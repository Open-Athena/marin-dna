"""Label-independent prompt-grammar preflight on released sequence recovery."""

from __future__ import annotations

import hashlib
import resource
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from datasets import load_dataset
from transformers import LogitsProcessorList

from marin_dna_carbon_conditioning_vep.prompts import (
    PromptGrammar,
    assert_prefix_outside_dna_mode,
)
from marin_dna_carbon_conditioning_vep.scoring import (
    load_carbon_model_and_tokenizer,
)
from marin_dna_carbon_conditioning_vep.tokenizer_snapshot import (
    assert_frozen_prefix_token_ids,
)


class PromptPreflightBlocked(RuntimeError):
    """Neither documented metadata grammar passes the direction check."""


@dataclass(frozen=True)
class SuppressTokenIds:
    """Greedy-generation logits processor matching Carbon's released runner."""

    token_ids: tuple[int, ...]

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        del input_ids
        for token_id in self.token_ids:
            scores[:, token_id] = -float("inf")
        return scores


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _sample_key(sequence: str, row_index: int) -> str:
    payload = f"{row_index}:{sequence}".encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def select_fixed_preflight_slice(
    dataset: pd.DataFrame,
    *,
    target_species: list[str],
    rows_per_species: int,
) -> pd.DataFrame:
    """Select a deterministic, balanced low-resource sequence-recovery slice."""
    required = {"sequence", "label", "type"}
    missing = sorted(required - set(dataset.columns))
    assert not missing, f"sequence-recovery dataset missing columns: {missing}"
    assert rows_per_species > 0
    working = dataset.reset_index(drop=False).rename(
        columns={"index": "source_row_index"}
    )
    working["species"] = working["type"].astype(str)
    working["sample_key"] = [
        _sample_key(str(row.sequence), int(row.source_row_index))
        for row in working.itertuples(index=False)
    ]
    selected: list[pd.DataFrame] = []
    for species in target_species:
        candidates = working.loc[working["species"] == species].sort_values(
            "sample_key"
        )
        assert len(candidates) >= rows_per_species, (
            f"sequence-recovery type {species!r} has {len(candidates)} rows; "
            f"need {rows_per_species}"
        )
        selected.append(candidates.head(rows_per_species))
    result = pd.concat(selected, ignore_index=True)
    assert len(result) == len(target_species) * rows_per_species
    assert not result["sample_key"].duplicated().any()
    return result


def _truncate_recovery_context(sequence: str, max_sequence_bp: int) -> str:
    """Match Carbon's pinned right truncation, including ambiguous bases."""
    n = (min(len(sequence), max_sequence_bp) // 6) * 6
    assert n > 0
    return sequence[-n:]


def continuation_accuracy(prediction: str, label: str, length_bp: int = 30) -> float:
    """Return Carbon's released fixed-denominator next-base accuracy."""
    matches = sum(
        prediction[index] == label[index]
        for index in range(min(len(prediction), len(label), length_bp))
    )
    return matches / length_bp


def _generate(
    records: list[dict[str, Any]],
    *,
    prefixes: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_sequence_bp: int,
    generated_tokens: int,
    batch_size: int,
) -> list[str]:
    prompts = [
        prefix + _truncate_recovery_context(str(record["sequence"]), max_sequence_bp)
        for record, prefix in zip(records, prefixes, strict=True)
    ]
    special_ids = tuple(int(value) for value in (tokenizer.all_special_ids or []))
    processor = LogitsProcessorList([SuppressTokenIds(special_ids)])
    predictions: list[str] = []
    tokenizer.padding_side = "left"
    for offset in range(0, len(prompts), batch_size):
        batch = prompts[offset : offset + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        encoded = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=generated_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                logits_processor=processor,
            )
        new_ids = output[:, -generated_tokens:]
        predictions.extend(
            tokenizer.decode(new_ids[index].tolist())
            for index in range(new_ids.shape[0])
        )
    return predictions


def choose_prompt_grammar(
    results: pd.DataFrame,
    *,
    grammar_names: list[str],
    tolerance: float,
) -> tuple[str, str, dict[str, dict[str, float]]]:
    """Select the uniquely best grammar with a positive correct-tag delta."""
    summaries: dict[str, dict[str, float]] = {}
    for grammar_name in grammar_names:
        subset = results.loc[results["grammar"] == grammar_name]
        assert not subset.empty
        summaries[grammar_name] = {
            "correct_accuracy": float(subset["correct_accuracy"].mean()),
            "untagged_accuracy": float(subset["untagged_accuracy"].mean()),
            "delta": float(subset["delta"].mean()),
        }
    ranked = sorted(
        grammar_names, key=lambda name: summaries[name]["delta"], reverse=True
    )
    selected, rejected = ranked
    if summaries[selected]["delta"] <= tolerance:
        raise PromptPreflightBlocked(
            "neither documented grammar increased low-resource continuation accuracy: "
            f"{summaries}"
        )
    if abs(summaries[selected]["delta"] - summaries[rejected]["delta"]) <= tolerance:
        raise PromptPreflightBlocked(
            f"documented grammars are tied within the configured tolerance: {summaries}"
        )
    return selected, rejected, summaries


def run_prompt_preflight(
    *,
    model_repo: str,
    model_revision: str,
    dtype_name: str,
    dataset_repo: str,
    dataset_revision: str,
    dataset_config: str,
    split: str,
    target_species: list[str],
    rows_per_species: int,
    max_sequence_bp: int,
    generated_tokens: int,
    batch_size: int,
    grammar_templates: dict[str, str],
    conditions: dict[str, str | None],
    selection_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the fixed preflight and return row-level evidence plus the frozen grammar."""
    assert split == "test", (
        "sequence-recovery preflight must use its released test split"
    )
    assert dtype_name == "bfloat16"
    started = time.monotonic()
    dataset = load_dataset(
        dataset_repo,
        dataset_config,
        split=split,
        revision=dataset_revision,
    ).to_pandas()
    selected_rows = select_fixed_preflight_slice(
        dataset,
        target_species=target_species,
        rows_per_species=rows_per_species,
    )
    records = selected_rows[["sequence", "label", "species", "sample_key"]].to_dict(
        "records"
    )
    if not torch.cuda.is_available():
        raise RuntimeError("prompt preflight requires a CUDA GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer = load_carbon_model_and_tokenizer(
        model_repo, model_revision, device, dtype_name
    )
    prefix_ids_by_grammar = assert_frozen_prefix_token_ids(
        tokenizer,
        grammar_templates,
        conditions,
    )
    untagged_prefix = "<dna>"
    untagged_ids = assert_prefix_outside_dna_mode(tokenizer, untagged_prefix)
    untagged_predictions = _generate(
        records,
        prefixes=[untagged_prefix] * len(records),
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_sequence_bp=max_sequence_bp,
        generated_tokens=generated_tokens,
        batch_size=batch_size,
    )
    untagged_accuracies = [
        continuation_accuracy(prediction, str(record["label"]))
        for prediction, record in zip(untagged_predictions, records, strict=True)
    ]

    result_rows: list[dict[str, Any]] = []
    for grammar_name, template in grammar_templates.items():
        grammar = PromptGrammar(grammar_name, template)
        prefixes = [grammar.render(str(record["species"])) for record in records]
        correct_predictions = _generate(
            records,
            prefixes=prefixes,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_sequence_bp=max_sequence_bp,
            generated_tokens=generated_tokens,
            batch_size=batch_size,
        )
        for record, prefix, prediction, untagged_prediction, untagged_accuracy in zip(
            records,
            prefixes,
            correct_predictions,
            untagged_predictions,
            untagged_accuracies,
            strict=True,
        ):
            prefix_ids = assert_prefix_outside_dna_mode(tokenizer, prefix)
            correct_accuracy = continuation_accuracy(prediction, str(record["label"]))
            result_rows.append(
                {
                    "sample_key": record["sample_key"],
                    "species": record["species"],
                    "grammar": grammar_name,
                    "prefix": prefix,
                    "prefix_ids": prefix_ids,
                    "correct_prediction": prediction,
                    "untagged_prediction": untagged_prediction,
                    "correct_accuracy": correct_accuracy,
                    "untagged_accuracy": untagged_accuracy,
                    "delta": correct_accuracy - untagged_accuracy,
                }
            )
    results = pd.DataFrame(result_rows)
    grammar_names = list(grammar_templates)
    assert len(grammar_names) == 2, "issue #486 defines exactly two candidate grammars"
    selected, rejected, summaries = choose_prompt_grammar(
        results,
        grammar_names=grammar_names,
        tolerance=selection_tolerance,
    )
    selected_grammar = PromptGrammar(selected, grammar_templates[selected])
    selected_prefixes = {
        condition: (
            "<dna>" if species is None else selected_grammar.render(str(species))
        )
        for condition, species in conditions.items()
    }
    summary: dict[str, Any] = {
        "status": "selected",
        "selected_grammar": selected,
        "rejected_grammar": rejected,
        "grammar_templates": grammar_templates,
        "grammar_summaries": summaries,
        "selected_prefixes": selected_prefixes,
        "prefix_ids": prefix_ids_by_grammar[selected],
        "all_prefix_ids": prefix_ids_by_grammar,
        "untagged_prefix_ids": untagged_ids,
        "model": model_repo,
        "model_revision": model_revision,
        "tokenizer_revision": model_revision,
        "dataset": dataset_repo,
        "dataset_revision": dataset_revision,
        "dataset_config": dataset_config,
        "split": split,
        "target_species": target_species,
        "rows_per_species": rows_per_species,
        "rows": len(records),
        "elapsed_seconds": time.monotonic() - started,
        "device": torch.cuda.get_device_name(device),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    return results, summary
