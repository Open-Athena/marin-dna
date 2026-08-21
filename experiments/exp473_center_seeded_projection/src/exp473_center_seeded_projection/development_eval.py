"""Isolated development-only scoring adapters for issue #473.

The maintained evals_v2 rules remain unchanged. This module calls their
official model runners and metric functions while enforcing two experiment
specific boundaries:

* download exactly train.parquet rather than asking a dataset builder to
  prepare every split; and
* load the fixed character-plus-BOS tokenizer explicitly, because its exported
  TokenizersBackend metadata is not registered with AutoTokenizer.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEVELOPMENT_SPLIT = "train"
ALLOWED_DEVELOPMENT_CHROMS = {
    *(str(chrom) for chrom in range(1, 23, 2)),
    "X",
}
EXPECTED_TOKEN_IDS = {
    "pad_token_id": 0,
    "unk_token_id": 1,
    "bos_token_id": 2,
}


def _normalized_chromosome(value: object) -> str:
    normalized = str(value).strip()
    if normalized.lower().startswith("chr"):
        normalized = normalized[3:]
    return normalized.upper()


def assert_development_chromosomes(frame: pd.DataFrame) -> None:
    """Reject any even-autosome, Y, or otherwise unregistered labeled row."""
    assert "chrom" in frame.columns, "development dataset is missing chrom"
    observed = {_normalized_chromosome(value) for value in frame["chrom"]}
    unexpected = observed - ALLOWED_DEVELOPMENT_CHROMS
    assert not unexpected, (
        "development dataset contains held-out or unknown chromosomes: "
        f"{sorted(unexpected)}"
    )


def load_development_dataset(
    repo_id: str,
    *,
    revision: str,
    filename: str,
    split: str,
    hub_download: Callable[..., str] | None = None,
    parquet_loader: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Download and open exactly the registered development parquet."""
    assert split == DEVELOPMENT_SPLIT, (
        f"issue #473 evaluation is development-only, got split={split!r}"
    )
    expected_filename = f"{DEVELOPMENT_SPLIT}.parquet"
    assert filename == expected_filename, (
        f"issue #473 must read exactly {expected_filename!r}, got {filename!r}"
    )
    assert len(revision) == 40, "dataset revision must be a full commit SHA"
    if hub_download is None:
        from huggingface_hub import hf_hub_download

        hub_download = hf_hub_download
    if parquet_loader is None:
        from datasets import load_dataset

        parquet_loader = load_dataset
    path = hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )
    dataset = parquet_loader(
        "parquet",
        data_files={DEVELOPMENT_SPLIT: path},
        split=DEVELOPMENT_SPLIT,
    )
    frame = dataset.to_pandas()
    assert len(frame) > 0, f"development dataset {repo_id}@{revision} is empty"
    assert_development_chromosomes(frame)
    return frame


def load_issue473_tokenizer(
    checkpoint_path: str | Path,
    *,
    tokenizer_factory: Callable[..., Any] | None = None,
) -> Any:
    """Load the exact character-plus-BOS tokenizer without AutoTokenizer."""
    checkpoint = Path(checkpoint_path)
    tokenizer_path = checkpoint / "tokenizer.json"
    config_path = checkpoint / "tokenizer_config.json"
    assert tokenizer_path.is_file(), f"missing tokenizer graph: {tokenizer_path}"
    assert config_path.is_file(), f"missing tokenizer metadata: {config_path}"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "bos_token": "[BOS]",
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
    }
    observed = {key: config.get(key) for key in expected}
    assert observed == expected, (
        f"unexpected issue #473 tokenizer metadata: {observed}; expected {expected}"
    )
    tokenizer_class = config.get("tokenizer_class")
    assert tokenizer_class in {"PreTrainedTokenizerFast", "TokenizersBackend"}, (
        f"unexpected issue #473 tokenizer class: {tokenizer_class!r}"
    )
    if tokenizer_class == "TokenizersBackend":
        assert config.get("backend") == "tokenizers", (
            "TokenizersBackend metadata must declare backend='tokenizers'"
        )
    if tokenizer_factory is None:
        from transformers import PreTrainedTokenizerFast

        tokenizer_factory = PreTrainedTokenizerFast
    tokenizer = tokenizer_factory(
        tokenizer_file=str(tokenizer_path),
        bos_token=config["bos_token"],
        pad_token=config["pad_token"],
        unk_token=config["unk_token"],
        clean_up_tokenization_spaces=bool(
            config.get("clean_up_tokenization_spaces", False)
        ),
        model_max_length=int(config["model_max_length"]),
    )
    for attribute, expected_id in EXPECTED_TOKEN_IDS.items():
        assert getattr(tokenizer, attribute) == expected_id, (
            f"{attribute}={getattr(tokenizer, attribute)} != {expected_id}"
        )
    assert tokenizer.encode("acgt") == [2, 3, 4, 5, 6]
    assert tokenizer.encode("acgt", add_special_tokens=False) == [3, 4, 5, 6]
    return tokenizer


def compute_issue473_variant_scores(
    checkpoint_path: str | Path,
    dataset: pd.DataFrame,
    genome_path: str | Path,
    *,
    context_size: int,
    batch_size: int,
    num_workers: int,
    data_transform_on_the_fly: bool,
    torch_compile: bool,
    rc: bool,
) -> pd.DataFrame:
    """Call the official variant-score runner with the fixed tokenizer loader."""
    assert rc, "issue #473 requires FWD+RC scoring"
    from datasets import Dataset
    from transformers import AutoModelForCausalLM

    from marin_dna.data.genome import Genome
    from marin_dna_evals.model.runner import run_variant_score_bundle

    checkpoint = Path(checkpoint_path)
    tokenizer = load_issue473_tokenizer(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=True,
    )
    results = run_variant_score_bundle(
        model,
        tokenizer,
        Dataset.from_pandas(dataset, preserve_index=False),
        Genome(genome_path),
        context_size,
        rc=True,
        return_embeddings=False,
        data_transform_on_the_fly=data_transform_on_the_fly,
        inference_kwargs={
            "per_device_eval_batch_size": batch_size,
            "torch_compile": torch_compile,
            "bf16_full_eval": True,
            "dataloader_num_workers": num_workers,
            "remove_unused_columns": False,
        },
    )
    assert set(results) == {"fwd", "rc"}, (
        f"official score runner returned unexpected strands: {sorted(results)}"
    )
    columns: dict[str, np.ndarray] = {}
    for strand in ("fwd", "rc"):
        values = np.asarray(results[strand])
        assert values.shape == (len(dataset), 2)
        assert np.isfinite(values).all()
        columns[f"llr_{strand}"] = values[:, 0]
        columns[f"jsd_{strand}"] = values[:, 1]
    return pd.DataFrame(columns)


def compute_issue473_metrics(
    scores: pd.DataFrame,
    *,
    model: str,
    dataset: str,
    split: str,
    score_protocol: str,
    eval_protocol: str,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Call the unchanged official metric functions for one development cell."""
    assert split == DEVELOPMENT_SPLIT
    assert_development_chromosomes(scores)
    from marin_dna_evals.conservation import (
        REQUIRED_VARIANT_COLUMNS,
        SGE_VARIANT_COLUMNS,
    )
    from marin_dna_evals.metrics import (
        SCORE_PROTOCOLS,
        compute_auprc_metrics,
        compute_sge_metrics,
    )

    assert eval_protocol in {"matched_pair", "sge"}, (
        f"issue #473 has no registered eval protocol {eval_protocol!r}"
    )
    transform = SCORE_PROTOCOLS[score_protocol]
    frame = scores.copy()
    frame["llr_avg"] = (frame["llr_fwd"] + frame["llr_rc"]) / 2
    frame["jsd_avg"] = (frame["jsd_fwd"] + frame["jsd_rc"]) / 2
    score_columns: list[str] = []
    for strand in ("fwd", "rc", "avg"):
        frame[f"{score_protocol}_{strand}"] = transform(frame[f"llr_{strand}"])
        score_columns.extend((f"{score_protocol}_{strand}", f"jsd_{strand}"))
    if eval_protocol == "sge":
        required = SGE_VARIANT_COLUMNS
        metrics = compute_sge_metrics(
            dataset=frame[list(required)],
            scores=frame[score_columns],
            score_columns=score_columns,
            n_bootstrap=n_bootstrap,
            rng=bootstrap_seed,
        )
    else:
        required = REQUIRED_VARIANT_COLUMNS
        metrics = compute_auprc_metrics(
            dataset=frame[list(required)],
            scores=frame[score_columns],
            score_columns=score_columns,
            n_bootstrap=n_bootstrap,
            rng=bootstrap_seed,
        )
    metrics["model"] = model
    metrics["dataset"] = dataset
    metrics["split"] = split
    return metrics


def compute_issue473_ll_atoms(
    checkpoint_path: str | Path,
    sequences: pd.DataFrame,
    *,
    window_size: int,
    batch_size: int,
    num_workers: int,
    torch_compile: bool,
) -> pd.DataFrame:
    """Call the official causal-LM loss runner with the fixed tokenizer loader."""
    assert "seq" in sequences.columns
    n_rows = len(sequences)
    assert n_rows > 0
    lengths = sequences["seq"].astype(str).str.len()
    assert (lengths == window_size).all(), (
        f"every seq must be {window_size} bp; got {sorted(lengths.unique())[:5]}"
    )
    from datasets import Dataset
    from transformers import AutoModelForCausalLM

    from marin_dna_evals.model.runner import run_ll_clm

    checkpoint = Path(checkpoint_path)
    tokenizer = load_issue473_tokenizer(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=True,
    )
    prediction = np.asarray(
        run_ll_clm(
            model,
            tokenizer,
            Dataset.from_pandas(sequences[["seq"]], preserve_index=False),
            data_transform_on_the_fly=True,
            inference_kwargs={
                "per_device_eval_batch_size": batch_size,
                "torch_compile": torch_compile,
                "bf16_full_eval": True,
                "dataloader_num_workers": num_workers,
                "remove_unused_columns": False,
            },
        )
    )
    if prediction.ndim == 1 and prediction.shape[0] == n_rows * 4:
        prediction = prediction.reshape(n_rows, 4)
    assert prediction.shape == (n_rows, 4)
    assert np.isfinite(prediction).all()
    output = pd.DataFrame(
        {
            "ll_sum_upper": prediction[:, 0].astype(np.float64),
            "ll_sum_lower": prediction[:, 1].astype(np.float64),
            "n_upper": prediction[:, 2].astype(np.int64),
            "n_lower": prediction[:, 3].astype(np.int64),
        }
    )
    if "id" in sequences.columns:
        output.insert(0, "id", sequences["id"].to_numpy())
    return output
