"""Frozen, matched FIM presentations from original held-out DNA windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from exp593_dna.annotations import load_annotations
from exp593_dna.audit import file_sha256
from exp593_dna.pipeline import (
    ALPHABET,
    DATASET_REVISION,
    AugmentationConfig,
    Augmenter,
    Encoder,
    Example,
    RawWindow,
    Span,
    load_encoder,
    rng_for,
)


def paired_config(config: AugmentationConfig) -> AugmentationConfig:
    """A FIM-only panel with a minimal member and an informative paired member."""
    return replace(
        config,
        kind_probabilities=(0, 0, 0.75, 0.25),
        minimal_fim_probability=0,
        species_probability=1,
        mask_description_probability=1,
    )


def matched_pair(
    augmenter: Augmenter, raw: RawWindow, occurrence: int
) -> tuple[Example, Example]:
    """Change presentation alone; the two examples share all DNA and gap choices.

    The formatted member determines the crop after budgeting its metadata.
    Both members use the larger of their text budgets, so neither presentation
    truncates context differently or overflows its allocated sequence length.
    """
    if raw.split != "validation":
        raise ValueError("Matched FIM validation requires held-out source rows")
    config = paired_config(augmenter.config)
    formatted = Augmenter(
        augmenter.encoder, config, augmenter.registry, augmenter.annotations
    ).build_validation(raw, occurrence)
    a = formatted.attributes
    dna = {s.role: s.text for s in formatted.spans if s.role != "text"}
    label_rng = rng_for(config.seed, occurrence, "fim_validation_labels")
    labels = {
        "left": str(label_rng.choice(("left", "prefix", "L"))),
        "right": str(label_rng.choice(("right", "suffix", "R"))),
    }
    minimal_spans = []
    roles = ("right", "left") if a["right_flank_first"] else ("left", "right")
    for role in roles:
        minimal_spans.extend(
            (
                Span(f"[{labels[role]}] ", "text"),
                Span(dna[role], role),
                Span("\n", "text"),
            )
        )
    if a["gap_length_visible"]:
        minimal_spans.append(Span(f"[gap:{a['gap_length']}]\n", "text"))
    minimal_spans.extend((Span("[fill] ", "text"), Span(dna["middle"], "middle", True)))
    minimal_text_tokens = sum(
        len(augmenter.encoder.text_ids(s.text))
        for s in minimal_spans
        if s.role == "text"
    )
    extra = max(0, minimal_text_tokens - a["text_tokens"])
    # Normally zero: the formatted member always includes species and mask text.
    # Still handle a tokenizer/template combination with longer minimal labels.
    trim_left = min(extra, len(dna["left"]) - 128)
    trim_right = extra - trim_left
    if trim_right > len(dna["right"]) - 128:
        raise ValueError("Insufficient shared flank context for paired text budgets")
    dna["left"] = dna["left"][trim_left:]
    if trim_right:
        dna["right"] = dna["right"][:-trim_right]
    pair_key = hashlib.sha256(
        f"{DATASET_REVISION}:{raw.row}:{config.seed}:{occurrence}".encode()
    ).hexdigest()
    result = []
    for presentation, spans in (
        ("minimal", minimal_spans),
        ("formatted", formatted.spans),
    ):
        spans = tuple(
            s if s.role == "text" else replace(s, text=dna[s.role]) for s in spans
        )
        token_ids, weights = augmenter.encoder.encode(spans)
        if len(token_ids) > formatted.bucket:
            raise ValueError("Matched validation presentation overflows its bucket")
        attributes = dict(a)
        attributes.update(
            pair_key=pair_key,
            presentation=presentation,
            input_tokens=len(token_ids),
            text_tokens=sum(
                len(augmenter.encoder.text_ids(s.text))
                for s in spans
                if s.role == "text"
            ),
            dna_tokens=a["dna_tokens"] - extra,
            gap_start=a["gap_start"] - trim_left,
            crop_start_in_oriented_window=a["crop_start_in_oriented_window"]
            + trim_left,
            minimal_fim=presentation == "minimal",
        )
        if presentation == "minimal":
            attributes.update(
                metadata_fields=[],
                metadata_structure="none",
                mask_description=False,
                description=False,
                layout="compact",
            )
        result.append(Example(token_ids, weights, spans, formatted.bucket, attributes))
    return result[0], result[1]


def prompt_and_target(example: Example) -> tuple[np.ndarray, np.ndarray]:
    """Expose the caller-known DNA length without teacher-forcing generated tokens."""
    if not example.attributes["kind"].endswith("fim"):
        raise ValueError("Expected a FIM completion")
    length = int(example.attributes["gap_length"])
    if length < 1 or example.scored_tokens != length:
        raise ValueError("FIM target length changed")
    expected = np.arange(
        len(example.token_ids) - length - 1, len(example.token_ids) - 1
    )
    if not np.array_equal(np.flatnonzero(example.loss_weights), expected):
        raise ValueError(
            "FIM completion must be the final, exclusively scored DNA span"
        )
    return example.token_ids[:-length], example.token_ids[-length:]


@dataclass(frozen=True)
class FIMObservation:
    """Teacher-forced predictions and optional independently generated completion."""

    example: Example
    target_nll: np.ndarray
    predicted_tokens: np.ndarray
    generated_tokens: np.ndarray | None = None


def fim_metrics(
    observations: Sequence[FIMObservation], encoder: Encoder
) -> dict[str, float | int]:
    """Aggregate example-mean NLL, token accuracy and actual autoregressive EM.

    Exact match is only reported for independently generated completions; an
    all-correct teacher-forced span is never treated as autoregressive success.
    Counts accompany every stratum, including the smaller generation panel.
    """
    if not observations:
        raise ValueError("FIM metrics require at least one observation")
    groups: dict[str, list[FIMObservation]] = defaultdict(list)
    for item in observations:
        _, target = prompt_and_target(item.example)
        nll = np.asarray(item.target_nll)
        if nll.shape != target.shape or not np.all(np.isfinite(nll)):
            raise ValueError("Expected finite NLL for each missing DNA token")
        if np.shape(item.predicted_tokens) != target.shape:
            raise ValueError("Teacher-forced prediction shape differs from the gap")
        if (
            item.generated_tokens is not None
            and np.shape(item.generated_tokens) != target.shape
        ):
            raise ValueError(
                "Caller-controlled generation must produce exactly the requested tokens"
            )
        a = item.example.attributes
        for group in (
            "",
            f"/{a['presentation']}",
            f"/gap_{len(target)}",
            "/short" if len(target) <= 3 else "/long",
            "/guided" if a["annotation_guided"] else "/random",
        ):
            groups[group].append(item)
    casefold = {int(encoder.dna_lookup[ord(c)]): c.casefold() for c in ALPHABET}
    result: dict[str, float | int] = {}
    for group, items in sorted(groups.items()):
        suffix = group.replace("/", "_", 1)
        prefix = f"eval/fim{suffix}"
        targets = [prompt_and_target(item.example)[1] for item in items]
        result[f"{prefix}_loss"] = float(
            np.mean([np.mean(i.target_nll) for i in items])
        )
        correct = sum(
            int(np.count_nonzero(np.asarray(i.predicted_tokens) == t))
            for i, t in zip(items, targets, strict=True)
        )
        count = sum(len(t) for t in targets)
        result[f"{prefix}_base_accuracy"] = correct / count
        result[f"{prefix}_examples"] = len(items)
        result[f"{prefix}_targets"] = count
        generated = [
            (i, t)
            for i, t in zip(items, targets, strict=True)
            if i.generated_tokens is not None
        ]
        result[f"{prefix}_generated_examples"] = len(generated)
        if generated:
            exact = dna_exact = 0
            for item, target in generated:
                prediction = np.asarray(item.generated_tokens)
                exact += int(np.array_equal(prediction, target))
                predicted_dna = [
                    casefold.get(int(token), "<non-DNA>") for token in prediction
                ]
                target_dna = [casefold[int(token)] for token in target]
                dna_exact += int(predicted_dna == target_dna)
            result[f"{prefix}_exact_match"] = exact / len(generated)
            result[f"{prefix}_dna_exact_match"] = dna_exact / len(generated)
    return result


def freeze_panel(
    manifest: Path, partitions: Path, annotations: Path, output: Path
) -> dict[str, Any]:
    """Join the fixed disjoint holdout to native DNA and freeze matched prompts."""
    source = json.loads(manifest.read_text())
    if source["dataset_revision"] != DATASET_REVISION:
        raise ValueError("Validation manifest dataset revision differs")
    selected = source["examples"]
    if len(selected) != 512 or len({r["row"] for r in selected}) != 512:
        raise ValueError("The fixed FIM panel must reuse the 512 disjoint source rows")
    raw_by_row = {}
    for code in sorted({r["assembly"] for r in selected}):
        rows = {r["row"] for r in selected if r["assembly"] == code}
        for batch in pq.ParquetFile(
            partitions / "validation" / f"{code}.parquet"
        ).iter_batches():
            for rec in batch.to_pylist():
                if rec["row"] in rows:
                    raw = RawWindow.from_dict(rec)
                    if raw.row in raw_by_row or raw.split != "validation":
                        raise ValueError("Validation row identity differs")
                    raw_by_row[raw.row] = raw
    encoder = load_encoder()
    pipeline = Augmenter(
        encoder,
        replace(
            AugmentationConfig(),
            seed=593,
            length_buckets=(8192,),
            bucket_probabilities=(1.0,),
        ),
        annotations=load_annotations(
            annotations, split="validation", rows=set(raw_by_row)
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    histogram: Counter[str] = Counter()
    generation_candidates: Counter[int] = Counter()
    path = output / "fim_validation_pairs.jsonl"
    temporary = path.with_suffix(".part")
    with temporary.open("w") as stream:
        for occurrence, metadata in enumerate(selected):
            raw = raw_by_row[metadata["row"]]
            if any(
                getattr(raw, k) != v
                for k, v in metadata.items()
                if k in RawWindow.__dataclass_fields__
            ):
                raise ValueError(
                    "Disjoint manifest metadata differs from the original row"
                )
            minimal, formatted = matched_pair(pipeline, raw, occurrence)
            length = minimal.attributes["gap_length"]
            histogram[f"gap_{length}"] += 1
            histogram[
                "guided" if minimal.attributes["annotation_guided"] else "random"
            ] += 1
            candidate = generation_candidates[length] < 2
            if candidate:
                generation_candidates[length] += 1
            stream.write(
                json.dumps(
                    {
                        "source_row": raw.row,
                        "source_sequence_sha256": hashlib.sha256(
                            raw.seq.encode()
                        ).hexdigest(),
                        "generation_candidate": candidate,
                        "members": [
                            {
                                "attributes": e.attributes,
                                "bucket": e.bucket,
                                "spans": [asdict(s) for s in e.spans],
                            }
                            for e in (minimal, formatted)
                        ],
                    }
                )
                + "\n"
            )
    temporary.replace(path)
    report = {
        "dataset_revision": DATASET_REVISION,
        "source_manifest_sha256": file_sha256(manifest),
        "annotation_sidecar_sha256": file_sha256(annotations),
        "tokenizer_sha256": encoder.digest,
        "pairs": len(selected),
        "presentations": len(selected) * 2,
        "species": len({r["assembly"] for r in selected}),
        "histogram_pairs": dict(histogram),
        "generation_candidate_pairs_by_gap": dict(generation_candidates),
        "generation_policy": "Fixed candidates only; decode budget is set after accelerator profiling. EM denominators report completions actually generated.",
        "pairs_sha256": file_sha256(path),
        "formatted_member_config": asdict(paired_config(pipeline.config)),
        "pair_policy": "Every shared DNA/gap has one minimal member and one formatted member; formatted members always include species and mask-availability text.",
    }
    (output / "fim_validation_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze_panel(args.manifest, args.partitions, args.annotations, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
