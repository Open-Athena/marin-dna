"""Run augmentation on sampled raw training DNA, without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from exp593_dna.pipeline import (
    DATASET_REVISION,
    KINDS,
    Augmenter,
    RawWindow,
    collate,
    load_encoder,
)
from exp593_dna.templates import catalog_manifest
from exp593_dna.validation import select_disjoint_validation, verify_disjoint_metadata

PRIMARY_SHA256 = "62334d75edfef471c248b36c01e1542885acb5c0ecf3ac88afbd164ebe56e112"


def verify_prep_identity(prep: Path) -> None:
    """Validate dataset and historical-sample evidence before reusing prep inputs."""
    if (
        json.loads((prep / "sources/dataset_api.json").read_text())["sha"]
        != DATASET_REVISION
    ):
        raise ValueError("Unexpected dataset revision")
    primary = [
        json.loads(line)
        for line in (prep / "primary_validation_512.jsonl").read_text().splitlines()
    ]
    if len(primary) != 512 or [r["row"] for r in primary] != [
        i * 329832 // 512 for i in range(512)
    ]:
        raise ValueError("Historical primary validation membership changed")
    digest = hashlib.sha256(
        "".join(r["seq"].lower() for r in primary).encode()
    ).hexdigest()
    if digest != PRIMARY_SHA256:
        raise ValueError("Historical primary validation sequences changed")
    evidence = json.loads((prep / "s3_validation_and_overlap.json").read_text())
    if not evidence["all_primary_512_match_raw_case"]:
        raise ValueError("Missing historical storage verification")


def run_audit(
    prep: Path, output: Path, count: int = 100_000, sample_count: int = 32
) -> dict[str, Any]:
    """Audit frequency over deterministic occurrences and actual train windows."""
    if count < 1 or sample_count < 0:
        raise ValueError("Invalid audit size")
    verify_prep_identity(prep)
    sources = sorted((prep / "metadata/train").glob("*.sample.jsonl"))
    raw = [
        RawWindow.from_dict(json.loads(line))
        for path in sources
        for line in path.read_text().splitlines()
    ]
    if not raw:
        raise ValueError("No sampled training windows found")
    output.mkdir(parents=True, exist_ok=True)
    pipeline = Augmenter(load_encoder())
    config = pipeline.config
    histograms: dict[str, Counter[Any]] = defaultdict(Counter)
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    input_tokens = padded_tokens = text_tokens = targets = 0
    input_by_kind: Counter[str] = Counter()
    targets_by_kind: Counter[str] = Counter()
    lengths = []
    examples = []
    samples = []
    signatures = set()
    started = time.monotonic()
    for occurrence in range(count):
        e = pipeline.build(raw[occurrence % len(raw)], occurrence)
        a = e.attributes
        kind = a["kind"]
        for field in [
            "kind",
            "case",
            "rc",
            "bucket",
            "mask_description",
            "description",
            "metadata_structure",
            "layout",
            "right_flank_first",
            "gap_length",
            "center_gap",
        ]:
            histograms[field][a[field]] += 1
        for field in a["metadata_fields"]:
            histograms["metadata_field"][field] += 1
        by_kind[kind]["examples"] += 1
        by_kind[kind]["rc"] += int(a["rc"])
        by_kind[kind][a["case"]] += 1
        histograms["species"][a["assembly"]] += 1
        input_tokens += len(e.token_ids)
        padded_tokens += e.bucket
        text_tokens += a["text_tokens"]
        targets += e.scored_tokens
        input_by_kind[kind] += len(e.token_ids)
        targets_by_kind[kind] += e.scored_tokens
        lengths.append(len(e.token_ids))
        if not np.isclose(e.loss_weights.sum(dtype=np.float64), 1, atol=1e-6):
            raise AssertionError("Per-example objective weights must sum to one")
        signatures.add(
            hashlib.sha256(
                "".join(
                    s.text if s.role == "text" else f"<{s.role}>" for s in e.spans
                ).encode()
            ).hexdigest()
        )
        # Collate complete global batches, checking no cross-document boundaries.
        examples.append(e)
        if len(examples) == config.global_batch_size:
            batch = collate(examples)
            assert np.all(
                np.count_nonzero(batch["segment_ids"] >= 0, axis=1) == batch["lengths"]
            )
            assert np.all(batch["loss_weights"][batch["segment_ids"] < 0] == 0)
            examples.clear()
        if (
            len(samples) < sample_count
            and occurrence % max(1, count // max(1, sample_count)) == 0
        ):
            samples.append(
                {"attributes": a, "text": e.text, "spans": [asdict(s) for s in e.spans]}
            )
        if (occurrence + 1) % 10_000 == 0:
            print(
                json.dumps(
                    {
                        "examples": occurrence + 1,
                        "elapsed_seconds": round(time.monotonic() - started, 2),
                    }
                ),
                flush=True,
            )
    if examples:
        collate(examples)
    probabilities = dict(zip(KINDS, config.kind_probabilities, strict=True))
    report = {
        "scope": "CPU augmentation only; 100% train-split sources. Raw record-batch samples are reused to exercise deterministic occurrence choices, not a full-corpus training-order replay.",
        "examples": count,
        "unique_raw_train_windows": len(raw),
        "species": len(histograms["species"]),
        "dataset_revision": DATASET_REVISION,
        "tokenizer_sha256": pipeline.encoder.digest,
        "config": asdict(config),
        "histograms": {
            k: {str(a): int(b) for a, b in v.items()} for k, v in histograms.items()
        },
        "kind_rates": {
            k: {
                "configured": probabilities[k],
                "observed": histograms["kind"][k] / count,
                "input_token_fraction": input_by_kind[k] / input_tokens,
                "scored_target_fraction": targets_by_kind[k] / targets,
                "normalized_objective_mass_fraction": histograms["kind"][k] / count,
            }
            for k in KINDS
        },
        "conditional_by_kind": {
            k: {
                name: (value / v["examples"] if name != "examples" else value)
                for name, value in v.items()
            }
            for k, v in by_kind.items()
        },
        "active_rates": {
            k: histograms[k][True] / count
            for k in [
                "rc",
                "mask_description",
                "description",
                "right_flank_first",
                "center_gap",
            ]
        },
        "metadata_rates": {
            k: v / count for k, v in histograms["metadata_field"].items()
        },
        "actual_input_tokens": input_tokens,
        "allocated_tokens": padded_tokens,
        "pad_fraction": 1 - input_tokens / padded_tokens,
        "text_token_fraction": text_tokens / input_tokens,
        "length_quantiles": {
            str(q): float(np.quantile(lengths, q))
            for q in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
        },
        "distinct_rendered_prompt_signatures": len(signatures),
        "semantic_component_count": catalog_manifest()["component_count"],
        "elapsed_seconds": time.monotonic() - started,
        "source_files": [
            {
                "path": str(p.relative_to(prep)),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sources
        ],
    }
    (output / "augmentation_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (output / "examples.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in samples)
    )
    (output / "template_catalog.json").write_text(
        json.dumps(catalog_manifest(), indent=2) + "\n"
    )
    candidates = pq.read_table(
        prep / "interval_disjoint_validation_candidates.parquet"
    ).to_pylist()
    holdout = select_disjoint_validation(candidates)
    checked = verify_disjoint_metadata(holdout, prep / "metadata/train")
    (output / "validation_disjoint_512.json").write_text(
        json.dumps(
            {
                "seed": 593,
                "dataset_revision": DATASET_REVISION,
                "candidate_pool_sha256": hashlib.sha256(
                    (
                        prep / "interval_disjoint_validation_candidates.parquet"
                    ).read_bytes()
                ).hexdigest(),
                "examples": holdout,
                "metric": "eval/loss_disjoint",
                "species": len({r["assembly"] for r in holdout}),
            },
            indent=2,
        )
        + "\n"
    )
    (output / "validation_primary_512.json").write_text(
        json.dumps(
            {
                "dataset_revision": DATASET_REVISION,
                "row_indices": [i * 329832 // 512 for i in range(512)],
                "historical_raw_case_verified": True,
                "raw_sequence_list_lowercase_sha256": PRIMARY_SHA256,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "holdout_verification.json").write_text(
        json.dumps(
            {
                "selected_examples": len(holdout),
                "species": len({r["assembly"] for r in holdout}),
                "training_metadata_rows_compared": checked,
                "intersecting_source_intervals": 0,
                "selected_manifest_sha256": hashlib.sha256(
                    (output / "validation_disjoint_512.json").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    return report


def main() -> None:
    """CLI entry point intentionally exposes no train/dispatch command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=100_000)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    report = run_audit(args.prep, args.output, args.examples, args.samples)
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "examples",
                    "kind_rates",
                    "conditional_by_kind",
                    "active_rates",
                    "metadata_rates",
                    "pad_fraction",
                    "length_quantiles",
                    "semantic_component_count",
                    "elapsed_seconds",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
