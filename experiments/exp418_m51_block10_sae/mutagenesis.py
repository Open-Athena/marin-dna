"""Causally map the selected exp418 splice features by base substitution.

This is a deliberately small exploratory follow-up. It mutates every position
in the center 61 bp of the ten highest-activating chromosome-22 donor and
acceptor examples, then measures the selected SAE feature and signed raw
residual dimension. Coordinates identify the original 0-based, half-open
reference window; mutated sequences are explicitly counterfactual.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl
import torch
from huggingface_hub import snapshot_download
from marin_dna.model.sae import M51_HIDDEN_SIZE, load_frozen_m51
from sae_lens.saes.sae import SAE

from interpret import (
    DEFAULT_BATCH_SIZE,
    FOCAL_INDEX,
    NUCLEOTIDES,
    WINDOW_BP,
    _extract_embeddings,
    _sha256,
    _write_json,
)
from launch import D_SAE, MODEL_ID, MODEL_REVISION

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS = ("donor", "acceptor")
DEFAULT_RADIUS = 30
NOOP_ABSOLUTE_TOLERANCE = 0.02


def counterfactual_sequences(sequence: str, *, radius: int) -> list[dict[str, Any]]:
    """Return all four target-base substitutions in a centered interval."""

    assert len(sequence) == WINDOW_BP
    assert set(sequence) <= set(NUCLEOTIDES)
    assert 0 <= radius <= FOCAL_INDEX
    output: list[dict[str, Any]] = []
    for relative_position in range(-radius, radius + 1):
        position = FOCAL_INDEX + relative_position
        reference_base = sequence[position]
        for target_base in NUCLEOTIDES:
            mutated = sequence[:position] + target_base + sequence[position + 1 :]
            assert len(mutated) == WINDOW_BP
            output.append(
                {
                    "relative_position": relative_position,
                    "reference_base": reference_base,
                    "target_base": target_base,
                    "changed": target_base != reference_base,
                    "sequence": mutated,
                }
            )
    assert len(output) == (2 * radius + 1) * len(NUCLEOTIDES)
    return output


def _mutation_rows(
    base_rows: Sequence[dict[str, Any]], *, radius: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for context_index, base_row in enumerate(base_rows):
        for specification in counterfactual_sequences(
            base_row["sequence"], radius=radius
        ):
            row = dict(base_row)
            row["sequence"] = specification["sequence"]
            row["row_id"] = (
                f"{base_row['row_id']}:mut:{specification['relative_position']}:"
                f"{specification['target_base']}"
            )
            rows.append(row)
            metadata.append(
                {
                    "context_index": context_index,
                    "source_row_id": base_row["row_id"],
                    "relative_position": specification["relative_position"],
                    "reference_base": specification["reference_base"],
                    "target_base": specification["target_base"],
                    "changed": specification["changed"],
                }
            )
    assert len(rows) == len(metadata)
    return rows, metadata


def _plot(summary: pl.DataFrame, output_dir: Path, radius: int) -> None:
    figure, axes = plt.subplots(
        len(TASKS),
        2,
        figsize=(14, 6.5),
        constrained_layout=True,
        squeeze=False,
    )
    score_columns = (
        ("mean_sae_delta", "SAE feature activation Δ"),
        ("mean_raw_delta", "signed raw-dimension activation Δ"),
    )
    positions = list(range(-radius, radius + 1))
    for task_index, task in enumerate(TASKS):
        task_frame = summary.filter(pl.col("task") == task)
        for score_index, (score_column, title) in enumerate(score_columns):
            matrix = np.empty((len(NUCLEOTIDES), len(positions)), dtype=np.float64)
            for base_index, target_base in enumerate(NUCLEOTIDES):
                values = (
                    task_frame.filter(pl.col("target_base") == target_base)
                    .sort("relative_position")[score_column]
                    .to_numpy()
                )
                assert values.shape == (len(positions),)
                matrix[base_index] = values
            limit = max(float(np.quantile(np.abs(matrix), 0.98)), 1e-6)
            axis = axes[task_index, score_index]
            image = axis.imshow(
                matrix,
                aspect="auto",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
                extent=(-radius - 0.5, radius + 0.5, 3.5, -0.5),
            )
            axis.axvline(0, color="black", linewidth=0.8, alpha=0.7)
            axis.set_yticks(range(len(NUCLEOTIDES)), list(NUCLEOTIDES))
            axis.set_xticks(range(-radius, radius + 1, 10))
            axis.set_xlabel("position relative to focal base")
            axis.set_ylabel("substitute target base")
            axis.set_title(f"{task}: {title}")
            figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle(
        "exp418 single-nucleotide dependency maps\n"
        "mean change over top 10 unique chromosome-22 contexts",
        fontsize=14,
    )
    figure.savefig(output_dir / "mutagenesis.png", dpi=180)
    figure.savefig(output_dir / "mutagenesis.svg")
    plt.close(figure)


@torch.inference_mode()
def run_mutagenesis(
    *,
    panel_path: Path,
    interpretation_results_path: Path,
    sae_path: Path,
    output_dir: Path,
    radius: int,
    batch_size: int,
) -> dict[str, Any]:
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    assert panel_path.exists()
    assert interpretation_results_path.exists()
    assert sae_path.exists()
    assert not output_dir.exists()
    started = time.monotonic()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert len(experiment_commit) == 40
    assert all(character in "0123456789abcdef" for character in experiment_commit)
    run_id = os.environ.get("RUN_ID", "")
    assert run_id

    interpretation = json.loads(interpretation_results_path.read_text())
    assert interpretation["panel"]["panel_sha256"] == _sha256(panel_path)
    rows = pl.read_parquet(panel_path).to_dicts()
    row_by_id = {row["row_id"]: row for row in rows}
    assert len(row_by_id) == len(rows)

    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    frozen = load_frozen_m51(checkpoint, device="cuda", dtype=torch.bfloat16)
    sae = SAE.load_from_disk(sae_path, device="cuda", dtype="float32")
    assert sae.cfg.architecture() == "jumprelu"
    assert sae.cfg.d_in == M51_HIDDEN_SIZE and sae.cfg.d_sae == D_SAE

    records: list[dict[str, Any]] = []
    task_metadata: dict[str, Any] = {}
    for task in TASKS:
        task_result = interpretation["tasks"][task]
        feature = int(task_result["sae"]["feature"])
        raw_dimension = int(task_result["raw"]["dimension"])
        raw_sign = int(task_result["raw"]["sign"])
        assert raw_sign in (-1, 1)
        top_contexts = task_result["top_test_contexts"]
        assert len(top_contexts) == 10
        assert all(context["label"] == 1 for context in top_contexts)
        base_rows = [row_by_id[context["row_id"]] for context in top_contexts]
        assert all(row["split"] == "test" and row["task"] == task for row in base_rows)
        assert len(
            {
                (row["chrom"], row["start"], row["end"], row["strand"])
                for row in base_rows
            }
        ) == len(base_rows)

        base_raw, base_features = _extract_embeddings(
            base_rows,
            frozen=frozen,
            sae=sae,
            batch_size=batch_size,
        )
        recorded_activations = np.asarray(
            [context["activation"] for context in top_contexts]
        )
        np.testing.assert_allclose(
            base_features[:, feature],
            recorded_activations,
            rtol=1e-5,
            atol=1e-5,
        )
        mutated_rows, mutation_metadata = _mutation_rows(base_rows, radius=radius)
        mutated_raw, mutated_features = _extract_embeddings(
            mutated_rows,
            frozen=frozen,
            sae=sae,
            batch_size=batch_size,
        )
        for index, metadata in enumerate(mutation_metadata):
            context_index = metadata["context_index"]
            sae_activation = float(mutated_features[index, feature])
            raw_activation = float(raw_sign * mutated_raw[index, raw_dimension])
            base_sae = float(base_features[context_index, feature])
            base_raw_activation = float(
                raw_sign * base_raw[context_index, raw_dimension]
            )
            records.append(
                {
                    "task": task,
                    "feature": feature,
                    "raw_dimension": raw_dimension,
                    "raw_sign": raw_sign,
                    **metadata,
                    "base_sae_activation": base_sae,
                    "mutated_sae_activation": sae_activation,
                    "sae_delta": sae_activation - base_sae,
                    "base_raw_activation": base_raw_activation,
                    "mutated_raw_activation": raw_activation,
                    "raw_delta": raw_activation - base_raw_activation,
                }
            )
        task_metadata[task] = {
            "feature": feature,
            "raw_dimension": raw_dimension,
            "raw_sign": raw_sign,
            "source_rows": [row["row_id"] for row in base_rows],
        }

    record_frame = pl.DataFrame(records)
    expected_rows = len(TASKS) * 10 * (2 * radius + 1) * len(NUCLEOTIDES)
    assert record_frame.height == expected_rows
    unchanged = record_frame.filter(~pl.col("changed"))
    assert unchanged.height == len(TASKS) * 10 * (2 * radius + 1)
    max_abs_noop_sae_delta = float(
        unchanged.select(pl.col("sae_delta").abs().max()).item()
    )
    max_abs_noop_raw_delta = float(
        unchanged.select(pl.col("raw_delta").abs().max()).item()
    )
    # The no-op sequences are byte-identical to the originals, but CUDA bf16
    # kernels can change their accumulation order when the batch shape differs.
    # Record the discrepancy and fail if it exceeds a few bf16 units rather
    # than requiring bitwise equality across differently shaped batches.
    assert max_abs_noop_sae_delta <= NOOP_ABSOLUTE_TOLERANCE
    assert max_abs_noop_raw_delta <= NOOP_ABSOLUTE_TOLERANCE
    summary = (
        record_frame.group_by("task", "relative_position", "target_base")
        .agg(
            pl.col("sae_delta").mean().alias("mean_sae_delta"),
            pl.col("sae_delta").median().alias("median_sae_delta"),
            pl.col("raw_delta").mean().alias("mean_raw_delta"),
            pl.col("raw_delta").median().alias("median_raw_delta"),
            pl.col("source_row_id").n_unique().alias("context_count"),
            pl.col("changed").sum().alias("changed_context_count"),
        )
        .sort("task", "relative_position", "target_base")
    )
    assert summary.height == len(TASKS) * (2 * radius + 1) * len(NUCLEOTIDES)
    assert summary.select(pl.col("context_count").min()).item() == 10

    output_dir.mkdir(parents=True)
    record_frame.write_parquet(output_dir / "mutations.parquet")
    summary.write_parquet(output_dir / "summary.parquet")
    _plot(summary, output_dir, radius)
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "experiment_commit": experiment_commit,
        "coordinate_semantics": (
            "loci are original 0-based half-open reference windows; sequences are counterfactual"
        ),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "block_index": 9},
        "inputs": {
            "panel_sha256": _sha256(panel_path),
            "interpretation_results_sha256": _sha256(interpretation_results_path),
            "sae_weights_sha256": _sha256(sae_path / "sae_weights.safetensors"),
        },
        "design": {
            "tasks": list(TASKS),
            "contexts_per_task": 10,
            "radius": radius,
            "target_bases_per_position": len(NUCLEOTIDES),
            "counterfactual_sequences": expected_rows,
            "noop_absolute_tolerance": NOOP_ABSOLUTE_TOLERANCE,
        },
        "numerical_noop_check": {
            "max_abs_sae_delta": max_abs_noop_sae_delta,
            "max_abs_raw_delta": max_abs_noop_raw_delta,
        },
        "task_features": task_metadata,
        "runtime": {
            "wall_seconds": time.monotonic() - started,
            "gpu_name": torch.cuda.get_device_name(0),
            "batch_size": batch_size,
        },
        "outputs": {
            "mutations": "mutations.parquet",
            "summary": "summary.parquet",
            "figure_png": "mutagenesis.png",
            "figure_svg": "mutagenesis.svg",
        },
    }
    _write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--interpretation-results", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    result = run_mutagenesis(
        panel_path=args.panel,
        interpretation_results_path=args.interpretation_results,
        sae_path=args.sae,
        output_dir=args.output_dir,
        radius=args.radius,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
