"""Analyze the frozen issue #429 causal panel on untouched test contexts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from analyze import sha256_file, write_json
from analyze_perturbations import (
    COMPARISON_SPEC,
    FEATURE_IDS,
    PRIMARY_SPECS,
    bootstrap_mean_interval,
    build_response_frame,
    codon_context_selectivity,
    codon_summaries,
    plot_codon_selectivity,
    splice_summaries,
    summarize_selectivity,
)
from sample_panel import assert_current_commit

ISSUE = 429
RANDOM_SEED = 429_2
EXPECTED_RADIUS = 15
EXPECTED_POSITIONS = 2 * EXPECTED_RADIUS + 1
EXPECTED_CONTEXTS_PER_CLASS = 64
EXPECTED_CONTEXT_GROUP = "untouched_test_hash"
EXPECTED_CLASSES = set(PRIMARY_SPECS)
SPLICE_TARGET_POSITIONS = {
    "splice_acceptor_variant": frozenset({-1, 0}),
    "splice_donor_5th_base_variant": frozenset({-4, -3}),
}


def validate_heldout_inputs(
    panel: pl.DataFrame,
    design_manifest: dict[str, Any],
    extraction_manifest: dict[str, Any],
    *,
    panel_path: Path,
    design_manifest_path: Path,
    extraction_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Verify the prospective design/extraction chain and load exact arrays."""

    rows = int(design_manifest["rows"])
    assert rows == panel.height > 0
    assert panel["perturbation_row"].to_list() == list(range(rows))
    assert set(panel["class"].unique()) == EXPECTED_CLASSES
    assert set(panel["context_group"].unique()) == {EXPECTED_CONTEXT_GROUP}
    assert set(panel["source_split"].unique()) == {"test"}
    assert set(panel["edit_distance"].unique()) == {1}
    assert panel.select("source_panel_row").n_unique() == (
        len(EXPECTED_CLASSES) * EXPECTED_CONTEXTS_PER_CLASS
    )
    assert design_manifest["protocol"]["source_selection"].endswith(
        "no SAE activation ranking"
    )
    assert design_manifest["artifacts"][panel_path.name]["sha256"] == sha256_file(
        panel_path
    )
    assert extraction_manifest["design"]["manifest_sha256"] == sha256_file(
        design_manifest_path
    )
    assert extraction_manifest["design"]["panel_sha256"] == sha256_file(panel_path)
    assert extraction_manifest["design"]["paired_rows"] == rows
    assert extraction_manifest["protocol"]["feature_ids"] == list(FEATURE_IDS)
    assert extraction_manifest["protocol"]["spatial_radius"] == EXPECTED_RADIUS
    for filename, metadata in extraction_manifest["artifacts"].items():
        path = extraction_dir / filename
        assert path.is_file() and path.stat().st_size == metadata["bytes"]
        assert sha256_file(path) == metadata["sha256"]

    reference_indices = np.load(
        extraction_dir / "reference_state_indices.npy", mmap_mode="r"
    )
    alternate_indices = np.load(
        extraction_dir / "alternate_state_indices.npy", mmap_mode="r"
    )
    forward = np.load(extraction_dir / "state_activations_forward.npy", mmap_mode="r")
    reverse = np.load(
        extraction_dir / "state_activations_reverse_complement.npy", mmap_mode="r"
    )
    assert reference_indices.shape == alternate_indices.shape == (rows,)
    assert forward.shape == reverse.shape
    assert forward.ndim == 3
    assert forward.shape[1:] == (EXPECTED_POSITIONS, len(FEATURE_IDS))
    assert reference_indices.min() >= 0 and alternate_indices.min() >= 0
    assert reference_indices.max() < forward.shape[0]
    assert alternate_indices.max() < forward.shape[0]
    return reference_indices, alternate_indices, forward, reverse


def splice_context_contrasts(
    responses: pl.DataFrame, *, bootstrap_samples: int = 2_000
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Contrast preregistered motif positions with the remaining saturation window."""

    assert bootstrap_samples > 0
    splice = responses.filter(
        (pl.col("response_role") == "primary")
        & (pl.col("perturbation_type") == "splice_saturation")
    )
    context_rows: list[dict[str, Any]] = []
    keys = ["class", "analysis_feature_id", "context_group", "source_panel_row"]
    for key, frame in splice.group_by(keys, maintain_order=True):
        class_name = str(key[0])
        target_positions = SPLICE_TARGET_POSITIONS[class_name]
        target = frame.filter(pl.col("relative_position").is_in(target_positions))
        other = frame.filter(~pl.col("relative_position").is_in(target_positions))
        assert target.height == len(target_positions) * 3
        assert other.height > 0
        target_mean = float(target["response_score"].mean())
        other_mean = float(other["response_score"].mean())
        context_rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "target_positions": ",".join(map(str, sorted(target_positions))),
                "target_mean_response": target_mean,
                "other_mean_response": other_mean,
                "target_minus_other": target_mean - other_mean,
            }
        )
    contexts = pl.DataFrame(context_rows).sort(keys)

    summary_rows: list[dict[str, Any]] = []
    summary_keys = ["class", "analysis_feature_id", "context_group", "target_positions"]
    for group_index, (key, frame) in enumerate(
        contexts.group_by(summary_keys, maintain_order=True)
    ):
        values = frame["target_minus_other"].to_numpy()
        assert len(values) == EXPECTED_CONTEXTS_PER_CLASS
        low, high = bootstrap_mean_interval(
            values,
            seed=RANDOM_SEED + group_index,
            samples=bootstrap_samples,
        )
        summary_rows.append(
            {
                **dict(zip(summary_keys, key, strict=True)),
                "contexts": frame.height,
                "mean_target_response": frame["target_mean_response"].mean(),
                "mean_other_response": frame["other_mean_response"].mean(),
                "mean_target_minus_other": float(values.mean()),
                "se_target_minus_other": float(
                    values.std(ddof=1) / np.sqrt(len(values))
                ),
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
            }
        )
    summary = pl.DataFrame(summary_rows).sort(summary_keys)
    return contexts, summary


def plot_splice_replication(position: pl.DataFrame, output_dir: Path) -> None:
    """Plot the frozen feature response across transcript-oriented positions."""

    classes = position["class"].unique(maintain_order=True).to_list()
    figure, axes = plt.subplots(1, len(classes), figsize=(12, 4), sharey=False)
    for axis, class_name in zip(np.atleast_1d(axes), classes, strict=True):
        frame = position.filter(pl.col("class") == class_name).sort("relative_position")
        x = frame["relative_position"].to_numpy()
        y = frame["mean_response"].to_numpy()
        se = frame["se_response"].to_numpy()
        axis.plot(x, y, marker="o", markersize=3, color="#4C78A8")
        axis.fill_between(x, y - se, y + se, color="#4C78A8", alpha=0.18)
        for position_value in SPLICE_TARGET_POSITIONS[class_name]:
            axis.axvline(position_value, color="#E45756", linestyle=":", linewidth=1)
        axis.set_title(class_name.removesuffix("_variant"))
        axis.set_xlabel("Transcript-oriented mutated position")
    np.atleast_1d(axes)[0].set_ylabel("Mean frozen SAE response (band = ±1 SE)")
    figure.suptitle("Untouched-context splice saturation replication")
    figure.tight_layout()
    figure.savefig(output_dir / "splice_replication.svg", bbox_inches="tight")
    figure.savefig(output_dir / "splice_replication.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze_heldout_perturbations(
    *,
    panel_path: Path,
    design_manifest_path: Path,
    extraction_dir: Path,
    output_dir: Path,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Evaluate the frozen causal hypotheses on untouched hash-sampled contexts."""

    assert bootstrap_samples > 0 and not output_dir.exists()
    analysis_commit = os.environ.get("HELDOUT_ANALYSIS_COMMIT", "")
    assert_current_commit(analysis_commit)
    design_manifest = json.loads(design_manifest_path.read_text())
    extraction_manifest_path = extraction_dir / "manifest.json"
    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    panel = pl.read_parquet(panel_path)
    reference_indices, alternate_indices, forward, reverse = validate_heldout_inputs(
        panel,
        design_manifest,
        extraction_manifest,
        panel_path=panel_path,
        design_manifest_path=design_manifest_path,
        extraction_dir=extraction_dir,
    )
    responses = build_response_frame(
        panel,
        reference_indices=reference_indices,
        alternate_indices=alternate_indices,
        forward_states=forward,
        reverse_states=reverse,
    )
    splice_position, splice_substitution = splice_summaries(responses)
    splice_contexts, splice_contrast = splice_context_contrasts(
        responses, bootstrap_samples=bootstrap_samples
    )
    codon_consequence, codon_identity = codon_summaries(responses)
    codon_input = responses.filter(
        (pl.col("response_role") == "primary")
        | (
            (pl.col("analysis_feature_id") == 4281)
            & (pl.col("class") == "stop_gained")
        )
    )
    codon_contexts = codon_context_selectivity(codon_input, edit_distance=1)
    codon_selectivity = summarize_selectivity(
        codon_contexts, bootstrap_samples=bootstrap_samples
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "perturbation_responses.parquet": responses,
        "splice_position_summary.parquet": splice_position,
        "splice_substitution_summary.parquet": splice_substitution,
        "splice_context_contrasts.parquet": splice_contexts,
        "splice_contrast_summary.parquet": splice_contrast,
        "codon_consequence_summary.parquet": codon_consequence,
        "codon_identity_summary.parquet": codon_identity,
        "codon_context_selectivity.parquet": codon_contexts,
        "codon_selectivity_summary.parquet": codon_selectivity,
    }
    for filename, frame in outputs.items():
        frame.write_parquet(output_dir / filename)
    plot_splice_replication(splice_position, output_dir)
    plot_codon_selectivity(codon_selectivity, output_dir)
    result = {
        "issue": ISSUE,
        "heldout_analysis_commit": analysis_commit,
        "design_manifest_sha256": sha256_file(design_manifest_path),
        "extraction_manifest_sha256": sha256_file(extraction_manifest_path),
        "protocol": {
            "primary_specs": PRIMARY_SPECS,
            "comparison_spec": COMPARISON_SPEC,
            "splice_target_positions": {
                key: sorted(value) for key, value in SPLICE_TARGET_POSITIONS.items()
            },
            "bootstrap_unit": "source genomic context",
            "bootstrap_samples": bootstrap_samples,
            "selection": "features, reducers, motif positions, coding targets, source split, and hash sampling frozen before held-out extraction",
        },
        "rows": {
            "panel": panel.height,
            "responses": responses.height,
            "source_contexts": panel.select("source_panel_row").n_unique(),
            "unique_states": int(forward.shape[0]),
        },
        "splice_contrast": splice_contrast.to_dicts(),
        "codon_selectivity": codon_selectivity.to_dicts(),
    }
    write_json(output_dir / "results.json", result)
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {**result, "artifacts": artifacts}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--design-manifest", type=Path, required=True)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    args = parser.parse_args()
    manifest = analyze_heldout_perturbations(
        panel_path=args.panel,
        design_manifest_path=args.design_manifest,
        extraction_dir=args.extraction_dir,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(manifest["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
