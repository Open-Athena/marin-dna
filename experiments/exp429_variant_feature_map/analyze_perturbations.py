"""Analyze issue #429 splice saturation and full-codon causal sweeps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from analyze import sha256_file, write_json
from analyze_spatial import (
    aligned_orientation_profile,
    oriented_profile,
    spatial_scores,
)
from sample_panel import assert_current_commit

ISSUE = 429
RANDOM_SEED = 429
EXPECTED_ROWS = 7_296
EXPECTED_STATES = 7_424
EXPECTED_RADIUS = 15
EXPECTED_POSITIONS = 2 * EXPECTED_RADIUS + 1
FEATURE_IDS = (3312, 4281, 6072, 11681, 11698)
PRIMARY_SPECS: dict[str, dict[str, Any]] = {
    "splice_acceptor_variant": {
        "feature_id": 11698,
        "orientation": "max_absolute",
        "transform": "signed",
        "direction": 1,
        "spatial_metric": "local_max",
    },
    "splice_donor_5th_base_variant": {
        "feature_id": 11681,
        "orientation": "max_absolute",
        "transform": "signed",
        "direction": 1,
        "spatial_metric": "focal",
    },
    "stop_gained": {
        "feature_id": 3312,
        "orientation": "max_absolute",
        "transform": "signed",
        "direction": 1,
        "spatial_metric": "local_max",
    },
    "synonymous_variant": {
        "feature_id": 6072,
        "orientation": "max_absolute",
        "transform": "signed",
        "direction": 1,
        "spatial_metric": "focal",
    },
}
COMPARISON_SPEC: dict[str, Any] = {
    "feature_id": 4281,
    "orientation": "forward",
    "transform": "signed",
    "direction": 1,
    "spatial_metric": "local_sum",
}


def candidate_response(
    forward: np.ndarray,
    reverse_complement: np.ndarray,
    *,
    orientation: Literal["forward", "reverse_complement", "mean", "max_absolute"],
    transform: Literal["signed", "absolute"],
    direction: int,
    spatial_metric: Literal["focal", "local_max", "local_sum"],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one frozen candidate reducer and return score and peak position."""

    profile = oriented_profile(
        aligned_orientation_profile(forward, reverse_complement, orientation),
        transform=transform,
        direction=direction,
    )
    scores = spatial_scores(profile, spatial_metric)
    peaks = np.argmax(profile, axis=1).astype(np.int64) - profile.shape[1] // 2
    assert scores.shape == peaks.shape == (forward.shape[0],)
    return scores, peaks


def bootstrap_mean_interval(
    values: np.ndarray, *, seed: int, samples: int
) -> tuple[float, float]:
    """Return a deterministic context-bootstrap 95% interval for a mean."""

    assert values.ndim == 1 and len(values) >= 2 and samples > 0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def codon_context_selectivity(
    responses: pl.DataFrame, *, edit_distance: int | None = None
) -> pl.DataFrame:
    """Compute within-context semantic-target versus other-codon responses."""

    coding = responses.filter(
        pl.col("perturbation_type").is_in(["codon_sweep", "coding_one_edit"])
    )
    if edit_distance is not None:
        assert edit_distance == 1 or edit_distance == 2 or edit_distance == 3
        coding = coding.filter(pl.col("edit_distance") == edit_distance)
    rows: list[dict[str, Any]] = []
    keys = [
        "analysis_feature_id",
        "response_role",
        "class",
        "context_group",
        "source_panel_row",
    ]
    for key, frame in coding.group_by(keys, maintain_order=True):
        feature_id = int(key[0])
        target_consequence = (
            "synonymous_variant" if feature_id == 6072 else "stop_gained"
        )
        target = frame.filter(pl.col("expected_consequence") == target_consequence)
        other = frame.filter(pl.col("expected_consequence") != target_consequence)
        assert target.height > 0 and other.height > 0
        target_mean = target["response_score"].mean()
        other_mean = other["response_score"].mean()
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "target_consequence": target_consequence,
                "target_states": target.height,
                "other_states": other.height,
                "target_mean_response": target_mean,
                "other_mean_response": other_mean,
                "target_minus_other": target_mean - other_mean,
            }
        )
    return pl.DataFrame(rows).sort(keys)


def summarize_selectivity(
    contexts: pl.DataFrame, *, bootstrap_samples: int
) -> pl.DataFrame:
    """Summarize context-level causal selectivity with context bootstrap CIs."""

    keys = [
        "analysis_feature_id",
        "response_role",
        "class",
        "context_group",
        "target_consequence",
    ]
    rows: list[dict[str, Any]] = []
    for group_index, (key, frame) in enumerate(
        contexts.group_by(keys, maintain_order=True)
    ):
        values = frame["target_minus_other"].to_numpy()
        low, high = bootstrap_mean_interval(
            values,
            seed=RANDOM_SEED + group_index,
            samples=bootstrap_samples,
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
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
    return pl.DataFrame(rows).sort(keys)


def validate_inputs(
    panel: pl.DataFrame,
    design_manifest: dict[str, Any],
    extraction_manifest: dict[str, Any],
    *,
    panel_path: Path,
    design_manifest_path: Path,
    extraction_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate manifests and load the exact state arrays and pair indices."""

    assert panel.height == EXPECTED_ROWS
    assert panel["perturbation_row"].to_list() == list(range(EXPECTED_ROWS))
    assert design_manifest["artifacts"][panel_path.name]["sha256"] == sha256_file(
        panel_path
    )
    assert extraction_manifest["design"]["manifest_sha256"] == sha256_file(
        design_manifest_path
    )
    assert extraction_manifest["design"]["panel_sha256"] == sha256_file(panel_path)
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
    assert reference_indices.shape == alternate_indices.shape == (EXPECTED_ROWS,)
    assert (
        forward.shape
        == reverse.shape
        == (
            EXPECTED_STATES,
            EXPECTED_POSITIONS,
            len(FEATURE_IDS),
        )
    )
    assert reference_indices.min() >= 0 and alternate_indices.min() >= 0
    assert reference_indices.max() < EXPECTED_STATES
    assert alternate_indices.max() < EXPECTED_STATES
    return reference_indices, alternate_indices, forward, reverse


def build_response_frame(
    panel: pl.DataFrame,
    *,
    reference_indices: np.ndarray,
    alternate_indices: np.ndarray,
    forward_states: np.ndarray,
    reverse_states: np.ndarray,
) -> pl.DataFrame:
    """Apply frozen feature reducers to every designed pair."""

    feature_index = {feature_id: index for index, feature_id in enumerate(FEATURE_IDS)}
    primary_frames: list[pl.DataFrame] = []
    for class_name, spec in PRIMARY_SPECS.items():
        rows = panel.filter(pl.col("class") == class_name)
        row_indices = rows["perturbation_row"].to_numpy()
        index = feature_index[int(spec["feature_id"])]
        forward = (
            forward_states[alternate_indices[row_indices], :, index]
            - forward_states[reference_indices[row_indices], :, index]
        )
        reverse = (
            reverse_states[alternate_indices[row_indices], :, index]
            - reverse_states[reference_indices[row_indices], :, index]
        )
        scores, peaks = candidate_response(
            np.asarray(forward),
            np.asarray(reverse),
            **{k: v for k, v in spec.items() if k != "feature_id"},
        )
        primary_frames.append(
            rows.drop("reference_sequence", "alternate_sequence").with_columns(
                pl.lit(int(spec["feature_id"])).alias("analysis_feature_id"),
                pl.lit("primary").alias("response_role"),
                pl.Series("response_score", scores),
                pl.Series("response_peak_relative_position", peaks),
            )
        )
    primary = pl.concat(primary_frames, how="vertical").sort("perturbation_row")
    assert primary.height == panel.height

    coding = panel.filter(
        pl.col("perturbation_type").is_in(["codon_sweep", "coding_one_edit"])
    )
    row_indices = coding["perturbation_row"].to_numpy()
    comparison_index = feature_index[int(COMPARISON_SPEC["feature_id"])]
    forward = (
        forward_states[alternate_indices[row_indices], :, comparison_index]
        - forward_states[reference_indices[row_indices], :, comparison_index]
    )
    reverse = (
        reverse_states[alternate_indices[row_indices], :, comparison_index]
        - reverse_states[reference_indices[row_indices], :, comparison_index]
    )
    scores, peaks = candidate_response(
        np.asarray(forward),
        np.asarray(reverse),
        **{k: v for k, v in COMPARISON_SPEC.items() if k != "feature_id"},
    )
    comparison = coding.drop("reference_sequence", "alternate_sequence").with_columns(
        pl.lit(int(COMPARISON_SPEC["feature_id"])).alias("analysis_feature_id"),
        pl.lit("stop_feature_comparison").alias("response_role"),
        pl.Series("response_score", scores),
        pl.Series("response_peak_relative_position", peaks),
    )
    responses = pl.concat((primary, comparison), how="vertical").sort(
        ["analysis_feature_id", "class", "perturbation_row"]
    )
    assert responses.filter(~pl.col("response_score").is_finite()).is_empty()
    return responses


def splice_summaries(
    responses: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Summarize position importance and base-specific splice mutations."""

    splice = responses.filter(
        (pl.col("response_role") == "primary")
        & (pl.col("perturbation_type") == "splice_saturation")
    )
    position = (
        splice.group_by(
            "class", "analysis_feature_id", "context_group", "relative_position"
        )
        .agg(
            pl.len().alias("mutations"),
            pl.col("source_panel_row").n_unique().alias("contexts"),
            pl.col("response_score").mean().alias("mean_response"),
            pl.col("response_score").median().alias("median_response"),
            (
                pl.col("response_score").std() / pl.col("response_score").count().sqrt()
            ).alias("se_response"),
        )
        .sort(["class", "context_group", "relative_position"])
    )
    substitution = (
        splice.group_by(
            "class",
            "analysis_feature_id",
            "context_group",
            "relative_position",
            "source_state",
            "alternate_state",
        )
        .agg(
            pl.len().alias("mutations"),
            pl.col("response_score").mean().alias("mean_response"),
            pl.col("response_score").median().alias("median_response"),
        )
        .sort(
            [
                "class",
                "context_group",
                "relative_position",
                "source_state",
                "alternate_state",
            ]
        )
    )
    return position, substitution


def codon_summaries(
    responses: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Summarize response by designed consequence and alternate codon identity."""

    coding = responses.filter(
        pl.col("perturbation_type").is_in(["codon_sweep", "coding_one_edit"])
    )
    consequence = (
        coding.group_by(
            "analysis_feature_id",
            "response_role",
            "class",
            "context_group",
            "expected_consequence",
            "edit_distance",
        )
        .agg(
            pl.len().alias("states"),
            pl.col("source_panel_row").n_unique().alias("contexts"),
            pl.col("response_score").mean().alias("mean_response"),
            pl.col("response_score").median().alias("median_response"),
        )
        .sort(
            [
                "analysis_feature_id",
                "class",
                "context_group",
                "expected_consequence",
                "edit_distance",
            ]
        )
    )
    codon = (
        coding.group_by(
            "analysis_feature_id",
            "response_role",
            "class",
            "context_group",
            "alternate_codon",
            "alternate_amino_acid",
            "expected_consequence",
        )
        .agg(
            pl.len().alias("contexts"),
            pl.col("response_score").mean().alias("mean_response"),
            pl.col("response_score").median().alias("median_response"),
        )
        .sort(
            ["analysis_feature_id", "class", "context_group", "mean_response"],
            descending=[False, False, False, True],
        )
    )
    return consequence, codon


def plot_splice_positions(position: pl.DataFrame, output_dir: Path) -> None:
    """Plot mean causal response by mutated splice-relative position."""

    classes = position["class"].unique(maintain_order=True).to_list()
    figure, axes = plt.subplots(1, len(classes), figsize=(12, 4), sharey=False)
    axes = np.atleast_1d(axes)
    colors = {"top": "#E45756", "rank_spaced_control": "#4C78A8"}
    for axis, class_name in zip(axes, classes, strict=True):
        for context_group in ("top", "rank_spaced_control"):
            frame = position.filter(
                (pl.col("class") == class_name)
                & (pl.col("context_group") == context_group)
            ).sort("relative_position")
            x = frame["relative_position"].to_numpy()
            y = frame["mean_response"].to_numpy()
            se = frame["se_response"].to_numpy()
            label = "strong contexts" if context_group == "top" else "controls"
            axis.plot(
                x, y, marker="o", markersize=3, color=colors[context_group], label=label
            )
            axis.fill_between(
                x, y - se, y + se, color=colors[context_group], alpha=0.18
            )
        axis.axvline(0, color="#777777", linestyle=":", linewidth=1)
        axis.set_title(class_name.removesuffix("_variant"))
        axis.set_xlabel("Mutated position in response orientation")
    axes[0].set_ylabel("Mean frozen SAE response (band = ±1 SE)")
    axes[-1].legend(frameon=False)
    figure.suptitle("Causal splice-motif saturation")
    figure.tight_layout()
    figure.savefig(output_dir / "splice_saturation.svg", bbox_inches="tight")
    figure.savefig(output_dir / "splice_saturation.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_codon_selectivity(summary: pl.DataFrame, output_dir: Path) -> None:
    """Plot within-context semantic-target selectivity with bootstrap intervals."""

    frame = summary.sort(["analysis_feature_id", "class", "context_group"])
    labels = [
        f"f{row['analysis_feature_id']}\n{row['class'].removesuffix('_variant')}\n{row['context_group'].replace('rank_spaced_', '')}"
        for row in frame.iter_rows(named=True)
    ]
    x = np.arange(frame.height)
    values = frame["mean_target_minus_other"].to_numpy()
    low = frame["bootstrap_ci95_low"].to_numpy()
    high = frame["bootstrap_ci95_high"].to_numpy()
    figure, axis = plt.subplots(figsize=(max(10, frame.height * 0.8), 5))
    axis.errorbar(
        x,
        values,
        yerr=np.stack((values - low, high - values)),
        fmt="o",
        capsize=0,
        color="#4C78A8",
    )
    axis.axhline(0, color="#777777", linestyle="--", linewidth=1)
    axis.set_xticks(x, labels, rotation=45, ha="right")
    axis.set_ylabel("Target-codon mean response − other-codon mean response")
    axis.set_title("Causal codon selectivity (95% context-bootstrap interval)")
    figure.tight_layout()
    figure.savefig(output_dir / "codon_selectivity.svg", bbox_inches="tight")
    figure.savefig(output_dir / "codon_selectivity.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze_perturbations(
    *,
    panel_path: Path,
    design_manifest_path: Path,
    extraction_dir: Path,
    output_dir: Path,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Run the commit-pinned causal analysis and write compact tables and plots."""

    assert bootstrap_samples > 0 and not output_dir.exists()
    analysis_commit = os.environ.get("PERTURBATION_ANALYSIS_COMMIT", "")
    assert_current_commit(analysis_commit)
    design_manifest = json.loads(design_manifest_path.read_text())
    extraction_manifest_path = extraction_dir / "manifest.json"
    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    panel = pl.read_parquet(panel_path)
    reference_indices, alternate_indices, forward, reverse = validate_inputs(
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
    codon_consequence, codon_identity = codon_summaries(responses)
    selectivity_contexts = codon_context_selectivity(responses)
    selectivity_summary = summarize_selectivity(
        selectivity_contexts, bootstrap_samples=bootstrap_samples
    )
    one_edit_input = responses.filter(
        (pl.col("response_role") == "primary")
        | ((pl.col("analysis_feature_id") == 4281) & (pl.col("class") == "stop_gained"))
    )
    one_edit_contexts = codon_context_selectivity(one_edit_input, edit_distance=1)
    one_edit_summary = summarize_selectivity(
        one_edit_contexts, bootstrap_samples=bootstrap_samples
    )
    top_positions = (
        splice_position.filter(pl.col("context_group") == "top")
        .sort(["class", "mean_response"], descending=[False, True])
        .group_by("class", maintain_order=True)
        .head(5)
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "perturbation_responses.parquet": responses,
        "splice_position_summary.parquet": splice_position,
        "splice_substitution_summary.parquet": splice_substitution,
        "codon_consequence_summary.parquet": codon_consequence,
        "codon_identity_summary.parquet": codon_identity,
        "codon_context_selectivity.parquet": selectivity_contexts,
        "codon_selectivity_summary.parquet": selectivity_summary,
        "codon_one_edit_context_selectivity.parquet": one_edit_contexts,
        "codon_one_edit_selectivity_summary.parquet": one_edit_summary,
    }
    for filename, frame in outputs.items():
        frame.write_parquet(output_dir / filename)
    plot_splice_positions(splice_position, output_dir)
    plot_codon_selectivity(one_edit_summary, output_dir)
    result = {
        "issue": ISSUE,
        "perturbation_analysis_commit": analysis_commit,
        "design_manifest_sha256": sha256_file(design_manifest_path),
        "extraction_manifest_sha256": sha256_file(extraction_manifest_path),
        "protocol": {
            "primary_specs": PRIMARY_SPECS,
            "comparison_spec": COMPARISON_SPEC,
            "bootstrap_unit": "source genomic context",
            "bootstrap_samples": bootstrap_samples,
            "selection": "all features, reducers, contexts, positions, and codon states frozen before extraction",
        },
        "rows": {
            "panel": panel.height,
            "responses": responses.height,
            "codon_contexts": selectivity_contexts.height,
        },
        "top_splice_positions": top_positions.to_dicts(),
        "codon_selectivity": selectivity_summary.to_dicts(),
        "codon_one_edit_selectivity": one_edit_summary.to_dicts(),
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
    manifest = analyze_perturbations(
        panel_path=args.panel,
        design_manifest_path=args.design_manifest,
        extraction_dir=args.extraction_dir,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(manifest["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
