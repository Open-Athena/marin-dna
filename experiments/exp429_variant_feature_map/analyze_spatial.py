"""Analyze selected SAE feature profiles around issue #429 variants.

Candidate feature IDs, response directions, and transforms are frozen by the
discovery/validation-only focal analysis. This follow-up uses validation blocks
to choose among three declared spatial summaries, then reports untouched-test
performance and position profiles.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from analyze import bootstrap_block_ap, sha256_file, write_json
from sample_panel import assert_current_commit

ISSUE = 429
EXPECTED_ROWS = 22_528
EXPECTED_CLASSES = 11
EXPECTED_SELECTED_ROWS = 44
EXPECTED_FEATURES = 18
EXPECTED_RADIUS = 15
EXPECTED_POSITIONS = 2 * EXPECTED_RADIUS + 1
RANDOM_SEED = 429
SPATIAL_METRICS = ("focal", "local_max", "local_sum")
ORIENTATIONS = ("forward", "reverse_complement", "mean", "max_absolute")


def aligned_orientation_profile(
    forward: np.ndarray,
    reverse_complement: np.ndarray,
    orientation: Literal["forward", "reverse_complement", "mean", "max_absolute"],
) -> np.ndarray:
    """Put RC positions in genomic order, then construct one orientation view."""

    assert forward.shape == reverse_complement.shape
    assert forward.ndim == 2 and forward.shape[1] % 2 == 1
    reverse_aligned = reverse_complement[:, ::-1]
    if orientation == "forward":
        output = forward
    elif orientation == "reverse_complement":
        output = reverse_aligned
    elif orientation == "mean":
        output = (forward + reverse_aligned) * 0.5
    elif orientation == "max_absolute":
        output = np.maximum(np.abs(forward), np.abs(reverse_aligned))
    else:
        raise AssertionError(orientation)
    assert output.shape == forward.shape and np.isfinite(output).all()
    return output


def oriented_profile(
    profile: np.ndarray,
    *,
    transform: Literal["signed", "absolute"],
    direction: int,
) -> np.ndarray:
    """Apply the feature transform and direction frozen before spatial analysis."""

    assert profile.ndim == 2 and direction in {-1, 1}
    if transform == "signed":
        transformed = profile
    elif transform == "absolute":
        transformed = np.abs(profile)
    else:
        raise AssertionError(transform)
    output = direction * transformed
    assert output.shape == profile.shape and np.isfinite(output).all()
    return output


def spatial_scores(
    profile: np.ndarray,
    metric: Literal["focal", "local_max", "local_sum"],
) -> np.ndarray:
    """Reduce a class-oriented position profile to one score per variant."""

    assert profile.ndim == 2 and profile.shape[1] % 2 == 1
    if metric == "focal":
        output = profile[:, profile.shape[1] // 2]
    elif metric == "local_max":
        output = profile.max(axis=1)
    elif metric == "local_sum":
        output = profile.sum(axis=1)
    else:
        raise AssertionError(metric)
    assert output.shape == (profile.shape[0],) and np.isfinite(output).all()
    return output


def choose_by_validation(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Select one row per key using validation AP with deterministic tie breaks."""

    metric_rank = {metric: index for index, metric in enumerate(SPATIAL_METRICS)}
    orientation_rank = {
        orientation: index for index, orientation in enumerate(ORIENTATIONS)
    }
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        choices = groups[key]
        best = min(
            choices,
            key=lambda row: (
                -float(row["validation_average_precision"]),
                orientation_rank.get(str(row["orientation"]), len(ORIENTATIONS)),
                metric_rank.get(str(row["spatial_metric"]), len(SPATIAL_METRICS)),
                int(row["dimension"]),
            ),
        )
        selected.append(dict(best))
    return selected


def validate_inputs(
    panel: pl.DataFrame,
    selected: pl.DataFrame,
    spatial_manifest: dict[str, Any],
    *,
    panel_path: Path,
    focal_analysis_dir: Path,
    spatial_dir: Path,
) -> list[int]:
    """Validate all frozen-input hashes, shapes, and split contracts."""

    assert panel.height == EXPECTED_ROWS
    assert panel["panel_row"].to_list() == list(range(EXPECTED_ROWS))
    assert panel["consequence_cre"].n_unique() == EXPECTED_CLASSES
    assert set(panel["split"].unique()) == {"discovery", "validation", "test"}
    assert selected.height == EXPECTED_SELECTED_ROWS
    required = {
        "class",
        "orientation",
        "transform",
        "dimension",
        "direction",
        "test_average_precision",
    }
    assert required <= set(selected.columns)
    assert set(selected["orientation"].unique()) == set(ORIENTATIONS)
    assert selected.group_by("class").len()["len"].unique().to_list() == [4]
    feature_ids = [int(value) for value in spatial_manifest["selection"]["feature_ids"]]
    assert feature_ids == sorted(set(selected["dimension"].to_list()))
    assert len(feature_ids) == EXPECTED_FEATURES
    assert spatial_manifest["panel"]["sha256"] == sha256_file(panel_path)
    assert spatial_manifest["panel"]["rows"] == EXPECTED_ROWS
    focal_manifest_path = focal_analysis_dir / "manifest.json"
    selected_path = focal_analysis_dir / "selected_individual_features.parquet"
    assert spatial_manifest["selection"]["analysis_manifest_sha256"] == sha256_file(
        focal_manifest_path
    )
    assert spatial_manifest["selection"]["selected_artifact_sha256"] == sha256_file(
        selected_path
    )
    for filename, metadata in spatial_manifest["artifacts"].items():
        path = spatial_dir / filename
        assert path.is_file(), path
        assert path.stat().st_size == metadata["bytes"]
        assert sha256_file(path) == metadata["sha256"]
    assert spatial_manifest["protocol"]["spatial_radius"] == EXPECTED_RADIUS
    assert spatial_manifest["protocol"]["relative_positions"] == list(
        range(-EXPECTED_RADIUS, EXPECTED_RADIUS + 1)
    )
    return feature_ids


def plot_spatial_ap(selected: pl.DataFrame, output_dir: Path) -> None:
    """Plot focal versus validation-selected spatial held-out AP with 95% CIs."""

    frame = selected.sort("class")
    labels = [value.removesuffix("_variant") for value in frame["class"]]
    x = np.arange(frame.height)
    width = 0.38
    figure, axis = plt.subplots(figsize=(13, 5.5))
    for offset, prefix, label, color in (
        (-width / 2, "focal", "focal base", "#4C78A8"),
        (width / 2, "spatial", "validation-selected spatial", "#F58518"),
    ):
        values = frame[f"test_{prefix}_average_precision"].to_numpy()
        low = frame[f"test_{prefix}_ap_ci95_low"].to_numpy()
        high = frame[f"test_{prefix}_ap_ci95_high"].to_numpy()
        yerr = np.stack((values - low, high - values))
        axis.bar(x + offset, values, width, label=label, color=color)
        axis.errorbar(
            x + offset,
            values,
            yerr=yerr,
            fmt="none",
            ecolor="#333333",
            capsize=2,
            linewidth=0.8,
        )
    axis.axhline(1 / EXPECTED_CLASSES, color="#666666", linestyle="--", linewidth=1)
    axis.set_xticks(x, labels, rotation=40, ha="right")
    axis.set_ylabel("Held-out one-vs-rest average precision")
    axis.set_title("Spatial SAE response can move away from the edited base")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "spatial_ap.svg", bbox_inches="tight")
    figure.savefig(output_dir / "spatial_ap.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_profiles(profiles: pl.DataFrame, output_dir: Path) -> None:
    """Plot held-out class-minus-background response profiles."""

    classes = profiles["class"].unique(maintain_order=True).to_list()
    figure, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True)
    for axis, class_name in zip(axes.flat, classes, strict=False):
        frame = profiles.filter(pl.col("class") == class_name).sort("relative_position")
        axis.plot(
            frame["relative_position"],
            frame["class_minus_background_mean"],
            color="#4C78A8",
            linewidth=1.8,
        )
        axis.axvline(0, color="#777777", linestyle=":", linewidth=1)
        axis.axhline(0, color="#BBBBBB", linewidth=0.8)
        axis.set_title(class_name.removesuffix("_variant"), fontsize=9)
    for axis in axes.flat[len(classes) :]:
        axis.set_visible(False)
    figure.supxlabel("Position relative to edited base (bp; genomic orientation)")
    figure.supylabel("Class mean − background mean oriented response")
    figure.suptitle("Held-out spatial profiles of validation-selected SAE features")
    figure.tight_layout()
    figure.savefig(output_dir / "spatial_profiles.svg", bbox_inches="tight")
    figure.savefig(output_dir / "spatial_profiles.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze_spatial(
    *,
    panel_path: Path,
    focal_analysis_dir: Path,
    spatial_dir: Path,
    output_dir: Path,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Select spatial summaries on validation and evaluate them on test blocks."""

    assert bootstrap_samples > 0 and not output_dir.exists()
    analysis_commit = os.environ.get("SPATIAL_ANALYSIS_COMMIT", "")
    assert_current_commit(analysis_commit)
    panel = pl.read_parquet(panel_path)
    selected_path = focal_analysis_dir / "selected_individual_features.parquet"
    selected = pl.read_parquet(selected_path)
    spatial_manifest_path = spatial_dir / "manifest.json"
    spatial_manifest = json.loads(spatial_manifest_path.read_text())
    feature_ids = validate_inputs(
        panel,
        selected,
        spatial_manifest,
        panel_path=panel_path,
        focal_analysis_dir=focal_analysis_dir,
        spatial_dir=spatial_dir,
    )
    feature_index = {feature_id: index for index, feature_id in enumerate(feature_ids)}
    arrays: dict[tuple[str, str], np.ndarray] = {}
    expected_shape = (EXPECTED_ROWS, EXPECTED_POSITIONS, EXPECTED_FEATURES)
    for orientation in ("forward", "reverse_complement"):
        for allele in ("ref", "alt"):
            path = spatial_dir / f"spatial_{allele}_{orientation}.npy"
            value = np.load(path, mmap_mode="r")
            assert value.shape == expected_shape and value.dtype == np.float32
            arrays[(orientation, allele)] = value

    labels = panel["consequence_cre"].to_numpy()
    split = panel["split"].to_numpy()
    blocks = panel["block_id"].to_numpy()
    validation = split == "validation"
    test = split == "test"
    assert validation.sum() == test.sum() == EXPECTED_CLASSES * 512

    metric_rows: list[dict[str, Any]] = []
    profile_cache: dict[tuple[str, str], np.ndarray] = {}
    for row in selected.iter_rows(named=True):
        dimension = int(row["dimension"])
        index = feature_index[dimension]
        forward = np.asarray(
            arrays[("forward", "alt")][:, :, index]
            - arrays[("forward", "ref")][:, :, index],
            dtype=np.float32,
        )
        reverse = np.asarray(
            arrays[("reverse_complement", "alt")][:, :, index]
            - arrays[("reverse_complement", "ref")][:, :, index],
            dtype=np.float32,
        )
        profile = oriented_profile(
            aligned_orientation_profile(forward, reverse, row["orientation"]),
            transform=row["transform"],
            direction=int(row["direction"]),
        )
        key = (str(row["class"]), str(row["orientation"]))
        assert key not in profile_cache
        profile_cache[key] = profile
        positive_validation = labels[validation] == row["class"]
        positive_test = labels[test] == row["class"]
        for metric in SPATIAL_METRICS:
            scores = spatial_scores(profile, metric)
            metric_rows.append(
                {
                    "class": row["class"],
                    "orientation": row["orientation"],
                    "transform": row["transform"],
                    "dimension": dimension,
                    "direction": int(row["direction"]),
                    "spatial_metric": metric,
                    "validation_average_precision": float(
                        average_precision_score(positive_validation, scores[validation])
                    ),
                    "test_average_precision": float(
                        average_precision_score(positive_test, scores[test])
                    ),
                }
            )
        focal_row = metric_rows[-len(SPATIAL_METRICS)]
        assert focal_row["spatial_metric"] == "focal"
        assert (
            abs(
                float(focal_row["test_average_precision"])
                - float(row["test_average_precision"])
            )
            < 1e-4
        )

    candidate_best = choose_by_validation(metric_rows, ("class", "orientation"))
    class_best = choose_by_validation(candidate_best, ("class",))
    selected_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    relative_positions = np.arange(-EXPECTED_RADIUS, EXPECTED_RADIUS + 1)
    for row in class_best:
        class_name = str(row["class"])
        profile = profile_cache[(class_name, str(row["orientation"]))]
        positive = labels[test] == class_name
        spatial_test_scores = spatial_scores(profile, row["spatial_metric"])[test]
        focal_test_scores = spatial_scores(profile, "focal")[test]
        spatial_low, spatial_high, n_blocks, positive_blocks = bootstrap_block_ap(
            spatial_test_scores,
            positive,
            blocks[test],
            seed=RANDOM_SEED + sum(map(ord, class_name)),
            samples=bootstrap_samples,
        )
        focal_low, focal_high, _, _ = bootstrap_block_ap(
            focal_test_scores,
            positive,
            blocks[test],
            seed=RANDOM_SEED + 10_000 + sum(map(ord, class_name)),
            samples=bootstrap_samples,
        )
        assert spatial_low is not None and spatial_high is not None
        assert focal_low is not None and focal_high is not None
        test_profile = profile[test]
        positive_mean = test_profile[positive].mean(axis=0, dtype=np.float64)
        negative_mean = test_profile[~positive].mean(axis=0, dtype=np.float64)
        excess = positive_mean - negative_mean
        peak_index = int(np.argmax(excess))
        positive_peak_positions = relative_positions[
            np.argmax(test_profile[positive], axis=1)
        ]
        selected_rows.append(
            {
                **row,
                "test_focal_average_precision": float(
                    average_precision_score(positive, focal_test_scores)
                ),
                "test_focal_ap_ci95_low": focal_low,
                "test_focal_ap_ci95_high": focal_high,
                "test_spatial_average_precision": float(
                    average_precision_score(positive, spatial_test_scores)
                ),
                "test_spatial_ap_ci95_low": spatial_low,
                "test_spatial_ap_ci95_high": spatial_high,
                "test_blocks": n_blocks,
                "test_positive_blocks": positive_blocks,
                "class_excess_peak_relative_position": int(
                    relative_positions[peak_index]
                ),
                "class_excess_at_focal": float(excess[EXPECTED_RADIUS]),
                "class_excess_at_peak": float(excess[peak_index]),
                "positive_fraction_peak_at_focal": float(
                    np.mean(positive_peak_positions == 0)
                ),
                "positive_median_absolute_peak_position": float(
                    np.median(np.abs(positive_peak_positions))
                ),
            }
        )
        for position, positive_value, negative_value, difference in zip(
            relative_positions,
            positive_mean,
            negative_mean,
            excess,
            strict=True,
        ):
            profile_rows.append(
                {
                    "class": class_name,
                    "orientation": row["orientation"],
                    "transform": row["transform"],
                    "dimension": int(row["dimension"]),
                    "direction": int(row["direction"]),
                    "spatial_metric": row["spatial_metric"],
                    "relative_position": int(position),
                    "class_mean": float(positive_value),
                    "background_mean": float(negative_value),
                    "class_minus_background_mean": float(difference),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_frame = pl.DataFrame(metric_rows).sort(
        ["class", "orientation", "spatial_metric"]
    )
    selected_frame = pl.DataFrame(selected_rows).sort("class")
    profiles_frame = pl.DataFrame(profile_rows).sort(["class", "relative_position"])
    assert metrics_frame.height == EXPECTED_SELECTED_ROWS * len(SPATIAL_METRICS)
    assert selected_frame.height == EXPECTED_CLASSES
    assert profiles_frame.height == EXPECTED_CLASSES * EXPECTED_POSITIONS
    metrics_frame.write_parquet(output_dir / "spatial_metrics.parquet")
    selected_frame.write_parquet(output_dir / "selected_spatial_features.parquet")
    profiles_frame.write_parquet(output_dir / "spatial_profiles.parquet")
    plot_spatial_ap(selected_frame, output_dir)
    plot_profiles(profiles_frame, output_dir)

    summary = {
        "issue": ISSUE,
        "spatial_analysis_commit": analysis_commit,
        "panel_sha256": sha256_file(panel_path),
        "focal_analysis_manifest_sha256": sha256_file(
            focal_analysis_dir / "manifest.json"
        ),
        "spatial_manifest_sha256": sha256_file(spatial_manifest_path),
        "classes": EXPECTED_CLASSES,
        "chance_one_vs_rest_ap": 1 / EXPECTED_CLASSES,
        "protocol": {
            "candidate_rule": "feature, direction, and transform frozen by discovery/validation focal analysis",
            "spatial_metric_selection": "validation AP only; deterministic ties focal, local_max, local_sum",
            "class_candidate_selection": "validation AP only after per-orientation spatial metric selection",
            "test_use": "one final report with genomic-block bootstrap intervals; no test reranking",
            "spatial_metrics": list(SPATIAL_METRICS),
            "radius": EXPECTED_RADIUS,
            "reverse_complement_alignment": "reverse the RC position axis into genomic coordinates before aggregation",
            "bootstrap_samples": bootstrap_samples,
        },
    }
    write_json(output_dir / "results.json", summary)
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {**summary, "artifacts": artifacts}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--focal-analysis-dir", type=Path, required=True)
    parser.add_argument("--spatial-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    args = parser.parse_args()
    manifest = analyze_spatial(
        panel_path=args.panel,
        focal_analysis_dir=args.focal_analysis_dir,
        spatial_dir=args.spatial_dir,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(manifest["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
