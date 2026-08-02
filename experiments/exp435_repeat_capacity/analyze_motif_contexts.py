"""Describe sequence contexts for the frozen repeat-feature interpretation panel.

This is a post-hoc interpretation stage. It does not read Mendelian labels and
does not select new feature IDs from sequence-context results.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import ISSUE, assert_commit, sha256_file, write_json
from extract_common import CONTEXTS, D_SAE, FOCAL_INDEX, ORIENTATIONS, WINDOW_BP
from motif_context_common import (
    MIN_CONTEXTS,
    MOTIF_CONTEXT_RUN_ID,
    MOTIF_RADIUS,
    PAIRED_ACTIVATION_MANIFEST_SHA256,
    PAIRED_ANALYSIS_ARCHIVE_SHA256,
    REFERENCE_ACTIVATION_ARCHIVE_SHA256,
    REFERENCE_ASSOCIATION_ARCHIVE_SHA256,
    SELECTED_FEATURES,
    TOP_CONTEXTS,
    TOP_VARIANTS,
    VARIANT_PANEL_ARCHIVE_SHA256,
    kmer_enrichment,
    match_controls,
    positional_enrichment,
    reverse_complement,
    select_top_contexts,
    sequence_consensus,
)
from variant_common import EXPECTED_VARIANTS

REFERENCE_ACTIVATION_RUN_ID = "dna-exp435-repeat-reference-activations-r1"
REFERENCE_ASSOCIATION_RUN_ID = "dna-exp435-repeat-reference-associations-r1"
REFERENCE_PANEL_RUN_ID = "dna-exp435-repeat-reference-panel-r1"
VARIANT_PANEL_RUN_ID = "dna-exp435-repeat-variant-panel-r1"
PAIRED_ACTIVATION_RUN_ID = "dna-exp436-mendelian-focal-seed288-r1"
PAIRED_ANALYSIS_RUN_ID = "dna-exp435-repeat-variant-deltas-r1"

BLOCK_TO_ARM = {1: "block01-25m", 10: "block10-25m", 19: "block19-25m"}
BASE_ORDER = ("A", "C", "G", "T")


def verify_archive(
    root: Path,
    *,
    expected_sha256: str,
    expected_run_id: str,
) -> dict[str, Any]:
    manifest_path = root / "archive_manifest.json"
    assert manifest_path.is_file(), manifest_path
    assert sha256_file(manifest_path) == expected_sha256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["issue"] == ISSUE
    assert manifest["run_id"] == expected_run_id
    for relative, expected in manifest["artifacts"].items():
        path = root / relative
        assert path.is_file(), path
        assert path.stat().st_size == expected["bytes"]
        assert sha256_file(path) == expected["sha256"]
    return manifest


def verify_paired_activations(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    assert manifest_path.is_file()
    assert sha256_file(manifest_path) == PAIRED_ACTIVATION_MANIFEST_SHA256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["issue"] == 436
    assert manifest["run_id"] == PAIRED_ACTIVATION_RUN_ID
    for arm in BLOCK_TO_ARM.values():
        for orientation in ORIENTATIONS:
            relative = f"{arm}/sae_focal_{orientation}.parquet"
            expected = manifest["artifacts"][relative]
            path = root / relative
            assert path.is_file() and path.stat().st_size == expected["bytes"]
            assert sha256_file(path) == expected["sha256"]
    return manifest


def feature_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "block": item.block,
                "arm": BLOCK_TO_ARM[item.block],
                "feature_id": item.feature_id,
                "selection_reason": item.reason,
            }
            for item in SELECTED_FEATURES
        ],
        schema_overrides={"block": pl.UInt8, "feature_id": pl.UInt32},
    )


def validate_selection_provenance(
    reference_association_root: Path,
    paired_analysis_root: Path,
    selected: pl.DataFrame,
) -> None:
    reference = pl.read_parquet(
        reference_association_root / "associations" / "top_hits.parquet",
        columns=["block", "feature_id"],
    ).unique()
    paired = pl.read_parquet(
        paired_analysis_root / "paired" / "top_hits.parquet",
        columns=["block", "feature_id"],
    ).unique()
    observed = pl.concat([reference, paired]).unique()
    missing = selected.join(observed, on=["block", "feature_id"], how="anti")
    assert missing.is_empty(), missing


def reference_contexts(reference_activation_root: Path) -> pl.DataFrame:
    path = reference_activation_root / "inputs" / "panel" / "panel" / "contexts.parquet"
    frame = pl.read_parquet(path).sort("context_id")
    assert frame.height == CONTEXTS
    assert frame["context_id"].to_list() == list(range(CONTEXTS))
    assert frame["sequence"].str.len_chars().unique().to_list() == [WINDOW_BP]
    assert "label" not in frame.columns
    return frame


def reference_activation_table(
    root: Path,
    *,
    arm: str,
    orientation: str,
    feature_ids: list[int],
) -> pl.DataFrame:
    path = root / "extraction" / arm / f"sae_focal_{orientation}.parquet"
    frame = (
        pl.scan_parquet(path).filter(pl.col("feature_id").is_in(feature_ids)).collect()
    )
    assert {"context_id", "feature_id", "activation"} == set(frame.columns)
    assert frame.filter(
        (pl.col("context_id") >= CONTEXTS)
        | (pl.col("feature_id") >= D_SAE)
        | (pl.col("activation") <= 0)
        | ~pl.col("activation").is_finite()
    ).is_empty()
    assert (
        frame.select(pl.struct("context_id", "feature_id").n_unique()).item()
        == frame.height
    )
    return frame


def metadata_summary(frame: pl.DataFrame) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for level, column in (
        ("class", "repeat_class"),
        ("family", "family_label"),
        ("subfamily", "subfamily_label"),
    ):
        current = (
            frame.with_columns(pl.col(column).fill_null("non_repeat"))
            .group_by("block", "feature_id", "orientation", "role", column)
            .agg(
                pl.len().alias("contexts"),
                pl.col("activation").drop_nulls().mean().alias("mean_activation"),
            )
            .rename({column: "category"})
            .with_columns(
                pl.lit(level).alias("level"),
                (
                    pl.col("contexts")
                    / pl.col("contexts")
                    .sum()
                    .over("block", "feature_id", "orientation", "role")
                ).alias("fraction"),
            )
            .select(
                "block",
                "feature_id",
                "orientation",
                "role",
                "level",
                "category",
                "contexts",
                "fraction",
                "mean_activation",
            )
        )
        frames.append(current)
    return pl.concat(frames).sort(
        "block",
        "feature_id",
        "orientation",
        "role",
        "level",
        "contexts",
        descending=[False, False, False, False, False, True],
    )


def top_category(frame: pl.DataFrame, column: str) -> str:
    current = (
        frame.with_columns(pl.col(column).fill_null("non_repeat"))
        .group_by(column)
        .len()
        .sort("len", column, descending=[True, False])
    )
    assert current.height
    return f"{current[column][0]} ({int(current['len'][0])}/{frame.height})"


def motif_rows(
    *,
    contexts: pl.DataFrame,
    activations: pl.DataFrame,
    block: int,
    feature_id: int,
    orientation: str,
    reason: str,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    dict[str, Any],
]:
    selected = select_top_contexts(activations, feature_id=feature_id)
    if selected.height < MIN_CONTEXTS:
        return (
            pl.DataFrame(),
            pl.DataFrame(),
            pl.DataFrame(),
            {
                "block": block,
                "arm": BLOCK_TO_ARM[block],
                "feature_id": feature_id,
                "orientation": orientation,
                "selection_reason": reason,
                "status": "insufficient_reference_support",
                "reference_nonzero_contexts": selected.height,
            },
        )
    top = selected.join(contexts, on="context_id", how="inner", validate="1:1")
    assert top.height == selected.height
    controls = match_controls(
        contexts,
        top["context_id"].to_list(),
        namespace=f"{MOTIF_CONTEXT_RUN_ID}|b{block}|f{feature_id}|{orientation}",
    )
    control = (
        controls.join(
            contexts,
            left_on="control_context_id",
            right_on="context_id",
            how="inner",
            validate="1:1",
        )
        .rename({"control_context_id": "context_id"})
        .with_columns(pl.lit(None).cast(pl.Float64).alias("activation"))
    )
    assert control.height == top.height

    top = top.with_columns(
        pl.lit("top").alias("role"),
        pl.lit(None).cast(pl.UInt32).alias("matched_top_context_id"),
        pl.lit(None).cast(pl.String).alias("match_level"),
    )
    control = control.with_columns(
        pl.lit("control").alias("role"),
        pl.col("top_context_id").cast(pl.UInt32).alias("matched_top_context_id"),
    )
    shared_columns = [
        "context_id",
        "chrom",
        "pos0",
        "sequence",
        "is_repeat",
        "repeat_strand",
        "repeat_name",
        "repeat_class",
        "repeat_family",
        "family_label",
        "subfamily_label",
        "milli_div",
        "boundary_distance",
        "overlap_count",
        "gc_fraction",
        "gc_bin",
        "cpg_count",
        "shannon_entropy",
        "max_homopolymer",
        "repeat_fraction",
        "activation",
        "role",
        "matched_top_context_id",
        "match_level",
    ]
    combined = pl.concat(
        [top.select(shared_columns), control.select(shared_columns)],
        how="vertical_relaxed",
    ).with_columns(
        pl.lit(block).cast(pl.UInt8).alias("block"),
        pl.lit(BLOCK_TO_ARM[block]).alias("arm"),
        pl.lit(feature_id).cast(pl.UInt32).alias("feature_id"),
        pl.lit(orientation).alias("orientation"),
        pl.lit(reason).alias("selection_reason"),
        pl.when(pl.lit(orientation) == "reverse_complement")
        .then(
            pl.col("sequence").map_elements(reverse_complement, return_dtype=pl.String)
        )
        .otherwise(pl.col("sequence"))
        .alias("model_sequence"),
    )
    top_sequences = combined.filter(pl.col("role") == "top")["model_sequence"].to_list()
    control_sequences = combined.filter(pl.col("role") == "control")[
        "model_sequence"
    ].to_list()
    motif_slice = slice(
        FOCAL_INDEX - MOTIF_RADIUS,
        FOCAL_INDEX + MOTIF_RADIUS + 1,
    )
    top_motif = [sequence[motif_slice] for sequence in top_sequences]
    control_motif = [sequence[motif_slice] for sequence in control_sequences]
    position = positional_enrichment(top_sequences, control_sequences).with_columns(
        pl.lit(block).cast(pl.UInt8).alias("block"),
        pl.lit(BLOCK_TO_ARM[block]).alias("arm"),
        pl.lit(feature_id).cast(pl.UInt32).alias("feature_id"),
        pl.lit(orientation).alias("orientation"),
    )
    kmers = kmer_enrichment(top_motif, control_motif).with_columns(
        pl.lit(block).cast(pl.UInt8).alias("block"),
        pl.lit(BLOCK_TO_ARM[block]).alias("arm"),
        pl.lit(feature_id).cast(pl.UInt32).alias("feature_id"),
        pl.lit(orientation).alias("orientation"),
    )
    significant_kmers = kmers.filter(
        (pl.col("q_value") < 0.05) & (pl.col("log2_odds") > 0)
    ).sort("log2_odds", "q_value", descending=[True, False])
    top_kmers = [
        f"{row['kmer']}:{float(row['log2_odds']):.2f}"
        for row in significant_kmers.head(10).iter_rows(named=True)
    ]
    top_only = combined.filter(pl.col("role") == "top")
    match_counts = {
        str(level): int(count)
        for level, count in combined.filter(pl.col("role") == "control")
        .group_by("match_level")
        .len()
        .sort("match_level")
        .iter_rows()
    }
    summary = {
        "block": block,
        "arm": BLOCK_TO_ARM[block],
        "feature_id": feature_id,
        "orientation": orientation,
        "selection_reason": reason,
        "status": "analyzed",
        "reference_nonzero_contexts": activations.filter(
            pl.col("feature_id") == feature_id
        ).height,
        "top_contexts": top_only.height,
        "maximum_activation": float(top_only["activation"].max()),
        "median_top_activation": float(top_only["activation"].median()),
        "consensus": sequence_consensus(position),
        "significant_position_base_tests": position.filter(
            (pl.col("q_value") < 0.05) & (pl.col("log2_odds").abs() >= 1)
        ).height,
        "significant_positive_kmers": significant_kmers.height,
        "top_enriched_kmers": ",".join(top_kmers),
        "top_repeat_class": top_category(top_only, "repeat_class"),
        "top_repeat_family": top_category(top_only, "family_label"),
        "top_repeat_subfamily": top_category(top_only, "subfamily_label"),
        "mean_repeat_fraction": float(top_only["repeat_fraction"].mean()),
        "mean_gc_fraction": float(top_only["gc_fraction"].mean()),
        "mean_entropy": float(top_only["shannon_entropy"].mean()),
        "mean_max_homopolymer": float(top_only["max_homopolymer"].mean()),
        "match_chrom_class_gc": match_counts.get("chrom_class_gc", 0),
        "match_class_gc": match_counts.get("class_gc", 0),
        "match_status_gc": match_counts.get("status_gc", 0),
        "match_status": match_counts.get("status", 0),
    }
    return combined, position, kmers, summary


def paired_activation_table(
    root: Path,
    *,
    arm: str,
    orientation: str,
    feature_ids: list[int],
) -> pl.DataFrame:
    path = root / arm / f"sae_focal_{orientation}.parquet"
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("feature_id").is_in(feature_ids))
        .select(
            "panel_row",
            "feature_id",
            "ref_activation",
            "alt_activation",
            "delta",
        )
        .collect()
    )
    assert frame.filter(
        (pl.col("panel_row") >= EXPECTED_VARIANTS)
        | (pl.col("feature_id") >= D_SAE)
        | (pl.col("ref_activation") < 0)
        | (pl.col("alt_activation") < 0)
        | ~pl.col("delta").is_finite()
    ).is_empty()
    assert frame.filter(
        (pl.col("delta") - (pl.col("alt_activation") - pl.col("ref_activation"))).abs()
        > 1e-5
    ).is_empty()
    return frame


def top_variant_rows(
    activation: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    block: int,
    feature_id: int,
    orientation: str,
    reason: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    current = activation.filter(
        (pl.col("feature_id") == feature_id) & (pl.col("delta") != 0)
    ).with_columns(pl.col("delta").abs().alias("abs_delta"))
    selected = (
        current.sort("abs_delta", "panel_row", descending=[True, False])
        .head(TOP_VARIANTS)
        .with_row_index("top_variant_rank")
    )
    if selected.is_empty():
        return pl.DataFrame(), {
            "block": block,
            "arm": BLOCK_TO_ARM[block],
            "feature_id": feature_id,
            "orientation": orientation,
            "selection_reason": reason,
            "paired_nonzero_variants": 0,
            "top_variants": 0,
        }
    result = selected.join(panel, on="panel_row", how="inner", validate="m:1").sort(
        "top_variant_rank"
    )
    assert result.height == selected.height and "label" not in result.columns
    result = result.with_columns(
        pl.lit(block).cast(pl.UInt8).alias("block"),
        pl.lit(BLOCK_TO_ARM[block]).alias("arm"),
        pl.lit(orientation).alias("orientation"),
        pl.lit(reason).alias("selection_reason"),
        pl.when(pl.col("delta") > 0)
        .then(pl.lit("positive"))
        .otherwise(pl.lit("negative"))
        .alias("delta_sign"),
        pl.when((pl.col("ref_activation") == 0) & (pl.col("alt_activation") > 0))
        .then(pl.lit("inactive_to_active"))
        .when((pl.col("ref_activation") > 0) & (pl.col("alt_activation") == 0))
        .then(pl.lit("active_to_inactive"))
        .otherwise(pl.lit("active_to_active_changed"))
        .alias("activation_transition"),
    )
    summary = {
        "block": block,
        "arm": BLOCK_TO_ARM[block],
        "feature_id": feature_id,
        "orientation": orientation,
        "selection_reason": reason,
        "paired_nonzero_variants": current.height,
        "top_variants": result.height,
        "maximum_abs_delta": float(result["abs_delta"].max()),
        "median_top_abs_delta": float(result["abs_delta"].median()),
        "focal_repeat_top_variants": result.filter(
            pl.col("position_status") == "focal_repeat"
        ).height,
        "near_repeat_top_variants": result.filter(
            pl.col("position_status") == "near_repeat"
        ).height,
        "repeat_free_top_variants": result.filter(
            pl.col("position_status") == "repeat_free_window"
        ).height,
    }
    return result, summary


def plot_feature(
    position: pl.DataFrame,
    *,
    block: int,
    feature_id: int,
    output_dir: Path,
) -> list[Path]:
    current = position.filter(
        (pl.col("block") == block) & (pl.col("feature_id") == feature_id)
    )
    if current.is_empty():
        return []
    figure, axes = plt.subplots(2, 1, figsize=(12, 4.8), sharex=True)
    image = None
    for axis, orientation in zip(axes, ORIENTATIONS, strict=True):
        view = current.filter(pl.col("orientation") == orientation)
        if view.is_empty():
            axis.text(0.5, 0.5, "support < 32", ha="center", va="center")
            axis.set_axis_off()
            continue
        matrix = np.asarray(
            [
                view.filter(pl.col("base") == base)
                .sort("offset")["log2_odds"]
                .to_numpy()
                for base in BASE_ORDER
            ]
        )
        image = axis.imshow(
            np.clip(matrix, -4, 4),
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-4,
            vmax=4,
            extent=[-MOTIF_RADIUS - 0.5, MOTIF_RADIUS + 0.5, 3.5, -0.5],
        )
        axis.axvline(0, color="black", linewidth=0.7)
        axis.set_yticks(range(4), BASE_ORDER)
        axis.set_ylabel(orientation.replace("_", " "))
    axes[-1].set_xlabel("offset from focal base in model-input orientation")
    figure.suptitle(
        f"Block {block} feature {feature_id}: top-context nucleotide log2 odds"
    )
    if image is not None:
        figure.colorbar(image, ax=axes, label="log2 odds vs matched controls")
    figure.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.12, hspace=0.32)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"block{block:02d}-feature{feature_id:05d}"
    svg = stem.with_suffix(".svg")
    png = stem.with_suffix(".png")
    figure.savefig(svg)
    figure.savefig(png, dpi=180)
    plt.close(figure)
    return [svg, png]


def artifact_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def analyze(
    *,
    reference_activation_root: Path,
    reference_association_root: Path,
    variant_panel_root: Path,
    paired_activation_root: Path,
    paired_analysis_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert not output_dir.exists()
    assert os.environ.get("RUN_ID") == MOTIF_CONTEXT_RUN_ID
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    started = time.perf_counter()

    reference_activation_manifest = verify_archive(
        reference_activation_root,
        expected_sha256=REFERENCE_ACTIVATION_ARCHIVE_SHA256,
        expected_run_id=REFERENCE_ACTIVATION_RUN_ID,
    )
    reference_association_manifest = verify_archive(
        reference_association_root,
        expected_sha256=REFERENCE_ASSOCIATION_ARCHIVE_SHA256,
        expected_run_id=REFERENCE_ASSOCIATION_RUN_ID,
    )
    variant_panel_manifest = verify_archive(
        variant_panel_root,
        expected_sha256=VARIANT_PANEL_ARCHIVE_SHA256,
        expected_run_id=VARIANT_PANEL_RUN_ID,
    )
    paired_activation_manifest = verify_paired_activations(paired_activation_root)
    paired_analysis_manifest = verify_archive(
        paired_analysis_root,
        expected_sha256=PAIRED_ANALYSIS_ARCHIVE_SHA256,
        expected_run_id=PAIRED_ANALYSIS_RUN_ID,
    )

    selected = feature_frame()
    validate_selection_provenance(
        reference_association_root, paired_analysis_root, selected
    )
    contexts = reference_contexts(reference_activation_root)
    panel = pl.read_parquet(variant_panel_root / "panel" / "variant_panel.parquet")
    assert panel.height == EXPECTED_VARIANTS
    assert panel["panel_row"].to_list() == list(range(EXPECTED_VARIANTS))
    assert "label" not in panel.columns

    output_dir.mkdir(parents=True)
    selected.write_parquet(output_dir / "selected_features.parquet", compression="zstd")
    context_frames: list[pl.DataFrame] = []
    position_frames: list[pl.DataFrame] = []
    kmer_frames: list[pl.DataFrame] = []
    feature_summaries: list[dict[str, Any]] = []
    variant_frames: list[pl.DataFrame] = []
    paired_summaries: list[dict[str, Any]] = []

    for block, arm in BLOCK_TO_ARM.items():
        block_selected = selected.filter(pl.col("block") == block)
        feature_ids = block_selected["feature_id"].to_list()
        reasons = {
            int(feature_id): str(reason)
            for feature_id, reason in block_selected.select(
                "feature_id", "selection_reason"
            ).iter_rows()
        }
        for orientation in ORIENTATIONS:
            reference_activations = reference_activation_table(
                reference_activation_root,
                arm=arm,
                orientation=orientation,
                feature_ids=feature_ids,
            )
            paired_activations = paired_activation_table(
                paired_activation_root,
                arm=arm,
                orientation=orientation,
                feature_ids=feature_ids,
            )
            for feature_id in feature_ids:
                context_frame, position, kmers, feature_summary = motif_rows(
                    contexts=contexts,
                    activations=reference_activations,
                    block=block,
                    feature_id=int(feature_id),
                    orientation=orientation,
                    reason=reasons[int(feature_id)],
                )
                feature_summaries.append(feature_summary)
                if not context_frame.is_empty():
                    context_frames.append(context_frame)
                    position_frames.append(position)
                    kmer_frames.append(kmers)
                variants, paired_summary = top_variant_rows(
                    paired_activations,
                    panel,
                    block=block,
                    feature_id=int(feature_id),
                    orientation=orientation,
                    reason=reasons[int(feature_id)],
                )
                paired_summaries.append(paired_summary)
                if not variants.is_empty():
                    variant_frames.append(variants)

    assert context_frames and position_frames and kmer_frames and variant_frames
    top_contexts = pl.concat(context_frames, how="diagonal_relaxed")
    positions = pl.concat(position_frames, how="vertical_relaxed")
    kmers = pl.concat(kmer_frames, how="vertical_relaxed")
    feature_summary = pl.DataFrame(feature_summaries)
    top_variants = pl.concat(variant_frames, how="diagonal_relaxed")
    paired_summary = pl.DataFrame(paired_summaries)
    category_summary = metadata_summary(top_contexts)
    assert "label" not in top_variants.columns
    assert feature_summary.height == selected.height * len(ORIENTATIONS)
    assert paired_summary.height == selected.height * len(ORIENTATIONS)

    tables = {
        "top_contexts.parquet": top_contexts,
        "position_base_enrichment.parquet": positions,
        "kmer_enrichment.parquet": kmers,
        "context_category_summary.parquet": category_summary,
        "feature_summary.parquet": feature_summary,
        "top_variants.parquet": top_variants,
        "paired_summary.parquet": paired_summary,
    }
    for name, frame in tables.items():
        frame.write_parquet(output_dir / name, compression="zstd")

    plot_paths: list[Path] = []
    for block, feature_id in selected.select("block", "feature_id").iter_rows():
        plot_paths.extend(
            plot_feature(
                positions,
                block=int(block),
                feature_id=int(feature_id),
                output_dir=output_dir / "plots",
            )
        )

    artifacts = {
        str(path.relative_to(output_dir)): artifact_record(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    elapsed = time.perf_counter() - started
    result = {
        "issue": ISSUE,
        "run_id": MOTIF_CONTEXT_RUN_ID,
        "analysis_status": "posthoc_repeat_motif_and_context_description",
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed,
        "experiment_commit": experiment_commit,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "label_used": False,
        "selection_status": "posthoc_from_reported_reference_and_paired_associations",
        "protocol": {
            "layers": sorted(BLOCK_TO_ARM),
            "orientations": list(ORIENTATIONS),
            "selected_feature_ids": selected.height,
            "views": selected.height * len(ORIENTATIONS),
            "top_context_limit": TOP_CONTEXTS,
            "minimum_reference_support": MIN_CONTEXTS,
            "top_variant_limit": TOP_VARIANTS,
            "motif_radius": MOTIF_RADIUS,
            "model_sequence_policy": (
                "forward genomic sequence for FWD; reverse-complemented genomic "
                "sequence for RC"
            ),
            "position_fdr": "BH across all 63 positions x 4 bases per feature/orientation",
            "kmer_fdr": "BH across all supported 3-6-mers per feature/orientation",
            "consensus": "q<0.05 and positive log2 odds >=1; otherwise dot",
            "control_relaxation": [
                "chromosome+repeat class+GC bin",
                "repeat class+GC bin",
                "repeat status+GC bin",
                "repeat status",
            ],
        },
        "inputs": {
            "reference_activation_archive": {
                "run_id": reference_activation_manifest["run_id"],
                "manifest_sha256": REFERENCE_ACTIVATION_ARCHIVE_SHA256,
            },
            "reference_association_archive": {
                "run_id": reference_association_manifest["run_id"],
                "manifest_sha256": REFERENCE_ASSOCIATION_ARCHIVE_SHA256,
            },
            "variant_panel_archive": {
                "run_id": variant_panel_manifest["run_id"],
                "manifest_sha256": VARIANT_PANEL_ARCHIVE_SHA256,
            },
            "paired_activations": {
                "run_id": paired_activation_manifest["run_id"],
                "manifest_sha256": PAIRED_ACTIVATION_MANIFEST_SHA256,
            },
            "paired_analysis_archive": {
                "run_id": paired_analysis_manifest["run_id"],
                "manifest_sha256": PAIRED_ANALYSIS_ARCHIVE_SHA256,
            },
            "reference_panel": {
                "run_id": REFERENCE_PANEL_RUN_ID,
                "contexts": contexts.height,
            },
        },
        "outputs": {
            "selected_features": selected.height,
            "analyzed_views": feature_summary.filter(
                pl.col("status") == "analyzed"
            ).height,
            "top_context_rows": top_contexts.height,
            "position_tests": positions.height,
            "kmer_tests": kmers.height,
            "top_variant_rows": top_variants.height,
            "plots": len(plot_paths),
        },
        "artifacts": artifacts,
    }
    write_json(output_dir / "results.json", result)
    result["artifacts"]["results.json"] = artifact_record(output_dir / "results.json")
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-activation-root", type=Path, required=True)
    parser.add_argument("--reference-association-root", type=Path, required=True)
    parser.add_argument("--variant-panel-root", type=Path, required=True)
    parser.add_argument("--paired-activation-root", type=Path, required=True)
    parser.add_argument("--paired-analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        reference_activation_root=args.reference_activation_root,
        reference_association_root=args.reference_association_root,
        variant_panel_root=args.variant_panel_root,
        paired_activation_root=args.paired_activation_root,
        paired_analysis_root=args.paired_analysis_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
