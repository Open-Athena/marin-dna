"""Interpret recurrent final-layer Mendelian-label SAE feature slots.

Candidate selection is outcome-aware and fixed in issue #436. Outputs from this script
are exploratory biological interpretation, not an independent confirmatory test.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy
from scipy.stats import pearsonr, spearmanr

from analyze_focal import PRIMARY_RESPONSES, RESPONSES, bh_adjust
from extract_focal import (
    BUDGETS,
    EXPECTED_ROWS,
    FOCAL_INDEX,
    ISSUE,
    ORIENTATIONS,
    WINDOW_BP,
    sha256_file,
    validate_panel,
    write_json,
)
from train import assert_commit

CANDIDATE_FEATURES = (9086, 4635, 7731, 1132, 10388, 12658)
TARGET_ORDER = (
    "overall",
    "missense_variant",
    "synonymous_variant",
    "splicing",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "tss_proximal",
    "non_coding_transcript_exon_variant",
    "distal",
)
TARGET_LABELS = {
    "overall": "Overall",
    "missense_variant": "Missense",
    "synonymous_variant": "Synonymous",
    "splicing": "Splicing",
    "5_prime_UTR_variant": "5′ UTR",
    "3_prime_UTR_variant": "3′ UTR",
    "tss_proximal": "TSS-proximal",
    "non_coding_transcript_exon_variant": "ncRNA exon",
    "distal": "Distal",
}
SCORE_OUTCOMES = ("minus_llr_avg", "probe_score")
CONTEXT_FLANK = 31
TOP_CONTEXTS = 50
DNA_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(sequence: str) -> str:
    assert set(sequence) <= set("ACGT")
    return sequence.translate(DNA_COMPLEMENT)[::-1]


def verify_manifest_artifacts(root: Path, manifest: dict[str, Any]) -> None:
    for relative, expected in manifest["artifacts"].items():
        path = root / relative
        assert path.is_file(), path
        assert path.stat().st_size == expected["bytes"], path
        assert sha256_file(path) == expected["sha256"], path


def read_annotations(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    contexts_path: Path,
) -> pl.DataFrame:
    panel_manifest = json.loads(panel_manifest_path.read_text())
    panel = pl.read_parquet(panel_path)
    validate_panel(panel, panel_manifest, panel_path)
    assert panel.height == EXPECTED_ROWS
    panel = panel.with_row_index("panel_row")

    contexts = pl.read_parquet(contexts_path).rename({"row_index": "panel_row"})
    required = {
        "panel_row",
        "ref_context",
        "alt_context",
        "flank_gc_count",
        "flank_gc_bin",
    }
    assert required <= set(contexts.columns)
    assert contexts.height == panel.height
    assert contexts["panel_row"].to_list() == panel["panel_row"].to_list()
    assert contexts.select(
        pl.col("ref_context").str.len_chars().unique()
    ).to_series().to_list() == [WINDOW_BP]
    assert contexts.select(
        pl.col("alt_context").str.len_chars().unique()
    ).to_series().to_list() == [WINDOW_BP]

    annotations = panel.join(contexts, on="panel_row", how="inner").with_columns(
        pl.concat_str("ref", "alt", separator=">").alias("substitution"),
        pl.col("ref_context")
        .str.slice(FOCAL_INDEX - 15, 31)
        .str.count_matches("[GC]")
        .alias("ref_local31_gc"),
        pl.col("alt_context")
        .str.slice(FOCAL_INDEX - 15, 31)
        .str.count_matches("[GC]")
        .alias("alt_local31_gc"),
        pl.col("ref_context")
        .str.slice(FOCAL_INDEX - 15, 31)
        .str.count_matches("CG")
        .alias("ref_local31_cpg"),
        pl.col("alt_context")
        .str.slice(FOCAL_INDEX - 15, 31)
        .str.count_matches("CG")
        .alias("alt_local31_cpg"),
    )
    assert annotations.height == EXPECTED_ROWS
    assert annotations.filter(
        pl.col("ref_context").str.slice(FOCAL_INDEX, 1) != pl.col("ref")
    ).is_empty()
    assert annotations.filter(
        pl.col("alt_context").str.slice(FOCAL_INDEX, 1) != pl.col("alt")
    ).is_empty()
    return annotations


def densify_candidates(
    sparse: pl.DataFrame,
    *,
    rows: int,
    candidates: tuple[int, ...] = CANDIDATE_FEATURES,
) -> pl.DataFrame:
    required = {
        "panel_row",
        "feature_id",
        "ref_activation",
        "alt_activation",
        "delta",
    }
    assert required <= set(sparse.columns)
    sparse = sparse.filter(pl.col("feature_id").is_in(candidates))
    assert (
        sparse.select(pl.struct("panel_row", "feature_id").n_unique()).item()
        == sparse.height
    )
    if sparse.height:
        assert int(sparse["panel_row"].max()) < rows
    grid = pl.DataFrame({"panel_row": np.arange(rows, dtype=np.uint32)}).join(
        pl.DataFrame({"feature_id": np.asarray(candidates, dtype=np.uint32)}),
        how="cross",
    )
    dense = (
        grid.join(sparse, on=["panel_row", "feature_id"], how="left")
        .with_columns(
            pl.col("ref_activation", "alt_activation", "delta").fill_null(0.0)
        )
        .with_columns(pl.col("delta").abs().alias("abs_delta"))
        .sort(["panel_row", "feature_id"])
    )
    assert dense.height == rows * len(candidates)
    assert np.allclose(
        dense["delta"].to_numpy(),
        dense["alt_activation"].to_numpy() - dense["ref_activation"].to_numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    return dense


def read_candidate_responses(
    *,
    extraction_root: Path,
    extraction_manifest: dict[str, Any],
    annotations: pl.DataFrame,
) -> pl.DataFrame:
    verify_manifest_artifacts(extraction_root, extraction_manifest)
    frames: list[pl.DataFrame] = []
    metadata_columns = [
        "panel_row",
        "chrom",
        "pos",
        "ref",
        "alt",
        "substitution",
        "label",
        "subset",
        "match_group",
        "trait",
        "consequence",
        "consequence_cre",
        "consequence_final",
        "consequence_group",
        "distance_exon",
        "distance_tss",
        "distance_tss_pc_bin",
        "distance_exon_pc_bin",
        "llr_fwd",
        "llr_rc",
        "minus_llr_avg",
        "probe_score",
        "flank_gc_count",
        "flank_gc_bin",
        "ref_local31_gc",
        "alt_local31_gc",
        "ref_local31_cpg",
        "alt_local31_cpg",
    ]
    metadata_columns = [
        column for column in metadata_columns if column in annotations.columns
    ]
    metadata = annotations.select(metadata_columns)
    for budget in BUDGETS:
        arm = f"block19-{budget // 1_000_000}m"
        for orientation in ORIENTATIONS:
            path = extraction_root / arm / f"sae_focal_{orientation}.parquet"
            sparse = (
                pl.scan_parquet(path)
                .filter(pl.col("feature_id").is_in(CANDIDATE_FEATURES))
                .collect()
            )
            dense = densify_candidates(sparse, rows=annotations.height)
            frames.append(
                dense.join(metadata, on="panel_row", how="inner").with_columns(
                    pl.lit(arm).alias("arm"),
                    pl.lit(19, dtype=pl.UInt8).alias("block"),
                    pl.lit(budget, dtype=pl.UInt32).alias("budget"),
                    pl.lit(orientation).alias("orientation"),
                )
            )
    result = pl.concat(frames, how="vertical")
    assert result.height == (
        EXPECTED_ROWS * len(CANDIDATE_FEATURES) * len(BUDGETS) * len(ORIENTATIONS)
    )
    assert (
        result.select(
            pl.struct("arm", "orientation", "panel_row", "feature_id").n_unique()
        ).item()
        == result.height
    )
    return result.sort(["arm", "orientation", "panel_row", "feature_id"])


def read_candidate_associations(
    associations_root: Path,
    associations_manifest: dict[str, Any],
) -> pl.DataFrame:
    verify_manifest_artifacts(associations_root, associations_manifest)
    frames: list[pl.DataFrame] = []
    for budget in BUDGETS:
        arm = f"block19-{budget // 1_000_000}m"
        for orientation in ORIENTATIONS:
            for response in RESPONSES:
                path = (
                    associations_root
                    / "families"
                    / arm
                    / orientation
                    / f"{response}.parquet"
                )
                frames.append(
                    pl.read_parquet(path).filter(
                        pl.col("feature_id").is_in(CANDIDATE_FEATURES)
                    )
                )
    result = pl.concat(frames, how="diagonal")
    assert set(result["response"].unique().to_list()) == set(RESPONSES)
    assert set(result["feature_id"].unique().to_list()) == set(CANDIDATE_FEATURES)
    return result.sort(["arm", "orientation", "response", "target", "feature_id"])


def add_target_views(frame: pl.DataFrame) -> pl.DataFrame:
    return pl.concat(
        [
            frame.with_columns(
                pl.lit("overall").alias("target_kind"),
                pl.lit("overall").alias("target"),
            ),
            frame.with_columns(
                pl.lit("within_subset").alias("target_kind"),
                pl.col("subset").alias("target"),
            ),
        ],
        how="vertical",
    )


def score_correlations(responses: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    targeted = add_target_views(responses)
    keys = ["arm", "block", "budget", "orientation", "feature_id"]
    for key, group in targeted.group_by(keys, maintain_order=True):
        metadata = dict(zip(keys, key, strict=True))
        for target_kind, target in (
            ("overall", "overall"),
            *(
                ("within_subset", subset)
                for subset in sorted(group["subset"].unique().to_list())
            ),
        ):
            selected = group.filter(
                (pl.col("target_kind") == target_kind) & (pl.col("target") == target)
            )
            for response in PRIMARY_RESPONSES:
                for outcome in SCORE_OUTCOMES:
                    x = selected[response].to_numpy()
                    y = selected[outcome].to_numpy()
                    finite = np.isfinite(x) & np.isfinite(y)
                    x = x[finite]
                    y = y[finite]
                    if x.size < 10 or np.unique(x).size < 2 or np.unique(y).size < 2:
                        continue
                    pearson = pearsonr(x, y)
                    spearman = spearmanr(x, y)
                    rows.append(
                        {
                            **metadata,
                            "target_kind": target_kind,
                            "target": target,
                            "response": response,
                            "outcome": outcome,
                            "n": int(x.size),
                            "pearson_r": float(pearson.statistic),
                            "pearson_p": float(pearson.pvalue),
                            "spearman_rho": float(spearman.statistic),
                            "spearman_p": float(spearman.pvalue),
                        }
                    )
    result = pl.DataFrame(rows)
    family_keys = ["arm", "orientation", "response", "outcome"]
    corrected: list[pl.DataFrame] = []
    for _, family in result.group_by(family_keys, maintain_order=True):
        corrected.append(
            family.with_columns(
                pl.Series(
                    "pearson_q",
                    bh_adjust(family["pearson_p"].to_numpy()),
                ),
                pl.Series(
                    "spearman_q",
                    bh_adjust(family["spearman_p"].to_numpy()),
                ),
            )
        )
    return pl.concat(corrected).sort([*family_keys, "target", "feature_id"])


def substitution_summary(responses: pl.DataFrame) -> pl.DataFrame:
    targeted = add_target_views(responses)
    keys = [
        "arm",
        "block",
        "budget",
        "orientation",
        "feature_id",
        "target_kind",
        "target",
        "substitution",
    ]
    return (
        targeted.group_by(keys)
        .agg(
            pl.len().alias("n"),
            pl.col("label").sum().alias("positives"),
            (pl.col("abs_delta") > 0).sum().alias("nonzero_abs_delta"),
            pl.col("delta")
            .filter(pl.col("label") == 1)
            .mean()
            .alias("positive_mean_delta"),
            pl.col("delta")
            .filter(pl.col("label") == 0)
            .mean()
            .alias("negative_mean_delta"),
            pl.col("abs_delta")
            .filter(pl.col("label") == 1)
            .mean()
            .alias("positive_mean_abs_delta"),
            pl.col("abs_delta")
            .filter(pl.col("label") == 0)
            .mean()
            .alias("negative_mean_abs_delta"),
        )
        .with_columns(
            (pl.col("positive_mean_delta") - pl.col("negative_mean_delta")).alias(
                "label_delta_difference"
            ),
            (
                pl.col("positive_mean_abs_delta") - pl.col("negative_mean_abs_delta")
            ).alias("label_abs_delta_difference"),
        )
        .sort(keys)
    )


def _oriented_window(sequence: str, orientation: str) -> str:
    start = FOCAL_INDEX - CONTEXT_FLANK
    stop = FOCAL_INDEX + CONTEXT_FLANK + 1
    window = sequence[start:stop]
    assert len(window) == 2 * CONTEXT_FLANK + 1
    if orientation == "reverse_complement":
        window = reverse_complement(window)
    else:
        assert orientation == "forward"
    return window


def top_contexts(
    responses: pl.DataFrame,
    annotations: pl.DataFrame,
    *,
    top_n: int = TOP_CONTEXTS,
) -> pl.DataFrame:
    context_columns = [
        "panel_row",
        "ref_context",
        "alt_context",
    ]
    frames: list[pl.DataFrame] = []
    keys = ["arm", "block", "budget", "orientation", "feature_id"]
    for _, group in responses.group_by(keys, maintain_order=True):
        for criterion, column, descending in (
            ("largest_abs_delta", "abs_delta", True),
            ("largest_positive_delta", "delta", True),
            ("largest_negative_delta", "delta", False),
        ):
            frames.append(
                group.sort(
                    [column, "panel_row"],
                    descending=[descending, False],
                )
                .head(top_n)
                .with_row_index("rank", offset=1)
                .with_columns(pl.lit(criterion).alias("criterion"))
            )
    selected = pl.concat(frames, how="vertical").join(
        annotations.select(context_columns),
        on="panel_row",
        how="inner",
    )
    ref_windows = [
        _oriented_window(sequence, orientation)
        for sequence, orientation in zip(
            selected["ref_context"].to_list(),
            selected["orientation"].to_list(),
            strict=True,
        )
    ]
    alt_windows = [
        _oriented_window(sequence, orientation)
        for sequence, orientation in zip(
            selected["alt_context"].to_list(),
            selected["orientation"].to_list(),
            strict=True,
        )
    ]
    selected = selected.with_columns(
        pl.Series("oriented_ref_window", ref_windows),
        pl.Series("oriented_alt_window", alt_windows),
    ).drop("ref_context", "alt_context")
    assert selected.height == (
        len(CANDIDATE_FEATURES) * len(BUDGETS) * len(ORIENTATIONS) * 3 * top_n
    )
    return selected.sort([*keys, "criterion", "rank"])


def plot_label_effects(
    associations: pl.DataFrame,
    output_dir: Path,
) -> list[Path]:
    selected = associations.filter(pl.col("response").is_in(PRIMARY_RESPONSES))
    selected = selected.filter(pl.col("target").is_in(TARGET_ORDER))
    value_limit = float(selected["rank_biserial"].abs().max())
    value_limit = max(value_limit, 0.1)
    row_specs = [
        (budget, response) for budget in BUDGETS for response in ("abs_delta", "delta")
    ]
    fig, axes = plt.subplots(
        len(row_specs),
        len(ORIENTATIONS),
        figsize=(14.5, 13.5),
        sharex=True,
        sharey=True,
    )
    image = None
    for row_index, (budget, response) in enumerate(row_specs):
        for column_index, orientation in enumerate(ORIENTATIONS):
            axis = axes[row_index, column_index]
            group = selected.filter(
                (pl.col("budget") == budget)
                & (pl.col("orientation") == orientation)
                & (pl.col("response") == response)
            )
            lookup = {
                (row["target"], row["feature_id"]): row["rank_biserial"]
                for row in group.select(
                    "target", "feature_id", "rank_biserial"
                ).to_dicts()
            }
            matrix = np.asarray(
                [
                    [lookup[(target, feature)] for feature in CANDIDATE_FEATURES]
                    for target in TARGET_ORDER
                ]
            )
            assert matrix.shape == (len(TARGET_ORDER), len(CANDIDATE_FEATURES))
            image = axis.imshow(
                matrix,
                cmap="coolwarm",
                vmin=-value_limit,
                vmax=value_limit,
                aspect="auto",
            )
            axis.set_title(
                f"{budget // 1_000_000}M · {orientation.replace('_', ' ')} · {response}"
            )
            axis.set_xticks(
                np.arange(len(CANDIDATE_FEATURES)),
                [str(feature) for feature in CANDIDATE_FEATURES],
                rotation=45,
                ha="right",
            )
            axis.set_yticks(
                np.arange(len(TARGET_ORDER)),
                [TARGET_LABELS[target] for target in TARGET_ORDER],
            )
    assert image is not None
    fig.colorbar(
        image,
        ax=axes,
        label="Rank-biserial effect (pathogenic − benign)",
        shrink=0.62,
        pad=0.02,
    )
    fig.suptitle(
        "Recurrent final-layer SAE candidates have domain- and strand-dependent label effects",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "Candidates were selected on these outcomes; the heatmap is exploratory interpretation.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.subplots_adjust(
        left=0.16,
        right=0.87,
        bottom=0.08,
        top=0.94,
        hspace=0.30,
        wspace=0.10,
    )
    paths = [
        output_dir / "candidate_label_effects.svg",
        output_dir / "candidate_label_effects.png",
    ]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def summarize(
    *,
    extraction_root: Path,
    associations_root: Path,
    panel_path: Path,
    panel_manifest_path: Path,
    contexts_path: Path,
    recurrence_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert extraction_root.is_dir()
    assert associations_root.is_dir()
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    started = time.monotonic()

    extraction_manifest = json.loads((extraction_root / "manifest.json").read_text())
    associations_manifest = json.loads(
        (associations_root / "manifest.json").read_text()
    )
    recurrence = pl.read_parquet(recurrence_path)
    observed_candidates = tuple(
        recurrence.head(len(CANDIDATE_FEATURES))["feature_id"].to_list()
    )
    assert observed_candidates == CANDIDATE_FEATURES

    annotations = read_annotations(
        panel_path=panel_path,
        panel_manifest_path=panel_manifest_path,
        contexts_path=contexts_path,
    )
    responses = read_candidate_responses(
        extraction_root=extraction_root,
        extraction_manifest=extraction_manifest,
        annotations=annotations,
    )
    associations = read_candidate_associations(associations_root, associations_manifest)
    correlations = score_correlations(responses)
    substitutions = substitution_summary(responses)
    contexts = top_contexts(responses, annotations)

    output_dir.mkdir(parents=True)
    tables = {
        "candidate_responses.parquet": responses,
        "candidate_associations.parquet": associations,
        "candidate_score_correlations.parquet": correlations,
        "candidate_substitution_summary.parquet": substitutions,
        "candidate_top_contexts.parquet": contexts,
    }
    for filename, frame in tables.items():
        frame.write_parquet(output_dir / filename, compression="zstd")
    plot_paths = plot_label_effects(associations, output_dir)

    artifact_paths = [
        *(output_dir / filename for filename in tables),
        *plot_paths,
    ]
    artifacts = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in artifact_paths
    }
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": os.environ.get("RUN_ID", ""),
        "experiment_commit": experiment_commit,
        "elapsed_seconds": time.monotonic() - started,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "scipy": scipy.__version__,
        "inputs": {
            "panel_sha256": sha256_file(panel_path),
            "contexts_sha256": sha256_file(contexts_path),
            "extraction_manifest_sha256": sha256_file(
                extraction_root / "manifest.json"
            ),
            "association_manifest_sha256": sha256_file(
                associations_root / "manifest.json"
            ),
            "recurrence_sha256": sha256_file(recurrence_path),
            "verified_extraction_artifacts": len(extraction_manifest["artifacts"]),
            "verified_association_artifacts": len(associations_manifest["artifacts"]),
        },
        "protocol": {
            "candidate_features": list(CANDIDATE_FEATURES),
            "candidate_selection": (
                "fixed top six overall-label recurrent block-19 slots; "
                "outcome-aware exploratory interpretation"
            ),
            "budgets": list(BUDGETS),
            "orientations": list(ORIENTATIONS),
            "primary_responses": sorted(PRIMARY_RESPONSES),
            "top_contexts_per_criterion": TOP_CONTEXTS,
            "context_window_bp": 2 * CONTEXT_FLANK + 1,
            "score_correlation_fdr": (
                "BH within arm x orientation x response x score outcome; "
                "exploratory because candidates are label-selected"
            ),
        },
        "rows": {
            "candidate_responses": responses.height,
            "candidate_associations": associations.height,
            "candidate_score_correlations": correlations.height,
            "candidate_substitution_summary": substitutions.height,
            "candidate_top_contexts": contexts.height,
        },
        "artifacts": artifacts,
    }
    write_json(output_dir / "results.json", result)
    result["artifacts"]["results.json"] = {
        "bytes": (output_dir / "results.json").stat().st_size,
        "sha256": sha256_file(output_dir / "results.json"),
    }
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--associations-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--recurrence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        extraction_root=args.extraction_root,
        associations_root=args.associations_root,
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        contexts_path=args.contexts,
        recurrence_path=args.recurrence,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
