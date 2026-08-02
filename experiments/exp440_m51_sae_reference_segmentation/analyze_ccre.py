"""Test archived focal SAE activations for ELS and PLS associations."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from analyze import (
    MINIMUM_SUPPORT,
    PRIMARY_ABS_RANK_BISERIAL,
    PRIMARY_Q,
    analyze_sparse_table,
    validate_extraction,
)
from build_panel import FOCAL_INDEX, STANDARD_CHROMS
from extract_focal import (
    BLOCK_INDICES,
    D_SAE,
    EXTRACTION_RUN_ID,
    ISSUE,
    ORIENTATIONS,
    PANEL_RUN_ID,
    arm_label,
    assert_commit,
    load_panel,
    sha256_file,
    write_json,
)

CCRE_RUN_ID = "dna-exp440-ccre-subtype-associations-seed288-r1"
CCRE_BYTES = 10_361_229
CCRE_SHA256 = "35c322243a347ddcbcfac478c825ca9bb1af1cdfbbd876b64e788b5105a7afd5"
EXPECTED_SUBTYPE_COUNTS = {
    "pELS": 1_193,
    "dELS": 2_922,
    "PLS": 1_309,
}
CONTRASTS = ("els", "pls")


def annotate_focal_ccre(panel: pl.DataFrame, ccre: pl.DataFrame) -> pl.DataFrame:
    """Overlay non-overlapping cCRE intervals at each panel focal coordinate."""
    required_panel = {"panel_row", "chrom", "start", "end"}
    required_ccre = {"chrom", "start", "end", "cre_class"}
    assert required_panel <= set(panel.columns)
    assert set(ccre.columns) == required_ccre
    assert panel["panel_row"].n_unique() == panel.height
    assert panel.filter(pl.col("end") <= pl.col("start")).is_empty()
    assert ccre.filter(pl.col("end") <= pl.col("start")).is_empty()
    assert ccre.filter(~pl.col("chrom").is_in(STANDARD_CHROMS)).is_empty()
    assert ccre.select(pl.all().null_count()).sum_horizontal().item() == 0

    intervals = ccre.sort("chrom", "start", "end").with_columns(
        pl.col("end").shift(1).over("chrom").alias("previous_end")
    )
    assert intervals.filter(pl.col("start") < pl.col("previous_end")).is_empty()
    intervals = intervals.drop("previous_end").rename(
        {
            "start": "ccre_start",
            "end": "ccre_end",
            "cre_class": "ccre_subtype",
        }
    )

    focal = panel.select(
        "panel_row",
        "chrom",
        (pl.col("start") + FOCAL_INDEX).alias("focal_position0"),
    ).sort("chrom", "focal_position0")
    overlay = focal.join_asof(
        intervals,
        left_on="focal_position0",
        right_on="ccre_start",
        by="chrom",
        strategy="backward",
        check_sortedness=False,
    ).with_columns(
        pl.when(
            pl.col("ccre_start").is_not_null()
            & (pl.col("focal_position0") < pl.col("ccre_end"))
        )
        .then(pl.col("ccre_subtype"))
        .otherwise(None)
        .alias("ccre_subtype")
    )
    overlay = (
        overlay.with_columns(
            pl.when(pl.col("ccre_subtype").is_in(["pELS", "dELS"]))
            .then(pl.lit("els"))
            .when(pl.col("ccre_subtype") == "PLS")
            .then(pl.lit("pls"))
            .otherwise(pl.lit("other_or_none"))
            .alias("ccre_group")
        )
        .select(
            "panel_row",
            "chrom",
            "focal_position0",
            "ccre_start",
            "ccre_end",
            "ccre_subtype",
            "ccre_group",
        )
        .sort("panel_row")
    )
    assert overlay.height == panel.height
    assert overlay["panel_row"].to_list() == panel.sort("panel_row")[
        "panel_row"
    ].to_list()
    assert overlay.filter(
        pl.col("ccre_subtype").is_null()
        & (
            pl.col("ccre_start").is_not_null()
            & (pl.col("focal_position0") < pl.col("ccre_end"))
        )
    ).is_empty()
    return overlay


def contrast_codes(overlay: pl.DataFrame, contrast: str) -> np.ndarray:
    assert contrast in CONTRASTS
    assert overlay["panel_row"].to_list() == list(range(overlay.height))
    codes = (
        overlay.select((pl.col("ccre_group") == contrast).cast(pl.UInt8))
        .to_series()
        .to_numpy()
        .astype(np.int64, copy=False)
    )
    assert codes.shape == (overlay.height,)
    assert set(np.unique(codes)) == {0, 1}
    return codes


def orientation_concordant_hits(associations: pl.DataFrame) -> pl.DataFrame:
    keys = ["block", "arm", "ccre_contrast", "feature_id"]
    columns = [
        "rank_biserial",
        "mean_difference",
        "welch_q",
        "mwu_q",
        "auprc",
    ]
    primary = associations.filter(pl.col("primary_association"))
    forward = primary.filter(pl.col("orientation") == "forward").select(
        *keys, *(pl.col(column).alias(f"forward_{column}") for column in columns)
    )
    reverse = primary.filter(
        pl.col("orientation") == "reverse_complement"
    ).select(
        *keys, *(pl.col(column).alias(f"reverse_{column}") for column in columns)
    )
    result = (
        forward.join(reverse, on=keys, how="inner", validate="1:1")
        .filter(
            pl.col("forward_rank_biserial")
            * pl.col("reverse_rank_biserial")
            > 0
        )
        .with_columns(
            pl.min_horizontal(
                pl.col("forward_rank_biserial").abs(),
                pl.col("reverse_rank_biserial").abs(),
            ).alias("minimum_abs_rank_biserial"),
            (
                pl.col("forward_rank_biserial")
                + pl.col("reverse_rank_biserial")
            )
            .sign()
            .cast(pl.Int8)
            .alias("effect_direction"),
        )
        .sort(
            "block",
            "ccre_contrast",
            "minimum_abs_rank_biserial",
            descending=[False, False, True],
        )
    )
    assert result.select(pl.struct(keys).n_unique()).item() == result.height
    return result


def analyze_ccre(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    ccre_path: Path,
    extraction_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    assert os.environ.get("RUN_ID") == CCRE_RUN_ID
    assert ccre_path.is_file()
    assert ccre_path.stat().st_size == CCRE_BYTES
    assert sha256_file(ccre_path) == CCRE_SHA256
    started = time.monotonic()

    panel = load_panel(panel_path, panel_manifest_path)
    extraction = validate_extraction(extraction_root)
    ccre = pl.read_parquet(ccre_path)
    overlay = annotate_focal_ccre(panel, ccre)
    subtype_counts = dict(
        overlay.filter(pl.col("ccre_subtype").is_not_null())
        .group_by("ccre_subtype")
        .len()
        .sort("ccre_subtype")
        .iter_rows()
    )
    assert {
        subtype: subtype_counts[subtype] for subtype in EXPECTED_SUBTYPE_COUNTS
    } == EXPECTED_SUBTYPE_COUNTS
    contrast_counts = {
        contrast: int((overlay["ccre_group"] == contrast).sum())
        for contrast in CONTRASTS
    }
    assert contrast_counts == {"els": 4_115, "pls": 1_309}

    output_dir.mkdir(parents=True)
    frames: list[pl.DataFrame] = []
    family_records: list[dict[str, Any]] = []
    for block_index in BLOCK_INDICES:
        arm = arm_label(block_index)
        for orientation in ORIENTATIONS:
            sparse = pl.read_parquet(
                extraction_root / arm / f"sae_focal_{orientation}.parquet"
            )
            for contrast in CONTRASTS:
                contrast_started = time.monotonic()
                associations = (
                    analyze_sparse_table(
                        sparse,
                        panel_classes=contrast_codes(overlay, contrast),
                        class_names=(f"non_{contrast}", contrast),
                        d_sae=D_SAE,
                        minimum_support=MINIMUM_SUPPORT,
                    )
                    .filter(pl.col("reference_class") == contrast)
                    .drop("reference_class")
                    .with_columns(
                        pl.lit(block_index + 1).cast(pl.UInt8).alias("block"),
                        pl.lit(arm).alias("arm"),
                        pl.lit(orientation).alias("orientation"),
                        pl.lit(contrast).alias("ccre_contrast"),
                    )
                )
                frames.append(associations)
                family_records.append(
                    {
                        "block": block_index + 1,
                        "arm": arm,
                        "orientation": orientation,
                        "ccre_contrast": contrast,
                        "n_positive": contrast_counts[contrast],
                        "n_negative": panel.height - contrast_counts[contrast],
                        "eligible_features": associations.height,
                        "primary_associations": int(
                            associations["primary_association"].sum()
                        ),
                        "positive_associations": associations.filter(
                            pl.col("primary_association")
                            & (pl.col("rank_biserial") > 0)
                        ).height,
                        "negative_associations": associations.filter(
                            pl.col("primary_association")
                            & (pl.col("rank_biserial") < 0)
                        ).height,
                        "maximum_abs_rank_biserial": float(
                            associations["rank_biserial"].abs().max()
                        ),
                        "maximum_auprc": float(associations["auprc"].max()),
                    }
                )
                print(
                    json.dumps(
                        {
                            "stage": "associate_ccre_subtype",
                            "arm": arm,
                            "orientation": orientation,
                            "ccre_contrast": contrast,
                            "eligible_features": associations.height,
                            "primary_associations": int(
                                associations["primary_association"].sum()
                            ),
                            "elapsed_seconds": time.monotonic()
                            - contrast_started,
                        }
                    ),
                    flush=True,
                )

    all_associations = pl.concat(frames).select(
        "block",
        "arm",
        "orientation",
        "ccre_contrast",
        "feature_id",
        "n_class",
        "n_rest",
        "nonzero_support_total",
        "nonzero_support_class",
        "mean_class",
        "mean_rest",
        "mean_difference",
        "cohen_d",
        "welch_t",
        "welch_df",
        "welch_p",
        "welch_q",
        "mwu_u",
        "mwu_p",
        "mwu_q",
        "rank_biserial",
        "auprc",
        "primary_association",
    )
    primary_hits = all_associations.filter(pl.col("primary_association")).sort(
        "block",
        "orientation",
        "ccre_contrast",
        pl.col("rank_biserial").abs(),
        descending=[False, False, False, True],
    )
    concordant = orientation_concordant_hits(all_associations)
    family_summary = pl.DataFrame(family_records).sort(
        "block", "orientation", "ccre_contrast"
    )

    frames_by_path = {
        "panel_ccre_overlay.parquet": overlay,
        "associations.parquet": all_associations,
        "family_summary.parquet": family_summary,
        "primary_hits.parquet": primary_hits,
        "orientation_concordant_hits.parquet": concordant,
    }
    artifacts: dict[str, Any] = {}
    for relative, frame in frames_by_path.items():
        path = output_dir / relative
        frame.write_parquet(path, compression="zstd")
        artifacts[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": frame.height,
        }

    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": CCRE_RUN_ID,
        "analysis_status": "preregistered_focal_ccre_els_pls_association_scan",
        "experiment_commit": experiment_commit,
        "elapsed_seconds": time.monotonic() - started,
        "platform": platform.platform(),
        "inputs": {
            "panel_run_id": PANEL_RUN_ID,
            "extraction_run_id": EXTRACTION_RUN_ID,
            "extraction_experiment_commit": extraction["experiment_commit"],
            "extraction_manifest_sha256": sha256_file(
                extraction_root / "manifest.json"
            ),
            "ccre": {
                "uri": "s3://oa-bolinas/snakemake/zoonomia_projection_dataset/results/human/intervals/cre/all.parquet",
                "bytes": CCRE_BYTES,
                "sha256": CCRE_SHA256,
            },
        },
        "protocol": {
            "focal_coordinate": "window start + 127 in 0-based half-open coordinates",
            "contrasts": {
                "els": "pELS or dELS versus every other focal coordinate",
                "pls": "PLS versus every other focal coordinate",
            },
            "views": list(ORIENTATIONS),
            "minimum_nonzero_support": MINIMUM_SUPPORT,
            "bh_family": "within block x orientation x cCRE contrast, separately for Welch and Mann-Whitney",
            "primary_call": (
                f"Welch q<{PRIMARY_Q} and Mann-Whitney q<{PRIMARY_Q} and "
                f"|rank-biserial|>={PRIMARY_ABS_RANK_BISERIAL}"
            ),
            "orientation_concordant_call": (
                "primary in FWD and RC with matching rank-biserial sign"
            ),
        },
        "subtype_counts": subtype_counts,
        "contrast_counts": contrast_counts,
        "association_rows": all_associations.height,
        "primary_associations": primary_hits.height,
        "orientation_concordant_hits": concordant.height,
        "family_summary": family_records,
        "artifacts": artifacts,
    }
    results_path = output_dir / "results.json"
    write_json(results_path, result)
    result["artifacts"]["results.json"] = {
        "bytes": results_path.stat().st_size,
        "sha256": sha256_file(results_path),
    }
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--ccre", type=Path, required=True)
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_ccre(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        ccre_path=args.ccre,
        extraction_root=args.extraction_root,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "association_rows": result["association_rows"],
                "primary_associations": result["primary_associations"],
                "orientation_concordant_hits": result[
                    "orientation_concordant_hits"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
