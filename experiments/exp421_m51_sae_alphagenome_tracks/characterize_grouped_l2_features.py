"""Post-hoc characterization of AlphaGenome-associated SAE features.

The complete-family scan selected these features without using sequence annotations.
This script asks what they respond to, while keeping orientations separate and
explicitly measuring whether associations survive removal of extreme variants.
The results are descriptive follow-up, not a new confirmatory screen.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from grouped_l2_association import (
    EXPECTED_ROWS,
    KEYS,
    benjamini_hochberg,
    load_grouped_outcomes,
    sha256_file,
    write_json,
)
from pyfaidx import Fasta
from scipy import stats

ISSUE = 421
CONTEXT_FLANK = 31
TOP_CONTEXTS = 100
MIN_CATEGORY_ROWS = 32
ORIENTATIONS = ("forward", "reverse_complement")
FEATURES_BY_ARM = {
    "block10-25m": (11_137,),
    "block19-25m": (219, 11_928),
}
FEATURE_ROLES = {
    ("block10-25m", 11_137): "broad_accessibility",
    ("block19-25m", 219): "broad_accessibility",
    ("block19-25m", 11_928): "tail_sensitive_candidate",
}
TARGETS_BY_FEATURE = {
    ("block10-25m", 11_137): ("ATAC", "DNASE", "PROCAP"),
    ("block19-25m", 219): ("ATAC", "DNASE", "PROCAP"),
    ("block19-25m", 11_928): ("all_tracks", "RNA_SEQ", "tissue|liver"),
}
DNA_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
SEQUENCE_BASES = ("A", "C", "G", "T", "N")


def _artifact_for_basename(manifest: dict[str, Any], basename: str) -> dict[str, Any]:
    matches = [
        value
        for relative, value in manifest["artifacts"].items()
        if Path(relative).name == basename
    ]
    assert len(matches) == 1, (basename, len(matches))
    return matches[0]


def verify_artifact(path: Path, expected: dict[str, Any]) -> None:
    assert path.is_file(), path
    assert path.stat().st_size == expected["bytes"], path
    assert sha256_file(path) == expected["sha256"], path


def reverse_complement(sequence: str) -> str:
    assert set(sequence) <= set(SEQUENCE_BASES)
    return sequence.translate(DNA_COMPLEMENT)[::-1]


def _entropy(sequence: str) -> float:
    counts = np.asarray([sequence.count(base) for base in "ACGT"], dtype=np.float64)
    counts = counts[counts > 0]
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _max_homopolymer(sequence: str) -> int:
    longest = 1
    current = 1
    for previous, base in itertools.pairwise(sequence):
        current = current + 1 if base == previous else 1
        longest = max(longest, current)
    return longest


def build_contexts(panel: pl.DataFrame, fasta_path: Path) -> pl.DataFrame:
    """Read 63-bp contexts using one 1-based-to-0-based boundary conversion."""

    assert panel.height == EXPECTED_ROWS
    fasta = Fasta(
        str(fasta_path),
        as_raw=True,
        sequence_always_upper=True,
        rebuild=False,
    )
    rows: list[dict[str, Any]] = []
    for panel_row, row in enumerate(panel.iter_rows(named=True)):
        pos0 = int(row["pos"]) - 1
        start0 = pos0 - CONTEXT_FLANK
        end0 = pos0 + CONTEXT_FLANK + 1
        assert start0 >= 0
        ref_context = str(fasta[str(row["chrom"])][start0:end0]).upper()
        assert len(ref_context) == 2 * CONTEXT_FLANK + 1
        assert ref_context[CONTEXT_FLANK] == row["ref"], (
            row["chrom"],
            row["pos"],
            row["ref"],
            ref_context[CONTEXT_FLANK],
        )
        assert set(ref_context) <= set(SEQUENCE_BASES)
        alt_context = (
            ref_context[:CONTEXT_FLANK]
            + str(row["alt"])
            + ref_context[CONTEXT_FLANK + 1 :]
        )
        gc_fraction = sum(base in "GC" for base in ref_context) / len(ref_context)
        rows.append(
            {
                "panel_row": panel_row,
                "ref_context": ref_context,
                "alt_context": alt_context,
                "substitution": f"{row['ref']}>{row['alt']}",
                "ref_gc_fraction": gc_fraction,
                "alt_gc_fraction": sum(base in "GC" for base in alt_context)
                / len(alt_context),
                "ref_cpg_count": ref_context.count("CG"),
                "alt_cpg_count": alt_context.count("CG"),
                "ref_entropy": _entropy(ref_context),
                "ref_max_homopolymer": _max_homopolymer(ref_context),
            }
        )
    fasta.close()
    contexts = pl.DataFrame(rows)
    assert contexts.height == EXPECTED_ROWS
    gc = contexts["ref_gc_fraction"].to_numpy()
    cutpoints = np.quantile(gc, (0.2, 0.4, 0.6, 0.8))
    quintile = np.digitize(gc, cutpoints, right=True) + 1
    contexts = contexts.with_columns(
        pl.Series("local_gc_quintile", quintile.astype(np.uint8)),
        (pl.col("alt_gc_fraction") - pl.col("ref_gc_fraction")).alias(
            "delta_gc_fraction"
        ),
        (pl.col("alt_cpg_count") - pl.col("ref_cpg_count")).alias("delta_cpg_count"),
    )
    assert contexts["local_gc_quintile"].is_between(1, 5).all()
    return contexts


def validate_repeat_panel(
    panel: pl.DataFrame,
    repeat_panel_path: Path,
    repeat_manifest_path: Path,
) -> pl.DataFrame:
    manifest = json.loads(repeat_manifest_path.read_text())
    verify_artifact(
        repeat_panel_path,
        _artifact_for_basename(manifest, repeat_panel_path.name),
    )
    repeat = pl.read_parquet(repeat_panel_path).sort("panel_row")
    assert repeat.height == panel.height
    assert repeat["panel_row"].to_list() == list(range(EXPECTED_ROWS))
    indexed = panel.with_row_index("panel_row")
    for key in KEYS:
        assert repeat[key].to_list() == indexed[key].to_list(), key
    required = {
        "panel_row",
        "position_status",
        "repeat_fraction",
        "repeat_name",
        "repeat_class",
        "repeat_family",
        "family_label",
        "subfamily_label",
        "boundary_distance",
        "overlap_count",
        "repeat_interior_32",
    }
    assert required <= set(repeat.columns)
    return repeat.select(sorted(required))


def densify_features(
    sparse: pl.DataFrame,
    *,
    rows: int,
    feature_ids: tuple[int, ...],
) -> pl.DataFrame:
    required = {
        "panel_row",
        "feature_id",
        "ref_activation",
        "alt_activation",
        "delta",
    }
    assert required <= set(sparse.columns)
    selected = sparse.filter(pl.col("feature_id").is_in(feature_ids))
    assert (
        selected.select(pl.struct("panel_row", "feature_id").n_unique()).item()
        == selected.height
    )
    grid = pl.DataFrame({"panel_row": np.arange(rows, dtype=np.uint32)}).join(
        pl.DataFrame({"feature_id": np.asarray(feature_ids, dtype=np.uint32)}),
        how="cross",
    )
    dense = (
        grid.join(selected, on=["panel_row", "feature_id"], how="left")
        .with_columns(
            pl.col("ref_activation", "alt_activation", "delta").fill_null(0.0)
        )
        .with_columns(pl.col("delta").abs().alias("abs_delta"))
        .sort(["feature_id", "panel_row"])
    )
    assert dense.height == rows * len(feature_ids)
    assert np.allclose(
        dense["delta"].to_numpy(),
        dense["alt_activation"].to_numpy() - dense["ref_activation"].to_numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    return dense


def load_responses(
    activation_root: Path,
    extraction_manifest_path: Path,
) -> pl.DataFrame:
    manifest = json.loads(extraction_manifest_path.read_text())
    frames: list[pl.DataFrame] = []
    for arm, feature_ids in FEATURES_BY_ARM.items():
        for orientation in ORIENTATIONS:
            relative = f"{arm}/sae_focal_{orientation}.parquet"
            path = activation_root / relative
            verify_artifact(path, manifest["artifacts"][relative])
            selected = (
                pl.scan_parquet(path)
                .filter(pl.col("feature_id").is_in(feature_ids))
                .collect()
            )
            frames.append(
                densify_features(
                    selected,
                    rows=EXPECTED_ROWS,
                    feature_ids=feature_ids,
                ).with_columns(
                    pl.lit(arm).alias("arm"),
                    pl.lit(int(arm.removeprefix("block").split("-")[0])).alias(
                        "report_block"
                    ),
                    pl.lit(orientation).alias("orientation"),
                )
            )
    result = pl.concat(frames, how="vertical").with_columns(
        pl.struct("arm", "feature_id")
        .map_elements(
            lambda value: FEATURE_ROLES[(value["arm"], value["feature_id"])],
            return_dtype=pl.String,
        )
        .alias("feature_role")
    )
    expected = (
        EXPECTED_ROWS
        * len(ORIENTATIONS)
        * sum(len(ids) for ids in FEATURES_BY_ARM.values())
    )
    assert result.height == expected
    assert (
        result.select(
            pl.struct("arm", "orientation", "feature_id", "panel_row").n_unique()
        ).item()
        == expected
    )
    return result.sort(["arm", "feature_id", "orientation", "panel_row"])


def feature_summary(responses: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["arm", "report_block", "feature_id", "feature_role", "orientation"]
    for key, group in responses.group_by(keys, maintain_order=True):
        values = group["abs_delta"].to_numpy()
        delta = group["delta"].to_numpy()
        nonzero = values > 0
        top_count = max(1, int(np.ceil(0.01 * len(values))))
        top_mass = np.sort(values)[-top_count:].sum()
        total_mass = values.sum()
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "rows": len(values),
                "support": int(nonzero.sum()),
                "prevalence": float(nonzero.mean()),
                "positive_delta": int((delta > 0).sum()),
                "negative_delta": int((delta < 0).sum()),
                "abs_delta_p50": float(np.quantile(values, 0.50)),
                "abs_delta_p90": float(np.quantile(values, 0.90)),
                "abs_delta_p95": float(np.quantile(values, 0.95)),
                "abs_delta_p99": float(np.quantile(values, 0.99)),
                "abs_delta_max": float(values.max()),
                "top_1pct_abs_delta_mass_fraction": (
                    float(top_mass / total_mass) if total_mass > 0 else 0.0
                ),
            }
        )
    return pl.DataFrame(rows).sort(["arm", "feature_id", "orientation"])


def _safe_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    statistic: str,
) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    result = stats.pearsonr(x, y) if statistic == "pearson" else stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def orientation_concordance(responses: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arm, feature_id), group in responses.group_by(
        ["arm", "feature_id"], maintain_order=True
    ):
        forward = group.filter(pl.col("orientation") == "forward").sort("panel_row")
        reverse = group.filter(pl.col("orientation") == "reverse_complement").sort(
            "panel_row"
        )
        assert forward["panel_row"].to_list() == reverse["panel_row"].to_list()
        f_delta = forward["delta"].to_numpy()
        r_delta = reverse["delta"].to_numpy()
        f_abs = np.abs(f_delta)
        r_abs = np.abs(r_delta)
        f_active = f_abs > 0
        r_active = r_abs > 0
        union = f_active | r_active
        both = f_active & r_active
        row: dict[str, Any] = {
            "arm": arm,
            "feature_id": feature_id,
            "feature_role": FEATURE_ROLES[(arm, feature_id)],
            "active_union": int(union.sum()),
            "active_intersection": int(both.sum()),
            "active_jaccard": float(both.sum() / union.sum()),
            "delta_sign_agreement_when_both_active": float(
                (np.sign(f_delta[both]) == np.sign(r_delta[both])).mean()
            ),
        }
        for statistic in ("pearson", "spearman"):
            row[f"{statistic}_delta"] = _safe_correlation(
                f_delta, r_delta, statistic=statistic
            )[0]
            row[f"{statistic}_abs_delta"] = _safe_correlation(
                f_abs, r_abs, statistic=statistic
            )[0]
        rows.append(row)
    return pl.DataFrame(rows).sort(["arm", "feature_id"])


def _bh_columns(
    frame: pl.DataFrame,
    *,
    family_columns: list[str],
    pvalue_columns: Iterable[str],
) -> pl.DataFrame:
    corrected: list[pl.DataFrame] = []
    for _, family in frame.group_by(family_columns, maintain_order=True):
        expressions = []
        for column in pvalue_columns:
            values = family[column].to_numpy()
            assert np.isfinite(values).all()
            expressions.append(
                pl.Series(column.replace("_p", "_q"), benjamini_hochberg(values))
            )
        corrected.append(family.with_columns(*expressions))
    return pl.concat(corrected, how="vertical")


def _finite_welch(inside: np.ndarray, outside: np.ndarray) -> tuple[float, float]:
    result = stats.ttest_ind(
        inside,
        outside,
        equal_var=False,
        alternative="two-sided",
    )
    if np.isfinite(result.statistic) and np.isfinite(result.pvalue):
        return float(result.statistic), float(result.pvalue)
    return 0.0, 1.0


def category_associations(annotated: pl.DataFrame) -> pl.DataFrame:
    frame = annotated.with_columns(
        pl.when(pl.col("label") == 1)
        .then(pl.lit("pathogenic"))
        .otherwise(pl.lit("benign"))
        .alias("label_name"),
        pl.col("repeat_class").fill_null("no_repeat").alias("repeat_class_clean"),
        pl.col("repeat_family").fill_null("no_repeat").alias("repeat_family_clean"),
        pl.concat_str(pl.lit("Q"), pl.col("local_gc_quintile")).alias("local_gc_group"),
    )
    dimensions = {
        "label": "label_name",
        "subset": "subset",
        "consequence": "consequence_cre",
        "repeat_position": "position_status",
        "repeat_class": "repeat_class_clean",
        "repeat_family": "repeat_family_clean",
        "local_gc_quintile": "local_gc_group",
        "substitution": "substitution",
    }
    rows: list[dict[str, Any]] = []
    keys = ["arm", "report_block", "feature_id", "feature_role", "orientation"]
    for key, group in frame.group_by(keys, maintain_order=True):
        metadata = dict(zip(keys, key, strict=True))
        values = group["abs_delta"].to_numpy()
        for dimension, column in dimensions.items():
            categories = group[column].cast(pl.String).to_numpy()
            for level in sorted(set(categories)):
                selected = categories == level
                if (
                    selected.sum() < MIN_CATEGORY_ROWS
                    or (~selected).sum() < MIN_CATEGORY_ROWS
                ):
                    continue
                inside = values[selected]
                outside = values[~selected]
                mann = stats.mannwhitneyu(
                    inside,
                    outside,
                    alternative="two-sided",
                    method="asymptotic",
                )
                welch_t, welch_p = _finite_welch(inside, outside)
                rows.append(
                    {
                        **metadata,
                        "dimension": dimension,
                        "level": level,
                        "n_level": len(inside),
                        "n_other": len(outside),
                        "mean_level": float(inside.mean()),
                        "mean_other": float(outside.mean()),
                        "median_level": float(np.median(inside)),
                        "median_other": float(np.median(outside)),
                        "rank_biserial": float(
                            2 * mann.statistic / (len(inside) * len(outside)) - 1
                        ),
                        "mann_whitney_p": float(mann.pvalue),
                        "mean_difference": float(inside.mean() - outside.mean()),
                        "welch_t": welch_t,
                        "welch_p": welch_p,
                    }
                )
        for subset in sorted(group["subset"].unique().to_list()):
            selected_subset = group.filter(pl.col("subset") == subset)
            labels = selected_subset["label"].to_numpy().astype(bool)
            if labels.sum() < 10 or (~labels).sum() < 10:
                continue
            subset_values = selected_subset["abs_delta"].to_numpy()
            inside = subset_values[labels]
            outside = subset_values[~labels]
            mann = stats.mannwhitneyu(inside, outside, method="asymptotic")
            welch_t, welch_p = _finite_welch(inside, outside)
            rows.append(
                {
                    **metadata,
                    "dimension": "label_within_subset",
                    "level": subset,
                    "n_level": len(inside),
                    "n_other": len(outside),
                    "mean_level": float(inside.mean()),
                    "mean_other": float(outside.mean()),
                    "median_level": float(np.median(inside)),
                    "median_other": float(np.median(outside)),
                    "rank_biserial": float(
                        2 * mann.statistic / (len(inside) * len(outside)) - 1
                    ),
                    "mann_whitney_p": float(mann.pvalue),
                    "mean_difference": float(inside.mean() - outside.mean()),
                    "welch_t": welch_t,
                    "welch_p": welch_p,
                }
            )
    result = pl.DataFrame(rows)
    assert result.height > 0
    return _bh_columns(
        result,
        family_columns=["arm", "feature_id", "orientation", "dimension"],
        pvalue_columns=("mann_whitney_p", "welch_p"),
    ).sort(["arm", "feature_id", "orientation", "dimension", "mann_whitney_p"])


def sequence_metric_associations(annotated: pl.DataFrame) -> pl.DataFrame:
    metrics = (
        "ref_gc_fraction",
        "ref_cpg_count",
        "ref_entropy",
        "ref_max_homopolymer",
        "delta_gc_fraction",
        "delta_cpg_count",
        "repeat_fraction",
        "boundary_distance",
    )
    rows: list[dict[str, Any]] = []
    keys = ["arm", "report_block", "feature_id", "feature_role", "orientation"]
    for key, group in annotated.group_by(keys, maintain_order=True):
        metadata = dict(zip(keys, key, strict=True))
        x = group["abs_delta"].to_numpy()
        for metric in metrics:
            y = group[metric].cast(pl.Float64).fill_null(float("nan")).to_numpy()
            for statistic in ("pearson", "spearman"):
                effect, pvalue = _safe_correlation(x, y, statistic=statistic)
                if np.isfinite(pvalue):
                    rows.append(
                        {
                            **metadata,
                            "metric": metric,
                            "statistic": statistic,
                            "effect": effect,
                            "pvalue": pvalue,
                        }
                    )
    result = pl.DataFrame(rows)
    corrected: list[pl.DataFrame] = []
    for _, family in result.group_by(
        ["arm", "feature_id", "orientation", "statistic"], maintain_order=True
    ):
        corrected.append(
            family.with_columns(
                pl.Series("qvalue", benjamini_hochberg(family["pvalue"].to_numpy()))
            )
        )
    return pl.concat(corrected).sort(
        ["arm", "feature_id", "orientation", "statistic", "pvalue"]
    )


def _oriented(sequence: str, orientation: str) -> str:
    if orientation == "forward":
        return sequence
    assert orientation == "reverse_complement"
    return reverse_complement(sequence)


def top_variants_and_profiles(
    annotated: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    top_frames: list[pl.DataFrame] = []
    profile_rows: list[dict[str, Any]] = []
    keys = ["arm", "report_block", "feature_id", "feature_role", "orientation"]
    for key, group in annotated.group_by(keys, maintain_order=True):
        metadata = dict(zip(keys, key, strict=True))
        top = (
            group.sort(["abs_delta", "panel_row"], descending=[True, False])
            .head(TOP_CONTEXTS)
            .with_row_index("rank", offset=1)
        )
        higher_sequences: list[str] = []
        for row in top.iter_rows(named=True):
            sequence = (
                row["alt_context"]
                if row["alt_activation"] >= row["ref_activation"]
                else row["ref_context"]
            )
            higher_sequences.append(_oriented(sequence, row["orientation"]))
        top = top.with_columns(pl.Series("higher_activation_context", higher_sequences))
        top_frames.append(top)

        active = group.filter(pl.col("abs_delta") > 0)
        background_sequences = []
        for row in active.iter_rows(named=True):
            sequence = (
                row["alt_context"]
                if row["alt_activation"] >= row["ref_activation"]
                else row["ref_context"]
            )
            background_sequences.append(_oriented(sequence, row["orientation"]))
        assert len(background_sequences) > 0
        for position in range(-CONTEXT_FLANK, CONTEXT_FLANK + 1):
            index = position + CONTEXT_FLANK
            top_bases = [sequence[index] for sequence in higher_sequences]
            background_bases = [sequence[index] for sequence in background_sequences]
            for base in SEQUENCE_BASES:
                top_frequency = top_bases.count(base) / len(top_bases)
                background_frequency = background_bases.count(base) / len(
                    background_bases
                )
                profile_rows.append(
                    {
                        **metadata,
                        "position": position,
                        "base": base,
                        "top_n": len(top_bases),
                        "background_n": len(background_bases),
                        "top_frequency": top_frequency,
                        "background_frequency": background_frequency,
                        "frequency_difference": top_frequency - background_frequency,
                    }
                )
    top_result = pl.concat(top_frames, how="diagonal_relaxed")
    keep = [
        *keys,
        "rank",
        "panel_row",
        "chrom",
        "pos",
        "ref",
        "alt",
        "label",
        "subset",
        "consequence_cre",
        "match_group",
        "delta",
        "abs_delta",
        "ref_activation",
        "alt_activation",
        "position_status",
        "repeat_class",
        "repeat_family",
        "repeat_name",
        "repeat_fraction",
        "substitution",
        "local_gc_quintile",
        "higher_activation_context",
    ]
    return (
        top_result.select([column for column in keep if column in top_result.columns]),
        pl.DataFrame(profile_rows).sort(
            ["arm", "feature_id", "orientation", "position", "base"]
        ),
    )


def target_sensitivity(
    responses: pl.DataFrame,
    outcomes: dict[str, tuple[np.ndarray, pl.DataFrame]],
) -> pl.DataFrame:
    target_lookup: dict[str, np.ndarray] = {}
    target_metadata: dict[str, dict[str, Any]] = {}
    for values, catalog in outcomes.values():
        for target in catalog.iter_rows(named=True):
            target_id = target["target_id"]
            target_lookup[target_id] = values[:, target["target_index"]]
            target_metadata[target_id] = target

    rows: list[dict[str, Any]] = []
    keys = ["arm", "report_block", "feature_id", "feature_role", "orientation"]
    for key, group in responses.group_by(keys, maintain_order=True):
        metadata = dict(zip(keys, key, strict=True))
        group = group.sort("panel_row")
        x_raw = group["abs_delta"].to_numpy()
        assert group["panel_row"].to_list() == list(range(EXPECTED_ROWS))
        for target_id in TARGETS_BY_FEATURE[(metadata["arm"], metadata["feature_id"])]:
            y = target_lookup[target_id]
            x_cutoff = float(np.quantile(x_raw, 0.99))
            y_cutoff = float(np.quantile(y, 0.99))
            protocols = {
                "raw": (x_raw, np.ones(len(x_raw), dtype=bool), "pearson"),
                "log1p_feature": (
                    np.log1p(x_raw),
                    np.ones(len(x_raw), dtype=bool),
                    "pearson",
                ),
                "spearman": (x_raw, np.ones(len(x_raw), dtype=bool), "spearman"),
                "trim_feature_top_1pct": (x_raw, x_raw <= x_cutoff, "pearson"),
                "trim_outcome_top_1pct": (x_raw, y <= y_cutoff, "pearson"),
                "trim_both_top_1pct": (
                    x_raw,
                    (x_raw <= x_cutoff) & (y <= y_cutoff),
                    "pearson",
                ),
            }
            for protocol, (x, mask, statistic) in protocols.items():
                effect, pvalue = _safe_correlation(
                    x[mask],
                    y[mask],
                    statistic=statistic,
                )
                rows.append(
                    {
                        **metadata,
                        "target_id": target_id,
                        "target_name": target_metadata[target_id]["target_name"],
                        "target_group_axis": target_metadata[target_id]["group_axis"],
                        "target_track_count": target_metadata[target_id]["track_count"],
                        "protocol": protocol,
                        "statistic": statistic,
                        "n": int(mask.sum()),
                        "effect": effect,
                        "pvalue": pvalue,
                        "feature_p99": x_cutoff,
                        "outcome_p99": y_cutoff,
                    }
                )
    result = pl.DataFrame(rows)
    assert result["pvalue"].is_finite().all()
    corrected: list[pl.DataFrame] = []
    for _, family in result.group_by(
        ["arm", "feature_id", "orientation"], maintain_order=True
    ):
        corrected.append(
            family.with_columns(
                pl.Series("qvalue", benjamini_hochberg(family["pvalue"].to_numpy()))
            )
        )
    return pl.concat(corrected).sort(
        ["arm", "feature_id", "orientation", "target_id", "protocol"]
    )


def _write_table(
    frame: pl.DataFrame,
    path: Path,
    artifacts: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    frame.write_parquet(path, compression="zstd")
    artifacts[str(path.relative_to(output_dir))] = {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": frame.height,
    }


def analyze(
    *,
    panel_path: Path,
    activation_root: Path,
    extraction_manifest_path: Path,
    repeat_panel_path: Path,
    repeat_manifest_path: Path,
    fasta_path: Path,
    alphagenome_uri: str,
    track_taxonomy_path: Path,
    taxonomy_manifest_path: Path,
    output_dir: Path,
    analysis_commit: str,
) -> dict[str, Any]:
    assert len(analysis_commit) == 40 and all(
        character in "0123456789abcdef" for character in analysis_commit
    )
    assert not output_dir.exists()
    started = time.monotonic()

    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    assert extraction_manifest["panel"]["sha256"] == sha256_file(panel_path)
    assert extraction_manifest["panel"]["rows"] == EXPECTED_ROWS
    panel = pl.read_parquet(panel_path)
    assert panel.height == EXPECTED_ROWS
    assert panel.select(pl.struct(KEYS).n_unique()).item() == EXPECTED_ROWS

    contexts = build_contexts(panel, fasta_path)
    repeats = validate_repeat_panel(
        panel,
        repeat_panel_path,
        repeat_manifest_path,
    )
    annotations = (
        panel.with_row_index("panel_row")
        .join(contexts, on="panel_row", how="inner", validate="1:1")
        .join(repeats, on="panel_row", how="inner", validate="1:1")
    )
    assert annotations.height == EXPECTED_ROWS

    responses = load_responses(activation_root, extraction_manifest_path)
    annotated = responses.join(
        annotations,
        on="panel_row",
        how="inner",
        validate="m:1",
    )
    assert annotated.height == responses.height

    taxonomy_manifest = json.loads(taxonomy_manifest_path.read_text())
    verify_artifact(
        track_taxonomy_path,
        taxonomy_manifest["artifacts"]["track_taxonomy.parquet"],
    )
    mapping = pl.read_parquet(track_taxonomy_path).sort("track_id")
    outcomes = load_grouped_outcomes(
        panel=panel,
        alphagenome_uri=alphagenome_uri,
        mapping=mapping,
    )

    output_dir.mkdir(parents=True)
    artifacts: dict[str, dict[str, Any]] = {}
    tables = {
        "variant_contexts.parquet": annotations,
        "selected_feature_responses.parquet": responses,
        "feature_summary.parquet": feature_summary(responses),
        "orientation_concordance.parquet": orientation_concordance(responses),
        "category_associations.parquet": category_associations(annotated),
        "sequence_metric_associations.parquet": sequence_metric_associations(annotated),
        "target_sensitivity.parquet": target_sensitivity(responses, outcomes),
    }
    top, profiles = top_variants_and_profiles(annotated)
    tables["top_variants.parquet"] = top
    tables["position_base_enrichment.parquet"] = profiles
    for filename, frame in tables.items():
        _write_table(frame, output_dir / filename, artifacts, output_dir)

    results = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": os.environ.get("RUN_ID", output_dir.name),
        "analysis_commit": analysis_commit,
        "elapsed_seconds": time.monotonic() - started,
        "scope": {
            "features": [
                {
                    "arm": arm,
                    "feature_id": feature_id,
                    "role": FEATURE_ROLES[(arm, feature_id)],
                }
                for arm, feature_ids in FEATURES_BY_ARM.items()
                for feature_id in feature_ids
            ],
            "orientations": list(ORIENTATIONS),
            "population": "all 16,140 Mendelian variants",
            "status": "post-hoc characterization after complete-family AlphaGenome L2 selection",
        },
        "protocol": {
            "sequence_context_bp": 2 * CONTEXT_FLANK + 1,
            "top_contexts_per_feature_orientation": TOP_CONTEXTS,
            "repeat_annotation": "outcome-blind audited issue #435 panel",
            "category_tests": "one-vs-rest Welch t-test and Mann-Whitney; BH within feature x orientation x annotation dimension",
            "continuous_tests": "Pearson and Spearman; BH within feature x orientation x statistic",
            "tail_sensitivity": "raw, log1p(feature), Spearman, and Pearson after separately/jointly trimming each variable's top 1%",
            "strand_handling": "forward and reverse complement are analyzed separately; concordance is descriptive",
        },
        "inputs": {
            "panel_sha256": sha256_file(panel_path),
            "extraction_manifest_sha256": sha256_file(extraction_manifest_path),
            "repeat_panel_sha256": sha256_file(repeat_panel_path),
            "repeat_manifest_sha256": sha256_file(repeat_manifest_path),
            "fasta": str(fasta_path),
            "alphagenome_uri": alphagenome_uri,
            "taxonomy_manifest_sha256": sha256_file(taxonomy_manifest_path),
        },
        "artifacts": artifacts,
    }
    write_json(output_dir / "results.json", results)
    artifacts["results.json"] = {
        "bytes": (output_dir / "results.json").stat().st_size,
        "sha256": sha256_file(output_dir / "results.json"),
    }
    write_json(output_dir / "manifest.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--activation-root", type=Path, required=True)
    parser.add_argument("--extraction-manifest", type=Path, required=True)
    parser.add_argument("--repeat-panel", type=Path, required=True)
    parser.add_argument("--repeat-manifest", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--alphagenome-uri", required=True)
    parser.add_argument("--track-taxonomy", type=Path, required=True)
    parser.add_argument("--taxonomy-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-commit", required=True)
    args = parser.parse_args()
    results = analyze(
        panel_path=args.panel,
        activation_root=args.activation_root,
        extraction_manifest_path=args.extraction_manifest,
        repeat_panel_path=args.repeat_panel,
        repeat_manifest_path=args.repeat_manifest,
        fasta_path=args.fasta,
        alphagenome_uri=args.alphagenome_uri,
        track_taxonomy_path=args.track_taxonomy,
        taxonomy_manifest_path=args.taxonomy_manifest,
        output_dir=args.output_dir,
        analysis_commit=args.analysis_commit,
    )
    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
