"""Per-variant conservation scoring via UCSC bigWig tracks.

Used by ``snakemake/conservation_eval/`` (issue #146) to score matched-pair
variant-effect datasets (e.g. ``bolinas-dna/evals_mendelian_traits``,
``bolinas-dna/evals_complex_traits``) with classical conservation tracks:

- ``phyloP_100v``    — UCSC 100-vertebrate phyloP (multiz alignment)
- ``phastCons_100v`` — UCSC 100-vertebrate phastCons (multiz alignment)
- ``phyloP_241m``    — Zoonomia 241-mammal Cactus phyloP
- ``phyloP_447m``    — UCSC 447-way phyloP (Zoonomia + densely-sampled primates, Cactus)
- ``phyloP_470m``    — UCSC 470-way phyloP (multiz; parallel work to the 447-way Cactus, not a successor)
- ``phastCons_470m`` — UCSC 470-way phastCons (multiz; parallel work to the 447-way Cactus, not a successor)
- ``phastCons_43p``  — Zoonomia 43-primate track (TraitGym name; underlying file is phyloP-over-primates)

The first three tracks (``phyloP_100v``, ``phyloP_241m``, ``phastCons_43p``)
are the original TraitGym set; their URLs are copied verbatim from
TraitGym's ``eval/workflow/rules/conservation.smk``. The same bigWigs are
also used elsewhere in this repo (enhancer_classification,
training_dataset/dataset_creation) but each pipeline manages its own
download to avoid coupling.

NaN policy: this module preserves NaNs from the bigWig (no alignment at that
locus). Callers decide how to fill them — the ``conservation_eval`` pipeline
applies ``fillna(0)`` only at the metrics-aggregation step.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

from bolinas.pipelines.evals.metrics import (
    GLOBAL_SUBSET,
    MACRO_AVG_SUBSET,
    compute_auprc_metrics,
)


CONSERVATION_TRACKS: dict[str, str] = {
    # 100-vertebrate UCSC multiz alignment.
    "phyloP_100v": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw",
    "phastCons_100v": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw",
    # Zoonomia 241-mammal Cactus alignment.
    "phyloP_241m": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/cactus241way/cactus241way.phyloP.bw",
    # UCSC 447-way Cactus (Zoonomia + densely-sampled primates, Kuderna et al. 2023).
    "phyloP_447m": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP447way/hg38.phyloP447way.bw",
    # UCSC 470-way: multiz alignment (per UCSC track description), parallel
    # work to the 447-way Cactus rather than a successor. Distinct aligner,
    # different coverage characteristics — not "newer/better mammal alignment".
    "phyloP_470m": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP470way/hg38.phyloP470way.bw",
    "phastCons_470m": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons470way/hg38.phastCons470way.bw",
    # phastCons_43p name is a TraitGym convention; the underlying file is
    # actually phyloP over 43 primates from the Zoonomia track hub. We keep
    # the name to stay consistent with TraitGym + existing config in this repo.
    "phastCons_43p": "https://cgl.gi.ucsc.edu/data/cactus/zoonomia-2021-track-hub/hg38/phyloPPrimates.bigWig",
}


# Variant columns the pipeline preserves end-to-end. Asserted by the score
# and aggregate stages so a schema drift fails fast. ``match_group`` links
# 1:k matched positives and negatives produced by ``snakemake/evals/`` (PR
# #194 switched the matched-pair datasets to 1:9 via k=9 nearest neighbors).
REQUIRED_VARIANT_COLUMNS: tuple[str, ...] = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "subset",
    "match_group",
)


def score_variants_at_positions(
    df: pd.DataFrame,
    bw_path: str | Path,
) -> np.ndarray:
    """Look up a bigWig value for each variant in ``df``.

    TraitGym variants are 1-based (VCF convention); pyBigWig is 0-based
    half-open. The conversion ``[pos - 1, pos)`` selects the single base at
    1-based ``pos``.

    NaNs are preserved: pyBigWig returns ``nan`` at positions with no data.

    Args:
        df: Must have integer ``pos`` and string ``chrom`` columns. ``chrom``
            entries without a ``chr`` prefix are auto-prefixed to match UCSC
            bigWig naming.
        bw_path: Path to the bigWig file.

    Returns:
        ``np.ndarray`` of shape ``(len(df),)`` with one float per row.
    """
    assert "chrom" in df.columns and "pos" in df.columns, (
        "df must have chrom + pos columns"
    )
    assert pd.api.types.is_integer_dtype(df["pos"]), (
        f"pos must be integer dtype, got {df['pos'].dtype}"
    )

    bw = pyBigWig.open(str(bw_path))
    try:
        bw_chroms = set(bw.chroms())
        scores = np.empty(len(df), dtype=np.float64)
        for i, (chrom, pos) in enumerate(zip(df["chrom"], df["pos"])):
            chrom_str = str(chrom)
            if not chrom_str.startswith("chr"):
                chrom_str = f"chr{chrom_str}"
            if chrom_str not in bw_chroms:
                # No track for this chromosome (e.g. patches, alt contigs).
                scores[i] = np.nan
                continue
            # 1-based pos -> 0-based half-open [pos-1, pos): one base.
            vals = bw.values(chrom_str, int(pos) - 1, int(pos))
            scores[i] = vals[0] if vals else np.nan
    finally:
        bw.close()

    return scores


def aggregate_conservation_metrics(
    parquet_paths: dict[str, str | Path],
    n_min: int = 30,
    *,
    n_bootstrap: int = 1000,
    bootstrap_seed: int | None = 0,
) -> tuple[pd.DataFrame, str]:
    """Aggregate per-score scored-variant parquets into a metrics DataFrame
    and a markdown report.

    Each input parquet is the output of ``score_variants_at_positions`` plus
    the matched-pair variant columns: must contain ``[chrom, pos, ref, alt,
    label, subset, match_group, score]``. The ``score`` column may contain
    NaN (positions with no alignment in the bigWig).

    For each score: NaN count is recorded per subset, then ``score`` is
    filled with 0 (semantically meaningful — see module docstring) before
    AUPRC + cluster-bootstrap SE is computed via
    ``bolinas.pipelines.evals.metrics.compute_auprc_metrics``. The bootstrap
    resamples ``match_group``s (the matched-pair clustering unit), so SE
    is honest under the 1:k structure of the PR #194 datasets.

    Args:
        parquet_paths: mapping ``score_name -> parquet path``. Order is
            preserved in the markdown table.
        n_min: forwarded to ``compute_auprc_metrics`` — minimum subset
            ``n_groups`` for inclusion in the macro-average aggregate row.
        n_bootstrap: bootstrap iterations per (subset, score).
        bootstrap_seed: seed for the cluster bootstrap; ``None`` for fresh
            randomness. Default ``0`` keeps outputs bit-stable across re-runs.

    Returns:
        ``(metrics_df, markdown)`` where ``metrics_df`` has columns
        ``[score_type, score_name, subset, value, se, n_groups, n_rows,
        n_nan, n_total]``. Includes ``_global_`` and ``_macro_avg_`` aggregate
        rows per score (used by downstream leaderboard rendering); these are
        excluded from the markdown report, which stays per-subset.
    """
    assert parquet_paths, "parquet_paths must be non-empty"

    score_names = list(parquet_paths)
    required = (*REQUIRED_VARIANT_COLUMNS, "score")
    all_metrics: list[pd.DataFrame] = []

    for score_name in score_names:
        df = pd.read_parquet(parquet_paths[score_name])
        for col in required:
            assert col in df.columns, (
                f"{parquet_paths[score_name]}: missing column {col!r}"
            )

        # Count NaN per subset before filling.
        nan_per_subset = df.groupby("subset")["score"].apply(
            lambda s: int(s.isna().sum())
        )
        total_per_subset = df.groupby("subset").size().astype(int)

        m = compute_auprc_metrics(
            dataset=df[list(REQUIRED_VARIANT_COLUMNS)],
            scores=df[["score"]].fillna(0),
            score_columns=["score"],
            n_min=n_min,
            n_bootstrap=n_bootstrap,
            rng=bootstrap_seed,
        )
        m["score_name"] = score_name

        # n_nan / n_total for aggregate rows: _global_ covers every variant;
        # _macro_avg_ covers only the subsets that contribute (n_groups >= n_min).
        qualifying = set(
            m.loc[
                ~m["subset"].isin([GLOBAL_SUBSET, MACRO_AVG_SUBSET])
                & (m["n_groups"] >= n_min),
                "subset",
            ]
        )
        total_nan = int(df["score"].isna().sum())
        total_rows = int(len(df))
        qualifying_nan = int(
            df.loc[df["subset"].isin(qualifying), "score"].isna().sum()
        )
        qualifying_total = int(df["subset"].isin(qualifying).sum())

        nan_map = {
            **nan_per_subset.to_dict(),
            GLOBAL_SUBSET: total_nan,
            MACRO_AVG_SUBSET: qualifying_nan,
        }
        total_map = {
            **total_per_subset.to_dict(),
            GLOBAL_SUBSET: total_rows,
            MACRO_AVG_SUBSET: qualifying_total,
        }
        m["n_nan"] = m["subset"].map(nan_map).astype(int)
        m["n_total"] = m["subset"].map(total_map).astype(int)
        all_metrics.append(m)

    metrics = pd.concat(all_metrics, ignore_index=True)
    md = _build_markdown(metrics, score_names)
    return metrics, md


def _build_markdown(metrics: pd.DataFrame, score_names: list[str]) -> str:
    """Render the metrics DataFrame as a two-table markdown report.

    AUPRC table: one row per ``subset`` (no ``global`` / ``mean`` aggregate
    row). Each cell is ``f"{value:.3f} ± {se:.3f}"``.

    NaN-counts table: per-subset NaN counts plus per-subset n_total.
    """
    # Drop aggregate rows so this per-pipeline report stays per-subset. The
    # aggregates still flow through to the returned metrics DataFrame (and
    # downstream parquet), where the dashboard data loader picks them up.
    metrics = metrics[~metrics["subset"].isin([GLOBAL_SUBSET, MACRO_AVG_SUBSET])]

    def _pivot(values_col: str) -> pd.DataFrame:
        return metrics.pivot_table(
            index="subset",
            columns="score_name",
            values=values_col,
            aggfunc="first",
        )

    # Per-subset coverage (n_groups / n_rows) from the first score — subset
    # coverage is score-independent.
    coverage = (
        metrics[metrics["score_name"] == score_names[0]][
            ["subset", "n_groups", "n_rows", "n_total"]
        ]
        .drop_duplicates(subset="subset")
        .set_index("subset")
    )

    val_pivot = _pivot("value")
    se_pivot = _pivot("se")
    nan_pivot = _pivot("n_nan")

    per_subset = list(coverage.sort_values("n_groups", ascending=False).index)

    lines: list[str] = []
    lines.append("### Conservation — AUPRC ± cluster-bootstrap SE")
    lines.append("")
    lines.append(
        "Per-subset AUPRC ± cluster-bootstrap SE. Each iteration resamples "
        "the unique `match_group` IDs (1:k matched positive + k negative "
        "variants) with replacement, gathers all rows in the sampled groups, "
        "and recomputes AUPRC; SE is the std of the bootstrap distribution. "
        "Clustering on `match_group` honors the non-iid pairing structure."
    )
    lines.append("")
    header = ["subset", "n_groups", *score_names]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for subset in per_subset:
        n_groups = int(coverage.loc[subset, "n_groups"])
        cells: list[str] = []
        for s in score_names:
            v = val_pivot.loc[subset, s] if s in val_pivot.columns else float("nan")
            e = se_pivot.loc[subset, s] if s in se_pivot.columns else float("nan")
            cells.append(f"{v:.3f} ± {e:.3f}" if pd.notna(v) and pd.notna(e) else "—")
        lines.append("| " + " | ".join([subset, str(n_groups), *cells]) + " |")

    # NaN counts: per-subset rows ordered the same way as the metric table.
    lines.append("")
    lines.append("### NaN counts")
    lines.append("")
    lines.append(
        "NaN = no alignment at that locus in the bigWig. AUPRC above is "
        "computed after `.fillna(0)`: 0 is semantically meaningful for both "
        "phyloP (neither conserved nor accelerated) and phastCons "
        "(non-conserved). `auprc_with_bootstrap_se` also asserts no NaN "
        "scores, so the fill is required upstream of the metric call."
    )
    lines.append("")
    nan_header = ["subset", "n_total", *score_names]
    lines.append("| " + " | ".join(nan_header) + " |")
    lines.append("| " + " | ".join(["---"] * len(nan_header)) + " |")
    for subset in per_subset:
        n_total = int(coverage.loc[subset, "n_total"])
        vals = [
            str(int(nan_pivot.loc[subset, s]))
            if s in nan_pivot.columns and pd.notna(nan_pivot.loc[subset, s])
            else "0"
            for s in score_names
        ]
        lines.append("| " + " | ".join([subset, str(n_total), *vals]) + " |")

    return "\n".join(lines) + "\n"
