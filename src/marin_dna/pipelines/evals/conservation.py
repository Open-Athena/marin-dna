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
also used elsewhere in this repo (training_dataset/dataset_creation)
but each pipeline manages its own
download to avoid coupling.

NaN policy: this module preserves NaNs from the bigWig (no alignment at that
locus). Callers decide how to fill them — the ``conservation_eval`` pipeline
applies ``fillna(0)`` at the metrics-aggregation step (every eval protocol,
SGE included; 0 is meaningful — neither conserved nor accelerated for phyloP,
non-conserved for phastCons).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

from marin_dna.pipelines.evals.metrics import (
    GLOBAL_SUBSET,
    MACRO_AVG_SUBSET,
    compute_auprc_metrics,
    compute_qtl_metrics,
    compute_sge_metrics,
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
# 1:k matched positives and negatives of the matched-pair eval datasets (PR
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

# Columns the unmatched DART-Eval QTL datasets (``caqtl`` / ``dsqtl``, PR
# #214) carry instead: no ``subset`` / ``match_group`` (no matching, no
# subsampling), but a continuous ``effect_size`` (unsigned ``|effect|``) used
# by the positives-only correlation metric. Asserted by the ``eval_protocol:
# qtl_global`` branch of each pipeline's score/metric rules. Shared home here
# mirrors ``REQUIRED_VARIANT_COLUMNS`` (imported by all three eval pipelines).
QTL_VARIANT_COLUMNS: tuple[str, ...] = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "label",
    "effect_size",
)

# Columns the saturation-genome-editing dataset (``evals_sge`` v3, issue #301)
# carries instead of ``match_group`` / ``effect_size``: a boolean ``label`` (True =
# impactful = calibrated abnormal — the AUPRC target), the MaveDB ``mavedb_urn``
# (the per-study grouping key — scores are non-comparable across studies, so AUPRC
# is computed per accession), the ``gene`` (display), and the consequence-group
# ``subset`` ∈ {missense_variant, splicing}. Asserted by the ``eval_protocol: sge``
# branch of each pipeline's score/metric rules. ``compute_variant_scores`` itself
# only reads chrom/pos/ref/alt; asserting the metric columns early fails fast on
# schema drift, since ``compute_sge_metrics`` depends on them surviving into the
# scores parquet.
SGE_VARIANT_COLUMNS: tuple[str, ...] = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "mavedb_urn",
    "gene",
    "subset",
    "label",
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


def fill_score_nan(scores: pd.DataFrame) -> pd.DataFrame:
    """The conservation NaN policy — the single shared home, used by **every**
    eval-protocol aggregate (matched_pair / qtl_global / sge).

    Unaligned bigWig loci have no value (NaN from ``score_variants_at_positions``).
    Every conservation metric fills the score with 0 before scoring, because 0 is
    semantically meaningful for both track types: neither conserved nor
    accelerated (phyloP), non-conserved (phastCons). Callers that report an
    ``n_nan`` diagnostic must count NaNs **before** calling this.
    """
    return scores.fillna(0)


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
    ``marin_dna.pipelines.evals.metrics.compute_auprc_metrics``. The bootstrap
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
            scores=fill_score_nan(df[["score"]]),
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


def aggregate_conservation_qtl_metrics(
    parquet_paths: dict[str, str | Path],
    *,
    n_bootstrap: int = 1000,
    bootstrap_seed: int | None = 0,
) -> tuple[pd.DataFrame, str]:
    """Global QTL metrics across per-track scored parquets (caqtl/dsqtl).

    The ``eval_protocol: qtl_global`` counterpart to
    ``aggregate_conservation_metrics``: the DART-Eval QTL datasets have no
    ``subset`` / ``match_group``, so there is no per-subset stratification or
    cluster bootstrap. Each input parquet is ``score_variants_at_positions``
    output plus the QTL variant columns: must contain ``[label, effect_size,
    score]``. ``score`` may be NaN (no alignment in the bigWig).

    Per track: NaN count recorded, then ``score`` is filled with 0 (same
    rationale as the matched-pair path — 0 is meaningful for phyloP/phastCons)
    before ``compute_qtl_metrics`` computes global AUPRC + positives-only
    Pearson/Spearman vs ``effect_size``.

    Args:
        parquet_paths: mapping ``score_name -> parquet path``. Order is
            preserved in the markdown table.
        n_bootstrap: bootstrap iterations per (metric, track).
        bootstrap_seed: seed for the bootstrap; ``None`` for fresh randomness.
            Default ``0`` keeps outputs bit-stable across re-runs.

    Returns:
        ``(metrics_df, markdown)`` where ``metrics_df`` has columns
        ``[metric, score_type, value, se, n_rows, n_pos, score_name, n_nan,
        n_total]``. ``score_type`` is always ``"score"``; ``score_name`` is
        the conservation track.
    """
    assert parquet_paths, "parquet_paths must be non-empty"

    score_names = list(parquet_paths)
    required = ("label", "effect_size", "score")
    all_metrics: list[pd.DataFrame] = []

    for score_name in score_names:
        df = pd.read_parquet(parquet_paths[score_name])
        for col in required:
            assert col in df.columns, (
                f"{parquet_paths[score_name]}: missing column {col!r}"
            )

        m = compute_qtl_metrics(
            dataset=df[["label", "effect_size"]],
            scores=fill_score_nan(df[["score"]]),
            score_columns=["score"],
            n_bootstrap=n_bootstrap,
            rng=bootstrap_seed,
        )
        m["score_name"] = score_name
        m["n_nan"] = int(df["score"].isna().sum())
        m["n_total"] = int(len(df))
        all_metrics.append(m)

    metrics = pd.concat(all_metrics, ignore_index=True)
    md = _build_qtl_markdown(metrics, score_names)
    return metrics, md


def _build_qtl_markdown(metrics: pd.DataFrame, score_names: list[str]) -> str:
    """Render the global QTL metrics as a markdown table (metric × track)."""
    lines: list[str] = []
    lines.append("### caQTL/dsQTL — global metrics (value ± bootstrap SE)")
    lines.append("")
    lines.append(
        "AUPRC over all variants (significant QTL vs control). Pearson / "
        "Spearman of the conservation score vs `effect_size`, over positive "
        "variants only. Scores are `.fillna(0)` before scoring (0 = no "
        "alignment at that locus — neither conserved nor accelerated). "
        "Bootstrap SE resamples rows (AUPRC) / positive rows (correlations)."
    )
    lines.append("")
    n_pos = int(metrics["n_pos"].iloc[0])
    n_total = int(metrics["n_total"].iloc[0])
    lines.append(f"n_total = {n_total}; n_positives = {n_pos}.")
    lines.append("")

    val = metrics.pivot_table(
        index="metric", columns="score_name", values="value", aggfunc="first"
    )
    se = metrics.pivot_table(
        index="metric", columns="score_name", values="se", aggfunc="first"
    )
    header = ["metric", *score_names]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for metric in ("AUPRC", "pearson", "spearman"):
        if metric not in val.index:
            continue
        cells = []
        for s in score_names:
            v = val.loc[metric, s] if s in val.columns else float("nan")
            e = se.loc[metric, s] if s in se.columns else float("nan")
            cells.append(f"{v:.3f} ± {e:.3f}" if pd.notna(v) and pd.notna(e) else "—")
        lines.append("| " + " | ".join([metric, *cells]) + " |")

    # Per-track NaN counts (no alignment; filled with 0 before scoring).
    nan_map = (
        metrics.drop_duplicates("score_name").set_index("score_name")["n_nan"].to_dict()
    )
    lines.append("")
    lines.append(
        "NaN scores per track (filled with 0): "
        + ", ".join(f"{s}={int(nan_map.get(s, 0))}" for s in score_names)
    )

    return "\n".join(lines) + "\n"


def aggregate_conservation_sge_metrics(
    parquet_paths: dict[str, str | Path],
    *,
    n_bootstrap: int = 1000,
    bootstrap_seed: int | None = 0,
) -> tuple[pd.DataFrame, str]:
    """SGE metrics across per-track scored parquets (``eval_protocol: sge``).

    The conservation counterpart to the evals_v2 ``sge`` path. Each input parquet
    is ``score_variants_at_positions`` output plus SGE_VARIANT_COLUMNS
    (``[mavedb_urn, gene, subset, label, score]``). The per-track ``score`` is fed
    into the **shared**
    ``compute_sge_metrics`` (``score_columns=["score"]``) — no metric logic is
    duplicated here; only the per-track loop + ``score_name`` stamping + markdown
    are conservation-specific. ``score`` is ``.fillna(0)`` before scoring, the
    same convention as the matched-pair / QTL conservation paths (0 is
    semantically meaningful: neither conserved nor accelerated for phyloP,
    non-conserved for phastCons). The pre-fill NaN count is reported per track.

    Args:
        parquet_paths: mapping ``score_name -> parquet path``; order preserved.
        n_bootstrap: bootstrap iterations per cell.
        bootstrap_seed: seed; ``None`` for fresh randomness. Default ``0`` keeps
            outputs bit-stable across re-runs.

    Returns:
        ``(metrics_df, markdown)`` where ``metrics_df`` is ``compute_sge_metrics``
        output (``[metric, subset, accession, gene, score_type, value, se, n,
        n_pos]``, ``score_type == "score"``) plus ``[score_name, n_nan, n_total]``
        per track.
    """
    assert parquet_paths, "parquet_paths must be non-empty"

    score_names = list(parquet_paths)
    metric_cols = ["mavedb_urn", "gene", "subset", "label"]
    required = (*metric_cols, "score")
    all_metrics: list[pd.DataFrame] = []

    for score_name in score_names:
        df = pd.read_parquet(parquet_paths[score_name])
        for col in required:
            assert col in df.columns, (
                f"{parquet_paths[score_name]}: missing column {col!r}"
            )

        m = compute_sge_metrics(
            dataset=df[metric_cols],
            scores=fill_score_nan(df[["score"]]),
            score_columns=["score"],
            n_bootstrap=n_bootstrap,
            rng=bootstrap_seed,
        )
        m["score_name"] = score_name
        m["n_nan"] = int(df["score"].isna().sum())
        m["n_total"] = int(len(df))
        all_metrics.append(m)

    metrics = pd.concat(all_metrics, ignore_index=True)
    md = _build_sge_markdown(metrics, score_names)
    return metrics, md


def _build_sge_markdown(metrics: pd.DataFrame, score_names: list[str]) -> str:
    """Render the SGE headline (across-accession × across-subset macro) as a
    markdown table (metric × track). The full per-accession × per-subset grid
    lives in the metrics parquet; this is the at-a-glance summary."""
    lines: list[str] = []
    lines.append("### SGE — headline macro (value ± bootstrap SE)")
    lines.append("")
    lines.append(
        "Per-accession (MaveDB study) AUPRC predicting the binary `label` "
        "(impactful vs not) from the conservation score, macro-averaged over "
        "consequence subsets (missense/splicing) then over accessions. Unaligned "
        "(NaN) scores are filled with 0 before scoring (the conservation-pipeline "
        "convention)."
    )
    lines.append("")
    headline = metrics[
        (metrics["accession"] == MACRO_AVG_SUBSET)
        & (metrics["subset"] == MACRO_AVG_SUBSET)
    ]
    val = headline.pivot_table(
        index="metric", columns="score_name", values="value", aggfunc="first"
    )
    se = headline.pivot_table(
        index="metric", columns="score_name", values="se", aggfunc="first"
    )
    header = ["metric", *score_names]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for metric in ("AUPRC",):
        if metric not in val.index:
            continue
        cells = []
        for s in score_names:
            v = val.loc[metric, s] if s in val.columns else float("nan")
            e = se.loc[metric, s] if s in se.columns else float("nan")
            cells.append(f"{v:.3f} ± {e:.3f}" if pd.notna(v) and pd.notna(e) else "—")
        lines.append("| " + " | ".join([metric, *cells]) + " |")

    nan_map = (
        metrics.drop_duplicates("score_name").set_index("score_name")["n_nan"].to_dict()
    )
    n_total = int(metrics["n_total"].iloc[0])
    lines.append("")
    lines.append(
        f"n_total = {n_total}. NaN (unaligned) scores per track: "
        + ", ".join(f"{s}={int(nan_map.get(s, 0))}" for s in score_names)
    )
    return "\n".join(lines) + "\n"
