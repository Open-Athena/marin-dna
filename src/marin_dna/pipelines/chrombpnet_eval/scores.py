"""Load ARSENAL ChromBPNet variant scores and align them to our QTL parquets.

ARSENAL's ``snp_score`` output (``variant_scores.tsv``) is one row per scored
variant carrying the study's ``allele1`` / ``allele2`` and a signed
``logfc = log2(allele2_pred_counts / allele1_pred_counts)``. To evaluate on
*our* caqtl/dsqtl splits we join those scores onto our variant parquets and
re-orient the score so it is signed to *our* ``alt`` — matching the orientation
of our signed ``effect`` so the signed correlation is meaningful.

Coordinate systems:

- **caQTL** is native hg38 on both sides → join directly on ``(chrom, pos)``.
- **dsQTL** ARSENAL scores are **hg19** (scored against ``male.hg19.fa``) while
  our parquet is hg38-lifted → lift the ARSENAL coords hg19→hg38 with the same
  ``lift_hg19_to_hg38`` used to build our parquet before joining.

The join is keyed on ``(chrom, pos, {ref,alt} as an unordered pair)`` so it is
robust to ref/alt-orientation differences (``check_ref_alt`` may have swapped our
ref/alt to match the reference) and to multi-allelic sites.
"""

from __future__ import annotations

import polars as pl

from marin_dna.pipelines.evals.variants import lift_hg19_to_hg38

ARSENAL_SCORE_COLUMNS = ["chr", "pos", "allele1", "allele2", "variant_id", "logfc"]


def load_arsenal_scores(path: str, *, flip_logfc: bool = False) -> pl.DataFrame:
    """Read an ARSENAL ``variant_scores.tsv``.

    Returns ``[chrom, pos, allele1, allele2, variant_id, logfc]`` with ``chrom``
    stripped of any ``chr`` prefix and alleles upper-cased.

    Args:
        path: TSV path.
        flip_logfc: negate ``logfc`` on load. **Needed for ARSENAL's released
            dsQTL (yoruban) scores**, whose ``logfc`` sign convention is the
            opposite of the study effect (``obs.estimate``) / DART-Eval — verified
            two ways: (1) ARSENAL's own ``supervised_variant_scoring_yoruba.ipynb``
            correlates ``-1 * obs.estimate`` against their ``logfc`` (the african
            caQTL notebook does *not* negate), and (2) the released dsQTL ``logfc``
            anti-correlates (r≈-0.97) with the in-benchmark
            ``pred.chrombpnet.encsr000emt.varscore.logfc`` despite identical
            allele order. Our ``effect`` follows the DART-Eval convention
            (``obs.estimate`` as-is), so we negate ARSENAL's released dsQTL
            ``logfc`` here to put both on the same axis. (Not needed for caQTL.
            The supervised scorer hits the *same* dsQTL flip — its predicted
            alt-log2FC anti-correlates with this ``effect`` — so it applies the
            equivalent per-dataset flip; see ``qtl_eval.QTL_DATASETS``.)
    """
    df = pl.read_csv(path, separator="\t")
    missing = [c for c in ARSENAL_SCORE_COLUMNS if c not in df.columns]
    assert not missing, f"ARSENAL score TSV missing {missing}; got {df.columns}"
    sign = -1.0 if flip_logfc else 1.0
    out = df.select(
        pl.col("chr").cast(pl.Utf8).str.replace(r"^chr", "").alias("chrom"),
        pl.col("pos").cast(pl.Int64),
        pl.col("allele1").cast(pl.Utf8).str.to_uppercase(),
        pl.col("allele2").cast(pl.Utf8).str.to_uppercase(),
        pl.col("variant_id").cast(pl.Utf8),
        (pl.col("logfc").cast(pl.Float64) * sign).alias("logfc"),
    )
    # SNP-only: a stray indel/N would build a malformed allele-pair key that
    # silently fails to join later — fail loud at the boundary instead.
    nuc = ["A", "C", "G", "T"]
    bad = out.filter(~pl.col("allele1").is_in(nuc) | ~pl.col("allele2").is_in(nuc))
    assert bad.height == 0, (
        f"load_arsenal_scores: {bad.height} rows with non-ACGT alleles (SNP-only "
        f"expected); e.g. {bad.select('chrom', 'pos', 'allele1', 'allele2').head(3)}"
    )
    return out


def _unordered_pair(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """An orientation-independent ``"X/Y"`` key for an allele pair (sorted)."""
    return (
        pl.when(a <= b)
        .then(pl.concat_str([a, pl.lit("/"), b]))
        .otherwise(pl.concat_str([b, pl.lit("/"), a]))
    )


def align_scores_to_variants(
    variants: pl.DataFrame,
    scores: pl.DataFrame,
    *,
    lift: bool,
    score_out: str = "score",
    min_coverage: float = 0.95,
) -> pl.DataFrame:
    """Join ARSENAL ``scores`` onto our ``variants``, re-oriented to our ``alt``.

    Args:
        variants: our QTL parquet, with at least ``chrom, pos, ref, alt``.
        scores: output of :func:`load_arsenal_scores`.
        lift: if True, lift ``scores`` hg19→hg38 before joining (dsQTL). caQTL
            is already hg38 → pass ``False``.
        score_out: name of the appended signed-score column.
        min_coverage: assert at least this fraction of ``variants`` receive a
            score (a loud guard against a coordinate/build mismatch).

    Returns:
        ``variants`` (unmatched rows dropped) plus a ``score_out`` column equal
        to ``logfc`` when our ``alt`` is ARSENAL's ``allele2`` and ``-logfc``
        when our ``alt`` is ARSENAL's ``allele1`` — i.e. the model's signed
        allelic score oriented to our ``alt``.
    """
    assert {"chrom", "pos", "ref", "alt"} <= set(variants.columns), (
        f"variants missing coordinate columns; got {variants.columns}"
    )
    n_variants = variants.height

    if lift:
        # lift_hg19_to_hg38 transforms chrom/pos/ref/alt (RCing alleles on the
        # minus strand) and preserves all other columns; map allele1/allele2 to
        # ref/alt around the call so the alleles are lifted consistently.
        lifted = (
            scores.rename({"allele1": "ref", "allele2": "alt"})
            .pipe(lift_hg19_to_hg38)
            .filter(pl.col("pos") != -1)
            .rename({"ref": "allele1", "alt": "allele2"})
        )
        scores = lifted

    v = variants.with_columns(
        _unordered_pair(pl.col("ref"), pl.col("alt")).alias("_pair")
    )
    s_paired = scores.with_columns(
        _unordered_pair(pl.col("allele1"), pl.col("allele2")).alias("_pair")
    )
    # A (chrom,pos,pair) group with conflicting logfc would make the dedup below
    # pick an arbitrary, order-dependent score — fail loud instead.
    conflicts = (
        s_paired.group_by(["chrom", "pos", "_pair"])
        .agg(pl.col("logfc").n_unique().alias("_n"))
        .filter(pl.col("_n") > 1)
    )
    assert conflicts.height == 0, (
        f"{conflicts.height} (chrom,pos,pair) score groups have conflicting logfc; "
        f"dedup would be arbitrary — e.g. {conflicts.head(3)}"
    )
    s = s_paired.unique(subset=["chrom", "pos", "_pair"], keep="first")

    joined = v.join(
        s.select(["chrom", "pos", "_pair", "allele1", "allele2", "logfc"]),
        on=["chrom", "pos", "_pair"],
        how="left",
    )
    # `s` is unique per (chrom,pos,_pair), so a left join cannot fan out — assert
    # it (a weakened dedup would otherwise double-count rows into the metric).
    assert joined.height == n_variants, (
        f"join fan-out: {joined.height} rows from {n_variants} variants"
    )
    n_matched = joined.filter(pl.col("logfc").is_not_null()).height
    coverage = n_matched / n_variants
    assert coverage >= min_coverage, (
        f"score join coverage {coverage:.3f} < {min_coverage} "
        f"({n_matched}/{n_variants}) — suspect a coordinate/build mismatch "
        f"(lift={lift})"
    )
    # Orientation invariant: a matched variant's alt must be exactly one of the
    # score's two alleles (guaranteed by the unordered-pair join). Assert so a
    # future join-key change can't silently route rows through the -logfc branch.
    bad_orient = joined.filter(
        pl.col("logfc").is_not_null()
        & (pl.col("alt") != pl.col("allele1"))
        & (pl.col("alt") != pl.col("allele2"))
    )
    assert bad_orient.height == 0, (
        f"{bad_orient.height} matched rows whose alt is neither ARSENAL allele"
    )

    # Orient to our alt: +logfc if our alt is allele2, -logfc if our alt is
    # allele1. The unordered-pair join guarantees {ref,alt}=={allele1,allele2},
    # so a matched row's alt is always exactly one of them.
    return (
        joined.with_columns(
            pl.when(pl.col("alt") == pl.col("allele2"))
            .then(pl.col("logfc"))
            .otherwise(-pl.col("logfc"))
            .alias(score_out)
        )
        .filter(pl.col("logfc").is_not_null())
        .drop("_pair", "allele1", "allele2", "logfc")
    )
