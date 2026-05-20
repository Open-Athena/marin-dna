"""Per-variant pos/neg labeling from per-(variant, study) PIP rows.

Used by both `complex_traits` (UKBB SuSiE+FINEMAP across 119 traits) and
`eqtl` (Finucane GTEx SuSiE across 49 tissues) eval datasets. Each row in
the input represents one fine-mapping observation: a variant tested in one
"study" — a trait for `complex_traits`, a tissue for `eqtl`. A variant
typically has multiple rows when it was tested in multiple studies.
"""

from __future__ import annotations

import polars as pl

from marin_dna.pipelines.evals.variants import COORDINATES


def label_variants_by_pip(
    df: pl.LazyFrame | pl.DataFrame,
    *,
    pip_pos_threshold: float,
    pip_neg_threshold: float,
    use_null_pip_guard: bool,
    extra_aggs: list[pl.Expr] | None = None,
) -> pl.LazyFrame | pl.DataFrame:
    """Group by variant and assign positive/negative/excluded labels from PIPs.

    The label cascade uses the per-variant *aggregated max* PIP across all
    studies (traits / tissues / etc.) that tested the variant:

    - ``max(pip) > pip_pos_threshold`` → ``True`` (positive).
    - ``max(pip) < pip_neg_threshold`` (and, if ``use_null_pip_guard``, no
      tested study reported a null PIP) → ``False`` (negative).
    - otherwise (including ``max(pip) ∈ [pip_neg_threshold, pip_pos_threshold]``
      and any null-guarded variant) → label set to ``None`` and the row is
      dropped via ``.filter(label.is_not_null())``.

    **Intermediate-PIP exclusion is handled by the cascade, not by a
    pre-filter.** Do NOT row-filter the input to extreme PIP rows before
    calling this function — that would silently mislabel variants whose
    only-extreme tissue PIP coexists with a mid-range tissue PIP. Concrete
    bug case: a variant with tissue PIPs ``[0.001, 0.5]`` should be
    excluded (max = 0.5, intermediate). With a ``(pip > pos) | (pip < neg)``
    pre-filter, only the 0.001 row survives, ``max = 0.001 < pip_neg`` and
    the variant gets mislabeled as a clean negative. See
    ``test_multiple_studies_max_excludes_when_intermediate``.

    **"Null PIP" vs. "study did not test variant".** Two distinct cases the
    aggregation handles differently:

    - *Null PIP*: variant has a row in the study but the PIP value is null.
      In ``complex_traits`` this is the SuSiE/FINEMAP combine-step
      disagreement output (when ``|pip_susie - pip_finemap| > 0.05`` the
      combined PIP is set to null as a quality filter). With
      ``use_null_pip_guard=True``, any null PIP among the tested studies
      forbids the variant from being a negative — we only call something a
      confident negative if every tested study agrees.

    - *Study did not test the variant*: variant has no row at all in that
      study (typically because the variant isn't in any fine-mapped region
      for that study). This case never enters the ``group_by`` aggregation,
      so it doesn't trigger the null guard. Negatives are NOT required to
      be tested in *all* studies — typical fine-mapping outputs only cover
      significant regions, so most variants are tested in only a small
      subset of studies and that's fine.

    Args:
        df: rows of ``(chrom, pos, ref, alt, pip, ...)``. May be a
            ``LazyFrame`` (recommended; the caller can ``sink_parquet`` the
            result without materializing intermediates).
        pip_pos_threshold: positive cutoff (typical: ``0.9``).
        pip_neg_threshold: negative cutoff (typical: ``0.01``).
        use_null_pip_guard: if ``True``, a variant cannot be labeled
            negative if any of its tested studies has a null PIP. Set
            ``False`` for single-method sources where null PIPs aren't
            produced (e.g. SuSiE-only ``eqtl``).
        extra_aggs: additional polars expressions to compute alongside
            ``pip.max()`` in the per-variant aggregation. Common patterns:
            ``pl.col("rsid").first()`` (carry-along constant per variant);
            ``pl.col("trait").filter(pl.col("pip") > pip_pos_threshold).unique()``
            (collect the studies in which the variant was a positive).

    Returns:
        Per-variant frame with ``COORDINATES + pip + label + extra_aggs``
        cols. Only ``label ∈ {True, False}`` rows survive; intermediate-PIP
        and null-guarded variants are dropped.
    """
    extra = list(extra_aggs) if extra_aggs else []
    aggs: list[pl.Expr] = [pl.col("pip").max(), *extra]
    if use_null_pip_guard:
        aggs.append(pl.col("pip").is_null().any().alias("_any_null_pip"))
    grouped = df.group_by(COORDINATES).agg(aggs)

    neg_cond = pl.col("pip") < pip_neg_threshold
    if use_null_pip_guard:
        neg_cond = neg_cond & ~pl.col("_any_null_pip")

    out = grouped.with_columns(
        pl.when(pl.col("pip") > pip_pos_threshold)
        .then(pl.lit(True))
        .when(neg_cond)
        .then(pl.lit(False))
        .otherwise(pl.lit(None))
        .alias("label")
    ).filter(pl.col("label").is_not_null())

    if use_null_pip_guard:
        out = out.drop("_any_null_pip")

    return out
