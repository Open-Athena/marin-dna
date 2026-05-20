"""Tests for ``label_variants_by_pip`` — per-variant PIP labeling shared by
the ``complex_traits`` and ``eqtl`` eval datasets.

These behaviours are critical correctness invariants for the eval datasets,
so the test suite is intentionally exhaustive: every label-cascade branch,
the eqtl pre-filter bug case, the null-guard semantics, the
"study coverage" semantics (variants tested in 1 vs. all studies), and
boundary values.
"""

from __future__ import annotations

import polars as pl

from marin_dna.pipelines.evals.labeling import label_variants_by_pip

# Standard thresholds used by complex_traits and eqtl in production.
POS = 0.9
NEG = 0.01


def make_rows(rows: list[dict]) -> pl.DataFrame:
    """Build a per-(variant, study) input frame from a list of dicts.

    `chrom` is forced to String to match the production schema (chr names
    are strings, not ints, since some chroms are X/Y/MT).
    """
    return pl.DataFrame(rows, schema_overrides={"chrom": pl.String})


def labels_only(df: pl.DataFrame | pl.LazyFrame) -> dict[tuple, bool]:
    """Collapse a labeled output to ``{(chrom, pos, ref, alt): label}``."""
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    return {
        (r["chrom"], r["pos"], r["ref"], r["alt"]): r["label"] for r in df.to_dicts()
    }


# ---------- single-row label cascade ----------


def test_single_positive():
    """max(pip) > pip_pos_threshold → True."""
    rows = make_rows([{"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.95}])
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    assert labels_only(out) == {("1", 100, "A", "G"): True}


def test_single_negative():
    """max(pip) < pip_neg_threshold → False."""
    rows = make_rows([{"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.005}])
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    assert labels_only(out) == {("1", 100, "A", "G"): False}


def test_single_intermediate_excluded():
    """max(pip) ∈ [pip_neg, pip_pos] → label None → filtered out."""
    rows = make_rows([{"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.5}])
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    assert labels_only(out) == {}


# ---------- multi-row aggregation by max ----------


def test_max_picks_positive_when_one_study_high():
    """[0.5, 0.95] → max = 0.95 → positive. Mid-range PIP in another
    study doesn't disqualify the call (matches TraitGym behaviour)."""
    rows = make_rows(
        [
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.5},
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.95},
        ]
    )
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    assert labels_only(out) == {("1", 100, "A", "G"): True}


def test_multiple_studies_max_excludes_when_intermediate():
    """Regression test for the eqtl row-level pre-filter bug.

    A variant with tissue PIPs ``[0.001, 0.5]``:
      - Correct: ``max = 0.5`` → intermediate → excluded.
      - Buggy pre-filter ``(pip > pos) | (pip < neg)``: 0.5 row dropped,
        only 0.001 survives, ``max = 0.001 < pip_neg`` → mislabeled
        negative.

    This test pins down the correct behaviour: the row with pip=0.5
    must enter the max aggregation and force exclusion.
    """
    rows = make_rows(
        [
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.001},
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.5},
        ]
    )
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    assert labels_only(out) == {}


def test_pos_paired_with_neg_labeled_positive():
    """[0.001, 0.95] → max = 0.95 → positive. QTLs are often
    tissue-specific; a low PIP in one tissue and high in another is a
    legitimate positive."""
    rows = make_rows(
        [
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.001},
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.95},
        ]
    )
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    assert labels_only(out) == {("1", 100, "A", "G"): True}


def test_all_negative_across_studies():
    """All studies have low PIP → max is low → negative."""
    rows = make_rows(
        [
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.001},
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.005},
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.008},
        ]
    )
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    assert labels_only(out) == {("1", 100, "A", "G"): False}


# ---------- coverage: study count doesn't affect the rule ----------


def test_coverage_one_vs_all_studies_label_identically():
    """A variant tested in 1 study and a variant tested in 5 studies
    label identically given the same ``max(pip)``. Negatives are NOT
    required to be tested in all studies."""
    rows = pl.concat(
        [
            make_rows(
                [{"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.005}]
            ),
            make_rows(
                [
                    {"chrom": "2", "pos": 200, "ref": "C", "alt": "T", "pip": 0.005},
                    {"chrom": "2", "pos": 200, "ref": "C", "alt": "T", "pip": 0.001},
                    {"chrom": "2", "pos": 200, "ref": "C", "alt": "T", "pip": 0.002},
                    {"chrom": "2", "pos": 200, "ref": "C", "alt": "T", "pip": 0.003},
                    {"chrom": "2", "pos": 200, "ref": "C", "alt": "T", "pip": 0.004},
                ]
            ),
        ]
    )
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    assert labels_only(out) == {
        ("1", 100, "A", "G"): False,
        ("2", 200, "C", "T"): False,
    }


def test_coverage_intermediate_in_one_of_many_excludes():
    """A variant tested in 5 studies with all-low except one
    intermediate ``[0.001, 0.001, 0.5, 0.001, 0.001]``:
    ``max = 0.5`` → excluded. Even one intermediate-PIP study
    disqualifies the variant from being a clean negative."""
    rows = make_rows(
        [
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.001},
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.001},
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.5},
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.001},
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.001},
        ]
    )
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    assert labels_only(out) == {}


def test_coverage_high_in_one_of_many_labeled_positive():
    """A variant tested in 5 studies with one high PIP and rest low:
    ``max = 0.95`` → positive. Tissue/trait specificity is the norm."""
    rows = make_rows(
        [
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.001},
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.001},
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.95},
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.001},
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.001},
        ]
    )
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    assert labels_only(out) == {("1", 100, "A", "G"): True}


# ---------- null-PIP guard semantics ----------


def test_null_pip_with_guard_blocks_negative():
    """[0.001, NULL] with null guard ON: even though ``max(pip) = 0.001``
    looks like a clean negative, the null PIP in another tested study
    means SuSiE/FINEMAP disagreed there → cannot label as confident
    negative → excluded."""
    rows = make_rows(
        [
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.001},
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": None},
        ]
    )
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=True,
    )
    assert labels_only(out) == {}


def test_null_pip_with_guard_does_not_block_positive():
    """[0.95, NULL] with null guard ON: max = 0.95 → still positive.
    Null guard only affects negatives, not positives — high PIP somewhere
    is enough signal regardless of disagreement elsewhere."""
    rows = make_rows(
        [
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.95},
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": None},
        ]
    )
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=True,
    )
    assert labels_only(out) == {("1", 100, "A", "G"): True}


def test_null_pip_without_guard_allows_negative():
    """[0.001, NULL] with null guard OFF (e.g. eqtl SuSiE-only):
    ``max(pip) = 0.001`` (max ignores nulls) → negative. Used for
    single-method sources where null PIPs aren't expected anyway."""
    rows = make_rows(
        [
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.001},
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": None},
        ]
    )
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    assert labels_only(out) == {("1", 100, "A", "G"): False}


def test_all_null_pip_excluded():
    """All rows have null PIP: ``max(pip) = NULL`` → cascade falls
    through to ``otherwise(None)`` → excluded."""
    rows = make_rows(
        [
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": None},
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": None},
        ]
    )
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    assert labels_only(out) == {}


# ---------- boundary values ----------


def test_pip_exactly_at_pos_threshold_excluded():
    """Boundary: ``pip = pip_pos`` exactly → not strictly > pip_pos
    (positive uses strict ``>``) → not strictly < pip_neg → falls to
    None → excluded."""
    rows = make_rows([{"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": POS}])
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    assert labels_only(out) == {}


def test_pip_exactly_at_neg_threshold_excluded():
    """Boundary: ``pip = pip_neg`` exactly → not strictly < pip_neg
    (negative uses strict ``<``) → not strictly > pip_pos → falls to
    None → excluded."""
    rows = make_rows([{"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": NEG}])
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    assert labels_only(out) == {}


# ---------- extra_aggs ----------


def test_extra_aggs_carried_through():
    """``extra_aggs`` columns appear in the output and aggregate
    per-variant. Common patterns: ``first()`` for carry-along constants,
    ``filter().unique()`` for collecting positive-study labels."""
    rows = make_rows(
        [
            {
                "chrom": "1",
                "pos": 100,
                "ref": "A",
                "alt": "G",
                "pip": 0.95,
                "tissue": "adipose",
                "rsid": "rs1",
            },
            {
                "chrom": "1",
                "pos": 100,
                "ref": "A",
                "alt": "G",
                "pip": 0.85,
                "tissue": "blood",
                "rsid": "rs1",
            },
        ]
    )
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
        extra_aggs=[
            pl.col("rsid").first(),
            pl.col("tissue")
            .filter(pl.col("pip") > POS)
            .unique()
            .sort()
            .alias("pos_tissues"),
        ],
    )
    if isinstance(out, pl.LazyFrame):
        out = out.collect()
    assert out.height == 1
    assert out["rsid"][0] == "rs1"
    # 0.95 > pip_pos → adipose enters pos_tissues; 0.85 ≤ pip_pos → blood doesn't.
    assert out["pos_tissues"][0].to_list() == ["adipose"]


def test_extra_aggs_filter_to_positive_studies_excludes_when_only_intermediate():
    """When all study PIPs are intermediate, the variant is excluded
    by max-then-filter; ``extra_aggs`` results never matter (no row
    survives) but the expression must still evaluate without error."""
    rows = make_rows(
        [
            {
                "chrom": "1",
                "pos": 100,
                "ref": "A",
                "alt": "G",
                "pip": 0.5,
                "tissue": "adipose",
            },
        ]
    )
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
        extra_aggs=[
            pl.col("tissue").filter(pl.col("pip") > POS).unique().alias("pos_tissues"),
        ],
    )
    if isinstance(out, pl.LazyFrame):
        out = out.collect()
    assert out.height == 0


# ---------- LazyFrame compatibility ----------


def test_lazyframe_input_returns_lazyframe():
    """Streaming pipelines pass LazyFrame; output stays lazy so the
    caller can chain ``sink_parquet`` without materializing."""
    rows = make_rows(
        [{"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.95}]
    ).lazy()
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    assert isinstance(out, pl.LazyFrame)
    assert labels_only(out) == {("1", 100, "A", "G"): True}


def test_dataframe_input_returns_dataframe():
    rows = make_rows([{"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.95}])
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    assert isinstance(out, pl.DataFrame)


# ---------- multi-variant + edge cases ----------


def test_multiple_variants_labeled_independently():
    rows = make_rows(
        [
            {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.95},
            {"chrom": "1", "pos": 200, "ref": "C", "alt": "T", "pip": 0.005},
            {"chrom": "1", "pos": 300, "ref": "G", "alt": "A", "pip": 0.5},
        ]
    )
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    assert labels_only(out) == {
        ("1", 100, "A", "G"): True,
        ("1", 200, "C", "T"): False,
        # ("1", 300, "G", "A") excluded — intermediate
    }


def test_empty_input():
    rows = pl.DataFrame(
        [],
        schema={
            "chrom": pl.String,
            "pos": pl.Int64,
            "ref": pl.String,
            "alt": pl.String,
            "pip": pl.Float64,
        },
    )
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=False,
    )
    if isinstance(out, pl.LazyFrame):
        out = out.collect()
    assert out.height == 0


def test_null_guard_does_not_leak_internal_column():
    """The internal ``_any_null_pip`` helper column should not appear in
    the output."""
    rows = make_rows([{"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "pip": 0.95}])
    out = label_variants_by_pip(
        rows,
        pip_pos_threshold=POS,
        pip_neg_threshold=NEG,
        use_null_pip_guard=True,
    )
    if isinstance(out, pl.LazyFrame):
        out = out.collect()
    assert "_any_null_pip" not in out.columns
