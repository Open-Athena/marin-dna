"""Tests for the conservation-score helpers (issue #146)."""

import numpy as np
import pandas as pd
import pyBigWig
import pytest

from bolinas.pipelines.evals.conservation import (
    CONSERVATION_TRACKS,
    aggregate_conservation_metrics,
    score_variants_at_positions,
)
from bolinas.pipelines.evals.metrics import GLOBAL_SUBSET, MACRO_AVG_SUBSET


def test_conservation_tracks_keys():
    """The expected tracks are present and URLs look like bigWigs."""
    assert set(CONSERVATION_TRACKS.keys()) == {
        "phyloP_100v",
        "phastCons_100v",
        "phyloP_241m",
        "phyloP_447m",
        "phyloP_470m",
        "phastCons_470m",
        "phastCons_43p",
    }
    for name, url in CONSERVATION_TRACKS.items():
        assert url.startswith("https://"), f"{name}: URL must be https"
        assert url.endswith((".bw", ".bigWig")), (
            f"{name}: URL must point to a bigWig file, got {url}"
        )


def _write_tiny_bigwig(path, entries, chrom_size=1000):
    """Write a minimal bigWig with the given (chrom, start, end, value) entries."""
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader([("chr1", chrom_size)])
    chroms, starts, ends, values = zip(*entries)
    bw.addEntries(list(chroms), list(starts), ends=list(ends), values=list(values))
    bw.close()


def test_score_variants_coordinate_conversion(tmp_path):
    """1-based VCF pos -> 0-based half-open bigWig query.

    bigWig has values at 0-based positions 99 and 199 only; everything
    else should be NaN.
    """
    bw_path = tmp_path / "tiny.bw"
    _write_tiny_bigwig(
        bw_path,
        [
            ("chr1", 99, 100, 1.5),
            ("chr1", 199, 200, 2.5),
        ],
    )

    df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr1"],
            "pos": [100, 200, 300, 500],  # 1-based
        }
    )

    scores = score_variants_at_positions(df, bw_path)

    assert scores.shape == (4,)
    assert scores[0] == pytest.approx(1.5)
    assert scores[1] == pytest.approx(2.5)
    assert np.isnan(scores[2])
    assert np.isnan(scores[3])


def test_score_variants_preserves_nan(tmp_path):
    """NaN must be preserved (not silently filled) when no data is present."""
    bw_path = tmp_path / "tiny.bw"
    _write_tiny_bigwig(bw_path, [("chr1", 0, 1, 0.0)])

    df = pd.DataFrame({"chrom": ["chr1"], "pos": [500]})  # outside the entry
    scores = score_variants_at_positions(df, bw_path)
    assert np.isnan(scores[0])


def test_score_variants_chrom_prefix_handling(tmp_path):
    """Chromosome names without 'chr' prefix should be auto-prefixed."""
    bw_path = tmp_path / "tiny.bw"
    _write_tiny_bigwig(bw_path, [("chr1", 99, 100, 3.14)])

    df = pd.DataFrame({"chrom": ["1"], "pos": [100]})  # no 'chr' prefix
    scores = score_variants_at_positions(df, bw_path)
    assert scores[0] == pytest.approx(3.14)


def test_score_variants_unknown_chromosome(tmp_path):
    """Variants on chromosomes not in the bigWig return NaN, don't crash."""
    bw_path = tmp_path / "tiny.bw"
    _write_tiny_bigwig(bw_path, [("chr1", 99, 100, 1.0)])

    df = pd.DataFrame({"chrom": ["chrM"], "pos": [100]})
    scores = score_variants_at_positions(df, bw_path)
    assert np.isnan(scores[0])


def test_score_variants_rejects_non_integer_pos(tmp_path):
    """Float pos column should fail loudly (silent rounding hides bugs)."""
    bw_path = tmp_path / "tiny.bw"
    _write_tiny_bigwig(bw_path, [("chr1", 99, 100, 1.0)])

    df = pd.DataFrame({"chrom": ["chr1"], "pos": [100.0]})
    with pytest.raises(AssertionError, match="pos must be integer"):
        score_variants_at_positions(df, bw_path)


def _scored_parquet(tmp_path, name, scores, labels, subsets, match_groups):
    """Write a fake scored-variant parquet matching the score_variants output schema."""
    path = tmp_path / f"{name}.parquet"
    n = len(scores)
    df = pd.DataFrame(
        {
            "chrom": ["chr1"] * n,
            "pos": list(range(1, n + 1)),
            "ref": ["A"] * n,
            "alt": ["T"] * n,
            "label": labels,
            "subset": subsets,
            "match_group": match_groups,
            "score": scores,
        }
    )
    df.to_parquet(path, index=False)
    return path


def test_aggregate_conservation_metrics_nan_accounting(tmp_path):
    """NaN counts should be correct per subset; AUPRC computed after
    fillna(0). 4 matched groups across 2 subsets; subset A has 1 NaN."""
    # 8 variants = 4 groups. NaN in pos of group 0 -> after fillna(0) the
    # positive is tied at 0 with the median negative; subset A's AUPRC
    # degrades vs subset B where every positive ranks above its negative.
    scores = [np.nan, 0.1, 0.8, 0.2, 0.9, 0.1, 0.7, 0.3]
    labels = [1, 0, 1, 0, 1, 0, 1, 0]
    subsets = ["A", "A", "A", "A", "B", "B", "B", "B"]
    match_groups = [0, 0, 1, 1, 2, 2, 3, 3]
    p = _scored_parquet(tmp_path, "phyloP_241m", scores, labels, subsets, match_groups)

    # Small n_bootstrap so the test is fast; we assert NaN/schema, not SE
    # values that need many iterations to settle.
    metrics, md = aggregate_conservation_metrics(
        {"phyloP_241m": p}, n_min=1, n_bootstrap=50
    )

    # Schema check (AUPRC schema: n_groups + n_rows replace n_pairs + n_ties).
    expected_cols = {
        "score_type",
        "score_name",
        "subset",
        "value",
        "se",
        "n_groups",
        "n_rows",
        "n_nan",
        "n_total",
    }
    assert expected_cols.issubset(set(metrics.columns))

    # NaN accounting.
    nan_a = metrics[
        (metrics["score_name"] == "phyloP_241m") & (metrics["subset"] == "A")
    ]["n_nan"].iloc[0]
    nan_b = metrics[
        (metrics["score_name"] == "phyloP_241m") & (metrics["subset"] == "B")
    ]["n_nan"].iloc[0]
    assert nan_a == 1
    assert nan_b == 0

    # Total counts.
    tot_a = metrics[
        (metrics["score_name"] == "phyloP_241m") & (metrics["subset"] == "A")
    ]["n_total"].iloc[0]
    assert tot_a == 4

    # Subset A AUPRC: group 0 has NaN→0 vs 0.1 (positive ranks below the
    # negative); group 1 has 0.8 vs 0.2 (positive ranks above). The mixed
    # ranking gives a strictly intermediate AUPRC.
    val_a = metrics[
        (metrics["score_name"] == "phyloP_241m") & (metrics["subset"] == "A")
    ]["value"].iloc[0]
    assert 0.0 < val_a < 1.0, f"expected intermediate AUPRC for subset A, got {val_a}"
    # Subset B: both positives rank above their negatives. AUPRC = 1.0.
    val_b = metrics[
        (metrics["score_name"] == "phyloP_241m") & (metrics["subset"] == "B")
    ]["value"].iloc[0]
    assert val_b == 1.0

    # n_groups equals the number of match_groups per subset (1:1 here).
    g_a = metrics[
        (metrics["score_name"] == "phyloP_241m") & (metrics["subset"] == "A")
    ]["n_groups"].iloc[0]
    assert int(g_a) == 2

    # Markdown contains the AUPRC and NaN-count tables.
    assert "AUPRC" in md
    assert "NaN" in md
    assert "phyloP_241m" in md


def test_aggregate_conservation_metrics_aggregate_row_nan_totals(tmp_path):
    """n_nan and n_total on the aggregate rows. _global_ covers every
    variant; _macro_avg_ covers only the qualifying subsets (n_groups >= n_min).

    Dataset: subset A has 4 groups with 1 NaN; subset B has 1 group with 1 NaN.
    With n_min=2, only A qualifies for the macro row."""
    scores = [np.nan, 0.1, 0.8, 0.2, 0.9, 0.1, 0.7, 0.3, np.nan, 0.5]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    subsets = ["A", "A", "A", "A", "A", "A", "A", "A", "B", "B"]
    match_groups = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    p = _scored_parquet(tmp_path, "phyloP_241m", scores, labels, subsets, match_groups)

    metrics, _ = aggregate_conservation_metrics(
        {"phyloP_241m": p}, n_min=2, n_bootstrap=50
    )

    g = metrics[metrics["subset"] == GLOBAL_SUBSET].iloc[0]
    m = metrics[metrics["subset"] == MACRO_AVG_SUBSET].iloc[0]

    # _global_: every variant counted.
    assert int(g["n_nan"]) == 2  # one NaN in A, one in B
    assert int(g["n_total"]) == 10

    # _macro_avg_ (n_min=2): only A qualifies (4 groups); B (1 group) excluded.
    assert int(m["n_nan"]) == 1  # only A's NaN
    assert int(m["n_total"]) == 8  # only A's 8 variants


def test_aggregate_conservation_metrics_multi_score_column_order(tmp_path):
    """Markdown columns appear in the order of parquet_paths.keys()."""
    scores = [0.7, 0.5, 0.3, 0.2]
    labels = [1, 0, 1, 0]
    subsets = ["x", "x", "x", "x"]
    match_groups = [0, 0, 1, 1]
    paths = {
        "phyloP_100v": _scored_parquet(
            tmp_path, "phyloP_100v", scores, labels, subsets, match_groups
        ),
        "phyloP_241m": _scored_parquet(
            tmp_path, "phyloP_241m", scores, labels, subsets, match_groups
        ),
        "phastCons_43p": _scored_parquet(
            tmp_path, "phastCons_43p", scores, labels, subsets, match_groups
        ),
    }
    _, md = aggregate_conservation_metrics(paths, n_min=1, n_bootstrap=50)

    pos_100v = md.find("phyloP_100v")
    pos_241m = md.find("phyloP_241m")
    pos_43p = md.find("phastCons_43p")
    assert 0 < pos_100v < pos_241m < pos_43p


def test_aggregate_conservation_metrics_no_global_or_mean_row(tmp_path):
    """AUPRC table has only per-subset rows — no 'global' or 'mean'
    aggregate row (per user requirements)."""
    # 8 variants = 4 groups across 2 subsets.
    scores = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]
    labels = [1, 0, 1, 0, 1, 0, 1, 0]
    subsets = ["A", "A", "A", "A", "B", "B", "B", "B"]
    match_groups = [0, 0, 1, 1, 2, 2, 3, 3]
    p = _scored_parquet(tmp_path, "score1", scores, labels, subsets, match_groups)

    _, md = aggregate_conservation_metrics({"score1": p}, n_min=1, n_bootstrap=50)

    # Split markdown into the two tables.
    auprc_section, nan_section = md.split("### NaN counts")
    assert "AUPRC" in auprc_section

    auprc_rows = [
        ln
        for ln in auprc_section.splitlines()
        if ln.startswith("| ") and "---" not in ln
    ]
    # First row is the header; data rows follow.
    data_rows = auprc_rows[1:]
    row_labels = [r.split("|")[1].strip() for r in data_rows]
    assert "global" not in row_labels
    assert "mean" not in row_labels
    assert set(row_labels) == {"A", "B"}


def test_aggregate_conservation_metrics_value_formatting(tmp_path):
    """Cells in the markdown table follow the ``value ± se`` format. With
    every positive ranking above its matched negative, AUPRC = 1.0 on every
    cluster-bootstrap resample, so SE = 0."""
    scores = [0.9, 0.1, 0.8, 0.2]
    labels = [1, 0, 1, 0]
    subsets = ["A", "A", "A", "A"]
    match_groups = [0, 0, 1, 1]
    p = _scored_parquet(tmp_path, "score1", scores, labels, subsets, match_groups)
    _, md = aggregate_conservation_metrics({"score1": p}, n_min=1, n_bootstrap=50)
    assert "1.000 ± 0.000" in md


def test_aggregate_conservation_metrics_handles_1_to_k_groups(tmp_path):
    """Regression guard for the PR #194/#195 mirror: the aggregator must
    accept 1:k matched groups (k=9 in production) without firing the
    1:1-only assertion that ``compute_pairwise_metrics`` enforces."""
    # 2 groups × 10 variants each (1 positive + 9 negatives) in one subset.
    n_groups = 2
    k_neg = 9
    per_group = 1 + k_neg
    scores: list[float] = []
    labels: list[int] = []
    match_groups: list[int] = []
    for g in range(n_groups):
        # Positive scores highest in each group so AUPRC is well-defined.
        scores.append(0.9 + 0.01 * g)
        labels.append(1)
        match_groups.append(g)
        for j in range(k_neg):
            scores.append(0.1 + 0.01 * j)
            labels.append(0)
            match_groups.append(g)
    subsets = ["A"] * (n_groups * per_group)

    p = _scored_parquet(tmp_path, "score1", scores, labels, subsets, match_groups)
    metrics, _ = aggregate_conservation_metrics({"score1": p}, n_min=1, n_bootstrap=50)

    a = metrics[(metrics["score_name"] == "score1") & (metrics["subset"] == "A")].iloc[
        0
    ]
    # n_groups = 2 (match_groups), n_rows = 20 (variants).
    assert int(a["n_groups"]) == n_groups
    assert int(a["n_rows"]) == n_groups * per_group
    # All positives rank above all negatives → AUPRC = 1.0.
    assert a["value"] == pytest.approx(1.0)


def test_score_variants_preserves_input_order(tmp_path):
    """Output is row-aligned with input — important since the pipeline
    concats it back as a column."""
    bw_path = tmp_path / "tiny.bw"
    _write_tiny_bigwig(
        bw_path,
        [
            ("chr1", 9, 10, 10.0),
            ("chr1", 19, 20, 20.0),
            ("chr1", 29, 30, 30.0),
        ],
    )

    # Deliberately unsorted positions
    df = pd.DataFrame({"chrom": ["chr1"] * 3, "pos": [30, 10, 20]})
    scores = score_variants_at_positions(df, bw_path)

    assert scores[0] == pytest.approx(30.0)
    assert scores[1] == pytest.approx(10.0)
    assert scores[2] == pytest.approx(20.0)
