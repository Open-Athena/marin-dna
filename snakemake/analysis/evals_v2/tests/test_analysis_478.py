from __future__ import annotations

import numpy as np
from marin_dna_evals.analysis_478 import (
    Score,
    _controlled_fit,
    make_scores,
    summarize_cds_secondary,
    summarize_primary,
)


def test_make_scores_has_expected_directions() -> None:
    small = np.array([[4.0, 3.0]])
    middle = np.array([[3.0, 3.5]])
    large = np.array([[2.0, 3.25]])
    scores = make_scores(
        {"small": small, "middle": middle, "large": large},
        np.array([[0.2, 0.3]]),
        ["small", "middle", "large"],
    )
    endpoint = next(score for score in scores if score.kind == "endpoint_delta")
    np.testing.assert_array_equal(endpoint.values, [[2.0, -0.25]])
    adjacent = [score for score in scores if score.kind == "adjacent_delta"]
    np.testing.assert_array_equal(adjacent[0].values, [[1.0, -0.5]])
    np.testing.assert_array_equal(adjacent[1].values, [[1.0, 0.25]])


def test_primary_summary_uses_central_span_and_blocks() -> None:
    score = Score(
        "endpoint_delta",
        "small",
        "large",
        np.array([[1.0, 2.0, -1.0, 4.0], [2.0, 3.0, 1.0, 5.0]]),
        True,
    )
    positions = np.tile(np.arange(4), 2)
    conserved = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=bool)
    repeat = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
    rows = summarize_primary(
        [score],
        region="cds",
        conserved=conserved,
        repeat=repeat,
        ambiguous=np.zeros(8, dtype=bool),
        block=np.repeat([10, 20], 4),
        positions=positions,
        primary_start=1,
        primary_end_exclusive=3,
        replicates=50,
        rng=np.random.default_rng(1),
    )
    central = [row for row in rows if row["span"] == "central_32_222"]
    assert sum(row["n_positions"] for row in central) == 4
    target = next(
        row for row in central if row["conserved"] is True and row["repeat"] is False
    )
    assert target["mean"] == 0.0
    assert target["fraction_positive"] == 0.5
    assert target["n_blocks"] == 2


def test_cds_secondary_keeps_codon_and_splice_separate() -> None:
    score = Score(
        "absolute_nll",
        "small",
        "small",
        np.arange(8, dtype=float).reshape(2, 4) + 1,
        False,
    )
    rows = summarize_cds_secondary(
        [score],
        conserved=np.zeros(8, dtype=bool),
        repeat=np.zeros(8, dtype=bool),
        ambiguous=np.zeros(8, dtype=bool),
        block=np.repeat([0, 1], 4),
        positions=np.tile(np.arange(4), 2),
        codon_position=np.array([1, 2, 3, 0, 1, 2, 3, 0]),
        codon_strand=np.ones(8, dtype=np.int8),
        splice_class=np.array([0, 0, 0, 1, 0, 0, 0, 2]),
        splice_strand=np.ones(8, dtype=np.int8),
        primary_start=0,
        primary_end_exclusive=4,
        replicates=20,
        rng=np.random.default_rng(2),
    )
    assert {row["analysis_family"] for row in rows} == {
        "secondary_codon",
        "secondary_splice",
    }
    assert {row["feature"] for row in rows} == {
        "codon_1",
        "codon_2",
        "codon_3",
        "splice_donor_2bp",
        "splice_acceptor_2bp",
    }


def test_controlled_fit_recovers_conservation_repeat_coefficients() -> None:
    rng = np.random.default_rng(4)
    n = 4000
    conserved = rng.integers(0, 2, n).astype(bool)
    repeat = rng.integers(0, 2, n).astype(bool)
    gc = rng.uniform(0.2, 0.8, n)
    kmer = rng.uniform(0.5, 2.0, n)
    positions = rng.integers(32, 223, n)
    blocks = np.repeat(np.arange(20), n // 20)
    y = (
        1.0
        + 0.7 * conserved
        - 0.4 * repeat
        + 0.25 * conserved * repeat
        + 0.2 * (gc - 0.5)
        - 0.1 * (kmer - np.log(4))
        + rng.normal(0, 0.02, n)
    )
    rows = _controlled_fit(
        y,
        conserved=conserved,
        repeat=repeat,
        window_gc=gc,
        kmer_nll=kmer,
        positions=positions,
        blocks=blocks,
        replicates=50,
        rng=np.random.default_rng(5),
    )
    estimates = {row["term"]: row["estimate"] for row in rows}
    np.testing.assert_allclose(estimates["conserved"], 0.7, atol=0.02)

    np.testing.assert_allclose(estimates["repeat"], -0.4, atol=0.02)
    np.testing.assert_allclose(estimates["conserved_x_repeat"], 0.25, atol=0.02)
