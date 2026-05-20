# Copyright The MarinDNA Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the lm_eval task class ``DnaVepLlrEvalTask``.

Pin down the per-strand + AVG aggregation logic — the load-bearing piece for
#179 parity with the offline batched VEP path. Avoids loading any HF dataset;
the tests construct synthetic doc dicts and call ``process_results`` /
``aggregation`` directly.
"""

import math

import pytest

pytest.importorskip("lm_eval", reason="install with `uv sync --extra marin` to run")

from marin_dna.pipelines.evals.lm_eval.dna_vep_llr_eval import (  # noqa: E402
    METRIC_REGISTRY,
    _PairwiseAccuracyAggregation,
)


def _items_from_per_variant(per_variant_rows):
    """Flatten per-variant rows into the
    (score, target, subset, variant_id, match_group, strand) tuples that
    ``DnaVepLlrEvalTask.aggregation()`` consumes.

    ``per_variant_rows`` is a list of
        (variant_id, target, subset, match_group, strand_to_score)
    where ``strand_to_score`` is a dict ``{"+": s}`` or ``{"+": s, "-": s}``.
    """
    items = []
    for variant_id, target, subset, match_group, strand_to_score in per_variant_rows:
        for strand, s in strand_to_score.items():
            items.append((s, target, subset, variant_id, match_group, strand))
    return items


def _run_aggregation(items, metric_name="pairwise_accuracy", task_name=None):
    store: dict[str, float] = {}
    agg = _PairwiseAccuracyAggregation(
        results_store=store, metric_name=metric_name, task_name=task_name
    )
    scalar = agg(items)
    return scalar, store


def test_metric_registry_has_pairwise_accuracy():
    assert "pairwise_accuracy" in METRIC_REGISTRY
    assert METRIC_REGISTRY["pairwise_accuracy"]["higher_is_better"] is True


def test_one_strand_per_variant_emits_only_avg():
    """1-strand dataset: only the avg keys are emitted (avg == single-strand score).

    No fwd/rc keys because they would be redundant with avg. 4 perfectly-separable
    matched pairs (in 30+ pairs to satisfy macro_avg n_min=30) => global PA = 1.0.
    """
    per_variant: list = []
    for i in range(30):
        per_variant.append((("1", 100 + i, "G", "A"), 1, "missense", i + 1, {"+": 0.9}))
        per_variant.append((("1", 200 + i, "T", "A"), 0, "missense", i + 1, {"+": 0.1}))
    items = _items_from_per_variant(per_variant)
    scalar, store = _run_aggregation(items)

    assert scalar == pytest.approx(1.0)
    assert store["_global_/avg/pairwise_accuracy"] == pytest.approx(1.0)
    assert store["missense/avg/pairwise_accuracy"] == pytest.approx(1.0)
    assert store["_macro_avg_/avg/pairwise_accuracy"] == pytest.approx(1.0)
    # No fwd/rc keys for 1-strand datasets.
    fwd_keys = [k for k in store if "/fwd/" in k]
    rc_keys = [k for k in store if "/rc/" in k]
    assert fwd_keys == [], f"unexpected fwd keys for 1-strand dataset: {fwd_keys}"
    assert rc_keys == [], f"unexpected rc keys for 1-strand dataset: {rc_keys}"


def test_two_strands_per_variant_emits_fwd_rc_avg_separately():
    """2-strand dataset: fwd / rc / avg PA all stored, avg returned as scalar.

    Set up so:
      - FWD scores would give PA = 0.5 (tied within each pair)
      - RC scores would give PA = 1.0 (perfectly separable)
      - AVG breaks the FWD tie in favor of the positive => AVG PA = 1.0
    """
    per_variant: list = []
    for i in range(30):
        per_variant.append(
            (("1", 100 + i, "G", "A"), 1, "missense", i + 1, {"+": 0.5, "-": 0.9})
        )
        per_variant.append(
            (("1", 200 + i, "T", "A"), 0, "missense", i + 1, {"+": 0.5, "-": 0.1})
        )
    items = _items_from_per_variant(per_variant)
    scalar, store = _run_aggregation(items)

    # AVG breaks FWD ties => 1.0 globally
    assert scalar == pytest.approx(1.0)
    assert store["_global_/avg/pairwise_accuracy"] == pytest.approx(1.0)
    # FWD: every pair tied => PA = 0.5
    assert store["_global_/fwd/pairwise_accuracy"] == pytest.approx(0.5)
    assert store["missense/fwd/pairwise_accuracy"] == pytest.approx(0.5)
    # RC: perfectly separated => PA = 1.0
    assert store["_global_/rc/pairwise_accuracy"] == pytest.approx(1.0)
    assert store["missense/rc/pairwise_accuracy"] == pytest.approx(1.0)
    # Macro-avg present for all three.
    assert "_macro_avg_/fwd/pairwise_accuracy" in store
    assert "_macro_avg_/rc/pairwise_accuracy" in store
    assert "_macro_avg_/avg/pairwise_accuracy" in store


def test_two_strands_se_present_for_each():
    """Each (subset, strand_tag) emits a paired ``..._se`` key."""
    per_variant: list = []
    for i in range(30):
        per_variant.append(
            (("1", 100 + i, "G", "A"), 1, "missense", i + 1, {"+": 0.9, "-": 0.8})
        )
        per_variant.append(
            (("1", 200 + i, "T", "A"), 0, "missense", i + 1, {"+": 0.1, "-": 0.2})
        )
    items = _items_from_per_variant(per_variant)
    _, store = _run_aggregation(items)
    for tag in ("fwd", "rc", "avg"):
        assert f"_global_/{tag}/pairwise_accuracy" in store
        assert f"_global_/{tag}/pairwise_accuracy_se" in store
        assert math.isfinite(store[f"_global_/{tag}/pairwise_accuracy_se"])


def test_inconsistent_target_within_variant_fails_loud():
    """Two rows with the same variant_id but different targets must fail."""
    items = [
        (0.5, 1, "missense", ("1", 11, "G", "A"), 1, "+"),
        (0.6, 0, "missense", ("1", 11, "G", "A"), 1, "-"),  # contradicting target
    ]
    with pytest.raises(AssertionError, match="inconsistent meta"):
        _run_aggregation(items)


def test_inconsistent_subset_within_variant_fails_loud():
    items = [
        (0.5, 1, "missense", ("1", 11, "G", "A"), 1, "+"),
        (0.6, 1, "splicing", ("1", 11, "G", "A"), 1, "-"),
    ]
    with pytest.raises(AssertionError, match="inconsistent meta"):
        _run_aggregation(items)


def test_inconsistent_match_group_within_variant_fails_loud():
    items = [
        (0.5, 1, "missense", ("1", 11, "G", "A"), 1, "+"),
        (0.6, 1, "missense", ("1", 11, "G", "A"), 99, "-"),
    ]
    with pytest.raises(AssertionError, match="inconsistent meta"):
        _run_aggregation(items)


def test_duplicate_strand_within_variant_fails_loud():
    """Two rows with the same variant_id AND same strand must fail."""
    items = [
        (0.5, 1, "missense", ("1", 11, "G", "A"), 1, "+"),
        (0.6, 1, "missense", ("1", 11, "G", "A"), 1, "+"),  # duplicate strand
    ]
    with pytest.raises(AssertionError, match="duplicate strand"):
        _run_aggregation(items)


def test_unknown_strand_fails_loud():
    items = [(0.5, 1, "missense", ("1", 11, "G", "A"), 1, "?")]
    with pytest.raises(AssertionError, match="unknown strand"):
        _run_aggregation(items)


def test_heterogeneous_strand_sets_fails_loud():
    """One variant has both strands, another only one — must fail.

    A mixed dataset would silently make ``score_avg`` mean different things
    across rows.
    """
    items = [
        # variant A has both strands
        (0.5, 1, "missense", ("1", 11, "G", "A"), 1, "+"),
        (0.7, 1, "missense", ("1", 11, "G", "A"), 1, "-"),
        # variant B has only "+"
        (0.3, 0, "missense", ("1", 12, "T", "A"), 1, "+"),
    ]
    with pytest.raises(AssertionError, match="has strands="):
        _run_aggregation(items)


def test_per_subset_rows_with_n_below_30_are_dropped():
    """Per-subset rows with fewer than 30 matched pairs are NOT stored
    (leaderboard convention). ``_global_`` and ``_macro_avg_`` are always stored.
    """
    per_variant: list = []
    # 30 pairs in "missense" — qualifying.
    for i in range(30):
        per_variant.append((("1", 100 + i, "G", "A"), 1, "missense", i + 1, {"+": 0.9}))
        per_variant.append((("1", 200 + i, "T", "A"), 0, "missense", i + 1, {"+": 0.1}))
    # 5 pairs in "splicing" — below threshold.
    for i in range(5):
        per_variant.append(
            (("2", 100 + i, "G", "A"), 1, "splicing", 100 + i, {"+": 0.9})
        )
        per_variant.append(
            (("2", 200 + i, "T", "A"), 0, "splicing", 100 + i, {"+": 0.1})
        )
    items = _items_from_per_variant(per_variant)
    _, store = _run_aggregation(items)

    # Qualifying subset is present.
    assert "missense/avg/pairwise_accuracy" in store
    # Below-threshold subset is dropped.
    assert "splicing/avg/pairwise_accuracy" not in store
    assert "splicing/avg/pairwise_accuracy_se" not in store
    # Aggregate rows always emitted.
    assert "_global_/avg/pairwise_accuracy" in store
    assert "_macro_avg_/avg/pairwise_accuracy" in store


def test_se_is_finite_and_consistent_with_wald():
    """Standard error matches the Wald binomial form ``sqrt(p*(1-p)/n)``.

    Uses 30 1-strand pairs (n_min=30 macro-avg gate) with 20 wins, 10 losses
    => global AVG PA = 20/30.
    """
    per_variant: list = []
    for i in range(20):
        per_variant.append((("1", 100 + i, "G", "A"), 1, "missense", i + 1, {"+": 0.9}))
        per_variant.append((("1", 200 + i, "T", "A"), 0, "missense", i + 1, {"+": 0.1}))
    for i in range(20, 30):
        per_variant.append((("1", 100 + i, "G", "A"), 1, "missense", i + 1, {"+": 0.1}))
        per_variant.append((("1", 200 + i, "T", "A"), 0, "missense", i + 1, {"+": 0.9}))
    items = _items_from_per_variant(per_variant)
    scalar, store = _run_aggregation(items)
    assert scalar == pytest.approx(20 / 30)
    n_pairs = 30
    expected_se = math.sqrt((20 / 30) * (1 - 20 / 30) / n_pairs)
    assert store["_global_/avg/pairwise_accuracy_se"] == pytest.approx(expected_se)


def test_aggregation_pushes_per_subset_to_levanter_tracker(monkeypatch):
    """All ``results_store`` entries are forwarded to ``levanter.tracker.log``
    under the ``lm_eval/<task_name>/<key>`` prefix so per-subset/per-strand cells
    actually surface in wandb (lm-eval itself only logs the scalar return value).

    Logged (not summary-only) so the wandb backend writes both run history —
    making the cells available as workspace charts — and auto-fills the
    summary panel from the latest logged value.
    """
    import levanter.tracker

    pushed: list[tuple[dict, int | None]] = []

    def fake_log(payload, *, step=None, commit=None):
        pushed.append((dict(payload), step))

    monkeypatch.setattr(levanter.tracker, "log", fake_log)

    per_variant: list = []
    for i in range(30):
        per_variant.append(
            (("1", 100 + i, "G", "A"), 1, "missense", i + 1, {"+": 0.9, "-": 0.8})
        )
        per_variant.append(
            (("1", 200 + i, "T", "A"), 0, "missense", i + 1, {"+": 0.1, "-": 0.2})
        )
    items = _items_from_per_variant(per_variant)
    _run_aggregation(items, task_name="mendelian_traits_255")

    assert len(pushed) == 1
    payload, step = pushed[0]
    # step=None lets the tracker backend fill in its current step — works for
    # both eval-only (step=0) and mid-training (step=current_training_step).
    assert step is None
    assert all(k.startswith("lm_eval/mendelian_traits_255/") for k in payload)
    expected = {
        f"lm_eval/mendelian_traits_255/{sub}/{tag}/pairwise_accuracy"
        for sub in ("_global_", "_macro_avg_", "missense")
        for tag in ("fwd", "rc", "avg")
    }
    assert expected.issubset(set(payload))


def test_aggregation_skips_tracker_push_without_task_name(monkeypatch):
    """When constructed without a task_name (e.g. unit tests), keys are
    prefixed with just ``lm_eval/`` — and tracker errors are swallowed
    so a missing/noop tracker doesn't tank the eval.
    """
    import levanter.tracker

    def raising_log(payload, *, step=None, commit=None):
        raise RuntimeError("no tracker set")

    monkeypatch.setattr(levanter.tracker, "log", raising_log)

    per_variant: list = []
    for i in range(30):
        per_variant.append((("1", 100 + i, "G", "A"), 1, "missense", i + 1, {"+": 0.9}))
        per_variant.append((("1", 200 + i, "T", "A"), 0, "missense", i + 1, {"+": 0.1}))
    items = _items_from_per_variant(per_variant)
    scalar, store = _run_aggregation(
        items
    )  # no task_name; tracker raises; must not crash
    assert scalar == pytest.approx(1.0)
    assert "_global_/avg/pairwise_accuracy" in store
