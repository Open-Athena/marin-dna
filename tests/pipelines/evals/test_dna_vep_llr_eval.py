# Copyright The MarinDNA Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the lm_eval task class ``DnaVepLlrEvalTask``.

Pin down the per-strand + AVG aggregation logic — the load-bearing piece for
parity with the offline ``snakemake/analysis/evals_v2/`` batched VEP path
(#225: FWD/RC-avg + AUPRC). Avoids loading any HF dataset; the tests construct
synthetic per-variant rows and call the collapse helper / aggregation directly.
"""

import numpy as np
import pytest

pytest.importorskip("lm_eval", reason="install with `uv sync --extra marin` to run")

from marin_dna.pipelines.evals.lm_eval.dna_vep_llr_eval import (  # noqa: E402
    _MIN_GROUPS_PER_SUBSET,
    LLR_TRANSFORMS,
    METRIC_REGISTRY,
    _AuprcAggregation,
    _collapse_variants,
)
from marin_dna.pipelines.evals.metrics import (  # noqa: E402
    compute_auprc_metrics,
)

_NEGATE = LLR_TRANSFORMS["negate"]
_IDENTITY = LLR_TRANSFORMS["identity"]


def _items_from_per_variant(per_variant_rows):
    """Flatten per-variant rows into the
    (llr, target, subset, variant_id, match_group, strand) tuples that
    ``DnaVepLlrEvalTask.aggregation()`` consumes. The first element is the
    **raw** LLR (the transform is applied inside the aggregation).

    ``per_variant_rows`` is a list of
        (variant_id, target, subset, match_group, strand_to_llr)
    where ``strand_to_llr`` is a dict ``{"+": llr}`` or ``{"+": llr, "-": llr}``.
    """
    items = []
    for variant_id, target, subset, match_group, strand_to_llr in per_variant_rows:
        for strand, llr in strand_to_llr.items():
            items.append((llr, target, subset, variant_id, match_group, strand))
    return items


def _run_aggregation(
    items, *, transform=_IDENTITY, metric_name="auprc", task_name=None
):
    store: dict[str, float] = {}
    agg = _AuprcAggregation(
        results_store=store,
        metric_name=metric_name,
        llr_transform=transform,
        task_name=task_name,
    )
    scalar = agg(items)
    return scalar, store


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_metric_registry_has_auprc():
    assert "auprc" in METRIC_REGISTRY
    assert METRIC_REGISTRY["auprc"]["higher_is_better"] is True
    # PA was retired here in #225 (invalid on the 1:9 matched-pair dataset).
    assert "pairwise_accuracy" not in METRIC_REGISTRY


# --------------------------------------------------------------------------- #
# _collapse_variants — raw-LLR averaging, then transform; column shape
# --------------------------------------------------------------------------- #
def test_collapse_one_strand_emits_only_avg_column():
    items = _items_from_per_variant(
        [
            (("1", 100, "G", "A"), 1, "missense", 1, {"+": -2.0}),
            (("1", 200, "T", "A"), 0, "missense", 1, {"+": 0.5}),
        ]
    )
    df, score_columns = _collapse_variants(items, _IDENTITY)
    assert score_columns == ["score_avg"]
    assert set(df.columns) == {"label", "subset", "match_group", "score_avg"}
    assert len(df) == 2


def test_collapse_two_strands_emits_fwd_rc_avg_columns():
    items = _items_from_per_variant(
        [
            (("1", 100, "G", "A"), 1, "missense", 1, {"+": -2.0, "-": -1.0}),
            (("1", 200, "T", "A"), 0, "missense", 1, {"+": 0.5, "-": 0.1}),
        ]
    )
    df, score_columns = _collapse_variants(items, _IDENTITY)
    assert score_columns == ["score_fwd", "score_rc", "score_avg"]
    assert set(df.columns) == {
        "label",
        "subset",
        "match_group",
        "score_fwd",
        "score_rc",
        "score_avg",
    }


def test_collapse_averages_raw_llr_then_transforms_negate():
    """``score_avg`` = transform(mean raw LLR), not mean(transform(LLR))."""
    items = _items_from_per_variant(
        [(("1", 100, "G", "A"), 1, "missense", 1, {"+": -3.0, "-": -1.0})]
    )
    df, _ = _collapse_variants(items, _NEGATE)
    row = df.iloc[0]
    # mean raw LLR = -2.0; negate => +2.0
    assert row["score_avg"] == pytest.approx(2.0)
    assert row["score_fwd"] == pytest.approx(3.0)  # -(-3)
    assert row["score_rc"] == pytest.approx(1.0)  # -(-1)


def test_collapse_average_before_abs_is_not_average_of_abs():
    """The ordering pin: ``abs`` does NOT commute with averaging. With
    llr_fwd=+2, llr_rc=-2 the raw mean is 0, so ``abs(mean)=0`` — whereas
    ``mean(abs)`` would be 2. We must do raw-mean-then-transform (offline
    ``abs_llr_avg`` semantics)."""
    items = _items_from_per_variant(
        [(("1", 100, "G", "A"), 1, "missense", 1, {"+": 2.0, "-": -2.0})]
    )
    df, _ = _collapse_variants(items, LLR_TRANSFORMS["abs"])
    row = df.iloc[0]
    assert row["score_avg"] == pytest.approx(0.0)  # abs(mean(+2,-2)) = abs(0)
    assert row["score_fwd"] == pytest.approx(2.0)
    assert row["score_rc"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# AUPRC values + exact parity with a direct compute_auprc_metrics call
# --------------------------------------------------------------------------- #
def test_perfectly_separable_global_auprc_is_one():
    """Positives ranked strictly above negatives => global AUPRC = 1.0."""
    per_variant = []
    for i in range(_MIN_GROUPS_PER_SUBSET):
        per_variant.append((("1", 100 + i, "G", "A"), 1, "missense", i + 1, {"+": 5.0}))
        per_variant.append((("1", 200 + i, "T", "A"), 0, "missense", i + 1, {"+": 0.0}))
    items = _items_from_per_variant(per_variant)
    scalar, store = _run_aggregation(items)
    assert scalar == pytest.approx(1.0)
    assert store["_global_/avg/auprc"] == pytest.approx(1.0)
    assert store["missense/avg/auprc"] == pytest.approx(1.0)
    assert store["_macro_avg_/avg/auprc"] == pytest.approx(1.0)


def test_aggregation_matches_direct_compute_auprc_metrics():
    """The aggregation is faithful glue: every emitted (subset, strand) cell's
    point AUPRC equals a direct ``compute_auprc_metrics`` call on the collapsed
    frame. Online is point-only (``n_bootstrap=0``), so no ``_se`` cells are
    emitted."""
    rng = np.random.default_rng(123)
    per_variant = []
    # missense: 40 groups (1:1) — qualifying. Positives get lower raw LLR
    # (pathogenic), so under `negate` (minus_llr) they score higher.
    for i in range(40):
        pos_llr = {"+": float(rng.normal(-2, 1)), "-": float(rng.normal(-2, 1))}
        neg_llr = {"+": float(rng.normal(0, 1)), "-": float(rng.normal(0, 1))}
        per_variant.append((("1", 100 + i, "G", "A"), 1, "missense", i + 1, pos_llr))
        per_variant.append((("1", 200 + i, "T", "A"), 0, "missense", i + 1, neg_llr))
    # tss: 35 groups (1:1) — qualifying, weaker signal.
    for i in range(35):
        pos_llr = {"+": float(rng.normal(-0.5, 1)), "-": float(rng.normal(-0.5, 1))}
        neg_llr = {"+": float(rng.normal(0, 1)), "-": float(rng.normal(0, 1))}
        per_variant.append((("2", 100 + i, "G", "A"), 1, "tss", 1000 + i, pos_llr))
        per_variant.append((("2", 200 + i, "T", "A"), 0, "tss", 1000 + i, neg_llr))
    items = _items_from_per_variant(per_variant)

    df, score_columns = _collapse_variants(items, _NEGATE)
    expected = compute_auprc_metrics(
        dataset=df[["label", "subset", "match_group"]],
        scores=df[score_columns],
        score_columns=score_columns,
        n_bootstrap=0,
        n_min=_MIN_GROUPS_PER_SUBSET,
    )

    _, store = _run_aggregation(items, transform=_NEGATE)

    for row in expected.to_dict("records"):
        tag = row["score_type"].removeprefix("score_")
        key = f"{row['subset']}/{tag}/auprc"
        assert store[key] == pytest.approx(row["value"]), key
        assert f"{key}_se" not in store, f"online is point-only; unexpected {key}_se"
    # Sanity: the strong subset out-scores the weak one and both beat baseline.
    assert store["missense/avg/auprc"] > store["tss/avg/auprc"]
    assert store["_global_/avg/auprc"] > 0.5


def test_point_only_no_se_cells_emitted():
    """Online is point-only (n_bootstrap=0): per-(subset, strand) point AUPRC
    cells are present, but NO ``_se`` cells are emitted (the cluster-bootstrap SE
    lives in the offline evals_v2 parquet)."""
    per_variant = []
    for i in range(_MIN_GROUPS_PER_SUBSET):
        per_variant.append(
            (("1", 100 + i, "G", "A"), 1, "missense", i + 1, {"+": -2.0, "-": -1.5})
        )
        per_variant.append(
            (("1", 200 + i, "T", "A"), 0, "missense", i + 1, {"+": 0.2, "-": 0.4})
        )
    items = _items_from_per_variant(per_variant)
    _, store = _run_aggregation(items, transform=_NEGATE)
    for tag in ("fwd", "rc", "avg"):
        assert f"_global_/{tag}/auprc" in store
    se_keys = [k for k in store if k.endswith("_se")]
    assert se_keys == [], f"online should emit no SE cells, got {se_keys}"


def test_per_subset_rows_below_n_min_groups_are_dropped():
    """Per-subset cells with fewer than ``_MIN_GROUPS_PER_SUBSET`` match_groups
    are NOT stored (leaderboard convention). ``_global_`` / ``_macro_avg_`` are
    always stored."""
    per_variant = []
    for i in range(_MIN_GROUPS_PER_SUBSET):  # qualifying
        per_variant.append(
            (("1", 100 + i, "G", "A"), 1, "missense", i + 1, {"+": -2.0})
        )
        per_variant.append((("1", 200 + i, "T", "A"), 0, "missense", i + 1, {"+": 0.5}))
    for i in range(5):  # below threshold
        per_variant.append(
            (("2", 100 + i, "G", "A"), 1, "splicing", 1000 + i, {"+": -2.0})
        )
        per_variant.append(
            (("2", 200 + i, "T", "A"), 0, "splicing", 1000 + i, {"+": 0.5})
        )
    items = _items_from_per_variant(per_variant)
    _, store = _run_aggregation(items)

    assert "missense/avg/auprc" in store
    assert "splicing/avg/auprc" not in store
    assert "splicing/avg/auprc_se" not in store
    assert "_global_/avg/auprc" in store
    assert "_macro_avg_/avg/auprc" in store


# --------------------------------------------------------------------------- #
# Defensive invariants (raised inside _collapse_variants)
# --------------------------------------------------------------------------- #
def test_inconsistent_target_within_variant_fails_loud():
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
    """One variant has both strands, another only one — must fail (a mixed
    dataset would silently make ``score_avg`` mean different things)."""
    items = [
        (0.5, 1, "missense", ("1", 11, "G", "A"), 1, "+"),
        (0.7, 1, "missense", ("1", 11, "G", "A"), 1, "-"),
        (0.3, 0, "missense", ("1", 12, "T", "A"), 1, "+"),
    ]
    with pytest.raises(AssertionError, match="has strands="):
        _run_aggregation(items)


# --------------------------------------------------------------------------- #
# Tracker push
# --------------------------------------------------------------------------- #
def test_aggregation_pushes_per_subset_to_levanter_tracker(monkeypatch):
    """All ``results_store`` entries are forwarded to ``levanter.tracker.log``
    under the ``lm_eval/<task_name>/<key>`` prefix so per-subset/per-strand cells
    surface in wandb (lm-eval itself only logs the scalar return value)."""
    import levanter.tracker

    pushed: list[tuple[dict, int | None]] = []

    def fake_log(payload, *, step=None, commit=None):
        pushed.append((dict(payload), step))

    monkeypatch.setattr(levanter.tracker, "log", fake_log)

    per_variant = []
    for i in range(_MIN_GROUPS_PER_SUBSET):
        per_variant.append(
            (("1", 100 + i, "G", "A"), 1, "missense", i + 1, {"+": -2.0, "-": -1.5})
        )
        per_variant.append(
            (("1", 200 + i, "T", "A"), 0, "missense", i + 1, {"+": 0.2, "-": 0.4})
        )
    items = _items_from_per_variant(per_variant)
    _run_aggregation(items, transform=_NEGATE, task_name="mendelian_traits_255")

    assert len(pushed) == 1
    payload, step = pushed[0]
    # step=None lets the tracker backend fill in its current step — works for
    # both eval-only (step=0) and mid-training (step=current_training_step).
    assert step is None
    assert all(k.startswith("lm_eval/mendelian_traits_255/") for k in payload)
    expected = {
        f"lm_eval/mendelian_traits_255/{sub}/{tag}/auprc"
        for sub in ("_global_", "_macro_avg_", "missense")
        for tag in ("fwd", "rc", "avg")
    }
    assert expected.issubset(set(payload))


def test_aggregation_skips_tracker_push_without_task_name(monkeypatch):
    """When the tracker raises (missing/noop), the eval must not crash."""
    import levanter.tracker

    def raising_log(payload, *, step=None, commit=None):
        raise RuntimeError("no tracker set")

    monkeypatch.setattr(levanter.tracker, "log", raising_log)

    # Separable under the default (identity) transform => global AUPRC = 1.0.
    per_variant = []
    for i in range(_MIN_GROUPS_PER_SUBSET):
        per_variant.append((("1", 100 + i, "G", "A"), 1, "missense", i + 1, {"+": 5.0}))
        per_variant.append((("1", 200 + i, "T", "A"), 0, "missense", i + 1, {"+": 0.0}))
    items = _items_from_per_variant(per_variant)
    scalar, store = _run_aggregation(items)  # tracker raises; must not crash
    assert scalar == pytest.approx(1.0)
    assert "_global_/avg/auprc" in store
