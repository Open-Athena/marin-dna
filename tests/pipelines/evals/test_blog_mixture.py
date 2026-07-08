"""Tests for the blog mixture-lineage offline data layer (``plots/blog/_mixture.py``).

The Figure-10 composition (root→leaf chain-stitching + per-phase cumulative-token
placement + fork truncation) is the intricate, easy-to-get-plausibly-wrong part of
the blog redo. These tests pin the token math against the vendored lineage's
cumulative-token accounting (the same numbers the Appendix mixture tree draws), so a
regression in the port crashes CI rather than shipping a mis-stitched figure.
"""

from __future__ import annotations

import math

import pytest

from plots.blog import _mixture as mx
from plots.blog import _mixture_lineage as ml

LEAVES = ("exp135-zoonomia-m5.1", "exp135-zoonomia-m1.3", "exp135-zoonomia-m3.3")


def _all_scored(mixv: str, step: int) -> dict[str, float]:
    """Synthetic world reader: every checkpoint 'scored', value monotone in step."""
    return {"missense_variant": 0.3 + step / 5e5, "splicing": 0.25 + step / 6e5}


def test_chain_to_leaf_walks_parents() -> None:
    assert mx.chain_to_leaf("exp135-zoonomia-m5.1") == [
        "uniform",
        "uniform_to_uniform_1",
        "exp135-zoonomia-m5.1",
    ]
    # Pre-cooldown chains are 4 stages deep.
    assert mx.chain_to_leaf("exp135-zoonomia-m1.3") == [
        "exp135-zoonomia-m1",
        "exp135-zoonomia-m1.1",
        "exp135-zoonomia-m1.2",
        "exp135-zoonomia-m1.3",
    ]


@pytest.mark.parametrize("leaf", LEAVES)
def test_composed_endpoint_matches_mixture_tree_total(leaf: str) -> None:
    """The composed trajectory's last token must land on the lineage's
    cumulative-token total — Eric's build-time consistency guard, ported."""
    tokens, values = mx.composed_curve(_all_scored, leaf, ("missense_variant",))
    assert len(tokens) == len(values) > 0
    # Token axis is sorted and strictly within [0, cumulative_total].
    assert list(tokens) == sorted(tokens)
    expected = ml.cumulative_total(leaf, mx.own_tokens())
    # The last HF checkpoint lands one optimizer step short of the configured end
    # (e.g. step-82823 vs num_train_steps 82824), so the composed endpoint sits
    # ≤2 steps below the tree total — exactly the rounding validate_consistency
    # tolerates (its lower bound is a far looser 1.5 eval-gaps).
    assert tokens[-1] == pytest.approx(expected, abs=2 * mx.TOKENS_PER_STEP)
    assert mx.validate_consistency(
        _all_scored, leaf, ("missense_variant",)
    ) == pytest.approx(expected, abs=2 * mx.TOKENS_PER_STEP)


def test_parent_off_path_tail_is_truncated() -> None:
    """uniform's final checkpoint (step-25004 ≈ 52.4B) is its own-run endpoint but
    lies past the m5.1 fork (uniform_to_uniform_1 inherits only 0.8 of uniform's
    tokens ≈ 41.9B), so it must be dropped from the composed m5.1 curve."""
    tokens, _ = mx.composed_curve(
        _all_scored, "exp135-zoonomia-m5.1", ("missense_variant",)
    )
    # uniform contributes only its 10k/20k steps (≈21B/42B); 25004 (≈52.4B) is
    # off-path. The first fork offset ≈ 41.9B, so no composed point sits in
    # (44B, 62B) — the dropped-tail gap before uniform_to_uniform_1's own portion.
    b = tokens / 1e9
    assert not ((b > 45) & (b < 60)).any(), f"off-path tail leaked: {b}"


def test_validate_consistency_skips_when_unscored() -> None:
    """A reader with nothing scored yet (scoring in flight) returns nan, not a crash."""

    def empty(mixv: str, step: int) -> dict[str, float]:
        return {}

    assert math.isnan(
        mx.validate_consistency(empty, "exp135-zoonomia-m5.1", ("missense_variant",))
    )


def test_config_entries_reuse_existing_ids_and_are_unique() -> None:
    entries = list(mx.config_entries())
    names = [e["name"] for e in entries]
    assert len(names) == len(set(names)) == 47
    # The three pre-existing config entries must be regenerated verbatim (else the
    # reused checkpoints get double-scored under a second id).
    for name in (
        "mix-v0.9-p1B-i0-uniform-step-25004",
        "mix-v0.9-p1B-i16-upstream-step-8333",
        "mix-v0.9-p1B-i24-exp135-m5.1-step-59158",
    ):
        assert name in names
    # Every entry is scored on both worlds' datasets, embeddings-on path.
    assert all(e["datasets"] == ["mendelian_traits", "sge"] for e in entries)


def test_final_name_is_highest_step() -> None:
    assert mx.final_name("uniform") == "mix-v0.9-p1B-i0-uniform-step-25004"
    assert (
        mx.final_name("exp135-zoonomia-m1.3")
        == "mix-v0.9-p1B-i30-exp135-zoonomia-m1.3-step-82823"
    )


def test_own_tokens_covers_every_lineage_run() -> None:
    ot = mx.own_tokens()
    assert set(mx.BY_MIX) <= set(ot)
    # Token budgets are positive and the ⅕-mix zoonomia roots are the ~62B 'L' tier.
    assert ot["exp135-zoonomia-m1"] == pytest.approx(62e9, rel=0.05)
    assert ot["uniform"] == pytest.approx(52.4e9, rel=0.05)
