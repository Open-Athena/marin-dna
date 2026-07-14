"""Regression tests for the complete issue-365 Figure 8 layout/statistics."""

from __future__ import annotations

import numpy as np
import pytest

from plots.blog import figure8_loss_vs_traitgym_correlation as figure8
from plots.blog._worlds import WORLDS


def test_figure8_uses_original_eight_model_sizes_and_shared_steps() -> None:
    assert [label for label, _run, _stem in figure8.SCALES] == [
        "46M",
        "76M",
        "128M",
        "255M",
        "476M",
        "1B",
        "2B",
        "4B",
    ]
    assert figure8.COMMON_STEPS == (
        160000,
        170000,
        180000,
        190000,
        200000,
        210000,
        215573,
    )


def test_figure8_variants_follow_new_figure5_order() -> None:
    assert [
        subset for subset, _label in figure8._traits_for(WORLDS["mendelian_llr"])
    ] == [
        "missense_variant",
        "synonymous_variant",
        "splicing",
        "tss_proximal",
        "5_prime_UTR_variant",
        "3_prime_UTR_variant",
    ]
    assert [subset for subset, _label in figure8._traits_for(WORLDS["sge_llr"])] == [
        "missense_variant",
        "splicing",
    ]


def test_figure8_spearman_and_pearson_are_distinct_and_labeled_standardly() -> None:
    x = np.arange(7, dtype=float)
    y = x**2
    assert figure8._correlation(x, y, "spearman") == pytest.approx(1.0)
    assert figure8._correlation(x, y, "pearson") < 1.0
    assert figure8._method_text("spearman") == ("Spearman", "ρ")
    assert figure8._method_text("pearson") == ("Pearson", "r")


def test_figure8_fails_when_a_shared_checkpoint_is_missing(monkeypatch) -> None:
    complete = {
        step: {"missense_variant": float(i)}
        for i, step in enumerate(figure8.COMMON_STEPS)
    }

    def fake_scored(stem, world):
        del world
        if stem.endswith("p46M"):
            return {step: values for step, values in complete.items() if step != 160000}
        return complete

    monkeypatch.setattr(figure8, "_scored", fake_scored)
    with pytest.raises(RuntimeError, match=r"46M: \[160000\]"):
        figure8._complete_scored(WORLDS["mendelian_llr"])
