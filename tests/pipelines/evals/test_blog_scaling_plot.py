"""Regression tests for the blog scaling figures' variant→loss mapping."""

from __future__ import annotations

import pandas as pd
import pytest

from plots.blog._scaling import add_relevant_region_loss


def test_add_relevant_region_loss_uses_variant_matched_column() -> None:
    data = pd.DataFrame(
        {
            "subset": [
                "missense_variant",
                "splicing",
                "tss_proximal",
                "5_prime_UTR_variant",
                "3_prime_UTR_variant",
            ]
        }
    )
    metadata = pd.Series(
        {
            "eval_loss": 9.9,
            "eval_loss_cds": 0.3,
            "eval_loss_upstream": 0.8,
            "eval_loss_downstream": 1.2,
        }
    )

    out = add_relevant_region_loss(data, metadata)

    assert out["loss_region"].tolist() == [
        "cds",
        "cds",
        "upstream",
        "upstream",
        "downstream",
    ]
    assert out["eval_loss"].tolist() == [0.3, 0.3, 0.8, 0.8, 1.2]
    assert (out["eval_loss"] != metadata["eval_loss"]).all()


def test_add_relevant_region_loss_rejects_unmapped_variant() -> None:
    data = pd.DataFrame({"subset": ["not_a_mapped_variant"]})
    metadata = pd.Series(
        {
            "eval_loss_cds": 0.3,
            "eval_loss_upstream": 0.8,
            "eval_loss_downstream": 1.2,
        }
    )

    with pytest.raises(AssertionError, match="missing variant→region mapping"):
        add_relevant_region_loss(data, metadata)
