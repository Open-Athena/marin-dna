from __future__ import annotations

import importlib

import numpy as np
import polars as pl
import torch

from analyze_replication import SPATIAL_RADIUS, response_views
from full_dictionary_synonymous import (
    CLASS_NAME,
    rank_feature_views,
    synonymous_panel,
)

extract_perturbations = importlib.import_module("extract_perturbations")


def test_synonymous_panel_filters_and_reindexes() -> None:
    frame = pl.DataFrame(
        {
            "perturbation_row": [2, 7, 9],
            "class": [CLASS_NAME, "stop_gained", CLASS_NAME],
            "source_panel_row": [10, 11, 12],
        }
    )
    result = synonymous_panel(frame)
    assert result["perturbation_row"].to_list() == [0, 1]
    assert result["original_perturbation_row"].to_list() == [2, 9]
    assert set(result["class"].unique()) == {CLASS_NAME}


def test_rank_feature_views_finds_target_contrast() -> None:
    panel = pl.DataFrame(
        {
            "perturbation_row": [0, 1, 2, 3],
            "class": [CLASS_NAME] * 4,
            "source_panel_row": [10, 10, 20, 20],
            "expected_consequence": [
                "synonymous_variant",
                "missense_variant",
                "synonymous_variant",
                "missense_variant",
            ],
            "relative_position": [0] * 4,
        }
    )
    positions = 2 * SPATIAL_RADIUS + 1
    fwd = np.zeros((4, positions, 2), dtype=np.float32)
    rc = np.zeros_like(fwd)
    fwd[[0, 2], SPATIAL_RADIUS, 1] = 4.0
    rc[[0, 2], SPATIAL_RADIUS, 1] = 2.0
    scores = rank_feature_views(panel, np.array([5, 9]), response_views(fwd, rc))
    best = scores.sort("abs_effect", descending=True).row(0, named=True)
    assert best["feature_id"] == 9
    assert best["effect"] > 0
    assert best["support_contexts"] == 2


def test_full_identity_feature_selection_uses_exact_sae_encoder() -> None:
    class FakeConfig:
        d_in = 2
        d_sae = 3
        normalize_activations = "none"

        @staticmethod
        def architecture() -> str:
            return "jumprelu"

    class FakeSae:
        cfg = FakeConfig()
        training = False
        hook_z_reshaping_mode = False

        def __init__(self) -> None:
            self.encode_calls = 0

        def encode(self, raw: torch.Tensor) -> torch.Tensor:
            self.encode_calls += 1
            return torch.ones((*raw.shape[:-1], self.cfg.d_sae))

    sae = FakeSae()
    raw = torch.zeros((2, 4, sae.cfg.d_in))
    actual = extract_perturbations.encode_selected_features(
        sae, raw, torch.arange(sae.cfg.d_sae)
    )
    assert sae.encode_calls == 1
    torch.testing.assert_close(actual, torch.ones((2, 4, sae.cfg.d_sae)))
