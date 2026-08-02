from __future__ import annotations

import numpy as np

from extract_common import D_SAE
from sensitivity_common import (
    decoder_set_geometry,
    nearest_dictionary_neighbors,
    normalize_decoders,
    scaled_category_support,
    scaled_uniform_support,
    support_matched_controls,
)


def test_scaled_support_preserves_declared_fraction_with_floors() -> None:
    assert scaled_uniform_support(4_132) == 16
    assert scaled_uniform_support(11_128) == 22
    assert scaled_uniform_support(26_019) == 51
    assert scaled_uniform_support(17_738) == 35
    assert scaled_category_support(32) == 4
    assert scaled_category_support(64) == 8
    assert scaled_category_support(128) == 16


def test_support_matched_controls_are_unique_and_deterministic() -> None:
    associated_ids = np.array([2, 5, 8])
    associated_support = np.array([10, 100, 1_000])
    candidates = np.array([1, 3, 4, 6, 7, 9])
    candidate_support = np.array([9, 11, 90, 110, 900, 1_100])
    first = support_matched_controls(
        associated_ids,
        associated_support,
        candidates,
        candidate_support,
        namespace="test",
    )
    second = support_matched_controls(
        associated_ids,
        associated_support,
        candidates,
        candidate_support,
        namespace="test",
    )
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 3
    assert set(first) <= set(candidates)


def test_decoder_geometry_and_neighbors_detect_redundancy() -> None:
    decoder = np.zeros((D_SAE, 3), dtype=np.float32)
    decoder[:, 2] = 1.0
    decoder[0] = [1, 0, 0]
    decoder[1] = [1, 0, 0]
    decoder[2] = [0, 1, 0]
    decoder[3] = [0, 1, 0.01]
    normalized, norms = normalize_decoders(decoder)
    assert np.all(norms > 0)
    geometry = decoder_set_geometry(normalized, np.array([0, 1, 2, 3]))
    assert geometry["features"] == 4
    assert np.isclose(geometry["effective_rank"], 2.0, atol=0.01)
    assert geometry["pairs_cosine_ge_095"] == 2
    assert geometry["components_cosine_ge_095"] == 2
    neighbors, cosine = nearest_dictionary_neighbors(
        normalized, np.array([0, 2]), chunk_size=1
    )
    assert neighbors.tolist() == [1, 3]
    assert np.all(cosine > 0.99)
