"""Frozen metadata filters and decoder geometry for issue-435 sensitivities."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from extract_common import D_SAE

MIN_CATEGORY_PAIRS = 32
ORIGINAL_UNIFORM_PAIRS = 32_768
ORIGINAL_UNIFORM_SUPPORT = 64
ORIGINAL_CATEGORY_PAIRS = 128
ORIGINAL_CATEGORY_SUPPORT = 16

EXPECTED_UNIFORM_COUNTS = {
    "composition_strict": 4_132,
    "composition_moderate": 11_128,
    "repeat_interior_32": 26_019,
    "full_repeat_window": 17_738,
    "strict_interior_32": 3_260,
}
EXPECTED_CATEGORY_TARGETS = {
    "overlap_unique": 192,
    "repeat_interior_32": 181,
    "composition_moderate": 152,
}


@dataclass(frozen=True)
class PairSubset:
    name: str
    positive_ids: np.ndarray
    negative_ids: np.ndarray
    minimum_nonzero_support: int

    def __post_init__(self) -> None:
        assert self.positive_ids.dtype == self.negative_ids.dtype == np.int64
        assert self.positive_ids.shape == self.negative_ids.shape
        assert self.positive_ids.size >= 2
        assert self.minimum_nonzero_support > 0


@dataclass(frozen=True)
class CategorySubset:
    sensitivity: str
    hierarchy: str
    target: str
    positive_ids: np.ndarray
    negative_ids: np.ndarray
    minimum_nonzero_support: int

    def __post_init__(self) -> None:
        assert self.hierarchy in {"class", "family", "subfamily"}
        assert self.positive_ids.dtype == self.negative_ids.dtype == np.int64
        assert self.positive_ids.shape == self.negative_ids.shape
        assert self.positive_ids.size >= MIN_CATEGORY_PAIRS
        assert self.minimum_nonzero_support >= 4


def _context_arrays(contexts: pl.DataFrame) -> dict[str, np.ndarray]:
    contexts = contexts.sort("context_id")
    context_ids = contexts["context_id"].to_numpy().astype(np.int64, copy=False)
    assert np.array_equal(context_ids, np.arange(contexts.height))
    columns = (
        "gc_count",
        "cpg_count",
        "shannon_entropy",
        "max_homopolymer",
        "overlap_count",
        "boundary_distance",
        "repeat_fraction",
    )
    result = {column: contexts[column].to_numpy() for column in columns}
    assert all(values.shape == (contexts.height,) for values in result.values())
    return result


def _composition_mask(
    arrays: dict[str, np.ndarray],
    positive_ids: np.ndarray,
    negative_ids: np.ndarray,
    *,
    gc: int,
    cpg: int,
    entropy: float,
    homopolymer: int,
) -> np.ndarray:
    return (
        (
            np.abs(arrays["gc_count"][positive_ids] - arrays["gc_count"][negative_ids])
            <= gc
        )
        & (
            np.abs(
                arrays["cpg_count"][positive_ids] - arrays["cpg_count"][negative_ids]
            )
            <= cpg
        )
        & (
            np.abs(
                arrays["shannon_entropy"][positive_ids]
                - arrays["shannon_entropy"][negative_ids]
            )
            <= entropy
        )
        & (
            np.abs(
                arrays["max_homopolymer"][positive_ids]
                - arrays["max_homopolymer"][negative_ids]
            )
            <= homopolymer
        )
    )


def scaled_uniform_support(pair_count: int) -> int:
    assert 0 < pair_count <= ORIGINAL_UNIFORM_PAIRS
    scaled = math.ceil(
        2 * pair_count * ORIGINAL_UNIFORM_SUPPORT / (2 * ORIGINAL_UNIFORM_PAIRS)
    )
    return max(16, scaled)


def scaled_category_support(pair_count: int) -> int:
    assert MIN_CATEGORY_PAIRS <= pair_count <= ORIGINAL_CATEGORY_PAIRS
    scaled = math.ceil(
        2 * pair_count * ORIGINAL_CATEGORY_SUPPORT / (2 * ORIGINAL_CATEGORY_PAIRS)
    )
    return max(4, scaled)


def uniform_sensitivity_subsets(
    contexts: pl.DataFrame, uniform_pairs: pl.DataFrame
) -> dict[str, PairSubset]:
    assert uniform_pairs.height == ORIGINAL_UNIFORM_PAIRS
    arrays = _context_arrays(contexts)
    positive_ids = uniform_pairs["repeat_context_id"].to_numpy().astype(np.int64)
    negative_ids = uniform_pairs["control_context_id"].to_numpy().astype(np.int64)
    strict = _composition_mask(
        arrays,
        positive_ids,
        negative_ids,
        gc=1,
        cpg=1,
        entropy=0.05,
        homopolymer=2,
    )
    moderate = _composition_mask(
        arrays,
        positive_ids,
        negative_ids,
        gc=2,
        cpg=2,
        entropy=0.10,
        homopolymer=3,
    )
    interior = arrays["boundary_distance"][positive_ids] >= 32
    full_window = arrays["repeat_fraction"][positive_ids] == 1.0
    masks = {
        "composition_strict": strict,
        "composition_moderate": moderate,
        "repeat_interior_32": interior,
        "full_repeat_window": full_window,
        "strict_interior_32": strict & interior,
    }
    assert {
        name: int(mask.sum()) for name, mask in masks.items()
    } == EXPECTED_UNIFORM_COUNTS
    return {
        name: PairSubset(
            name=name,
            positive_ids=positive_ids[mask],
            negative_ids=negative_ids[mask],
            minimum_nonzero_support=scaled_uniform_support(int(mask.sum())),
        )
        for name, mask in masks.items()
    }


def category_sensitivity_subsets(
    contexts: pl.DataFrame, comparisons: pl.DataFrame
) -> dict[str, dict[str, list[CategorySubset]]]:
    assert comparisons.height == 24_576
    arrays = _context_arrays(contexts)
    positive_ids = comparisons["positive_context_id"].to_numpy().astype(np.int64)
    negative_ids = comparisons["negative_context_id"].to_numpy().astype(np.int64)
    masks = {
        "overlap_unique": (
            (arrays["overlap_count"][positive_ids] == 1)
            & (arrays["overlap_count"][negative_ids] == 1)
        ),
        "repeat_interior_32": (
            (arrays["boundary_distance"][positive_ids] >= 32)
            & (arrays["boundary_distance"][negative_ids] >= 32)
        ),
        "composition_moderate": _composition_mask(
            arrays,
            positive_ids,
            negative_ids,
            gc=2,
            cpg=2,
            entropy=0.10,
            homopolymer=3,
        ),
    }
    indexed = comparisons.with_row_index("comparison_row")
    result: dict[str, dict[str, list[CategorySubset]]] = {}
    eligible_counts: dict[str, int] = {}
    for sensitivity, mask in masks.items():
        retained = indexed.filter(pl.Series(mask))
        result[sensitivity] = {level: [] for level in ("class", "family", "subfamily")}
        eligible = 0
        for group in retained.partition_by("level", "label", maintain_order=True):
            if group.height < MIN_CATEGORY_PAIRS:
                continue
            hierarchy = str(group["level"][0])
            target = str(group["label"][0])
            result[sensitivity][hierarchy].append(
                CategorySubset(
                    sensitivity=sensitivity,
                    hierarchy=hierarchy,
                    target=target,
                    positive_ids=group["positive_context_id"]
                    .to_numpy()
                    .astype(np.int64),
                    negative_ids=group["negative_context_id"]
                    .to_numpy()
                    .astype(np.int64),
                    minimum_nonzero_support=scaled_category_support(group.height),
                )
            )
            eligible += 1
        eligible_counts[sensitivity] = eligible
    assert eligible_counts == EXPECTED_CATEGORY_TARGETS
    return result


def stable_hash(namespace: str, value: int) -> int:
    digest = hashlib.sha256(f"{namespace}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def support_matched_controls(
    associated_ids: np.ndarray,
    associated_support: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_support: np.ndarray,
    *,
    namespace: str,
) -> np.ndarray:
    associated_ids = np.asarray(associated_ids, dtype=np.int64)
    associated_support = np.asarray(associated_support, dtype=np.float64)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    candidate_support = np.asarray(candidate_support, dtype=np.float64)
    assert associated_ids.ndim == associated_support.ndim == 1
    assert candidate_ids.ndim == candidate_support.ndim == 1
    assert associated_ids.size == associated_support.size > 1
    assert candidate_ids.size == candidate_support.size >= associated_ids.size
    assert len(np.unique(associated_ids)) == associated_ids.size
    assert len(np.unique(candidate_ids)) == candidate_ids.size
    assert not set(associated_ids) & set(candidate_ids)
    assert np.all(associated_support > 0) and np.all(candidate_support > 0)

    unused = np.ones(candidate_ids.size, dtype=bool)
    selected: list[int] = []
    order = np.lexsort((associated_ids, np.log1p(associated_support)))
    for index in order:
        available = np.flatnonzero(unused)
        distance = np.abs(
            np.log1p(candidate_support[available])
            - math.log1p(float(associated_support[index]))
        )
        best_distance = float(distance.min())
        tied = available[np.isclose(distance, best_distance, rtol=0, atol=1e-12)]
        chosen = min(
            tied, key=lambda item: stable_hash(namespace, int(candidate_ids[item]))
        )
        selected.append(int(candidate_ids[chosen]))
        unused[chosen] = False
    result = np.array(selected, dtype=np.int64)
    assert result.size == associated_ids.size and len(np.unique(result)) == result.size
    return result


def normalize_decoders(decoder: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    decoder = np.asarray(decoder, dtype=np.float32)
    assert decoder.shape[0] == D_SAE and decoder.ndim == 2
    assert np.isfinite(decoder).all()
    norms = np.linalg.norm(decoder, axis=1)
    assert np.all(norms > 0)
    normalized = decoder / norms[:, None]
    return normalized, norms


def _component_count(similarity: np.ndarray, threshold: float) -> int:
    adjacency = csr_matrix(similarity >= threshold)
    return int(connected_components(adjacency, directed=False, return_labels=False))


def decoder_set_geometry(
    normalized_decoder: np.ndarray, feature_ids: np.ndarray
) -> dict[str, float | int]:
    feature_ids = np.asarray(feature_ids, dtype=np.int64)
    assert feature_ids.ndim == 1 and feature_ids.size > 1
    assert len(np.unique(feature_ids)) == feature_ids.size
    vectors = normalized_decoder[feature_ids]
    similarity = vectors @ vectors.T
    assert np.isfinite(similarity).all()
    np.fill_diagonal(similarity, -np.inf)
    nearest = similarity.max(axis=1)
    finite_similarity = similarity[np.isfinite(similarity)]
    singular_values = np.linalg.svd(vectors, compute_uv=False)
    energy = np.square(singular_values.astype(np.float64))
    probabilities = energy / energy.sum()
    probabilities = probabilities[probabilities > 0]
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    stable_rank = float(energy.sum() / energy.max())
    return {
        "features": int(feature_ids.size),
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "median_nearest_within_cosine": float(np.median(nearest)),
        "maximum_within_cosine": float(finite_similarity.max()),
        "pairs_cosine_ge_090": int(np.sum(finite_similarity >= 0.90) // 2),
        "pairs_cosine_ge_095": int(np.sum(finite_similarity >= 0.95) // 2),
        "components_cosine_ge_090": _component_count(similarity, 0.90),
        "components_cosine_ge_095": _component_count(similarity, 0.95),
    }


def nearest_dictionary_neighbors(
    normalized_decoder: np.ndarray,
    feature_ids: np.ndarray,
    *,
    chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    feature_ids = np.asarray(feature_ids, dtype=np.int64)
    assert feature_ids.ndim == 1 and len(np.unique(feature_ids)) == feature_ids.size
    neighbor_ids = np.empty(feature_ids.size, dtype=np.int64)
    similarities = np.empty(feature_ids.size, dtype=np.float32)
    for offset in range(0, feature_ids.size, chunk_size):
        stop = min(offset + chunk_size, feature_ids.size)
        query_ids = feature_ids[offset:stop]
        values = normalized_decoder[query_ids] @ normalized_decoder.T
        values[np.arange(query_ids.size), query_ids] = -np.inf
        local_neighbors = np.argmax(values, axis=1)
        neighbor_ids[offset:stop] = local_neighbors
        similarities[offset:stop] = values[np.arange(query_ids.size), local_neighbors]
    assert np.all(neighbor_ids != feature_ids) and np.isfinite(similarities).all()
    return neighbor_ids, similarities
