"""Validated presentation helpers for MarinDNA variant-score bundles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VariantScoreBundleView:
    """Row-aligned score atoms and optional FWD/RC-averaged allele embeddings."""

    scores: pd.DataFrame
    ref_embeddings: np.ndarray | None
    alt_embeddings: np.ndarray | None


def fwd_rc_average_fp32(strand_embeddings: list[np.ndarray]) -> np.ndarray:
    """Average equally shaped strand embedding matrices in fp32."""
    assert strand_embeddings, "need at least one strand's embeddings to average"
    first = np.asarray(strand_embeddings[0], dtype=np.float32)
    assert first.ndim == 2, f"expected embeddings [N,D], got {first.shape}"
    assert np.isfinite(first).all(), "non-finite strand embeddings"
    acc = np.zeros_like(first, dtype=np.float32)
    for embedding in strand_embeddings:
        embedding = np.asarray(embedding, dtype=np.float32)
        assert embedding.shape == first.shape, (
            f"strand embedding shape mismatch: {embedding.shape} vs {first.shape}"
        )
        assert np.isfinite(embedding).all(), "non-finite strand embeddings"
        acc += embedding
    acc /= len(strand_embeddings)
    assert np.isfinite(acc).all(), "non-finite averaged embeddings"
    return acc


def variant_score_bundle_view(
    results: dict[str, np.ndarray],
    *,
    hidden_size: int | None = None,
) -> VariantScoreBundleView:
    """Validate and unpack ``run_variant_score_bundle`` outputs.

    The score frame retains the raw per-strand LLR/JSD atoms. When
    ``hidden_size`` is supplied, both FWD and RC arrays must contain the
    ``[scores, ref embedding, alt embedding]`` layout; each allele embedding is
    averaged across strands in fp32 and returned as an ``[N,D]`` matrix.
    """
    assert results, "empty variant-score bundle"
    assert "fwd" in results, "variant-score bundle is missing the forward strand"
    assert set(results) <= {"fwd", "rc"}, (
        f"unexpected variant-score bundle strands: {sorted(results)}"
    )
    if hidden_size is not None:
        assert hidden_size >= 1
        assert set(results) == {"fwd", "rc"}, (
            "embedding aggregation requires both forward and reverse-complement "
            "bundle arrays"
        )

    expected_width = 2 if hidden_size is None else 2 + 2 * hidden_size
    arrays: dict[str, np.ndarray] = {}
    n_rows: int | None = None
    for strand in ("fwd", "rc"):
        if strand not in results:
            continue
        array = np.asarray(results[strand])
        assert array.ndim == 2, (
            f"{strand} bundle must be [N,W], got shape {array.shape}"
        )
        assert array.shape[1] == expected_width, (
            f"{strand} bundle width {array.shape[1]} != expected {expected_width}"
        )
        if n_rows is None:
            n_rows = array.shape[0]
        assert array.shape[0] == n_rows, (
            f"{strand} row count {array.shape[0]} != expected {n_rows}"
        )
        assert np.isfinite(array).all(), f"{strand} bundle contains non-finite values"
        arrays[strand] = array
    assert n_rows is not None and n_rows > 0, "variant-score bundle has no rows"

    score_columns: dict[str, np.ndarray] = {}
    for strand in ("fwd", "rc"):
        if strand not in arrays:
            continue
        score_columns[f"llr_{strand}"] = arrays[strand][:, 0]
        score_columns[f"jsd_{strand}"] = arrays[strand][:, 1]
    scores = pd.DataFrame(score_columns)
    assert len(scores) == n_rows

    if hidden_size is None:
        return VariantScoreBundleView(
            scores=scores,
            ref_embeddings=None,
            alt_embeddings=None,
        )

    ref_embeddings = fwd_rc_average_fp32(
        [arrays[strand][:, 2 : 2 + hidden_size] for strand in ("fwd", "rc")]
    )
    alt_embeddings = fwd_rc_average_fp32(
        [
            arrays[strand][:, 2 + hidden_size : 2 + 2 * hidden_size]
            for strand in ("fwd", "rc")
        ]
    )
    assert ref_embeddings.shape == (n_rows, hidden_size)
    assert alt_embeddings.shape == (n_rows, hidden_size)
    return VariantScoreBundleView(
        scores=scores,
        ref_embeddings=ref_embeddings,
        alt_embeddings=alt_embeddings,
    )
