"""Exact canonical features, independent tiling, and locus-budget scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from kmer_conservation.fixture import stable_hash

INVALID = np.uint64(2**64 - 1)


def truth_loci(source: list[dict], target: list[dict], component: str) -> set[str]:
    """Known physical target loci; used only after unrestricted candidate scoring."""
    groups = {
        r["group"]
        for r in source
        if r["kind"] == "anchor" and r["component"] == component
    }
    return {
        r["component"] for r in target if r["kind"] == "anchor" and r["group"] in groups
    }


def rank_truth(hits: list[dict], truth: set[str]) -> dict[str, int | None]:
    ranks = {h["component"]: i + 1 for i, h in enumerate(hits)}
    assert truth
    return {name: ranks.get(name) for name in sorted(truth)}


def recall(rows: list[dict], budget: int) -> float:
    ranks = [rank for r in rows for rank in r["ranks"].values()]
    return sum(rank is not None and rank <= budget for rank in ranks) / len(ranks)


def canonical_codes(sequence: str, k: int) -> np.ndarray:
    """One lossless 2-bit canonical code per start, sentinel at ambiguous k-mers."""
    if not 1 <= k <= 31:
        raise ValueError("k must be between 1 and 31")
    output = np.full(max(0, len(sequence) - k + 1), INVALID, dtype=np.uint64)
    forward = reverse = valid = 0
    mask = (1 << (2 * k)) - 1
    alphabet = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, base in enumerate(sequence.upper()):
        code = alphabet.get(base)
        if code is None:
            forward = reverse = valid = 0
            continue
        forward = ((forward << 2) | code) & mask
        reverse = (reverse >> 2) | ((3 - code) << (2 * (k - 1)))
        valid += 1
        if valid >= k:
            output[i - k + 1] = min(forward, reverse)
    return output


def window_starts(
    start: int, length: int, width: int, stride: int, phase: int
) -> list[int]:
    """Offsets within a fixed interval; two edge windows ensure identical coverage."""
    if not 0 < width <= length or stride < 1:
        raise ValueError("invalid window geometry")
    first = (phase - start) % stride
    return sorted({0, length - width, *range(first, length - width + 1, stride)})


@dataclass
class Windows:
    features: list[np.ndarray]
    owners: np.ndarray
    starts: np.ndarray
    records: list[dict]
    width: int
    edge_count: int


def make_windows(
    records: list[dict], width: int, k: int, divisor: int, mask: bool = False
) -> Windows:
    features, owners, starts = [], [], []
    stride = max(1, width // divisor)
    seen: set[tuple] = set()
    edges = 0
    for owner, row in enumerate(records):
        seq = row["sequence"]
        if mask:
            seq = "".join("N" if b.islower() else b for b in seq)
        codes = canonical_codes(seq, k)
        phase = stable_hash(f"568:{row['species']}:{row['chrom']}") % stride
        for start in window_starts(row["start"], len(seq), width, stride, phase):
            # Shuffled records share source-coordinate metadata but are distinct sequences.
            identity = row["id"] if row["kind"] == "shuffled_decoy" else row["chrom"]
            key = (identity, row["start"] + start, row["component"])
            if key in seen:
                continue
            seen.add(key)
            values = np.unique(codes[start : start + width - k + 1])
            values = values[values != INVALID]
            features.append(values)
            owners.append(owner)
            starts.append(start)
            edges += (row["start"] + start - phase) % stride != 0
    return Windows(
        features, np.asarray(owners), np.asarray(starts), records, width, edges
    )


def exact_index(windows: Windows) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
    """Sparse binary feature matrix; transpose is the exact inverted posting index."""
    lengths = np.array([len(values) for values in windows.features], dtype=np.int64)
    flat = np.concatenate(windows.features)
    vocab, inverse = np.unique(flat, return_inverse=True)
    matrix = sparse.csr_matrix(
        (np.ones(len(inverse), dtype=np.int32), inverse, np.r_[0, lengths.cumsum()]),
        shape=(len(lengths), len(vocab)),
    )
    # Column storage is the inverted index. Its transpose is CSR and can be reused
    # for all queries without converting the full target matrix on every multiply.
    return matrix.tocsc(), vocab, lengths


def query_matrix(
    features: list[np.ndarray], vocab: np.ndarray
) -> tuple[sparse.csr_matrix, np.ndarray]:
    rows, cols = [], []
    for i, values in enumerate(features):
        positions = np.searchsorted(vocab, values)
        valid = positions < len(vocab)
        positions = positions[valid]
        positions = positions[vocab[positions] == values[valid]]
        rows.extend([i] * len(positions))
        cols.extend(positions)
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(len(features), len(vocab)),
    )
    # Full query lengths include features absent from the target vocabulary.
    return matrix, np.array([len(x) for x in features])


def exact_scores(
    query_features: list[np.ndarray],
    index: sparse.csr_matrix,
    vocab: np.ndarray,
    target_lengths: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    query, query_lengths = query_matrix(query_features, vocab)
    intersection = (query @ index.T).tocoo()
    scores = np.zeros(index.shape[0], dtype=np.float64)
    denom = (
        query_lengths[intersection.row]
        + target_lengths[intersection.col]
        - intersection.data
    )
    np.maximum.at(scores, intersection.col, intersection.data / denom)
    # Every posting expansion contributes one shared-feature occurrence to a window pair.
    return scores, intersection.nnz, int(intersection.data.sum())


def rank_loci(scores: np.ndarray, windows: Windows, limit: int = 100) -> list[dict]:
    """Collapse a target locus before spending a unique-candidate budget."""
    best: dict[str, dict] = {}
    for wi in np.flatnonzero(scores > 0):
        row = windows.records[windows.owners[wi]]
        locus = row["component"]
        value = float(scores[wi])
        if locus not in best or value > best[locus]["score"]:
            best[locus] = {
                "component": locus,
                "id": row["id"],
                "kind": row["kind"],
                "score": value,
                "window": int(wi),
            }
    return sorted(
        best.values(), key=lambda x: (-x["score"], stable_hash(x["component"]))
    )[:limit]
