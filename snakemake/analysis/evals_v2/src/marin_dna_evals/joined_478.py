"""Versioned validation/repeat/control join for issue #478."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from marin_dna_evals.cds_annotations import (
    annotate_cds_windows,
    load_cds_and_exons,
)

WINDOW_ID = re.compile(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
ARTIFACT_SCHEMA_VERSION = "v1"
PRIMARY_START = 32
PRIMARY_END = 223
KMER_ORDER = 6
KMER_PSEUDOCOUNT = 0.5


def parse_windows(
    sequences: pd.DataFrame,
    *,
    window_size: int,
) -> list[tuple[str, int, int]]:
    """Parse FASTA-style IDs as 0-based half-open genomic intervals."""
    assert {"id", "seq"} <= set(sequences.columns)
    assert sequences["id"].is_unique, "window ids must be unique"
    windows: list[tuple[str, int, int]] = []
    for window_id, seq in sequences[["id", "seq"]].itertuples(index=False):
        match = WINDOW_ID.fullmatch(str(window_id))
        assert match is not None, f"invalid window id {window_id!r}"
        chrom = match.group("chrom")
        start, end = int(match.group("start")), int(match.group("end"))
        assert start >= 0 and end - start == window_size, (
            f"{window_id}: expected a {window_size}-bp 0-based half-open interval"
        )
        assert len(seq) == window_size, f"{window_id}: sequence length mismatch"
        windows.append((chrom, start, end))
    return windows


def _encode_sequence(seq: str) -> np.ndarray:
    return np.fromiter(
        (BASE_TO_INT.get(base, -1) for base in seq.upper()),
        dtype=np.int8,
        count=len(seq),
    )


def _reverse_complement_encoded(encoded: np.ndarray) -> np.ndarray:
    out = encoded[::-1].copy()
    valid = out >= 0
    out[valid] = 3 - out[valid]
    out[~valid] = -1
    return out


def _context_index(context: np.ndarray) -> int:
    index = 0
    for value in context:
        index = index * 4 + int(value)
    return index


def _add_order6_counts(encoded: np.ndarray, counts: np.ndarray) -> None:
    for oriented in (encoded, _reverse_complement_encoded(encoded)):
        for target_pos in range(KMER_ORDER, len(oriented)):
            context = oriented[target_pos - KMER_ORDER : target_pos]
            target = int(oriented[target_pos])
            if target < 0 or (context < 0).any():
                continue
            counts[_context_index(context), target] += 1


def _score_order6(encoded: np.ndarray, counts: np.ndarray) -> np.ndarray:
    out = np.full(len(encoded), np.nan, dtype=np.float32)
    for target_pos in range(KMER_ORDER, len(encoded)):
        context = encoded[target_pos - KMER_ORDER : target_pos]
        target = int(encoded[target_pos])
        if target < 0 or (context < 0).any():
            continue
        row = counts[_context_index(context)]
        probability = (row[target] + KMER_PSEUDOCOUNT) / (
            row.sum() + 4 * KMER_PSEUDOCOUNT
        )
        out[target_pos] = -np.log(probability)
    return out


def leave_one_chrom_7mer_nll(
    sequences: list[str],
    chroms: list[str],
) -> list[np.ndarray]:
    """Strand-symmetric order-6 Markov NLL with chromosome-held-out counts.

    A 7-mer consists of six causal context bases plus the target. Counts include
    both orientations. Every window on the target chromosome is excluded from
    its estimator, preventing exact same-locus overlap from becoming a control.
    """
    assert len(sequences) == len(chroms) and sequences
    shape = (4**KMER_ORDER, 4)
    total = np.zeros(shape, dtype=np.int64)
    by_chrom: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(shape, dtype=np.int64)
    )
    encoded = [_encode_sequence(seq) for seq in sequences]
    for array, chrom in zip(encoded, chroms):
        _add_order6_counts(array, total)
        _add_order6_counts(array, by_chrom[chrom])

    results: list[np.ndarray] = []
    for array, chrom in zip(encoded, chroms):
        held_out = total - by_chrom[chrom]
        fwd = _score_order6(array, held_out)
        rc = _score_order6(_reverse_complement_encoded(array), held_out)[::-1]

        valid_fwd, valid_rc = np.isfinite(fwd), np.isfinite(rc)
        combined = np.full(len(array), np.nan, dtype=np.float32)
        both = valid_fwd & valid_rc
        combined[both] = (fwd[both] + rc[both]) / 2
        combined[valid_fwd & ~valid_rc] = fwd[valid_fwd & ~valid_rc]
        combined[valid_rc & ~valid_fwd] = rc[valid_rc & ~valid_fwd]
        results.append(combined)
    return results


def build_joined_windows(
    sequences: pd.DataFrame,
    *,
    region: str,
    window_size: int,
    repeat_twobit_path: str | Path,
    cds_gtf_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join conservation case, exact RefSeq soft mask, controls, and CDS strata.

    Codon/splice annotation is intentionally accepted only for ``region="cds"``.
    Those columns are secondary diagnostics and are absent from the upstream and
    downstream artifacts.
    """
    assert region in {"cds", "upstream", "downstream"}, region
    assert (region == "cds") == (cds_gtf_path is not None), (
        "the exact RefSeq GTF is required for CDS and forbidden for other regions"
    )
    windows = parse_windows(sequences, window_size=window_size)

    import py2bit

    two_bit = py2bit.open(str(repeat_twobit_path), True)
    repeat_masks: list[np.ndarray] = []
    conserved_masks: list[np.ndarray] = []
    ambiguous_masks: list[np.ndarray] = []
    window_gc: list[float] = []
    try:
        known_chroms = two_bit.chroms()
        for (chrom, start, end), (window_id, case_seq) in zip(
            windows, sequences[["id", "seq"]].itertuples(index=False)
        ):
            assert chrom in known_chroms, f"{chrom} absent from RefSeq 2bit"
            reference = two_bit.sequence(chrom, start, end)
            assert len(reference) == window_size
            assert reference.upper() == str(case_seq).upper(), (
                f"assembly/coordinate mismatch at {window_id}"
            )
            reference_chars = np.asarray(list(reference))
            case_chars = np.asarray(list(str(case_seq)))
            canonical = np.isin(np.char.upper(reference_chars), list("ACGT"))
            repeat_masks.append(np.char.islower(reference_chars) & canonical)
            conserved_masks.append(np.char.isupper(case_chars) & canonical)
            ambiguous_masks.append(~canonical)
            upper = np.char.upper(reference_chars)
            denom = int(np.isin(upper, list("ACGT")).sum())
            window_gc.append(
                float(np.isin(upper, list("GC")).sum() / denom)
                if denom
                else float("nan")
            )
    finally:
        two_bit.close()

    chroms = [chrom for chrom, _, _ in windows]
    kmer_nll = leave_one_chrom_7mer_nll(sequences["seq"].str.upper().tolist(), chroms)
    joined = pd.DataFrame(
        {
            "window_id": sequences["id"].to_numpy(),
            "region": region,
            "chrom": chroms,
            "start": [start for _, start, _ in windows],
            "end": [end for _, _, end in windows],
            "sequence_upper": sequences["seq"].str.upper().to_numpy(),
            "is_conserved": conserved_masks,
            "is_repeat": repeat_masks,
            "is_ambiguous": ambiguous_masks,
            "window_gc": np.asarray(window_gc, dtype=np.float32),
            "kmer7_nll": kmer_nll,
        }
    )

    if region == "cds":
        assert cds_gtf_path is not None
        cds, exons = load_cds_and_exons(str(cds_gtf_path))
        codon, codon_strand, splice, splice_strand = annotate_cds_windows(
            windows, cds, exons
        )
        joined["codon_position"] = codon
        joined["codon_strand"] = codon_strand
        joined["splice_class"] = splice
        joined["splice_strand"] = splice_strand

    ambiguous = int(sum(mask.sum() for mask in ambiguous_masks))
    total = len(joined) * window_size
    primary_per_window = max(
        0, min(window_size, PRIMARY_END) - min(window_size, PRIMARY_START)
    )
    manifest: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "region": region,
        "window_size": window_size,
        "n_windows": len(joined),
        "n_positions": total,
        "n_ambiguous_positions": ambiguous,
        "n_repeat_positions": int(sum(mask.sum() for mask in repeat_masks)),
        "n_conserved_positions": int(sum(mask.sum() for mask in conserved_masks)),
        "primary_span": {
            "start": PRIMARY_START,
            "end_exclusive": PRIMARY_END,
            "n_positions_per_window": primary_per_window,
            "n_edge_positions_excluded": total - len(joined) * primary_per_window,
        },
        "kmer_control": {
            "name": "chromosome-held-out strand-averaged order-6 Markov NLL",
            "pseudocount": KMER_PSEUDOCOUNT,
        },
        "secondary_cds_strata": region == "cds",
        "codebook": {
            "codon_position": "-1 ambiguous, 0 non-CDS, 1/2/3 codon position",
            "splice_class": "-1 ambiguous, 0 other, 1 donor, 2 acceptor",
            "strand": "-1 minus, 0 none, 1 plus, 2 ambiguous",
        }
        if region == "cds"
        else None,
    }
    return joined, manifest
