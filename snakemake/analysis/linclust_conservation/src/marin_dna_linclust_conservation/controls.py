"""Deterministic nucleotide controls for the MMseqs2 release gate."""

from __future__ import annotations

import random
from collections.abc import Mapping

from marin_dna_linclust_conservation.mmseqs import parse_cluster_assignments

COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def _substitute(sequence: str, count: int) -> str:
    assert 0 <= count <= len(sequence)
    replacements = {"A": "C", "C": "G", "G": "T", "T": "A"}
    output = list(sequence)
    positions = [
        round(index * (len(sequence) - 1) / max(count - 1, 1)) for index in range(count)
    ]
    for position in positions:
        output[position] = replacements[output[position]]
    return "".join(output)


def synthetic_sequences(seed: int = 521) -> dict[str, str]:
    """Return controls covering strand, identity, indel, masking, and ordering."""
    rng = random.Random(seed)
    base = "".join(rng.choices("ACGT", k=255))
    insertion = base[:96] + "GATTACA" + base[96:-7]
    deletion = base[:128] + base[143:]
    controls = {
        "base": base,
        "exact_duplicate_a": base,
        "exact_duplicate_b": base,
        "exact_reverse_complement": reverse_complement(base),
        "identity_95": _substitute(base, 13),
        "identity_80": _substitute(base, 51),
        "identity_50": _substitute(base, 128),
        "short_insertion": insertion,
        "short_deletion": deletion,
        "low_complexity": "A" * 255,
        "soft_masked_25pct": base[:64].lower() + base[64:],
    }
    assert len(controls["short_insertion"]) == 255
    assert len(controls["short_deletion"]) == 240
    return controls


def write_fasta(records: Mapping[str, str], path: str) -> None:
    """Write deterministic records in mapping order."""
    with open(path, "w") as handle:
        for name, sequence in records.items():
            assert name and not any(character.isspace() for character in name)
            handle.write(f">{name}\n{sequence}\n")


def check_release_gate(assignments_path: str) -> dict[str, object]:
    """Require exact forward and reverse-complement recovery in one cluster."""
    assignments = parse_cluster_assignments(assignments_path)
    cluster_for = dict(
        zip(assignments["member"].to_list(), assignments["representative"].to_list())
    )
    required = {
        "base",
        "exact_duplicate_a",
        "exact_duplicate_b",
        "exact_reverse_complement",
    }
    missing = required - set(cluster_for)
    assert not missing, f"controls missing from assignments: {sorted(missing)}"
    representatives = {cluster_for[name] for name in required}
    assert len(representatives) == 1, (
        "MMseqs2 release gate failed: exact forward/reverse-complement records "
        f"span clusters {sorted(representatives)}"
    )
    return {
        "release_gate_passed": True,
        "required_records": sorted(required),
        "representative": next(iter(representatives)),
        "cluster_count": assignments["representative"].n_unique(),
        "record_count": assignments.height,
    }


def canonical_partition(assignments_path: str) -> tuple[tuple[str, ...], ...]:
    """Represent a cluster assignment independently of representative names."""
    assignments = parse_cluster_assignments(assignments_path)
    clusters = assignments.group_by("representative").agg("member")
    return tuple(
        sorted(tuple(sorted(members)) for members in clusters["member"].to_list())
    )
