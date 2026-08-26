"""Build bounded cross-genome candidate graphs from repeat-capped DNA seeds."""

from __future__ import annotations

import csv
import heapq
import itertools
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from marin_dna_linclust_conservation.background_scaling import _iter_fasta_records

_BASE_BITS = {"A": 0, "C": 1, "G": 2, "T": 3}
_MASK_64 = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class SeedGraphConfiguration:
    """One deterministic bounded seed-graph configuration."""

    kmer_length: int
    selected_seeds_per_sequence: int
    max_seed_frequency: int
    min_shared_seeds: int
    hash_seed: int

    def __post_init__(self) -> None:
        assert 1 <= self.kmer_length <= 31
        assert self.selected_seeds_per_sequence > 0
        assert self.max_seed_frequency >= 2
        assert self.min_shared_seeds > 0
        assert 0 <= self.hash_seed <= _MASK_64


class _DisjointSet:
    def __init__(self, source_masks: list[int]) -> None:
        self.parent = list(range(len(source_masks)))
        self.size = [1] * len(source_masks)
        self.source_mask = source_masks.copy()

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union_if_sources_disjoint(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.source_mask[left_root] & self.source_mask[right_root]:
            return False
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]
        self.source_mask[left_root] |= self.source_mask[right_root]
        return True


def source_label(identifier: str) -> str:
    """Return the source-genome label encoded by a pipeline record identifier."""
    if identifier.startswith("anchor") and "__" in identifier:
        label = identifier.rsplit("__", maxsplit=1)[1]
    else:
        label = identifier.split("|", maxsplit=1)[0]
    assert label
    return label


def _mix_64(value: int) -> int:
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & _MASK_64
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & _MASK_64
    return value ^ (value >> 31)


def _canonical_kmer_codes(sequence: str, kmer_length: int) -> Iterable[int]:
    sequence = sequence.upper()
    assert set(sequence) <= _BASE_BITS.keys()
    mask = (1 << (2 * kmer_length)) - 1
    forward = 0
    reverse = 0
    for index, base in enumerate(sequence):
        bits = _BASE_BITS[base]
        forward = ((forward << 2) | bits) & mask
        reverse = (reverse >> 2) | ((3 - bits) << (2 * (kmer_length - 1)))
        if index + 1 >= kmer_length:
            yield min(forward, reverse)


def selected_sequence_seeds(
    sequence: str,
    *,
    kmer_length: int,
    selected_count: int,
    hash_seed: int,
) -> tuple[int, ...]:
    """Select the bottom mixed canonical seeds from one sequence."""
    unique = {
        _mix_64(code ^ hash_seed)
        for code in _canonical_kmer_codes(sequence, kmer_length)
    }
    return tuple(sorted(heapq.nsmallest(selected_count, unique)))


def _truth_pair_keys(
    truth_path: str | Path,
    identifier_indices: dict[str, int],
) -> set[int]:
    with Path(truth_path).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    by_anchor: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_anchor[row["query_name"]].append(identifier_indices[row["record_id"]])
    return {
        (left << 32) | right
        for indices in by_anchor.values()
        for left, right in itertools.combinations(sorted(indices), 2)
    }


def build_seed_graph(
    *,
    fasta_path: str | Path,
    truth_path: str | Path,
    assignments_path: str | Path,
    configuration: SeedGraphConfiguration,
    source_aliases: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build a repeat-capped cross-source graph and write its partition."""
    identifiers: list[str] = []
    identifier_indices: dict[str, int] = {}
    sources: list[str] = []
    source_indices: dict[str, int] = {}
    buckets: dict[int, list[int] | None] = {}
    selected_posting_count = 0
    suppressed_seed_count = 0

    aliases = source_aliases or {}
    with Path(fasta_path).open("rb") as handle:
        for identifier, sequence in _iter_fasta_records(handle):
            assert identifier not in identifier_indices
            sequence_index = len(identifiers)
            identifiers.append(identifier)
            identifier_indices[identifier] = sequence_index
            raw_source = source_label(identifier)
            source = aliases.get(raw_source, raw_source)
            sources.append(source)
            source_indices.setdefault(source, len(source_indices))
            seeds = selected_sequence_seeds(
                sequence,
                kmer_length=configuration.kmer_length,
                selected_count=configuration.selected_seeds_per_sequence,
                hash_seed=configuration.hash_seed,
            )
            selected_posting_count += len(seeds)
            for seed in seeds:
                if seed not in buckets:
                    buckets[seed] = [sequence_index]
                    continue
                bucket = buckets[seed]
                if bucket is None:
                    continue
                bucket.append(sequence_index)
                if len(bucket) > configuration.max_seed_frequency:
                    buckets[seed] = None
                    suppressed_seed_count += 1

    assert identifiers
    assert len(identifiers) < (1 << 32)
    edge_counts: dict[int, int] = {}
    retained_seed_count = 0
    cross_source_pair_emission_count = 0
    for bucket in buckets.values():
        if bucket is None or len(bucket) < 2:
            continue
        source_count = len({sources[index] for index in bucket})
        if source_count < 2:
            continue
        retained_seed_count += 1
        for left, right in itertools.combinations(bucket, 2):
            if sources[left] == sources[right]:
                continue
            if left > right:
                left, right = right, left
            pair_key = (left << 32) | right
            edge_counts[pair_key] = edge_counts.get(pair_key, 0) + 1
            cross_source_pair_emission_count += 1

    truth_pairs = _truth_pair_keys(truth_path, identifier_indices)
    qualifying_edges_by_support: dict[int, list[int]] = defaultdict(list)
    candidate_true_pair_count = 0
    for pair_key, support in edge_counts.items():
        if support < configuration.min_shared_seeds:
            continue
        qualifying_edges_by_support[support].append(pair_key)
        if pair_key in truth_pairs:
            candidate_true_pair_count += 1

    source_masks = [1 << source_indices[source] for source in sources]
    components = _DisjointSet(source_masks)
    accepted_edge_count = 0
    source_conflict_edge_count = 0
    for support in sorted(qualifying_edges_by_support, reverse=True):
        for pair_key in sorted(qualifying_edges_by_support[support]):
            left = pair_key >> 32
            right = pair_key & ((1 << 32) - 1)
            if components.union_if_sources_disjoint(left, right):
                accepted_edge_count += 1
            else:
                source_conflict_edge_count += 1

    representatives: dict[int, str] = {}
    component_sizes: Counter[int] = Counter()
    for index, identifier in enumerate(identifiers):
        root = components.find(index)
        component_sizes[root] += 1
        current = representatives.get(root)
        if current is None or identifier < current:
            representatives[root] = identifier
    assert max(component_sizes.values()) <= len(source_indices)

    assignments = Path(assignments_path)
    assignments.parent.mkdir(parents=True, exist_ok=True)
    with assignments.open("w") as handle:
        for index, identifier in enumerate(identifiers):
            handle.write(f"{representatives[components.find(index)]}\t{identifier}\n")

    qualifying_candidate_pair_count = sum(
        len(keys) for keys in qualifying_edges_by_support.values()
    )
    singleton_count = sum(size == 1 for size in component_sizes.values())
    return {
        "accepted_edge_count": accepted_edge_count,
        "candidate_true_pair_count": candidate_true_pair_count,
        "candidate_true_pair_recall": candidate_true_pair_count / len(truth_pairs),
        "cluster_count": len(component_sizes),
        "cross_source_pair_emission_count": cross_source_pair_emission_count,
        "max_cluster_size": max(component_sizes.values()),
        "qualifying_candidate_pair_count": qualifying_candidate_pair_count,
        "retained_seed_count": retained_seed_count,
        "selected_posting_count": selected_posting_count,
        "sequence_count": len(identifiers),
        "singleton_cluster_count": singleton_count,
        "singleton_sequence_fraction": singleton_count / len(identifiers),
        "source_conflict_edge_count": source_conflict_edge_count,
        "source_count": len(source_indices),
        "source_record_counts": dict(sorted(Counter(sources).items())),
        "suppressed_seed_count": suppressed_seed_count,
        "truth_pair_count": len(truth_pairs),
        "unique_cross_source_candidate_pair_count": len(edge_counts),
    }
