"""Evaluate clustering against shared projected-anchor truth."""

from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from pathlib import Path

from marin_dna_linclust_conservation.mmseqs import (
    parse_alignments,
    parse_cluster_assignments,
)


def _read_truth(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows
    assert len({row["record_id"] for row in rows}) == len(rows)
    return rows


def evaluate_homology_clusters(
    *,
    truth_path: str | Path,
    assignments_path: str | Path,
    alignments_path: str | Path,
) -> dict[str, object]:
    """Measure true-anchor recovery, cluster purity, and retained-edge alignments."""
    truth = _read_truth(truth_path)
    by_record = {row["record_id"]: row for row in truth}
    by_anchor: dict[str, set[str]] = defaultdict(set)
    for row in truth:
        by_anchor[row["query_name"]].add(row["record_id"])
    species_counts = {len(records) for records in by_anchor.values()}
    assert len(species_counts) == 1
    species_count = next(iter(species_counts))
    assert species_count >= 2

    assignments = parse_cluster_assignments(assignments_path)
    assert set(assignments["member"].to_list()) == set(by_record)
    clusters: dict[str, set[str]] = defaultdict(set)
    for representative, member in assignments.iter_rows():
        clusters[str(representative)].add(str(member))

    true_pairs = {
        frozenset(pair)
        for members in by_anchor.values()
        for pair in itertools.combinations(sorted(members), 2)
    }
    clustered_pairs = {
        frozenset(pair)
        for members in clusters.values()
        for pair in itertools.combinations(sorted(members), 2)
    }
    recovered_true_pairs = true_pairs & clustered_pairs
    false_pairs = clustered_pairs - true_pairs
    true_pair_counts_by_species_pair: dict[str, int] = defaultdict(int)
    recovered_pair_counts_by_species_pair: dict[str, int] = defaultdict(int)
    for pair in true_pairs:
        left, right = sorted(pair)
        species_pair = "--".join(
            sorted(
                (
                    by_record[left]["source_label"],
                    by_record[right]["source_label"],
                )
            )
        )
        true_pair_counts_by_species_pair[species_pair] += 1
        if pair in recovered_true_pairs:
            recovered_pair_counts_by_species_pair[species_pair] += 1
    true_pair_recall_by_species_pair = {
        species_pair: recovered_pair_counts_by_species_pair[species_pair] / count
        for species_pair, count in sorted(true_pair_counts_by_species_pair.items())
    }
    exact_anchor_clusters = sum(
        1
        for members in clusters.values()
        if len({by_record[member]["query_name"] for member in members}) == 1
        and len(members) == species_count
    )
    impure_clusters = sum(
        1
        for members in clusters.values()
        if len({by_record[member]["query_name"] for member in members}) > 1
    )

    alignments = parse_alignments(alignments_path)
    nonself = alignments.filter(alignments["query"] != alignments["target"])
    true_edge_alignments = nonself.filter(
        [
            by_record[str(query)]["query_name"] == by_record[str(target)]["query_name"]
            for query, target in nonself.select("query", "target").iter_rows()
        ]
    )
    aligned_true_edges = true_edge_alignments.height
    return {
        "aligned_true_representative_member_edges": aligned_true_edges,
        "anchor_count": len(by_anchor),
        "cluster_count": len(clusters),
        "cluster_count_over_ideal": len(clusters) / len(by_anchor),
        "exact_anchor_cluster_count": exact_anchor_clusters,
        "exact_anchor_recovery_fraction": exact_anchor_clusters / len(by_anchor),
        "false_clustered_pair_count": len(false_pairs),
        "impure_cluster_count": impure_clusters,
        "mean_true_edge_identity": (
            float(true_edge_alignments["fident"].mean()) if aligned_true_edges else 0.0
        ),
        "mean_true_edge_query_coverage": (
            float(true_edge_alignments["qcov"].mean()) if aligned_true_edges else 0.0
        ),
        "mean_true_edge_target_coverage": (
            float(true_edge_alignments["tcov"].mean()) if aligned_true_edges else 0.0
        ),
        "pair_precision": (
            len(recovered_true_pairs) / len(clustered_pairs) if clustered_pairs else 1.0
        ),
        "sequence_count": len(truth),
        "species_count": species_count,
        "true_pair_count": len(true_pairs),
        "true_pair_recall": len(recovered_true_pairs) / len(true_pairs),
        "true_pair_recall_by_species_pair": true_pair_recall_by_species_pair,
        "recovered_true_pair_count": len(recovered_true_pairs),
    }
