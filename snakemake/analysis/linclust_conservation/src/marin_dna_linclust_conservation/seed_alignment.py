"""Prepare and evaluate bounded alignment diagnostics for seed-graph clusters."""

from __future__ import annotations

import csv
import itertools
from collections import Counter, defaultdict
from pathlib import Path

from marin_dna_linclust_conservation.background_scaling import _iter_fasta_records


def _truth_anchors(truth_path: str | Path) -> dict[str, str]:
    with Path(truth_path).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    anchors = {row["record_id"]: row["query_name"] for row in rows}
    assert len(anchors) == len(rows)
    return anchors


def prepare_seed_alignment_subset(
    *,
    fasta_path: str | Path,
    truth_path: str | Path,
    assignments_path: str | Path,
    subset_fasta_path: str | Path,
    pairs_path: str | Path,
) -> dict[str, object]:
    """Extract complete seed-graph components that contain projected truth records."""
    truth_anchors = _truth_anchors(truth_path)
    truth_clusters: set[str] = set()
    truth_records_seen: set[str] = set()
    with Path(assignments_path).open() as handle:
        for line in handle:
            representative, member = line.rstrip("\n").split("\t")
            if member in truth_anchors:
                truth_clusters.add(representative)
                truth_records_seen.add(member)
    assert truth_records_seen == set(truth_anchors)

    members_by_cluster: dict[str, list[str]] = defaultdict(list)
    member_clusters: dict[str, str] = {}
    with Path(assignments_path).open() as handle:
        for line in handle:
            representative, member = line.rstrip("\n").split("\t")
            if representative not in truth_clusters:
                continue
            assert member not in member_clusters
            member_clusters[member] = representative
            members_by_cluster[representative].append(member)

    pair_counts: Counter[str] = Counter()
    pairs = Path(pairs_path)
    pairs.parent.mkdir(parents=True, exist_ok=True)
    with pairs.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["left", "right", "pair_class", "cluster"])
        for representative in sorted(members_by_cluster):
            members = sorted(members_by_cluster[representative])
            for left, right in itertools.combinations(members, 2):
                left_anchor = truth_anchors.get(left)
                right_anchor = truth_anchors.get(right)
                if left_anchor is not None and right_anchor is not None:
                    pair_class = (
                        "true_pair"
                        if left_anchor == right_anchor
                        else "false_truth_pair"
                    )
                elif left_anchor is not None or right_anchor is not None:
                    pair_class = "truth_decoy_pair"
                else:
                    pair_class = "background_pair"
                pair_counts[pair_class] += 1
                writer.writerow([left, right, pair_class, representative])

    subset = Path(subset_fasta_path)
    subset.parent.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    with Path(fasta_path).open("rb") as source, subset.open("w") as target:
        for identifier, sequence in _iter_fasta_records(source):
            if identifier not in member_clusters:
                continue
            target.write(f">{identifier}\n{sequence}\n")
            written.add(identifier)
    assert written == set(member_clusters)

    return {
        "pair_class_counts": dict(sorted(pair_counts.items())),
        "selected_cluster_count": len(members_by_cluster),
        "selected_record_count": len(member_clusters),
        "truth_cluster_count": len(truth_clusters),
        "truth_record_count": len(truth_anchors),
    }


def evaluate_seed_alignments(
    *,
    truth_path: str | Path,
    pairs_path: str | Path,
    alignments_path: str | Path,
    thresholds: list[dict[str, object]],
) -> dict[str, object]:
    """Measure which graph pairs survive configured nucleotide-alignment gates."""
    truth_anchors = _truth_anchors(truth_path)
    truth_by_anchor: dict[str, list[str]] = defaultdict(list)
    for record_id, anchor in truth_anchors.items():
        truth_by_anchor[anchor].append(record_id)
    global_true_pair_count = sum(
        sum(1 for _ in itertools.combinations(records, 2))
        for records in truth_by_anchor.values()
    )
    assert global_true_pair_count > 0

    graph_pairs: dict[tuple[str, str], str] = {}
    with Path(pairs_path).open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["left"], row["right"])
            assert key not in graph_pairs
            graph_pairs[key] = row["pair_class"]

    best_alignments: dict[tuple[str, str], dict[str, float]] = {}
    with Path(alignments_path).open(newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            assert len(row) == 12
            query, target = row[0], row[1]
            if query == target:
                continue
            key = tuple(sorted((query, target)))
            if key not in graph_pairs:
                continue
            alignment = {
                "fident": float(row[2]),
                "qcov": float(row[4]),
                "tcov": float(row[5]),
                "evalue": float(row[10]),
                "bits": float(row[11]),
            }
            previous = best_alignments.get(key)
            rank = (alignment["evalue"], -alignment["bits"])
            if previous is None or rank < (previous["evalue"], -previous["bits"]):
                best_alignments[key] = alignment

    pair_class_counts = Counter(graph_pairs.values())
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        name = str(threshold["name"])
        min_identity = float(threshold["min_sequence_identity"])
        min_coverage = float(threshold["coverage"])
        max_evalue = float(threshold["evalue"])
        retained_counts: Counter[str] = Counter()
        for key, pair_class in graph_pairs.items():
            alignment = best_alignments.get(key)
            if alignment is None:
                continue
            if (
                alignment["fident"] >= min_identity
                and min(alignment["qcov"], alignment["tcov"]) >= min_coverage
                and alignment["evalue"] <= max_evalue
            ):
                retained_counts[pair_class] += 1
        retained_total = sum(retained_counts.values())
        retained_true = retained_counts["true_pair"]
        graph_true = pair_class_counts["true_pair"]
        graph_decoy = pair_class_counts["truth_decoy_pair"]
        rows.append(
            {
                "name": name,
                "min_sequence_identity": min_identity,
                "coverage": min_coverage,
                "evalue": max_evalue,
                "retained_pair_class_counts": dict(sorted(retained_counts.items())),
                "retained_pair_count": retained_total,
                "retained_true_pair_count": retained_true,
                "global_true_pair_recall": retained_true / global_true_pair_count,
                "graph_true_pair_retention": retained_true / graph_true,
                "truth_decoy_pair_retention": (
                    retained_counts["truth_decoy_pair"] / graph_decoy
                    if graph_decoy
                    else 0.0
                ),
                "retained_pair_precision": (
                    retained_true / retained_total if retained_total else 1.0
                ),
            }
        )

    return {
        "aligned_graph_pair_count": len(best_alignments),
        "global_true_pair_count": global_true_pair_count,
        "graph_pair_class_counts": dict(sorted(pair_class_counts.items())),
        "thresholds": rows,
    }
