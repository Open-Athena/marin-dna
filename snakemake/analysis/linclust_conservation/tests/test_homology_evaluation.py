from pathlib import Path

from marin_dna_linclust_conservation.homology_evaluation import (
    evaluate_homology_clusters,
)


def _alignment(query: str, target: str, identity: float = 1.0) -> str:
    return f"{query}\t{target}\t{identity}\t255\t1.0\t1.0\t1\t255\t1\t255\t0\t255\n"


def test_homology_evaluation_measures_pair_recall(tmp_path: Path) -> None:
    truth = tmp_path / "truth.tsv"
    truth.write_text(
        "anchor_index\tquery_name\trecord_id\tsource_label\tspecies\tassembly\t"
        "region_label\tsource_chrom\tsource_start\tsource_end\n"
        "0\tq0\ta0h\th\thuman\thg\tcds\tchr1\t0\t255\n"
        "0\tq0\ta0m\tm\tmouse\tmm\tcds\tchr1\t0\t255\n"
        "0\tq0\ta0a\ta\tarmadillo\tda\tcds\tchr1\t0\t255\n"
        "1\tq1\ta1h\th\thuman\thg\tcds\tchr1\t128\t383\n"
        "1\tq1\ta1m\tm\tmouse\tmm\tcds\tchr1\t128\t383\n"
        "1\tq1\ta1a\ta\tarmadillo\tda\tcds\tchr1\t128\t383\n"
    )
    assignments = tmp_path / "clusters.tsv"
    assignments.write_text(
        "a0h\ta0h\na0h\ta0m\na0h\ta0a\na1h\ta1h\na1h\ta1m\na1a\ta1a\n"
    )
    alignments = tmp_path / "alignments.tsv"
    alignments.write_text(
        "".join(
            [
                _alignment("a0h", "a0h"),
                _alignment("a0h", "a0m", 0.8),
                _alignment("a0h", "a0a", 0.7),
                _alignment("a1h", "a1h"),
                _alignment("a1h", "a1m", 0.6),
                _alignment("a1a", "a1a"),
            ]
        )
    )

    result = evaluate_homology_clusters(
        truth_path=truth,
        assignments_path=assignments,
        alignments_path=alignments,
    )

    assert result["anchor_count"] == 2
    assert result["cluster_count"] == 3
    assert result["cluster_count_over_ideal"] == 1.5
    assert result["exact_anchor_cluster_count"] == 1
    assert result["exact_anchor_recovery_fraction"] == 0.5
    assert result["true_pair_count"] == 6
    assert result["recovered_true_pair_count"] == 4
    assert result["true_pair_recall"] == 4 / 6
    assert result["true_pair_recall_by_species_pair"] == {
        "a--h": 0.5,
        "a--m": 0.5,
        "h--m": 1.0,
    }
    assert result["pair_precision"] == 1.0
    assert result["impure_cluster_count"] == 0
    assert result["false_clustered_pair_count"] == 0
    assert result["aligned_true_representative_member_edges"] == 3
