import csv
from pathlib import Path

from marin_dna_linclust_conservation.seed_alignment import (
    evaluate_seed_alignments,
    prepare_seed_alignment_subset,
)


def test_prepare_and_evaluate_seed_alignment_subset(tmp_path: Path) -> None:
    fasta = tmp_path / "windows.fasta"
    fasta.write_text(
        ">anchor000000__human\nAAAAAAAAAA\n"
        ">anchor000000__mouse\nAAAAAAAAAT\n"
        ">anchor000001__armadillo\nCCCCCCCCCC\n"
        ">opossum|decoy\nTTTTTTTTTT\n"
    )
    truth = tmp_path / "truth.tsv"
    with truth.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["query_name", "record_id"])
        writer.writerow(["anchor000000", "anchor000000__human"])
        writer.writerow(["anchor000000", "anchor000000__mouse"])
        writer.writerow(["anchor000001", "anchor000001__armadillo"])
    assignments = tmp_path / "clusters.tsv"
    assignments.write_text(
        "anchor000000__human\tanchor000000__human\n"
        "anchor000000__human\tanchor000000__mouse\n"
        "anchor000001__armadillo\tanchor000001__armadillo\n"
        "anchor000000__human\topossum|decoy\n"
    )
    subset = tmp_path / "subset.fasta"
    pairs = tmp_path / "pairs.tsv"
    receipt = prepare_seed_alignment_subset(
        fasta_path=fasta,
        truth_path=truth,
        assignments_path=assignments,
        subset_fasta_path=subset,
        pairs_path=pairs,
    )
    assert receipt["selected_record_count"] == 4
    assert receipt["pair_class_counts"] == {
        "true_pair": 1,
        "truth_decoy_pair": 2,
    }

    alignments = tmp_path / "alignments.tsv"
    alignments.write_text(
        "anchor000000__human\tanchor000000__mouse\t0.8\t10\t1.0\t1.0\t1\t10\t1\t10\t1e-9\t50\n"
        "anchor000000__human\topossum|decoy\t0.3\t5\t0.5\t0.5\t1\t5\t1\t5\t0.1\t5\n"
    )
    evaluation = evaluate_seed_alignments(
        truth_path=truth,
        pairs_path=pairs,
        alignments_path=alignments,
        thresholds=[
            {
                "name": "relaxed",
                "min_sequence_identity": 0.4,
                "coverage": 0.7,
                "evalue": 0.001,
            }
        ],
    )
    result = evaluation["thresholds"][0]
    assert result["global_true_pair_recall"] == 1.0
    assert result["truth_decoy_pair_retention"] == 0.0
    assert result["retained_pair_precision"] == 1.0
