import csv
from pathlib import Path

from marin_dna_linclust_conservation.mmseqs import parse_cluster_assignments
from marin_dna_linclust_conservation.seed_graph import (
    SeedGraphConfiguration,
    build_seed_graph,
    selected_sequence_seeds,
    source_label,
)


def test_selected_seeds_are_reverse_complement_invariant() -> None:
    sequence = "ACGTTGCA" * 4
    reverse_complement = "TGCAACGT" * 4
    assert selected_sequence_seeds(
        sequence, kmer_length=7, selected_count=12, hash_seed=521
    ) == selected_sequence_seeds(
        reverse_complement, kmer_length=7, selected_count=12, hash_seed=521
    )


def test_source_label_handles_background_and_truth_identifiers() -> None:
    assert source_label("GCF_1|chr1|0|255|+") == "GCF_1"
    assert source_label("anchor000001__armadillo") == "armadillo"


def test_seed_graph_caps_repeats_and_prevents_duplicate_source_components(
    tmp_path: Path,
) -> None:
    records = {
        "H|chr1|0|255|+": "A" * 255,
        "M|chr1|0|255|+": "A" * 255,
        "anchor000000__human": "ACGT" * 63 + "ACG",
        "anchor000000__mouse": "ACGT" * 63 + "ACG",
        "anchor000001__human": "ACGT" * 63 + "ACA",
        "anchor000001__mouse": "ACGT" * 63 + "ACA",
    }
    fasta = tmp_path / "records.fasta"
    fasta.write_text("".join(f">{key}\n{value}\n" for key, value in records.items()))
    truth = tmp_path / "truth.tsv"
    with truth.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("query_name", "record_id"))
        for anchor in (0, 1):
            for source in ("human", "mouse"):
                writer.writerow((f"q{anchor}", f"anchor{anchor:06d}__{source}"))
    assignments = tmp_path / "assignments.tsv"

    receipt = build_seed_graph(
        fasta_path=fasta,
        truth_path=truth,
        assignments_path=assignments,
        configuration=SeedGraphConfiguration(
            kmer_length=7,
            selected_seeds_per_sequence=16,
            max_seed_frequency=4,
            min_shared_seeds=1,
            hash_seed=521,
        ),
        source_aliases={"H": "human", "M": "mouse"},
    )

    parsed = parse_cluster_assignments(assignments)
    assert parsed.height == len(records)
    assert receipt["max_cluster_size"] <= receipt["source_count"]
    assert receipt["candidate_true_pair_recall"] > 0
    clusters = parsed.group_by("representative").agg("member")
    for members in clusters["member"]:
        labels = [source_label(member) for member in members]
        assert len(labels) == len(set(labels))
