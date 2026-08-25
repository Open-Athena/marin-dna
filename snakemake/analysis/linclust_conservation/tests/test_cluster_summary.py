from marin_dna_linclust_conservation.cluster_summary import (
    summarize_grouped_assignments,
)


def test_streaming_assignment_summary_counts_cross_genome_clusters() -> None:
    summary = summarize_grouped_assignments(
        [
            "GCF_A|chr1|0|255|+\tGCF_A|chr1|0|255|+\n",
            "GCF_A|chr1|0|255|+\tGCF_B|chr2|128|383|+\n",
            "GCF_A|chr1|256|511|+\tGCF_A|chr1|256|511|+\n",
            "GCF_B|chr2|0|255|+\tGCF_B|chr2|0|255|+\n",
            "GCF_B|chr2|0|255|+\tGCF_B|chr3|128|383|+\n",
            "GCF_B|chr2|0|255|+\tGCF_B|chr3|256|511|+\n",
        ],
        expected_accessions={"GCF_A", "GCF_B"},
    )
    assert summary.assignment_count == 6
    assert summary.cluster_count == 3
    assert summary.singleton_cluster_count == 1
    assert summary.cross_genome_cluster_count == 1
    assert summary.cross_genome_member_count == 2
    assert summary.member_count_by_accession == {"GCF_A": 2, "GCF_B": 4}
    assert summary.size_bucket_histogram == {"1": 1, "2": 1, "3": 1}
    assert summary.distinct_genome_histogram == {"1": 2, "2": 1}
    assert summary.max_cluster_size == 3
    assert summary.max_distinct_genomes == 2
