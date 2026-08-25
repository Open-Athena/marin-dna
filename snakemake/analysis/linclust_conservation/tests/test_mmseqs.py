from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from marin_dna_linclust_conservation.mmseqs import (
    cluster_membership_features,
    parse_alignments,
    parse_cluster_assignments,
    validate_score_features,
)


def test_parse_cluster_assignments_requires_one_self_row_per_cluster(
    tmp_path: Path,
) -> None:
    path = tmp_path / "clusters.tsv"
    path.write_text("rep1\trep1\nrep1\tmember1\nrep2\trep2\n")
    assignments = parse_cluster_assignments(path)
    assert assignments.height == 3

    path.write_text("rep1\tmember1\n")
    with pytest.raises(AssertionError, match="representative"):
        parse_cluster_assignments(path)


def test_parse_alignments_converts_one_based_coordinates_at_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "alignments.tsv"
    path.write_text("q\tt\t0.95\t200\t0.8\t0.9\t1\t200\t220\t21\t1e-20\t100\n")
    alignment = parse_alignments(path).row(0, named=True)
    assert (alignment["query_start"], alignment["query_end"]) == (0, 200)
    assert (alignment["target_start"], alignment["target_end"]) == (20, 220)
    assert alignment["reverse_strand"] is True


def test_membership_features_count_distinct_genomes_and_multiplicity() -> None:
    assignments = pl.DataFrame(
        {
            "representative": ["a", "a", "a", "d"],
            "member": ["a", "b", "c", "d"],
        }
    )
    member_to_genome = pl.DataFrame(
        {
            "member": ["a", "b", "c", "d"],
            "source_genome": ["g1", "g1", "g2", "g3"],
        }
    )
    features = cluster_membership_features(
        assignments,
        member_to_genome=member_to_genome,
    )
    a = features.filter(pl.col("member") == "a").row(0, named=True)
    d = features.filter(pl.col("member") == "d").row(0, named=True)
    assert a["cluster_member_count"] == 3
    assert a["distinct_genome_count"] == 2
    assert a["max_members_per_genome"] == 2
    assert a["dominant_genome_fraction"] == pytest.approx(2 / 3)
    assert a["singleton"] is False
    assert d["singleton"] is True


def test_deployed_feature_contract_excludes_diagnostics() -> None:
    validate_score_features(["cluster_member_count", "distinct_genome_count"])
    with pytest.raises(AssertionError, match="forbidden"):
        validate_score_features(["cluster_member_count", "phylop_fraction"])
    with pytest.raises(AssertionError, match="undeclared"):
        validate_score_features(["mystery_feature"])
