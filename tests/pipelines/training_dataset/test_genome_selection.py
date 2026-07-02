import pandas as pd

from marin_dna.pipelines.training_dataset.genome_selection import (
    read_accessions_tsv,
    select_one_per_rank,
)


def _genomes() -> pd.DataFrame:
    # Order O1 spans families F1/F2; order O2 spans families F3/F4.
    return pd.DataFrame(
        {
            "Assembly Accession": ["A", "B", "C", "D", "E"],
            "Assembly Level": [
                "Scaffold",
                "Chromosome",
                "Contig",
                "Chromosome",
                "Chromosome",
            ],
            "Assembly Stats Total Sequence Length": [10, 20, 5, 30, 9_999],
            "Organism Name": ["a", "b", "c", "d", "e"],
            "family": ["F1", "F1", "F2", "F3", "F4"],
            "order": ["O1", "O1", "O1", "O2", "O2"],
        }
    )


def test_picks_best_assembly_level_then_smallest():
    # O1: B (Chromosome) beats A (Scaffold) and C (Contig).
    # O2: D and E both Chromosome; D smaller wins.
    picks = _picks(select_one_per_rank(_genomes(), "order"))
    assert picks == {"O1": "B", "O2": "D"}


def test_priority_overrides_ranking():
    picks = _picks(select_one_per_rank(_genomes(), "order", priority=["C"]))
    assert picks["O1"] == "C"  # forced to top despite Contig level


def test_exclude_size_and_nan_rank():
    g = _genomes()
    g.loc[g["Assembly Accession"] == "E", "order"] = None  # NaN rank -> dropped
    picks = _picks(select_one_per_rank(g, "order", exclude=["B"], max_genome_size=25))
    # O1: B excluded -> A (Scaffold) beats C (Contig).
    assert picks["O1"] == "A"
    # O2: E has no order, D (30) exceeds the size cap -> order absent entirely.
    assert "O2" not in picks


def test_order_winners_are_a_subset_of_family_winners():
    g = _genomes()
    fam = set(select_one_per_rank(g, "family")["Assembly Accession"])
    order = set(select_one_per_rank(g, "order")["Assembly Accession"])
    assert order <= fam


def test_one_row_per_rank_value():
    out = select_one_per_rank(_genomes(), "order")
    assert out["order"].is_unique
    assert out["Assembly Accession"].is_unique


def test_read_accessions_tsv(tmp_path):
    p = tmp_path / "species.tsv"
    p.write_text("Assembly Accession\torder\nGCF_1\tO1\nGCF_2\tO2\n")
    assert read_accessions_tsv(str(p)) == ["GCF_1", "GCF_2"]


def _picks(out: pd.DataFrame) -> dict[str, str]:
    return dict(zip(out["order"], out["Assembly Accession"]))
