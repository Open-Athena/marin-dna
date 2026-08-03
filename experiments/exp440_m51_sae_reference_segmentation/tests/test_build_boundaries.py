from __future__ import annotations

import polars as pl

from build_boundaries import ccre_edge_candidates, gene_boundary_candidates


def _attribute(transcript_id: str, gene_id: str, *, canonical: bool = False) -> str:
    tag = ' tag "Ensembl_canonical";' if canonical else ""
    return (
        f'gene_id "{gene_id}"; transcript_id "{transcript_id}"; '
        f'gene_name "{gene_id}_name"; transcript_biotype "protein_coding";{tag}'
    )


def test_gene_boundary_candidates_handle_both_strands() -> None:
    rows: list[dict[str, object]] = []

    def add(
        chrom: str,
        feature: str,
        start: int,
        end: int,
        strand: str,
        transcript_id: str,
        gene_id: str,
    ) -> None:
        rows.append(
            {
                "chrom": chrom,
                "feature": feature,
                "start": start,
                "end": end,
                "strand": strand,
                "attribute": _attribute(
                    transcript_id, gene_id, canonical=feature == "transcript"
                ),
            }
        )

    add("1", "transcript", 100, 600, "+", "tx_plus", "g_plus")
    for feature, start, end in (
        ("exon", 100, 250),
        ("exon", 400, 600),
        ("five_prime_utr", 100, 150),
        ("CDS", 150, 250),
        ("CDS", 400, 550),
        ("stop_codon", 550, 553),
        ("three_prime_utr", 553, 600),
    ):
        add("1", feature, start, end, "+", "tx_plus", "g_plus")
    add("2", "transcript", 1_000, 1_450, "-", "tx_minus", "g_minus")
    for feature, start, end in (
        ("exon", 1_000, 1_200),
        ("exon", 1_300, 1_450),
        ("three_prime_utr", 1_000, 1_050),
        ("stop_codon", 1_050, 1_053),
        ("CDS", 1_053, 1_200),
        ("CDS", 1_300, 1_400),
        ("five_prime_utr", 1_400, 1_450),
    ):
        add("2", feature, start, end, "-", "tx_minus", "g_minus")

    observed = gene_boundary_candidates(pl.DataFrame(rows))
    lookup = {
        (record["transcript_id"], record["boundary_type"]): (
            record["boundary_position0"],
            record["direction"],
        )
        for record in observed
    }
    assert lookup == {
        ("tx_plus", "cds_to_intron"): (250, 1),
        ("tx_plus", "intron_to_cds"): (400, 1),
        ("tx_plus", "utr5_to_cds"): (150, 1),
        ("tx_plus", "cds_to_utr3"): (553, 1),
        ("tx_minus", "cds_to_intron"): (1_300, -1),
        ("tx_minus", "intron_to_cds"): (1_200, -1),
        ("tx_minus", "utr5_to_cds"): (1_400, -1),
        ("tx_minus", "cds_to_utr3"): (1_050, -1),
    }


def test_ccre_edge_candidates_alternate_edges_deterministically() -> None:
    ccre = pl.DataFrame(
        {
            "chrom": ["1", "2", "3", "4"],
            "start": [100, 200, 300, 400],
            "end": [150, 250, 350, 450],
            "cre_class": ["pELS", "dELS", "PLS", "CA"],
        }
    )
    first = ccre_edge_candidates(ccre)
    second = ccre_edge_candidates(ccre)
    assert first == second
    assert len(first) == 3
    for record in first:
        assert record["boundary_type"] in {"els_edge", "pls_edge"}
        assert record["edge_side"] in {"start", "end"}
        assert record["direction"] in {-1, 1}
        assert record["state_before"] == "outside_ccre"


def test_mixed_candidate_records_infer_nullable_string_columns() -> None:
    gene_records = [
        {
            "boundary_type": "cds_to_intron",
            "ccre_subtype": None,
        }
        for _ in range(101)
    ]
    ccre_records = [
        {
            "boundary_type": "els_edge",
            "ccre_subtype": "pELS",
        }
    ]

    frame = pl.DataFrame(gene_records + ccre_records, infer_schema_length=None)

    assert frame.schema["ccre_subtype"] == pl.String
