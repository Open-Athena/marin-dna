from __future__ import annotations

from pathlib import Path

import polars as pl

from marin_dna.pipelines.vertebrate_projection_dataset.contract import (
    ACCEPTED_SCHEMA,
    REJECTION_SCHEMA,
)
from marin_dna.pipelines.vertebrate_projection_dataset.inspection import (
    assert_zrs_broad_recovery,
    build_inspection_sample,
    build_rejection_inspection_sample,
    render_inspection_report,
)
from marin_dna.pipelines.vertebrate_projection_dataset.pipeline_io import (
    write_inspection_files,
)


def _row(
    query_name: str,
    species: str,
    alignment_source: str,
    clade: str,
    phylogenetic_rank: int,
    sequence: str = "Aa" + "C" * 253,
    region_label: str = "ccre_non_promoter",
) -> dict[str, object]:
    return {
        "query_name": query_name,
        "source_chrom": "chr7" if query_name.startswith("zrs_") else "chr1",
        "source_start": 10,
        "source_end": 265,
        "region_label": region_label,
        "species": species,
        "alignment_name": species.replace(" ", "_"),
        "assembly": species.replace(" ", "_assembly"),
        "taxonomy_id": 9606 if species == "Homo sapiens" else 1,
        "family": "family",
        "clade": clade,
        "phylogenetic_rank": phylogenetic_rank,
        "alignment_source": alignment_source,
        "t_chrom": "chr1",
        "t_start": 100,
        "t_end": 355,
        "t_strand": "+",
        "t_src_size": 1000,
        "pre_resize_t_start": 100,
        "pre_resize_t_end": 355,
        "fragment_count": 1,
        "aligned_bases": 255,
        "sequence": sequence,
    }


def test_inspection_sample_includes_zrs_scopes_and_is_deterministic() -> None:
    schema = {**ACCEPTED_SCHEMA, "sequence": pl.String}
    rows = pl.DataFrame(
        [
            _row("zrs_one", "Homo sapiens", "human_reference", "mammals", 0),
            _row("zrs_one", "Mus musculus", "zoonomia_cactus", "mammals", 1),
            _row("zrs_one", "Gallus gallus", "ucsc_multiz100way", "birds", 2),
            _row("zrs_one", "Danio rerio", "ucsc_multiz100way", "fish", 6),
            _row(
                "other",
                "Mus musculus",
                "zoonomia_cactus",
                "mammals",
                1,
                region_label="cds",
            ),
            _row("other", "Gallus gallus", "ucsc_multiz100way", "birds", 2),
        ],
        schema=schema,
    )
    first = build_inspection_sample(rows, seed=7, rows_per_region=1)
    second = build_inspection_sample(rows.reverse(), seed=7, rows_per_region=1)
    assert first.to_dicts() == second.to_dicts()
    zrs = first.filter(pl.col("query_name") == "zrs_one")
    assert set(zrs["species"].to_list()) == {
        "Homo sapiens",
        "Mus musculus",
        "Gallus gallus",
        "Danio rerio",
    }
    assert first["valid_iupac"].all()
    assert first["sequence_length"].to_list() == [255] * first.height

    rejected = pl.DataFrame(
        [
            {
                "query_name": "other",
                "source_chrom": "chr1",
                "source_start": 10,
                "source_end": 265,
                "region_label": "cds",
                "species": "Xenopus tropicalis",
                "assembly": "xenTro7",
                "taxonomy_id": 8364,
                "family": "Pipidae",
                "clade": "amphibians",
                "phylogenetic_rank": 4,
                "alignment_source": "ucsc_multiz100way",
                "rejection_reason": "multi_strand",
                "detail": "+,-",
                "fragment_count": 2,
            }
        ]
    )
    rejected_sample = build_rejection_inspection_sample(rejected, seed=7)
    report = render_inspection_report(first, rejected_sample, seed=7)
    assert "pending human review" in report
    assert "zrs_one" in report
    assert "Aa" + "C" * 253 in report
    assert "multi_strand" in report
    assert_zrs_broad_recovery(rows)


def test_inspection_rejects_invalid_sequence() -> None:
    schema = {**ACCEPTED_SCHEMA, "sequence": pl.String}
    rows = pl.DataFrame(
        [_row("zrs_one", "Homo sapiens", "human_reference", "mammals", 0, "X" * 255)],
        schema=schema,
    )
    try:
        build_inspection_sample(rows)
    except AssertionError:
        pass
    else:
        raise AssertionError("invalid sequence should fail inspection prechecks")


def test_streaming_inspection_materializes_only_sample_candidates(
    tmp_path: Path,
) -> None:
    schema = {**ACCEPTED_SCHEMA, "sequence": pl.String}
    rows = pl.DataFrame(
        [
            _row("zrs_one", "Homo sapiens", "human_reference", "mammals", 0),
            _row("zrs_one", "Mus musculus", "zoonomia_cactus", "mammals", 1),
            _row("zrs_one", "Gallus gallus", "ucsc_multiz100way", "birds", 2),
            _row("zrs_one", "Danio rerio", "ucsc_multiz100way", "fish", 6),
            _row(
                "cds_one",
                "Mus musculus",
                "zoonomia_cactus",
                "mammals",
                1,
                region_label="cds",
            ),
            _row(
                "ccre_one",
                "Gallus gallus",
                "ucsc_multiz100way",
                "birds",
                2,
            ),
        ],
        schema=schema,
    ).with_columns(
        pl.when(pl.col("query_name") == "ccre_one")
        .then(pl.lit(2))
        .otherwise(pl.col("fragment_count"))
        .alias("fragment_count")
    )
    rejected = pl.DataFrame(
        [
            {
                "query_name": "cds_rejected",
                "source_chrom": "chr1",
                "source_start": 10,
                "source_end": 265,
                "region_label": "cds",
                "species": "Xenopus tropicalis",
                "assembly": "xenTro7",
                "taxonomy_id": 8364,
                "family": "Pipidae",
                "clade": "amphibians",
                "phylogenetic_rank": 4,
                "alignment_source": "ucsc_multiz100way",
                "rejection_reason": "multi_strand",
                "detail": "+,-",
                "fragment_count": 2,
            }
        ],
        schema=REJECTION_SCHEMA,
    )
    sequences_path = tmp_path / "sequences.parquet"
    rejected_path = tmp_path / "rejected.parquet"
    rows.write_parquet(sequences_path)
    rejected.write_parquet(rejected_path)
    sample_path = tmp_path / "sample.tsv"
    rejected_sample_path = tmp_path / "rejected.tsv"
    report_path = tmp_path / "report.md"

    write_inspection_files(
        sequences_path,
        [str(rejected_path)],
        sample_path,
        rejected_sample_path,
        report_path,
        seed=7,
        rows_per_region=1,
        fragmented_rows=1,
        rejected_rows_per_reason=1,
    )

    sample = pl.read_csv(sample_path, separator="\t")
    assert {"zrs_one", "cds_one", "ccre_one"} <= set(sample["query_name"])
    assert pl.read_csv(rejected_sample_path, separator="\t").height == 1
    assert "pending human review" in report_path.read_text()

    full_sequences_path = tmp_path / "full-sequences.parquet"
    rows.filter(
        ~pl.col("query_name").str.to_lowercase().str.starts_with("zrs_")
    ).write_parquet(full_sequences_path)
    full_report_path = tmp_path / "full-report.md"
    write_inspection_files(
        full_sequences_path,
        [str(rejected_path)],
        tmp_path / "full-sample.tsv",
        tmp_path / "full-rejected.tsv",
        full_report_path,
        seed=7,
        rows_per_region=1,
        fragmented_rows=1,
        rejected_rows_per_reason=1,
        require_zrs=False,
    )

    full_report = full_report_path.read_text()
    assert "separate sidecar QC" in full_report
    assert "intentionally absent from the conservation-filtered grid" in full_report
