from pathlib import Path

import polars as pl
from marin_dna_linclust_conservation.homology_fixture import (
    ProjectionSource,
    build_projection_fixture,
)


def _source(label: str) -> ProjectionSource:
    return ProjectionSource(
        label=label,
        species=label,
        assembly=f"assembly_{label}",
        uri=f"s3://bucket/{label}.parquet",
        etag=f"etag_{label}",
        size_bytes=1,
    )


def _write_projection(
    path: Path, source: ProjectionSource, sequences: dict[str, str]
) -> None:
    pl.DataFrame(
        [
            {
                "query_name": query_name,
                "source_chrom": "chr1",
                "source_start": index * 128,
                "source_end": index * 128 + 255,
                "region_label": "background",
                "species": source.species,
                "assembly": source.assembly,
                "sequence": sequence,
            }
            for index, (query_name, sequence) in enumerate(sequences.items())
        ]
    ).write_parquet(path)


def test_projection_fixture_keeps_complete_clean_anchor_groups(tmp_path: Path) -> None:
    sources = [_source("human"), _source("mouse"), _source("armadillo")]
    paths = [tmp_path / f"{source.label}.parquet" for source in sources]
    for source, path in zip(sources, paths, strict=True):
        sequences = {
            "anchor_a": "ACGT" * 63 + "ACG",
            "anchor_b": "ACGT" * 63 + "ACN",
            "anchor_c": "acgt" * 63 + "acg",
        }
        _write_projection(path, source, sequences)

    receipt = build_projection_fixture(
        sources=sources,
        paths=paths,
        max_anchors=1,
        candidate_anchors=3,
        window_length=255,
        fasta_path=tmp_path / "fixture.fasta",
        truth_path=tmp_path / "truth.tsv",
    )

    assert receipt["common_anchor_count"] == 3
    assert receipt["selected_anchor_count"] == 1
    assert receipt["input_sequence_count"] == 3
    assert (tmp_path / "fixture.fasta").read_text().count(">") == 3
    truth = pl.read_csv(tmp_path / "truth.tsv", separator="\t")
    assert truth["query_name"].to_list() == ["anchor_a"] * 3
    assert truth["record_id"].to_list() == [
        "anchor000000__human",
        "anchor000000__mouse",
        "anchor000000__armadillo",
    ]
