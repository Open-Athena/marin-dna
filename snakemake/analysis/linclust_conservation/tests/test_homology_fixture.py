from pathlib import Path

import polars as pl
from marin_dna_linclust_conservation.homology_fixture import (
    ProjectionSource,
    build_center_expanded_projection_fixture,
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


def test_center_expanded_fixture_preserves_centers_and_source_order(
    tmp_path: Path, monkeypatch
) -> None:
    sources = [_source("human"), _source("mouse"), _source("armadillo")]
    projection_paths = [tmp_path / f"{source.label}.parquet" for source in sources]
    genome_paths = [tmp_path / f"{source.label}.2bit" for source in sources]
    for source, projection_path, genome_path in zip(
        sources, projection_paths, genome_paths, strict=True
    ):
        pl.DataFrame(
            [
                {
                    "query_name": query_name,
                    "source_chrom": "chr1",
                    "source_start": index * 4,
                    "source_end": index * 4 + 5,
                    "region_label": "background",
                    "species": source.species,
                    "assembly": source.assembly,
                    "sequence": "ACGTA",
                    "t_chrom": "target1",
                    "t_strand": "-" if source.label == "armadillo" else "+",
                    "t_src_size": 100,
                    "pre_resize_t_start": 40 + index * 10,
                    "pre_resize_t_end": 41 + index * 10,
                }
                for index, query_name in enumerate(("anchor_a", "anchor_b"))
            ]
        ).write_parquet(projection_path)
        genome_path.write_bytes(b"fake twobit")

    def fake_twobit_to_fa(command: list[str], *, check: bool) -> None:
        assert check
        bed_path = Path(
            next(
                value.removeprefix("-bed=")
                for value in command
                if value.startswith("-bed=")
            )
        )
        output_path = Path(command[2])
        records = []
        for line in bed_path.read_text().splitlines():
            _, start, end, record_id, _, strand = line.split("\t")
            length = int(end) - int(start)
            sequence = ("A" if strand == "+" else "T") * length
            records.append(f">{record_id}\n{sequence}\n")
        output_path.write_text("".join(records))

    monkeypatch.setattr(
        "marin_dna_linclust_conservation.homology_fixture.subprocess.run",
        fake_twobit_to_fa,
    )
    receipt = build_center_expanded_projection_fixture(
        sources=sources,
        projection_paths=projection_paths,
        genome_paths=genome_paths,
        max_anchors=1,
        candidate_anchors=2,
        selection_window_length=5,
        window_length=9,
        fasta_path=tmp_path / "expanded.fasta",
        truth_path=tmp_path / "expanded_truth.tsv",
    )

    assert receipt["selected_anchor_count"] == 1
    assert receipt["matched_embedded_prefix_anchor_count"] == 1
    assert receipt["window_length"] == 9
    fasta_lines = (tmp_path / "expanded.fasta").read_text().splitlines()
    assert fasta_lines[1::2] == ["A" * 9, "A" * 9, "T" * 9]
    truth = pl.read_csv(tmp_path / "expanded_truth.tsv", separator="\t")
    assert truth["target_start"].to_list() == [36, 36, 36]
    assert truth["target_end"].to_list() == [45, 45, 45]
    assert truth["target_strand"].to_list() == ["+", "+", "-"]
