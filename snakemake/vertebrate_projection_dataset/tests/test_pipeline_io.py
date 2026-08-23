from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from marin_dna_vertebrate_projection.contract import ACCEPTED_SCHEMA
from marin_dna_vertebrate_projection.pipeline_io import (
    combine_sequence_parquets,
    write_dataset_card,
    write_filtered_anchor_bed,
    write_human_reference_sequences,
    write_twobit_sequences,
)

from .helpers import species_manifest


def test_write_filtered_anchor_bed_is_valid_deterministic_gzip(
    tmp_path: Path,
) -> None:
    scored_paths: list[str] = []
    for chrom, offset in [("chr1", 0), ("chr2", 1_000)]:
        path = tmp_path / f"{chrom}.parquet"
        pl.DataFrame(
            {
                "chrom": [chrom, chrom],
                "start": [offset, offset + 255],
                "end": [offset + 255, offset + 510],
                "name": [f"{chrom}-keep", f"{chrom}-drop"],
                "proportion_conserved": [0.8, 0.2],
            }
        ).write_parquet(path)
        scored_paths.append(str(path))

    first = tmp_path / "first.bed.gz"
    second = tmp_path / "second.bed.gz"
    for output in [first, second]:
        write_filtered_anchor_bed(
            scored_paths,
            output,
            min_proportion_conserved=0.65,
        )

    assert first.read_bytes() == second.read_bytes()
    assert pl.read_csv(
        first,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end", "name"],
    ).to_dicts() == [
        {"chrom": 1, "start": 0, "end": 255, "name": "chr1-keep"},
        {"chrom": 2, "start": 1_000, "end": 1_255, "name": "chr2-keep"},
    ]


def test_write_twobit_sequences_batches_bed6_and_preserves_tool_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = tmp_path / "accepted.parquet"
    pl.DataFrame(
        [
            {
                "query_name": "anchor-1",
                "source_chrom": "chr1",
                "source_start": 100,
                "source_end": 104,
                "region_label": "cds",
                "species": "Test species",
                "alignment_name": "testSpecies",
                "assembly": "test-assembly",
                "taxonomy_id": 1,
                "family": "Testidae",
                "clade": "mammals",
                "phylogenetic_rank": 1,
                "alignment_source": "zoonomia_cactus",
                "t_chrom": "chr2",
                "t_start": 200,
                "t_end": 204,
                "t_strand": "-",
                "t_src_size": 1_000,
                "pre_resize_t_start": 200,
                "pre_resize_t_end": 204,
                "fragment_count": 1,
                "aligned_bases": 4,
            }
        ],
        schema=ACCEPTED_SCHEMA,
    ).write_parquet(accepted)
    two_bit = tmp_path / "genome.2bit"
    two_bit.write_bytes(b"test stub")

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check
        assert command[:2] == ["twoBitToFa", str(two_bit)]
        bed = Path(command[2].removeprefix("-bed="))
        assert bed.read_text() == "chr2\t200\t204\tanchor-1\t0\t-\n"
        Path(command[3]).write_text(">anchor-1\naC\ngT\n")

    monkeypatch.setattr(
        "marin_dna_vertebrate_projection.pipeline_io.subprocess.run",
        fake_run,
    )
    sequences = tmp_path / "sequences.parquet"
    rejected = tmp_path / "rejected.parquet"
    write_twobit_sequences(
        accepted,
        two_bit,
        sequences,
        rejected,
        target_length=4,
    )

    assert pl.read_parquet(sequences)["sequence"].to_list() == ["aCgT"]
    assert pl.read_parquet(rejected).is_empty()


def test_write_human_reference_sequences_uses_batched_twobit_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    anchors = tmp_path / "anchors.parquet"
    pl.DataFrame(
        {
            "query_name": ["anchor-1"],
            "source_chrom": ["chr1"],
            "source_start": [0],
            "source_end": [4],
            "region_label": ["cds"],
        }
    ).write_parquet(anchors)
    sizes = tmp_path / "chrom.sizes"
    sizes.write_text("chr1\t100\n")
    two_bit = tmp_path / "human.2bit"
    two_bit.write_bytes(b"test stub")

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check
        assert command[:2] == ["twoBitToFa", str(two_bit)]
        bed = Path(command[2].removeprefix("-bed="))
        assert bed.read_text() == "chr1\t0\t4\tanchor-1\t0\t+\n"
        Path(command[3]).write_text(">anchor-1\naC\ngT\n")

    monkeypatch.setattr(
        "marin_dna_vertebrate_projection.pipeline_io.subprocess.run",
        fake_run,
    )
    output = tmp_path / "human.parquet"
    write_human_reference_sequences(
        anchors,
        two_bit,
        sizes,
        output,
        target_length=4,
    )

    result = pl.read_parquet(output)
    assert result["sequence"].to_list() == ["aCgT"]
    assert result.select("species", "alignment_source").row(0) == (
        "Homo sapiens",
        "human_reference",
    )


def test_combine_sequence_parquets_accepts_one_species_per_input(
    tmp_path: Path,
) -> None:
    paths: list[str] = []
    for index, species in enumerate(["Homo sapiens", "Mus musculus"]):
        path = tmp_path / f"species-{index}.parquet"
        pl.DataFrame(
            {
                "query_name": ["anchor-1", "anchor-2"],
                "species": [species, species],
                "sequence": ["A" * 255, "C" * 255],
            }
        ).write_parquet(path)
        paths.append(str(path))

    output = tmp_path / "combined.parquet"
    combine_sequence_parquets(paths, output)

    combined = pl.read_parquet(output)
    assert combined.height == 4
    assert set(combined["species"]) == {"Homo sapiens", "Mus musculus"}


def test_combine_sequence_parquets_accepts_schema_valid_empty_input(
    tmp_path: Path,
) -> None:
    populated = tmp_path / "populated.parquet"
    empty = tmp_path / "empty.parquet"
    frame = pl.DataFrame(
        {
            "query_name": ["anchor-1"],
            "species": ["Homo sapiens"],
            "sequence": ["A" * 255],
        }
    )
    frame.write_parquet(populated)
    frame.clear().write_parquet(empty)

    output = tmp_path / "combined.parquet"
    combine_sequence_parquets([str(populated), str(empty)], output)

    assert pl.read_parquet(output).to_dicts() == frame.to_dicts()


def test_combine_sequence_parquets_rejects_empty_input_with_wrong_schema(
    tmp_path: Path,
) -> None:
    populated = tmp_path / "populated.parquet"
    malformed = tmp_path / "malformed.parquet"
    pl.DataFrame(
        {
            "query_name": ["anchor-1"],
            "species": ["Homo sapiens"],
            "sequence": ["A" * 255],
        }
    ).write_parquet(populated)
    pl.DataFrame(schema={"query_name": pl.String}).write_parquet(malformed)

    with pytest.raises(AssertionError):
        combine_sequence_parquets(
            [str(populated), str(malformed)], tmp_path / "combined.parquet"
        )


def test_dataset_card_distinguishes_anchor_filter_from_repeat_mask_case(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.parquet"
    validation = tmp_path / "validation.parquet"
    manifest = tmp_path / "species.tsv"
    card = tmp_path / "README.md"
    frame = pl.DataFrame(
        {
            "query_name": ["anchor-1"],
            "sequence": ["aCgT"],
        }
    )
    frame.write_parquet(train)
    frame.write_parquet(validation)
    species_manifest().write_csv(manifest, separator="\t")

    write_dataset_card(
        train,
        validation,
        manifest,
        card,
        pipeline_commit="a" * 40,
        hf_repo="marin-dna/vertebrate-v1-cds",
        region_label="cds",
        species_scope="all",
    )

    text = " ".join(card.read_text().split())
    assert (
        "configs: - config_name: default data_files: - split: train path: "
        "data/train/*.jsonl.zst - split: validation path: "
        "data/validation/*.jsonl.zst" in text
    )
    assert (
        "Anchor eligibility uses the pipeline's pinned phyloP conservation filter."
        in text
    )
    assert "lowercase bases preserve source repeat masking" in text
    assert "Sequence case is independent of that filter" in text
    assert "project only the central human nucleotide" in text
    assert "conservation scores never rewrite emitted characters or case" in text
    assert "loss weight" not in text
