from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from marin_dna_vertebrate_projection.projection.liftover import (
    attach_target_sizes,
    chain_filename,
    liftover_records_to_fragments,
    read_liftover_chain_manifest,
    validate_liftover_chain_manifest,
    validate_liftover_partition,
)
from marin_dna_vertebrate_projection.projection.requests import (
    build_projection_requests,
)

from .helpers import species_manifest


def _write_bed(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + ("\n" if rows else ""))


def test_chain_filename_preserves_ucsc_database_suffix() -> None:
    assert chain_filename("galGal4") == "hg38ToGalGal4.over.chain.gz"
    assert chain_filename("fr3") == "hg38ToFr3.over.chain.gz"


def test_liftover_manifest_is_complete_for_selected_non_mammals() -> None:
    root = Path(__file__).parents[1]
    chains = read_liftover_chain_manifest(root / "config/liftover_chain_manifest.tsv")
    names = (
        pl.read_csv(root / "config/species_selected.tsv", separator="\t")
        .filter((pl.col("backend") == "ucsc_multiz100way") & pl.col("selected"))[
            "alignment_name"
        ]
        .to_list()
    )
    validate_liftover_chain_manifest(chains, names)
    assert len(chains) == 28
    assert sum(item.byte_size for item in chains.values()) == 279_346_460


def test_liftover_partition_reconciles_mapped_and_unmapped(tmp_path: Path) -> None:
    source = tmp_path / "source.bed"
    mapped = tmp_path / "mapped.bed"
    unmapped = tmp_path / "unmapped.bed"
    _write_bed(source, ["chr1\t227\t228\ta1\t0\t+", "chr2\t327\t328\ta2\t0\t+"])
    _write_bed(mapped, ["chr5\t99\t100\ta1\t0\t-"])
    _write_bed(unmapped, ["#Deleted in new", "chr2\t327\t328\ta2\t0\t+"])

    assert validate_liftover_partition(source, mapped, unmapped, multiple=False) == {
        "input_queries": 2,
        "mapped_queries": 1,
        "mapped_rows": 1,
        "unmapped_queries": 1,
    }


def test_liftover_partition_rejects_silent_loss(tmp_path: Path) -> None:
    source = tmp_path / "source.bed"
    mapped = tmp_path / "mapped.bed"
    unmapped = tmp_path / "unmapped.bed"
    _write_bed(source, ["chr1\t227\t228\ta1\t0\t+", "chr2\t327\t328\ta2\t0\t+"])
    _write_bed(mapped, ["chr5\t99\t100\ta1\t0\t-"])
    _write_bed(unmapped, [])

    with pytest.raises(AssertionError, match="missing"):
        validate_liftover_partition(source, mapped, unmapped, multiple=False)


def test_liftover_records_preserve_anchor_and_center_coordinates(
    tmp_path: Path,
) -> None:
    anchors = pl.DataFrame(
        {
            "query_name": ["a1"],
            "source_chrom": ["chr1"],
            "source_start": [100],
            "source_end": [355],
            "region_label": ["cds"],
        }
    )
    requests = build_projection_requests(anchors)
    records = pl.DataFrame(
        {
            "t_chrom": ["chr2"],
            "t_start": [200],
            "t_end": [201],
            "query_name": ["a1"],
            "score": [0],
            "t_strand": ["-"],
        }
    )
    sizes = tmp_path / "sizes.tsv"
    sizes.write_text("chr2\t1000\n")
    records = attach_target_sizes(records, sizes)

    fragments = liftover_records_to_fragments(
        records,
        requests,
        species_manifest(),
        alignment_name="galGal4",
    )

    assert fragments.height == 1
    row = fragments.row(0, named=True)
    assert (row["source_start"], row["source_end"]) == (100, 355)
    assert (row["source_fragment_start"], row["source_fragment_end"]) == (
        227,
        228,
    )
    assert (row["t_start"], row["t_end"], row["t_strand"]) == (200, 201, "-")
    assert row["t_src_size"] == 1000
    assert row["alignment_name"] == "galGal4"
    assert row["alignment_source"] == "ucsc_multiz100way"
