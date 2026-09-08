from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from marin_dna_vertebrate_projection.projection.chains import (
    file_sha256,
    prepare_chain_requests,
    read_bed6,
    read_chain_assets,
    read_sizes,
    validate_chain_inputs,
    write_chain_projections,
)

PROJECT = Path(__file__).parents[2]
FIXTURE = PROJECT / "tests/fixtures/chains"
SPECIES = PROJECT / "config/species_selected.tsv"


def assets() -> dict[str, str]:
    return read_chain_assets(FIXTURE / "assets.tsv", SPECIES)["Papio_anubis"]


def inputs(tmp_path: Path) -> tuple[Path, Path]:
    requests, bed = tmp_path / "requests.parquet", tmp_path / "input.bed"
    prepare_chain_requests(FIXTURE / "anchors.tsv", FIXTURE / "source.sizes", requests, bed)
    validate_chain_inputs(
        FIXTURE / "human_to_target.chain", FIXTURE / "source.sizes",
        FIXTURE / "target.sizes", assets(), tmp_path / "validated.json",
    )
    return requests, bed


def project(tmp_path: Path) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, object]]:
    write_chain_projections(
        tmp_path / "requests.parquet", tmp_path / "mapped.bed", tmp_path / "unmapped.bed",
        FIXTURE / "target.sizes", SPECIES, "Papio_anubis", tmp_path / "validated.json",
        tmp_path / "accepted.parquet", tmp_path / "rejected.parquet", tmp_path / "audit.json",
    )
    return (
        pl.read_parquet(tmp_path / "accepted.parquet"),
        pl.read_parquet(tmp_path / "rejected.parquet"),
        json.loads((tmp_path / "audit.json").read_text()),
    )


def test_center_identity_and_strand_resizing(tmp_path: Path) -> None:
    requests, bed = inputs(tmp_path)
    assert read_bed6(bed)["t_start"].to_list() == [227, 427, 727]
    assert pl.read_parquet(requests)["source_start"].to_list() == [100, 300, 600]
    (tmp_path / "mapped.bed").write_text("chrA\t527\t528\tplus\t0\t+\nchrA\t1272\t1273\tminus\t0\t-\n")
    (tmp_path / "unmapped.bed").write_text("#Deleted in new\nchr1\t727\t728\tunmapped\t0\t+\n")
    accepted, rejected, audit = project(tmp_path)
    assert accepted.select("query_name", "t_start", "t_end", "t_strand").rows() == [
        ("minus", 1145, 1400, "-"), ("plus", 400, 655, "+"),
    ]
    assert accepted["alignment_source"].to_list() == ["zoonomia_cactus"] * 2
    assert rejected.select("query_name", "rejection_reason").rows() == [("unmapped", "unmapped")]
    assert audit["accepted_queries"] == 2
    assert audit["input_queries"] == 3


def test_ambiguous_and_boundary_projections_are_rejected(tmp_path: Path) -> None:
    inputs(tmp_path)
    (tmp_path / "mapped.bed").write_text(
        "chrA\t10\t11\tplus\t0\t+\n"
        "chrA\t500\t501\tminus\t0\t-\nchrA\t600\t601\tminus\t0\t-\n"
    )
    (tmp_path / "unmapped.bed").write_text("chr1\t727\t728\tunmapped\t0\t+\n")
    accepted, rejected, audit = project(tmp_path)
    assert accepted.is_empty()
    assert set(rejected["rejection_reason"]) == {"unmapped", "duplicated_mapping", "target_window_out_of_bounds"}
    assert audit["mapped_rows"] == 3
    assert audit["mapped_queries"] == 2


def test_all_unmapped_preserves_output_schemas(tmp_path: Path) -> None:
    _, bed = inputs(tmp_path)
    (tmp_path / "mapped.bed").write_text("")
    (tmp_path / "unmapped.bed").write_text(bed.read_text())
    accepted, rejected, audit = project(tmp_path)
    assert accepted.is_empty() and rejected.height == 3
    assert audit["unmapped_queries"] == 3


@pytest.mark.parametrize("bad_mapped,bad_unmapped", [
    ("", ""),
    ("chrA\t527\t528\tunknown\t0\t+\n", ""),
    ("chrA\t527\t528\tplus\t0\t+\n", "chr1\t227\t228\tplus\t0\t+\n"),
])
def test_partition_fails_closed(tmp_path: Path, bad_mapped: str, bad_unmapped: str) -> None:
    inputs(tmp_path)
    (tmp_path / "mapped.bed").write_text(bad_mapped)
    (tmp_path / "unmapped.bed").write_text(bad_unmapped)
    with pytest.raises(ValueError):
        project(tmp_path)
    assert not (tmp_path / "audit.json").exists()


def test_unmapped_coordinates_are_checked(tmp_path: Path) -> None:
    _, bed = inputs(tmp_path)
    (tmp_path / "mapped.bed").write_text("")
    (tmp_path / "unmapped.bed").write_text(bed.read_text().replace("227\t228", "226\t227"))
    with pytest.raises(ValueError, match="unmapped coordinates"):
        project(tmp_path)


@pytest.mark.parametrize("record", ["chrMissing\t500\t501", "chrA\t2000\t2001"])
def test_unknown_or_out_of_bounds_target_fails(tmp_path: Path, record: str) -> None:
    _, bed = inputs(tmp_path)
    (tmp_path / "mapped.bed").write_text(f"{record}\tplus\t0\t+\n")
    (tmp_path / "unmapped.bed").write_text("\n".join(bed.read_text().splitlines()[1:]) + "\n")
    with pytest.raises(ValueError, match="target bounds"):
        project(tmp_path)


def test_checksums_and_direction_are_checked(tmp_path: Path) -> None:
    expected = assets()
    with pytest.raises(ValueError, match="SHA-256"):
        validate_chain_inputs(FIXTURE / "human_to_target.chain", FIXTURE / "source.sizes", FIXTURE / "target.sizes", {**expected, "chain_sha256": "0" * 64}, tmp_path / "audit.json")
    # Pin the wrong dictionaries: hashes match, but the direction must fail.
    expected["source_sizes_sha256"], expected["target_sizes_sha256"] = expected["target_sizes_sha256"], expected["source_sizes_sha256"]
    with pytest.raises(ValueError, match="direction/assembly"):
        validate_chain_inputs(FIXTURE / "human_to_target.chain", FIXTURE / "target.sizes", FIXTURE / "source.sizes", expected, tmp_path / "audit.json")


def test_asset_manifest_rejects_wrong_assembly_and_duplicate(tmp_path: Path) -> None:
    text = (FIXTURE / "assets.tsv").read_text()
    bad = tmp_path / "assets.tsv"
    bad.write_text(text.replace("GCA_000264685.2", "wrong-build"))
    with pytest.raises(ValueError, match="version-matched"):
        read_chain_assets(bad, SPECIES)
    bad.write_text(text + text.splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_chain_assets(bad, SPECIES)


def test_request_checks_exact_source_dictionary(tmp_path: Path) -> None:
    bad = tmp_path / "source.sizes"
    bad.write_text("1\t1000\n")
    with pytest.raises(ValueError, match="human bounds"):
        prepare_chain_requests(FIXTURE / "anchors.tsv", bad, tmp_path / "r.parquet", tmp_path / "r.bed")
    bad.write_text("chr1\t500\n")
    with pytest.raises(ValueError, match="human bounds"):
        prepare_chain_requests(FIXTURE / "anchors.tsv", bad, tmp_path / "r.parquet", tmp_path / "r.bed")


def test_duplicate_chromosome_and_invalid_bed_fail(tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.write_text("chr1\t1000\nchr1\t1000\n")
    with pytest.raises(ValueError):
        read_sizes(bad)
    bad.write_text("chr1\t100\t102\tname\t0\t+\n")
    with pytest.raises(ValueError, match="center-1"):
        read_bed6(bad)


def test_empty_chain_fails_even_if_hash_matches(tmp_path: Path) -> None:
    bad = tmp_path / "empty.chain"
    bad.write_text("")
    with pytest.raises(ValueError, match="no headers"):
        validate_chain_inputs(bad, FIXTURE / "source.sizes", FIXTURE / "target.sizes", {**assets(), "chain_sha256": file_sha256(bad)}, tmp_path / "audit.json")
