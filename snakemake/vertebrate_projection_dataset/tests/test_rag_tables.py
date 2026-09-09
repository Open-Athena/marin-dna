import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marin_dna_vertebrate_projection.rag.documents import Locus
from marin_dna_vertebrate_projection.rag.tables import (
    WindowStore,
    read_catalog,
    sha256_file,
    write_training_datasets,
)


def sequence_row(locus, species="hg38", sequence="A" * 255):
    return {
        "query_name": locus.query_name,
        "source_chrom": locus.chrom,
        "source_start": locus.start,
        "source_end": locus.end,
        "alignment_name": species,
        "t_start": 1000,
        "t_end": 1255,
        "t_strand": "+",
        "t_src_size": 2000,
        "pre_resize_t_start": 1127,
        "pre_resize_t_end": 1128,
        "sequence": sequence,
    }


def test_unsplit_sequence_assembly_preserves_human_only_validation_and_rc_training(
    tmp_path,
):
    train = Locus("chr1", 0, 255)
    validation = Locus("chr18", 0, 255)
    unused_chr18 = Locus("chr18", 500, 755)
    loci = {train, validation, unused_chr18}
    human = tmp_path / "human.parquet"
    pq.write_table(
        pa.Table.from_pylist([sequence_row(locus) for locus in sorted(loci)]), human
    )
    bird = tmp_path / "bird.parquet"
    pq.write_table(pa.Table.from_pylist([sequence_row(train, "bird", "c" * 255)]), bird)
    rejected = tmp_path / "rejected.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_chrom": locus.chrom,
                    "source_start": locus.start,
                    "source_end": locus.end,
                    "rejection_reason": "no_mapping",
                }
                for locus in sorted(loci - {train})
            ]
        ),
        rejected,
    )
    store = WindowStore(tmp_path / "windows.sqlite")
    try:
        assert store.ingest(human, species="hg38", accepted=True, expected=loci) == 3
        assert store.ingest(bird, species="bird", accepted=True, expected=loci) == 1
        assert (
            store.ingest(rejected, species="bird", accepted=False, expected=loci) == 2
        )
        windows, provenance = store.get(validation, {"hg38", "bird"})
        assert set(windows) == {"hg38"}
        assert provenance["bird"]["rejection_reason"] == "no_mapping"
        summary = write_training_datasets(
            {
                "cds": {
                    locus: f"original-{index}"
                    for index, locus in enumerate(sorted(loci))
                }
            },
            store,
            species={"hg38", "bird"},
            directory=tmp_path / "datasets",
            validation_rows=1,
        )
        training = pq.read_table(tmp_path / "datasets/cds/train.parquet").to_pylist()
        held_out = pq.read_table(
            tmp_path / "datasets/cds/validation.parquet"
        ).to_pylist()
        assert len(training) == 2
        assert {row["orientation"] for row in training} == {"forward", "rc"}
        assert {row["source_chrom"] for row in training} == {"chr1"}
        assert len(held_out) == 1
        assert held_out[0]["sequence"] == "A" * 255
        assert held_out[0]["species_order"] == ["hg38"]
        assert summary["regions"]["cds"]["unused_chr18_anchors"] == 1
        assert (
            json.loads((tmp_path / "datasets/split_summary.json").read_text())
            == summary
        )
    finally:
        store.close()


def test_incomplete_outcomes_and_duplicate_inputs_fail(tmp_path):
    locus = Locus("chr1", 0, 255)
    source = tmp_path / "human.parquet"
    pq.write_table(pa.Table.from_pylist([sequence_row(locus)]), source)
    store = WindowStore(tmp_path / "windows.sqlite")
    try:
        store.ingest(source, species="hg38", accepted=True, expected={locus})
        with pytest.raises(ValueError, match="incomplete species"):
            store.get(locus, {"hg38", "bird"})
        with pytest.raises(ValueError, match="duplicate locus/species"):
            store.ingest(source, species="hg38", accepted=True, expected={locus})
    finally:
        store.close()


def test_invalid_projected_center_fails_before_document_construction(tmp_path):
    locus = Locus("chr1", 0, 255)
    row = sequence_row(locus, "bird")
    row["pre_resize_t_start"] += 1
    row["pre_resize_t_end"] += 1
    source = tmp_path / "bird.parquet"
    pq.write_table(pa.Table.from_pylist([row]), source)
    store = WindowStore(tmp_path / "windows.sqlite")
    try:
        with pytest.raises(ValueError, match="center"):
            store.ingest(source, species="bird", accepted=True, expected={locus})
    finally:
        store.close()


def test_catalog_adapter_keeps_exact_membership_and_checks_pinned_bytes(tmp_path):
    path = tmp_path / "source.tsv"
    path.write_text(
        "id\tchrom\tstart\tend\tarm\na\t18\t0\t255\tncrna\nb\t1\t500\t755\tenhancer\n"
    )
    spec = {
        "columns": {
            "id": "id",
            "chrom": "chrom",
            "start": "start",
            "end": "end",
            "region": "arm",
        },
        "chrom_names": "ensembl",
        "sha256": sha256_file(path),
        "region_value": "ncrna",
        "expected_rows": 1,
    }
    assert read_catalog(path, spec) == {Locus("chr18", 0, 255): "a"}
    path.write_text(path.read_text().replace("255", "256"))
    with pytest.raises(ValueError, match="SHA-256"):
        read_catalog(path, spec)
