from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl

from common import PRIMARY_CHROMS, RUN_ID
from prepare_repeat_inventory import (
    build_category_inventory,
    build_chrom_inventory,
    materialize,
    read_fai,
    read_repeatmasker,
    union_length,
)


def write_fai(path: Path) -> None:
    rows = [f"{chrom}\t100\t0\t0\t0" for chrom in PRIMARY_CHROMS]
    rows.append("MT\t50\t0\t0\t0")
    path.write_text("\n".join(rows) + "\n")


def rmsk_row(
    chrom: str,
    start: int,
    end: int,
    *,
    name: str,
    repeat_class: str,
    family: str,
    record_id: int,
) -> str:
    fields = [
        0,
        100 + record_id,
        20,
        0,
        0,
        chrom,
        start,
        end,
        -1,
        "+",
        name,
        repeat_class,
        family,
        0,
        end - start,
        0,
        record_id,
    ]
    return "\t".join(map(str, fields))


def write_rmsk(path: Path) -> None:
    rows = [
        rmsk_row(
            "chr1", 0, 10, name="L1A", repeat_class="LINE", family="L1", record_id=1
        ),
        rmsk_row(
            "chr1", 5, 15, name="AluA", repeat_class="SINE", family="Alu", record_id=2
        ),
        rmsk_row(
            "chr2", 10, 20, name="L1A", repeat_class="LINE", family="L1", record_id=3
        ),
        rmsk_row(
            "chrX",
            30,
            40,
            name="Low",
            repeat_class="Low_complexity",
            family="Low_complexity",
            record_id=4,
        ),
        rmsk_row(
            "chrUn_KI270442v1",
            0,
            10,
            name="Other",
            repeat_class="Unknown",
            family="Unknown",
            record_id=5,
        ),
    ]
    with gzip.open(path, "wt") as handle:
        handle.write("\n".join(rows) + "\n")


def test_inventory_coordinates_union_and_hierarchy(tmp_path: Path) -> None:
    fai_path = tmp_path / "reference.fa.fai"
    rmsk_path = tmp_path / "rmsk.txt.gz"
    write_fai(fai_path)
    write_rmsk(rmsk_path)
    lengths = read_fai(fai_path)
    assert set(lengths) == set(PRIMARY_CHROMS)
    annotations, source_rows = read_repeatmasker(rmsk_path, lengths)
    assert source_rows == 5
    assert annotations.height == 4
    assert annotations["chrom"].to_list() == ["1", "1", "2", "X"]
    assert annotations.select("start0", "end0").rows()[:2] == [(0, 10), (5, 15)]

    assert union_length(
        annotations.filter(pl.col("chrom") == "1")["start0"].to_numpy(),
        annotations.filter(pl.col("chrom") == "1")["end0"].to_numpy(),
    ) == (15, 1)
    chrom = build_chrom_inventory(annotations, lengths)
    assert chrom.filter(pl.col("chrom") == "1")["repeat_union_bp"].item() == 15
    assert chrom.filter(pl.col("chrom") == "2")["repeat_union_bp"].item() == 10
    assert chrom.filter(pl.col("chrom") == "X")["repeat_union_bp"].item() == 10

    categories = build_category_inventory(annotations)
    line = categories.filter(
        (pl.col("level") == "class") & (pl.col("label") == "LINE")
    ).row(0, named=True)
    assert line["record_count"] == 2
    assert line["raw_annotated_bp"] == 20
    l1a = categories.filter(
        (pl.col("level") == "subfamily") & (pl.col("label") == "LINE|L1|L1A")
    ).row(0, named=True)
    assert l1a["chrom_count"] == 2


def test_run_id_is_frozen() -> None:
    assert RUN_ID == "dna-exp435-repeat-inventory-r1"


def test_materialize_writes_hash_complete_inventory(
    tmp_path: Path, monkeypatch
) -> None:
    fai_path = tmp_path / "reference.fa.fai"
    rmsk_path = tmp_path / "rmsk.txt.gz"
    output_dir = tmp_path / "inventory"
    write_fai(fai_path)
    write_rmsk(rmsk_path)
    monkeypatch.setenv("EXPERIMENT_COMMIT", "a" * 40)
    monkeypatch.setenv("RUN_ID", RUN_ID)
    manifest = materialize(rmsk_path, fai_path, output_dir)
    assert manifest["analysis_status"] == "outcome_blind_annotation_inventory"
    assert manifest["primary_records"] == 4
    assert manifest["repeat_union_bp"] == 35
    assert set(manifest["artifacts"]) == {
        "annotations.parquet",
        "category_inventory.parquet",
        "chrom_inventory.parquet",
        "results.json",
    }
    assert (output_dir / "manifest.json").is_file()
