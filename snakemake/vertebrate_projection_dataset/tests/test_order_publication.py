from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from marin_dna_vertebrate_projection.order_manifest import read_order_manifest
from marin_dna_vertebrate_projection.order_publication import (
    TARGET_SELECTION,
    write_order_dataset_card,
    write_order_dataset_split_files,
    write_order_source_audit,
)


def _paths() -> tuple[Path, Path]:
    project = Path(__file__).resolve().parents[1]
    return (
        project / "config/species_vertebrate_order.tsv",
        project / "config/species_selected.tsv",
    )


def _combined_rows() -> pl.DataFrame:
    order_path, source_path = _paths()
    manifest = read_order_manifest(order_path, source_path)
    sources = [
        {
            "alignment_name": "hg38",
            "species": "Homo sapiens",
            "alignment_source": "human_reference",
        }
    ] + [
        {
            "alignment_name": row["alignment_name"],
            "species": row["scientific_name"],
            "alignment_source": row["backend"],
        }
        for row in manifest.to_dicts()
    ]
    # This source-family row must be excluded because human occupies Primates.
    sources.append(
        {
            "alignment_name": "Microcebus_murinus",
            "species": "Microcebus murinus",
            "alignment_source": "zoonomia_cactus",
        }
    )
    rows: list[dict[str, object]] = []
    for anchor_index in range(2):
        for source_index, source in enumerate(sources):
            start = anchor_index * 10_000 + source_index * 300
            rows.append(
                {
                    "query_name": f"enhancer_{anchor_index}_{source_index}",
                    "source_chrom": "chr1",
                    "source_start": start,
                    "source_end": start + 255,
                    "species": source["species"],
                    "alignment_name": source["alignment_name"],
                    "alignment_source": source["alignment_source"],
                    "region_label": "enhancer",
                    "sequence": "ACgt" + "A" * 251,
                }
            )
    return pl.DataFrame(rows)


def test_order_split_filters_nonhuman_primates_before_sampling(tmp_path: Path) -> None:
    order_path, source_path = _paths()
    combined = tmp_path / "combined.parquet"
    _combined_rows().write_parquet(combined)

    write_order_dataset_split_files(
        combined,
        order_path,
        source_path,
        tmp_path / "train.parquet",
        tmp_path / "validation.parquet",
        tmp_path / "selection.tsv",
        tmp_path / "composition.tsv",
        tmp_path / "summary.json",
        add_rc=True,
        validation_rows=4,
        seed=517,
    )

    train = pl.read_parquet(tmp_path / "train.parquet")
    validation = pl.read_parquet(tmp_path / "validation.parquet")
    assert "Microcebus_murinus" not in set(train["alignment_name"])
    assert "Microcebus_murinus" not in set(validation["alignment_name"])
    assert set(train["alignment_name"]) >= {"hg38", "Mus_musculus", "galGal4"}
    assert train.height == 152
    assert validation.height == 4
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["source_rows"] == 80
    assert summary["target_selection"] == TARGET_SELECTION


def test_order_audit_and_card_record_whole_dataset_contract(tmp_path: Path) -> None:
    order_path, source_path = _paths()
    combined = tmp_path / "combined.parquet"
    _combined_rows().write_parquet(combined)
    audit_path = tmp_path / "audit.json"
    write_order_source_audit(combined, order_path, source_path, audit_path)
    audit = json.loads(audit_path.read_text())
    assert audit["candidate_region_rows"] == 82
    assert audit["source_rows"] == 80
    assert audit["order_manifest_targets"] == 39
    assert audit["represented_orders_including_human"] == 40
    assert audit["human_is_sole_primates_source"] is True

    write_order_dataset_split_files(
        combined,
        order_path,
        source_path,
        tmp_path / "train.parquet",
        tmp_path / "validation.parquet",
        tmp_path / "selection.tsv",
        tmp_path / "composition.tsv",
        tmp_path / "summary.json",
        add_rc=True,
        validation_rows=4,
        seed=517,
    )
    card_path = tmp_path / "README.md"
    write_order_dataset_card(
        tmp_path / "train.parquet",
        tmp_path / "validation.parquet",
        order_path,
        source_path,
        card_path,
        pipeline_commit="a" * 40,
        hf_repo="marin-dna/phylop-uniform-v1-enhancer-arm-a-vertebrate-order",
        validation_seed=517,
        source_pipeline_commit="b" * 40,
        source_config_sha256="c" * 64,
    )
    card = card_path.read_text()
    assert "human is the sole Primates source" in card
    assert "39 non-human projection targets" in card
    assert "exactly one sequence source per represented NCBI order" in card
    assert "license: openmdw-1.1" in card
