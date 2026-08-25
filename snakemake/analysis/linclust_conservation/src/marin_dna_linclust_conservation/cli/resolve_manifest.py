"""Resolve a pinned order-deduplicated manifest from NCBI JSONL reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from marin_dna_linclust_conservation.manifest import (
    MANIFEST_COLUMNS,
    parse_ncbi_jsonl,
    select_order_representatives,
    validate_pinned_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genomes", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path)
    parser.add_argument("--datasets-version", required=True)
    parser.add_argument("--retrieved-at", required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--selected-output", type=Path, required=True)
    args = parser.parse_args()

    candidates = parse_ncbi_jsonl(
        args.genomes,
        args.taxonomy,
        datasets_version=args.datasets_version,
        retrieved_at=args.retrieved_at,
    )
    selected = select_order_representatives(candidates)
    if args.source_inventory is not None:
        inventory = pl.read_csv(args.source_inventory, separator="\t")
        inventory_columns = {
            "accession",
            "download_uri",
            "source_checksum_type",
            "source_checksum",
            "source_size_bytes",
        }
        assert inventory_columns.issubset(inventory.columns), sorted(
            inventory_columns - set(inventory.columns)
        )
        assert inventory["accession"].n_unique() == inventory.height
        selected = selected.drop(
            "download_uri",
            "source_checksum_type",
            "source_checksum",
            "source_size_bytes",
        ).join(inventory, on="accession", how="left", validate="1:1")
    selected = selected.select(MANIFEST_COLUMNS)
    if args.source_inventory is not None:
        validate_pinned_manifest(selected)

    args.candidates_output.parent.mkdir(parents=True, exist_ok=True)
    args.selected_output.parent.mkdir(parents=True, exist_ok=True)
    selected.write_csv(args.candidates_output, separator="\t")
    selected.filter(pl.col("selected")).write_csv(
        args.selected_output,
        separator="\t",
    )
    print(
        f"wrote {selected.height} candidates and "
        f"{selected.filter(pl.col('selected')).height} selected assemblies"
        + (
            " with pinned sources"
            if args.source_inventory is not None
            else " provisionally"
        )
    )


if __name__ == "__main__":
    main()
