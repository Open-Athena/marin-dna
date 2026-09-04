"""Regenerate the pinned all-vertebrate one-per-order target manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import polars as pl

from marin_dna_vertebrate_projection.cli.build_species_manifest import (
    _fetch_taxonomy,
)
from marin_dna_vertebrate_projection.manifest import validate_species_manifest
from marin_dna_vertebrate_projection.order_manifest import (
    select_order_representatives,
    validate_order_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_MANIFEST = PROJECT_ROOT / "config/species_selected.tsv"
MAMMAL_TSV = PROJECT_ROOT / "config/zoonomia_447_family_dedup.tsv"


def build_order_manifest() -> pl.DataFrame:
    """Resolve NCBI orders and apply the historical deterministic policy."""
    source = pl.read_csv(SOURCE_MANIFEST, separator="\t")
    validate_species_manifest(source)
    mammal_rows = {
        row["species"]: row
        for row in csv.DictReader(MAMMAL_TSV.open(), delimiter="\t")
    }
    scientific_names = source["scientific_name"].to_list()
    taxonomy = _fetch_taxonomy(scientific_names)

    candidates: list[dict[str, object]] = []
    for row in source.filter(pl.col("selected")).to_dicts():
        scientific_name = str(row["scientific_name"])
        classification = taxonomy[scientific_name]["taxonomy"]["classification"]
        order = str(classification["order"]["name"])
        alignment_name = str(row["alignment_name"])
        mammal = mammal_rows.get(alignment_name)
        if mammal is not None:
            assert mammal["order"] == order, (
                f"order drift for {alignment_name}: {mammal['order']} -> {order}"
            )
            quality_source = mammal["quality_source"]
        else:
            assert row["backend"] == "ucsc_multiz100way"
            quality_source = "unknown"
        candidates.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"selected", "selection_reason"}
            }
            | {"order": order, "quality_source": quality_source}
        )

    selected = select_order_representatives(pl.DataFrame(candidates))
    validate_order_manifest(selected, source)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "config/species_vertebrate_order.tsv",
    )
    args = parser.parse_args()
    manifest = build_order_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(args.output, separator="\t")
    print(
        f"wrote {manifest.height} targets spanning "
        f"{manifest['order'].n_unique()} NCBI orders"
    )


if __name__ == "__main__":
    main()
