"""Rebuild issue #417's combined Zoonomia + MultiZ species manifests.

The Zoonomia inputs are the frozen family-deduplicated v1 cohort.  Human is
removed because it is emitted separately as ``human_reference``.  The MultiZ
inputs are every non-mammalian leaf listed by the UCSC hg38 100-way README.
NCBI Datasets taxonomy supplies stable taxon IDs and families; the actual
selection policy lives in the tested library module.
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path
from urllib.parse import quote

import polars as pl

from marin_dna_vertebrate_projection.manifest import (
    select_family_representatives,
    validate_species_manifest,
)
from marin_dna_vertebrate_projection.projection.taxonomy import normalize_zoonomia_leaf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
NCBI_TAX_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/taxonomy/taxon"
MAMMAL_TSV = PROJECT_ROOT / "config/zoonomia_447_family_dedup.tsv"

# alignment name, scientific name, UCSC assembly label, major clade,
# explicit selection priority.  Priority is only non-equal within a family;
# it pins the audited assembly choice instead of relying on runtime taxonomy.
MULTIZ_CANDIDATES: tuple[tuple[str, str, str, str, int], ...] = (
    ("falChe1", "Falco cherrug", "F_cherrug_v1.0", "birds", 10),
    ("falPer1", "Falco peregrinus", "F_peregrinus_v1.0", "birds", 5),
    ("ficAlb2", "Ficedula albicollis", "FicAlb1.5", "birds", 10),
    ("zonAlb1", "Zonotrichia albicollis", "ASM38545v1", "birds", 10),
    ("geoFor1", "Geospiza fortis", "GeoFor_1.0", "birds", 10),
    ("taeGut2", "Taeniopygia guttata", "taeGut324", "birds", 10),
    ("pseHum1", "Pseudopodoces humilis", "PseHum1.0", "birds", 10),
    ("melUnd1", "Melopsittacus undulatus", "WUSTL_v6.3", "birds", 10),
    ("amaVit1", "Amazona vittata", "AV1", "birds", 5),
    ("araMac1", "Ara macao", "SMACv1.1", "birds", 10),
    ("colLiv1", "Columba livia", "Cliv_1.0", "birds", 10),
    ("anaPla1", "Anas platyrhynchos", "BGI_duck_1.0", "birds", 10),
    ("galGal4", "Gallus gallus", "Gallus_gallus-4.0", "birds", 10),
    ("melGal1", "Meleagris gallopavo", "Turkey_2.01", "birds", 5),
    (
        "allMis1",
        "Alligator mississippiensis",
        "allMis0.2",
        "reptiles",
        10,
    ),
    ("cheMyd1", "Chelonia mydas", "CheMyd_1.0", "reptiles", 10),
    ("chrPic2", "Chrysemys picta bellii", "v3.0.3", "reptiles", 10),
    ("pelSin1", "Pelodiscus sinensis", "PelSin_1.0", "reptiles", 10),
    ("apaSpi1", "Apalone spinifera", "ASM38561v1", "reptiles", 5),
    ("anoCar2", "Anolis carolinensis", "AnoCar2.0", "reptiles", 10),
    ("xenTro7", "Xenopus tropicalis", "JGI_7.0", "amphibians", 10),
    ("latCha1", "Latimeria chalumnae", "Broad", "lobe-finned_fish", 10),
    ("tetNig2", "Tetraodon nigroviridis", "Genoscope_8.0", "ray-finned_fish", 5),
    ("fr3", "Takifugu rubripes", "FUGU5", "ray-finned_fish", 10),
    ("takFla1", "Takifugu flavidus", "Takifugu_flavidus_v1", "ray-finned_fish", 5),
    ("oreNil2", "Oreochromis niloticus", "oreNil1.1", "ray-finned_fish", 10),
    ("neoBri1", "Neolamprologus brichardi", "NeoBri1.0", "ray-finned_fish", 5),
    ("hapBur1", "Haplochromis burtoni", "AstBur1.0", "ray-finned_fish", 5),
    ("mayZeb1", "Maylandia zebra", "MetZeb1.1", "ray-finned_fish", 5),
    ("punNye1", "Pundamilia nyererei", "PunNye1.0", "ray-finned_fish", 5),
    ("oryLat2", "Oryzias latipes", "MEDAKA1", "ray-finned_fish", 10),
    ("xipMac1", "Xiphophorus maculatus", "X_maculatus-4.4.2", "ray-finned_fish", 10),
    ("gasAcu1", "Gasterosteus aculeatus", "Broad", "ray-finned_fish", 10),
    ("gadMor1", "Gadus morhua", "GadMor_May2010", "ray-finned_fish", 10),
    ("danRer10", "Danio rerio", "Zv10", "ray-finned_fish", 10),
    (
        "astMex1",
        "Astyanax mexicanus",
        "Astyanax_mexicanus-1.0.2",
        "ray-finned_fish",
        10,
    ),
    ("lepOcu1", "Lepisosteus oculatus", "LepOcu1", "ray-finned_fish", 10),
    ("petMar2", "Petromyzon marinus", "WUGSC_7.0", "jawless_vertebrates", 10),
)

CLADE_RANK = {
    "mammals": 1,
    "birds": 2,
    "reptiles": 3,
    "amphibians": 4,
    "lobe-finned_fish": 5,
    "ray-finned_fish": 6,
    "jawless_vertebrates": 7,
}


def _fetch_taxonomy(queries: list[str]) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for start in range(0, len(queries), 50):
        batch = queries[start : start + 50]
        encoded = ",".join(quote(query, safe="") for query in batch)
        url = f"{NCBI_TAX_URL}/{encoded}/dataset_report?page_size={len(batch) + 5}"
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
        for report in payload.get("reports", []):
            for query in report.get("query", []):
                reports[str(query)] = report
    missing = set(queries) - set(reports)
    assert not missing, f"NCBI taxonomy did not resolve: {sorted(missing)}"
    return reports


def _taxonomy_fields(report: dict) -> tuple[str, int, str]:
    taxonomy = report["taxonomy"]
    scientific_name = str(taxonomy["current_scientific_name"]["name"])
    taxonomy_id = int(taxonomy["tax_id"])
    family = str(taxonomy["classification"]["family"]["name"])
    assert scientific_name and taxonomy_id > 0 and family
    return scientific_name, taxonomy_id, family


def build_manifest() -> pl.DataFrame:
    """Resolve pinned inputs and return the fully annotated candidate table."""
    with MAMMAL_TSV.open() as handle:
        mammal_rows = list(csv.DictReader(handle, delimiter="\t"))
    mammal_rows = [row for row in mammal_rows if row["species"] != "Homo_sapiens"]
    mammal_queries = {
        row["species"]: normalize_zoonomia_leaf(row["species"]).replace("_", " ")
        for row in mammal_rows
    }
    multiz_queries = [row[1] for row in MULTIZ_CANDIDATES]
    taxonomy = _fetch_taxonomy(
        sorted(set(mammal_queries.values()) | set(multiz_queries))
    )

    candidates: list[dict[str, object]] = []
    for row in mammal_rows:
        query = mammal_queries[row["species"]]
        scientific_name, taxonomy_id, family = _taxonomy_fields(taxonomy[query])
        assert family == row["family"], (
            f"family drift for {row['species']}: {row['family']} -> {family}"
        )
        candidates.append(
            {
                "alignment_name": row["species"],
                "scientific_name": scientific_name,
                "assembly": row["accession"],
                "taxonomy_id": taxonomy_id,
                "family": family,
                "clade": "mammals",
                "phylogenetic_rank": CLADE_RANK["mammals"],
                "backend": "zoonomia_cactus",
                "selection_priority": 10,
                "assembly_level": row["assembly_level"] or "unknown",
                "contig_n50": int(row["contig_n50"] or 0),
            }
        )
    for alignment_name, query, assembly, clade, priority in MULTIZ_CANDIDATES:
        scientific_name, taxonomy_id, family = _taxonomy_fields(taxonomy[query])
        candidates.append(
            {
                "alignment_name": alignment_name,
                "scientific_name": scientific_name,
                "assembly": f"{assembly}/{alignment_name}",
                "taxonomy_id": taxonomy_id,
                "family": family,
                "clade": clade,
                "phylogenetic_rank": CLADE_RANK[clade],
                "backend": "ucsc_multiz100way",
                "selection_priority": priority,
                # UCSC's 2015 README does not publish comparable assembly
                # level/N50 fields.  The explicit priority above pins audited
                # within-family choices; these unknown values are honest.
                "assembly_level": "unknown",
                "contig_n50": 0,
            }
        )
    manifest = select_family_representatives(pl.DataFrame(candidates))
    validate_species_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = PROJECT_ROOT / "config"
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=default_dir / "species_candidates.tsv",
    )
    parser.add_argument(
        "--selected-output",
        type=Path,
        default=default_dir / "species_selected.tsv",
    )
    args = parser.parse_args()
    manifest = build_manifest()
    args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(args.candidate_output, separator="\t")
    manifest.filter(pl.col("selected")).write_csv(args.selected_output, separator="\t")
    print(
        f"wrote {manifest.height} candidates and "
        f"{manifest.filter(pl.col('selected')).height} selected targets"
    )


if __name__ == "__main__":
    main()
