"""Resolved cohorts, pinned assets, and producer identity for chain projection."""

from pathlib import Path
import json
import re

import polars as pl

from marin_dna_vertebrate_projection.manifest import (
    read_species_manifest,
)
from marin_dna_vertebrate_projection.genome_assets import read_genome_assets
from marin_dna_vertebrate_projection.projection.chains import (
    file_sha256,
    read_chain_assets,
)
from marin_dna_vertebrate_projection.provenance import (
    hash_pipeline_config,
    resolve_pipeline_commit,
    write_producer_manifest,
)

PIPELINE_VERSION = str(config["pipeline_version"])
HF_REPO_PREFIX = f"vertebrate-{PIPELINE_VERSION}"
TIER = str(config["tier"])
assert TIER in {"smoke", "full"}
VALIDATION_ROWS = int(
    config["smoke_validation_rows"] if TIER == "smoke" else config["validation_rows"]
)
assert VALIDATION_ROWS > 0
PIPELINE_COMMIT = resolve_pipeline_commit()
CHAIN_ASSETS_PATH = str(config["chain_assets"])
GENOME_ASSETS_PATH = str(config["genome_assets"])
SPECIES_SELECTED = str(config["species_selected"])
ASSETS = read_chain_assets(CHAIN_ASSETS_PATH, SPECIES_SELECTED)
GENOMES = read_genome_assets(GENOME_ASSETS_PATH, ASSETS)
RESOLVED_CONFIG = {
    **config,
    "chain_assets_sha256": file_sha256(CHAIN_ASSETS_PATH),
    "genome_assets_sha256": file_sha256(GENOME_ASSETS_PATH),
    "species_selected_sha256": file_sha256(SPECIES_SELECTED),
}
if TIER == "smoke" and not config.get("anchors"):
    RESOLVED_CONFIG["smoke_anchors_sha256"] = file_sha256(str(config["smoke_anchors"]))
PIPELINE_CONFIG_SHA256 = hash_pipeline_config(RESOLVED_CONFIG)
RESULTS = (
    f"results/{PIPELINE_VERSION}/{PIPELINE_COMMIT}/{PIPELINE_CONFIG_SHA256}/{TIER}"
)
PRODUCER_MANIFEST = f"{RESULTS}/metadata/producer.json"
ASSET_PROVENANCE = f"{RESULTS}/metadata/assets.json"

WINDOW_SIZE = int(config["window_size"])
assert WINDOW_SIZE == 255

selected_manifest = read_species_manifest(SPECIES_SELECTED)
assert selected_manifest["selected"].all()

all_mammals = selected_manifest.filter(pl.col("backend") == "zoonomia_cactus")[
    "alignment_name"
].to_list()
all_non_mammals = selected_manifest.filter(pl.col("backend") == "ucsc_multiz100way")[
    "alignment_name"
].to_list()
MAMMALS = sorted(set(ASSETS) & set(all_mammals))
NON_MAMMALS = sorted(set(ASSETS) & set(all_non_mammals))
if TIER == "smoke":
    CHROMS = list(config["smoke_chroms"])
    ANCHOR_CATALOG = str(config["smoke_anchors"])
    COHORTS = list(config["smoke_cohorts"])
else:
    assert set(ASSETS) == set(
        all_mammals + all_non_mammals
    ), "full tier requires the complete selected cohort"
    assert all(
        row["chain_origin"] != "synthetic-test-only" for row in ASSETS.values()
    ), "synthetic chains cannot build a full dataset"
    CHROMS = list(config["standard_chroms"])
    ANCHOR_CATALOG = f"{RESULTS}/anchors/catalog.parquet"
    COHORTS = list(config["region_cohorts"])


def asset_input(path):
    return storage.s3(path) if path.startswith("s3://") else local(path)


SPECIES_SELECTED_INPUT = local(SPECIES_SELECTED)
if config.get("anchors"):
    ANCHOR_CATALOG = str(config["anchors"])
    assert re.fullmatch(
        r"[0-9a-f]{64}", str(config.get("anchors_sha256", ""))
    ), "external anchors require SHA-256"
ANCHOR_CATALOG_INPUT = (
    asset_input(ANCHOR_CATALOG)
    if TIER == "smoke" or config.get("anchors")
    else ANCHOR_CATALOG
)
PROJECTION_REQUESTS = f"{RESULTS}/anchors/projection_requests.parquet"

assert set(MAMMALS) <= set(all_mammals)
assert set(NON_MAMMALS) <= set(all_non_mammals)
assert set(CHROMS) <= set(config["standard_chroms"])
assert (MAMMALS or NON_MAMMALS) and CHROMS

ACTIVE_SPECIES = MAMMALS + NON_MAMMALS
ACTIVE_MANIFEST = f"{RESULTS}/metadata/species_active.tsv"
SPECIES_RE = "|".join(re.escape(name) for name in ACTIVE_SPECIES)
GENOME_RE = "|".join(re.escape(name) for name in GENOMES)
CHROM_RE = "|".join(CHROMS)
COHORT_RE = "|".join(COHORTS)
PUBLICATION_TRAIN_SHARD_COUNT = int(
    config[
        (
            "publication_smoke_train_shards"
            if TIER == "smoke"
            else "publication_train_shards"
        )
    ]
)
PUBLICATION_VALIDATION_SHARD_COUNT = int(config["publication_validation_shards"])
PUBLICATION_SHUFFLE_SEED = int(config["publication_shuffle_seed"])
assert PUBLICATION_TRAIN_SHARD_COUNT > 0 and PUBLICATION_VALIDATION_SHARD_COUNT > 0
PUBLICATION_TRAIN_SHARDS = [
    f"shard_{i:04d}" for i in range(PUBLICATION_TRAIN_SHARD_COUNT)
]
PUBLICATION_VALIDATION_SHARDS = [
    f"shard_{i:04d}" for i in range(PUBLICATION_VALIDATION_SHARD_COUNT)
]
HF_RESULTS = f"{RESULTS}/hf"
HF_MANIFEST = f"{RESULTS}/hf_validation/hf_publication_manifest.json"

HUMAN_SEQUENCES = f"{RESULTS}/sequences/human_reference.parquet"
COMBINED_SEQUENCES = f"{RESULTS}/sequences/all_sources.parquet"


rule producer_manifest:
    output:
        PRODUCER_MANIFEST,
    run:
        write_producer_manifest(
            output[0],
            pipeline_commit=PIPELINE_COMMIT,
            config_sha256=PIPELINE_CONFIG_SHA256,
            pipeline_version=PIPELINE_VERSION,
            tier=TIER,
        )


rule asset_provenance:
    input:
        chains=local(CHAIN_ASSETS_PATH),
        genomes=local(GENOME_ASSETS_PATH),
        species=SPECIES_SELECTED_INPUT,
    output:
        ASSET_PROVENANCE,
    run:
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(output[0]).write_text(
            json.dumps(
                {
                    "pipeline_commit": PIPELINE_COMMIT,
                    "config_sha256": PIPELINE_CONFIG_SHA256,
                    "config": RESOLVED_CONFIG,
                    "chains": ASSETS,
                    "genomes": GENOMES,
                    "projector": "UCSC liftOver",
                    "projection_policy": "center_1",
                },
                indent=2,
            )
            + "\n"
        )


rule active_species_manifest:
    input:
        SPECIES_SELECTED_INPUT,
    output:
        ACTIVE_MANIFEST,
    run:
        frame = pl.read_csv(input[0], separator="\t").filter(
            pl.col("alignment_name").is_in(ACTIVE_SPECIES)
        )
        assert frame.height == len(ACTIVE_SPECIES)
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        frame.write_csv(output[0], separator="\t")
