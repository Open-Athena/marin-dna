"""Constants and fail-fast configuration checks for issue #517."""

from pathlib import Path

import polars as pl

from marin_dna_vertebrate_projection.functional_anchors import FUNCTIONAL_ARMS
from marin_dna_vertebrate_projection.manifest import read_species_manifest
from marin_dna_vertebrate_projection.mirror import (
    read_mirror_manifest,
    validate_multiz_mirror_contents,
)
from marin_dna_vertebrate_projection.provenance import (
    hash_pipeline_config,
    resolve_pipeline_commit,
    write_producer_manifest,
)
from marin_dna_vertebrate_projection.sequence_sources import (
    read_twobit_manifest,
    validate_twobit_manifest,
)

PIPELINE_VERSION = str(config["pipeline_version"])
HF_REPO_PREFIX = str(config["hf_repo_prefix"])
TIER = str(config["tier"])
assert TIER in {"smoke", "full"}
VALIDATION_ROWS = int(
    config["smoke_validation_rows"] if TIER == "smoke" else config["validation_rows"]
)
assert VALIDATION_ROWS > 0
PIPELINE_COMMIT = resolve_pipeline_commit()
PIPELINE_CONFIG_SHA256 = hash_pipeline_config(config)
RESULTS = (
    f"results/{PIPELINE_VERSION}/{PIPELINE_COMMIT}/{PIPELINE_CONFIG_SHA256}/{TIER}"
)
PRODUCER_MANIFEST = f"{RESULTS}/metadata/producer.json"

WINDOW_SIZE = int(config["window_size"])
assert WINDOW_SIZE == 255
assert tuple(config["functional_arms"]) == FUNCTIONAL_ARMS
COHORTS = list(FUNCTIONAL_ARMS)

SPECIES_CANDIDATES = str(config["species_candidates"])
SPECIES_SELECTED = str(config["species_selected"])
MIRROR_MANIFEST = str(config["multiz_mirror_manifest"])
TWOBIT_MANIFEST = str(config["twobit_manifest"])
mirror_objects = read_mirror_manifest(MIRROR_MANIFEST)
validate_multiz_mirror_contents(mirror_objects, list(config["standard_chroms"]))
candidate_manifest = read_species_manifest(SPECIES_CANDIDATES)
selected_manifest = read_species_manifest(SPECIES_SELECTED)
assert selected_manifest["selected"].all()

all_mammals = selected_manifest.filter(pl.col("backend") == "zoonomia_cactus")[
    "alignment_name"
].to_list()
all_non_mammals = selected_manifest.filter(pl.col("backend") == "ucsc_multiz100way")[
    "alignment_name"
].to_list()
twobit_objects = read_twobit_manifest(TWOBIT_MANIFEST)
validate_twobit_manifest(twobit_objects, ["hg38", *all_non_mammals])
if TIER == "smoke":
    MAMMALS = list(config["smoke_mammals"])
    NON_MAMMALS = list(config["smoke_non_mammals"])
    CHROMS = list(config["smoke_chroms"])
else:
    MAMMALS = all_mammals
    NON_MAMMALS = all_non_mammals
    CHROMS = list(config["standard_chroms"])

SPECIES_SELECTED_INPUT = local(SPECIES_SELECTED)
MIRROR_MANIFEST_INPUT = local(MIRROR_MANIFEST)
TWOBIT_MANIFEST_INPUT = local(TWOBIT_MANIFEST)
ANCHOR_CATALOG = f"{RESULTS}/anchors/projection.parquet"
ANCHOR_CATALOG_INPUT = ANCHOR_CATALOG
TRAINING_CATALOG = f"{RESULTS}/anchors/training.parquet"
DEFERRED_CATALOG = f"{RESULTS}/anchors/deferred.parquet"
PROJECTION_REQUESTS = f"{RESULTS}/anchors/projection_requests.parquet"

assert set(MAMMALS) <= set(all_mammals)
assert set(NON_MAMMALS) <= set(all_non_mammals)
assert set(CHROMS) <= set(config["standard_chroms"])
assert MAMMALS and NON_MAMMALS and CHROMS

ACTIVE_SPECIES = MAMMALS + NON_MAMMALS
ACTIVE_MANIFEST = f"{RESULTS}/metadata/species_active.tsv"
HAL_PATH = str(config["hal_stage_path"])
HAL_VALIDATION = f"{RESULTS}/metadata/hal_stage_validated.txt"
MULTIZ_STAGE_DIR = str(config["multiz_stage_dir"])
MAMMAL_RE = "|".join(MAMMALS)
NON_MAMMAL_RE = "|".join(NON_MAMMALS)
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
    f"shard_{index:04d}" for index in range(PUBLICATION_TRAIN_SHARD_COUNT)
]
PUBLICATION_VALIDATION_SHARDS = [
    f"shard_{index:04d}" for index in range(PUBLICATION_VALIDATION_SHARD_COUNT)
]
HF_RESULTS = f"{RESULTS}/hf"
HF_MANIFEST = f"{RESULTS}/hf_validation/hf_publication_manifest.json"

HUMAN_SEQUENCES = f"{RESULTS}/sequences/human_reference.parquet"
COMBINED_SEQUENCES = f"{RESULTS}/sequences/all_sources.parquet"
TRAINING_SEQUENCES = f"{RESULTS}/sequences/training_eligible.parquet"


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
