"""Fail-fast constants for issue #517 GPN-Star-P dataset publication."""

from pathlib import Path

from marin_dna_vertebrate_projection.gpn_star_anchors import GPN_ARMS
from marin_dna_vertebrate_projection.provenance import (
    hash_pipeline_config,
    resolve_pipeline_commit,
    write_producer_manifest,
)

PIPELINE_VERSION = str(config["pipeline_version"])
TIER = str(config["tier"])
assert TIER == "full"
PIPELINE_COMMIT = resolve_pipeline_commit()
PIPELINE_CONFIG_SHA256 = hash_pipeline_config(config)
RESULTS = (
    f"results/{PIPELINE_VERSION}/{PIPELINE_COMMIT}/{PIPELINE_CONFIG_SHA256}/{TIER}"
)
PRODUCER_MANIFEST = f"{RESULTS}/metadata/producer.json"

SOURCE_PIPELINE_VERSION = str(config["source_pipeline_version"])
SOURCE_PIPELINE_COMMIT = str(config["source_pipeline_commit"])
SOURCE_CONFIG_SHA256 = str(config["source_config_sha256"])
SOURCE_TIER = str(config["source_tier"])
assert len(SOURCE_PIPELINE_COMMIT) == 40
assert len(SOURCE_CONFIG_SHA256) == 64
assert SOURCE_TIER == "full"
SOURCE_RESULTS = (
    f"results/{SOURCE_PIPELINE_VERSION}/{SOURCE_PIPELINE_COMMIT}/"
    f"{SOURCE_CONFIG_SHA256}/{SOURCE_TIER}"
)
SOURCE_PRODUCER_MANIFEST = f"{SOURCE_RESULTS}/metadata/producer.json"
COMBINED_SEQUENCES = f"{SOURCE_RESULTS}/sequences/all_sources.parquet"
ACTIVE_MANIFEST = f"{SOURCE_RESULTS}/metadata/species_active.tsv"
SOURCE_ROWS = int(config["source_rows"])
SOURCE_ARM_ROWS = {
    str(cohort): int(rows) for cohort, rows in config["source_arm_rows"].items()
}

COHORTS = list(config["region_cohorts"])
assert tuple(COHORTS) == GPN_ARMS
assert set(SOURCE_ARM_ROWS) == set(COHORTS)
assert sum(SOURCE_ARM_ROWS.values()) == SOURCE_ROWS
COHORT_RE = "|".join(COHORTS)
VALIDATION_ROWS = int(config["validation_rows"])
assert VALIDATION_ROWS == 16_384

HF_OWNER = str(config["hf_owner"])
HF_REPO_NAMES = {
    str(cohort): str(repo_name) for cohort, repo_name in config["hf_repo_names"].items()
}
assert set(HF_REPO_NAMES) == set(COHORTS)
assert len(set(HF_REPO_NAMES.values())) == len(COHORTS)
assert all("/" not in repo_name for repo_name in HF_REPO_NAMES.values())


def gpn_hf_repo(cohort):
    assert cohort in HF_REPO_NAMES
    return f"{HF_OWNER}/{HF_REPO_NAMES[cohort]}"


PUBLICATION_TRAIN_SHARD_COUNT = int(config["publication_train_shards"])
PUBLICATION_VALIDATION_SHARD_COUNT = int(config["publication_validation_shards"])
PUBLICATION_SHUFFLE_SEED = int(config["publication_shuffle_seed"])
assert PUBLICATION_TRAIN_SHARD_COUNT > 0
assert PUBLICATION_VALIDATION_SHARD_COUNT > 0
PUBLICATION_TRAIN_SHARDS = [
    f"shard_{index:04d}" for index in range(PUBLICATION_TRAIN_SHARD_COUNT)
]
PUBLICATION_VALIDATION_SHARDS = [
    f"shard_{index:04d}" for index in range(PUBLICATION_VALIDATION_SHARD_COUNT)
]
HF_RESULTS = f"{RESULTS}/hf"
HF_MANIFEST = f"{RESULTS}/hf_validation/hf_publication_manifest.json"


rule gpn_publication_producer_manifest:
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
