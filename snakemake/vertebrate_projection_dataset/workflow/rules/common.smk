"""Constants and fail-fast configuration checks for issue #417."""

from pathlib import Path

import polars as pl

from marin_dna.pipelines.vertebrate_projection_dataset.manifest import (
    read_species_manifest,
)
from marin_dna.pipelines.vertebrate_projection_dataset.mirror import (
    read_mirror_manifest,
    validate_multiz_mirror_contents,
)
from marin_dna.pipelines.vertebrate_projection_dataset.provenance import (
    resolve_pipeline_commit,
)

PIPELINE_VERSION = str(config["pipeline_version"])
TIER = str(config["tier"])
assert TIER in {"smoke", "full"}
RESULTS = f"snakemake/vertebrate_projection_dataset/results/{PIPELINE_VERSION}/{TIER}"

WINDOW_SIZE = int(config["window_size"])
TARGET_LENGTH = int(config["target_length"])
PRE_RESIZE_MIN = int(config["pre_resize_min_length"])
PRE_RESIZE_MAX = int(config["pre_resize_max_length"])
assert WINDOW_SIZE == TARGET_LENGTH == 255
assert 0 < PRE_RESIZE_MIN <= TARGET_LENGTH <= PRE_RESIZE_MAX

SPECIES_CANDIDATES = str(config["species_candidates"])
SPECIES_SELECTED = str(config["species_selected"])
MIRROR_MANIFEST = str(config["multiz_mirror_manifest"])
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
if TIER == "smoke":
    MAMMALS = list(config["smoke_mammals"])
    NON_MAMMALS = list(config["smoke_non_mammals"])
    CHROMS = list(config["smoke_chroms"])
    ANCHOR_CATALOG = str(config["smoke_anchors"])
    COHORTS = ["all", "cds", "ccre_non_promoter", "background"]
else:
    MAMMALS = all_mammals
    NON_MAMMALS = all_non_mammals
    CHROMS = list(config["standard_chroms"])
    ANCHOR_CATALOG = f"{RESULTS}/anchors/catalog.parquet"
    COHORTS = list(config["region_cohorts"])

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
    f"shard_{i:04d}" for i in range(PUBLICATION_TRAIN_SHARD_COUNT)
]
PUBLICATION_VALIDATION_SHARDS = [
    f"shard_{i:04d}" for i in range(PUBLICATION_VALIDATION_SHARD_COUNT)
]
HF_RESULTS = f"{RESULTS}/hf"

HUMAN_SEQUENCES = f"{RESULTS}/sequences/human_reference.parquet"
COMBINED_SEQUENCES = f"{RESULTS}/sequences/all_sources.parquet"


rule active_species_manifest:
    input:
        SPECIES_SELECTED,
    output:
        ACTIVE_MANIFEST,
    run:
        frame = pl.read_csv(input[0], separator="\t").filter(
            pl.col("alignment_name").is_in(ACTIVE_SPECIES)
        )
        assert frame.height == len(ACTIVE_SPECIES)
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        frame.write_csv(output[0], separator="\t")
