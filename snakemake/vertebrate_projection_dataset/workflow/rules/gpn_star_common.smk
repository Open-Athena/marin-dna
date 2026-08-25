"""Constants and fail-fast checks for the issue #517 GPN-Star-P grid."""

from pathlib import Path

import polars as pl

from marin_dna_vertebrate_projection.gpn_star_anchors import (
    GPN_ARMS,
    GPN_ASSIGNMENT_RECIPE,
    read_gpn_entropy_manifest,
)
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
TIER = str(config["tier"])
assert TIER in {"smoke", "full"}
PIPELINE_COMMIT = resolve_pipeline_commit()
PIPELINE_CONFIG_SHA256 = hash_pipeline_config(config)
RESULTS = (
    f"results/{PIPELINE_VERSION}/{PIPELINE_COMMIT}/{PIPELINE_CONFIG_SHA256}/{TIER}"
)
PRODUCER_MANIFEST = f"{RESULTS}/metadata/producer.json"

WINDOW_SIZE = int(config["window_size"])
STEP_SIZE = int(config["step_size"])
assert WINDOW_SIZE == 255 and STEP_SIZE == 128
assert str(config["gpn_assignment_recipe"]) == GPN_ASSIGNMENT_RECIPE
assert tuple(config["assignment_arms"]) == GPN_ARMS

FULL_CHROMS = list(config["standard_chroms"])
CHROMS = list(config["smoke_chroms"] if TIER == "smoke" else FULL_CHROMS)
assert set(CHROMS) <= set(FULL_CHROMS) and CHROMS
CHROM_RE = "|".join(CHROMS)

GPN_ENTROPY_MANIFEST = str(config["gpn_entropy_manifest"])
GPN_ENTROPY_MANIFEST_INPUT = local(GPN_ENTROPY_MANIFEST)
gpn_entropy_shards = read_gpn_entropy_manifest(GPN_ENTROPY_MANIFEST)
assert set(gpn_entropy_shards) == set(FULL_CHROMS)
GPN_DATASET_REPO = str(config["gpn_dataset_repo"])
GPN_DATASET_REVISION = str(config["gpn_dataset_revision"])
GPN_SCORE_SET = str(config["gpn_score_set"])
GPN_ENTROPY_CUTOFF = float(config["gpn_entropy_cutoff"])
GPN_MIN_SELECTED_BASES = int(config["gpn_min_selected_bases"])
assert GPN_DATASET_REPO == "songlab/gpn-star-scores"
assert GPN_SCORE_SET == "gpn-star-hg38-p243-200m"
assert len(GPN_DATASET_REVISION) == 40
assert 0.0 < GPN_ENTROPY_CUTOFF < 1.0
assert GPN_MIN_SELECTED_BASES == 51

SPECIES_SELECTED = str(config["species_selected"])
MIRROR_MANIFEST = str(config["multiz_mirror_manifest"])
TWOBIT_MANIFEST = str(config["twobit_manifest"])
mirror_objects = read_mirror_manifest(MIRROR_MANIFEST)
validate_multiz_mirror_contents(mirror_objects, FULL_CHROMS)
selected_manifest = read_species_manifest(SPECIES_SELECTED)
assert selected_manifest["selected"].all()

all_mammals = selected_manifest.filter(pl.col("backend") == "zoonomia_cactus")[
    "alignment_name"
].to_list()
all_non_mammals = selected_manifest.filter(
    pl.col("backend") == "ucsc_multiz100way"
)["alignment_name"].to_list()
twobit_objects = read_twobit_manifest(TWOBIT_MANIFEST)
validate_twobit_manifest(twobit_objects, ["hg38", *all_non_mammals])
if TIER == "smoke":
    MAMMALS = list(config["smoke_mammals"])
    NON_MAMMALS = list(config["smoke_non_mammals"])
else:
    MAMMALS = all_mammals
    NON_MAMMALS = all_non_mammals

assert set(MAMMALS) <= set(all_mammals)
assert set(NON_MAMMALS) <= set(all_non_mammals)
assert MAMMALS and NON_MAMMALS

SPECIES_SELECTED_INPUT = local(SPECIES_SELECTED)
MIRROR_MANIFEST_INPUT = local(MIRROR_MANIFEST)
TWOBIT_MANIFEST_INPUT = local(TWOBIT_MANIFEST)
ANCHOR_CATALOG = f"{RESULTS}/anchors/catalog.parquet"
ANCHOR_CATALOG_INPUT = ANCHOR_CATALOG
ASSIGNMENTS = f"{RESULTS}/anchors/assignments.parquet"
PROJECTION_REQUESTS = f"{RESULTS}/anchors/projection_requests.parquet"

ACTIVE_SPECIES = MAMMALS + NON_MAMMALS
ACTIVE_MANIFEST = f"{RESULTS}/metadata/species_active.tsv"
HAL_PATH = str(config["hal_stage_path"])
HAL_VALIDATION = f"{RESULTS}/metadata/hal_stage_validated.txt"
MULTIZ_STAGE_DIR = str(config["multiz_stage_dir"])
MAMMAL_RE = "|".join(MAMMALS)
NON_MAMMAL_RE = "|".join(NON_MAMMALS)

HUMAN_SEQUENCES = f"{RESULTS}/sequences/human_reference.parquet"
COMBINED_SEQUENCES = f"{RESULTS}/sequences/all_sources.parquet"


rule gpn_producer_manifest:
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


rule gpn_active_species_manifest:
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
