"""Additive whole-genome HAL-chain generation and validation for issue #523."""

from pathlib import Path

from marin_dna_vertebrate_projection.mirror import (
    s3_object_size,
    stage_hal_object,
    verify_hal_object,
)
from marin_dna_vertebrate_projection.projection.hal_chains import (
    run_hal_to_chain,
    run_liftover_benchmark,
    write_chain_parity_audit,
    write_hal_genome_assets,
    write_uniform_grid_center_bed,
)
from marin_dna_vertebrate_projection.provenance import (
    hash_pipeline_config,
    resolve_pipeline_commit,
    write_producer_manifest,
)


PIPELINE_VERSION = str(config["pipeline_version"])
TIER = str(config["tier"])
assert TIER == "full"
assert str(config["cactus_version"]) == "3.3.0"
PIPELINE_COMMIT = resolve_pipeline_commit()
PIPELINE_CONFIG_SHA256 = hash_pipeline_config(config)
RESULTS = (
    f"results/{PIPELINE_VERSION}/{PIPELINE_COMMIT}/{PIPELINE_CONFIG_SHA256}/{TIER}"
)
PRODUCER_MANIFEST = f"{RESULTS}/metadata/producer.json"

SOURCE_GENOME = str(config["source_genome"])
assert SOURCE_GENOME == "Homo_sapiens"
PILOT_SPECIES = list(config["pilot_species"])
assert PILOT_SPECIES == ["Papio_anubis", "Mus_musculus", "Loxodonta_africana"]
CHAIN_RECIPES = list(config["chain_recipes"])
assert CHAIN_RECIPES == ["default", "no_dupes"]
RECIPE_NO_DUPES = {"default": False, "no_dupes": True}
LINEAR_GAP = str(config["linear_gap"])
assert LINEAR_GAP == "medium"
GENOMES = [SOURCE_GENOME, *PILOT_SPECIES]
GENOME_RE = "|".join(GENOMES)
SPECIES_RE = "|".join(PILOT_SPECIES)
RECIPE_RE = "|".join(CHAIN_RECIPES)

HAL_PATH = str(config["hal_stage_path"])
HAL_VALIDATION = str(config["hal_validation_path"])
BASELINE_ROOT = str(config["strict_phylop_baseline_root"])
VALIDATION_BED = f"{BASELINE_ROOT}/hal/input.bed"
VALIDATION_EXPECTED_QUERIES = int(config["validation_expected_queries"])
GRID_CHROM_SIZES = f"{BASELINE_ROOT}/anchors/chrom.sizes"
GRID_UNDEFINED_BED = f"{BASELINE_ROOT}/anchors/undefined.ucsc.bed"
GRID_EXPECTED_QUERIES = int(config["uniform_grid_expected_queries"])
WINDOW_SIZE = int(config["window_size"])
STEP_SIZE = int(config["step_size"])
STANDARD_CHROMS = list(config["standard_chroms"])
assert GRID_EXPECTED_QUERIES == 22_948_560
assert WINDOW_SIZE == 255 and STEP_SIZE == 128 and len(STANDARD_CHROMS) == 24

CHAIN_PATH = f"{RESULTS}/chains/{{species}}/{{recipe}}.human_to_species.chain.gz"
CHAIN_METRICS = f"{RESULTS}/chains/{{species}}/{{recipe}}.generation.json"
VALIDATION_MAPPED = f"{RESULTS}/validation/{{species}}/{{recipe}}.mapped.bed"
VALIDATION_UNMAPPED = f"{RESULTS}/validation/{{species}}/{{recipe}}.unmapped.bed"
VALIDATION_METRICS = f"{RESULTS}/validation/{{species}}/{{recipe}}.liftover.json"
PARITY_SUMMARY = f"{RESULTS}/validation/{{species}}/{{recipe}}.parity.json"
PARITY_DISCREPANCIES = (
    f"{RESULTS}/validation/{{species}}/{{recipe}}.discrepancies.parquet"
)
FULL_GRID_BED = f"{RESULTS}/full_grid/input.center1.bed"
FULL_GRID_MAPPED = f"{RESULTS}/full_grid/{{species}}.mapped.bed"
FULL_GRID_UNMAPPED = f"{RESULTS}/full_grid/{{species}}.unmapped.bed"
FULL_GRID_METRICS = f"{RESULTS}/full_grid/{{species}}.liftover.json"


rule hal_chain_pilot_producer_manifest:
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


rule stage_hal_for_chain_pilot:
    output:
        local(HAL_PATH),
    resources:
        mem_mb=1000,
    params:
        source=str(config["hal_s3_uri"]),
    run:
        stage_hal_object(params.source, output[0])


rule validate_hal_for_chain_pilot:
    input:
        local(HAL_PATH),
    output:
        local(HAL_VALIDATION),
    resources:
        mem_mb=1000,
    params:
        source=str(config["hal_s3_uri"]),
    run:
        expected_size = s3_object_size(params.source)
        verify_hal_object(input[0], expected_size=expected_size)
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(output[0]).write_text(
            f"source={params.source}\nbytes={expected_size}\n"
            f"required_genome={SOURCE_GENOME}\n"
        )


rule prepare_hal_chain_genome_assets:
    input:
        hal=local(HAL_PATH),
        validation=local(HAL_VALIDATION),
    output:
        sizes=temp(local(f"{RESULTS}/genomes/{{genome}}.chrom.sizes")),
        bed=temp(local(f"{RESULTS}/genomes/{{genome}}.whole_genome.bed")),
        twobit=temp(local(f"{RESULTS}/genomes/{{genome}}.2bit")),
    wildcard_constraints:
        genome=GENOME_RE,
    conda:
        "../envs/bioinformatics.yaml"
    resources:
        mem_mb=8000,
        genome_assets=1,
    run:
        write_hal_genome_assets(
            input.hal,
            wildcards.genome,
            output.sizes,
            output.bed,
            output.twobit,
        )


rule generate_human_to_mammal_chain:
    input:
        hal=local(HAL_PATH),
        validation=local(HAL_VALIDATION),
        query_sizes=local(f"{RESULTS}/genomes/{{species}}.chrom.sizes"),
        query_bed=local(f"{RESULTS}/genomes/{{species}}.whole_genome.bed"),
        query_twobit=local(f"{RESULTS}/genomes/{{species}}.2bit"),
        target_sizes=local(f"{RESULTS}/genomes/{SOURCE_GENOME}.chrom.sizes"),
        target_twobit=local(f"{RESULTS}/genomes/{SOURCE_GENOME}.2bit"),
    output:
        chain=CHAIN_PATH,
        metrics=CHAIN_METRICS,
    wildcard_constraints:
        species=SPECIES_RE,
        recipe=RECIPE_RE,
    resources:
        mem_mb=170000,
        chain_generation=1,
    run:
        run_hal_to_chain(
            hal_path=input.hal,
            query_genome=wildcards.species,
            query_bed=input.query_bed,
            query_chrom_sizes=input.query_sizes,
            query_twobit=input.query_twobit,
            target_genome=SOURCE_GENOME,
            target_chrom_sizes=input.target_sizes,
            target_twobit=input.target_twobit,
            output_chain=output.chain,
            output_metrics=output.metrics,
            no_dupes=RECIPE_NO_DUPES[wildcards.recipe],
            linear_gap=LINEAR_GAP,
        )


rule validation_chain_liftover:
    input:
        bed=VALIDATION_BED,
        chain=CHAIN_PATH,
    output:
        mapped=temp(local(VALIDATION_MAPPED)),
        unmapped=temp(local(VALIDATION_UNMAPPED)),
        metrics=VALIDATION_METRICS,
    wildcard_constraints:
        species=SPECIES_RE,
        recipe=RECIPE_RE,
    conda:
        "../envs/liftover.yaml"
    resources:
        mem_mb=8000,
    run:
        run_liftover_benchmark(
            input_bed=input.bed,
            chain_path=input.chain,
            mapped_bed=output.mapped,
            unmapped_bed=output.unmapped,
            metrics_path=output.metrics,
            expected_queries=VALIDATION_EXPECTED_QUERIES,
        )


rule audit_chain_parity:
    input:
        queries=VALIDATION_BED,
        direct=lambda wc: f"{BASELINE_ROOT}/hal/raw/{wc.species}.bed",
        chain=local(VALIDATION_MAPPED),
    output:
        summary=PARITY_SUMMARY,
        discrepancies=PARITY_DISCREPANCIES,
    wildcard_constraints:
        species=SPECIES_RE,
        recipe=RECIPE_RE,
    resources:
        mem_mb=24000,
    run:
        write_chain_parity_audit(
            input_bed=input.queries,
            direct_bed=input.direct,
            chain_bed=input.chain,
            summary_path=output.summary,
            discrepancies_path=output.discrepancies,
            expected_queries=VALIDATION_EXPECTED_QUERIES,
        )


rule prepare_full_uniform_grid_centers:
    input:
        sizes=GRID_CHROM_SIZES,
        undefined=GRID_UNDEFINED_BED,
    output:
        temp(local(FULL_GRID_BED)),
    resources:
        mem_mb=2000,
    run:
        write_uniform_grid_center_bed(
            chrom_sizes_path=input.sizes,
            undefined_bed_path=input.undefined,
            output_bed_path=output[0],
            standard_chroms=STANDARD_CHROMS,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
            expected_queries=GRID_EXPECTED_QUERIES,
        )


rule benchmark_full_grid_chain_liftover:
    input:
        bed=local(FULL_GRID_BED),
        chain=lambda wc: CHAIN_PATH.format(species=wc.species, recipe="no_dupes"),
    output:
        mapped=temp(local(FULL_GRID_MAPPED)),
        unmapped=temp(local(FULL_GRID_UNMAPPED)),
        metrics=FULL_GRID_METRICS,
    wildcard_constraints:
        species=SPECIES_RE,
    conda:
        "../envs/liftover.yaml"
    resources:
        mem_mb=12000,
        full_grid_liftover=1,
    run:
        run_liftover_benchmark(
            input_bed=input.bed,
            chain_path=input.chain,
            mapped_bed=output.mapped,
            unmapped_bed=output.unmapped,
            metrics_path=output.metrics,
            expected_queries=GRID_EXPECTED_QUERIES,
        )
