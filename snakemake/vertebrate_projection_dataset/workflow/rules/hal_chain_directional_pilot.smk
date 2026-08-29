"""Direction-matched HAL-chain smoke gate and baboon whole-genome pilot."""

from pathlib import Path

from marin_dna_vertebrate_projection.mirror import (
    s3_object_size,
    stage_hal_object,
    verify_hal_object,
)
from marin_dna_vertebrate_projection.projection.hal_chains import (
    run_direct_hal_benchmark,
    run_direction_matched_hal_to_chain,
    run_liftover_benchmark,
    write_chain_parity_audit,
    write_exact_chain_parity_audit,
    write_hal_genome_assets,
    write_regional_smoke_beds,
    write_uniform_grid_center_bed,
)
from marin_dna_vertebrate_projection.provenance import (
    hash_pipeline_config,
    resolve_pipeline_commit,
    write_producer_manifest,
)


PIPELINE_VERSION = str(config["pipeline_version"])
TIER = str(config["tier"])
assert PIPELINE_VERSION == "hal-chains-directional-pilot-v1"
assert TIER == "full"
assert str(config["cactus_version"]) == "3.3.0"
assert int(config["kent_version"]) == 482
PIPELINE_COMMIT = resolve_pipeline_commit()
PIPELINE_CONFIG_SHA256 = hash_pipeline_config(config)
RESULTS = (
    f"results/{PIPELINE_VERSION}/{PIPELINE_COMMIT}/{PIPELINE_CONFIG_SHA256}/{TIER}"
)
PRODUCER_MANIFEST = f"{RESULTS}/metadata/producer.json"

SOURCE_GENOME = str(config["source_genome"])
assert SOURCE_GENOME == "Homo_sapiens"
SMOKE_SPECIES = list(config["smoke_species"])
assert SMOKE_SPECIES == ["Papio_anubis", "Mus_musculus", "Loxodonta_africana"]
FULL_SPECIES = str(config["full_species"])
assert FULL_SPECIES == "Papio_anubis"
GENOMES = [SOURCE_GENOME, *SMOKE_SPECIES]
GENOME_RE = "|".join(GENOMES)
SPECIES_RE = "|".join(SMOKE_SPECIES)
LINEAR_GAP = str(config["linear_gap"])
MIN_SCORE = int(config["chain_min_score"])
assert LINEAR_GAP == "medium" and MIN_SCORE == -1_000_000

HAL_PATH = str(config["hal_stage_path"])
HAL_VALIDATION = str(config["hal_validation_path"])
SMOKE_REGIONS = list(config["smoke_regions"])
SMOKE_EXPECTED_QUERIES = int(config["smoke_expected_queries"])
assert len(SMOKE_REGIONS) == 12 and SMOKE_EXPECTED_QUERIES == 9_374

BASELINE_ROOT = str(config["strict_phylop_baseline_root"])
VALIDATION_BED = f"{BASELINE_ROOT}/hal/input.bed"
VALIDATION_EXPECTED_QUERIES = int(config["validation_expected_queries"])
GRID_CHROM_SIZES = f"{BASELINE_ROOT}/anchors/chrom.sizes"
GRID_UNDEFINED_BED = f"{BASELINE_ROOT}/anchors/undefined.ucsc.bed"
GRID_EXPECTED_QUERIES = int(config["uniform_grid_expected_queries"])
WINDOW_SIZE = int(config["window_size"])
STEP_SIZE = int(config["step_size"])
STANDARD_CHROMS = list(config["standard_chroms"])
assert VALIDATION_EXPECTED_QUERIES == 1_136_854
assert GRID_EXPECTED_QUERIES == 22_948_560
assert WINDOW_SIZE == 255 and STEP_SIZE == 128 and len(STANDARD_CHROMS) == 24

GENOME_SIZES = f"{RESULTS}/genomes/{{genome}}.chrom.sizes"
GENOME_BED = f"{RESULTS}/genomes/{{genome}}.whole_genome.bed"
GENOME_TWOBIT = f"{RESULTS}/genomes/{{genome}}.2bit"

SMOKE_REGION_BED = f"{RESULTS}/smoke/source_regions.bed"
SMOKE_CENTER_BED = f"{RESULTS}/smoke/source_centers.bed"
SMOKE_DIRECT_BED = f"{RESULTS}/smoke/{{species}}/direct_hal.bed"
SMOKE_DIRECT_METRICS = f"{RESULTS}/smoke/{{species}}/direct_hal.json"
SMOKE_CHAIN = f"{RESULTS}/smoke/{{species}}/human_to_species.chain.gz"
SMOKE_CHAIN_METRICS = f"{RESULTS}/smoke/{{species}}/chain_generation.json"
SMOKE_MAPPED = f"{RESULTS}/smoke/{{species}}/chain_mapped.bed"
SMOKE_UNMAPPED = f"{RESULTS}/smoke/{{species}}/chain_unmapped.bed"
SMOKE_LIFTOVER_METRICS = f"{RESULTS}/smoke/{{species}}/chain_liftover.json"
SMOKE_PARITY_SUMMARY = f"{RESULTS}/smoke/{{species}}/parity.json"
SMOKE_PARITY_DISCREPANCIES = f"{RESULTS}/smoke/{{species}}/discrepancies.parquet"

FULL_CHAIN = f"{RESULTS}/full/{FULL_SPECIES}/human_to_species.chain.gz"
FULL_CHAIN_METRICS = f"{RESULTS}/full/{FULL_SPECIES}/chain_generation.json"
FULL_VALIDATION_MAPPED = f"{RESULTS}/full/{FULL_SPECIES}/validation.mapped.bed"
FULL_VALIDATION_UNMAPPED = f"{RESULTS}/full/{FULL_SPECIES}/validation.unmapped.bed"
FULL_VALIDATION_METRICS = f"{RESULTS}/full/{FULL_SPECIES}/validation.liftover.json"
FULL_PARITY_SUMMARY = f"{RESULTS}/full/{FULL_SPECIES}/parity.json"
FULL_PARITY_DISCREPANCIES = f"{RESULTS}/full/{FULL_SPECIES}/discrepancies.parquet"
FULL_GRID_BED = f"{RESULTS}/full_grid/input.center1.bed"
FULL_GRID_MAPPED = f"{RESULTS}/full_grid/{FULL_SPECIES}.mapped.bed"
FULL_GRID_UNMAPPED = f"{RESULTS}/full_grid/{FULL_SPECIES}.unmapped.bed"
FULL_GRID_METRICS = f"{RESULTS}/full_grid/{FULL_SPECIES}.liftover.json"


rule hal_chain_directional_producer_manifest:
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


rule stage_hal_for_directional_chain:
    output:
        local(HAL_PATH),
    resources:
        mem_mb=1000,
    params:
        source=str(config["hal_s3_uri"]),
    run:
        stage_hal_object(params.source, output[0])


rule validate_hal_for_directional_chain:
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


rule prepare_directional_chain_genome_assets:
    input:
        hal=local(HAL_PATH),
        validation=local(HAL_VALIDATION),
    output:
        sizes=temp(local(GENOME_SIZES)),
        bed=temp(local(GENOME_BED)),
        twobit=temp(local(GENOME_TWOBIT)),
    wildcard_constraints:
        genome=GENOME_RE,
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


rule prepare_directional_smoke_queries:
    output:
        regions=temp(local(SMOKE_REGION_BED)),
        centers=temp(local(SMOKE_CENTER_BED)),
    resources:
        mem_mb=1000,
    run:
        write_regional_smoke_beds(
            regions=SMOKE_REGIONS,
            region_bed_path=output.regions,
            centers_bed_path=output.centers,
            step_size=STEP_SIZE,
            center_offset=WINDOW_SIZE // 2,
            expected_queries=SMOKE_EXPECTED_QUERIES,
        )


rule benchmark_directional_smoke_direct_hal:
    input:
        hal=local(HAL_PATH),
        validation=local(HAL_VALIDATION),
        centers=local(SMOKE_CENTER_BED),
    output:
        bed=temp(local(SMOKE_DIRECT_BED)),
        metrics=SMOKE_DIRECT_METRICS,
    wildcard_constraints:
        species=SPECIES_RE,
    resources:
        mem_mb=8000,
    run:
        run_direct_hal_benchmark(
            hal_path=input.hal,
            source_genome=SOURCE_GENOME,
            source_bed=input.centers,
            destination_genome=wildcards.species,
            output_bed=output.bed,
            metrics_path=output.metrics,
            expected_queries=SMOKE_EXPECTED_QUERIES,
        )


rule generate_directional_smoke_chain:
    input:
        hal=local(HAL_PATH),
        validation=local(HAL_VALIDATION),
        source_bed=local(SMOKE_REGION_BED),
        source_sizes=local(GENOME_SIZES.format(genome=SOURCE_GENOME)),
        source_twobit=local(GENOME_TWOBIT.format(genome=SOURCE_GENOME)),
        destination_sizes=local(GENOME_SIZES.replace("{genome}", "{species}")),
        destination_twobit=local(GENOME_TWOBIT.replace("{genome}", "{species}")),
    output:
        chain=SMOKE_CHAIN,
        metrics=SMOKE_CHAIN_METRICS,
    wildcard_constraints:
        species=SPECIES_RE,
    resources:
        mem_mb=24000,
        chain_generation=1,
    run:
        run_direction_matched_hal_to_chain(
            hal_path=input.hal,
            source_genome=SOURCE_GENOME,
            source_bed=input.source_bed,
            source_chrom_sizes=input.source_sizes,
            source_twobit=input.source_twobit,
            destination_genome=wildcards.species,
            destination_chrom_sizes=input.destination_sizes,
            destination_twobit=input.destination_twobit,
            output_chain=output.chain,
            output_metrics=output.metrics,
            min_score=MIN_SCORE,
            linear_gap=LINEAR_GAP,
        )


rule liftover_directional_smoke_chain:
    input:
        bed=local(SMOKE_CENTER_BED),
        chain=SMOKE_CHAIN,
    output:
        mapped=temp(local(SMOKE_MAPPED)),
        unmapped=temp(local(SMOKE_UNMAPPED)),
        metrics=SMOKE_LIFTOVER_METRICS,
    wildcard_constraints:
        species=SPECIES_RE,
    resources:
        mem_mb=8000,
    run:
        run_liftover_benchmark(
            input_bed=input.bed,
            chain_path=input.chain,
            mapped_bed=output.mapped,
            unmapped_bed=output.unmapped,
            metrics_path=output.metrics,
            expected_queries=SMOKE_EXPECTED_QUERIES,
        )


rule gate_directional_smoke_parity:
    input:
        queries=local(SMOKE_CENTER_BED),
        direct=local(SMOKE_DIRECT_BED),
        chain=local(SMOKE_MAPPED),
    output:
        summary=SMOKE_PARITY_SUMMARY,
        discrepancies=SMOKE_PARITY_DISCREPANCIES,
    wildcard_constraints:
        species=SPECIES_RE,
    resources:
        mem_mb=8000,
    run:
        write_exact_chain_parity_audit(
            input_bed=input.queries,
            direct_bed=input.direct,
            chain_bed=input.chain,
            summary_path=output.summary,
            discrepancies_path=output.discrepancies,
            expected_queries=SMOKE_EXPECTED_QUERIES,
        )


rule generate_full_baboon_directional_chain:
    input:
        smoke_gate=expand(SMOKE_PARITY_SUMMARY, species=SMOKE_SPECIES),
        hal=local(HAL_PATH),
        validation=local(HAL_VALIDATION),
        source_bed=local(GENOME_BED.format(genome=SOURCE_GENOME)),
        source_sizes=local(GENOME_SIZES.format(genome=SOURCE_GENOME)),
        source_twobit=local(GENOME_TWOBIT.format(genome=SOURCE_GENOME)),
        destination_sizes=local(GENOME_SIZES.format(genome=FULL_SPECIES)),
        destination_twobit=local(GENOME_TWOBIT.format(genome=FULL_SPECIES)),
    output:
        chain=FULL_CHAIN,
        metrics=FULL_CHAIN_METRICS,
    resources:
        mem_mb=170000,
        chain_generation=1,
    run:
        run_direction_matched_hal_to_chain(
            hal_path=input.hal,
            source_genome=SOURCE_GENOME,
            source_bed=input.source_bed,
            source_chrom_sizes=input.source_sizes,
            source_twobit=input.source_twobit,
            destination_genome=FULL_SPECIES,
            destination_chrom_sizes=input.destination_sizes,
            destination_twobit=input.destination_twobit,
            output_chain=output.chain,
            output_metrics=output.metrics,
            min_score=MIN_SCORE,
            linear_gap=LINEAR_GAP,
        )


rule validate_full_baboon_directional_chain:
    input:
        bed=VALIDATION_BED,
        chain=FULL_CHAIN,
    output:
        mapped=temp(local(FULL_VALIDATION_MAPPED)),
        unmapped=temp(local(FULL_VALIDATION_UNMAPPED)),
        metrics=FULL_VALIDATION_METRICS,
    resources:
        mem_mb=12000,
    run:
        run_liftover_benchmark(
            input_bed=input.bed,
            chain_path=input.chain,
            mapped_bed=output.mapped,
            unmapped_bed=output.unmapped,
            metrics_path=output.metrics,
            expected_queries=VALIDATION_EXPECTED_QUERIES,
        )


rule audit_full_baboon_directional_parity:
    input:
        queries=VALIDATION_BED,
        direct=f"{BASELINE_ROOT}/hal/raw/{FULL_SPECIES}.bed",
        chain=local(FULL_VALIDATION_MAPPED),
    output:
        summary=FULL_PARITY_SUMMARY,
        discrepancies=FULL_PARITY_DISCREPANCIES,
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


rule prepare_directional_full_uniform_grid_centers:
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


rule benchmark_full_baboon_directional_grid:
    input:
        bed=local(FULL_GRID_BED),
        chain=FULL_CHAIN,
    output:
        mapped=temp(local(FULL_GRID_MAPPED)),
        unmapped=temp(local(FULL_GRID_UNMAPPED)),
        metrics=FULL_GRID_METRICS,
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
