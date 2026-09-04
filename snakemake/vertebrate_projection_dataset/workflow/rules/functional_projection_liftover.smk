"""Issue #517 HAL and UCSC chain adapters feeding one projection contract."""

from marin_dna_vertebrate_projection.projection.hal import (
    attach_src_size,
    parse_halliftover_bed,
    run_halliftover,
    write_chrom_sizes,
)
from marin_dna_vertebrate_projection.projection.center import write_hal_request_bed6
from marin_dna_vertebrate_projection.projection.liftover import (
    stage_liftover_chain,
    write_liftover_fragments,
)
from marin_dna_vertebrate_projection.projection.requests import (
    build_projection_requests,
)
from marin_dna_vertebrate_projection.pipeline_io import (
    combine_sequence_parquets,
    read_anchor_catalog,
    write_contract_outputs,
    write_hal_fragments,
)
from marin_dna_vertebrate_projection.sequence_compatibility import (
    validate_projected_twobit_sizes,
)
from marin_dna_vertebrate_projection.sequence_sources import stage_twobit


rule download_human_twobit:
    input:
        TWOBIT_MANIFEST_INPUT,
    output:
        f"{RESULTS}/reference/hg38.2bit",
    resources:
        ucsc_downloads=1,
    run:
        stage_twobit(twobit_objects["hg38"], output[0])


rule human_chrom_sizes:
    input:
        f"{RESULTS}/reference/hg38.2bit",
    output:
        f"{RESULTS}/reference/hg38.chrom.sizes",
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} {output}"


rule human_reference_sequences:
    input:
        anchors=ANCHOR_CATALOG_INPUT,
        twobit=f"{RESULTS}/reference/hg38.2bit",
        sizes=f"{RESULTS}/reference/hg38.chrom.sizes",
    output:
        HUMAN_SEQUENCES,
    conda:
        "../envs/bioinformatics.yaml"
    resources:
        mem_mb=4000,
    shell:
        "uv run python -m "
        "marin_dna_vertebrate_projection.sequence_cli "
        "human {input.anchors} {input.twobit} {input.sizes} {output}"


rule projection_requests:
    input:
        ANCHOR_CATALOG_INPUT,
    output:
        PROJECTION_REQUESTS,
    run:
        requests = build_projection_requests(read_anchor_catalog(input[0]))
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        requests.write_parquet(output[0])


rule prepare_hal_bed:
    input:
        PROJECTION_REQUESTS,
    output:
        f"{RESULTS}/hal/input.bed",
    run:
        write_hal_request_bed6(input[0], output[0])


rule hal_chrom_sizes:
    input:
        hal=local(HAL_PATH),
        validation=local(HAL_VALIDATION),
    output:
        f"{RESULTS}/hal/chrom_sizes/{{species}}.tsv",
    wildcard_constraints:
        species=MAMMAL_RE,
    resources:
        mem_mb=2000,
    run:
        write_chrom_sizes(input.hal, wildcards.species, output[0])


rule hal_liftover:
    input:
        hal=local(HAL_PATH),
        bed=f"{RESULTS}/hal/input.bed",
        validation=local(HAL_VALIDATION),
    output:
        f"{RESULTS}/hal/raw/{{species}}.bed",
    wildcard_constraints:
        species=MAMMAL_RE,
    threads: 1
    resources:
        mem_mb=2000,
    run:
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        run_halliftover(
            input.hal,
            "Homo_sapiens",
            input.bed,
            wildcards.species,
            output[0],
            no_dupes=True,
        )


rule hal_fragments:
    input:
        raw=f"{RESULTS}/hal/raw/{{species}}.bed",
        sizes=f"{RESULTS}/hal/chrom_sizes/{{species}}.tsv",
        requests=PROJECTION_REQUESTS,
        manifest=ACTIVE_MANIFEST,
    output:
        f"{RESULTS}/hal/fragments/{{species}}.parquet",
    wildcard_constraints:
        species=MAMMAL_RE,
    resources:
        mem_mb=8000,
    run:
        records = attach_src_size(
            parse_halliftover_bed(input.raw, wildcards.species), input.sizes
        )
        write_hal_fragments(records, input.requests, input.manifest, output[0])


rule hal_contract:
    input:
        f"{RESULTS}/hal/fragments/{{species}}.parquet",
    output:
        accepted=f"{RESULTS}/hal/accepted/{{species}}.parquet",
        rejected=f"{RESULTS}/hal/rejected/{{species}}.parquet",
    wildcard_constraints:
        species=MAMMAL_RE,
    resources:
        mem_mb=10000,
    run:
        write_contract_outputs(input[0], output.accepted, output.rejected)


rule hal_to_fasta:
    input:
        hal=local(HAL_PATH),
        validation=local(HAL_VALIDATION),
    output:
        local(f"{RESULTS}/hal/genomes/{{species}}.fa"),
    wildcard_constraints:
        species=MAMMAL_RE,
    threads: 4
    resources:
        mem_mb=4000,
    shell:
        "hal2fasta {input.hal} {wildcards.species} > {output}"


rule hal_fasta_to_twobit:
    input:
        local(f"{RESULTS}/hal/genomes/{{species}}.fa"),
    output:
        f"{RESULTS}/hal/genomes/{{species}}.2bit",
    wildcard_constraints:
        species=MAMMAL_RE,
    conda:
        "../envs/bioinformatics.yaml"
    threads: 2
    resources:
        mem_mb=4000,
    shell:
        "faToTwoBit {input} {output}"


rule hal_sequences:
    input:
        accepted=f"{RESULTS}/hal/accepted/{{species}}.parquet",
        twobit=f"{RESULTS}/hal/genomes/{{species}}.2bit",
    output:
        sequences=f"{RESULTS}/sequences/hal/{{species}}.parquet",
        rejected=f"{RESULTS}/hal/sequence_rejected/{{species}}.parquet",
    wildcard_constraints:
        species=MAMMAL_RE,
    conda:
        "../envs/bioinformatics.yaml"
    resources:
        mem_mb=4000,
    shell:
        "uv run python -m "
        "marin_dna_vertebrate_projection.sequence_cli "
        "projected {input.accepted} {input.twobit} {output.sequences} {output.rejected}"


rule download_liftover_chain:
    input:
        LIFTOVER_CHAIN_MANIFEST_INPUT,
    output:
        f"{RESULTS}/liftover/chains/{{species}}.over.chain.gz",
    wildcard_constraints:
        species=NON_MAMMAL_RE,
    resources:
        ucsc_downloads=1,
        mem_mb=1000,
    run:
        stage_liftover_chain(liftover_chains[wildcards.species], output[0])


rule run_non_mammal_liftover:
    input:
        bed=f"{RESULTS}/hal/input.bed",
        chain=f"{RESULTS}/liftover/chains/{{species}}.over.chain.gz",
    output:
        mapped=f"{RESULTS}/liftover/raw/{{species}}.mapped.bed",
        unmapped=f"{RESULTS}/liftover/raw/{{species}}.unmapped.bed",
    wildcard_constraints:
        species=NON_MAMMAL_RE,
    conda:
        "../envs/liftover.yaml"
    threads: 1
    resources:
        mem_mb=2000,
    params:
        options=(
            f"-minMatch={LIFTOVER_MIN_MATCH:g}"
            + (" -multiple" if LIFTOVER_MULTIPLE else "")
        ),
    shell:
        "liftOver {params.options} {input.bed} {input.chain} "
        "{output.mapped} {output.unmapped}"


rule download_liftover_twobit:
    input:
        TWOBIT_MANIFEST_INPUT,
    output:
        f"{RESULTS}/liftover/genomes/{{species}}.2bit",
    wildcard_constraints:
        species=NON_MAMMAL_RE,
    resources:
        ucsc_downloads=1,
    run:
        stage_twobit(twobit_objects[wildcards.species], output[0])


rule liftover_twobit_chrom_sizes:
    input:
        f"{RESULTS}/liftover/genomes/{{species}}.2bit",
    output:
        f"{RESULTS}/liftover/genomes/{{species}}.chrom.sizes",
    wildcard_constraints:
        species=NON_MAMMAL_RE,
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} {output}"


rule liftover_fragments:
    input:
        input_bed=f"{RESULTS}/hal/input.bed",
        mapped=f"{RESULTS}/liftover/raw/{{species}}.mapped.bed",
        unmapped=f"{RESULTS}/liftover/raw/{{species}}.unmapped.bed",
        requests=PROJECTION_REQUESTS,
        manifest=ACTIVE_MANIFEST,
        sizes=f"{RESULTS}/liftover/genomes/{{species}}.chrom.sizes",
    output:
        f"{RESULTS}/liftover/fragments/{{species}}.parquet",
    wildcard_constraints:
        species=NON_MAMMAL_RE,
    resources:
        mem_mb=8000,
    run:
        write_liftover_fragments(
            input.input_bed,
            input.mapped,
            input.unmapped,
            input.requests,
            input.manifest,
            input.sizes,
            output[0],
            alignment_name=wildcards.species,
            multiple=LIFTOVER_MULTIPLE,
        )


rule liftover_contract:
    input:
        f"{RESULTS}/liftover/fragments/{{species}}.parquet",
    output:
        accepted=f"{RESULTS}/liftover/accepted/{{species}}.parquet",
        rejected=f"{RESULTS}/liftover/rejected/{{species}}.parquet",
    wildcard_constraints:
        species=NON_MAMMAL_RE,
    resources:
        mem_mb=8000,
    run:
        write_contract_outputs(input[0], output.accepted, output.rejected)


rule validate_liftover_twobit_compatibility:
    input:
        accepted=f"{RESULTS}/liftover/accepted/{{species}}.parquet",
        sizes=f"{RESULTS}/liftover/genomes/{{species}}.chrom.sizes",
    output:
        f"{RESULTS}/liftover/genomes/{{species}}.compatibility.json",
    wildcard_constraints:
        species=NON_MAMMAL_RE,
    run:
        validate_projected_twobit_sizes(input.accepted, input.sizes, output[0])


rule liftover_sequences:
    input:
        accepted=f"{RESULTS}/liftover/accepted/{{species}}.parquet",
        twobit=f"{RESULTS}/liftover/genomes/{{species}}.2bit",
        compatibility=f"{RESULTS}/liftover/genomes/{{species}}.compatibility.json",
    output:
        sequences=f"{RESULTS}/sequences/liftover/{{species}}.parquet",
        rejected=f"{RESULTS}/liftover/sequence_rejected/{{species}}.parquet",
    wildcard_constraints:
        species=NON_MAMMAL_RE,
    conda:
        "../envs/bioinformatics.yaml"
    resources:
        mem_mb=4000,
    shell:
        "uv run python -m "
        "marin_dna_vertebrate_projection.sequence_cli "
        "projected {input.accepted} {input.twobit} {output.sequences} {output.rejected}"


rule combine_sequences:
    input:
        [HUMAN_SEQUENCES]
        + expand(f"{RESULTS}/sequences/hal/{{species}}.parquet", species=MAMMALS)
        + expand(
            f"{RESULTS}/sequences/liftover/{{species}}.parquet",
            species=NON_MAMMALS,
        ),
    output:
        COMBINED_SEQUENCES,
    run:
        combine_sequence_parquets(list(input), output[0])
