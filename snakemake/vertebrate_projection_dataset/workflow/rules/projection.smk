"""HAL and MAF adapters feeding one projection/sequence contract."""

from marin_dna_zoonomia_projection.projection.hal import (
    attach_src_size,
    parse_halliftover_bed,
    run_halliftover,
    write_chrom_sizes,
)
from marin_dna_vertebrate_projection.pipeline_io import (
    combine_sequence_parquets,
    merge_parquets_streaming,
    write_contract_outputs,
    write_contract_outputs_for_alignment,
    write_hal_bed6,
    write_hal_fragments,
    write_maf_candidates,
)
from marin_dna_vertebrate_projection.sequence_compatibility import (
    validate_projected_twobit_sizes,
)
from marin_dna_vertebrate_projection.sequence_sources import stage_twobit


rule download_human_twobit:
    input:
        TWOBIT_MANIFEST,
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
        anchors=ANCHOR_CATALOG,
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


rule prepare_hal_bed:
    input:
        ANCHOR_CATALOG,
    output:
        f"{RESULTS}/hal/input.bed",
    run:
        write_hal_bed6(input[0], output[0])


rule hal_chrom_sizes:
    input:
        hal=HAL_PATH,
        validation=HAL_VALIDATION,
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
        hal=HAL_PATH,
        bed=f"{RESULTS}/hal/input.bed",
        validation=HAL_VALIDATION,
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
        anchors=ANCHOR_CATALOG,
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
        write_hal_fragments(records, input.anchors, input.manifest, output[0])


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
        write_contract_outputs(
            input[0],
            output.accepted,
            output.rejected,
            target_length=TARGET_LENGTH,
            pre_resize_min_length=PRE_RESIZE_MIN,
            pre_resize_max_length=PRE_RESIZE_MAX,
        )


rule hal_to_fasta:
    input:
        hal=HAL_PATH,
        validation=HAL_VALIDATION,
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
        f"{RESULTS}/hal/genomes/{{species}}.fa",
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


rule multiz_candidates:
    input:
        maf=f"{MULTIZ_STAGE_DIR}/maf/{{chrom}}.maf.gz",
        anchors=ANCHOR_CATALOG,
        manifest=ACTIVE_MANIFEST,
    output:
        f"{RESULTS}/multiz/fragments/{{chrom}}.parquet",
    wildcard_constraints:
        chrom=CHROM_RE,
    threads: 4
    resources:
        mem_mb=16000,
    run:
        write_maf_candidates(input.maf, input.anchors, input.manifest, output[0])


rule multiz_contract:
    input:
        f"{RESULTS}/multiz/fragments/{{chrom}}.parquet",
    output:
        accepted=f"{RESULTS}/multiz/accepted/by_chrom/{{chrom}}/{{species}}.parquet",
        rejected=f"{RESULTS}/multiz/rejected/by_chrom/{{chrom}}/{{species}}.parquet",
    wildcard_constraints:
        chrom=CHROM_RE,
        species=NON_MAMMAL_RE,
    resources:
        mem_mb=8000,
    run:
        write_contract_outputs_for_alignment(
            input[0],
            wildcards.species,
            output.accepted,
            output.rejected,
            target_length=TARGET_LENGTH,
            pre_resize_min_length=PRE_RESIZE_MIN,
            pre_resize_max_length=PRE_RESIZE_MAX,
        )


rule merge_multiz_accepted:
    input:
        lambda wc: expand(
            f"{RESULTS}/multiz/accepted/by_chrom/{{chrom}}/{{species}}.parquet",
            chrom=CHROMS,
            species=[wc.species],
        ),
    output:
        f"{RESULTS}/multiz/accepted/{{species}}.parquet",
    wildcard_constraints:
        species=NON_MAMMAL_RE,
    resources:
        mem_mb=4000,
    run:
        merge_parquets_streaming(list(input), output[0])


rule merge_multiz_rejected:
    input:
        lambda wc: expand(
            f"{RESULTS}/multiz/rejected/by_chrom/{{chrom}}/{{species}}.parquet",
            chrom=CHROMS,
            species=[wc.species],
        ),
    output:
        f"{RESULTS}/multiz/rejected/{{species}}.parquet",
    wildcard_constraints:
        species=NON_MAMMAL_RE,
    resources:
        mem_mb=4000,
    run:
        merge_parquets_streaming(list(input), output[0])


rule download_multiz_twobit:
    input:
        TWOBIT_MANIFEST,
    output:
        f"{RESULTS}/multiz/genomes/{{species}}.2bit",
    wildcard_constraints:
        species=NON_MAMMAL_RE,
    resources:
        ucsc_downloads=1,
    run:
        stage_twobit(twobit_objects[wildcards.species], output[0])


rule multiz_twobit_chrom_sizes:
    input:
        f"{RESULTS}/multiz/genomes/{{species}}.2bit",
    output:
        f"{RESULTS}/multiz/genomes/{{species}}.chrom.sizes",
    wildcard_constraints:
        species=NON_MAMMAL_RE,
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input} {output}"


rule validate_multiz_twobit_compatibility:
    input:
        accepted=f"{RESULTS}/multiz/accepted/{{species}}.parquet",
        sizes=f"{RESULTS}/multiz/genomes/{{species}}.chrom.sizes",
    output:
        f"{RESULTS}/multiz/genomes/{{species}}.compatibility.json",
    wildcard_constraints:
        species=NON_MAMMAL_RE,
    run:
        validate_projected_twobit_sizes(input.accepted, input.sizes, output[0])


rule multiz_sequences:
    input:
        accepted=f"{RESULTS}/multiz/accepted/{{species}}.parquet",
        twobit=f"{RESULTS}/multiz/genomes/{{species}}.2bit",
        compatibility=f"{RESULTS}/multiz/genomes/{{species}}.compatibility.json",
    output:
        sequences=f"{RESULTS}/sequences/multiz/{{species}}.parquet",
        rejected=f"{RESULTS}/multiz/sequence_rejected/{{species}}.parquet",
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
            f"{RESULTS}/sequences/multiz/{{species}}.parquet",
            species=NON_MAMMALS,
        ),
    output:
        COMBINED_SEQUENCES,
    run:
        combine_sequence_parquets(list(input), output[0])
