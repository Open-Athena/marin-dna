"""HAL and MAF adapters feeding one projection/sequence contract."""

from marin_dna.pipelines.projection.hal import (
    attach_src_size,
    parse_halliftover_bed,
    run_halliftover,
    write_chrom_sizes,
)
from marin_dna.pipelines.vertebrate_projection_dataset.pipeline_io import (
    combine_sequence_parquets,
    write_contract_outputs,
    write_hal_bed6,
    write_hal_fragments,
    write_human_reference_sequences,
    write_maf_candidates,
    write_twobit_sequences,
)


rule download_human_twobit:
    output:
        f"{RESULTS}/reference/hg38.2bit",
    params:
        url=str(config["human_twobit_url"]),
    shell:
        "wget -q -O {output} {params.url}"


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
    run:
        write_human_reference_sequences(
            input.anchors, input.twobit, input.sizes, output[0]
        )


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
    output:
        f"{RESULTS}/hal/chrom_sizes/{{species}}.tsv",
    wildcard_constraints:
        species=MAMMAL_RE,
    run:
        write_chrom_sizes(input.hal, wildcards.species, output[0])


rule hal_liftover:
    input:
        hal=HAL_PATH,
        bed=f"{RESULTS}/hal/input.bed",
    output:
        f"{RESULTS}/hal/raw/{{species}}.bed",
    wildcard_constraints:
        species=MAMMAL_RE,
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
    output:
        local(f"{RESULTS}/hal/genomes/{{species}}.fa"),
    wildcard_constraints:
        species=MAMMAL_RE,
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
    run:
        write_twobit_sequences(
            input.accepted, input.twobit, output.sequences, output.rejected
        )


rule multiz_candidates:
    input:
        maf=f"{MULTIZ_STAGE_DIR}/maf/{{chrom}}.maf.gz",
        anchors=ANCHOR_CATALOG,
        manifest=ACTIVE_MANIFEST,
    output:
        f"{RESULTS}/multiz/fragments/{{chrom}}.parquet",
    wildcard_constraints:
        chrom=CHROM_RE,
    run:
        write_maf_candidates(input.maf, input.anchors, input.manifest, output[0])


rule merge_multiz_fragments:
    input:
        expand(f"{RESULTS}/multiz/fragments/{{chrom}}.parquet", chrom=CHROMS),
    output:
        f"{RESULTS}/multiz/fragments/all.parquet",
    run:
        pl.concat([pl.read_parquet(path) for path in input]).write_parquet(output[0])


rule multiz_contract:
    input:
        f"{RESULTS}/multiz/fragments/all.parquet",
    output:
        accepted=f"{RESULTS}/multiz/accepted/all.parquet",
        rejected=f"{RESULTS}/multiz/rejected/all.parquet",
    run:
        write_contract_outputs(
            input[0],
            output.accepted,
            output.rejected,
            target_length=TARGET_LENGTH,
            pre_resize_min_length=PRE_RESIZE_MIN,
            pre_resize_max_length=PRE_RESIZE_MAX,
        )


rule subset_multiz_accepted:
    input:
        accepted=f"{RESULTS}/multiz/accepted/all.parquet",
        manifest=ACTIVE_MANIFEST,
    output:
        f"{RESULTS}/multiz/accepted/{{species}}.parquet",
    wildcard_constraints:
        species=NON_MAMMAL_RE,
    run:
        manifest = pl.read_csv(input.manifest, separator="\t")
        scientific_name = manifest.filter(
            pl.col("alignment_name") == wildcards.species
        )["scientific_name"].item()
        frame = pl.read_parquet(input.accepted).filter(
            pl.col("species") == scientific_name
        )
        frame.write_parquet(output[0])


rule download_multiz_twobit:
    output:
        f"{RESULTS}/multiz/genomes/{{species}}.2bit",
    wildcard_constraints:
        species=NON_MAMMAL_RE,
    params:
        # gbdb is the stable UCSC endpoint across both current and legacy
        # assemblies in the 2015 hg38 MultiZ release.
        url=lambda wc: (
            f"https://hgdownload.soe.ucsc.edu/gbdb/{wc.species}/{wc.species}.2bit"
        ),
    shell:
        "wget -q -O {output} {params.url}"


rule multiz_sequences:
    input:
        accepted=f"{RESULTS}/multiz/accepted/{{species}}.parquet",
        twobit=f"{RESULTS}/multiz/genomes/{{species}}.2bit",
    output:
        sequences=f"{RESULTS}/sequences/multiz/{{species}}.parquet",
        rejected=f"{RESULTS}/multiz/sequence_rejected/{{species}}.parquet",
    wildcard_constraints:
        species=NON_MAMMAL_RE,
    run:
        write_twobit_sequences(
            input.accepted, input.twobit, output.sequences, output.rejected
        )


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
