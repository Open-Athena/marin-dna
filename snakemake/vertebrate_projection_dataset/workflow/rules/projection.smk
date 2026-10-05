"""One chain projector and assembly-matched sequence path for every target."""

from marin_dna_vertebrate_projection.projection.chains import (
    prepare_chain_requests,
    validate_chain_inputs,
    write_chain_projections,
)
from marin_dna_vertebrate_projection.genome_assets import (
    validate_genome_source,
    validate_genome_dictionary,
    validate_human_dictionary,
)
from marin_dna_vertebrate_projection.pipeline_io import combine_sequence_parquets
from marin_dna_vertebrate_projection.sequence_compatibility import (
    validate_projected_twobit_sizes,
)

SOURCE = ASSETS[ACTIVE_SPECIES[0]]


rule chain_requests:
    input:
        anchors=ANCHOR_CATALOG_INPUT,
        sizes=asset_input(SOURCE["source_sizes"]),
        human_sizes=asset_input(GENOMES["hg38"]["chrom_sizes"]),
        provenance=ASSET_PROVENANCE,
    output:
        requests=PROJECTION_REQUESTS,
        bed=f"{RESULTS}/anchors/centers.bed",
    resources:
        mem_mb=int(config["table_mem_mb"]),
    run:
        if (
            config.get("anchors")
            and file_sha256(input.anchors) != config["anchors_sha256"]
        ):
            raise ValueError("anchor catalog SHA-256 mismatch")
        if file_sha256(input.sizes) != SOURCE["source_sizes_sha256"]:
            raise ValueError("source chromosome dictionary SHA-256 mismatch")
        if file_sha256(input.human_sizes) != GENOMES["hg38"]["chrom_sizes_sha256"]:
            raise ValueError("human genome dictionary SHA-256 mismatch")
        validate_human_dictionary(input.sizes, input.human_sizes)
        prepare_chain_requests(
            input.anchors, input.sizes, output.requests, output.bed
        )


rule chain_validate:
    input:
        chain=lambda w: asset_input(ASSETS[w.species]["chain"]),
        source=lambda w: asset_input(ASSETS[w.species]["source_sizes"]),
        target=lambda w: asset_input(ASSETS[w.species]["target_sizes"]),
        provenance=ASSET_PROVENANCE,
    output:
        f"{RESULTS}/chains/{{species}}/validated.json",
    wildcard_constraints:
        species=SPECIES_RE,
    resources:
        mem_mb=1000,
    run:
        validate_chain_inputs(
            input.chain,
            input.source,
            input.target,
            ASSETS[wildcards.species],
            output[0],
        )


rule chain_liftover:
    input:
        bed=f"{RESULTS}/anchors/centers.bed",
        chain=lambda w: asset_input(ASSETS[w.species]["chain"]),
        validated=f"{RESULTS}/chains/{{species}}/validated.json",
    output:
        mapped=f"{RESULTS}/chains/{{species}}/mapped.bed",
        unmapped=f"{RESULTS}/chains/{{species}}/unmapped.bed",
    log:
        f"{RESULTS}/chains/{{species}}/liftover.log",
    benchmark:
        f"{RESULTS}/chains/{{species}}/liftover.benchmark.tsv"
    conda:
        "../envs/chains.yaml"
    threads: 1
    resources:
        mem_mb=int(config["liftover_mem_mb"]),
    shell:
        "liftOver -minMatch=0.95 -multiple {input.bed:q} {input.chain:q} "
        "{output.mapped:q} {output.unmapped:q} 2> {log:q}"


rule chain_contract:
    input:
        requests=PROJECTION_REQUESTS,
        mapped=f"{RESULTS}/chains/{{species}}/mapped.bed",
        unmapped=f"{RESULTS}/chains/{{species}}/unmapped.bed",
        sizes=lambda w: asset_input(ASSETS[w.species]["target_sizes"]),
        species=SPECIES_SELECTED_INPUT,
        validated=f"{RESULTS}/chains/{{species}}/validated.json",
    output:
        accepted=f"{RESULTS}/chains/{{species}}/accepted.parquet",
        rejected=f"{RESULTS}/chains/{{species}}/rejected.parquet",
        audit=f"{RESULTS}/chains/{{species}}/audit.json",
    wildcard_constraints:
        species=SPECIES_RE,
    resources:
        mem_mb=int(config["table_mem_mb"]),
    run:
        write_chain_projections(
            input.requests,
            input.mapped,
            input.unmapped,
            input.sizes,
            input.species,
            wildcards.species,
            input.validated,
            output.accepted,
            output.rejected,
            output.audit,
        )


rule genome_source_validation:
    input:
        sequence=lambda w: asset_input(GENOMES[w.genome]["sequence"]),
        sizes=lambda w: asset_input(GENOMES[w.genome]["chrom_sizes"]),
        provenance=ASSET_PROVENANCE,
    output:
        local(f"{RESULTS}/reference/{{genome}}.source.json"),
    wildcard_constraints:
        genome=GENOME_RE,
    resources:
        mem_mb=1000,
    run:
        validate_genome_source(
            input.sequence, input.sizes, GENOMES[wildcards.genome], output[0]
        )


rule genome_twobit:
    input:
        sequence=lambda w: asset_input(GENOMES[w.genome]["sequence"]),
        validated=local(f"{RESULTS}/reference/{{genome}}.source.json"),
    output:
        local(f"{RESULTS}/reference/{{genome}}.2bit"),
    wildcard_constraints:
        genome=GENOME_RE,
    conda:
        "../envs/bioinformatics.yaml"
    resources:
        mem_mb=4000,
    params:
        format=lambda w: GENOMES[w.genome]["format"],
    shell:
        "if [ {params.format:q} = fasta ]; then "
        "faToTwoBit {input.sequence:q} {output:q}; "
        "else cp {input.sequence:q} {output:q}; fi"


rule genome_chrom_sizes:
    input:
        local(f"{RESULTS}/reference/{{genome}}.2bit"),
    output:
        f"{RESULTS}/reference/{{genome}}.chrom.sizes",
    wildcard_constraints:
        genome=GENOME_RE,
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitInfo {input:q} {output:q}"


rule genome_compatibility:
    input:
        sizes=f"{RESULTS}/reference/{{genome}}.chrom.sizes",
        pinned=lambda w: asset_input(GENOMES[w.genome]["chrom_sizes"]),
    output:
        f"{RESULTS}/reference/{{genome}}.compatibility.json",
    wildcard_constraints:
        genome=GENOME_RE,
    run:
        validate_genome_dictionary(input.sizes, input.pinned, output[0])


rule human_reference_sequences:
    input:
        anchors=ANCHOR_CATALOG_INPUT,
        requests=PROJECTION_REQUESTS,
        twobit=local(f"{RESULTS}/reference/hg38.2bit"),
        sizes=f"{RESULTS}/reference/hg38.chrom.sizes",
        compatibility=f"{RESULTS}/reference/hg38.compatibility.json",
    output:
        HUMAN_SEQUENCES,
    conda:
        "../envs/bioinformatics.yaml"
    resources:
        mem_mb=4000,
    shell:
        "uv run --locked python -m marin_dna_vertebrate_projection.sequence_cli "
        "human {input.anchors:q} {input.twobit:q} {input.sizes:q} {output:q}"


rule projected_genome_compatibility:
    input:
        accepted=f"{RESULTS}/chains/{{species}}/accepted.parquet",
        sizes=f"{RESULTS}/reference/{{species}}.chrom.sizes",
        validated=f"{RESULTS}/reference/{{species}}.compatibility.json",
    output:
        f"{RESULTS}/chains/{{species}}/sequence_compatibility.json",
    wildcard_constraints:
        species=SPECIES_RE,
    run:
        validate_projected_twobit_sizes(input.accepted, input.sizes, output[0])


rule chain_sequences:
    input:
        accepted=f"{RESULTS}/chains/{{species}}/accepted.parquet",
        twobit=local(f"{RESULTS}/reference/{{species}}.2bit"),
        compatibility=f"{RESULTS}/chains/{{species}}/sequence_compatibility.json",
    output:
        sequences=f"{RESULTS}/sequences/chains/{{species}}.parquet",
        rejected=f"{RESULTS}/chains/{{species}}/sequence_rejected.parquet",
    wildcard_constraints:
        species=SPECIES_RE,
    conda:
        "../envs/bioinformatics.yaml"
    resources:
        mem_mb=4000,
    shell:
        "uv run --locked python -m marin_dna_vertebrate_projection.sequence_cli "
        "projected {input.accepted:q} {input.twobit:q} {output.sequences:q} {output.rejected:q}"


rule combine_sequences:
    input:
        [HUMAN_SEQUENCES]
        + expand(
            f"{RESULTS}/sequences/chains/{{species}}.parquet", species=ACTIVE_SPECIES
        ),
    output:
        COMBINED_SEQUENCES,
    resources:
        mem_mb=8000,
    run:
        combine_sequence_parquets(list(input), output[0])
