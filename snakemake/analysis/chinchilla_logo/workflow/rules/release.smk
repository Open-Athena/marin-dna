"""Construct BigWigs, UCSC hub, dataset card, and release manifest."""

from marin_dna.pipelines.chinchilla_logo import (
    list_score_shards,
    write_bigwig_sets,
    write_dataset_readme,
    write_release_manifest,
    write_ucsc_hub,
)


rule build_bigwigs:
    input:
        done=SCORE_DONE_OUTPUTS,
        chrom_sizes=CHROM_SIZES,
    output:
        BIGWIG_OUTPUTS,
    run:
        shard_paths = list_score_shards(
            [f"results/shards/{scaffold}" for scaffold in SCAFFOLDS]
        )
        written = write_bigwig_sets(
            shard_paths,
            input.chrom_sizes,
            "results/release",
        )
        assert {str(path) for path in written.values()} == set(output)


rule write_hub:
    input:
        BIGWIG_OUTPUTS,
    output:
        hub="results/release/ucsc/hub.txt",
        genomes="results/release/ucsc/genomes.txt",
        track_db=f"results/release/ucsc/{ASSEMBLY}/trackDb.txt",
    run:
        paths = write_ucsc_hub("results/release", assembly_accession=ASSEMBLY)
        assert {str(path) for path in paths.values()} == set(output)


rule write_dataset_card:
    output:
        "results/release/README.md",
    params:
        application_commit=GIT_COMMIT_SHA,
    run:
        write_dataset_readme(
            output[0],
            application_commit=params.application_commit,
            model_repository=config["model"]["repository"],
            model_revision=config["model"]["revision"],
            assembly_accession=ASSEMBLY,
        )


rule write_manifest:
    input:
        bigwigs=BIGWIG_OUTPUTS,
        hub="results/release/ucsc/hub.txt",
        genomes="results/release/ucsc/genomes.txt",
        track_db=f"results/release/ucsc/{ASSEMBLY}/trackDb.txt",
        readme="results/release/README.md",
        plans=PLAN_METADATA_OUTPUTS,
        runtimes=RUNTIME_OUTPUTS,
        chrom_sizes=CHROM_SIZES,
    output:
        "results/release/manifest/release.json",
    params:
        application_commit=GIT_COMMIT_SHA,
    run:
        write_release_manifest(
            "results/release",
            input.chrom_sizes,
            input.plans,
            input.runtimes,
            application_commit=params.application_commit,
            model_repository=config["model"]["repository"],
            model_revision=config["model"]["revision"],
            assembly_accession=ASSEMBLY,
            context_size=CONTEXT_SIZE,
            stride=STRIDE,
            retain_start=RETAIN_START,
            retain_end=RETAIN_END,
        )
