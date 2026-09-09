"""Additional RAG targets; legacy projection and publication rules are unchanged."""

from marin_dna_vertebrate_projection.rag.pipeline import (
    assemble_outputs,
    compile_requests,
    download_benchmark,
)

if config.get("rag"):
    RAG = config["rag"]
    RAG_RESULTS = f"{RESULTS}/rag"
    RAG_REGIONS = sorted(RAG["catalogs"])
    RAG_BENCHMARKS = sorted(RAG["benchmarks"])
    RAG_UNION = f"{RAG_RESULTS}/anchors/catalog.parquet"
    RAG_REQUESTS = f"{RAG_RESULTS}/anchors/requests.parquet"
    RAG_CATALOGS = {name: asset_input(spec["uri"]) for name, spec in RAG["catalogs"].items()}
    RAG_BENCHMARK_FILES = {name: f"{RAG_RESULTS}/sources/{name}.parquet" for name in RAG_BENCHMARKS}

    rule rag_benchmark_source:
        output:
            f"{RAG_RESULTS}/sources/{{benchmark}}.parquet",
        wildcard_constraints:
            benchmark="|".join(RAG_BENCHMARKS),
        run:
            download_benchmark(RAG["benchmarks"][wildcards.benchmark], output[0])

    rule rag_union_catalog:
        input:
            catalogs=RAG_CATALOGS.values(),
            benchmarks=RAG_BENCHMARK_FILES.values(),
            sizes=asset_input(GENOMES["hg38"]["chrom_sizes"]),
            provenance=ASSET_PROVENANCE,
        output:
            catalog=RAG_UNION,
            memberships=f"{RAG_RESULTS}/anchors/source_memberships.parquet",
            audit=f"{RAG_RESULTS}/anchors/request_audit.json",
        resources:
            mem_mb=int(config["table_mem_mb"]),
        run:
            compile_requests(
                dict(zip(RAG_CATALOGS, input.catalogs, strict=True)),
                dict(zip(RAG_BENCHMARK_FILES, input.benchmarks, strict=True)),
                RAG, ASSETS, GENOMES, input.sizes,
                output.catalog, output.memberships, output.audit,
            )

    use rule chain_requests as rag_chain_requests with:
        input:
            anchors=RAG_UNION,
            sizes=asset_input(SOURCE["source_sizes"]),
            human_sizes=asset_input(GENOMES["hg38"]["chrom_sizes"]),
            provenance=ASSET_PROVENANCE,
        output:
            requests=RAG_REQUESTS,
            bed=f"{RAG_RESULTS}/anchors/centers.bed",

    use rule chain_liftover as rag_chain_liftover with:
        input:
            bed=f"{RAG_RESULTS}/anchors/centers.bed",
            chain=lambda w: asset_input(ASSETS[w.species]["chain"]),
            validated=f"{RESULTS}/chains/{{species}}/validated.json",
        output:
            mapped=f"{RAG_RESULTS}/chains/{{species}}/mapped.bed",
            unmapped=f"{RAG_RESULTS}/chains/{{species}}/unmapped.bed",
        log:
            f"{RAG_RESULTS}/chains/{{species}}/liftover.log",
        benchmark:
            f"{RAG_RESULTS}/chains/{{species}}/liftover.benchmark.tsv"

    use rule chain_contract as rag_chain_contract with:
        input:
            requests=RAG_REQUESTS,
            mapped=f"{RAG_RESULTS}/chains/{{species}}/mapped.bed",
            unmapped=f"{RAG_RESULTS}/chains/{{species}}/unmapped.bed",
            sizes=lambda w: asset_input(ASSETS[w.species]["target_sizes"]),
            species=SPECIES_SELECTED_INPUT,
            validated=f"{RESULTS}/chains/{{species}}/validated.json",
        output:
            accepted=f"{RAG_RESULTS}/chains/{{species}}/accepted.parquet",
            rejected=f"{RAG_RESULTS}/chains/{{species}}/rejected.parquet",
            audit=f"{RAG_RESULTS}/chains/{{species}}/audit.json",

    use rule human_reference_sequences as rag_human_sequences with:
        input:
            anchors=RAG_UNION,
            requests=RAG_REQUESTS,
            twobit=local(f"{RESULTS}/reference/hg38.2bit"),
            sizes=f"{RESULTS}/reference/hg38.chrom.sizes",
            compatibility=f"{RESULTS}/reference/hg38.compatibility.json",
        output:
            f"{RAG_RESULTS}/sequences/hg38.parquet",

    use rule projected_genome_compatibility as rag_projected_compatibility with:
        input:
            accepted=f"{RAG_RESULTS}/chains/{{species}}/accepted.parquet",
            sizes=f"{RESULTS}/reference/{{species}}.chrom.sizes",
            validated=f"{RESULTS}/reference/{{species}}.compatibility.json",
        output:
            f"{RAG_RESULTS}/chains/{{species}}/sequence_compatibility.json",

    use rule chain_sequences as rag_chain_sequences with:
        input:
            accepted=f"{RAG_RESULTS}/chains/{{species}}/accepted.parquet",
            twobit=local(f"{RESULTS}/reference/{{species}}.2bit"),
            compatibility=f"{RAG_RESULTS}/chains/{{species}}/sequence_compatibility.json",
        output:
            sequences=f"{RAG_RESULTS}/sequences/{{species}}.parquet",
            rejected=f"{RAG_RESULTS}/chains/{{species}}/sequence_rejected.parquet",

    rule rag_documents:
        input:
            catalogs=RAG_CATALOGS.values(),
            benchmarks=RAG_BENCHMARK_FILES.values(),
            union=RAG_UNION,
            human=f"{RAG_RESULTS}/sequences/hg38.parquet",
            sequences=expand(f"{RAG_RESULTS}/sequences/{{species}}.parquet", species=ACTIVE_SPECIES),
            rejections=expand(f"{RAG_RESULTS}/chains/{{species}}/rejected.parquet", species=ACTIVE_SPECIES),
            sequence_rejections=expand(f"{RAG_RESULTS}/chains/{{species}}/sequence_rejected.parquet", species=ACTIVE_SPECIES),
        output:
            train=expand(f"{RAG_RESULTS}/datasets/{{region}}/train.parquet", region=RAG_REGIONS),
            validation=expand(f"{RAG_RESULTS}/datasets/{{region}}/validation.parquet", region=RAG_REGIONS),
            summary=f"{RAG_RESULTS}/datasets/split_summary.json",
            harness=f"{RAG_RESULTS}/evaluation/combined_development.parquet",
            harness_counts=f"{RAG_RESULTS}/evaluation/combined_development.counts.json",
        resources:
            mem_mb=16000,
        run:
            assemble_outputs(
                dict(zip(RAG_CATALOGS, input.catalogs, strict=True)),
                dict(zip(RAG_BENCHMARK_FILES, input.benchmarks, strict=True)), RAG,
                input.union, input.human,
                dict(zip(ACTIVE_SPECIES, input.sequences, strict=True)),
                dict(zip(ACTIVE_SPECIES, input.rejections, strict=True)),
                dict(zip(ACTIVE_SPECIES, input.sequence_rejections, strict=True)),
                str(Path(output.summary).parent), output.harness,
            )

    rule rag_all_documents:
        input:
            PRODUCER_MANIFEST,
            f"{RAG_RESULTS}/datasets/split_summary.json",
            f"{RAG_RESULTS}/evaluation/combined_development.parquet",
