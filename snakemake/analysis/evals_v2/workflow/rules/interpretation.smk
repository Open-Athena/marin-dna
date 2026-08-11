"""Nucleotide dependency maps (categorical Jacobian) — interpretation, issue #237.

GPU-bound compute rule + a plot rule. These targets are intentionally kept OUT
of the default ``rule all`` (which is metrics-only), so they never perturb
score/metric reruns. Build them explicitly:

    snakemake nuc_dep                                   # all (locus, model) plots
    snakemake results/plots/nuc_dep/LDLR/<model>.svg    # one map

The compute rule reuses the existing ``download_model`` rule for the checkpoint
and reads the GRCh38 reference straight from S3, exactly like ``compute_scores``.
"""


rule compute_nuc_dep:
    input:
        checkpoint="results/checkpoints/{model}",
    output:
        "results/interpretation/nuc_dep/{combine}/{locus}/{model}.parquet",
    wildcard_constraints:
        combine="|".join(NUC_DEP_COMBINES),
        model="|".join(NUC_DEP_MODELS),
        locus="|".join(NUC_DEP_LOCI),
    threads: config["inference"]["num_workers"]
    params:
        # Output-affecting fields only (tracked by snakemake's `params` rerun
        # trigger). batch_size is execution-only and read inside `run:`.
        chrom=lambda wc: str(config["nuc_dep"]["loci"][wc.locus]["chrom"]),
        start=lambda wc: config["nuc_dep"]["loci"][wc.locus]["start"],
        end=lambda wc: config["nuc_dep"]["loci"][wc.locus]["end"],
        strand=lambda wc: config["nuc_dep"]["loci"][wc.locus]["strand"],
        window_size=lambda wc: get_nuc_dep_window(wc.model),
        norm_ord=get_nuc_dep_ord(),
    run:
        from marin_dna_evals.interpretation import compute_dependency_map

        df = compute_dependency_map(
            checkpoint_path=input.checkpoint,
            # S3 URI; pyfaidx + fsspec/s3fs reads sequence by byte-range.
            genome_path=config["genome_path"],
            chrom=params.chrom,
            start=int(params.start),
            end=int(params.end),
            strand=params.strand,
            window_size=int(params.window_size),
            combine=wildcards.combine,  # mean | max — from the output path
            norm_ord=params.norm_ord,
            batch_size=config["nuc_dep"].get("batch_size", 32),
        )
        n = int(params.end) - int(params.start)
        assert df.shape == (n, n), f"expected {n}x{n} map, got {df.shape}"
        df.to_parquet(output[0])
        print(
            f"[nuc_dep] {wildcards.model} {wildcards.locus}: {df.shape[0]}x{df.shape[1]}"
        )


rule plot_nuc_dep:
    input:
        "results/interpretation/nuc_dep/{combine}/{locus}/{model}.parquet",
    output:
        "results/plots/nuc_dep/{combine}/{locus}/{model}.svg",
    run:
        from marin_dna_evals.interpretation import plot_dependency_map

        df = pd.read_parquet(input[0])
        # parquet stringifies column names; restore int genomic coordinates.
        df.columns = df.columns.astype(int)
        df.index = df.index.astype(int)
        # No title: the dashboard labels each map with the model line, and a
        # per-model title would also make stacked maps crop to different widths
        # (#240). The locus/model/combine live in the output path.
        plot_dependency_map(
            df,
            output[0],
            chrom=str(config["nuc_dep"]["loci"][wildcards.locus]["chrom"]),
            dpi=config["nuc_dep"].get("dpi", 150),
        )


rule nuc_dep:
    """Aggregate convenience target: every (combine, locus, model) dependency-map
    plot. Not part of `rule all`."""
    input:
        [
            f"results/plots/nuc_dep/{combine}/{locus}/{model}.svg"
            for combine in NUC_DEP_COMBINES
            for model in NUC_DEP_MODELS
            for locus in NUC_DEP_LOCI
        ],
