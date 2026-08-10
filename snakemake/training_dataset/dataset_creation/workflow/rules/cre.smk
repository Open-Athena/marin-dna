rule cre_download:
    output:
        temp("results/cre/all.tsv"),
    shell:
        "wget -O {output} https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed"


rule cre_process:
    input:
        "results/cre/all.tsv",
    output:
        "results/cre/all.parquet",
    run:
        (
            pl.read_csv(
                input[0],
                separator="\t",
                has_header=False,
                columns=[0, 1, 2, 5],
                new_columns=["chrom", "start", "end", "cre_class"],
            )
            .with_columns(pl.col("chrom").str.replace("chr", ""))
            .filter(pl.col("chrom").is_in(STANDARD_CHROMS))
            .write_parquet(output[0])
        )


rule cre_filter_enhancers:
    input:
        "results/cre/all.parquet",
    output:
        "results/cre/ELS.parquet",
    run:
        (
            pl.read_parquet(input[0])
            .filter(pl.col("cre_class").is_in(ENHANCER_CRE_CLASSES))
            .select(["chrom", "start", "end"])
            .write_parquet(output[0])
        )


rule cre_conservation:
    input:
        cre="results/cre/ELS.parquet",
        conservation="results/conservation/cactus241way.phyloP.bw",
    output:
        "results/cre/ELS_cactus241way.phyloP.parquet",
    run:
        phylop_cutoff = config["conservation"]["phylop_cutoff"]
        df = pl.read_parquet(input.cre)
        bw = pyBigWig.open(input.conservation)
        pct_conserved = df.select(
            pl.struct(["chrom", "start", "end"])
            .map_elements(
                lambda x: np.mean(
                    bw.values("chr" + x["chrom"], x["start"], x["end"], numpy=True)
                    >= phylop_cutoff
                ),
                return_dtype=pl.Float64,
            )
            .cast(pl.Float32)
            .alias("pct_conserved")
        )
        bw.close()
        df.hstack(pct_conserved).write_parquet(output[0])


rule cre_filter_conserved_enhancers:
    input:
        "results/cre/ELS_cactus241way.phyloP.parquet",
    output:
        "results/cre/ELS_conserved_{n}.parquet",
    run:
        min_conserved = int(wildcards.n)
        df = pl.read_parquet(input[0])
        size = df["end"] - df["start"]
        (
            df.filter((df["pct_conserved"] * size) >= min_conserved)
            .select(["chrom", "start", "end"])
            .write_parquet(output[0])
        )


# Parallel of cre_conservation but using the Zoonomia 43-primate phastCons
# track. Threshold (0.961) is calibrated to ~3.5% conserved-base fraction,
# matching phyloP-241way 2.27 for cross-track comparability.
rule cre_conservation_phastcons_43p:
    input:
        cre="results/cre/ELS.parquet",
        conservation="results/conservation/phastCons_43p.bw",
    output:
        "results/cre/ELS_phastCons_43p.parquet",
    run:
        cutoff = config["conservation"]["phastcons_43p_cutoff"]
        df = pl.read_parquet(input.cre)
        bw = pyBigWig.open(input.conservation)
        pct_conserved = df.select(
            pl.struct(["chrom", "start", "end"])
            .map_elements(
                lambda x: np.mean(
                    bw.values("chr" + x["chrom"], x["start"], x["end"], numpy=True)
                    >= cutoff
                ),
                return_dtype=pl.Float64,
            )
            .cast(pl.Float32)
            .alias("pct_conserved")
        )
        bw.close()
        df.hstack(pct_conserved).write_parquet(output[0])


rule cre_filter_phastcons_43p_conserved_enhancers:
    input:
        "results/cre/ELS_phastCons_43p.parquet",
    output:
        "results/cre/ELS_phastCons_43p_conserved_{n}.parquet",
    run:
        min_conserved = int(wildcards.n)
        df = pl.read_parquet(input[0])
        size = df["end"] - df["start"]
        (
            df.filter((df["pct_conserved"] * size) >= min_conserved)
            .select(["chrom", "start", "end"])
            .write_parquet(output[0])
        )
