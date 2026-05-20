FINEMAPPING_METHODS = ["SuSiE", "FINEMAP"]
FINEMAPPING_PIP_DIFF_THRESHOLD = 0.05
COMPLEX_TRAITS = pl.read_csv("config/complex_traits.csv")["trait"].to_list()


rule complex_traits_download_finemapping:
    output:
        "results/complex_traits/finemapping/{trait}/{method}.parquet",
    wildcard_constraints:
        method="|".join(FINEMAPPING_METHODS),
    params:
        url=lambda wc: f"https://huggingface.co/datasets/{config['complex_traits']['finemapping_repo']}/resolve/main/UKBB.{wc.trait}.{wc.method}.tsv.bgz",
    run:
        (
            pl.read_csv(
                params.url,
                separator="\t",
                null_values=["NA"],
                schema_overrides={"chromosome": pl.String},
                columns=[
                    "chromosome",
                    "position",
                    "allele1",
                    "allele2",
                    "rsid",
                    "pip",
                ],
            )
            .rename(
                {
                    "chromosome": "chrom",
                    "position": "pos",
                    "allele1": "ref",
                    "allele2": "alt",
                }
            )
            .pipe(filter_snp)
            .write_parquet(output[0])
        )


rule complex_traits_combine_methods:
    input:
        susie="results/complex_traits/finemapping/{trait}/SuSiE.parquet",
        finemap="results/complex_traits/finemapping/{trait}/FINEMAP.parquet",
    output:
        "results/complex_traits/finemapping/{trait}/combined.parquet",
    run:
        pip_defined_in_both_methods = (
            pl.col("pip_susie").is_not_null() & pl.col("pip_finemap").is_not_null()
        )
        (
            pl.read_parquet(input.susie)
            .join(
                pl.read_parquet(input.finemap),
                on=COORDINATES,
                how="full",
                suffix="_finemap",
            )
            .rename({"pip": "pip_susie", "rsid": "rsid_susie"})
            .with_columns(
                *(pl.coalesce(col, f"{col}_finemap").alias(col) for col in COORDINATES),
                pl.coalesce("rsid_susie", "rsid_finemap").alias("rsid"),
                pl.when(pip_defined_in_both_methods)
                .then(
                    pl.when(
                        (pl.col("pip_susie") - pl.col("pip_finemap")).abs()
                        <= FINEMAPPING_PIP_DIFF_THRESHOLD
                    )
                    .then((pl.col("pip_susie") + pl.col("pip_finemap")) / 2)
                    .otherwise(pl.lit(None))
                )
                .otherwise(pl.coalesce("pip_susie", "pip_finemap"))
                .alias("pip"),
            )
            .select([*COORDINATES, "rsid", "pip"])
            .write_parquet(output[0])
        )


rule complex_traits_aggregate_traits:
    input:
        expand(
            "results/complex_traits/finemapping/{trait}/combined.parquet",
            trait=COMPLEX_TRAITS,
        ),
    output:
        "results/complex_traits/finemapping/aggregated.parquet",
    run:
        high = config["complex_traits"]["pip_pos_threshold"]
        low = config["complex_traits"]["pip_neg_threshold"]
        # Per-variant labeling delegated to `label_variants_by_pip` (in
        # `src/marin_dna/evals/labeling.py`, fully unit-tested in
        # `tests/evals/test_labeling.py`). Notes:
        #   - Labels come from `max(pip)` across the traits that
        #     fine-mapped this variant. Intermediate `max(pip)`
        #     (in `[low, high]`) is excluded by the cascade's
        #     `otherwise(None)` + `filter(label.is_not_null())`. Don't
        #     row-filter extreme PIPs before the helper — that would
        #     mislabel `[low_pip, mid_pip]` variants as clean negatives.
        #   - `use_null_pip_guard=True`: SuSiE/FINEMAP combine-step
        #     disagreement (`|pip_susie - pip_finemap| > 0.05` in
        #     `complex_traits_combine_methods`) sets pip to null. Any null
        #     pip among the tested traits forbids the variant from being
        #     a confident negative.
        #   - A variant only has rows for the traits that fine-mapped
        #     it. Negatives are NOT required to be tested in all 119
        #     traits — typical fine-mapping outputs only cover
        #     significant regions.
        # Streaming: scan_parquet + sink_parquet so polars can chunk
        # through the 119-trait concat + group_by without materializing
        # all ~50 GB of per-trait fine-mapping data at once.
        labeled = label_variants_by_pip(
            pl.concat(
                [
                    pl.scan_parquet(path).with_columns(trait=pl.lit(trait))
                    for path, trait in zip(input, COMPLEX_TRAITS)
                ]
            ),
            pip_pos_threshold=high,
            pip_neg_threshold=low,
            use_null_pip_guard=True,
            extra_aggs=[
                pl.col("rsid").first(),
                pl.col("trait").filter(pl.col("pip") > high).unique(),
            ],
        )
        (
            labeled.with_columns(
                pl.col("trait").list.sort().list.join(",").alias("traits")
            )
            .drop("trait")
            .sort(COORDINATES)
            .sink_parquet(output[0])
        )


rule complex_traits_annotate:
    input:
        "results/complex_traits/finemapping/aggregated.parquet",
        "results/ldscore/UKBB.EUR.ldscore.parquet",
        genome="results/genome.fa.gz",
        consequences=expand("results/consequences/{chrom}.parquet", chrom=CHROMS),
    output:
        "results/complex_traits/annotated.parquet",
    run:
        ldscore = pl.read_parquet(input[1], columns=COORDINATES + ["MAF", "ld_score"])
        genome = Genome(input.genome)
        V = (
            pl.read_parquet(input[0])
            .join(ldscore, on=COORDINATES, how="left")
            # Drops high-PIP variants with very low MAF not in the LD-score file
            .filter(pl.col("ld_score").is_not_null())
            .pipe(lift_hg19_to_hg38)
            .filter(pl.col("pos") != -1)
            .pipe(filter_chroms)
            .pipe(check_ref_alt, genome)
            .sort(COORDINATES)
        )
        attach_per_chrom_consequences(V, list(input.consequences), CHROMS).write_parquet(
            output[0]
        )


rule complex_traits_dataset_all:
    input:
        "results/complex_traits/annotated.parquet",
        "results/intervals/exon_pc.parquet",
        "results/intervals/exon_nc.parquet",
        "results/intervals/tss_pc.parquet",
        "results/intervals/tss_nc.parquet",
    output:
        "results/complex_traits/dataset_all.parquet",
    run:
        build_dataset(
            pl.read_parquet(input[0]),
            exon_pc=pl.read_parquet(input[1]),
            exon_nc=pl.read_parquet(input[2]),
            tss_pc=pl.read_parquet(input[3]),
            tss_nc=pl.read_parquet(input[4]),
            exclude_consequences=config["exclude_consequences"],
            exon_proximal_dist=config["exon_proximal_dist"],
            tss_proximal_dist=config["tss_proximal_dist"],
            consequence_groups=config["consequence_groups"],
        ).write_parquet(output[0])


# complex_traits uses the BASE scheme as-is — no leak in distal that
# warrants the extra bin mendelian adds.
COMPLEX_DISTANCE_BIN_SCHEME = BASE_DISTANCE_BIN_SCHEME


rule complex_traits_dataset:
    input:
        "results/complex_traits/dataset_all.parquet",
    output:
        "results/dataset_unsplit/complex_traits.parquet",
    run:
        # RobustScaler in matching needs finite MAF; filter null/NaN up-front.
        V = add_subset_distance_bins_v2(
            pl.read_parquet(input[0]).filter(
                pl.col("MAF").is_finite() & pl.col("MAF").is_not_null()
            ),
            COMPLEX_DISTANCE_BIN_SCHEME,
        )
        (
            match_features(
                V.filter(pl.col("label")),
                V.filter(~pl.col("label")),
                [
                    "distance_tss_pc",
                    "distance_tss_nc",
                    "distance_exon_pc",
                    "distance_exon_nc",
                    "MAF",
                ],
                CAT_BASE
                + [
                    "distance_tss_pc_bin",
                    "distance_exon_pc_bin",
                ],
                k=9,
            )
            .with_columns(subset=pl.col("consequence_group"))
            .write_parquet(output[0])
        )
