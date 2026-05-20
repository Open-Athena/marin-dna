ENHANCER_CONFIG = config["enhancer_prediction"]
SEGMENTATION_CONFIG = config["enhancer_prediction_segmentation"]


wildcard_constraints:
    g="[^/]+",


rule extract_exons:
    input:
        "results/annotation/{g}.gtf.gz",
    output:
        "results/intervals/exons/{g}.parquet",
    run:
        ann = load_annotation(input[0])
        exons = get_exons_for_masking(ann)
        exons.write_parquet(output[0])


rule scannable_regions:
    input:
        defined="results/intervals/defined/{g}.bed.gz",
        exons="results/intervals/exons/{g}.parquet",
    output:
        "results/intervals/scannable/{g}.bed.gz",
    run:
        defined = GenomicSet.read_bed(input.defined)
        exons = GenomicSet.read_parquet(input.exons)
        scannable = defined - exons
        scannable.write_bed(output[0])


rule enhancer_prediction_windows:
    input:
        "results/intervals/scannable/{g}.bed.gz",
    output:
        "results/enhancer_predictions/windows/{g}.parquet",
    run:
        window_size = ENHANCER_CONFIG["window_size"]
        step_size = ENHANCER_CONFIG["step_size"]

        regions = pd.read_csv(
            input[0], sep="\t", header=None, names=["chrom", "start", "end"],
            dtype={"chrom": str},
        )

        chroms, starts, ends = [], [], []
        for _, row in regions.iterrows():
            for w_start, w_end in sliding_windows(
                row["start"], row["end"], window_size, step_size
            ):
                chroms.append(row["chrom"])
                starts.append(w_start)
                ends.append(w_end)

        windows = pl.DataFrame({
            "chrom": chroms,
            "start": starts,
            "end": ends,
        })
        windows.write_parquet(output[0])


rule predict_enhancers:
    input:
        genome="results/genome/{g}.2bit",
        windows="results/enhancer_predictions/windows/{g}.parquet",
        checkpoint=storage(ENHANCER_CONFIG["checkpoint"]),
    output:
        "results/enhancer_predictions/{g}.parquet",
    params:
        batch_size=ENHANCER_CONFIG["batch_size"],
        num_workers=ENHANCER_CONFIG["num_workers"],
    threads: workflow.cores
    shell:
        """
        uv run python -m marin_dna.pipelines.enhancer_classification.predict_genome \
            --genome {input.genome} \
            --checkpoint {input.checkpoint} \
            --windows {input.windows} \
            --batch-size {params.batch_size} \
            --num-workers {params.num_workers} \
            --output {output}
        """


rule segmentation_prediction_windows:
    input:
        "results/genome/{g}.2bit",
    output:
        "results/enhancer_predictions_segmentation/windows/{g}.parquet",
    run:
        tb = py2bit.open(input[0])
        chrom_sizes = tb.chroms()
        tb.close()

        tiles = tile_chromosomes(chrom_sizes, SEGMENTATION_CONFIG["window_size"])
        pl.DataFrame(tiles, schema=["chrom", "start", "end"], orient="row").write_parquet(
            output[0]
        )


rule predict_enhancers_segmentation:
    input:
        genome="results/genome/{g}.2bit",
        windows="results/enhancer_predictions_segmentation/windows/{g}.parquet",
        checkpoint=storage(SEGMENTATION_CONFIG["checkpoint"]),
    output:
        "results/enhancer_predictions_segmentation/{g}.parquet",
    params:
        bin_size=SEGMENTATION_CONFIG["bin_size"],
        batch_size=SEGMENTATION_CONFIG["batch_size"],
        num_workers=SEGMENTATION_CONFIG["num_workers"],
        max_windows=SEGMENTATION_CONFIG["max_windows"],
    threads: workflow.cores
    shell:
        # Persist the torch.compile cache across per-genome invocations so
        # every genome after the first skips the ~30s Inductor compile step.
        """
        TORCHINDUCTOR_CACHE_DIR=.snakemake/torchinductor_cache \
        uv run python -m marin_dna.pipelines.enhancer_segmentation.predict_genome \
            --genome {input.genome} \
            --checkpoint {input.checkpoint} \
            --windows {input.windows} \
            --bin-size {params.bin_size} \
            --batch-size {params.batch_size} \
            --num-workers {params.num_workers} \
            --max-windows {params.max_windows} \
            --output {output}
        """


rule all_enhancer_predictions_segmentation:
    input:
        expand(
            "results/enhancer_predictions_segmentation/{g}.parquet",
            g=SEGMENTATION_CONFIG["genomes"],
        ),


rule intervals_recipe_v19:
    input:
        "results/enhancer_predictions/{g}.parquet",
    output:
        "results/intervals/recipe/v19/{g}.bed.gz",
    run:
        threshold = ENHANCER_CONFIG["threshold"]
        df = pl.read_parquet(input[0])
        df = df.filter(pl.col("logit") >= threshold)
        intervals = GenomicSet(df.select(["chrom", "start", "end"]))
        intervals.write_bed(output[0])


# Segmentation-based enhancer recipe: top-quantile bins from the per-bin
# segmentation predictions (#118 / PR #126), each resized to a fixed window
# centered on its midpoint, then exon-masked and clipped to defined regions.
rule intervals_recipe_v20:
    input:
        predictions="results/enhancer_predictions_segmentation/{g}.parquet",
        exons="results/intervals/exons/{g}.parquet",
        defined="results/intervals/defined/{g}.bed.gz",
    output:
        "results/intervals/recipe/v20/{g}.bed.gz",
    run:
        top_quantile = SEGMENTATION_CONFIG["recipe_top_quantile"]
        target_size = SEGMENTATION_CONFIG["recipe_target_size"]

        bin_logits = pl.read_parquet(input.predictions)
        windows = top_quantile_bins_to_windows(
            bin_logits, top_quantile=top_quantile, target_size=target_size
        )
        exons = GenomicSet.read_parquet(input.exons)
        defined = GenomicSet.read_bed(input.defined)
        windows = (windows - exons) & defined
        windows.write_bed(output[0])
