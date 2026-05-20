"""Per-bin enhancer segmentation (issue #115).

Tiles the genome into non-overlapping fixed-size windows, labels each
``bin_size`` bin by fractional overlap with conserved enhancers, subsamples
non-enhancer-containing windows for training (val keeps full coverage), and
trains a Conv1d-head segmenter. Window size is parameterized per dataset
via the `w{window_size}` path segment and a per-dataset `window_size` key
in `seg_datasets`.
"""

SEG_CFG = config["segmentation"]
SEG_BIN_SIZE = SEG_CFG["bin_size"]
SEG_THRESHOLD = SEG_CFG["bin_overlap_threshold"]
SEG_BG_MAX = SEG_CFG["background_fraction_max"]


def _seg_dataset_window_size(dataset: str) -> int:
    return int(config["seg_datasets"][dataset]["window_size"])


def _num_bins_for(window_size: int) -> int:
    return window_size // SEG_BIN_SIZE


rule tile_genome_segmentation:
    input:
        chrom_sizes="results/genome/{species}.chrom.sizes.filtered",
        genome_bed="results/genome/{species}.genome.bed",
        undefined="results/genome/{species}.undefined.bed",
    output:
        "results/segmentation/w{window_size}/intervals/{species}/windows.bed",
    run:
        ws = int(wildcards.window_size)
        chrom_sizes = pd.read_csv(
            input.chrom_sizes,
            sep="\t",
            header=None,
            names=["chrom", "size"],
            dtype={"chrom": str},
        )
        frames = []
        for chrom, size in zip(chrom_sizes["chrom"], chrom_sizes["size"]):
            starts = np.arange(0, size - ws + 1, ws)
            ends = starts + ws
            frames.append(pd.DataFrame({"chrom": chrom, "start": starts, "end": ends}))
        windows_df = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["chrom", "start", "end"])
        )

        genome_set = GenomicSet.read_bed(input.genome_bed)
        undefined = GenomicSet.read_bed(input.undefined)
        defined = genome_set - undefined
        windows = GenomicList(windows_df).filter_within(defined)
        windows.write_bed(output[0])


rule label_segmentation_windows:
    threads: workflow.cores
    input:
        windows="results/segmentation/w{window_size}/intervals/{species}/windows.bed",
        enhancers="results/cre/{species}/{intervals}.parquet",
    output:
        "results/segmentation/w{window_size}/intervals/{intervals}/{species}/labeled_windows.parquet",
    run:
        ws = int(wildcards.window_size)
        num_bins = _num_bins_for(ws)
        windows = pd.read_csv(
            input.windows,
            sep="\t",
            header=None,
            names=["chrom", "start", "end"],
            dtype={"chrom": str},
        )
        enhancers = pl.read_parquet(input.enhancers).to_pandas()

        label_matrix = label_windows_by_bin_overlap(
            windows,
            enhancers,
            bin_size=SEG_BIN_SIZE,
            num_bins=num_bins,
            threshold=SEG_THRESHOLD,
        )
        labels_list = label_matrix.tolist()
        n_pos = label_matrix.sum(axis=1).astype(np.int32)

        out = pl.DataFrame(
            {
                "chrom": pl.Series(windows["chrom"].tolist(), dtype=pl.Utf8),
                "start": pl.Series(windows["start"].to_numpy(), dtype=pl.Int64),
                "end": pl.Series(windows["end"].to_numpy(), dtype=pl.Int64),
                "labels": pl.Series(labels_list, dtype=pl.List(pl.UInt8)),
                "n_positive_bins": pl.Series(n_pos, dtype=pl.Int32),
            }
        )
        out.write_parquet(output[0])


rule subsample_segmentation_windows:
    threads: workflow.cores
    input:
        "results/segmentation/w{window_size}/intervals/{intervals}/{species}/labeled_windows.parquet",
    output:
        "results/segmentation/w{window_size}/intervals/{intervals}/{species}/sampled_windows.parquet",
    run:
        seed = config["seed"]
        df = pl.read_parquet(input[0])
        pos_mask = pl.col("n_positive_bins") >= 1
        pos_df = df.filter(pos_mask)
        neg_df = df.filter(~pos_mask)

        n_pos = pos_df.height
        n_neg = neg_df.height
        total = n_pos + n_neg
        # Keep all enhancer-containing windows. Cap non-enhancer windows so
        # they are at most SEG_BG_MAX fraction of the final dataset. If the
        # natural fraction is already <= SEG_BG_MAX, keep them all.
        if total == 0 or n_neg == 0:
            kept_neg = neg_df
        else:
            max_neg_from_frac = (
                int(np.floor(SEG_BG_MAX * n_pos / (1 - SEG_BG_MAX)))
                if SEG_BG_MAX < 1
                else n_neg
            )
            target_neg = min(n_neg, max_neg_from_frac)
            if target_neg >= n_neg:
                kept_neg = neg_df
            else:
                kept_neg = neg_df.sample(n=target_neg, seed=seed)

        sampled = pl.concat([pos_df, kept_neg]).sort(["chrom", "start"])
        sampled.write_parquet(output[0])


# {windows_kind} selects upstream: "sampled_windows" (post-subsample, used
# for train) or "labeled_windows" (full coverage, used for validation).
rule segmentation_windows_to_bed:
    wildcard_constraints:
        windows_kind="labeled_windows|sampled_windows",
    input:
        "results/segmentation/w{window_size}/intervals/{intervals}/{species}/{windows_kind}.parquet",
    output:
        temp(
            "results/segmentation/w{window_size}/intervals/{intervals}/{species}/{windows_kind}.4col.bed"
        ),
    run:
        df = pl.read_parquet(input[0], columns=["chrom", "start", "end"]).to_pandas()
        df["name"] = "."
        df.to_csv(output[0], sep="\t", header=False, index=False)


rule extract_segmentation_sequences:
    wildcard_constraints:
        windows_kind="labeled_windows|sampled_windows",
    input:
        twobit="results/genome/{species}.2bit",
        bed="results/segmentation/w{window_size}/intervals/{intervals}/{species}/{windows_kind}.4col.bed",
    output:
        temp(
            "results/segmentation/w{window_size}/sequences/{intervals}/{species}/{windows_kind}.fa"
        ),
    conda:
        "../envs/bioinformatics.yaml"
    shell:
        "twoBitToFa {input.twobit} {output} -bed={input.bed} -bedPos"


rule make_segmentation_species_parquet:
    threads: workflow.cores
    wildcard_constraints:
        windows_kind="labeled_windows|sampled_windows",
    input:
        windows="results/segmentation/w{window_size}/intervals/{intervals}/{species}/{windows_kind}.parquet",
        fa="results/segmentation/w{window_size}/sequences/{intervals}/{species}/{windows_kind}.fa",
    output:
        "results/segmentation/w{window_size}/parquet/{intervals}/{species}/{windows_kind}.parquet",
    run:
        ws = int(wildcards.window_size)
        labeled = pl.read_parquet(input.windows)
        seqs = load_fasta(input.fa)
        # FASTA ids are "chrom:start-end" produced by twoBitToFa -bedPos.
        seq_df = seqs.to_frame().reset_index(names="id")
        coords = seq_df["id"].str.split(":", expand=True)
        seq_df["chrom"] = coords[0].astype(str)
        start_end = coords[1].str.split("-", expand=True)
        seq_df["start"] = start_end[0].astype(int)
        seq_df["end"] = start_end[1].astype(int)
        seq_df = seq_df[["chrom", "start", "end", "seq"]]

        merged = labeled.to_pandas().merge(
            seq_df, on=["chrom", "start", "end"], how="inner"
        )
        assert len(merged) == len(
            labeled
        ), f"Sequence/label join dropped rows: {len(labeled)} -> {len(merged)}"
        assert (
            merged["seq"].str.len() == ws
        ).all(), f"Not all sequences are {ws} bp"
        # Defined-regions filter guarantees no Ns; assert to catch drift.
        assert (
            not merged["seq"].str.upper().str.contains("N").any()
        ), "Sequences contain N despite defined-regions filter"

        merged["strand"] = "+"
        merged["genome"] = wildcards.species
        merged["labels"] = labeled["labels"].to_list()

        out = merged[["genome", "chrom", "start", "end", "strand", "seq", "labels"]]
        pl.from_pandas(out).with_columns(
            pl.col("labels").cast(pl.List(pl.UInt8))
        ).write_parquet(output[0])


def _seg_dataset_parquets(wc) -> list[str]:
    cfg = config["seg_datasets"][wc.dataset]
    split_config = config["splits"][cfg["split"]]
    ws = _seg_dataset_window_size(wc.dataset)
    # All splits source from the (positive-enriched) subsampled windows. The
    # non-subsampled full-coverage val is built by a separate rule below so
    # existing trained checkpoints' train/val inputs remain byte-identical.
    return [
        f"results/segmentation/w{ws}/parquet/"
        f"{resolve_intervals(cfg['intervals'], species)}/"
        f"{species}/sampled_windows.parquet"
        for species, splits in split_config.items()
        if wc.split_name in splits
    ]


def _seg_dataset_fullcov_parquets(wc) -> list[str]:
    """Per-species parquets built from *labeled_windows* (no subsampling) so
    the validation split covers every bin of the validation chromosomes.
    Used only by the offline full-coverage eval rule."""
    cfg = config["seg_datasets"][wc.dataset]
    split_config = config["splits"][cfg["split"]]
    ws = _seg_dataset_window_size(wc.dataset)
    return [
        f"results/segmentation/w{ws}/parquet/"
        f"{resolve_intervals(cfg['intervals'], species)}/"
        f"{species}/labeled_windows.parquet"
        for species, splits in split_config.items()
        if "validation" in splits
    ]


rule build_segmentation_dataset:
    threads: workflow.cores
    input:
        _seg_dataset_parquets,
    output:
        "results/segmentation/dataset/{dataset}/{split_name}.parquet",
    run:
        # RC augmentation happens in SegmentationDataset via virtual length
        # doubling, not here. Baking RC into the parquet peaks at ~4-5x the
        # train data size during Python-based reverse_complement, which OOMs
        # on a 15 GB box for human+mouse.
        seed = config["seed"]
        dataset_config = config["seg_datasets"][wildcards.dataset]
        split_config = config["splits"][dataset_config["split"]]

        dfs = []
        for parquet_path in input:
            # Path ends in .../{species}/{windows_kind}.parquet — species is
            # the second-to-last segment.
            parts = parquet_path.rstrip("/").split("/")
            species = parts[-2]
            chroms = split_config[species][wildcards.split_name]
            df = pl.read_parquet(parquet_path)
            df = df.filter(pl.col("chrom").is_in(chroms))
            dfs.append(df)

        combined = pl.concat(dfs)

        max_samples = config.get("seg_max_samples", {}).get(wildcards.split_name)
        if max_samples and combined.height > max_samples:
            combined = combined.sample(n=max_samples, seed=seed)

        combined = combined.sample(fraction=1.0, seed=seed, shuffle=True)
        combined.write_parquet(output[0])


rule build_segmentation_fullcov_validation:
    """Assemble a validation parquet that covers every bin of the validation
    chromosomes (no subsampling, no max_samples cap). Used by the offline
    full-coverage eval rule so different window-size variants can be compared
    on the identical set of genomic bins."""
    threads: workflow.cores
    input:
        _seg_dataset_fullcov_parquets,
    output:
        "results/segmentation/dataset/{dataset}/validation_fullcov.parquet",
    run:
        dataset_config = config["seg_datasets"][wildcards.dataset]
        split_config = config["splits"][dataset_config["split"]]

        dfs = []
        for parquet_path in input:
            parts = parquet_path.rstrip("/").split("/")
            species = parts[-2]
            chroms = split_config[species]["validation"]
            df = pl.read_parquet(parquet_path)
            df = df.filter(pl.col("chrom").is_in(chroms))
            dfs.append(df)

        combined = pl.concat(dfs).sort(["genome", "chrom", "start"])
        combined.write_parquet(output[0])


def get_seg_model_config(model_name: str) -> dict:
    defaults = config["seg_models"]["default"]
    return {**defaults, **config["seg_models"].get(model_name, {})}


rule train_segmentation_model:
    threads: workflow.cores
    input:
        train="results/segmentation/dataset/{dataset}/train.parquet",
        val="results/segmentation/dataset/{dataset}/validation.parquet",
        weights=WEIGHTS_FILE,
    output:
        ckpt="results/segmentation/model/{model}/{dataset}/best.ckpt",
        metrics="results/segmentation/model/{model}/{dataset}/metrics.json",
        val_predictions="results/segmentation/model/{model}/{dataset}/val_predictions.parquet",
    params:
        freeze_flag=lambda wc: (
            "--freeze-backbone"
            if get_seg_model_config(wc.model)["freeze_backbone"]
            else "--no-freeze-backbone"
        ),
        learning_rate=lambda wc: get_seg_model_config(wc.model)["learning_rate"],
        weight_decay=lambda wc: get_seg_model_config(wc.model)["weight_decay"],
        gradient_clip_val=lambda wc: get_seg_model_config(wc.model)["gradient_clip_val"],
        batch_size=lambda wc: get_seg_model_config(wc.model)["batch_size"],
        warmup_fraction=lambda wc: get_seg_model_config(wc.model)["warmup_fraction"],
        max_steps=lambda wc: get_seg_model_config(wc.model).get("max_steps", -1),
        limit_train_batches=lambda wc: get_seg_model_config(wc.model).get(
            "limit_train_batches", 1.0
        ),
        limit_val_batches=lambda wc: get_seg_model_config(wc.model).get(
            "limit_val_batches", 1.0
        ),
        n_transformer_layers=lambda wc: get_seg_model_config(wc.model).get(
            "n_transformer_layers", 0
        ),
        seed=lambda wc: get_seg_model_config(wc.model).get("seed", config["seed"]),
        accumulate_grad_batches=lambda wc: get_seg_model_config(wc.model).get(
            "accumulate_grad_batches", 1
        ),
    shell:
        """
        uv run python -m marin_dna.pipelines.enhancer_segmentation.train \
            --train-parquet {input.train} \
            --val-parquet {input.val} \
            --weights-path {input.weights} \
            --output-ckpt {output.ckpt} \
            --output-metrics {output.metrics} \
            --output-val-predictions {output.val_predictions} \
            --learning-rate {params.learning_rate} \
            --weight-decay {params.weight_decay} \
            --batch-size {params.batch_size} \
            --gradient-clip-val {params.gradient_clip_val} \
            --warmup-fraction {params.warmup_fraction} \
            --max-steps {params.max_steps} \
            --limit-train-batches {params.limit_train_batches} \
            --limit-val-batches {params.limit_val_batches} \
            --n-transformer-layers {params.n_transformer_layers} \
            --accumulate-grad-batches {params.accumulate_grad_batches} \
            {params.freeze_flag} \
            --seed {params.seed} \
            --num-workers {threads} \
            --wandb-run seg-{wildcards.model}-{wildcards.dataset}
        """


rule evaluate_segmentation_model_fullcov:
    """Run a trained segmenter over the full-coverage validation parquet and
    write a val_predictions_fullcov.parquet compatible with the AUPRC /
    precision_recall tooling. Reuses an existing best.ckpt; no training."""
    threads: workflow.cores
    input:
        checkpoint="results/segmentation/model/{model}/{dataset}/best.ckpt",
        val="results/segmentation/dataset/{dataset}/validation_fullcov.parquet",
    output:
        val_predictions="results/segmentation/eval/{model}/{dataset}/val_predictions_fullcov.parquet",
    params:
        batch_size=lambda wc: get_seg_model_config(wc.model)["batch_size"],
    shell:
        """
        uv run python -m marin_dna.pipelines.enhancer_segmentation.evaluate \
            --checkpoint {input.checkpoint} \
            --val-parquet {input.val} \
            --output {output.val_predictions} \
            --batch-size {params.batch_size} \
            --num-workers {threads}
        """


SEG_VIS_CFG = config.get("segmentation_visualization", {})
SEG_VIS_REGIONS = SEG_VIS_CFG.get("regions", [])


def _get_seg_region(name: str) -> dict:
    for region in SEG_VIS_REGIONS:
        if region["name"] == name:
            return region
    raise ValueError(f"Unknown segmentation visualization region: {name}")


rule predict_segmentation_region:
    input:
        genome=lambda wc: f"results/genome/{_get_seg_region(wc.name)['genome']}.fa.gz",
        checkpoint="results/segmentation/model/{model}/{dataset}/best.ckpt",
    output:
        "results/segmentation/visualization/{model}/{dataset}/{name}.bedgraph",
    threads: workflow.cores
    params:
        chrom=lambda wc: _get_seg_region(wc.name)["chrom"],
        start=lambda wc: _get_seg_region(wc.name)["start"],
        end=lambda wc: _get_seg_region(wc.name)["end"],
        window_size=lambda wc: _seg_dataset_window_size(wc.dataset),
        bin_size=SEG_BIN_SIZE,
    shell:
        """
        uv run python -m marin_dna.pipelines.enhancer_segmentation.predict \
            --genome {input.genome} \
            --checkpoint {input.checkpoint} \
            --chrom {params.chrom} \
            --start {params.start} \
            --end {params.end} \
            --window-size {params.window_size} \
            --bin-size {params.bin_size} \
            --output {output} \
            --name "Seg enhancer prob ({wildcards.model}/{wildcards.dataset})"
        """


rule segmentation_precision_recall:
    input:
        val_predictions="results/segmentation/model/{model}/{dataset}/val_predictions.parquet",
    output:
        parquet="results/segmentation/model/{model}/{dataset}/precision_recall.parquet",
    run:
        df = pl.read_parquet(input.val_predictions)
        labels = df["label"].to_numpy()
        probabilities = expit(df["logit"].to_numpy())

        precision, recall, thresholds = precision_recall_curve(labels, probabilities)
        pr = pl.DataFrame(
            {
                "threshold": np.append(thresholds, np.nan),
                "precision": precision,
                "recall": recall,
            }
        )
        pr.write_parquet(output.parquet)
