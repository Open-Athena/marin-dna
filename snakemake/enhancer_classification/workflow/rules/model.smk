WEIGHTS_FILE = f"results/weights/{config['alphagenome_weights']['filename']}"


rule download_alphagenome_weights:
    output:
        "results/weights/{filename}",
    params:
        repo=config["alphagenome_weights"]["repo"],
    run:
        from pathlib import Path

        out = Path(output[0])
        hf_hub_download(params.repo, wildcards.filename, local_dir=str(out.parent))


rule train_model:
    threads: workflow.cores
    input:
        train="results/dataset/{dataset}/train.parquet",
        val="results/dataset/{dataset}/validation.parquet",
        weights=lambda wc: [] if get_model_config(wc.model).get("random_init") else WEIGHTS_FILE,
    output:
        ckpt="results/model/{model}/{dataset}/best.ckpt",
        metrics="results/model/{model}/{dataset}/metrics.json",
        val_predictions="results/model/{model}/{dataset}/val_predictions.parquet",
    params:
        freeze_flag=lambda wc: (
            "--freeze-backbone"
            if get_model_config(wc.model)["freeze_backbone"]
            else "--no-freeze-backbone"
        ),
        weights_flag=lambda wc, input: (
            f"--weights-path {input.weights}" if input.weights else ""
        ),
        learning_rate=lambda wc: get_model_config(wc.model)["learning_rate"],
        weight_decay=lambda wc: get_model_config(wc.model)["weight_decay"],
        gradient_clip_val=lambda wc: get_model_config(wc.model)["gradient_clip_val"],
        batch_size=lambda wc: get_model_config(wc.model)["batch_size"],
        warmup_fraction=lambda wc: get_model_config(wc.model)["warmup_fraction"],
        mlp_hidden_dim=lambda wc: get_model_config(wc.model).get("mlp_hidden_dim", 0),
    shell:
        """
        uv run python -m marin_dna.pipelines.enhancer_classification.train \
            --train-parquet {input.train} \
            --val-parquet {input.val} \
            {params.weights_flag} \
            --output-ckpt {output.ckpt} \
            --output-metrics {output.metrics} \
            --output-val-predictions {output.val_predictions} \
            --learning-rate {params.learning_rate} \
            --weight-decay {params.weight_decay} \
            --batch-size {params.batch_size} \
            --gradient-clip-val {params.gradient_clip_val} \
            --warmup-fraction {params.warmup_fraction} \
            --mlp-hidden-dim {params.mlp_hidden_dim} \
            {params.freeze_flag} \
            --seed {config[seed]} \
            --num-workers {threads} \
            --wandb-run {wildcards.model}-{wildcards.dataset}
        """


TOP_N_MISCLASSIFIED = 10


rule misclassified_regions:
    input:
        val_predictions="results/model/{model}/{dataset}/val_predictions.parquet",
    output:
        parquet="results/model/{model}/{dataset}/misclassified.parquet",
    run:
        df = pl.read_parquet(input.val_predictions)

        false_positives = (
            df.filter(pl.col("label") == 0)
            .sort("logit", descending=True)
            .head(TOP_N_MISCLASSIFIED)
            .with_columns(error_type=pl.lit("false_positive"))
        )
        false_negatives = (
            df.filter(pl.col("label") == 1)
            .sort("logit")
            .head(TOP_N_MISCLASSIFIED)
            .with_columns(error_type=pl.lit("false_negative"))
        )

        result = pl.concat([false_positives, false_negatives])
        result = result.with_columns(
            probability=pl.Series(expit(result["logit"].to_numpy()))
        )
        result = result.select(
            "error_type", "genome", "chrom", "start", "end", "strand",
            "label", "logit", "probability",
        )
        result.write_parquet(output.parquet)


# Caveat: precision-recall is computed on a balanced 1:1 validation set.
# In the real genome, negatives vastly outnumber enhancers, so precision
# at a given recall will be lower than reported here.
rule precision_recall:
    input:
        val_predictions="results/model/{model}/{dataset}/val_predictions.parquet",
    output:
        parquet="results/model/{model}/{dataset}/precision_recall.parquet",
    run:
        df = pl.read_parquet(input.val_predictions)
        labels = df["label"].to_numpy()
        probabilities = expit(df["logit"].to_numpy())

        precision, recall, thresholds = precision_recall_curve(labels, probabilities)
        # precision_recall_curve returns n+1 precision/recall values; last has recall=0
        pr = pl.DataFrame({
            "threshold": np.append(thresholds, np.nan),
            "precision": precision,
            "recall": recall,
        })
        pr.write_parquet(output.parquet)
