"""Compute the variant-score bundle (LLR + embedding distances + next-token JSD)
per (model, dataset), with FWD+RC averaging.

GPU-bound: meant to run on a SkyPilot GPU node, not the local CPU box.
"""


rule compute_scores:
    input:
        checkpoint="results/checkpoints/{model}",
    output:
        "results/scores/{model}/{dataset}.parquet",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(DATASETS),
    threads: config["inference"]["num_workers"]
    params:
        # 255 for BOS-using checkpoints (e.g. exp136), 256 for older runs;
        # the tokenizer baked into each checkpoint handles BOS itself.
        # NOTE: only output-affecting fields belong in `params:` — values
        # here are tracked by snakemake's `params` rerun trigger. Execution-
        # only knobs (e.g. batch_size) are read inside `run:` instead so
        # tuning them doesn't force a re-run of finished work.
        window_size=lambda wc: get_model_config(wc.model)["window_size"],
        hf_path=lambda wc: f"{config['input_hf_prefix']}_{wc.dataset}",
        # Pin the HF dataset commit. Bumping it triggers rerun via the
        # `params:` hash. `load_dataset(revision=…)` raises
        # `RevisionNotFoundError` on an unknown SHA — no silent fallback
        # to `main`.
        hf_revision=lambda wc: get_dataset_config(wc.dataset)["hf_revision"],
        rc=config["inference"]["rc"],
    run:
        # batch_size is per-model but execution-only (numerics are batch-
        # size-invariant modulo float-reduction noise), so we read it here
        # rather than declare it as a snakemake param. See note in `params:`.
        batch_size = get_model_batch_size(wildcards.model)
        # eval_accumulation_steps is execution-only (CPU-offload cadence for the
        # wide embedding predictions; doesn't change the stored values).
        eval_accumulation_steps = get_model_eval_accumulation_steps(wildcards.model)
        ds = load_dataset(
            params.hf_path, split=config["split"], revision=params.hf_revision
        ).to_pandas()
        # qtl_global datasets (caqtl/dsqtl) carry no subset/match_group; they
        # require effect_size instead. `compute_variant_scores` only reads
        # chrom/pos/ref/alt, and the concat below preserves every ds column,
        # so effect_size reaches the scores parquet for the metric step.
        for col in get_dataset_variant_columns(wildcards.dataset):
            assert col in ds.columns, f"dataset missing column {col!r}"
        scores = compute_variant_scores(
            checkpoint_path=input.checkpoint,
            dataset=ds,
            # S3 URI; pyfaidx + fsspec/s3fs reads sequence by byte-range,
            # no full download. Requires `--group genome-s3`.
            genome_path=config["genome_path"],
            context_size=params.window_size,
            batch_size=batch_size,
            num_workers=config["inference"]["num_workers"],
            data_transform_on_the_fly=config["inference"][
                "data_transform_on_the_fly"
            ],
            torch_compile=config["inference"]["torch_compile"],
            bf16=config["inference"]["bf16"],
            rc=params.rc,
            return_embeddings=config["inference"]["return_embeddings"],
            eval_accumulation_steps=eval_accumulation_steps,
        )
        assert len(scores) == len(ds)
        # Preserve all variant columns (chrom, pos, ref, alt, label, subset,
        # match_group, …) alongside the score columns.
        out = pd.concat(
            [ds.reset_index(drop=True), scores.reset_index(drop=True)], axis=1
        )
        out.to_parquet(output[0], index=False)
        print(
            f"[evals_v2] {wildcards.model} {wildcards.dataset} "
            f"({config['split']}): n={len(out)}"
        )
