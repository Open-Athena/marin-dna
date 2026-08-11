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
        scorer=lambda wc: get_model_scorer(wc.model),
        rag_dataset_repo=lambda wc: get_model_config(wc.model).get(
            "rag_dataset_repo", ""
        ),
        rag_dataset_revision=lambda wc: get_model_config(wc.model).get(
            "rag_dataset_revision", ""
        ),
        hf_path=lambda wc: f"{config['input_hf_prefix']}_{wc.dataset}",
        # Pin the HF dataset commit. Bumping it triggers rerun via the
        # `params:` hash. `load_dataset(revision=…)` raises
        # `RevisionNotFoundError` on an unknown SHA — no silent fallback
        # to `main`.
        hf_revision=lambda wc: get_dataset_config(wc.dataset)["hf_revision"],
        rc=config["inference"]["rc"],
        # Output-affecting (#318): when true, the parquet gains the pooled
        # `emb_ref`/`emb_alt` columns, so it belongs in `params:`. Global toggle.
        return_embeddings=config["inference"]["return_embeddings"],
    run:
        # batch_size is per-model but execution-only (numerics are batch-
        # size-invariant modulo float-reduction noise), so we read it here
        # rather than declare it as a snakemake param. See note in `params:`.
        batch_size = get_model_batch_size(wildcards.model)
        # eval_accumulation_steps is execution-only (CPU-offload cadence for the
        # wide embedding predictions; does not change the stored values).
        eval_accumulation_steps = config["inference"].get("eval_accumulation_steps")
        ds = load_dataset(
            params.hf_path, split=config["split"], revision=params.hf_revision
        ).to_pandas()
        for col in get_dataset_variant_columns(wildcards.dataset):
            assert col in ds.columns, f"dataset missing column {col!r}"

        if params.scorer == "rag_glm":
            assert wildcards.dataset == "mendelian_traits"
            rag_rows = load_rag_eval_split(
                "mendelian_traits",
                config["split"],
                repo=params.rag_dataset_repo,
                revision=params.rag_dataset_revision,
            )
            # Exact-row contract: coordinates/alleles alone are insufficient;
            # labels, subsets, and match groups must also match the official
            # train dataset before a standard score artifact can be written.
            assert_rag_mendelian_variant_parity(rag_rows, pl.from_pandas(ds))
            out = score_rag_checkpoint_hf(
                input.checkpoint,
                rag_rows,
                benchmark="mendelian_traits",
                batch_size=batch_size,
                device="cuda",
                return_embeddings=params.return_embeddings,
            ).to_pandas()
        else:
            # qtl_global datasets (caqtl/dsqtl) carry no subset/match_group; they
            # require effect_size instead. `compute_variant_scores` only reads
            # chrom/pos/ref/alt, and the concat below preserves every ds column,
            # so effect_size reaches the scores parquet for the metric step.
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
                rc=params.rc,
                return_embeddings=params.return_embeddings,
                eval_accumulation_steps=eval_accumulation_steps,
            )
            assert len(scores) == len(ds)
            # Preserve all variant columns (chrom, pos, ref, alt, label, subset,
            # match_group, ...) alongside the score columns.
            out = pd.concat(
                [ds.reset_index(drop=True), scores.reset_index(drop=True)], axis=1
            )

        assert len(out) == len(ds)
        out.to_parquet(output[0], index=False)
        print(
            f"[evals_v2] {wildcards.model} {wildcards.dataset} "
            f"({config['split']}, scorer={params.scorer}): n={len(out)}"
        )
