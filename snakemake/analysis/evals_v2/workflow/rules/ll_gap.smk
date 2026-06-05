"""Functional/non-functional LL-gap eval (issue #274).

A GPU per-(model, region) scoring rule + a CPU summary gather, parallel to the
embedding-UMAP eval (rules/embedding_umap.smk) and kept OFF the default
``rule all`` (metrics-only) so it never perturbs score/metric reruns. Build:

    snakemake ll_gap                                          # summary over all models x regions
    snakemake results/ll_gap/scores/<model>/<region>.parquet  # one cell (one sky cluster)

``compute_hf_ll_gap`` reuses the existing ``download_model`` rule for the
checkpoint and pulls the mixed-case validation-interval sequences from
HuggingFace (uppercase = phyloP-functional, lowercase = non-functional). FWD
strand only — this matches the training-logged ``val_*_{functional,nonfunctional}``
loss the m5.1 gate is checked against.
"""


rule compute_ll_gap:
    input:
        checkpoint="results/checkpoints/{model}",
    output:
        "results/ll_gap/scores/{model}/{region}.parquet",
    wildcard_constraints:
        model="|".join(LL_GAP_MODELS),
        region="|".join(LL_GAP_DATASETS),
    threads: config["inference"]["num_workers"]
    params:
        # Output-affecting fields (snakemake `params` rerun trigger); batch_size
        # is execution-only and read inside `run:`.
        window_size=lambda wc: get_model_config(wc.model)["window_size"],
        hf_repo=lambda wc: get_ll_gap_dataset_config(wc.region)["hf_repo"],
        hf_revision=lambda wc: get_ll_gap_dataset_config(wc.region)["hf_revision"],
        split=LL_GAP_CFG.get("split", "validation"),
    run:
        from marin_dna.pipelines.evals.ll_gap import compute_hf_ll_gap

        batch_size = get_model_batch_size(wildcards.model)
        seqs = load_dataset(
            params.hf_repo, split=params.split, revision=params.hf_revision
        ).to_pandas()
        out = compute_hf_ll_gap(
            checkpoint_path=input.checkpoint,
            sequences=seqs,
            window_size=int(params.window_size),
            batch_size=batch_size,
            num_workers=config["inference"]["num_workers"],
            torch_compile=config["inference"].get("torch_compile", False),
        )
        out.to_parquet(output[0], index=False)
        print(
            f"[ll_gap] {wildcards.model}/{wildcards.region}: {len(out)} seqs "
            f"(upper={int(out['n_upper'].sum())}, lower={int(out['n_lower'].sum())})"
        )


rule ll_gap:
    """Aggregate every (model, region) score-atom parquet into one summary
(token-weighted LL_upper / LL_lower / gap per cell). Off `rule all` —
build with `snakemake ll_gap`."""
    input:
        [
            f"results/ll_gap/scores/{model}/{region}.parquet"
            for model in LL_GAP_MODELS
            for region in LL_GAP_DATASETS
        ],
    output:
        "results/ll_gap/summary.parquet",
    run:
        from marin_dna.pipelines.evals.ll_gap import aggregate_ll_gap

        rows = []
        for path in input:
            # results/ll_gap/scores/<model>/<region>.parquet
            model = Path(path).parent.name
            region = Path(path).stem
            atoms = pd.read_parquet(path)[
                ["ll_sum_upper", "ll_sum_lower", "n_upper", "n_lower"]
            ].to_numpy()
            rows.append(
                {"model": model, "region": region, **aggregate_ll_gap(atoms)}
            )
        summary = (
            pd.DataFrame(rows)
            .sort_values(["model", "region"])
            .reset_index(drop=True)
        )
        summary.to_parquet(output[0], index=False)
        print(f"[ll_gap] summary: {len(summary)} (model x region) rows")
