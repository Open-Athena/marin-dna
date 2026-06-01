rule score_variants:
    input:
        bw="results/conservation/{score}.bw",
    output:
        "results/{dataset}/{score}_{split}.parquet",
    wildcard_constraints:
        score="|".join(CONSERVATION_TRACKS),
        split="|".join(SPLITS),
        dataset="|".join(DATASETS),
    params:
        # Pin the HF dataset commit (mirrors evals_v2's pattern in
        # snakemake/analysis/evals_v2/workflow/rules/inference.smk). Bumping
        # the SHA in config triggers re-execution via snakemake's `params:`
        # hash. `load_dataset(revision=…)` raises RevisionNotFoundError on
        # an unknown SHA — no silent fallback to `main`.
        hf_revision=lambda wc: get_dataset_config(wc.dataset)["hf_revision"],
    run:
        hf_path = f"{INPUT_HF_PREFIX}_{wildcards.dataset}"
        ds = load_dataset(
            hf_path, split=wildcards.split, revision=params.hf_revision
        ).to_pandas()
        # qtl_global datasets (caqtl/dsqtl) require effect_size (no
        # subset/match_group); carry it through to the aggregate step.
        variant_cols = get_dataset_variant_columns(wildcards.dataset)
        for col in variant_cols:
            assert col in ds.columns, f"dataset missing column {col!r}"

        scores = score_variants_at_positions(ds, input.bw)
        assert len(scores) == len(ds)

        out = ds[list(variant_cols)].copy()
        out["score"] = scores
        n_nan = int(out["score"].isna().sum())
        print(
            f"[conservation_eval] {wildcards.dataset} {wildcards.score} "
            f"{wildcards.split}: n={len(out)} n_nan={n_nan} "
            f"({100 * n_nan / len(out):.2f}%) "
            f"score_min={out['score'].min():.3f} max={out['score'].max():.3f}"
        )
        out.to_parquet(output[0], index=False)


rule aggregate_metrics:
    input:
        parquets=lambda wc: expand(
            "results/{{dataset}}/{score}_{{split}}.parquet",
            score=SCORES,
        ),
    output:
        metrics="results/{dataset}/metrics_{split}.parquet",
        markdown="results/{dataset}/results_table_{split}.md",
    wildcard_constraints:
        split="|".join(SPLITS),
        dataset="|".join(DATASETS),
    params:
        n_bootstrap=config["inference"]["n_bootstrap"],
        bootstrap_seed=config["inference"]["bootstrap_seed"],
    run:
        # input.parquets are S3-fetched local temp paths, in expand() order (= SCORES).
        parquet_paths = dict(zip(SCORES, input.parquets))
        if get_dataset_protocol(wildcards.dataset) == "qtl_global":
            metrics, md = aggregate_conservation_qtl_metrics(
                parquet_paths,
                n_bootstrap=params.n_bootstrap,
                bootstrap_seed=params.bootstrap_seed,
            )
        else:
            metrics, md = aggregate_conservation_metrics(
                parquet_paths,
                n_bootstrap=params.n_bootstrap,
                bootstrap_seed=params.bootstrap_seed,
            )
        metrics["split"] = wildcards.split
        metrics["dataset"] = wildcards.dataset
        metrics.to_parquet(output.metrics, index=False)
        Path(output.markdown).write_text(md)
        print(f"[conservation_eval] wrote {output.markdown}")
        print(md)
