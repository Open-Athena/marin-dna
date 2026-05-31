"""AUPRC + cluster-bootstrap SE per (dataset).

Cluster = `match_group`; resamples groups (not rows) so the SE accounts
for the 1:k matched structure. Output schema mirrors
`marin_dna.pipelines.evals.metrics.compute_auprc_metrics`:
`[score_type, subset, value, se, n_groups, n_rows]` plus `_global_` and
`_macro_avg_` aggregate rows per score_type.

For `eval_protocol: qtl_global` datasets (caqtl/dsqtl) the unmatched path
runs instead via `compute_qtl_metrics`: one row per (metric × score_type),
metric ∈ {AUPRC, pearson, spearman} — global AUPRC + Pearson/Spearman of
`alphagenome_max_l2` vs `effect_size` over positives only (a `metric`
column, no subset rows).
"""


rule compute_metrics:
    input:
        "results/scores/{dataset}.parquet",
    output:
        "results/metrics/{dataset}.parquet",
    wildcard_constraints:
        dataset="|".join(DATASETS),
    params:
        n_bootstrap=config["n_bootstrap"],
        bootstrap_seed=config["bootstrap_seed"],
    run:
        score_col = config["score_column"]
        df = pd.read_parquet(input[0])
        eval_protocol = get_dataset_protocol(wildcards.dataset)
        for col in get_dataset_variant_columns(wildcards.dataset):
            assert col in df.columns, f"scores parquet missing column {col!r}"
        assert (
            score_col in df.columns
        ), f"scores parquet missing score column {score_col!r}"

        if eval_protocol == "qtl_global":
            metrics = compute_qtl_metrics(
                dataset=df[["label", "effect_size"]],
                scores=df[[score_col]],
                score_columns=[score_col],
                n_bootstrap=params.n_bootstrap,
                rng=params.bootstrap_seed,
            )
        else:
            metrics = compute_auprc_metrics(
                dataset=df[list(REQUIRED_VARIANT_COLUMNS)],
                scores=df[[score_col]],
                score_columns=[score_col],
                n_bootstrap=params.n_bootstrap,
                rng=params.bootstrap_seed,
            )
        metrics["dataset"] = wildcards.dataset
        metrics["split"] = config["split"]
        metrics.to_parquet(output[0], index=False)
        print(
            f"[alphagenome_eval] {wildcards.dataset} ({eval_protocol}): "
            f"{len(metrics)} metric rows"
        )
