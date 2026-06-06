"""Per-dataset AUPRC + cluster-bootstrap SE on the 4 score columns × 3 models.

Output schema (one row per ``(model, score_type, subset)``):
``[score_type, subset, value, se, n_groups, n_rows, model, dataset]``.
Includes ``_global_`` and ``_macro_avg_`` sentinel subset rows from
``compute_auprc_metrics`` (``n_min`` from config). AUPRC + cluster bootstrap on
``match_group`` matches evals_v2, so these rows are directly comparable to our
models on the same dataset revisions. The dashboard reads this parquet directly
(``leaderboard._parquet_path`` case ``gpn_star``), filtering by ``model`` +
``score_type`` — there is no ``split`` column (all rows are the train split, the
only split this pipeline emits). (The previous PairwiseAccuracy path asserted
1 pos + 1 neg per ``match_group`` and would fail on the 1:9 k=9 datasets.)
"""


rule compute_metrics:
    input:
        "results/scores/{dataset}.parquet",
    output:
        "results/metrics/{dataset}.parquet",
    wildcard_constraints:
        dataset="|".join(DATASETS),
    run:
        df = pd.read_parquet(input[0])
        for col in REQUIRED_VARIANT_COLUMNS:
            assert col in df.columns, f"input scores parquet missing column {col!r}"
        for col in SCORE_COLUMNS:
            assert col in df.columns, f"input scores parquet missing column {col!r}"

        expected_models = {f"GPN-Star-{m}" for m in MODELS}
        assert set(df["model"].unique()) == expected_models, (
            f"unexpected model set in scores parquet: "
            f"{set(df['model'].unique())} vs {expected_models}"
        )

        per_model = []
        for m in MODELS:
            model = f"GPN-Star-{m}"
            sub = df[df["model"] == model]
            metrics = compute_auprc_metrics(
                dataset=sub[["label", "subset", "match_group"]],
                scores=sub[SCORE_COLUMNS],
                score_columns=SCORE_COLUMNS,
                n_bootstrap=config["n_bootstrap"],
                rng=config["bootstrap_seed"],
                n_min=config["n_min"],
            )
            metrics["model"] = model
            per_model.append(metrics)

        out = pd.concat(per_model, ignore_index=True)
        out["dataset"] = wildcards.dataset
        out.to_parquet(output[0], index=False)
        print(
            f"[gpn_star_eval] {wildcards.dataset}: {len(out)} metric rows "
            f"({len(per_model)} models × {len(SCORE_COLUMNS)} score columns × "
            f"per-subset rows incl. _global_ / _macro_avg_)"
        )
