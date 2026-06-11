"""Per-dataset AUPRC + cluster-bootstrap SE on the 4 score columns × 3 models.

The metric path is keyed by `eval_protocol` (`get_dataset_protocol`), matching
the score columns the dashboard surfaces per dataset:

- **matched_pair** (mendelian_traits / complex_traits) → `compute_auprc_metrics`:
  per-consequence-subset AUPRC + cluster bootstrap on `match_group`, plus
  `_global_` / `_macro_avg_` sentinel rows (`n_min` from config). One row per
  `(model, score_type, subset)`: `[score_type, subset, value, se, n_groups,
  n_rows, model, dataset]`.
- **sge** (#301) → `compute_sge_metrics`: per-accession (MaveDB study) ×
  consequence-subset AUPRC on the binary `label`, macro-averaged over subsets
  then accessions (`_macro_avg_` sentinels; gated cells emit NaN value with real
  counts). One row per `(model, score_type, metric, subset, accession)`:
  `[metric, subset, accession, gene, score_type, value, se, n, n_pos, model,
  dataset]`. Same shared `compute_sge_metrics` as evals_v2 / conservation, so the
  rows are directly comparable to our models on the same evals_sge revision.

Both are AUPRC + bootstrap (seed from config) on exactly the variants our models
are scored on. There is no ``split`` column (all rows are the train split, the
only split this pipeline emits). The dashboard reads this parquet directly
(``leaderboard._parquet_path`` / ``sge_normalized_rows`` case ``gpn_star``),
filtering by ``model`` (+ ``score_type`` for matched_pair).
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
        protocol = get_dataset_protocol(wildcards.dataset)
        variant_cols = get_dataset_variant_columns(wildcards.dataset)
        for col in (*variant_cols, *SCORE_COLUMNS):
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
            if protocol == "sge":
                # Per-accession × consequence-subset AUPRC on the binary `label`.
                # Leave `n_min_auprc` at the compute_sge_metrics default (the SGE
                # per-label-class cell floor) so this matches evals_v2 +
                # conservation_eval exactly — they don't pass it either. (config
                # `n_min` is the matched_pair macro-entry gate, a different
                # semantic, and is used only in the else branch below.)
                metrics = compute_sge_metrics(
                    dataset=sub[["mavedb_urn", "gene", "subset", "label"]],
                    scores=sub[SCORE_COLUMNS],
                    score_columns=SCORE_COLUMNS,
                    n_bootstrap=config["n_bootstrap"],
                    rng=config["bootstrap_seed"],
                )
            else:
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
            f"[gpn_star_eval] {wildcards.dataset} ({protocol}): {len(out)} metric "
            f"rows ({len(per_model)} models × {len(SCORE_COLUMNS)} score columns)"
        )
