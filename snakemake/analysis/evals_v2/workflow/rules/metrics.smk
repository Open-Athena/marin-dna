"""Compute AUPRC + cluster-bootstrap SE per (model, dataset).

One rule, fired per (model, dataset) — no cross-model aggregation, no markdown
rendering. The parquet is the deliverable.

The scores parquet from `compute_scores` stores per-strand LLR/JSD atoms
only (`llr_fwd`, `llr_rc`, `jsd_fwd`, `jsd_rc`). This rule materializes
the derived `_avg`, `minus_llr_*`, `abs_llr_*` variants in-place, then
evaluates AUPRC + cluster-bootstrap SE (cluster = `match_group`) for each.

Score-type fan-out:
  - `{protocol}_{fwd,rc,avg}` where `protocol` is the dataset's
    `score_protocol` (`minus_llr` for mendelian, `abs_llr` for complex).
  - `jsd_{fwd,rc,avg}` — JSD doesn't sign-flip so we keep it as-is.

`_avg` semantics: average raw LLR first, then apply the protocol
transform (so `abs_llr_avg = |(llr_fwd + llr_rc)/2|`, matching the
prior in-runner averaging behavior).

Output parquet (matched_pair datasets) has one row per (subset × score_type)
plus aggregate rows `_global_` and `_macro_avg_` per score_type — see
`marin_dna_evals.metrics.compute_auprc_metrics`.

For `eval_protocol: qtl_global` datasets (caqtl/dsqtl) the unmatched path
runs instead: `compute_qtl_metrics` emits one row per (metric × score_type)
where metric ∈ {AUPRC, pearson, spearman} — global AUPRC over all variants
plus Pearson/Spearman of the score vs `effect_size` over positives only. No
subset/match_group, so the parquet has a `metric` column and no subset rows.
"""


rule compute_metrics:
    input:
        "results/scores/{model}/{dataset}.parquet",
    output:
        "results/metrics/{model}/{dataset}.parquet",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(DATASETS),
    params:
        n_bootstrap=config["inference"]["n_bootstrap"],
        bootstrap_seed=config["inference"]["bootstrap_seed"],
        score_protocol=lambda wc: get_dataset_config(wc.dataset)["score_protocol"],
    run:
        protocol = params.score_protocol
        transform = SCORE_PROTOCOLS[protocol]
        df = pd.read_parquet(input[0])
        eval_protocol = get_dataset_protocol(wildcards.dataset)
        for col in get_dataset_variant_columns(wildcards.dataset):
            assert col in df.columns, f"scores parquet missing column {col!r}"
        if "llr_rc" in df.columns:
            df["llr_avg"] = (df["llr_fwd"] + df["llr_rc"]) / 2
            df["jsd_avg"] = (df["jsd_fwd"] + df["jsd_rc"]) / 2
        score_cols: list[str] = []
        for strand in ("fwd", "rc", "avg"):
            llr_col = f"llr_{strand}"
            jsd_col = f"jsd_{strand}"
            if llr_col in df.columns:
                df[f"{protocol}_{strand}"] = transform(df[llr_col])
                score_cols.append(f"{protocol}_{strand}")
            if jsd_col in df.columns:
                score_cols.append(jsd_col)
        assert score_cols, "no score columns to evaluate — scores parquet schema?"
        if eval_protocol == "qtl_global":
            # Global AUPRC + positives-only effect_size correlation. No
            # subset/match_group; metrics frame carries a `metric` column.
            metrics = compute_qtl_metrics(
                dataset=df[["label", "effect_size"]],
                scores=df[score_cols],
                score_columns=score_cols,
                n_bootstrap=params.n_bootstrap,
                rng=params.bootstrap_seed,
            )
        elif eval_protocol == "sge":
            # Per-accession (mavedb_urn) × consequence-subset AUPRC on the binary
            # `label` (impactful = calibrated abnormal), macro-averaged over
            # subsets and accessions. Frame carries `metric` / `subset` /
            # `accession` columns.
            metrics = compute_sge_metrics(
                dataset=df[list(SGE_VARIANT_COLUMNS)],
                scores=df[score_cols],
                score_columns=score_cols,
                n_bootstrap=params.n_bootstrap,
                rng=params.bootstrap_seed,
            )
        else:
            metrics = compute_auprc_metrics(
                dataset=df[list(REQUIRED_VARIANT_COLUMNS)],
                scores=df[score_cols],
                score_columns=score_cols,
                n_bootstrap=params.n_bootstrap,
                rng=params.bootstrap_seed,
            )
        metrics["model"] = wildcards.model
        metrics["dataset"] = wildcards.dataset
        metrics["split"] = config["split"]
        metrics.to_parquet(output[0], index=False)
        print(
            f"[evals_v2] {wildcards.model} {wildcards.dataset} "
            f"({eval_protocol}): {len(metrics)} metric rows (score_cols={score_cols})"
        )
