"""Frozen-embedding linear-probe training (issue #320).

Per (model, dataset): train a per-subset, leave-one-chromosome-out (LOOC) L2-logistic
probe on the in-bundle pooled embeddings (`emb_ref`/`emb_alt`, #318) and emit two
artifacts — the **LOOC predictions** for every variant (`probe_score`, consumed by the
downstream metrics step) and the **fitted per-subset classifiers** (serialized for
reuse on other datasets). The protocol (feature rule, nested-C tuning, C-edge guard)
lives in `marin_dna_evals.variant_probe.run_subset_probes`; this rule is thin
glue.

CPU-only (sklearn on cached embeddings — no GPU). Requires the input scores parquet to
carry `emb_ref`/`emb_alt`, i.e. it must have been produced with
`inference.return_embeddings: true` (the #318 overlay); the rule fails fast otherwise.

Kept OFF `rule all` (a few-models analysis, like `umap` / `ll_gap`). Pass
`--rerun-triggers mtime` on every invocation: the input scores parquet was built with
the #318 overlay (`return_embeddings: true`), which differs from the committed default
(`false`), so snakemake's default `params` trigger would otherwise try to rebuild it —
and a rebuild on this CPU node (no GPU/gcloud) would drop the very `emb_ref`/`emb_alt`
the rule asserts on. Build by name:

    snakemake probe --rerun-triggers mtime                              # every configured probe cell
    snakemake results/probe/<model>/<dataset>.parquet --rerun-triggers mtime
"""


rule compute_probe:
    input:
        "results/scores/{model}/{dataset}.parquet",
    output:
        predictions="results/probe/{model}/{dataset}.parquet",
        classifiers="results/probe/{model}/{dataset}.joblib",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(DATASETS),
    threads: config["probe"]["n_jobs"]
    params:
        # Output-affecting → tracked by snakemake's `params` rerun trigger. The
        # feature combo is dataset-derived (directional vs swap-invariant); the rest
        # are the probe hyperparameters. `n_jobs` is execution-only (GridSearchCV
        # parallelism) so it's read via `threads`, not declared here.
        feature=lambda wc: get_probe_feature(wc.dataset),
        min_variants=config["probe"]["min_variants"],
        min_chroms=config["probe"]["min_chroms"],
        c_grid=config["probe"]["c_grid"],
        inner_splits=config["probe"]["inner_splits"],
    run:
        lo, hi, num = params.c_grid
        c_grid = np.logspace(lo, hi, num)
        df = pd.read_parquet(input[0])
        assert "emb_ref" in df.columns and "emb_alt" in df.columns, (
            f"{input[0]} lacks emb_ref/emb_alt — re-score {wildcards.model}/"
            f"{wildcards.dataset} with inference.return_embeddings=true (#318 overlay)"
        )
        predictions, classifiers = run_subset_probes(
            df,
            feature_combo=params.feature,
            c_grid=c_grid,
            min_variants=params.min_variants,
            min_chroms=params.min_chroms,
            inner_splits=params.inner_splits,
            n_jobs=threads,
        )
        predictions.to_parquet(output.predictions, index=False)
        joblib.dump(classifiers, output.classifiers)
        n_scored = int(predictions["probe_score"].notna().sum())
        print(
            f"[evals_v2] probe {wildcards.model} {wildcards.dataset} "
            f"({params.feature}): {len(classifiers)} subset probes, "
            f"n={len(predictions)} ({n_scored} scored)"
        )


rule probe:
    """Aggregate convenience target: probe artifacts for every configured probe cell
    (model × its listed datasets). Not part of `rule all`."""
    input:
        [
            f"results/probe/{pm['name']}/{dataset}.parquet"
            for pm in PROBE_MODELS
            for dataset in pm["datasets"]
        ],


rule compute_probe_metrics:
    """Probe-vs-zero-shot AUPRC for a probe cell — the metrics step of the #320/#331
    protocol (wires #319; #341; matched-pair SE + `_macro_avg_` row per #347), routed by
    eval_protocol:

    - **matched_pair** (mendelian / complex): per-subset per-chromosome-weighted AUPRC
      (TraitGym / #331), via `per_chrom_ap_table` — needs `subset` + `chrom`.
    - **sge**: per-accession (mavedb_urn) × consequence-subset AUPRC, macro-averaged,
      via `compute_sge_metrics` on the probe score — the same per-accession path the
      zero-shot SGE metrics use (#353), needs `mavedb_urn` + `gene` + `subset`. Rows
      the probe skipped (NaN `probe_score`, a subset that failed the
      min-variants/min-chroms gate) are dropped so the per-accession AUPRC and its
      paired baseline cover only probe-scored cells.

    `qtl_global` has no subset/accession structure and is rejected.

    In every case the probe predictions parquet carries BOTH `probe_score` AND the raw
    `llr_fwd`/`llr_rc` atoms (`compute_probe` drops only `emb_ref`/`emb_alt`), so the
    probe and its zero-shot baseline are scored on IDENTICAL rows under the identical
    metric — a paired comparison in one parquet. The baseline is the dataset's own
    `score_protocol` applied to the FWD/RC-averaged LLR (`minus_llr_avg` for mendelian /
    sge), matching `compute_metrics`'s `_avg` semantics.

    Thin glue over `marin_dna_evals.metrics.{per_chrom_ap_table,
    compute_sge_metrics}`. CPU-only; off `rule all`. Reads a probe parquet, so the same
    `--rerun-triggers mtime` note as `compute_probe` applies (its upstream scores parquet
    was built with the #318 overlay)."""
    input:
        "results/probe/{model}/{dataset}.parquet",
    output:
        "results/probe_metrics/{model}/{dataset}.parquet",
    wildcard_constraints:
        model="|".join(MODELS),
        dataset="|".join(DATASETS),
    params:
        score_protocol=lambda wc: get_dataset_config(wc.dataset)["score_protocol"],
        n_bootstrap=config["probe"]["n_bootstrap"],
        n_min=config["probe"]["n_min"],
    run:
        eval_protocol = get_dataset_protocol(wildcards.dataset)
        assert eval_protocol in ("matched_pair", "sge"), (
            f"compute_probe_metrics supports matched_pair (per-chrom AUPRC) + sge "
            f"(per-accession AUPRC); dataset {wildcards.dataset!r} is eval_protocol "
            f"{eval_protocol!r}"
        )
        df = pd.read_parquet(input[0])
        for col in ("probe_score", "label", "subset", "llr_fwd", "llr_rc"):
            assert col in df.columns, (
                f"{input[0]} missing column {col!r} — expected a compute_probe "
                f"predictions parquet"
            )
        if eval_protocol == "matched_pair":
            # Per-subset per-chromosome-weighted AUPRC + chromosome-cluster bootstrap SE +
            # `_macro_avg_` row (TraitGym / #331 / #347). Zero-shot baseline on the identical
            # rows = the dataset's score protocol applied to the FWD/RC-averaged raw LLR.
            assert (
                "chrom" in df.columns
            ), f"{input[0]} missing 'chrom' — matched_pair per-chrom AUPRC needs it"
            transform = SCORE_PROTOCOLS[params.score_protocol]
            baseline_col = f"{params.score_protocol}_avg"
            df[baseline_col] = transform((df["llr_fwd"] + df["llr_rc"]) / 2)
            metrics = per_chrom_ap_table(
                df,
                ["probe_score", baseline_col],
                n_bootstrap=params.n_bootstrap,
                rng=0,
                n_min=params.n_min,
            )
        else:  # sge — per-accession × subset AUPRC (probe vs its paired zero-shot).
            for col in ("mavedb_urn", "gene"):
                assert col in df.columns, (
                    f"{input[0]} missing {col!r} — sge per-accession AUPRC needs the "
                    f"accession key + gene"
                )
            metrics = compute_sge_probe_metrics(
                df,
                params.score_protocol,
                n_bootstrap=params.n_bootstrap,
                rng=0,
            )
        metrics["model"] = wildcards.model
        metrics["dataset"] = wildcards.dataset
        metrics["split"] = config["split"]
        metrics.to_parquet(output[0], index=False)
        print(
            f"[evals_v2] probe_metrics {wildcards.model} {wildcards.dataset} "
            f"({eval_protocol}): {len(metrics)} rows"
        )


rule probe_metrics:
    """Aggregate convenience target: probe_metrics for every configured probe cell.
    Not part of `rule all`."""
    input:
        [
            f"results/probe_metrics/{pm['name']}/{dataset}.parquet"
            for pm in PROBE_MODELS
            for dataset in pm["datasets"]
        ],
