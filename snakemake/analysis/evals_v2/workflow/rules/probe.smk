"""Frozen-embedding linear-probe training (issue #320).

Per (model, dataset): train a per-subset, leave-one-chromosome-out (LOOC) L2-logistic
probe on the in-bundle pooled embeddings (`emb_ref`/`emb_alt`, #318) and emit two
artifacts — the **LOOC predictions** for every variant (`probe_score`, consumed by the
downstream metrics step) and the **fitted per-subset classifiers** (serialized for
reuse on other datasets). The protocol (feature rule, nested-C tuning, C-edge guard)
lives in `marin_dna.pipelines.evals.variant_probe.run_subset_probes`; this rule is thin
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
    threads: config["probe"]["n_jobs"]
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
    """Per-subset per-chromosome-weighted AUPRC + chromosome-cluster bootstrap SE and a
    `_macro_avg_` aggregate row (#331 / TraitGym; SE + macro per #347) for the probe vs
    its matched zero-shot LLR baseline — the metrics step of the #320/#331 protocol (wires
    #319; #341).

    The probe predictions parquet carries BOTH `probe_score` AND the raw `llr_fwd`/
    `llr_rc` atoms (`compute_probe` drops only `emb_ref`/`emb_alt`), so the probe and
    its zero-shot baseline are scored on IDENTICAL rows under the identical metric — a
    paired probe-vs-LLR comparison in one parquet. The baseline is the dataset's own
    `score_protocol` applied to the FWD/RC-averaged LLR (`minus_llr_avg` for mendelian),
    matching `compute_metrics`'s `_avg` semantics. The per-subset per-chrom-weighted
    metric is the matched_pair (subset + chrom) path, so qtl_global/sge are rejected.

    Thin glue over `marin_dna.pipelines.evals.metrics.per_chrom_ap_table`. CPU-only; off
    `rule all`. Reads a probe parquet, so the same `--rerun-triggers mtime` note as
    `compute_probe` applies (its upstream scores parquet was built with the #318
    overlay)."""
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
        assert eval_protocol == "matched_pair", (
            f"compute_probe_metrics is the per-subset per-chrom-weighted AUPRC "
            f"(matched_pair / TraitGym) path — needs subset + chrom; dataset "
            f"{wildcards.dataset!r} is eval_protocol {eval_protocol!r}"
        )
        transform = SCORE_PROTOCOLS[params.score_protocol]
        df = pd.read_parquet(input[0])
        for col in ("probe_score", "label", "subset", "chrom", "llr_fwd", "llr_rc"):
            assert col in df.columns, (
                f"{input[0]} missing column {col!r} — expected a compute_probe "
                f"predictions parquet"
            )
        # Zero-shot baseline on the identical rows: the dataset's score protocol applied
        # to the FWD/RC-averaged raw LLR (== compute_metrics `_avg`).
        baseline_col = f"{params.score_protocol}_avg"
        df[baseline_col] = transform((df["llr_fwd"] + df["llr_rc"]) / 2)

        metrics = per_chrom_ap_table(
            df,
            ["probe_score", baseline_col],
            n_bootstrap=params.n_bootstrap,
            rng=0,
            n_min=params.n_min,
        )
        metrics["model"] = wildcards.model
        metrics["dataset"] = wildcards.dataset
        metrics["split"] = config["split"]
        metrics.to_parquet(output[0], index=False)
        print(
            f"[evals_v2] probe_metrics {wildcards.model} {wildcards.dataset}: "
            f"{len(metrics)} rows (per-subset + macro_avg × 2 score types: "
            f"probe_score, {baseline_col}); SE from {params.n_bootstrap} "
            f"chromosome-cluster bootstraps"
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
