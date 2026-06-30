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

Kept OFF `rule all` (a few-models analysis, like `calibration` / `umap`). Pass
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
