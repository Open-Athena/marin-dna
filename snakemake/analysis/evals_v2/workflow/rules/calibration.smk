"""cLLR mutation-rate calibration tables (stage 3, #267/#270).

Score the pinned, subsampled neutral-site set (built by ``snakemake/neutral_sites``)
with the fast 2-forward LLR bundle (FWD+RC) and bin by ``pentanuc_mut`` into a
per-checkpoint ``llr_neutral_mean`` table that stage 4 (#271) subtracts to
calibrate variant LLRs. Entropy calibration is a separate, deferred path.

GPU-bound — runs on the same SkyPilot GPU node as ``compute_scores``. Kept OFF
``rule all`` (a few-models analysis, like ``nuc_dep`` / ``umap``); build by name:

    snakemake calibration                    # every configured calibration model
    snakemake results/calibration/<model>/llr_neutral_mean_n100.parquet
"""


rule compute_llr_neutral_mean:
    input:
        checkpoint="results/checkpoints/{model}",
        # The neutral set lives in another pipeline's S3 prefix. `storage()` uses
        # the boto3-backed s3 plugin (no s3fs async loop), so it stages to a LOCAL
        # path that `pd.read_parquet` reads *without* initializing s3fs in this
        # parent process — which would otherwise deadlock the forked DataLoader
        # workers that read the genome lazily in-worker (see issue #270 /
        # `compute_variant_scores`). The `{n}` wildcard ties the input cap to the
        # output name so several caps can coexist.
        neutral=lambda wc: storage(
            f"{config['calibration']['neutral_sites_s3_prefix']}/neutral_sites_n{wc.n}.parquet"
        ),
    output:
        "results/calibration/{model}/llr_neutral_mean_n{n}.parquet",
    wildcard_constraints:
        model="|".join(MODELS),
        n=r"\d+",
    params:
        # 255 for BOS checkpoints, 256/512 for older runs (same as compute_scores).
        window_size=lambda wc: get_model_config(wc.model)["window_size"],
        # FWD+RC to match the eval; falls back to the global inference default.
        rc=config["calibration"].get("rc", config["inference"]["rc"]),
        min_bin_count=config["calibration"]["min_bin_count"],
    threads: config["inference"]["num_workers"]
    run:
        # batch_size is execution-only (numerics are batch-size-invariant modulo
        # float-reduction noise) — read here, not declared in `params:`, so tuning
        # it doesn't force a re-run. Same convention as compute_scores.
        import shlex

        batch_size = get_model_batch_size(wildcards.model)

        # Drop neutral sites whose model-window contains a non-ACGT base (assembly-gap
        # N near telomeres/centromeres): the scoring kernel asserts ACGT on the
        # downstream window, so an N trips a CUDA device-side assert. FWD checks the
        # right flank and RC the left, so the *whole* window must be clean. This MUST
        # run in a CHILD process — it reads the genome (initializing s3fs), and doing
        # that here would deadlock the DataLoader workers compute_variant_scores forks
        # below. The child writes the filtered sites to a local parquet.
        filtered = output[0] + ".acgt_sites.parquet"
        shell(
            "python -m marin_dna.pipelines.evals.calibration "
            f"--sites {shlex.quote(str(input.neutral))} "
            f"--genome {shlex.quote(str(config['genome_path']))} "
            f"--window {int(params.window_size)} "
            f"--out {shlex.quote(filtered)}"
        )

        # LOCAL read — never pd.read_parquet("s3://…") in this (forking) process.
        sites = pd.read_parquet(filtered)
        table = compute_llr_neutral_mean(
            checkpoint_path=input.checkpoint,
            sites=sites,
            # S3 URI; pyfaidx reads by byte-range lazily inside each worker.
            genome_path=config["genome_path"],
            window_size=params.window_size,
            subsample_n=int(wildcards.n),
            min_bin_count=params.min_bin_count,
            batch_size=batch_size,
            num_workers=config["inference"]["num_workers"],
            rc=params.rc,
            data_transform_on_the_fly=config["inference"]["data_transform_on_the_fly"],
            torch_compile=config["inference"]["torch_compile"],
        )
        table.to_parquet(output[0], index=False)
        print(
            f"[evals_v2] calibration {wildcards.model} n={wildcards.n}: "
            f"{len(table)} cells -> {output[0]}"
        )


rule calibration:
    """Aggregate convenience target: the llr_neutral_mean table for every
    configured calibration model. Not part of `rule all`."""
    input:
        [
            f"results/calibration/{model}/llr_neutral_mean_n{config['calibration']['subsample_n']}.parquet"
            for model in CALIBRATION_MODELS
        ],
