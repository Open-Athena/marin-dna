"""cLLR mutation-rate calibration tables (stage 3, #267/#270).

Score the pinned, **pre-filtered + subsampled** neutral-site set (built once by
``snakemake/neutral_sites``: ACGT-window-filtered, then ``n`` sites per 5-mer) with
the fast 2-forward LLR bundle (FWD+RC) and bin by ``pentanuc_mut`` into a
per-checkpoint ``llr_neutral_mean`` table that stage 4 (#271) subtracts to calibrate
variant LLRs. Entropy calibration is a separate, deferred path.

The N-window filtering is model-independent and lives upstream (one set per window,
reused by every model), so this GPU rule never touches the genome for filtering — it
just reads the clean set and scores. Kept OFF ``rule all`` (a few-models analysis,
like ``nuc_dep`` / ``umap``); build by name:

    snakemake calibration                    # every configured calibration model
    snakemake results/calibration/<model>/llr_neutral_mean_n100.parquet
"""


rule compute_llr_neutral_mean:
    input:
        checkpoint="results/checkpoints/{model}",
        # Pre-filtered (ACGT-window, at `scoreable_window`) + subsampled neutral set
        # from the neutral_sites pipeline — built once per window and reused by every
        # model, so this rule does no genome filtering. `storage()` (boto3, no s3fs
        # async loop) stages it to a LOCAL path, so `pd.read_parquet` never inits s3fs
        # in this process — which would deadlock the DataLoader workers that read the
        # genome lazily in-worker (issue #270 / `compute_variant_scores`).
        neutral=lambda wc: storage(
            f"{config['calibration']['neutral_sites_s3_prefix']}/"
            f"neutral_sites_n{wc.n}_w{config['calibration']['scoreable_window']}.parquet"
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
        scoreable_window=config["calibration"]["scoreable_window"],
    threads: config["inference"]["num_workers"]
    run:
        # batch_size is execution-only (numerics are batch-size-invariant modulo
        # float-reduction noise) — read here, not declared in `params:`, so tuning
        # it doesn't force a re-run. Same convention as compute_scores.
        batch_size = get_model_batch_size(wildcards.model)

        # The pinned neutral set is ACGT-clean for windows up to `scoreable_window`;
        # a model with a wider window could see an N the kernel rejects. Fail fast.
        assert params.scoreable_window >= params.window_size, (
            f"scoreable_window {params.scoreable_window} < model window_size "
            f"{params.window_size}: neutral set not guaranteed scoreable — "
            f"build a wider-window neutral_sites_n{wildcards.n}_w… set"
        )

        # LOCAL read (storage staged it); the set is already filtered + subsampled.
        sites = pd.read_parquet(input.neutral)
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
