"""AlphaGenome scoring + per-track aggregation.

Two rules so the per-track parquet is preserved on S3 — a future change to
the aggregation protocol (e.g. per-assay) won't have to re-spend the API
budget.
"""


rule compute_per_track_l2:
    output:
        "results/per_track_l2/{dataset}.parquet",
    wildcard_constraints:
        dataset="|".join(DATASETS),
    threads: config["num_workers"]
    params:
        # Pin the HF dataset commit. Bumping it triggers rerun via the
        # `params:` hash. `load_dataset(revision=…)` raises
        # `RevisionNotFoundError` on an unknown SHA — no silent fallback
        # to `main`.
        hf_path=lambda wc: f"{config['input_hf_prefix']}_{wc.dataset}",
        hf_revision=lambda wc: get_dataset_config(wc.dataset)["hf_revision"],
    run:
        ds = load_dataset(
            params.hf_path, split=config["split"], revision=params.hf_revision
        ).to_pandas()
        # qtl_global datasets (caqtl/dsqtl) carry effect_size, no
        # subset/match_group.
        variant_cols = get_dataset_variant_columns(wildcards.dataset)
        for col in variant_cols:
            assert col in ds.columns, f"dataset missing column {col!r}"

        # subset_n_pairs is a matched-pair smoke knob (slices on match_group);
        # QTL datasets have none, so it's ignored for them (null in production).
        n_pairs = config.get("subset_n_pairs")
        if n_pairs is not None and get_dataset_protocol(wildcards.dataset) == "matched_pair":
            keep = ds["match_group"].drop_duplicates().head(int(n_pairs))
            ds = ds[ds["match_group"].isin(keep)].reset_index(drop=True)
            print(
                f"[alphagenome_eval] {wildcards.dataset}: "
                f"subset_n_pairs={n_pairs} → {len(ds)} variants"
            )

        per_track = score_variants_alphagenome(
            ds[["chrom", "pos", "ref", "alt"]],
            num_workers=config["num_workers"],
        )

        out = pd.concat(
            [
                ds[list(variant_cols)].reset_index(drop=True),
                per_track.reset_index(drop=True),
            ],
            axis=1,
        )
        out.to_parquet(output[0], index=False)
        print(
            f"[alphagenome_eval] {wildcards.dataset} ({config['split']}): "
            f"n={len(out)} tracks={len(per_track.columns)}"
        )


rule aggregate_max:
    input:
        "results/per_track_l2/{dataset}.parquet",
    output:
        "results/scores/{dataset}.parquet",
    wildcard_constraints:
        dataset="|".join(DATASETS),
    run:
        score_col = config["score_column"]
        df = pd.read_parquet(input[0])
        # Exclude the variant columns (which for QTL datasets include
        # effect_size) so only the real per-track L2 columns are max-reduced.
        variant_cols = get_dataset_variant_columns(wildcards.dataset)
        track_cols = [c for c in df.columns if c not in variant_cols]
        assert track_cols, "no per-track columns found in input parquet"

        out = df[list(variant_cols)].copy()
        out[score_col] = df[track_cols].max(axis=1)
        assert (
            out[score_col].notna().all()
        ), f"NaN in {score_col} after max-across-tracks aggregation"
        out.to_parquet(output[0], index=False)
        print(
            f"[alphagenome_eval] {wildcards.dataset}: max-aggregated "
            f"{len(track_cols)} tracks → '{score_col}' "
            f"min={out[score_col].min():.3f} max={out[score_col].max():.3f}"
        )
