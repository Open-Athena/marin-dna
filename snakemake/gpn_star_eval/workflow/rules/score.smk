"""Per-dataset scoring: load HF + GPN-Star prediction parquets, align, write a
combined long parquet with one row per (model, variant).

Output schema:
- REQUIRED_VARIANT_COLUMNS (chrom, pos, ref, alt, label, subset, match_group)
- model: ``"GPN-Star-V" | "GPN-Star-M" | "GPN-Star-P"``
- llr, abs_llr, llr_calibrated, abs_llr_calibrated (passthrough from
  predictions parquet)
- minus_llr, minus_llr_calibrated (derived: ``-llr`` / ``-llr_calibrated``
  for the leaderboard convention)

Total rows = ``len(MODELS) * n_variants_in_split``.
"""


rule score_variants:
    output:
        "results/scores/{dataset}.parquet",
    wildcard_constraints:
        dataset="|".join(DATASETS),
    run:
        hf_path = f"{config['input_hf_prefix']}_{wildcards.dataset}"
        hf = load_dataset(
            hf_path,
            split=config["split"],
            revision=HF_REVISION[wildcards.dataset],
        ).to_pandas()
        for col in REQUIRED_VARIANT_COLUMNS:
            assert col in hf.columns, f"HF dataset missing column {col!r}"

        per_model = []
        for model in MODELS:
            url = predictions_url(wildcards.dataset, model)
            preds = pd.read_parquet(url)
            scores = score_variants_gpn_star(hf, preds, split=config["split"])
            combined = pd.concat(
                [
                    hf[list(REQUIRED_VARIANT_COLUMNS)].reset_index(drop=True),
                    scores.reset_index(drop=True),
                ],
                axis=1,
            )
            combined["model"] = f"GPN-Star-{model}"
            per_model.append(combined)

        out = pd.concat(per_model, ignore_index=True)
        out.to_parquet(output[0], index=False)
        print(
            f"[gpn_star_eval] {wildcards.dataset}: {len(out)} rows "
            f"({len(MODELS)} models × {len(hf)} variants)"
        )
