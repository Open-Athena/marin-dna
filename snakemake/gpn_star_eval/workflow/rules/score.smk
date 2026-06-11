"""Per-dataset scoring: load HF + GPN-Star prediction parquets, align, write a
combined long parquet with one row per (model, variant).

Output schema:
- variant columns, keyed by eval protocol (`get_dataset_variant_columns`):
  matched_pair → REQUIRED_VARIANT_COLUMNS (chrom, pos, ref, alt, label, subset,
  match_group); sge → SGE_VARIANT_COLUMNS (chrom, pos, ref, alt, mavedb_urn,
  gene, subset, label).
- model: ``"GPN-Star-V" | "GPN-Star-M" | "GPN-Star-P"``
- llr, abs_llr, llr_calibrated, abs_llr_calibrated (passthrough from
  predictions parquet)
- minus_llr, minus_llr_calibrated (derived: ``-llr`` / ``-llr_calibrated``
  for the leaderboard convention)

The align (`score_variants_gpn_star`) is protocol-independent — only the carried
variant columns differ. Total rows = ``len(MODELS) * n_variants_in_split``.
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
        variant_cols = get_dataset_variant_columns(wildcards.dataset)
        for col in variant_cols:
            assert col in hf.columns, f"HF dataset missing column {col!r}"

        per_model = []
        for model in MODELS:
            url = predictions_url(wildcards.dataset, model)
            preds = pd.read_parquet(url)
            scores = score_variants_gpn_star(hf, preds, split=config["split"])
            combined = pd.concat(
                [
                    hf[list(variant_cols)].reset_index(drop=True),
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
