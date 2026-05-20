# gpn_star_eval — GPN-Star V/M/P baseline on matched-pair eval datasets

PairwiseAccuracy ± binomial SE for the three GPN-Star variants
(V = vertebrate-100way, M = mammal-447way, P = primate-243way; all 200M params)
on the matched-pair eval datasets ([#161 Mendelian](https://github.com/Open-Athena/marin-dna/issues/161),
[#162 Complex traits](https://github.com/Open-Athena/marin-dna/issues/162),
[#172 eQTL](https://github.com/Open-Athena/marin-dna/issues/172)).

GPN-Star scoring runs in [songlab-cal/TraitGym](https://github.com/songlab-cal/TraitGym),
not this repo. Predictions are pulled from a gist (commit pinned in
`marin_dna.evals.gpn_star.GPN_STAR_GIST_BASE`); see
[#145 comment](https://github.com/Open-Athena/marin-dna/issues/145#issuecomment-4444680280)
for the upstream definition. This pipeline is the align + aggregate step:
load HF eval dataset, row-align with the predictions parquet, compute PA per
subset.

## What it does

For each `dataset` in `config["datasets"]`:

1. **Score** — for each of the 3 model variants, load the GPN-Star prediction
   parquet (`predictions_url`), align row-by-row with the HF dataset's `train`
   split, derive `minus_llr` and `minus_llr_calibrated` for the leaderboard
   convention. Output: one long parquet per dataset, `model` column
   distinguishing V / M / P.

   Row alignment is asserted element-wise, no key-based merge — TraitGym's
   `bolinas_pack_predictions` rule builds the predictions parquet by
   horizontal-concat with HF row order, so a `split == "train"` filter is
   sufficient.

2. **Compute** PairwiseAccuracy ± SE per consequence subset on all 4 score
   columns (`minus_llr`, `abs_llr`, `minus_llr_calibrated`,
   `abs_llr_calibrated`). Output: metrics parquet keyed by
   `(model, score_type, subset)`, plus the `_global_` / `_macro_avg_`
   sentinel rows from `compute_pairwise_metrics` (n_min=30).

Downstream consumers (e.g. `snakemake/evals/scratch/leaderboard_gen.py`)
filter by `score_type` to pick the leaderboard-convention column:

- **Mendelian:** `minus_llr_calibrated` (pathogenic ⇒ alt depleted under
  purifying selection ⇒ negative LLR ⇒ positive `minus_llr`).
- **Complex traits / eQTL:** `abs_llr_calibrated` (direction-agnostic
  magnitude).

## Outputs

S3 bucket `s3://oa-bolinas/snakemake/gpn_star_eval/`:

```
results/
├── scores/{dataset}.parquet    # 3 × N rows: variant cols + scores per model
└── metrics/{dataset}.parquet   # PA per (model, score_type, subset)
```

## Conventions

- **Train split only.** Test held out for the final-eval pass (matches the
  other matched-pair pipelines in this repo).
- **Calibration happens upstream.** `*_calibrated` columns are the producer's
  pentanucleotide-context background-subtracted variant; see
  [#145 comment](https://github.com/Open-Athena/marin-dna/issues/145#issuecomment-4444680280)
  for the formula.
- **Reverse-complement averaging happens upstream.** GPN-Star averages
  forward + RC strand predictions. The `exp*` gLM rows in this repo's
  leaderboards are forward-only — RC averaging is a planned addition.

## Usage

Pure CPU, ~20 s wallclock for all 3 datasets. Network: ~12 MB (downloads
9 prediction parquets from the pinned gist).

```bash
cd snakemake/gpn_star_eval

# Dry-run to inspect the DAG.
uv run snakemake -n

# Real run — writes outputs to S3 per the default profile.
uv run snakemake
```

No SkyPilot launch yaml — the workload is too small to be worth it.

## Library

Pipeline rules are thin glue around `marin_dna.evals.gpn_star`:

- `score_variants_gpn_star(hf_df, predictions, split)` — load + row-align
  predictions, derive `minus_*` columns. Asserts row-count + key-order match.
- `predictions_url(dataset, model)` — gist raw URL for one prediction parquet.
- `GPN_STAR_MODELS`, `GPN_STAR_MODEL_INFO`, `GPN_STAR_SCORE_COLUMN` — metadata.

Tests at [`tests/evals/test_gpn_star.py`](../../tests/evals/test_gpn_star.py)
cover alignment, NaN detection, chrom-dtype handling, and the predictions-URL
helper.
