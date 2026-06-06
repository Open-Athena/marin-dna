# gpn_star_eval — GPN-Star V/M/P baseline on matched-pair eval datasets

AUPRC ± cluster-bootstrap SE for the three GPN-Star variants
(V = vertebrate-100way, M = mammal-447way, P = primate-243way; all 200M params)
on the matched-pair eval datasets ([#161 Mendelian](https://github.com/Open-Athena/marin-dna/issues/161),
[#162 Complex traits](https://github.com/Open-Athena/marin-dna/issues/162)).
Same metric, dataset revisions, and bootstrap config as
[`evals_v2`](../analysis/evals_v2), so these rows are directly comparable to our
own models — and they **reproduce the GPN-Star numbers the dashboard reads** from
the #145 metrics gist.

GPN-Star scoring runs in [songlab-cal/TraitGym](https://github.com/songlab-cal/TraitGym),
not this repo. Predictions are pulled from a gist (commit pinned in
`marin_dna.pipelines.evals.gpn_star.GPN_STAR_GIST_BASE`); see the
[#145 calibration definition](https://github.com/Open-Athena/marin-dna/issues/145#issuecomment-4444680280)
and the [current-revision upload](https://github.com/Open-Athena/marin-dna/issues/145#issuecomment-4489509362).
This pipeline is the align + aggregate step: load HF eval dataset, row-align with
the predictions parquet, compute AUPRC per subset.

eQTL (#172) was retired in the 1:9 / k=9 migration (#194) and has no
current-revision GPN-Star upload, so it is not evaluated here.

## What it does

For each `dataset` in `config["datasets"]` (a `{name, hf_revision}` list pinned
to the k=9 HF commits):

1. **Score** — for each of the 3 model variants, load the GPN-Star prediction
   parquet (`predictions_url`), align row-by-row with the HF dataset's `train`
   split **at the pinned revision**, derive `minus_llr` and
   `minus_llr_calibrated` for the leaderboard convention. Output: one long
   parquet per dataset, `model` column distinguishing V / M / P.

   Row alignment is asserted element-wise, no key-based merge — TraitGym's
   `bolinas_pack_predictions` rule builds the predictions parquet by
   horizontal-concat with HF row order, so a `split == "train"` filter on the
   matching revision is sufficient (and `score_variants_gpn_star` fails loud if
   it ever isn't).

2. **Compute** AUPRC + cluster-bootstrap SE (cluster = `match_group`) per
   consequence subset on all 4 score columns (`minus_llr`, `abs_llr`,
   `minus_llr_calibrated`, `abs_llr_calibrated`). Output: metrics parquet keyed
   by `(model, score_type, subset)`, plus the `_global_` / `_macro_avg_`
   sentinel rows from `compute_auprc_metrics` (`n_min` from config). Baseline
   AUPRC ≈ 0.10 under 1:9.

Downstream consumers filter by `score_type` to pick the leaderboard-convention
column:

- **Mendelian:** `minus_llr_calibrated` (pathogenic ⇒ alt depleted under
  purifying selection ⇒ negative LLR ⇒ positive `minus_llr`).
- **Complex traits:** `abs_llr_calibrated` (direction-agnostic magnitude).

## Outputs

S3 bucket `s3://oa-bolinas/snakemake/gpn_star_eval/`:

```
results/
├── scores/{dataset}.parquet    # 3 × N rows: variant cols + scores per model
└── metrics/{dataset}.parquet   # AUPRC per (model, score_type, subset);
                                 # cols [score_type, subset, value, se,
                                 #       n_groups, n_rows, model, dataset, split]
```

## Conventions

- **Train split only.** Test held out for the final-eval pass (matches the
  other matched-pair pipelines in this repo).
- **Pinned revisions.** Each dataset's `hf_revision` mirrors `evals_v2` so the
  prediction parquets row-align deterministically and the AUPRC is computed on
  exactly the variants our models are scored on.
- **Calibration happens upstream.** `*_calibrated` columns are the producer's
  pentanucleotide-context background-subtracted variant; see the
  [#145 comment](https://github.com/Open-Athena/marin-dna/issues/145#issuecomment-4444680280)
  for the formula.
- **Reverse-complement averaging happens upstream.** GPN-Star averages
  forward + RC strand predictions.

## Usage

Pure CPU, no GPU. Network: a few MB (downloads 6 prediction parquets from the
pinned gist). The bootstrap dominates wallclock (~3 min CPU per dataset at
`n_bootstrap: 1000`).

```bash
cd snakemake/gpn_star_eval

# Dry-run to inspect the DAG.
uv run snakemake -n

# Real run — writes outputs to S3 per the default profile.
uv run snakemake
```

No SkyPilot launch yaml — the workload is too small to be worth it.

## Library

Pipeline rules are thin glue around `marin_dna.pipelines.evals.gpn_star`:

- `score_variants_gpn_star(hf_df, predictions, split)` — load + row-align
  predictions, derive `minus_*` columns. Asserts row-count + key-order match.
- `predictions_url(dataset, model)` — gist raw URL for one prediction parquet.
- `GPN_STAR_MODELS`, `GPN_STAR_MODEL_INFO`, `GPN_STAR_SCORE_COLUMN` — metadata.

Tests at [`tests/pipelines/evals/test_gpn_star.py`](../../../tests/pipelines/evals/test_gpn_star.py)
cover alignment, NaN detection, chrom-dtype handling, and the predictions-URL
helper.
