# gpn_star_eval — GPN-Star V/M/P baseline on the eval datasets

AUPRC ± cluster-bootstrap SE for the three GPN-Star variants
(V = vertebrate-100way, M = mammal-447way, P = primate-243way; all 200M params)
on the eval datasets: the matched-pair benchmarks
([#161 Mendelian](https://github.com/Open-Athena/marin-dna/issues/161),
[#162 Complex traits](https://github.com/Open-Athena/marin-dna/issues/162)) and
saturation genome editing ([#301 SGE](https://github.com/Open-Athena/marin-dna/issues/301)).
Same metric, dataset revisions, and bootstrap config as
[`evals_v2`](../analysis/evals_v2), so these rows are directly comparable to our
own models. **This pipeline's S3 metrics parquet is the dashboard's GPN-Star
source** (`leaderboard._parquet_path` / `sge_normalized_rows` case `gpn_star`) —
the pipeline is the single source of truth; the old #145 metrics gist is kept
only as a provenance record.

GPN-Star scoring runs in [songlab-cal/TraitGym](https://github.com/songlab-cal/TraitGym),
not this repo. Predictions are pulled from gists (commits pinned in
`marin_dna.pipelines.evals.gpn_star`: `GPN_STAR_GIST_BASE` for the matched-pair
datasets, `GPN_STAR_SGE_GIST_BASE` for SGE — a separate gist); see the
[#145 calibration definition](https://github.com/Open-Athena/marin-dna/issues/145#issuecomment-4444680280),
the [matched-pair upload](https://github.com/Open-Athena/marin-dna/issues/145#issuecomment-4489509362),
and the [SGE upload](https://github.com/Open-Athena/marin-dna/issues/145#issuecomment-4683700490).
This pipeline is the align + aggregate step: load HF eval dataset, row-align with
the predictions parquet, compute AUPRC.

eQTL (#172) was retired in the 1:9 / k=9 migration (#194) and has no
current-revision GPN-Star upload, so it is not evaluated here.

## What it does

Each `dataset` in `config["datasets"]` (a `{name, hf_revision, eval_protocol}`
list pinned to the revisions evals_v2 targets) runs through two steps. The
`eval_protocol` (`matched_pair` default, or `sge`) selects the metric path —
mirroring [`conservation_eval`](../conservation_eval):

1. **Score** — for each of the 3 model variants, load the GPN-Star prediction
   parquet (`predictions_url`), align row-by-row with the HF dataset's `train`
   split **at the pinned revision**, derive `minus_llr` and
   `minus_llr_calibrated` for the leaderboard convention. Output: one long
   parquet per dataset, `model` column distinguishing V / M / P. The carried
   variant columns differ by protocol (`get_dataset_variant_columns`):
   matched-pair keeps `[…, label, subset, match_group]`; SGE keeps
   `[…, mavedb_urn, gene, subset, label]`.

   Row alignment is asserted element-wise, no key-based merge — TraitGym's
   `bolinas_pack_predictions` rule builds the predictions parquet by
   horizontal-concat with HF row order, so a `split == "train"` filter on the
   matching revision is sufficient (and `score_variants_gpn_star` fails loud if
   it ever isn't). The align is protocol-independent.

2. **Compute** AUPRC + cluster-bootstrap SE on all 4 score columns (`minus_llr`,
   `abs_llr`, `minus_llr_calibrated`, `abs_llr_calibrated`):
   - **matched_pair** → `compute_auprc_metrics`: per consequence subset, cluster
     bootstrap on `match_group`, keyed by `(model, score_type, subset)` + the
     `_global_` / `_macro_avg_` sentinel rows (`n_min` from config). Baseline
     AUPRC ≈ 0.10 under 1:9.
   - **sge** → `compute_sge_metrics`: per accession (MaveDB study) × consequence
     subset, macro-averaged over subsets then accessions, keyed by
     `(model, score_type, metric, subset, accession)` (the `_macro_avg_`
     sentinels; `n_min_auprc` from config `n_min`). Same shared metric as
     evals_v2 / conservation. The abnormal base rate (~5–16%) varies per gene.

Downstream consumers filter by `score_type` to pick the leaderboard-convention
column:

- **Mendelian:** `minus_llr_calibrated` (pathogenic ⇒ alt depleted under
  purifying selection ⇒ negative LLR ⇒ positive `minus_llr`).
- **Complex traits:** `abs_llr_calibrated` (direction-agnostic magnitude).
- **SGE:** `minus_llr_calibrated` (abnormal = loss of function = deleterious, a
  directional signal like Mendelian).

## Outputs

S3 bucket `s3://oa-bolinas/snakemake/gpn_star_eval/`:

```
results/
├── scores/{dataset}.parquet    # 3 × N rows: variant cols + scores per model
└── metrics/{dataset}.parquet   # AUPRC, keyed by eval_protocol:
                                 #  matched_pair → (model, score_type, subset),
                                 #    cols [score_type, subset, value, se,
                                 #          n_groups, n_rows, model, dataset]
                                 #  sge → (model, score_type, metric, subset,
                                 #    accession), cols [metric, subset, accession,
                                 #    gene, score_type, value, se, n, n_pos,
                                 #    model, dataset]
                                 # (no `split` col — train only; the dashboard
                                 #  reads this parquet, filtering on model)
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

Pure CPU, no GPU. Network: a few MB (downloads 9 prediction parquets — 3 models ×
3 datasets — from the pinned gists). The bootstrap dominates wallclock (~3 min CPU
per dataset at `n_bootstrap: 1000`).

```bash
cd snakemake/gpn_star_eval

# Dry-run to inspect the DAG.
uv run snakemake -n

# Real run — writes outputs to S3 per the default profile.
uv run snakemake

# Build one dataset only (e.g. just SGE).
uv run snakemake results/metrics/sge.parquet
```

No SkyPilot launch yaml — the workload is too small to be worth it.

## Library

Pipeline rules are thin glue around `marin_dna.pipelines.evals.gpn_star`:

- `score_variants_gpn_star(hf_df, predictions, split)` — load + row-align
  predictions, derive `minus_*` columns. Asserts row-count + key-order match.
  Protocol-independent (matched-pair + SGE share it).
- `predictions_url(dataset, model)` — gist raw URL for one prediction parquet
  (the hosting gist differs by dataset; SGE uses `GPN_STAR_SGE_GIST_BASE`).
- `GPN_STAR_MODELS`, `GPN_STAR_MODEL_INFO`, `GPN_STAR_SCORE_COLUMN` — metadata.

The metric functions are the same shared ones our models use:
`compute_auprc_metrics` (matched-pair) / `compute_sge_metrics` (SGE) in
`marin_dna.pipelines.evals.metrics`; SGE also reuses `SGE_VARIANT_COLUMNS` from
`marin_dna.pipelines.evals.conservation`.

Tests at [`tests/pipelines/evals/test_gpn_star.py`](../../tests/pipelines/evals/test_gpn_star.py)
cover alignment, NaN detection, chrom-dtype handling, and the predictions-URL
helper (incl. the SGE gist).
