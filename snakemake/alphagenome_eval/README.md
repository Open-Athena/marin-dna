# alphagenome_eval — AlphaGenome predictions for the eval datasets

Runs the [AlphaGenome](https://github.com/google-deepmind/alphagenome) API to produce
per-variant predictions, on a small CPU node, writing to S3. Two independent paths share
this pipeline (and its one AlphaGenome API setup):

1. **Matched-group baseline** (`mendelian_traits` / `complex_traits`) — AUPRC ±
   cluster-bootstrap SE, the AlphaGenome baseline row for the mendelian/complex
   leaderboards ([#161](https://github.com/Open-Athena/marin-dna/issues/161) /
   [#162](https://github.com/Open-Athena/marin-dna/issues/162)).
2. **DNase-LFC QTL predictions** (`caqtl` / `dsqtl`) — the single GM12878-DNase LFC
   scorer for the supervised accessibility-QTL benchmark
   ([#311](https://github.com/Open-Athena/marin-dna/issues/311) / #309). This pipeline
   produces only AlphaGenome's per-variant predictions; the model-agnostic **metrics and
   leaderboard** (ChromBPNet/Enformer baselines + the official metrics) are the
   `scripts/qtl_benchmark.py` driver, not a rule here.

## What it does

### Matched-group baseline (`mendelian_traits`, `complex_traits`)

For each `dataset` in `config["datasets"]`:

1. **Score** every variant via the AlphaGenome API — one forward-strand call returns
   L2_DIFF_LOG1P scores across 7 assays (ATAC, DNASE, CHIP_TF, CHIP_HISTONE, CAGE,
   PROCAP, RNA_SEQ); each assay yields many tracks (cell types / tissues), so the
   per-variant output is a wide row with hundreds of track columns.
2. **Aggregate** to one column by max across all tracks: `alphagenome_max_l2`. The full
   per-track table is preserved on S3 so the aggregation can change later (e.g.
   per-assay) without re-spending the API budget.
3. **Compute** AUPRC ± cluster-bootstrap SE per consequence subset on `alphagenome_max_l2`,
   with `match_group` (the 1:k matched set) as the resampling cluster. Output includes
   per-subset rows plus `_global_` (pooled) and `_macro_avg_` aggregates.

### DNase-LFC QTL predictions (`caqtl`, `dsqtl`)

`score_dnase_lfc` scores each variant with AlphaGenome's recommended accessibility
scorer (Suppl Table 9): `CenterMaskScorer(DNASE, width=501, DIFF_LOG2_SUM)`,
GM12878-matched (ontology `EFO:0002784`) — one signed log2 fold-change (alt vs ref) per
variant. The scorer is **resumable + S3-checkpointed** (`score_dnase_lfc_resumable`): it
seeds from `dnase_lfc.s3_prefix/{ds}.parquet` and scores only missing variants.

> **caQTL/dsQTL spend 0 API.** Their genome-native predictions already exist (the #262
> run, corrected once into the dataset's sign convention by
> `scripts/correct_ag_predictions.py`), so the rule seeds them and makes no API calls
> (`max_new_calls: 0` is a fail-loud guard). Raise `max_new_calls` only to score a
> genuinely new dataset.

## Outputs

S3 bucket `s3://oa-bolinas/snakemake/alphagenome_eval/`:

```
results/
├── per_track_l2/{dataset}.parquet   # matched-group: variant cols + per-track L2
├── scores/{dataset}.parquet         # matched-group: variant cols + alphagenome_max_l2
├── metrics/{dataset}.parquet         # matched-group: AUPRC ± cluster-bootstrap SE
├── dnase_lfc/{ds}.parquet            # caqtl/dsqtl: genome-native GM12878-DNase LFC
└── dnase_lfc_raw/{ds}.parquet        # caqtl/dsqtl: raw #262 originals (pre sign-alignment)
```

The supervised-benchmark `scores/{model}/{ds}` and `metrics/{model}/{ds}` parquets live
under a separate prefix, `s3://oa-bolinas/qtl_benchmark/`, written by
`scripts/qtl_benchmark.py`.

## Conventions

- **Matched-group: train split only.** Test is held out for the final-eval pass.
- **No reverse-complement averaging.** TraitGym's reference averages forward + reverse
  strand; we score forward only to halve the API budget (metric loss is small — it is
  the ~0.004 gap on the caQTL AlphaGenome *direction* reproduction).
- **DNase-LFC sign convention.** AlphaGenome's raw DNase-LFC is uniformly sign-flipped on
  dsQTL relative to the study (a lift artifact of the #262 run). The genome-native
  predictions are aligned once to the carried ChromBPNet baseline by
  `scripts/correct_ag_predictions.py`; downstream there is no per-dataset flip.
- **Retries transient `INTERNAL` errors.** AlphaGenome's backend intermittently returns
  `StatusCode.INTERNAL` ("bad machine" outages). The SDK's `@retry_rpc` only retries
  `RESOURCE_EXHAUSTED` / `UNAVAILABLE`, so the scorers re-wrap `score_variant` to also
  retry `INTERNAL` / `DEADLINE_EXCEEDED` (10 attempts, exponential backoff). A single
  un-retried bad-machine hit would otherwise abort a whole dataset.

## Setup

`ALPHA_GENOME_API_KEY` env var (request one at
[deepmind.google/alphagenome](https://deepmind.google/science/alphagenome)). Export it in
`~/.bashrc` so every shell — including the one you launch SkyPilot from — has it set.

The official Google `alphagenome` client is in the optional `alphagenome-eval` dep group:

```bash
uv sync --group alphagenome-eval
```

(SkyPilot's `setup:` does this automatically.) caQTL/dsQTL reuse the cached predictions,
so a run that only refreshes `dnase_lfc` makes no API calls.

## Usage

### SkyPilot (recommended)

```bash
export ALPHA_GENOME_API_KEY=...
sky launch snakemake/alphagenome_eval/sky/run.yaml -c alphagenome-eval \
    --env ALPHA_GENOME_API_KEY
```

Provisions a small CPU node (us-east-2), runs the full pipeline, writes to S3. Tear down
with `sky down alphagenome-eval`.

> **Re-running matched-group scoring:** a rule-code or library change re-triggers
> `compute_per_track_l2` and burns the API budget again. If the change is known to
> produce identical outputs, skip with
> `snakemake --cleanup-metadata results/per_track_l2/{dataset}.parquet`.

### Local

```bash
cd snakemake/alphagenome_eval
uv run snakemake -n          # dry-run; confirm score_dnase_lfc is NOT scheduled to score
uv run snakemake             # caqtl/dsqtl: 0 API; matched-group: ~11K+ API calls each
```

## Configuration (`config/config.yaml`)

| Key | Purpose |
| --- | --- |
| `input_hf_prefix` | HF prefix for `f"{prefix}_{dataset}"`. |
| `split` | Matched-group split (`train`; test held out). |
| `datasets` | Matched-group `{name, hf_revision}` entries (mendelian/complex). The SHA pins the HF commit; bumping it re-runs scoring (re-spends API). |
| `num_workers` | API ThreadPoolExecutor threads. Keep ≤ 4. |
| `score_column` | Column written by `aggregate_max`, read by `compute_metrics`. |
| `n_bootstrap` / `bootstrap_seed` | Matched-group AUPRC cluster-bootstrap. |
| `subset_n_pairs` | Smoke knob: keep only the first N match groups (null in production). |
| `dnase_lfc` | `{s3_prefix, datasets:[{name,hf_revision}], num_workers, chunk_size, max_new_calls}` for the caqtl/dsqtl DNase-LFC predictions. `max_new_calls: 0` ⇒ reuse-only (fail loud if the cache is incomplete). |

The 7 assays, the 1MB sequence length, and the aggregation types are **code constants**
in `marin_dna.pipelines.evals.alphagenome`, not config.

## Library

Rules are thin glue around `marin_dna.pipelines.evals.alphagenome`:

- `score_variants_alphagenome(V, num_workers=4)` — matched-group 7-assay L2 scorer.
- `score_variants_dnase_lfc(V, num_workers=4)` / `make_dnase_lfc_scorer` /
  `select_gm12878_dnase_lfc` — the single GM12878-DNase LFC scorer.
- `score_dnase_lfc_resumable(variants, checkpoint_path, …)` — resumable, S3-checkpointed
  DNase-LFC scoring (seed + score-missing + checkpoint, with a `max_new_calls` cap).

Tests at `tests/pipelines/evals/test_alphagenome.py` cover the parsers + the resumable
scorer without touching the API (the scorer-construction tests are import-gated).
