# alphagenome_eval — AlphaGenome baseline on matched-pair eval datasets

AUPRC ± cluster-bootstrap SE (cluster = `match_group`) for
[AlphaGenome](https://github.com/google-deepmind/alphagenome) on the
matched-pair eval datasets `bolinas-dna/evals_mendelian_traits` and
`bolinas-dna/evals_complex_traits` (1:k matched groups, PR #194 rebuild).
Provides issue [#154](https://github.com/Open-Athena/bolinas-dna/issues/154)'s
baseline row for the leaderboards in
[#161](https://github.com/Open-Athena/bolinas-dna/issues/161) and
[#162](https://github.com/Open-Athena/bolinas-dna/issues/162). The metric
mirrors `snakemake/analysis/evals_v2/`'s post-PR-#195 schema.

## What it does

For each `dataset` in `config["datasets"]`:

1. **Score** every variant via the AlphaGenome API. One forward-strand call per
   variant returns L2_DIFF_LOG1P aggregated scores across 7 assays (ATAC,
   DNASE, CHIP_TF, CHIP_HISTONE, CAGE, PROCAP, RNA_SEQ); each assay yields
   many tracks (one per cell type / tissue), so the per-variant output is a
   wide row with hundreds of track columns.
2. **Aggregate** to a single column by taking the max across all track
   columns: `alphagenome_max_l2`. The full per-track table is preserved on S3
   so the aggregation protocol can change later (e.g. per-assay) without
   re-spending the API budget.
3. **Compute** AUPRC ± cluster-bootstrap SE per consequence subset on
   `alphagenome_max_l2`, with `match_group` as the cluster (resamples
   groups, not rows, so SE reflects the 1:k matched structure). Same
   column for both datasets — L2 is direction-agnostic and fits both the
   mendelian and complex-trait protocols. Output includes per-subset
   rows plus `_global_` (pooled) and `_macro_avg_` (mean of qualifying
   subsets) aggregates.

## Outputs

S3 bucket `s3://oa-bolinas/snakemake/alphagenome_eval/`:

```
results/
├── per_track_l2/{dataset}.parquet    # variant cols + per-track L2 columns
├── scores/{dataset}.parquet          # variant cols + alphagenome_max_l2
└── metrics/{dataset}.parquet         # AUPRC ± cluster-bootstrap SE per subset
```

## Conventions

- **Train split only.** Test is held out for the final-eval pass.
- **No reverse-complement averaging.** TraitGym's reference pipeline averages
  forward + reverse strand calls. We skip the RC pass to halve the API budget;
  the metric loss has been small in practice.
- **No edge filtering.** The 1MB sequence context wraps near chromosome ends;
  AlphaGenome handles this internally.

## Setup

`ALPHA_GENOME_API_KEY` env var (request one at
[deepmind.google/alphagenome](https://deepmind.google/science/alphagenome)).
Recommended: export it in `~/.bashrc` so every shell — including the one you
launch SkyPilot from — has it set.

```bash
# In ~/.bashrc:
export ALPHA_GENOME_API_KEY=...
```

The pipeline reads it from the env at the start of `compute_per_track_l2`;
SkyPilot inherits it via the `envs:` block in `sky/run.yaml`.

The official Google `alphagenome` Python client is in the optional
`alphagenome-eval` dep group:

```bash
uv sync --group alphagenome-eval
```

(SkyPilot's `setup:` does this automatically.)

## Usage

### SkyPilot (recommended)

```bash
# Once per session (or persisted in ~/.bashrc):
export ALPHA_GENOME_API_KEY=...

sky launch snakemake/alphagenome_eval/sky/run.yaml -c alphagenome-eval \
    --env ALPHA_GENOME_API_KEY
```

The launch yaml provisions a small CPU EC2 node (`m6i.xlarge`-class — 16 GB
to leave headroom for the per-variant API responses, us-east-2), runs the
full pipeline (~2-3 h end-to-end), and writes outputs to S3. Tear down with
`sky down alphagenome-eval`.

> **Re-running an existing dataset:** if the rule code or library changes,
> Snakemake's provenance check will re-trigger `compute_per_track_l2` and
> burn the API budget again. If the change is known to produce identical
> outputs, skip the rerun with `snakemake --cleanup-metadata results/per_track_l2/{dataset}.parquet`.

### Local (small subsets only)

```bash
cd snakemake/alphagenome_eval

# Dry-run to inspect the DAG.
uv run snakemake -n

# Real run — will hit the AlphaGenome API for ~12K variants if you don't
# subsample first; only do this if you mean to pay the wallclock.
uv run snakemake
```

## Configuration (`config/config.yaml`)

| Key | Purpose |
| --- | --- |
| `input_hf_prefix` | HF prefix for `f"{prefix}_{dataset}"`. |
| `split` | `train` (test held out). |
| `datasets` | List of `{name, hf_revision}` entries. The SHA pins the HF commit consumed; bumping it forces a re-run via snakemake's `params:` hash (re-spends API budget). SHAs mirror `snakemake/analysis/evals_v2/config/config.yaml`. |
| `num_workers` | Threads in the API ThreadPoolExecutor. Keep ≤ 4. |
| `score_column` | Column name written by `aggregate_max` and consumed by `compute_metrics`. |
| `n_bootstrap` | AUPRC cluster-bootstrap iterations per subset. |
| `bootstrap_seed` | RNG seed for the bootstrap; bumping it re-triggers `compute_metrics`. |

The 7 assays, the 1MB sequence length, and `L2_DIFF_LOG1P` aggregation type
are **code constants** in `bolinas.evals.alphagenome`, not config.

## Library

Pipeline rules are thin glue around `bolinas.evals.alphagenome`:

- `score_variants_alphagenome(V, num_workers=4)` — main entry; threads through
  forward-strand `model.score_variant` calls and returns a wide DataFrame.
- `parse_score_response(tidy, scorer_repr_to_assay)` — pure helper converting
  AlphaGenome's `tidy_scores` long-format output to a 1-row wide DataFrame
  with `{assay}_{idx}` column names.
- `make_scorers()` — the 7 `CenterMaskScorer(width=None, L2_DIFF_LOG1P)`
  scorers and their reverse map.

Tests at `tests/evals/test_alphagenome.py` cover the parser without touching
the API; the scorer-construction test is gated on `import alphagenome`.
