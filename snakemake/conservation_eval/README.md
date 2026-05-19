# Conservation scores baseline (issue #146)

Per-variant conservation scores on the matched-pair eval datasets produced by
`snakemake/evals/`:

- `bolinas-dna/evals_mendelian_traits` — Mendelian-disease pathogenic SNVs vs
  gnomAD common (MAF ≥ 5 %), 1:9 k-nearest-neighbor matched (PR #194).
- `bolinas-dna/evals_complex_traits` — UKBB fine-mapped (PIP > 0.9) vs low-PIP
  (< 0.01), 1:9 k-nearest-neighbor matched (PR #194).

A classical baseline alongside model-based evals — same metric (AUPRC +
cluster-bootstrap SE) and same pinned HF revisions as the `evals_v2`
pipeline (`snakemake/analysis/evals_v2/`), so conservation rows on the
leaderboard are apples-to-apples with bolinas rows.

## What it does

For each `dataset` in `config["datasets"]`, each `split` in `config["splits"]`
(default: `train` only — test is held out for the final-eval pass), and each
`score` in `config["scores"]`:

1. **Download** the bigWig from UCSC / Zoonomia (`results/conservation/{score}.bw`).
2. **Score** each variant by single-base lookup at its 1-based `pos` (converted
   to pyBigWig's 0-based half-open `[pos-1, pos)`). NaN is preserved where the
   bigWig has no aligned data. The HF dataset commit is pinned per-dataset via
   `hf_revision`; bumping the SHA in config triggers re-execution via
   snakemake's `params:` hash.
3. **Aggregate** the scored parquets into one AUPRC table per `(dataset,
   split)` (`results/{dataset}/results_table_{split}.md`) plus a long-form
   `metrics_{split}.parquet`.

## Metric: AUPRC ± cluster-bootstrap SE

The eval datasets are matched groups: every positive variant shares a
`match_group` with its k nearest-neighbor negatives (k=9 since PR #194). Rows
within a group are not iid — a naive row bootstrap would resample positives
and negatives independently and underestimate SE.

We use a **cluster bootstrap on `match_group`**: each iteration resamples the
unique `match_group` IDs with replacement, gathers all rows belonging to the
sampled groups (with multiplicity), and recomputes AUPRC. SE is the std of the
bootstrap distribution. The point estimate is AUPRC on the original rows.

- `n_bootstrap` (default 1000) — bootstrap iterations per (subset, score).
- `bootstrap_seed` (default 0) — pinned seed so re-runs reproduce bit-for-bit.

The markdown report shows per-subset rows (consequence groups: missense,
distal, splicing, promoter, …) — `_global_` and `_macro_avg_` aggregate rows
flow through the parquet (used by the dashboard) but are excluded from the
markdown.

## Tracks

| Name | Source URL | Notes |
| --- | --- | --- |
| `phyloP_100v` | `hgdownload.soe.ucsc.edu/.../hg38.phyloP100way.bw` | 100-vertebrate phyloP (UCSC multiz) |
| `phastCons_100v` | `hgdownload.soe.ucsc.edu/.../hg38.phastCons100way.bw` | 100-vertebrate phastCons (UCSC multiz) |
| `phyloP_241m` | `hgdownload.soe.ucsc.edu/.../cactus241way.phyloP.bw` | Zoonomia 241-mammal Cactus phyloP |
| `phyloP_447m` | `hgdownload.soe.ucsc.edu/.../hg38.phyloP447way.bw` | UCSC 447-way phyloP (Zoonomia + densely-sampled primates, Kuderna et al. 2023) |
| `phyloP_470m` | `hgdownload.soe.ucsc.edu/.../hg38.phyloP470way.bw` | UCSC 470-way phyloP (multiz; parallel work to 447-way Cactus, not a successor) |
| `phastCons_470m` | `hgdownload.soe.ucsc.edu/.../hg38.phastCons470way.bw` | UCSC 470-way phastCons (multiz; parallel work to 447-way Cactus, not a successor) |
| `phastCons_43p` | `cgl.gi.ucsc.edu/.../phyloPPrimates.bigWig` | Zoonomia 43-primate. Name follows TraitGym; underlying file is phyloP-over-primates. |

URLs are owned by `bolinas.evals.conservation.CONSERVATION_TRACKS` (single source of truth).

## NaN handling

NaN values arise when the bigWig has no alignment at a given locus (e.g.
patches, alt contigs, or genuinely unalignable bases). Before computing AUPRC
we **`fillna(0)`**:

- For phyloP, **0 is semantically meaningful**: neither conserved nor accelerated.
- For phastCons, **0 is also semantically meaningful**: non-conserved.

This matches the choice in TraitGym's own `eval/workflow/rules/conservation.smk`.
`auprc_with_bootstrap_se` also asserts no NaN scores, so the fill is required
upstream of the metric call.

The per-variant parquets (`{score}_{split}.parquet`) preserve raw NaNs so you
can apply a different policy without re-running scoring. Filling happens only
at the aggregation stage.

NaN counts are surfaced in `results_table_{split}.md` (per `(score, subset)`)
so the size of the choice is visible.

## Usage

```bash
cd snakemake/conservation_eval

# Dry-run to inspect the DAG
uv run snakemake -n

# Run locally (CPU-only — wall time is dominated by bigWig downloads on a
# cold S3 cache; the full 7-track set is ~50 GB total). The cluster bootstrap
# (1000 iters × 7 tracks × ~10 subsets × 2 datasets ≈ 140k iters) is fast.
uv run snakemake
```

Heavy first-time runs should go on SkyPilot — see `sky/run.yaml`:

```bash
sky launch snakemake/conservation_eval/sky/run.yaml -c conservation-eval
```

The default profile (`workflow/profiles/default/config.yaml`) uses S3 storage
at `s3://oa-bolinas/snakemake/conservation_eval/`. AWS credentials need S3
access — same setup as `snakemake/evals/`.

## Outputs

```
results/
├── conservation/
│   └── {score}.bw                          # one per track in config["scores"]
└── {dataset}/
    ├── {score}_{split}.parquet             # per-variant score (NaN preserved)
    ├── metrics_{split}.parquet             # long-form metrics table
    └── results_table_{split}.md            # markdown report (AUPRC ± SE)
```

Each `metrics_{split}.parquet` has columns `[score_type, score_name, subset,
value, se, n_groups, n_rows, n_nan, n_total, split, dataset]`.

## Library

Snakemake rules are thin glue around `bolinas.evals.conservation`:

- `score_variants_at_positions(df, bw_path)` — bigWig lookup, NaN preserved.
- `aggregate_conservation_metrics(parquet_paths, *, n_bootstrap, bootstrap_seed)`
  — `(metrics_df, markdown)` with AUPRC + cluster-bootstrap SE.

Tests live at `tests/pipelines/evals/test_conservation.py` and
`tests/pipelines/evals/test_pairwise_accuracy.py` (CPU-only, no real bigWig download).
