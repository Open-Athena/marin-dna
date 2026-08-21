# evals_v2 — gLM evaluation on matched-pair datasets

AUPRC ± cluster-bootstrap SE and Group SMD on the matched-pair eval datasets
(`bolinas-dna/evals_mendelian_traits`, `bolinas-dna/evals_complex_traits`).
Each HF dataset revision is pinned per-dataset in `config.yaml` via
`hf_revision` so bumping the underlying data triggers re-execution
deterministically.
Stripped-down successor to `evals_v1`: one score rule, one metric rule, one split, no plotting, and GCS- or HF-stored checkpoints.

## What it does

For each `model` × `dataset` in the config:

1. **Download** the model checkpoint dir (from GCS or HF Hub depending on
   the model entry). The genome reference is read directly from S3 by
   pyfaidx (byte-range reads — no full download).
   Before tokenizer or model construction, `marin_dna_evals.hf_compat` reads the local `config.json` and validates its effective RoPE semantics.
   Transformers-5-only `rope_parameters` are translated in memory for the pinned Transformers 4 consumer.
   Consistent dual-schema exports are accepted, while conflicting, malformed, incomplete, or unrepresentable schemas fail before weights load.
2. **Score** every variant with `compute_variant_scores`. The score
   bundle is per-strand LLR + JSD (`down_jsd_mean` in issue #175 — the
   per-position 4-nuc next-token JSD averaged over downstream positions).
   `inference.rc=true` (default) computes both strands; the scores
   parquet then carries the raw atoms `llr_fwd`, `llr_rc`, `jsd_fwd`,
   `jsd_rc`. The metrics rule derives `_avg`, `minus_llr_*`, and
   `abs_llr_*` from these — no redundant storage. FWD+RC is the
   validated default per #175 conclusion 2.
3. **Compute** AUPRC ± cluster-bootstrap SE and Group SMD per consequence subset and score column.
   Cluster bootstrap resamples `match_group` values with replacement.
   The dataset-specific LLR protocol comes from `score_protocol` (`minus_llr` for mendelian and `abs_llr` for complex).
   Each protocol and JSD are evaluated on FWD, RC, and AVG.

Outputs land in S3 at `s3://oa-bolinas/snakemake/analysis/evals_v2/results/`:

```
results/
├── checkpoints/{model}/                         # cached HF model dir
├── scores/{model}/{dataset}.parquet             # variant cols + score atoms + emb_ref/emb_alt
└── metrics/{model}/{dataset}.parquet            # AUPRC rows enriched with Group SMD columns
```

The metrics parquet retains `[score_type, subset, value, se, n_groups, n_rows, model, dataset, split]` and appends the `group_smd_*` columns described below.
It keeps `_global_` and `_macro_avg_` rows per `score_type`.

### Matched-pair metrics (AUPRC + Group SMD, #464)

`compute_metrics` writes one row per `(score_type, subset)` to the existing `results/metrics/{model}/{dataset}.parquet` path.
The row set and the `value` and `se` columns retain the established AUPRC contract.
Direct subset and `_global_` rows also contain Group SMD summary columns.
The `_macro_avg_` rows mark Group SMD unavailable because averaging subset-standardized effects defines a different statistic.

For each match group `g`, Group SMD uses `gap_g = positive_g - mean(negatives_g)` and reports `mean(gap_g) / sample_sd(gap_g)`.
The metric is invariant to a positive affine transformation of the score.
Each match group must contain exactly one positive and at least one negative, and each group must belong to one subset.
Missing grouping columns, null grouping values, incompatible groups, and groups that span subsets raise a validation error.
A direct scope with one match group or zero gap variance reports an explicit unavailable state.

The appended columns are:

```text
group_smd_value
group_smd_se
group_smd_ci_low
group_smd_ci_high
group_smd_confidence_level
group_smd_available
group_smd_unavailable_reason
group_smd_uncertainty_method
group_smd_n_bootstrap
group_smd_n_bootstrap_valid
```

One match-group draw matrix is reused across score columns within each direct scope and discarded after the summary is computed.
No grouped-only score path, metrics path, or bootstrap sidecar is produced.
Legacy metrics parquets without the appended columns remain valid AUPRC-only artifacts and are not backfilled automatically.
The leaderboard continues selecting `value`, `se`, `n_groups`, and `n_rows`, so the added columns do not change its output.

### QTL datasets (`caqtl` / `dsqtl`, `eval_protocol: qtl_global`)

The DART-Eval Task-5 chromatin-accessibility QTL benchmarks (PR #214) are
**unmatched** — no `subset`, no `match_group`, no subsampling — so they take a
separate global path selected by `eval_protocol: qtl_global` on the dataset
entry. Scoring is identical (they still set `score_protocol: abs_llr`, so the
score columns are `abs_llr_{fwd,rc,avg}` + `jsd_{fwd,rc,avg}` — abs-LLR and
JSD), but the metric step calls
`marin_dna_evals.metrics.compute_qtl_metrics` instead, emitting **one
row per (metric × score_type)** with a `metric` column ∈ `{AUPRC, pearson,
spearman}`:

- **AUPRC** over *all* variants (significant QTL vs control via `label`), with
  a plain row-bootstrap SE.
- **Pearson / Spearman** of the score vs the dataset's `effect_size` (unsigned
  `|effect|`), over the **positive variants only** — controls are excluded
  (for `dsqtl` they carry no measured effect at all).

These metrics parquets have columns
`[metric, score_type, value, se, n_rows, n_pos, model, dataset, split]` — note
the `metric` column and the absence of subset rows.

### SGE dataset (`sge`, `eval_protocol: sge`)

The saturation-genome-editing benchmark (`bolinas-dna/evals_sge`, issue #301; v3
label build) is **unmatched** and frames the task as a binary VEP: each variant
carries a boolean `label` (True = impactful = ClinGen/ExCALIBR-calibrated
abnormal) and a consequence-group `subset` ∈ {`missense_variant`, `splicing`}.
The v3 build keeps only labeled variants (abnormal/normal); the continuous
`function_score_aligned` + `calibrated_class` columns stay for provenance.
`eval_protocol: sge` selects
`marin_dna_evals.metrics.compute_sge_metrics`. Scoring uses
`score_protocol: minus_llr` (signed — the assayed ALT is the
deleterious-*candidate*, so its sign is informative; not `abs`), giving score
columns `minus_llr_{fwd,rc,avg}` + `jsd_{fwd,rc,avg}`.

Scores are **non-comparable across studies**, so AUPRC is computed **per
accession** (`mavedb_urn`) then macro-averaged. AUPRC is rank-based, so it
compares fairly with the conservation tracks (Spearman vs the continuous
function score was dropped in #301 to keep one classification metric and let the
dataset shed unlabeled variants for faster inference).

- **AUPRC** predicting `label` (impactful vs not) from the deleteriousness score;
  requires ≥ 30 rows per label class per cell.

It runs on a 2-axis grid: `subset` ∈ {`missense_variant`, `splicing`, `both`
(pooled), `_macro_avg_` (mean of the two subsets)} × `accession` ∈ {each
`mavedb_urn`, `_macro_avg_` (mean over accessions)}. Parquet columns:
`[metric, subset, accession, gene, score_type, value, se, n, n_pos, model,
dataset, split]` (`metric` is always `AUPRC`). The **same** `compute_sge_metrics`
is reused by the `conservation_eval` baseline pipeline (now branch-run — #332). Scoped
to the three #292 gLMs via their per-model `datasets:` lists.

### Pooled embeddings (#318)

Every newly computed VEP score parquet contains pooled `emb_ref` and `emb_alt` vectors.
The global `inference.return_embeddings`, `inference.torch_compile`, and `inference.bf16` settings are all `true` and cannot be overridden by a checkpoint entry.
The embeddings come from the same FWD and RC forward passes that produce LLR and JSD.

Each allele column stores a length-`D` Float16 vector.
The model hidden states are mean-pooled across the complete DNA window with special tokens excluded.
Pooling and the FWD/RC average accumulate in Float32 before the stored vector is cast to Float16.

Embedding predictions are wider than scalar score predictions, so execution sizing remains checkpoint-specific.
Checkpoint entries may set `batch_size` and `eval_accumulation_steps`; smaller models use the global `128` and `null` fallbacks.
These execution settings do not change the intended output schema.

Score parquets created before this default may lack `emb_ref` and `emb_alt`.
They remain valid inputs for zero-shot AUPRC and Group SMD because those metrics use scalar score atoms.
They are not backfilled automatically.
A probe that targets a legacy score parquet must explicitly rerun that one `compute_scores` target first.

## Conventions

- **Train split only.** Test is held out for the final-eval pass; train is
  the development split.
- **Three context conventions are supported.** Per-model `window_size`
  config field selects the number of DNA bases extracted. The tokenizer
  loaded from each checkpoint handles BOS itself.
  - 255 = BOS-using runs (e.g. `exp136-proj_v30-step-9999`, `exp166-v0.1-p1B-step-27329`).
  - 256 = no-BOS runs at 256-token context (e.g. `exp55/58/59`).
  - 512 = no-BOS runs trained at 512 bp context (e.g. `exp21` promoter-yolo).
    Pair with a per-model `batch_size:` override to fit on an A10G; the
    global default of 128 is tuned for 256-context.

## Setup

### Supported Sky GPU runtime

The standard AWS GPU task pins the complete runtime instead of inheriting SkyPilot's changing default image.

| Component | Standard runtime | PyTorch 2.8 baseline |
| --- | --- | --- |
| Sky image | AWS `ami-0324f0ad73bdcd087`, Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20260721 | SkyPilot default GPU image used by issue #459 |
| OS | Ubuntu 24.04.4 LTS | Ubuntu 22.04 |
| NVIDIA driver | 595.71.05 | 535.216.01 |
| PyTorch | 2.13.0 | 2.8.0 |
| Compiled CUDA | 13.0 | 12.8 |
| Accelerator | NVIDIA A10G | NVIDIA A10G |

The AWS image is immutable and its release notes list A10G support, driver 595.71.05, and CUDA 13.0 in the installed stack.
CUDA 13.x requires an R580-or-newer driver, so the R595 image satisfies the major-version compatibility boundary.
The exact contract and the fixed parity cell live in [`config/gpu_runtime_validation.yaml`](config/gpu_runtime_validation.yaml).
The locked environment pins `torch==2.13.0`, whose Linux wheel reports `torch.version.cuda == "13.0"`.

Every `sky/run.yaml` setup executes `evals-gpu-runtime-check smoke` after the locked environment is installed.
The smoke gate requires `torch.cuda.is_available()`, the exact image runtime metadata, A10G bf16 support, and a finite bf16 CUDA matrix multiplication.

The numerical gate re-scores the full 16,140-row `exp351-centered-step-1000` × `mendelian_traits@4aed58e` train cell in memory with compilation, BF16, and pooled embeddings enabled.
It validates finite, equally sized Float16 embedding vectors and checks the four raw score atoms against the checksummed PyTorch 2.8 parquet, using `rtol=1e-4, atol=0.15` for LLR and `rtol=1e-3, atol=1e-4` for mean JSD.
The LLR tolerance covers the accumulated bf16 kernel differences observed when moving the fixed cell to the new driver, CUDA, and PyTorch stack.
Run the gate on a fresh one-GPU cluster with:

```bash
sky launch snakemake/analysis/evals_v2/sky/run.yaml \
  -c evals-v2-gpu-runtime-462 \
  --env GPU_RUNTIME_PARITY=true \
  --down
```

The scalar-score predecessor of this gate passed on 2026-08-20 on a fresh AWS `g5.xlarge` spot instance.
That run matched the runtime metadata and read only the pinned `train.parquet` file.
The pooled-embedding extension has not been launched as part of this change.
All 16,140 rows passed for both strands:

| Score atom | Mean absolute difference | 95th percentile | Maximum | Outside tolerance |
| --- | ---: | ---: | ---: | ---: |
| `llr_fwd` | 0.0179263 | 0.0450959 | 0.130265 | 0 |
| `llr_rc` | 0.0176735 | 0.0443012 | 0.100058 | 0 |
| `mean_jsd_fwd` | 1.23838e-6 | 3.62520e-6 | 1.92486e-5 | 0 |
| `mean_jsd_rc` | 1.21606e-6 | 3.61578e-6 | 1.49733e-5 | 0 |

The validation writes no persistent score outputs and does not access held-out labels.

The pinned image is documented by the [AWS DLAMI release index](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-24-04.html), and the driver boundary comes from the [NVIDIA CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

On a GPU node (a small EC2 GPU is sufficient for the approximately 0.6B-parameter models):

```bash
cd snakemake/analysis/evals_v2
gcloud auth application-default login
gcloud storage ls gs://marin-us-central1/checkpoints/ | head
aws s3 ls s3://oa-bolinas/snakemake/analysis/evals_v2/ 2>&1 | head

uv sync --locked --group dev --group genome-s3
uv run --locked --group genome-s3 pytest
```

## Usage

```bash
cd snakemake/analysis/evals_v2
uv run --locked --group genome-s3 snakemake -n
uv run --locked --group genome-s3 snakemake
```

The default profile (`workflow/profiles/default/config.yaml`) uses S3 storage
at `s3://oa-bolinas/snakemake/analysis/evals_v2/`.

### Interpretation targets (off `rule all`)

Two visual-interpretation analyses live alongside the metrics DAG but are kept
**off `rule all`** (so they never perturb score/metric reruns); build them by
name:

- **Nucleotide dependency maps** (categorical Jacobian, #237) — `snakemake nuc_dep`.
- **Embedding UMAP** (GPN-Star Fig 4A/4B, #246) — `snakemake umap`. Embeds the
  labeled 100 bp windows from `songlab/gpn-star-umap-regions`, fits UMAP, and
  writes `results/plots/umap/{model}/{region,conservation}.svg`. It needs the
  optional `umap` group (a ~56 MB LLVM wheel via numba/llvmlite), so install it
  alongside `--group genome-s3`:

  ```bash
  uv sync --locked --group genome-s3 --group umap
  uv run --locked --group genome-s3 --group umap snakemake umap
  ```

  On a sky cluster, pass `EXTRA_UV_GROUPS` (threaded into both `uv sync` and
  `uv run` by `sky/run.yaml`):

  ```bash
  sky launch sky/run.yaml -c evals-umap \
    --env EXTRA_UV_GROUPS="--group umap" \
    --env SNAKEMAKE_ARGS="-- umap"
  ```

- **LL gap** (functional vs non-functional log-likelihood, #274) — `snakemake ll_gap`.
  For each `(model, region)` in the `ll_gap:` config it scores the model's mean
  log-likelihood on uppercase (phyloP-functional) vs lowercase (non-functional)
  target tokens over the mixed-case `genomes-v5` validation intervals
  (`cds`/`upstream`/`downstream` = v5/v1/v15), then aggregates to
  `results/ll_gap/summary.parquet` (`LL_upper`, `LL_lower`, `gap` per cell). A
  metric rather than an interpretation, but kept off `rule all` for the same
  reason. FWD strand only — matches the training-logged
  `val_*_{functional,nonfunctional}` loss. For one sky cluster per cell, build
  the per-cell `results/ll_gap/scores/{model}/{region}.parquet` targets, then
  gather with `snakemake ll_gap`.


### m1.3 likelihood dynamics (#489)

`snakemake likelihood_dynamics_489_pilot` scores a 128-window prefix of each blog validation probe at the earliest and terminal selected checkpoints.
The pilot is off `rule all` and runs one forward orientation.
It validates token identities, row counts, finite NLL and A/C/G/T entropy, and agreement between per-token NLL sums and the existing aggregate LL kernel.

The five pinned probes are CDS, upstream, downstream, ncRNA, and enhancer.
The first three are the issue #478 `genomes-v5-validation-*` datasets on RefSeq `GCF_000001405.40`.
The ncRNA and enhancer probes are `zoonomia-v1-val_ncrna` and `zoonomia-v1-val_enhancer` on the Ensembl release 115 GRCh38 primary assembly.
The workflow reads each matching soft-masked reference and rejects any uppercase sequence mismatch before assigning repeat labels.
Lowercase conservation case means either below-threshold phyloP or missing alignment, and the atom schema records that unresolved provenance.

The five checkpoints correspond to consolidated steps 10,000, 30,000, 50,000, 70,000, and 82,823.
Their exact cumulative token counts are stored in `config/config.yaml`.
The first four checkpoints are separated by 41,943,040,000 tokens, and the terminal gap is 26,891,780,096 tokens.

Build reusable metadata and full score atoms with:

```bash
snakemake likelihood_dynamics_489_metadata
snakemake likelihood_dynamics_489_atoms
```

Outputs use the versioned prefix `results/m13_likelihood_dynamics_489/v1/`.
Each score cell stores one row per source token with `(region, row_index, target_pos)` as its stable identity.
The atom cache includes genomic coordinates, source case, conservation status and provenance, repeat status, GC, chromosome-held-out 7-mer NLL, forward NLL, and four-nucleotide entropy.

### Linear probe (frozen-embedding VEP, #320)

`snakemake probe` trains a **frozen-embedding linear probe** per `(model,
dataset)` — the productionized form of #314's settled protocol — also kept **off
`rule all`**.
It consumes the in-bundle pooled embeddings (`emb_ref`/`emb_alt`, the #318 columns).
New score outputs include those columns by default.
A legacy score parquet without them fails fast and requires an explicit targeted score rerun.
The rule is CPU-only; the probe logic lives in
`marin_dna_evals.variant_probe.run_subset_probes`.

The protocol is **one** approach, no sweeps:

- **Feature** — `emb_ref`/`emb_alt` upcast f16→**f32** (a cancellation guard, #318),
  then one per-dataset pair-feature: `concat_ref_delta = [ref, alt−ref]` for
  directional datasets (`score_protocol: minus_llr`), `sum_absdiff = [ref+alt,
  |alt−ref|]` for swap-invariant ones (`abs_llr`). Override per dataset with an
  optional `probe_feature:`.
- **Per consequence `subset`** (or one synthetic `all` group when the dataset has no
  `subset`, e.g. caqtl/dsqtl), trained only if it clears `min_variants` **and**
  `min_chroms`; smaller subsets get `NaN` `probe_score` and no classifier.
- **Probe** — `StandardScaler → LogisticRegression(L2)`; the only tuned knob is the
  L2 strength `C`.
- **CV** — leave-one-chromosome-out predictions with an inner `GroupKFold`
  `GridSearchCV` re-tuning `C` per fold (leakage-free, the TraitGym protocol); the
  reusable classifier is the same pipeline fit on all the subset's variants.
- **C-edge diagnostic (verified)** — `c_grid = logspace(-12, 4, 17)` is anchored at
  both regularization limits: the high end (`1e4`) is a saturation cap whose ranking
  equals the unregularized `C→∞` limit (no `inf` needed), and the low end (`1e-12`)
  is a heavy-reg floor (as `C→0` the L2 fit shrinks to the scale-free mean-difference
  direction — *not* a constant predictor, since AUPRC is rank-based). For each subset
  the joblib `c_summary` records the pin counts **and verifies** them: from the
  all-data inner-CV curve, `high_edge_gain` / `low_edge_gain` measure whether a
  pinned edge is still improving vs its interior neighbor, and `truncation_risk`
  fires only if it is (which the anchored grid avoids). So an edge pin is confirmed
  *saturated/flat (benign)* rather than assumed.

Two artifacts per cell:

```
results/probe/{model}/{dataset}.parquet   # variant cols (minus emb_ref/emb_alt) + probe_score (NaN for skipped subsets)
results/probe/{model}/{dataset}.joblib    # {subset: {pipeline, C, feature, n, n_pos, c_summary}}
```

The predictions parquet (the LOOC `probe_score` per variant) is consumed by the
**`compute_probe_metrics`** rule below; the joblib classifiers are
serialized for **reuse on other datasets**. Configured under `probe:` in
`config.yaml` (`min_variants`, `min_chroms`, `c_grid` = `logspace(lo, hi, num)`,
`inner_splits`, `n_jobs`, and `models: [{name, datasets}]`). Build:

```bash
snakemake probe
snakemake results/probe/<model>/<dataset>.parquet

# A legacy cell without embedding columns requires an explicit targeted rerun.
snakemake --forcerun compute_scores -- results/scores/<model>/<dataset>.parquet
```

### Linear-probe metrics (per-subset per-chrom AUPRC, #331/#341)

`snakemake probe_metrics` scores the probe against its matched zero-shot LLR
baseline, per `(model, dataset)`, also **off `rule all`**. The `compute_probe_metrics`
rule reads a `results/probe/{model}/{dataset}.parquet` — which carries **both**
`probe_score` **and** the raw `llr_fwd`/`llr_rc` atoms (only `emb_ref`/`emb_alt` are
dropped) — so the probe and its baseline are scored on **identical rows** under one
metric. It emits, per consequence `subset`, the **per-chromosome-weighted AUPRC** (the
TraitGym / #314 headline; `marin_dna_evals.metrics.per_chrom_ap_table` →
`per_chrom_weighted_ap`) for two score types: `probe_score` and the dataset's
zero-shot baseline (its `score_protocol` applied to the FWD/RC-averaged LLR, e.g.
`minus_llr_avg` for mendelian). Routed by `eval_protocol`: **`matched_pair`** (mendelian /
complex; needs `subset` + `chrom`) takes the per-chromosome-weighted path above, while
**`sge`** takes a per-accession (`mavedb_urn`) × consequence-subset AUPRC macro-averaged over
genes (`compute_sge_probe_metrics` → `compute_sge_metrics`) — dropping the pooled `both`
scope, since the separate per-subset probe classifiers aren't comparable across subsets.
`qtl_global` is rejected.

```
# matched_pair: [score_type, subset, value, se, n, n_pos, n_chrom, model, dataset, split]
results/probe_metrics/{model}/{dataset}.parquet
# sge:          [metric, subset, accession, gene, score_type, value, se, n, n_pos, model, dataset, split]
```

```bash
snakemake probe_metrics
snakemake results/probe_metrics/<model>/<dataset>.parquet
```

### Parallel sky-cluster sweep (one cluster per target)

For a grid of independent targets — e.g. all checkpoints of one model arm,
or one cluster per (model, dataset) combination — use
[`sky/parallel_sweep.sh`](sky/parallel_sweep.sh). It dispatches one
g5.xlarge per target with `--down` on idle, and waits for all to finish.

The helper `cd`s to the repo root internally, so it's safe to invoke
from anywhere (e.g. from this pipeline dir or from `~`). Target paths
are interpreted relative to *this* pipeline dir, since that's what the
cluster's snakemake sees after its own `cd snakemake/analysis/evals_v2`.
Example:

```bash
snakemake/analysis/evals_v2/sky/parallel_sweep.sh \
  results/metrics/exp55-humans-step-16999/mendelian_traits.parquet \
  results/metrics/exp55-primates-step-16999/mendelian_traits.parquet
```

Cluster name = `evals-v2-{model}` derived from the target's parent dir,
so you can't pass `mendelian_traits` and `complex_traits` for the *same*
model in one invocation — split into two batches.

Two unavoidable AWS-side failure modes worth knowing about:

- **`VcpuLimitExceeded`**: bursting more g5.xlarge in one invocation than
  `vCPU_limit / 4` (us-east-2 default: 128 / 4 = 32 simultaneous
  g5.xlarge) hits the account-level vCPU limit. Re-run the helper with
  the failed target names after other clusters `--down`.
- **`ResourcesUnavailableError: Failed to acquire resources in all zones
  in us-east-2`**: occasional transient AZ saturation, even when well
  under the vCPU limit. Single-target retry usually succeeds on the
  next AZ rotation.

## Likelihood dynamics analysis (#489)

The likelihood_dynamics_489_analysis named target consumes the immutable full forward-token cache under results/m13_likelihood_dynamics_489/v1/.
It does not rescore checkpoints.

Run it after likelihood_dynamics_489_atoms:

~~~bash
snakemake likelihood_dynamics_489_analysis
~~~

The primary population is target positions [32, 223) after excluding repeats, ambiguous bases, and unscorable tokens.
All conservation labels retain the source dataset's case-derived missingness limitation.
The analysis writes:

- exact conservation AUPRC from negative loss and negative entropy, globally and by region;
- earliest-versus-terminal mean-loss trajectory groups;
- exact Jaccard overlap for the lowest-score 10% selected independently within each region;
- next-checkpoint and terminal loss reduction by region-specific current-loss decile;
- loss and entropy distribution summaries by conservation label;
- controlled negative-score contrasts using GC content, local 7-mer NLL, and target-position polynomials;
- a validation manifest and six SVG research figures.

Mean summaries and controlled contrasts use 500 resamples of region-specific 10 Mb genomic blocks for 95% intervals.
Exact AUPRC and Jaccard values describe the complete fixed validation population and are not resampled.
The complete output namespace is results/m13_likelihood_dynamics_489/v1/analysis/.

## Configuration (`config/config.yaml`)

| Key | Purpose |
| --- | --- |
| `input_hf_prefix` | HF prefix for `f"{prefix}_{dataset.name}"`. |
| `genome_path` | Canonical GRCh38 FASTA. fsspec URI (e.g. `s3://...`) or local path. The S3 path requires `--group genome-s3` at install time. |
| `split` | `train` (or `test` once held-out eval is unlocked). |
| `datasets` | List of `{name, hf_revision, score_protocol, [eval_protocol]}`. `hf_revision` is the pinned HF dataset commit SHA — bumping it triggers re-execution. `score_protocol` ∈ `{minus_llr, abs_llr}`. Optional `eval_protocol` ∈ `{matched_pair (default), qtl_global, sge}` — `qtl_global` selects the global AUPRC + positives-only `effect_size` correlation path for the unmatched caqtl/dsqtl datasets; `sge` selects the per-accession × consequence-subset AUPRC-on-`label` path for `evals_sge` (see the SGE section above). |
| `models` | List of `{name, window_size, ...}`. Each entry has exactly one checkpoint source. Optional execution fields are `batch_size` and `eval_accumulation_steps`; `datasets` restricts evaluation coverage. Semantic inference switches are rejected here. |
| `inference.*` | Global `return_embeddings`, `torch_compile`, and `bf16` settings are required to be `true`. `batch_size: 128` and `eval_accumulation_steps: null` are execution fallbacks. The section also configures workers, RC scoring, transforms, and metric bootstrap settings. |
| `nuc_dep` | Optional; nucleotide-dependency maps (#237, off `rule all`). `{combines, ord, batch_size, dpi, models: [...], loci: {...}}`. See `rules/interpretation.smk`. |
| `umap_embeddings` | Optional; embedding UMAP (#246, off `rule all`). `{dataset, layer_index, n_center_bp, random_state, dpi, models: [...]}` — `models` reuse the `models:` registry (each needs `window_size`). Build needs `--group umap` (+ `--group genome-s3`). See `rules/embedding_umap.smk`. |
| `ll_gap` | Optional; functional/non-functional LL gap (#274, off `rule all`). `{split, datasets: [{name, hf_repo, hf_revision}], models: [...]}` — `datasets` are mixed-case `seq` HF datasets (the v5/v1/v15 validation intervals; NOT the variant `datasets:` above); `models` reuse the `models:` registry. See `rules/ll_gap.smk`. |
| `likelihood_dynamics_489` | Optional; versioned forward per-token NLL and entropy cache through the five-checkpoint m1 lineage. The section pins five blog validation probes, their matching soft-masked reference families, cumulative token counts, pilot size, and per-checkpoint batch sizes. See `rules/likelihood_dynamics_489.smk`. |

## Library

Pipeline rules are thin glue around:

- `marin_dna_evals.hf_compat` — fail-closed Transformers 4/5 config normalization plus the `TokenizersBackend` tokenizer fallback used by every maintained HF model-loading path.
- `marin_dna_evals.inference.compute_variant_scores` — model + genome
  → per-strand score atoms (`llr_fwd`, `llr_rc`, `jsd_fwd`, `jsd_rc`).
- `marin_dna_evals.metrics.compute_auprc_metrics` — score columns
  → AUPRC ± cluster-bootstrap SE per subset (cluster = `match_group`).
- `marin_dna_evals.grouped_vep_metrics.compute_grouped_vep_metrics` → the unchanged AUPRC table with additive Group SMD columns and explicit unavailable states.
- `marin_dna_evals.grouped_vep_metrics.group_smd` — mean matched-group gap divided by the sample SD of matched-group gaps.
- `marin_dna_evals.metrics.compute_qtl_metrics` — score columns
  → global AUPRC + positives-only Pearson/Spearman vs `effect_size`
  (the `eval_protocol: qtl_global` path for caqtl/dsqtl).
- `marin_dna_evals.ll_gap.compute_hf_ll_gap` — HF checkpoint +
  mixed-case `seq` dataset → per-sequence functional/non-functional LL atoms
  (`ll_sum_upper`, `ll_sum_lower`, `n_upper`, `n_lower`); `aggregate_ll_gap`
  collapses them to token-weighted `LL_upper` / `LL_lower` / `gap`.
- `marin_dna_evals.likelihood_dynamics_489.compute_hf_per_token_stats` maps each 255 bp source window to forward NLL and A/C/G/T entropy arrays.
- `marin_dna_evals.likelihood_dynamics_489.build_window_metadata` joins conservation case and repeat-mask case against the dataset's matching assembly.
- `marin_dna_evals.likelihood_dynamics_489.assemble_token_atoms` expands those arrays into stable versioned token rows.

These are tested at `tests/evals/test_grouped_vep_metrics.py`,
`tests/evals/test_hf_compat.py`,
`tests/evals/test_metrics.py`,
`tests/evals/test_inference.py`,
`tests/evals/test_ll_gap.py`, and `tests/model/test_scoring.py`.
