# evals_v2 — gLM evaluation on matched-pair datasets

AUPRC ± cluster-bootstrap SE on the matched-pair eval datasets
(`bolinas-dna/evals_mendelian_traits`, `bolinas-dna/evals_complex_traits`).
Each HF dataset revision is pinned per-dataset in `config.yaml` via
`hf_revision` so bumping the underlying data triggers re-execution
deterministically. Stripped-down successor to `evals_v1`: one metric, one
split, no plotting, GCS- or HF-stored checkpoints.

## What it does

For each `model` × `dataset` in the config:

1. **Download** the model checkpoint dir (from GCS or HF Hub depending on
   the model entry). The genome reference is read directly from S3 by
   pyfaidx (byte-range reads — no full download).
2. **Score** every variant with `compute_variant_scores`. The score
   bundle is per-strand LLR + JSD (`down_jsd_mean` in issue #175 — the
   per-position 4-nuc next-token JSD averaged over downstream positions).
   `inference.rc=true` (default) computes both strands; the scores
   parquet then carries the raw atoms `llr_fwd`, `llr_rc`, `jsd_fwd`,
   `jsd_rc`. The metrics rule derives `_avg`, `minus_llr_*`, and
   `abs_llr_*` from these — no redundant storage. FWD+RC is the
   validated default per #175 conclusion 2.
3. **Compute** AUPRC ± cluster-bootstrap SE per consequence subset, per
   score column. Cluster bootstrap resamples `match_group`s with
   replacement (preserving the matched 1:k structure) so SE reflects
   the actual sampling unit. The dataset-appropriate LLR protocol comes
   from `score_protocol` in the config (`minus_llr` for mendelian,
   `abs_llr` for complex); each is evaluated on FWD, RC, and AVG, as
   is JSD.

Outputs land in S3 at `s3://oa-bolinas/snakemake/analysis/evals_v2/results/`:

```
results/
├── checkpoints/{model}/                         # cached HF model dir
├── scores/{model}/{dataset}.parquet             # variant cols + per-strand score atoms (+ emb_ref/emb_alt if return_embeddings)
└── metrics/{model}/{dataset}.parquet            # AUPRC ± bootstrap SE per (subset × score_type)
```

The metrics parquet has columns
`[score_type, subset, value, se, n_groups, n_rows, model, dataset, split]`,
with aggregate rows `_global_` and `_macro_avg_` per `score_type` —
see `marin_dna_evals.metrics.compute_auprc_metrics` for details.

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

### Pooled embeddings (`inference.return_embeddings`, #318)

For the frozen-embedding linear-probe VEP protocol (#314 → productionized in
#321/#320), the **same two (FWD, RC) forward passes** that produce LLR/JSD can
*also* emit a pooled both-allele embedding — no second pass, no parallel cache.
Set the global toggle `inference.return_embeddings: true` and the scores parquet
gains two columns:

- `emb_ref`, `emb_alt` — each a length-`D` (model hidden size) `float16` vector
  per variant: the **entire-DNA-window mean-pooled** (the `window_size` DNA
  positions, BOS/special tokens excluded — the #314 `entire_window` extent),
  **FWD+RC-averaged** last-layer hidden state for that allele. Pooling and the
  strand average accumulate in **fp32**; only the stored vector is cast to f16.

The probe feature (`concat_ref_delta` / `sum_absdiff`, #320) is built downstream
from these. Storage is ~`2·D·2` B/variant ≈ **1.5–2.3 GB/model** over the full
suite (~500× smaller than the per-token cache #314 used). The kernel captures the
last layer via a forward hook on `model.base_model` (grabbing `last_hidden_state`,
*not* `output_hidden_states=True` — which would materialize every layer), so the
extra VRAM is just the final `[B, L, D]` hidden states + the wider `[N, 2+2D]`
predictions; still heavier than the slim LLR/JSD bundle, so pair an embedding run
with a smaller per-model `batch_size` (and optionally
`inference.eval_accumulation_steps` to offload the predictions to CPU). Requires
`inference.rc: true` (the stored vector is the FWD+RC average; asserted at config
load).

Run embedding extractions **eager** (`torch_compile: false`): compiling the
hooked forward is unvalidated, and a small run doesn't need it. The ready-made overlay
[`config/overlays/return_embeddings.yaml`](config/overlays/return_embeddings.yaml)
deep-merges `return_embeddings: true` + `batch_size: 32` + `torch_compile: false`
over the config (preserving `rc` etc.), e.g.
`snakemake --configfile config/overlays/return_embeddings.yaml --forcerun
compute_scores -- results/scores/<model>/<dataset>.parquet`. Eager makes the
stored `llr_*` differ from the compiled default by float-reduction noise that
accumulates in the LLR **sum** (JSD, a mean, is unaffected); the difference is
AUPRC-invariant — a deliberate execution tradeoff for embedding runs, not a
correctness issue (the measured parity numbers live in #318 / the PR, not here).

**Operational note.** `return_embeddings` is output-affecting (it lives in the
rule's `params:`). To extract embeddings into a cell whose scores parquet already
exists, **force that specific target**:
`snakemake --configfile config/overlays/return_embeddings.yaml --forcerun
compute_scores -- results/scores/<model>/<dataset>.parquet`. (A bare
`--rerun-triggers mtime` does *not* help here — it drops the `params` trigger that
detects the `return_embeddings` flip, so a cell whose output already exists would
be skipped with no embedding columns.) Name targeted targets rather than
`snakemake all` so unrelated already-scored cells are left untouched.

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

### Soft Mendelian VEP metrics (issue #459)

`soft-vep-analysis` is a CPU-only, inference-free analysis of the 48 existing
exp232 development-split Mendelian score parquets. It reads only `label`,
`subset`, `match_group`, `llr_fwd`, and `llr_rc`, loads one checkpoint step at
a time, and excludes the exp232 cCRE arm, `distal`, and the underpowered
`mature_miRNA_variant` subset.

```bash
cd snakemake/analysis/evals_v2
uv run --locked soft-vep-analysis \
  --output-dir results/soft_vep/exp232 \
  --n-bootstrap 1000 \
  --seed 459
```

The analysis reproduces the stored `minus_llr_avg` AUPRC before computing the
global and group-balanced mean gaps, group-standardized and median/MAD
separation, fixed-temperature soft pairwise win rate, and grouped-CV calibrated
log loss and Brier score. `SoftWin` uses one temperature: the median absolute
within-group pairwise margin from `exp232-v4_bg-step-500`, pooled over the seven
non-distal subsets. That scalar is recorded in `metadata.json` and reused for
every arm, step, and subset.

The same command also runs a separate no-match-group sensitivity analysis. It
computes pooled within-class variant SMD (Cohen's `d`), the mean gap divided by
the SD of all variants, Student's pooled-variance `t`, and Welch's
unequal-variance `t`. Positive and negative variants are sampled separately
with replacement, and the same row multiplicities are applied to all five arms.
AUPRC is recomputed from those exact class-stratified variant-bootstrap draws;
the resulting intervals must not be confused with the primary `match_group`
bootstrap. Student's `t` is a fixed rescaling of pooled SMD when class counts
are fixed. Dividing the gap by the grand-mean standard error
`sd(all variants) / sqrt(n)` is omitted because it is a fixed rescaling of the
all-variant-SD gap within each subset and is not the standard error of a
difference in class means.

Outputs include:

- `point_metrics.parquet`: all 48 arm/checkpoint cells × seven subsets × eight
  metrics, with 95% intervals from joint `match_group` bootstrap draws.
- `pairwise_deltas.parquet`: paired arm differences from the same draws.
- `specialist_detectability.parquet`: for AUPRC and all seven candidate
  metrics at each synchronized checkpoint, the home arm's point rank, oriented
  margin to the best non-home arm, percentile interval on that joint margin,
  and fraction of bootstrap draws where the home arm ranks first.
  `specialist_detection_timing.parquet` records the first of two consecutive
  checkpoints whose home-minus-best-non-home 95% interval is above zero.
  `metric_detection_comparison.parquet` counts earlier, tied, AUPRC-earlier,
  and jointly undetected subsets for each candidate.
- `rank_agreement.parquet`, `rank_reversals.parquet`, and
  `confident_rank_reversals.parquet`: same-step and final-step AUPRC rank
  comparisons, including a view restricted to pairs whose joint-bootstrap
  intervals exclude zero for both metrics.
- `ungrouped_point_metrics.parquet`, `ungrouped_pairwise_deltas.parquet`, and
  `ungrouped_specialist_detectability.parquet`: the no-match-group point,
  interval, and home-versus-best-non-home results under a class-stratified
  variant bootstrap. `ungrouped_specialist_detection_timing.parquet` and
  `ungrouped_metric_detection_comparison.parquet` apply the same two-consecutive
  checkpoint rule against the row-bootstrap AUPRC baseline.
- `ungrouped_rank_agreement.parquet`, `ungrouped_rank_reversals.parquet`, and
  `ungrouped_confident_rank_reversals.parquet`: same-step and final-step AUPRC
  rank comparisons for the four no-group candidates.
- `specialist_wins.parquet`: the earliest synchronized checkpoint where the
  mapped specialist ranks first for two consecutive stored checkpoints.
  `supported_specialist_wins.parquet` applies the same persistence rule only
  when the specialist's paired 95% interval clears every competing arm.
- `auprc_reproduction.parquet`: computed-versus-stored AUPRC parity.
- `controls.parquet` and `control_summary.parquet`: positive score rescaling,
  sign reversal, within-group label permutation, and FWD-only diagnostics at
  step 4999. `fwd_metrics.parquet` contains the FWD-only trajectory at every
  stored checkpoint.
- `plots/*.svg`: one seven-subset, full-cross-arm trajectory figure per metric.
  Ribbons are 95% joint cluster-bootstrap intervals. Proper-score intervals are
  conditional on the fixed out-of-fold calibration fits; bootstrap draws
  resample their held-out row losses without refitting the calibrator.
- `plots/specialist_auprc_vs_brier.{svg,png}`: all five arms for each subset,
  with the mapped home arm highlighted. AUPRC and `1 - calibrated Brier` use
  independent axes so higher is better for both; home-arm ribbons show 95%
  intervals. The PNG is intended for inline GitHub display.
- `plots/specialist_detectability.{svg,png}`: the bootstrap frequency that the
  home arm ranks first under AUPRC and calibrated Brier at every synchronized
  step. `plots/specialist_detection_timing.{svg,png}` compares their earliest
  persistent confidence-supported separation directly.
- `plots/specialist_metric_detectability_summary.{svg,png}`: all eight
  metrics' earliest persistent steps and each candidate's timing counts relative
  to AUPRC.
- `plots/{variant_pooled_smd,variant_total_sd_gap,student_t,welch_t}.svg` and
  `plots/specialist_ungrouped_metric_detectability_summary.{svg,png}`: all-arm
  trajectories and earliest-detection comparison under the explicit no-group
  assumption.
- `distributions/*.svg`: final-step POS/NEG score ECDFs and matched-group
  positive-minus-mean-negative difference ECDFs for every non-distal subset.
  Each arm keeps its raw score scale so tail separation and scale drift remain
  visible.

The command never reads held-out test artifacts, interpolates a missing
checkpoint, or writes an exp326/exp351 soft metric. Distal aggregate trajectories
must be patched into the final issue summary separately because no compatible
per-variant score bundle is currently available. The separate aggregate-only
patch preserves every finite W&B history record, including duplicate resumed-run
log records, and labels the exp232 offline-vs-online exp326/351 protocol
difference:

```bash
uv run --locked --group wandb soft-vep-distal-patch \
  --output-dir results/soft_vep/distal \
  --exp232-results-dir results/soft_vep/exp232
```

It writes `distal_aggregate_trajectories.parquet`, `patched_panel_summary.parquet`,
an SVG with experiment-local comparisons, and metadata with exact point counts
and the W&B client version. The panel summary records the issue's two-evaluation
point-estimate rule and marks the unavailable distal soft/bootstrap cells. It
intentionally emits no distal soft metric or uncertainty interval.

#### exp351-centered replacement assessment

For the issue #459 sensitivity assessment, `soft-vep-augmented-analysis` treats
the exp351-centered enhancer arm from issue #351 as the replacement for exp232's
contaminated cCRE/distal arm. It keeps the five uncontaminated exp232 arms and
adds exp351-centered as the sixth arm. The comparison therefore covers all eight
Mendelian consequence subsets: exp351-centered is the mapped home arm for
`distal` and a non-home competitor for the original seven subsets.

The synchronized checkpoint grid is `500, 1500, 2000, 3000, 3500, 4000, 4500,
4999`. Step 1000 is deliberately absent because exp351-centered has a native
checkpoint but no durable HF export; this assessment does not export or
interpolate it.

After the eight exp351-centered offline Mendelian score bundles have been
produced, run:

```bash
uv run --locked soft-vep-augmented-analysis \
  --output-dir ../../../.agents/artifacts/459-soft-vep/augmented-exp232-exp351 \
  --n-bootstrap 1000 \
  --seed 459
```

This emits the same matched-group and no-group detectability tables as the
original exp232 assessment, with six-arm home-versus-best-non-home margins.
`plots/augmented_distal_metric_trajectories.{svg,png}` shows distal AUPRC, Group
SMD, and variant pooled SMD for all six arms. The two augmented detectability
summary figures add `distal` to the original seven consequence subsets. Only the
development split is read.

The companion current-leaderboard pass selects every `family: marin_dna` model
with Mendelian coverage from `dashboard/models.yaml`, computes the same metric
panel and joint model bootstrap, and cross-fits an isotonic soft-to-AUPRC map by
holding out whole experiment groups:

```bash
uv run --locked soft-vep-leaderboard-analysis \
  --models-yaml ../../../dashboard/models.yaml \
  --output-dir results/soft_vep/leaderboard \
  --n-bootstrap 1000 \
  --seed 459
```

Its `auprc_reproduction.parquet` is a required artifact rather than an implicit
assumption: score-bundle/stored-metric mismatches are retained with
`reproduces=false` and summarized in `metadata.json`. `projection.parquet`
contains every leave-one-experiment-out prediction; `projection_summary.parquet`
reports held-out MAE and rank agreement, so correlated models from one experiment
are never split across fit and evaluation. `confident_rank_reversals.parquet`
restricts model-pair disagreements to comparisons resolved by the joint
bootstrap for both the candidate metric and AUPRC.

### Linear probe (frozen-embedding VEP, #320)

`snakemake probe` trains a **frozen-embedding linear probe** per `(model,
dataset)` — the productionized form of #314's settled protocol — also kept **off
`rule all`**. It consumes the in-bundle pooled embeddings (`emb_ref`/`emb_alt`, the
#318 columns), so the cell's scores parquet **must** have been produced with
`inference.return_embeddings: true` (the rule fails fast otherwise). CPU-only — no
GPU; the probe logic lives in
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
`inner_splits`, `n_jobs`, and `models: [{name, datasets}]` — datasets listed
explicitly since embeddings are per-cell). Build:

```bash
# `--rerun-triggers mtime` is required: the scores parquet was built with the #318
# overlay (return_embeddings: true) but the committed default is false, so the default
# `params` trigger would otherwise rebuild it — dropping the embeddings the rule needs
# (and a rebuild needs a GPU the probe node doesn't have).
snakemake probe --rerun-triggers mtime                              # all configured probe cells
snakemake results/probe/<model>/<dataset>.parquet --rerun-triggers mtime   # one cell

# A cell whose scores parquet predates the embeddings must be re-scored first:
snakemake --configfile config/overlays/return_embeddings.yaml --forcerun \
  compute_scores -- results/scores/<model>/<dataset>.parquet
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
# Same `--rerun-triggers mtime` note as `probe` (its upstream scores parquet was
# built with the embedding overlay, differing from the committed default).
snakemake probe_metrics --rerun-triggers mtime                                    # all configured probe cells
snakemake results/probe_metrics/<model>/<dataset>.parquet --rerun-triggers mtime  # one cell
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

## Configuration (`config/config.yaml`)

| Key | Purpose |
| --- | --- |
| `input_hf_prefix` | HF prefix for `f"{prefix}_{dataset.name}"`. |
| `genome_path` | Canonical GRCh38 FASTA. fsspec URI (e.g. `s3://...`) or local path. The S3 path requires `--group genome-s3` at install time. |
| `split` | `train` (or `test` once held-out eval is unlocked). |
| `datasets` | List of `{name, hf_revision, score_protocol, [eval_protocol]}`. `hf_revision` is the pinned HF dataset commit SHA — bumping it triggers re-execution. `score_protocol` ∈ `{minus_llr, abs_llr}`. Optional `eval_protocol` ∈ `{matched_pair (default), qtl_global, sge}` — `qtl_global` selects the global AUPRC + positives-only `effect_size` correlation path for the unmatched caqtl/dsqtl datasets; `sge` selects the per-accession × consequence-subset AUPRC-on-`label` path for `evals_sge` (see the SGE section above). |
| `models` | List of `{name, window_size, ...}`. Each entry has exactly one of `gcs_path` (full GCS URI incl. `/hf/step-{N}`) or `hf_repo` (HuggingFace Hub repo ID), plus two optional fields: `datasets: [...]` to restrict which `datasets` this checkpoint evaluates on (defaults to all), and `batch_size: N` to override the global `inference.batch_size` for this checkpoint (useful when context size differs from the global default's tuning). |
| `inference.*` | Batch size, workers, `data_transform_on_the_fly`, `torch_compile`; `rc` (also score the reverse-complement strand — doubles inference time); `n_bootstrap` (AUPRC bootstrap iterations per subset × score_type); `bootstrap_seed` (reproducibility seed; bumping triggers metrics re-execution). |
| `nuc_dep` | Optional; nucleotide-dependency maps (#237, off `rule all`). `{combines, ord, batch_size, dpi, models: [...], loci: {...}}`. See `rules/interpretation.smk`. |
| `umap_embeddings` | Optional; embedding UMAP (#246, off `rule all`). `{dataset, layer_index, n_center_bp, random_state, dpi, models: [...]}` — `models` reuse the `models:` registry (each needs `window_size`). Build needs `--group umap` (+ `--group genome-s3`). See `rules/embedding_umap.smk`. |
| `ll_gap` | Optional; functional/non-functional LL gap (#274, off `rule all`). `{split, datasets: [{name, hf_repo, hf_revision}], models: [...]}` — `datasets` are mixed-case `seq` HF datasets (the v5/v1/v15 validation intervals; NOT the variant `datasets:` above); `models` reuse the `models:` registry. See `rules/ll_gap.smk`. |

## Library

Pipeline rules are thin glue around:

- `marin_dna_evals.inference.compute_variant_scores` — model + genome
  → per-strand score atoms (`llr_fwd`, `llr_rc`, `jsd_fwd`, `jsd_rc`).
- `marin_dna_evals.metrics.compute_auprc_metrics` — score columns
  → AUPRC ± cluster-bootstrap SE per subset (cluster = `match_group`).
- `marin_dna_evals.metrics.compute_qtl_metrics` — score columns
  → global AUPRC + positives-only Pearson/Spearman vs `effect_size`
  (the `eval_protocol: qtl_global` path for caqtl/dsqtl).
- `marin_dna_evals.ll_gap.compute_hf_ll_gap` — HF checkpoint +
  mixed-case `seq` dataset → per-sequence functional/non-functional LL atoms
  (`ll_sum_upper`, `ll_sum_lower`, `n_upper`, `n_lower`); `aggregate_ll_gap`
  collapses them to token-weighted `LL_upper` / `LL_lower` / `gap`.

These are tested at `tests/evals/test_metrics.py`,
`tests/evals/test_inference.py`,
`tests/evals/test_ll_gap.py`, and `tests/model/test_scoring.py`.
