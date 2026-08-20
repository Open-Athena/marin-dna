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

### Grouped VEP report (AUPRC + Group SMD, #464)

AUPRC remains the primary endpoint for matched-pair datasets.
The [#459 analysis](https://github.com/Open-Athena/marin-dna/issues/459) selected Group SMD as the secondary endpoint when meaningful matched groups exist.
Both metrics are higher-is-better.

For each match group `g`, the report computes `gap_g = positive_g - mean(negatives_g)`.
Group SMD is `mean(gap_g) / sample_sd(gap_g)` across match groups.
It is invariant to a positive affine transformation of the score.
It is a one-sample standardized distribution of matched-group gaps.
It is not a macro average of per-gene Cohen's d values, which would standardize separate positive and negative score samples within genes before averaging effects.

The input contract requires `label`, `subset`, and `match_group`.
Each match group must contain exactly one positive and at least one negative.
Each match group must belong to one subset.
Missing grouping columns, null grouping values, incompatible groups, and groups that span subsets raise a validation error.
A direct scope with one match group or zero SD across group gaps emits an unavailable Group SMD row with a machine-readable `unavailable_reason`.
The `_macro_avg_` Group SMD row is explicitly unavailable because averaging subset effects would define a different statistic.
The implementation never creates a synthetic global match group.

Build all configured matched-pair cells with:

```bash
snakemake grouped_vep_metrics
```

The named target is off `rule all` so existing `results/metrics/` artifacts keep their current producer and schema.
Each cell writes:

```text
results/grouped_vep_scores/{split}/{model}/{dataset}.parquet
results/grouped_vep_metrics/{split}/{model}/{dataset}.parquet
results/grouped_vep_bootstrap/{split}/{model}/{dataset}.parquet
```

The grouped report uses its own split-scoped score bundle rather than the legacy `results/scores/` artifact.
Changing `config.split` therefore selects a different score, summary, and bootstrap identity instead of reusing or overwriting artifacts from another split.

The summary schema is `[metric, higher_is_better, score_type, subset, value, se, ci_low, ci_high, confidence_level, n_groups, n_rows, available, unavailable_reason, uncertainty_method, n_bootstrap, n_bootstrap_valid, model, dataset, split, bootstrap_seed]`.
Filtering summary rows to `metric == "AUPRC"` preserves the existing `[score_type, subset, value, se, n_groups, n_rows]` output from `compute_auprc_metrics`.
Direct subset and `_global_` AUPRC rows retain their existing match-group bootstrap SE.
The `_macro_avg_` AUPRC row retains the existing independent-subset SE-of-mean and reports `n_bootstrap_valid = 0` because it has no stored macro draws.
All AUPRC interval values are null in this report.
Group SMD uses a joint match-group bootstrap and reports the 2.5th and 97.5th percentiles as a 95% interval.

The bootstrap schema is `[draw, metric, score_type, subset, value, n_groups, model, dataset, split, bootstrap_seed]`.
One match-group multiplicity vector is reused across both metrics and every score column within a scope.
Outputs from different models are aligned by `draw` when they use the same dataset revision, row order, `n_bootstrap`, and integer `bootstrap_seed`.
The bootstrap artifact omits `_macro_avg_` for both metrics because Group SMD has no macro definition and AUPRC uses the composed legacy macro SE without stored draws.

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
| `inference.*` | Batch size, workers, `data_transform_on_the_fly`, `torch_compile`; `rc` (also score the reverse-complement strand — doubles inference time); `n_bootstrap` (metric bootstrap iterations per subset × score_type); `bootstrap_seed` (reproducibility and cross-model draw-alignment seed; bumping it triggers metric re-execution). |
| `nuc_dep` | Optional; nucleotide-dependency maps (#237, off `rule all`). `{combines, ord, batch_size, dpi, models: [...], loci: {...}}`. See `rules/interpretation.smk`. |
| `umap_embeddings` | Optional; embedding UMAP (#246, off `rule all`). `{dataset, layer_index, n_center_bp, random_state, dpi, models: [...]}` — `models` reuse the `models:` registry (each needs `window_size`). Build needs `--group umap` (+ `--group genome-s3`). See `rules/embedding_umap.smk`. |
| `ll_gap` | Optional; functional/non-functional LL gap (#274, off `rule all`). `{split, datasets: [{name, hf_repo, hf_revision}], models: [...]}` — `datasets` are mixed-case `seq` HF datasets (the v5/v1/v15 validation intervals; NOT the variant `datasets:` above); `models` reuse the `models:` registry. See `rules/ll_gap.smk`. |

## Library

Pipeline rules are thin glue around:

- `marin_dna_evals.inference.compute_variant_scores` — model + genome
  → per-strand score atoms (`llr_fwd`, `llr_rc`, `jsd_fwd`, `jsd_rc`).
- `marin_dna_evals.metrics.compute_auprc_metrics` — score columns
  → AUPRC ± cluster-bootstrap SE per subset (cluster = `match_group`).
- `marin_dna_evals.grouped_vep_metrics.compute_grouped_vep_metrics` — unchanged AUPRC rows plus Group SMD, explicit unavailable states, 95% intervals, and aligned match-group bootstrap draws.
- `marin_dna_evals.grouped_vep_metrics.group_smd` — mean matched-group gap divided by the sample SD of matched-group gaps.
- `marin_dna_evals.metrics.compute_qtl_metrics` — score columns
  → global AUPRC + positives-only Pearson/Spearman vs `effect_size`
  (the `eval_protocol: qtl_global` path for caqtl/dsqtl).
- `marin_dna_evals.ll_gap.compute_hf_ll_gap` — HF checkpoint +
  mixed-case `seq` dataset → per-sequence functional/non-functional LL atoms
  (`ll_sum_upper`, `ll_sum_lower`, `n_upper`, `n_lower`); `aggregate_ll_gap`
  collapses them to token-weighted `LL_upper` / `LL_lower` / `gap`.

These are tested at `tests/evals/test_grouped_vep_metrics.py`,
`tests/evals/test_metrics.py`,
`tests/evals/test_inference.py`,
`tests/evals/test_ll_gap.py`, and `tests/model/test_scoring.py`.
