---
topic: issue-485-m5-1-inference-context
issue: https://github.com/Open-Athena/marin-dna/issues/485
description: MarinDNA m5.1 Mendelian VEP across inference context sizes
author: gonzalobenegas
---

# m5.1 Inference Context: Research Logbook

## Current TL;DR

- Status: the six-context zero-shot and trained frozen-probe experiment is complete, all artifacts are validated, and no issue #485 cluster remains live.
- Holding the m5.1 checkpoint fixed, macro AUPRC rises from 0.1941 to 0.3945 zero-shot and from 0.3174 to 0.4779 with the probe as inference context increases from 31 to the 255 bp training context.
- Extending inference to 511 bp changes macro AUPRC by +0.0175 zero-shot (paired 95% CI [+0.0057, +0.0278], p=0.0024) and -0.0124 with a newly trained probe (paired 95% CI [-0.0408, +0.0087], p=0.3138).
- Extending again to 1023 bp reduces macro AUPRC by 0.3060 zero-shot and 0.2993 with a newly trained probe relative to 511 bp; both paired 95% intervals exclude zero with p=0.0001 at the 10,000-draw resolution.
- The 511 and 1023 bp arms are inference-only extrapolations of a checkpoint trained at 255 bp, so they do not estimate the value of training a model at those contexts.
- The reviewed figures use the blog's 3×3 consequence order, independent y-axes, uncapped SE bars, compact spacing, a three-regime legend, and exclude mature miRNA.

## Scope

- Goal: measure how one fixed m5.1 checkpoint's Mendelian VEP changes when its inference window is cropped below its native 255 bp context.
- Primary metrics: standard matched-pair `minus_llr_avg` AUPRC with `match_group` cluster-bootstrap SE; frozen-embedding `probe_score` per-chromosome-weighted AUPRC with chromosome-cluster-bootstrap SE.
- Primary strata: every consequence subset emitted by the pinned dataset and `_macro_avg_`.
- Constraints: development `train` split only; do not access held-out labels, predictions, or aggregate metrics; 0-based half-open coordinates internally; odd centered windows; one checkpoint and unchanged scoring/probe protocols; no public-leaderboard registration for context aliases; no paid remote compute without explicit approval.
- Coordinating issue: https://github.com/Open-Athena/marin-dna/issues/485
- Branch: `codex/issue-485-context-size`
- Experiment prefix: `CTX-VEP`
- Shared tags: `CTX-VEP`, `issue-485`, `evals-v2`

## Baseline

- Date: 2026-08-20
- Code ref: `d40a56acd2011f96fe4f60e87c58098bd04914b5`
- Model: `mix-v0.9-p1B-i24-exp135-m5.1-step-59158`, 255 bp DNA plus BOS.
- Checkpoint: `gs://marin-us-east5/checkpoints/dna-bolinas-mix-v0.9-p1B-i24-exp135-zoonomia-m5.1-bef41e/hf/step-59158`.
- Dataset: `bolinas-dna/evals_mendelian_traits` at `4aed58e50c5dea0b878a665007af2ef9e5108e9f`, split `train`.
- Baseline numbers: not yet audited for this experiment. The existing 255 bp score, metric, and embedding artifacts may be reused only if their schema and provenance match issue #485.

## Hypothesis Queue

### Active

- `CTX-VEP-001`: zero-shot Mendelian VEP has a measurable context-response curve for at least one consequence subset or the macro average. Falsifier: all four point estimates are descriptively stable relative to their protocol-appropriate SEs.
- `CTX-VEP-002`: frozen-probe context-response curves differ by consequence subset and may differ qualitatively from the zero-shot curves. Falsifier: every eligible probe panel is descriptively stable or follows the same context ordering as its zero-shot panel.

### Blocked

- None.

### Falsified / Dead End

- None.

### Promoted

- None.

## Decision Log

- 2026-08-20: hold the checkpoint fixed and compare 255, 127, 63, and 31 bp odd centered inference windows.
- 2026-08-20: use only the Mendelian development split for scoring, probing, model selection, and interpretation.
- 2026-08-20: run the existing zero-shot and frozen-embedding probe protocols unchanged and report them in separate figures.
- 2026-08-20: treat arm-wise SE comparisons as descriptive; formal deltas would require paired `match_group` or chromosome bootstraps.
- 2026-08-20: do not register context aliases on the public leaderboard because they are inference configurations of one trained model.

## Background Research Brief

- Effort: low.
- Stop rule: stop after the issue, the linked Marin experiments and question page, the current `evals_v2` implementation, and the official m5.1 blog post agree on the experiment gap and no repository search reveals an existing matched inference-time crop ablation.
- Date: 2026-08-20.

### Question

How sensitive are m5.1 Mendelian zero-shot likelihood scoring and frozen-embedding probing to shortening only the inference window?

### Current Marin Context

- Issue #37 compared models pretrained with 256 and 512 bp contexts and found no clear VEP difference, but changed the training run and context together.
- Issue #314 settled the frozen-probe recipe at last-layer entire-window mean pooling, FWD/RC averaging, `concat_ref_delta`, and chromosome-grouped logistic regression.
- The current `evals_v2` code already accepts arbitrary per-model odd `window_size` values, emits embeddings in the scoring pass, and registers probe cells explicitly.
- The official m5.1 blog explains that 255 bp was chosen to model individual functional elements efficiently and leaves context extension as future work.

### Negative / Failed Leads

- No matched m5.1 inference-time context ablation was found in the current research question, `evals_v2` documentation, or repository searches for inference context, context size, and context length.
- Issue #37 is not a substitute because it changes pretraining context and model run.

### Evidence Map

#### Claim: issue #485 fills a distinct inference-time context gap

- Support:
  - Issue #37: only a 256-versus-512 bp pretraining comparison, with no clear VEP difference.
  - `evals_v2/config/config.yaml` and `src/marin_dna_evals/transforms.py`: per-model windows and odd centered cropping are already supported.
  - Issue #314 and the current probe pipeline: the supervised readout protocol is fixed and productionized.
- Contradictions: none found; existing evidence concerns training context, longer-context acquisition, or the native 255 bp probe.
- Directness to Marin: high.
- Confidence: high that the experiment is non-duplicative; no directional performance prediction is justified.
- Action: run the four-arm fixed-checkpoint experiment in issue #485.

### Recommended Next Experiments

#### 1. `CTX-VEP-001/002`: fixed-checkpoint inference crop ladder

- Minimum experiment: score and probe m5.1 at 255, 127, 63, and 31 bp on the pinned Mendelian development split.
- Baseline/control: native 255 bp model ID and unchanged production protocols.
- Expected signal: descriptive per-subset and macro context-response curves; no directional prediction.
- Falsifier: no descriptive movement beyond arm-wise uncertainty in either protocol.
- Cost/risk: four scoring passes plus four CPU probe fits; duplicate checkpoint caching under alias IDs and paid remote compute require review.
- Sources: issue #485, issue #37, issue #314, `evals_v2`, and the official MarinDNA m5.1 blog post.

### Hypothesis Queue Update

- Add: `CTX-VEP-001` and `CTX-VEP-002`.
- Revise: none.
- Falsify / stop: none.
- Promote: none.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
| --- | --- | --- | --- | --- | --- |
| Issue #485 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/485 | Fixed-checkpoint design and completion criteria | High | Coordinating issue |
| Issue #37 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/37 | Prior 256-versus-512 bp pretraining comparison | High | Training and inference context are confounded |
| Issue #314 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/314 | Settled frozen-probe protocol | High | Productionized in `evals_v2` |
| Context question | Knowledge base | `docs/research/questions/long-context.md` | No direct matched inference-crop result recorded | High | Current scope is long-context acquisition |
| `evals_v2` | Marin code | `snakemake/analysis/evals_v2/` at `d40a56ac` | Window, score, embedding, and probe support | High | Development split and dataset revision are pinned |
| MarinDNA m5.1 blog | Official blog | https://www.openathena.ai/blog/marin-dna/ | Motivation for native 255 bp context | High | Published 2026-08-03 |

### Handoff

- Suggested issue `Prior work` block: issue #37 changed pretraining context and model run; issue #314 fixed the probe protocol; no matched m5.1 inference-time crop ablation was found.
- Suggested logbook entry: prologue below.
- Open questions: whether the native 255 bp embedding artifact has matching schema and provenance; whether alias-keyed checkpoint caching creates material duplicate transfer or storage cost.
- Stop reason: additional searching no longer changed the planned four-arm experiment.

## Negative Results Index

- No matched m5.1 inference-time context ablation was found in the searched Marin artifacts.

## Entry Log

### 2026-08-20 17:00 UTC - CTX-VEP-001 prologue and prior-work audit

- Hypothesis: the existing `evals_v2` window, score, embedding, and probe paths can execute the four-arm experiment without changing maintained inference or probe behavior.
- Commit Hash: `d40a56acd2011f96fe4f60e87c58098bd04914b5` (starting point).
- Command: GitHub reads of issues #485, #37, and #314; targeted repository `rg` for context-size and frozen-probe work; read of the `evals_v2` README, manifest, profile, configuration, and rules; read of the official m5.1 blog post.
- Config: 255/127/63/31 bp; m5.1 step 59158; Mendelian revision `4aed58e50c5dea0b878a665007af2ef9e5108e9f`; split `train`; zero-shot `minus_llr_avg`; probe `concat_ref_delta`.
- Result: the issue fills a distinct inference-time gap. The existing pipeline supports the planned windows and production metrics; three alias model entries and three additional probe registrations are required. No scoring or paid compute has run.
- Interpretation: proceed with additive configuration changes and targeted DAG validation. Preserve the native 255 bp artifact namespace and keep aliases off the public leaderboard.
- Next action: register the context aliases, validate their exact configuration, run the owning project tests, and inspect the targeted four-arm dry-run before requesting paid-compute approval.

### 2026-08-20 17:10 UTC - CTX-VEP-001 registry and DAG gate

- Hypothesis: additive model aliases can isolate the four context artifacts while reusing the unchanged checkpoint, scoring kernel, and probe protocol.
- Commit Hash: working tree on `codex/issue-485-context-size` atop `d40a56acd2011f96fe4f60e87c58098bd04914b5`.
- Command: focused registry test:

```bash
uv run --locked pytest tests/evals/test_context_ablation_485.py
```

- Command: targeted four-arm DAG audit:

```bash
uv run --locked --group genome-s3 snakemake -n \
  --configfile config/overlays/return_embeddings.yaml \
  --forcerun compute_scores \
  --rerun-triggers mtime -- \
  results/metrics/mix-v0.9-p1B-i24-exp135-m5.1-step-59158/mendelian_traits.parquet \
  results/probe/mix-v0.9-p1B-i24-exp135-m5.1-step-59158/mendelian_traits.parquet \
  results/probe_metrics/mix-v0.9-p1B-i24-exp135-m5.1-step-59158/mendelian_traits.parquet \
  results/metrics/mix-v0.9-p1B-i24-exp135-m5.1-step-59158-ctx127/mendelian_traits.parquet \
  results/probe/mix-v0.9-p1B-i24-exp135-m5.1-step-59158-ctx127/mendelian_traits.parquet \
  results/probe_metrics/mix-v0.9-p1B-i24-exp135-m5.1-step-59158-ctx127/mendelian_traits.parquet \
  results/metrics/mix-v0.9-p1B-i24-exp135-m5.1-step-59158-ctx63/mendelian_traits.parquet \
  results/probe/mix-v0.9-p1B-i24-exp135-m5.1-step-59158-ctx63/mendelian_traits.parquet \
  results/probe_metrics/mix-v0.9-p1B-i24-exp135-m5.1-step-59158-ctx63/mendelian_traits.parquet \
  results/metrics/mix-v0.9-p1B-i24-exp135-m5.1-step-59158-ctx31/mendelian_traits.parquet \
  results/probe/mix-v0.9-p1B-i24-exp135-m5.1-step-59158-ctx31/mendelian_traits.parquet \
  results/probe_metrics/mix-v0.9-p1B-i24-exp135-m5.1-step-59158-ctx31/mendelian_traits.parquet
```

- Config: embedding overlay enabled; eager inference; batch size 96; all four score rules forced; only `mendelian_traits`; downstream rerun triggers restricted to `mtime`.
- Result: the focused test passed 2/2.
- Result: the DAG contains exactly 3 alias `download_model`, 4 `compute_scores`, 4 `compute_metrics`, 4 `compute_probe`, and 4 `compute_probe_metrics` jobs, for 19 jobs total.
- Result: the native checkpoint cache exists; all three alias caches are missing, confirming the anticipated duplicate checkpoint materialization.
- Shared-node record: the focused test ran approximately 17:07:18–17:07:20 UTC with status 0, 1.31 s elapsed, and 59,108 kB peak RSS.
- Shared-node record: the dry-run ran 17:08:02–17:08:18 UTC with status 0, 15.92 s elapsed, and an unexpected 1,090,132 kB peak RSS.
- Safety decision: do not repeat the full DAG load or run the full evals_v2 suite on this shared node; use focused local checks and remote compute or CI for the full gate.
- Cost estimate: SkyPilot reports `g5.xlarge` A10G pricing in `aws/us-east-2` at $0.365/hour Spot and $1.006/hour on demand.
- Interpretation: the registry and DAG satisfy the setup criteria with no unintended model, dataset, or rule family.
- Next action: after explicit paid-compute approval, run the four GPU score-plus-zero-shot-metric cells, then the four CPU probe-plus-probe-metric cells, with a $5 spend cap and active first-minutes monitoring.

### 2026-08-20 17:33 UTC - CTX-VEP-001 remote runtime gate and 31 bp result

- Hypothesis: a fresh Sky A10G node can reproduce the locked `evals_v2` environment and complete one development-only scoring arm before the remaining paid cells are released.
- Commit Hash: working tree on `codex/issue-485-context-size` atop `d40a56acd2011f96fe4f60e87c58098bd04914b5`.
- Command: initial 31 bp pilot launch:

```bash
sky launch -c evals-v2-485-ctx31 --image-id ami-0324f0ad73bdcd087 \
  --env "SNAKEMAKE_ARGS=--configfile config/overlays/return_embeddings.yaml --forcerun compute_scores --rerun-triggers mtime -- results/metrics/mix-v0.9-p1B-i24-exp135-m5.1-step-59158-ctx31/mendelian_traits.parquet" \
  --idle-minutes-to-autostop=1 --down -y sky/run.yaml
```

- Negative result: the initial node failed in setup because the fresh image supplied `uv 0.12.5` while `uv.lock` requires exact `uv 0.11.31`; no scoring ran and `sky down -y evals-v2-485-ctx31` terminated the node immediately.
- General fix: pin the validated AWS DLAMI `ami-0324f0ad73bdcd087` in `sky/run.yaml` and downgrade or upgrade fresh-node `uv` to `0.11.31` before `uv sync --locked`.
- Command: corrected 31 bp pilot launch:

```bash
sky launch -c evals-v2-485-ctx31 \
  --env "SNAKEMAKE_ARGS=--configfile config/overlays/return_embeddings.yaml --forcerun compute_scores --rerun-triggers mtime -- results/metrics/mix-v0.9-p1B-i24-exp135-m5.1-step-59158-ctx31/mendelian_traits.parquet" \
  --idle-minutes-to-autostop=1 --down -y sky/run.yaml
```

- Result: the corrected pilot resolved exactly one `download_model`, one `compute_scores`, and one `compute_metrics` job; it processed 16,140 `mendelian_traits` development rows and uploaded the embedding-bearing score table plus 66 matched-pair metric rows.
- Result: both reference and alternate prediction passes completed in approximately 49 seconds each; a late CUDA allocator warning was non-fatal and the job exited with status 0.
- Interpretation: the pinned image and `uv` bootstrap repair the fresh-node gate, and the intended development-only score/metric contract is working at 31 bp.
- Next action: complete and validate the 63, 127, and 255 bp GPU arms, then run the four frozen probes on CPU.

### 2026-08-20 17:48 UTC - CTX-VEP-001 four-arm GPU completion

- Hypothesis: the fixed checkpoint can score the same 16,140 Mendelian development variants at 63, 127, and native 255 bp with the same embedding-bearing protocol that passed at 31 bp.
- Commit Hash: `d40a56acd2011f96fe4f60e87c58098bd04914b5` plus the issue registry and fresh-node bootstrap changes in the synced working tree.
- Command: one named `compute_scores` plus downstream `compute_metrics` target per context on an A10G `g5.xlarge`, using `config/overlays/return_embeddings.yaml`, `--forcerun compute_scores`, and `--rerun-triggers mtime`.
- Result: all four arms exited with status 0 and uploaded 16,140-row score tables plus 66-row zero-shot metric tables.
- Result: score and metric uploads completed at 17:30/17:32 UTC for 31 bp, 17:38/17:39 for 63 bp, 17:40/17:42 for 127 bp, and 17:45/17:47 for 255 bp.
- Result: score Parquets are 125,646,546 to 125,653,158 bytes and metric Parquets are 7,312 to 7,355 bytes.
- Result: reference and alternate passes took approximately 49 seconds each at 31 bp, 91 seconds at 63 bp, 174 seconds at 127 bp, and 342 seconds at 255 bp.
- Diagnostic: each arm emitted a late non-fatal CUDA allocator warning after useful work had completed; every Snakemake job continued to successful upload.
- Interpretation: the zero-shot cells are complete and directly comparable because checkpoint, dataset revision, split, row set, scoring method, and downstream metric code are fixed while only inference context changes.
- Next action: fit the four independent production frozen probes and compute their probe metrics.

### 2026-08-20 19:42 UTC - CTX-VEP-002 frozen-probe completion and reusable bootstrap fix

- Hypothesis: the locked `concat_ref_delta` nested chromosome-held-out probe can consume every context's finite embeddings and preserve identical eligibility and sample contracts.
- Commit Hash: remote job snapshot `d40a56ac83ac414bc5c31625bc3996007edbd407` plus the issue registry and exact-`uv` probe bootstrap change in the synced working tree.
- Negative result: the first `c6i.2xlarge` Spot node successfully downgraded fresh-image `uv 0.12.5` to project-required `0.11.31`, completed the locked sync, staged two score files, and was then preempted before producing a probe artifact.
- Recovery: relaunch the same four explicit `probe_metrics` targets on an on-demand `c6i.2xlarge` with `--no-use-spot`; do not change the scientific protocol.
- Result: all four `compute_probe` and four `compute_probe_metrics` jobs exited with status 0.
- Result: every probe processed 16,140 variants, fitted 8 eligible consequence classifiers, scored 16,100 rows, and left only the 40 mature-miRNA rows unscored under `min_variants=300`.
- Result: every probe-metric artifact contains 20 rows, including `probe_score` and the paired `minus_llr_avg` diagnostic; only `probe_score` is used in the figure.
- Diagnostic: the 63 bp 5′ UTR classifier has `truncation_risk=true`, with median and full-data `C=1e-8` and a `+0.016005` high-edge AUPRC gain; preserve the production-grid result but flag this point as diagnostic rather than silently widening the frozen protocol.
- Diagnostic: all other recorded C-grid edges are flat under the production tolerance and have `truncation_risk=false`.
- General fix: `sky/probe.yaml` also needs the exact project `uv` pin before `uv sync --locked`; a separate `codex/issue-485-pin-probe-uv` worktree contains only that change and a regression covering every locked Sky task.
- Test: the new Sky bootstrap test passes alone at approximately 53 MB peak RSS and the fresh remote node demonstrated the downgrade and successful locked sync.
- Shared-node exception: an earlier adjacent 10-test check passed 10/10 but peaked at 1,051,680 kB RSS, above the 500 MiB local-work ceiling; do not repeat that suite locally and leave the torch-heavy runtime gate to CI or remote compute.
- Interpretation: all planned probe cells are complete with consistent sample and eligibility contracts; the single 63 bp 5′ UTR grid diagnostic limits that subset point but does not invalidate the remaining production-protocol measurements.
- Next action: validate embedding widths and artifact inventory, then build and inspect the two context-response figures.

### 2026-08-20 19:47 UTC - Context-response analysis, artifact gate, and disposition

- Branch base: at the user's direction, rebase `codex/issue-485-context-size` onto `origin/main` at `32470ee6dd4ddb500dfc426a912694b2cdeff83c`; retain `d40a56ac83ac414bc5c31625bc3996007edbd407` as the remote artifact-producing code provenance.
- Validation: all four score Parquets have 16,140 rows and observed `emb_ref`/`emb_alt` widths of 1,920; successful probe feature construction also asserts equal two-dimensional shapes and finiteness across every embedding row.
- Shared-node record: the lock-protected, ranged embedding-width read ran 18:52:04–18:52:11 UTC with status 0 and 460,448 kB peak RSS.
- Validation: all 20 expected S3 outputs exist with fresh timestamps: four scores, four zero-shot metrics, four probe predictions, four classifier joblibs, and four probe metrics.
- Validation: post-rebase focused registry and Sky bootstrap tests passed 3/3 in 0.16 seconds at 53,244 kB peak RSS.
- Plotting: apply the rebased `plot-research-results` guidance, render square 2×5 facets with shared log2 x and 0–1 y scales, and inspect PNG previews before retaining SVG output.
- Plotting: the final reviewed SVGs are `.agents/artifacts/issue-485-context-size/zero_shot_context.svg` and `.agents/artifacts/issue-485-context-size/probe_context.svg`; compact plotted data are `zero_shot_source.csv` and `probe_source.csv`.
- Plotting: `probe_source.csv` includes the saved regularization-grid diagnostics, and the frozen-probe figure explicitly labels the 63 bp 5′ UTR truncation risk.
- Shared-node record: the final lock-protected render exited with status 0 in 3.36 seconds at 263,020 kB peak RSS.

| Protocol | 31 bp | 63 bp | 127 bp | 255 bp |
| --- | ---: | ---: | ---: | ---: |
| Zero-shot macro AUPRC ± SE | 0.1941 ± 0.0102 | 0.3064 ± 0.0143 | 0.3658 ± 0.0149 | 0.3945 ± 0.0155 |
| Probe macro AUPRC ± SE | 0.3174 ± 0.0269 | 0.3354 ± 0.0266 | 0.4130 ± 0.0255 | 0.4779 ± 0.0216 |

- Result: relative to native 255 bp, zero-shot macro AUPRC changes by -0.2004 at 31 bp, -0.0881 at 63 bp, and -0.0287 at 127 bp.
- Result: relative to native 255 bp, probe macro AUPRC changes by -0.1604 at 31 bp, -0.1424 at 63 bp, and -0.0648 at 127 bp.
- Result: the macro curve improves with context in both protocols, while individual subsets are heterogeneous and sometimes peak at 127 bp or shorter.
- Interpretation: broader inference context is descriptively associated with better macro Mendelian VEP for this fixed m5.1 checkpoint under both zero-shot and frozen-probe protocols.
- Boundary: these are arm-wise ±1 SE summaries, not formal pairwise tests, and the 63 bp 5′ UTR probe point carries the production C-grid truncation warning.
- Cost: issue-specific SkyPilot estimates total $1.60, comprising $0.96 for the four successful GPU arms, $0.06 for the failed GPU bootstrap pilot, $0.01 for the preempted Spot probe attempt, and $0.63 for the successful on-demand probe job; all issue clusters are terminated.
- Knowledge-base disposition: `pending` until a human reviews and accepts the interpretation; do not promote an experiment page or revise a question page yet.
- Next action: publish the permanent experiment branch and issue summary after Git staging, commit, and push authorization; publish the separate general probe-bootstrap fix as a draft pull request through its own branch.

### 2026-08-20 20:08 UTC - User-directed blog-layout figure revision

- Direction: do not use mature miRNA in the presentation, follow the 3×3 subplot layout and row-major order of the official MarinDNA blog figures, and do not share y-axes.
- Revision: both source tables and figures now contain exactly nine panels ordered Macro Avg, Missense, Splicing, Synonymous, Promoter, 5′ UTR, 3′ UTR, Distal, and ncRNA.
- Revision: mature miRNA is filtered before the plotted CSVs are written and is absent from both SVGs; macro values are unchanged because the low-count mature-miRNA subset was already excluded from aggregation.
- Revision: every panel retains the shared log2 context x-axis but uses an independent y-range derived from its values and ±1 SE interval.
- Diagnostic: the 63 bp 5′ UTR probe C-grid warning remains attached to that point.
- Validation: each plotted source table has 36 rows, four per requested panel, and a case-insensitive search finds no mature-miRNA content in either source table or SVG.
- Shared-node record: the final lock-protected render ran 20:06:40–20:06:42 UTC with status 0 and 257,076 kB peak RSS.
- Review: visually inspect both final previews at resized detail; subplot order, independent y ticks, titles, legend, uncertainty bars, and outer labels are legible without overlap.

### 2026-08-20 20:23 UTC - Final figure spacing and uncertainty styling

- Direction: remove the legend, reduce horizontal and vertical subplot spacing, and render SE bars without end caps while retaining uppercase `SE` in the axis labels.
- Revision: shorten the canvas from 8.1×9.4 to 8.1×8.5 inches, move the grid top to 0.90, and reduce `wspace` from 0.35 to 0.22 and `hspace` from 0.42 to 0.25.
- Validation: inspect full zero-shot output and cropped probe output; all nine titles, independent y ticks, uncapped uncertainty bars, diagnostic text, and outer labels remain legible without overlap.
- Shared-node record: the final lock-protected render ran 20:15:42–20:15:45 UTC with status 0 and 256,256 kB peak RSS.

### 2026-08-20 20:23 UTC - CTX-VEP-003 proposed 511 bp zero-shot extrapolation

- Background-research effort: low; stop after the checkpoint configuration, locked Qwen3 implementation, existing issue results, and eval-pipeline contracts made the minimum experiment unambiguous.
- Stop rule: proceed only if 511 bp is executable without changing model weights, position code, held-out data, or the established zero-shot metric.
- Question: does the fixed m5.1 checkpoint benefit from 511 DNA bases plus BOS at inference time despite training at 255 DNA bases plus BOS?
- Current Marin context: zero-shot macro AUPRC rises from 0.1941 at 31 bp to 0.3945 at native 255 bp on the pinned Mendelian `train` split.
- Implementation evidence: the checkpoint declares `max_position_embeddings=256`, uses Llama-3-style rotary positions, and the locked Qwen3 forward computes rotary sine/cosine values directly from supplied position IDs without a learned lookup or hard length check.
- Negative result: no existing 511 bp m5.1 artifact or prior result was found in the issue registry; the new arm cannot be treated as an interpolation or as evidence about a model trained at 511 bp.
- Hypothesis: a 511 bp forward is executable and may improve zero-shot Mendelian VEP if useful longer-range sequence signal outweighs out-of-distribution positional behavior.
- Minimum experiment: one zero-shot-only 511 bp score and matched-pair metric cell; do not emit embeddings and do not fit a supervised probe.
- Baseline: compare with the already complete native 255 bp zero-shot cell under the same checkpoint, dataset revision, split, FWD+RC protocol, metric code, and bootstrap seed.
- Primary metric: macro matched-pair AUPRC with the existing match-group bootstrap SE; preserve all eligible subset rows for heterogeneity checks.
- Falsifier: a model/runtime length failure, non-finite scores, a schema or row-contract mismatch, or a clear macro degradation relative to 255 bp.
- Boundary: label the result exploratory and out of distribution because the model saw only 256-token sequences during training.
- Config: add experiment-only alias `mix-v0.9-p1B-i24-exp135-m5.1-step-59158-ctx511` with `window_size=511`, `batch_size=24`, and `datasets=[mendelian_traits]`; deliberately omit it from `probe.models`.
- Test: the updated registry contract passes 2/2 at 53,200 kB peak RSS and asserts that the 511 bp alias is excluded from probe fitting.
- Planned target: `results/metrics/mix-v0.9-p1B-i24-exp135-m5.1-step-59158-ctx511/mendelian_traits.parquet` after a remote dry-run of that exact target.
- Cost/risk: full attention at twice the token length may approach four times the attention work; batch size 24 is conservative and the pilot remains within the previously approved $5 session budget.
- Source ledger: pinned checkpoint `config.json` on GCS; locked `transformers` Qwen3 rotary and model forward source; `evals_v2` README, registry, inference rule, and issue #485 logbook.
- Next action: remote dry-run the exact metric target, inspect the planned DAG, then launch and monitor one GPU cell if the plan contains only checkpoint download, scoring, and metric computation.

### 2026-08-20 21:03 UTC - CTX-VEP-003 511 bp zero-shot result

- Approval: run the inference-only 511 bp arm under a separate $1.50 cap, without retraining or supervised probe fitting.
- Compute: SkyPilot could not provision Spot capacity in the requested AWS region and fell back to one on-demand `g5.xlarge` with an A10G at $1.01/hour.
- Dry-run gate: the exact metrics target expanded to only `download_model`, `compute_scores`, and `compute_metrics`, with one job each.
- Runtime validation: the fresh image downgraded `uv 0.12.5` to the locked project version `0.11.31`, and the GPU smoke test passed with bf16 support on an NVIDIA A10G.
- Execution: both reference and alternate 673-batch passes completed at approximately 1.32 batches/second without a context-length rejection, OOM, retry, or non-finite-score failure.
- Score contract: `s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores/mix-v0.9-p1B-i24-exp135-m5.1-step-59158-ctx511/mendelian_traits.parquet` contains exactly 16,140 `train` rows and no embedding columns.
- Metric contract: `s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/mix-v0.9-p1B-i24-exp135-m5.1-step-59158-ctx511/mendelian_traits.parquet` contains 66 matched-pair metric rows across the existing six score types.
- Primary result: macro matched-pair AUPRC increased from 0.394530 ± 0.015540 at native 255 bp to 0.411633 ± 0.015220 at 511 bp, an absolute change of +0.017103 and a descriptive relative increase of 4.34%.
- Heterogeneity: six of the eight macro-eligible subsets improved at 511 bp, while missense changed by -0.005932 and splicing by -0.026317.
- Largest descriptive gains: synonymous changed by +0.085445 and ncRNA by +0.040265; the remaining positive subset changes ranged from +0.006713 to +0.017698.
- Boundary: the arm is an inference-only out-of-distribution extrapolation because the fixed checkpoint trained on 255 DNA bases plus BOS; overlapping marginal SE bars are not a paired test of the 255-to-511 difference.
- Interpretation: the executable 511 bp context produces a small descriptive macro gain rather than the preregistered clear-degradation falsifier, but the heterogeneous subset response and absent paired uncertainty do not establish a general benefit.
- Presentation: retain the trained-range 31–255 bp figures unchanged and place the 255-versus-511 result in the separate `zero_shot_extrapolation_context.svg` figure and `zero_shot_extrapolation_source.csv` table.
- Presentation validation: the supplementary figure uses the requested 3×3 blog order, independent y-axes, no legend, reduced spacing, uncapped uppercase-SE bars, and excludes mature miRNA from both the plotted source and SVG.
- Shared-node record: the lock-protected supplementary render ran 21:00:05–21:00:06 UTC with status 0 and 164,764 kB peak RSS.
- Cost: SkyPilot reports 29m25s and an estimated $0.49, below the separate $1.50 cap; the cluster was explicitly terminated and is absent from `sky status`.
- General fix disposition: the fresh-node downgrade again validates the need for the separate `codex/issue-485-pin-probe-uv` bootstrap-fix branch; remove its duplicate edit and test from this experiment branch.
- Knowledge-base disposition: remain `pending` until human review accepts the interpretation.
- Next action: run final lightweight registry checks and publish only after explicit staging, commit, push, and GitHub-write authorization.

### 2026-08-20 22:08 UTC - CTX-VEP-004 combined 511 bp zero-shot and trained-probe extension

- Direction: include 511 bp in the same zero-shot and frozen-probe figures as the 31–255 bp context reduction, restore a legend distinguishing context reduction, the 255 bp training context, and the 511 bp inference extension, and train the downstream probe.
- Boundary: the language model remains the fixed 255 bp-trained m5.1 checkpoint; only its inference window is extended to 511 DNA bases plus BOS, while the frozen linear probe is fitted anew on development labels under the unchanged nested chromosome-held-out protocol.
- Registry: add the 511 bp alias to `probe.models` and preserve the same `concat_ref_delta` features, C grid, folds, seed, dataset revision, and Mendelian `train` split used by the shorter contexts.
- GPU dry-run: forcing the embedding-bearing 511 bp score target expanded to exactly one `compute_scores` job because the checkpoint was already present and downstream targets were not requested.
- GPU result: both 673-batch reference and alternate passes completed and uploaded a 16,140-row score artifact whose `emb_ref` and `emb_alt` vectors are finite and exactly 1,920 elements wide for every row.
- GPU diagnostic: the final reference batch logged a non-fatal CUDA allocator warning while requesting approximately 249.6 MB with approximately 249.0 MB free; the pass completed, the alternate pass completed, validation passed, and the artifact was uploaded, but this configuration is near the A10G memory edge.
- CPU dry-run: the exact probe-metric target expanded to only `compute_probe` and `compute_probe_metrics`.
- CPU result: the probe fitted all eight eligible consequence classifiers, scored 16,100 of 16,140 rows, skipped the 40 mature-miRNA rows below `min_variants=300`, and uploaded predictions, the classifier joblib, and 20 metric rows.
- Probe diagnostic: the 511 bp classifiers add no `truncation_risk=true` point; several folds select a C-grid edge but their measured edge gain remains below the production truncation threshold.
- Refresh: because the embedding-bearing rerun replaced the zero-shot score artifact, recompute the one downstream zero-shot metric target and treat the refreshed values below as authoritative.

| Protocol | 31 bp | 63 bp | 127 bp | 255 bp | 511 bp inference extension |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot macro AUPRC ± SE | 0.1941 ± 0.0102 | 0.3064 ± 0.0143 | 0.3658 ± 0.0149 | 0.3945 ± 0.0155 | 0.4121 ± 0.0153 |
| Probe macro AUPRC ± SE | 0.3174 ± 0.0269 | 0.3354 ± 0.0266 | 0.4130 ± 0.0255 | 0.4779 ± 0.0216 | 0.4654 ± 0.0280 |

- Zero-shot result: 511 bp changes macro AUPRC by +0.017546, or +4.45%, relative to 255 bp; six of eight subsets improve, while missense changes by -0.006148 and splicing by -0.026575.
- Probe result: 511 bp changes macro AUPRC by -0.012431, or -2.60%, relative to 255 bp; promoter and ncRNA improve by +0.074596 and +0.037030, while the other six subsets decline.
- Interpretation: the 511 bp extension yields a small descriptive zero-shot gain but not a corresponding frozen-probe macro gain, so it does not support a general claim that inference-only context doubling improves variant prediction.
- Boundary: these are arm-wise ±1 SE summaries without a paired 255-to-511 significance test, and 511 bp positions are out of the language model's training distribution even though the rotary implementation can execute them without a learned positional table.
- Presentation: overwrite the two main 3×3 blog-order SVGs and source tables with five context points per panel; use independent y-axes, compact spacing, uncapped SE bars, and a three-regime legend, and exclude mature miRNA from all plotted data.
- Presentation cleanup: remove the superseded standalone 511 bp figure and retain `plot_context_extension.py` as the reproducible integration script.
- Shared-node record: the final lock-protected render ran 22:05:31–22:05:34 UTC with status 0 and 256,952 kB peak RSS.
- Visual review: inspect both full-resolution previews; subplot order, independent y ticks, titles, three-regime legend, uncertainty bars, diagnostic text, and outer labels are legible without overlap.
- Cost: the embedding GPU rescore cost approximately $0.52 and the Spot CPU probe plus metric refresh cost approximately $0.04; including the earlier 511 bp zero-shot arm, total 511 bp work cost approximately $1.05 and all issue #485 SkyPilot work totals approximately $2.65.
- Teardown: `evals-v2-485-ctx511-emb` and `evals-v2-485-ctx511-probe` are terminated, and no issue #485 cluster remains live.
- General fix disposition: the exact-UV bootstrap used on the fresh probe node remains isolated in `codex/issue-485-pin-probe-uv`; remove it from this experiment branch and leave publishing its general PR for explicit GitHub-write authorization.
- Knowledge-base disposition: remain `pending` until a human reviews and accepts the interpretation.
- Next action: run final lightweight registry and artifact checks, then publish only after explicit staging, commit, push, and GitHub-write authorization.

### 2026-08-20 22:21 UTC - Paired 255-versus-511 inference

- Question: determine whether the macro 255-to-511 zero-shot improvement and frozen-probe drop are statistically significant rather than inferring from marginal SE bars.
- Method: run 10,000 reproducible paired bootstrap draws on exactly aligned development rows, resampling match groups within each zero-shot subset and resampling chromosomes jointly across both probe arms and all macro-eligible subsets.
- Zero-shot result: the macro delta is +0.017546 with paired SE 0.005650, 95% percentile CI [+0.005707, +0.027788], and two-sided bootstrap p=0.0024.
- Zero-shot interpretation: the 511 bp macro improvement is statistically significant at the unadjusted 0.05 level under the paired analysis.
- Probe result: the macro delta is -0.012431 with paired SE 0.012872, 95% percentile CI [-0.040806, +0.008653], and two-sided bootstrap p=0.3138.
- Probe interpretation: the observed 511 bp macro drop is not statistically significant.
- Subset boundary: several subset comparisons have nominal p<0.05, but they are exploratory multiple comparisons; do not promote them as headline findings without an explicit correction policy.
- Reproducibility: retain `paired_context_extension.py` and `paired_255_511.csv` beside the figures; the CSV records every subset and macro delta, paired SE, interval, p-value, cluster count, and row count.
- Shared-node record: the lock-protected analysis ran 22:17:30–22:20:09 UTC with status 0 and 202,632 kB peak RSS; live monitoring observed 9.3 GiB available memory and one-minute load 0.99.
- Presentation revision: simplify both figure y-axis labels to `AUPRC (±1 SE)` and retain the uncertainty-method details in the analysis record rather than the axes.
- Shared-node record: the final label-only render ran 22:13:18–22:13:21 UTC with status 0 and 257,192 kB peak RSS, and both full previews passed visual inspection.

### 2026-08-20 22:25 UTC - CTX-VEP-005 proposed 1023 bp zero-shot and trained-probe extension

- Direction: run the 1023 bp zero-shot and frozen-probe arms together rather than gating the probe on the zero-shot result.
- Boundary: keep the fixed 255 bp-trained m5.1 checkpoint and use 1023 DNA bases plus BOS only at inference; train a new downstream frozen linear probe on the same Mendelian `train` rows and unchanged nested chromosome-held-out protocol.
- Execution design: produce one embedding-bearing score artifact and use it for both the zero-shot metrics and the probe, avoiding a second GPU rescore.
- Config: add experiment-only alias `mix-v0.9-p1B-i24-exp135-m5.1-step-59158-ctx1023` with `window_size=1023`, `batch_size=6`, `datasets=[mendelian_traits]`, and explicit probe registration.
- Memory rationale: relative to the successful 511 bp batch-24 run, batch 6 keeps the leading batch-times-length-squared attention term approximately constant while reducing batch-times-length activation volume.
- OOD boundary: 1023 DNA bases plus BOS is a 1024-token forward, four times the checkpoint's 256-token training context, so any result is a stronger positional extrapolation than 511 bp.
- Cost gate: issue #485 has accrued approximately $2.65 against the prior $5 session allowance; target one on-demand A10G equivalent plus a short Spot CPU probe and stop if observed throughput projects total issue cost beyond $5.
- Data gate: use only the pinned Mendelian development split and do not access held-out labels, predictions, or aggregate test metrics.
- Test: the focused registry contract passes 2/2 and asserts the shared checkpoint, odd 1023 bp window, batch size 6, dataset restriction, and probe registration.

### 2026-08-20 23:53 UTC - CTX-VEP-005 1023 bp zero-shot and trained-probe result

- GPU dry-run: the exact embedding-bearing zero-shot target expanded to one each of `download_model`, `compute_scores`, and `compute_metrics`.
- GPU compute: one Spot `g5.xlarge` with an A10G completed 2,690 batches per strand at approximately 1.76 batches/second, or 25.5 minutes per strand.
- GPU diagnostics: the runtime logged the existing rotary-configuration warning and one non-fatal allocator warning during the reference pass; scoring recovered immediately, both strands completed, and no retry was needed.
- Score contract: the uploaded artifact contains 16,140 aligned `train` rows, finite FWD and RC scores, and finite 1,920-element `emb_ref` and `emb_alt` vectors for every row.
- CPU dry-run: the exact probe-metric target expanded to only `compute_probe` and `compute_probe_metrics`.
- Probe result: the Spot `c6i.2xlarge` fitted all eight eligible classifiers, scored 16,100 rows, skipped the 40 mature-miRNA rows below the existing probe gate, and emitted 20 metric rows.
- Probe diagnostics: the fitted classifiers add no new plotted `truncation_risk=true` warning.

| Protocol | 255 bp training context | 511 bp extension | 1023 bp extension |
| --- | ---: | ---: | ---: |
| Zero-shot macro AUPRC ± SE | 0.3945 ± 0.0155 | 0.4121 ± 0.0153 | 0.1061 ± 0.0048 |
| Probe macro AUPRC ± SE | 0.4779 ± 0.0216 | 0.4654 ± 0.0280 | 0.1661 ± 0.0106 |

- Zero-shot paired result: 1023 bp changes macro AUPRC by -0.306024 relative to 511 bp, with paired SE 0.015351, 95% percentile CI [-0.335873, -0.276216], and two-sided bootstrap p=0.0001 at the 10,000-draw resolution.
- Probe paired result: 1023 bp changes macro AUPRC by -0.299312 relative to 511 bp, with paired SE 0.030834, 95% percentile CI [-0.352517, -0.230420], and two-sided bootstrap p=0.0001 at the 10,000-draw resolution.
- Subset pattern: all eight zero-shot subset drops exclude zero; seven of eight probe subset drops exclude zero, while distal has CI [-0.380289, +0.016885] and p=0.0688.
- Score diagnostic: aligned 511 and 1023 bp zero-shot scores have Pearson correlation 0.1053, Spearman correlation 0.3510, and standard deviations 6.8234 and 0.9024, respectively.
- Interpretation: inference-only extension to 1023 bp breaks down for both raw likelihood scoring and a newly trained frozen probe under this checkpoint and protocol.
- Boundary: 1023 DNA bases plus BOS is four times the checkpoint's training sequence length; the result applies to this out-of-distribution inference extension and does not test a model trained at 1023 bp.
- Presentation: the two main SVGs and source tables now contain 31, 63, 127, 255, 511, and 1023 bp in the requested 3×3 blog order, with independent y-axes, compact spacing, uncapped SE bars, a three-regime legend, and no mature-miRNA panel or row.
- Artifact contracts: both source CSVs contain 54 rows, each paired CSV contains 18 rows, and both full-resolution previews passed visual inspection.
- Shared-node record: score validation ran 23:33:09–23:33:11 UTC with status 0 and 267,184 kB peak RSS; the final render ran 23:49:14–23:49:17 UTC with status 0 and 257,796 kB peak RSS; the paired analysis ran 23:49:49–23:52:29 UTC with status 0 and 202,884 kB peak RSS.
- Cost: 1023 bp GPU and CPU work cost approximately $0.44; all issue #485 SkyPilot work totals approximately $3.09, below the $5 session ceiling.
- Teardown: both 1023 bp clusters are terminated, and `sky status` shows no live issue #485 cluster.
- General fix disposition: the fresh-node exact-UV bootstrap remains isolated in `codex/issue-485-pin-probe-uv`; this experiment branch has no `sky/probe.yaml` diff.
- Knowledge-base disposition: remain `pending` until a human reviews and accepts the interpretation.
- Next action: publish only after explicit staging, commit, push, and GitHub-write authorization.

### 2026-08-21 00:30 UTC - Research snapshot, interpretation PR, and independent review

- Snapshot: commit `3d15954dafdaff07b0bc3d083f4a5fd8ca23fa88` and annotated tag `issue-485-inference-context-20260821` preserve the experiment configuration, scripts, compact source tables, paired outputs, figures, and logbook.
- Publication: draft interpretation PR #492 adds the accepted experiment-page proposal, broadens the research question to Context size, and updates the root question index.
- Statistical correction: the paired CSV encodes `p_two_sided=0.0001` as a one-draw floor when no opposing-tail draw occurs; a two-sided 10,000-draw test has resolution 0.0002, so the 511-to-1023 results are reported as p < 0.0002 rather than exact p=0.0001.
- Independent review: the first pass found the p-value-resolution issue and one stale inbound title after the question rename; both were fixed in `952b509a7589dca2c6d653c5933e8d3007e8bd80`.
- Independent re-review: no actionable findings remain.
- Validation: the full PR CI rerun is green, including build, pre-commit, selection, and tests.
- Issue record: the issue body and final comments now contain the corrected inference, commit-pinned provenance, cost and teardown status, and PR link.
- Knowledge-base disposition: `interpretation PR open`.
- Next action: human interpretation review of #492 before merge; keep issue #485 open while the disposition remains temporary.
