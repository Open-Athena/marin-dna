---
topic: soft-vep-metrics
issue: https://github.com/Open-Athena/marin-dna/issues/459
description: Explore score-magnitude VEP metrics as early proxies for AUPRC.
author: gonzalobenegas
---

# Soft VEP Metrics: Task Logbook

## Current TL;DR

- `VEP-SOFT-001` is implementing the CPU-only Mendelian first pass from issue #459.
- AUPRC remains the endpoint. The candidate metrics are early proxies and diagnostics.
- Development data only: the `train` split (odd autosomes and chromosome X). No held-out labels, predictions, or aggregates are in scope.

## Scope

- Goal: Test whether continuous summaries of FWD/RC-averaged Mendelian `-LLR` expose useful training progress before AUPRC rankings change.
- Primary metrics: AUPRC and the global positive-minus-negative mean score gap.
- Secondary metrics: group-balanced gap, standardized group gap, robust median/MAD separation, fixed-temperature soft pairwise win rate, and calibrated proper scores.
- Constraints: Reuse existing score parquets; preserve `match_group` in joint uncertainty estimates; do not run inference or backfill checkpoints; keep the local working set below 500 MiB.
- Coordinating issue: [Open-Athena/marin-dna#459](https://github.com/Open-Athena/marin-dna/issues/459)
- Experiment prefix: `VEP-SOFT`

## Current Baseline

- Date: 2026-08-13
- Code ref: `5e7b406e71c212589bf2654b69904e455b8a96c1`
- Baseline artifact inventory: 48 existing exp232 Mendelian score parquets across five non-distal arms, with nine checkpoints shared by all arms and step 2500 present for `bg`, `cds`, and `utr3` only.
- Baseline endpoint: exp232 issue #232 reports a 7/7 non-distal specialist diagonal at step 4999 and a late missense AUPRC decline.
- Distal coverage: issue #459 reports no exp326/exp351 `evals_v2` score parquets. Their logged AUPRC/LL-gap trajectories remain aggregate-only unless durable per-variant artifacts surface.

## Hypothesis Queue

### Active

- `VEP-SOFT-H1`: A scale-normalized soft metric agrees with final AUPRC ordering more reliably than the raw mean gap across model families and consequence subsets. Next test: implement the metric panel and rescaling controls.
- `VEP-SOFT-H2`: At least one soft metric identifies a mapped exp232 specialist win at an earlier stored checkpoint than AUPRC without hiding the late missense reversal. Next test: analyze all 48 stored exp232 cells with joint match-group resampling.
- `VEP-SOFT-H3`: A leave-one-experiment-out map from a soft metric to AUPRC has useful out-of-trajectory error. Next test: defer until the leaderboard and exp232 metric tables exist.

### Blocked

- `VEP-SOFT-H4`: Soft metrics recover the exp326/exp351 distal ordering by steps 2000-3000. Blocker: no per-variant score artifacts were found when issue #459 was drafted. Resume when an existing durable score bundle is located.

### Falsified / Dead End

- None yet.

### Promoted

- None yet.

## Decision Log

- 2026-08-13: Use the raw global class-conditional mean gap as the headline candidate, while treating it as calibration- and scale-sensitive.
- 2026-08-13: Resample whole `match_group`s jointly across models and metrics. Point estimates remain the ordinary unweighted formulas.
- 2026-08-13: Keep the exp232 cCRE arm and distal subset out of the synchronized non-distal matrix. Patch the distal slot with exp326/exp351 aggregate trajectories and mark unavailable soft cells explicitly.

## Negative Results Index

- No session-local negative result yet. The coordinating issue records the missing exp326/exp351 score prefixes and the failed exp255 distal follow-up.

## Background Research Brief

- Effort: low
- Stop rule: stop once the issue's cited prior work and current implementation identify a concrete first experiment; no paid compute or new inference is being selected.
- Date: 2026-08-13

### Question

Can score-magnitude summaries of per-variant Mendelian LLR provide a stable, earlier signal than AUPRC while preserving the final task ranking and known failure cases?

### Current Marin Context

- [`compute_auprc_metrics`](../../snakemake/analysis/evals_v2/src/marin_dna_evals/metrics.py) already defines score direction, subset gating, and match-group bootstrap conventions.
- [`evals_v2`](../../snakemake/analysis/evals_v2/README.md) stores `label`, `subset`, `match_group`, `llr_fwd`, and `llr_rc` in the per-variant score bundle, so the first pass needs no model inference.
- Issue #232 provides the specialist routing and the late missense counterexample. Issues #326 and #351 provide the distal aggregate comparators.

### External Prior Art

- [Delphi soft downstream metrics](https://openathena.ai/blog/delphi/) motivates preserving probability magnitude and fitting any soft-to-hard map as a separate observational stage.
- [Schaeffer, Miranda, and Koyejo (2023)](https://arxiv.org/abs/2304.15004) show how nonlinear endpoint metrics can turn smooth underlying changes into apparently abrupt transitions.
- [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html) motivates explicit score-scale and calibration controls.
- [Saito and Rehmsmeier (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4349800/) supports retaining precision-recall evaluation as the endpoint for imbalanced tasks.

### Evidence Map

#### Claim: raw score gaps can expose changes that rank metrics ignore

- Support: AUPRC depends only on score ordering, while the proposed mean gap changes when within-order margins change.
- Contradiction: a positive score rescaling changes the raw gap arbitrarily without changing AUPRC.
- Directness to Marin: direct; both quantities are computed from the same persisted Mendelian `-LLR` rows.
- Confidence: mathematical identity for the sensitivity claim; empirical usefulness remains exploratory.
- Action: pair the raw gap with rescaling controls and scale-normalized alternatives.

#### Claim: the early proxy must preserve known biological reversals

- Support: issue #279 and the exp232 late missense trajectory show that stronger likelihood structure or longer specialist training need not improve zero-shot Mendelian AUPRC.
- Contradiction: a smoother soft trajectory may continue improving through a real AUPRC decline.
- Directness to Marin: direct.
- Confidence: replicated within existing Marin issue evidence; the new metrics are untested.
- Action: make missense a named adversarial subset, not an averaged-away diagnostic.

### Recommended Next Experiments

#### 1. `VEP-SOFT-001`: metric primitives and controls

- Minimum experiment: compute AUPRC, raw/global and grouped mean gaps, standardized/robust group separation, and fixed-temperature soft win rate on synthetic matched data.
- Baseline/control: exact AUPRC parity with `sklearn`; complete 1:9 global/group gap equality; score rescaling, sign reversal, and within-group label permutation.
- Expected signal: standardized and rank-like metrics remain stable under positive rescaling; directional metrics reverse or collapse when the score sign/labels are corrupted.
- Falsifier: a candidate fails its defining invariance or cannot use one joint match-group resample across compared score columns.
- Cost/risk: CPU-only unit tests; negligible data cost.

#### 2. `VEP-SOFT-002`: exp232 all-S3 trajectory

- Minimum experiment: read only the five required columns from all 48 score parquets and emit per-arm/checkpoint/subset metrics plus nine synchronized rank tables.
- Baseline/control: reproduce stored `minus_llr_avg` AUPRC and retain the `bg` arm as the negative control.
- Expected signal: at least one normalized soft metric finds a persistent specialist win earlier on one subset without erasing the late missense reversal.
- Falsifier: no metric is earlier and directionally correct, or apparent gains disappear under rescaling/paired uncertainty.
- Cost/risk: S3 I/O and bounded CPU bootstrap; no inference.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Issue #459 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/459 | Scope, inventory, formulas, stop criteria | high | Coordinating specification |
| evals_v2 metrics | Marin code | `snakemake/analysis/evals_v2/src/marin_dna_evals/metrics.py` | AUPRC and cluster-bootstrap conventions | high | Base ref recorded above |
| evals_v2 README | reference docs | `snakemake/analysis/evals_v2/README.md` | Persisted score atoms and split | high | Current pipeline contract |
| Delphi | blog | https://openathena.ai/blog/delphi/ | Soft-to-hard motivation | medium | Analogy, not VEP evidence |
| Schaeffer et al. 2023 | paper | https://arxiv.org/abs/2304.15004 | Metric-induced apparent transitions | medium | General capability metrics |
| Guo et al. 2017 | paper | https://proceedings.mlr.press/v70/guo17a.html | Calibration/scale confounding | medium | General neural calibration |

### Handoff

- Suggested issue update: wait until the metric primitives and their invariance tests pass.
- Open questions: fixed `SoftWin` temperature reference; exact leaderboard model inclusion; S3 object sizes and bounded execution plan.
- Stop reason: the next decision is implementation, not additional prior-art search.

## Entry Log

### 2026-08-13 21:01 UTC - `VEP-SOFT-001` prologue

- Hypothesis: The candidate metric panel can be defined with explicit invariance contracts and one joint match-group bootstrap interface before touching S3 artifacts.
- Commit Hash: `5e7b406e71c212589bf2654b69904e455b8a96c1` (starting point)
- Command: inspected issue #459, `evals_v2` README/config/source/tests, and current repository state.
- Config: development `train` split only; no inference; `n_bootstrap=0` for point-parity unit tests unless uncertainty is under test.
- Result: The existing project already has the score-direction and cluster-bootstrap primitives needed for AUPRC. The soft metric analysis should remain project-local and consume persisted score atoms.
- Interpretation: Start with testable metric primitives, then add an off-`rule all` analysis harness over the explicit 48-object manifest.
- Next action: implement and test the Mendelian soft metric panel and joint bootstrap draws.

### 2026-08-13 21:41 UTC - `VEP-SOFT-002` exact bounded bootstrap

- Hypothesis: Vectorizing the joint cluster bootstrap over matched-group atoms can
  make the 48-cell exp232 analysis practical without changing AUPRC semantics.
- Commit Hash: task snapshot pending.
- Command: `uv run --locked soft-vep-analysis --n-bootstrap 1000 --seed 459`
  under the shared-node lock, two Polars/Rayon threads, single BLAS threads,
  `nice -n 10`, and low-priority I/O.
- Config: 48 explicit exp232 score objects; seven non-distal subsets; fixed
  SoftWin temperature 0.6455015093 from pooled `exp232-v4_bg-step-500` margins;
  one joint 1,000-draw match-group bootstrap per arm/checkpoint/subset cell.
- Result: The first row-expansion implementation was stopped after load crossed
  the node's 3.0 threshold. Profiling identified repeated sklearn AUPRC calls as
  the hot path. Replacing them with a tested weighted, tie-aware AP calculation
  and vectorized group atoms completed in 73 seconds at 407,448 KiB peak RSS.
  An explicit duplicated-row test proves exact parity with sklearn AP.
- Interpretation: The optimized implementation is bounded and reproduces the
  ordinary metric while preserving joint bootstrap draws.
- Next action: analyze the current MarinDNA leaderboard and add confidence-only
  comparisons.

### 2026-08-13 22:06 UTC - `VEP-SOFT-003` current leaderboard

- Hypothesis: A normalized or calibrated candidate will preserve current
  cross-model AUPRC ordering better than the raw score gaps.
- Commit Hash: task snapshot pending.
- Command: `uv run --locked soft-vep-leaderboard-analysis --n-bootstrap 1000
  --seed 459` over the 22 current `family: marin_dna` registry models with
  Mendelian coverage.
- Config: seven development subsets; joint model bootstrap; equal-weight macro
  average; isotonic soft-to-AUPRC projection cross-fit by ten whole experiment
  groups.
- Result: Calibrated Brier is best: macro Spearman 0.9955/Kendall 0.9654, top-three
  overlap 3/3, LOEO macro MAE 0.00937, and zero confidence-supported reversals
  among 988 informative subset/model pairs. Calibrated log loss is second
  (0.9864/0.9394, MAE 0.01150, zero of 995 supported reversals). Raw gaps have
  macro Spearman 0.8509, LOEO MAE 0.03954, and 41 supported reversals among 1,035
  informative pairs.
- Negative result: only `scaling-v0.5-h2944-p4B-step-215573` fails stored AUPRC
  parity, on all seven subsets; the largest absolute difference is 0.002643 on
  3′UTR. The artifact records the mismatch instead of aborting or hiding it.
- Resource record: 49.5 seconds wall time, 406,816 KiB peak RSS, status 0.
- Next action: patch distal aggregates, add distribution figures, and decide
  whether the apparent early signal survives paired uncertainty.

### 2026-08-13 22:13 UTC - `VEP-SOFT-004` final exp232 and distal artifacts

- Hypothesis: A useful soft metric must both track the known late missense
  degradation and produce an earlier bootstrap-supported specialist win.
- Commit Hash: task snapshot pending.
- Commands: final 1,000-draw exp232 pass; `soft-vep-distal-patch` with the locked
  optional W&B client; exact W&B `scan_history` rather than sampled histories.
- Result: exp232 AUPRC parity is exact for all 336 arm/step/subset cells. Brier
  follows the missense/CDS decline after step 4000 (0.07920 to 0.08010, lower is
  better), while the global gap rises 5.45 to 8.06 and hides the reversal. The
  point-estimate early wins do not establish a robust decision lead: missense
  AUPRC, Brier, log loss, raw gaps, group SMD, and SoftWin all first sustain a
  supported specialist win at step 1500. The raw gap/SoftWin lead on 3′UTR does
  not survive the cross-model scale/robustness criteria; Brier has no persistent
  supported 3′UTR win.
- Distal result: preserved 9 exp232 baseline, 14/14 exp326, and 14/11 exp351
  finite log records. Final AUPRC is 0.1268 baseline, 0.2990/0.2719 exp326, and
  0.3082/0.3663 exp351. No soft value or interval was imputed.
- Negative result: the exp232 cCRE baseline has no stored step 3500 metric. The
  first collector attempt correctly failed on that nonexistent object; the
  explicit manifest now omits it and performs no interpolation.
- Resource record: final exp232 81.7 seconds, 428,960 KiB peak RSS, status 0;
  distal pull 11.1 seconds, 333,248 KiB peak RSS, status 0.
- Next action: run the complete project tests, snapshot, and publish the decision.

### 2026-08-13 22:18 UTC - `VEP-SOFT-005` decision and verification

- Hypothesis disposition:
  - `VEP-SOFT-H1` supported for calibrated Brier, not for raw gaps: Brier is the
    most rank-faithful and best LOEO proxy across current model families.
  - `VEP-SOFT-H2` not established: no candidate provides a consistently earlier,
    confidence-supported specialist signal while passing robustness controls.
  - `VEP-SOFT-H3` supported for further work: Brier's macro LOEO MAE is 0.00937,
    but this does not authorize replacing the AUPRC endpoint.
  - `VEP-SOFT-H4` remains blocked by absent exp326/351 per-variant score bundles;
    aggregate AUPRC only is reported.
- Decision: keep AUPRC as the official endpoint. Reject raw gaps, group SMD,
  median/MAD, and SoftWin for standard integration. Carry grouped-CV calibrated
  Brier forward only as an optional diagnostic / second validation candidate.
- Verification: `uv run --locked pytest` completed with 370 passed, 5 skipped,
  and 44 pre-existing warnings in 38.3 seconds; status 0. `git diff --check`
  passed. Representative Brier, missense-distribution, and distal SVGs were
  rasterized and visually reviewed; crowded trajectory axis labels were shortened
  and the plots regenerated.
- Resource note: the full suite unexpectedly peaked at 1,184,644 KiB RSS, above
  the 500 MiB local-workload planning bound despite the pre-run capacity gate.
  Do not repeat that full suite on this shared node; use remote CI for subsequent
  whole-project reruns. Memory remained well above the 4 GiB stop threshold.
- Artifact: `.agents/artifacts/459-soft-vep/summary.md` contains the full decision,
  numeric evidence, limitations, and artifact index.
- Next action: create a stable task snapshot and post a concise issue update.

### 2026-08-13 22:20 UTC - `VEP-SOFT-006` stable task snapshot

- Commit Hash: `dacdf895a5ab765e86bc99676d5b891e39873c71`
- Snapshot scope: metric and bootstrap primitives, exp232/leaderboard/distal
  analysis commands, 22 focused tests, full README instructions, compact Parquet
  tables, reviewed SVG figures, decision summary, and this logbook.
- Reproduction commands: the three `soft-vep-*` commands in the owning README,
  all with locked dependencies and seed 459; `uv run --locked pytest` is the
  project verification command but should run in remote CI after the observed
  local 1.13 GiB peak.
- Interpretation: this commit is the stable evidence snapshot for issue #459.
  Subsequent prose-only issue updates should pin links to this hash.
- Next action: publish the result summary without closing the issue or promoting
  Brier into the production metric path.

### 2026-08-13 22:25 UTC - `VEP-SOFT-007` composite panel completion

- Commit Hash: completion snapshot pending.
- Result: both eight-subset AUPRC composites first meet the issue's
  two-recorded-evaluation point-estimate rule at step 3000. For the supported
  exp326 panel, the distal arm first beats both recorded comparators twice at
  steps 3000 and 4000 (the next step shared with the exp232 baseline). For the
  exp351 panel, centered first beats tiled twice at steps 2500 and 3000; the
  seven non-distal exp232 slots are not all ready until step 3000.
- Limitation: this is not an all-subset confidence result. Distal has no
  per-variant scores, and synonymous has no persistent bootstrap-supported
  specialist win under AUPRC or any candidate metric.
- Artifact: `distal/patched_panel_summary.parquet` records both point-estimate
  timings and the unavailable soft/bootstrap coverage explicitly.
- Verification: 23 focused issue tests passed after the addition and all eight
  new Python files pass `ruff format --check`. The full 370-test project suite
  was not repeated because its measured memory exceeds the shared-node local
  planning bound; it passed before this focused, independently tested addition.
- Next action: snapshot and post the issue update.

### 2026-08-13 22:26 UTC - `VEP-SOFT-008` completion snapshot

- Commit Hash: `afd94fecdb45c1b294584cc6691ef3c0c8638d31`
- Result: stable snapshot now includes the explicit two-panel timing table,
  formatted implementation, focused tests, and updated reproduction command.
- Next action: use this hash for the issue's implementation, summary, and figure
  permalinks.
