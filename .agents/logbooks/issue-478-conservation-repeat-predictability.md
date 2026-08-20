---
topic: issue-478-conservation-repeat-predictability
issue: https://github.com/Open-Athena/marin-dna/issues/478
description: Conservation by repeat predictability across the 46M-4B scaling ladder
author: gonzalobenegas
---

# Conservation by Repeat Predictability: Research Logbook

## Current TL;DR

- Status: complete. The original 8-checkpoint audit and the non-repeat conservation-classification follow-up are durable and validated.
- Absolute loss and entropy classify conserved non-repeat positions above prevalence globally and in CDS, upstream, and downstream; AUPRC increases monotonically with model size.
- The practical 46M-to-76M loss delta is above prevalence in every scope but weaker than either absolute small-model score.
- FWD-only and RC-only AUPRC are nearly identical; one orientation preserves most absolute-loss ranking lift at half the inference compute, while loss-delta ranking degrades more.
- The primary central-span analysis covers 3,129,344 bases per region after excluding 32 bases from each window edge, with 1,000 10-Mb block-bootstrap replicates and zero ambiguous bases.
- Adjusted 46M-to-4B loss reduction for conserved nonrepeat sequence was 0.364 nats/base in CDS (95% CI 0.356–0.372), 0.292 upstream (0.283–0.302), and 0.242 downstream (0.232–0.252).
- Repeat interactions were negative in every region: repeats improved less with scale, but conservation remained positively associated with improvement within repeats. The broad claim that repeats are intrinsically easier was not supported after composition controls.
- 46M absolute NLL and predictive entropy also tracked conserved nonrepeat sequence and advance as cheaper baseline weighting candidates; scale-differential loss is the primary causal-test candidate.
- All eight checkpoints use the same mixture: CDS 0.7319, upstream 0.2062, downstream 0.0619; uppercase weight 1.0 and lowercase weight 0.01.
- CDS-only codon position passed as a positive control on both strands; splice donor/acceptor results remain descriptive secondary evidence.
- FWD-only and RC-only analyses preserved the group-level result. Relative to the FWD/RC mean, one-orientation endpoint scores had 0.69–0.81 Spearman correlation, 0.58–0.72 top-decile overlap, and 0.76–0.86 gain-sign agreement across regions; neither orientation was consistently better.
- GPU spend was approximately $4.18; CPU analyses added about $0.31, below the $20 cap. Exact corpus exposure and homology density were unavailable and remain limitations.
- The permanent research branch contains the full token cache contract, classification metrics, and reviewed plots; interpretation PR #482 should be updated after the human plot iteration.

## Scope

- Goal: decide whether absolute loss, 46M predictive entropy, or loss reduction with model scale should advance to a fixed-compute training-weight experiment.
- Primary metrics: per-base NLL; 46M predictive entropy; 46M-to-4B and adjacent-rung loss reduction; fraction of positive loss reductions; block-bootstrap uncertainty for the issue #478 contrasts.
- Primary strata: region x validation-case conservation x RefSeq repeat status.
- Controls: GC content, target position, and a 7-mer predictability baseline. Homology or corpus-exposure density will be included only if pinned compatible metadata is available.
- Constraints: 0-based half-open coordinates internally; exact validation-matched RefSeq assembly and sequence names; additive S3 namespace; no local workload with an estimated working set above 500 MiB; $20 paid-GPU limit; smallest model first.
- Coordinating issue: https://github.com/Open-Athena/marin-dna/issues/478
- Branch: `codex/issue-478-conservation-repeat-predictability`
- Experiment prefix: `CRP`
- Shared tags: `CRP`, `issue-478`, `evals-v2`

## Baseline

- Date: 2026-08-19
- Code ref: `330bd32a8812a81fb561d3f3d1d85aba8d3e4a5b`
- Issue #274 cached scores: `s3://oa-bolinas/snakemake/analysis/evals_v2/results/ll_gap/scores/`
- Issue #296 durable artifacts: `s3://oa-bolinas/analysis/issue296/`
- RefSeq soft-mask source: `s3://oa-bolinas/snakemake/training_dataset/dataset_creation/results/genome/GCF_000001405.40.2bit`
- Baseline numbers: no issue #478 result yet. The forward-strand regression target is the issue #274 per-window sums and counts for each model x region cell.

## Hypothesis Queue

### Active

- None.

### Blocked

- None.

### Falsified / Dead End

- `CRP-001`, lower-small-model-loss clause: repeats had lower raw 46M NLL in several cells, but this did not survive the prespecified composition controls as a broad repeat effect.
- A fixed raw per-window maximum was unsuitable as a cross-runtime issue #274 regression gate because drift scales with the number of labeled bases; it was replaced by exact identities/counts plus correlation, q99, aggregate-drift, and per-base maximum gates.

### Promoted

- `CRP-001`, smaller-scale-gain clause: supported in all regions, including negative repeat interactions in the controlled endpoint models.
- `CRP-002`: supported in all three regions; advance scale-differential loss to a fixed-compute causal weighting test.
- `CRP-003`: supported on both strands; codon positions 1 and 2 improved more with scale than position 3.
- `CRP-004`: strand/context-specific donor and acceptor differences were observed, but remain secondary and do not change the training recommendation.

- `CRP-005`: supported globally and within all three regions; absolute loss, entropy, and the practical 46M-to-76M loss delta all exceeded conserved prevalence among non-repeat positions.
## Decision Log

- 2026-08-19: use the three pinned validation datasets and revisions from issue #478 and `evals_v2/config/config.yaml`.
- 2026-08-19: use the matching RefSeq `GCF_000001405.40` soft-mask asset. Do not substitute the canonical Ensembl release 115 reference used by other evals.
- 2026-08-19: extend `evals_v2` additively under an issue-specific target and S3 namespace. Preserve the existing issue #274 `ll_gap` rules and artifacts.
- 2026-08-19: port the validated issue #296 per-token kernel into the owning `evals_v2` package instead of designing a second kernel.
- 2026-08-19: run the 46M checkpoint first and stop on schema, coordinate, strand-mapping, or #274 regression failure.
- 2026-08-19: keep total paid GPU spend at or below $20.
- 2026-08-19: keep codon and splice feature strata secondary and CDS-only; do not apply them to upstream or downstream windows.
- 2026-08-19: use full-vocabulary true-base NLL for #274 parity and nucleotide-renormalized entropy for the 46M predictability diagnostic.
- 2026-08-20: accept same-corpus scale-differential loss as the primary fixed-compute weighting candidate; accept 46M absolute NLL and entropy as cheaper baseline candidates, not as causal proxies.
- 2026-08-20: reject a broad repeat-intrinsically-easier interpretation after composition controls. Keep issue #87 as the direct test of the current repeat downweighting because this inference-only audit did not ablate it.
- 2026-08-20: interpret the result as scale-dependent learnability, not Rho-1 reducible loss or proof of functional discovery; exact corpus exposure and homology density were unavailable.
- 2026-08-20: accept one-orientation scoring as a half-inference-compute screen because all group-level conclusions are stable. Retain FWD/RC averaging for a finalized per-base weighting function unless a downstream fixed-compute ablation shows that the observed rank and tail-set disagreement is harmless; FWD and RC are tied, so there is no empirical reason to privilege either direction.

## Background Research Brief

- Effort: medium. Searched Marin issues, branches, artifacts, run configs, and primary literature on reducible loss, conservation weighting, repeats, and homology leakage.
- Prior Marin evidence: #296 validated the token kernel and found codon-position ordering plus splice-site strand asymmetry on a separate Ensembl CDS-centric set. #177 pinned training repeat treatment to the source soft mask with lowercase loss weight 0.01. #274 supplies the exact FWD regression cache.
- External evidence: Rho-1 defines reducible loss against a reference model trained on a curated target distribution, so same-corpus 46M-to-4B loss reduction is a scale-sensitivity signal, not Rho-1 reducible loss. GPN/GPN-MSA support evaluating conservation/repeat weighting and warn that repeat behavior is heterogeneous. Homology-leakage work motivates an explicit exposure-metadata limitation.
- Negative result: the pinned HF validation datasets expose only `id` and mixed-case `seq`; no compatible per-window exposure or homology-density field was found.
- Source ledger: #274, #296, #177; Rho-1 (arXiv:2404.07965); GPN-MSA (PMCID: PMC10592768); GPN (PMCID: PMC10622914); homology-leakage preprint (bioRxiv 2025.01.22.634321); Dark Regulome preprint (arXiv:2606.06834, exploratory only).

## Negative Results Index

- No compatible exact training-corpus exposure or homology-density metadata was available for the pinned validation windows.
- Repeat sequence was not broadly lower-loss after the prespecified GC, 7-mer, position, conservation, and interaction controls.
- Two conserved-negative repeat strata had tiny negative 46M-to-76M mean changes (upstream -0.00064 and downstream -0.00056 nats/base); later adjacent rungs and all endpoint changes were positive.
- FWD and RC agreed only modestly with each other at individual bases. A single orientation was much closer to their mean but still did not reproduce the exact endpoint ranking or top-decile set.

## Entry Log

### 2026-08-19 22:25 UTC - CRP-001 prologue and access audit

- Hypothesis: the existing issue #296 per-token scorer and issue #274 pipeline can support the issue #478 experiment through an additive extension.
- Commit Hash: `330bd32a8812a81fb561d3f3d1d85aba8d3e4a5b` (starting point)
- Command: GitHub issue and prior-work reads; targeted `rg`; read-only AWS, GCS, Hugging Face, SkyPilot, and GitHub access checks.
- Config: issue #478 pinned datasets and eight scaling checkpoints; RefSeq `GCF_000001405.40`; 46M-first; $20 paid-GPU cap.
- Result: all required stores and launch credentials are reachable. Issue #296 contains a validated per-token loss kernel and full-ladder FWD caches, but it excludes reverse-complement scoring, repeats, two region families, and `evals_v2` integration.
- Interpretation: implementation can reuse prior tested logic. The first new remote run should validate the joined repeat artifact and FWD/RC per-base mapping on 46M before launching larger checkpoints.
- Next action: finish the medium prior-work brief, inspect and port the issue #296 kernel, and define the additive `evals_v2` artifact contracts.

### 2026-08-19 23:04 UTC - CRP-001 implementation and local gates

- Hypothesis: an additive `evals_v2` path can preserve exact per-base genomic alignment while reproducing issue #274.
- Commit Hash: working tree on `codex/issue-478-conservation-repeat-predictability`.
- Command: focused `uv run --locked pytest`; `snakemake -n predictability_478_pilot`; `snakemake -n predictability_478`.
- Config: version `v1`; RefSeq `GCF_000001405.40`; central `[32,223)`; 10-Mb genomic blocks; 1,000 bootstrap replicates; CDS-only codon and canonical 2-bp splice diagnostics.
- Result: 21 focused tests pass. Both the 46M pilot and full-ladder DAGs resolve through the S3 storage provider. No model inference has run.
- Interpretation: local schema, BOS, RC reversal, annotation ambiguity, exact-sequence, 7-mer, regression-comparison, score-direction, bootstrap, and control-model gates are ready for the remote pilot.
- Next action: run the full evals_v2 test gate and launch only `predictability_478_pilot`.

### 2026-08-20 00:18 UTC - CRP-001 pilot passed

- Hypothesis: the 46M per-base implementation preserves window identity, genomic alignment, and issue #274 FWD case aggregation before any larger checkpoint is scored.
- Commit Hash: working tree atop `14cfc93316d72593c3b0bc271e969ab190e2ca56`.
- Command: credential-free `sky launch snakemake/analysis/evals_v2/sky/predictability_478.yaml -c evals-v2-478-pilot-dlami --env "SNAKEMAKE_ARGS=--resources gpu=1 -- predictability_478_pilot"`.
- Config: AWS `g5.xlarge` A10G; official DLAMI `ami-0a15d33a6697fe677`; driver 595.91.07; PyTorch 2.13 CUDA 13; exact `uv==0.11.31`; 16,384 windows per region.
- Result: all three regression reports passed with exact IDs and uppercase/lowercase counts. Worst observed upper-case metrics were CDS correlation 0.99999743, q99 absolute per-window drift 0.48036 nats, max 1.44257 nats, and aggregate drift -1.4344e-05 nats/token. Worst lower-case q99 was 0.24040 nats, while every lower correlation exceeded 0.99999948. A same-runtime control comparing the old and new kernels on 2,048 upstream windows differed by at most 4.57e-05 nats/window with token-weighted drift below 5e-10.
- Result: the joined artifacts contain 4,177,920 positions per region, no ambiguous sequence bases, and 1,048,576 edge positions excluded from the primary span. CDS labels include 798,522/798,094/798,568 bases at codon positions 1/2/3, 25,703 donor bases, and 26,600 acceptor bases; these fields are CDS-only.
- Interpretation: the new kernel and coordinate mapping are correct. The legacy cache was created under a different PyTorch/CUDA runtime, so a single 5e-4 per-window maximum was not a valid cross-runtime gate. The replacement retains exact identities/counts and adds correlation, raw-sum q99, per-base-normalized maximum, and aggregate-drift bounds.
- Operational finding: issue #462 is the same Sky default-image CUDA incompatibility encountered here, with upstream SkyPilot issue #9406. Pin the current official DLAMI; do not file a duplicate MarinDNA issue.
- Operational finding: all eight final ladder checkpoints now have public, byte-identical Hugging Face mirrors. Pinning those immutable revisions enables a credential-free Sky task and avoids mounting local GCP application-default credentials on AWS.
- Next action: run the remaining 76M-4B cells, all regression gates, controlled analysis, and compact figure within the $20 cap.

### 2026-08-20 01:52 UTC - CRP-001 full-ladder execution guardrails

- Hypothesis: the larger-rung regression failures should distinguish a mapping error from bounded cross-runtime accumulation before relaxing any gate.
- Commit Hash: working tree atop `14cfc93316d72593c3b0bc271e969ab190e2ca56`.
- Command: resumable `sky exec`; completed-report audit; exact remote import-path and function-signature checks.
- Result: 255M upstream produced one 2.55787-nat upper-case window-sum outlier across 230 labeled bases, or 0.01112 nats/base. Its upper-case correlation is 0.99999957, q99 drift is 0.21149 nats, and aggregate drift is -7.43e-06 nats/token. The next-largest window drift is 1.0129 nats. The fixed raw-sum maximum therefore scaled with labeled-base count rather than an alignment defect.
- Result: an exact follow-up audit found the 255M upstream maximum was 0.05999 nats/base, while 4B upstream reached 0.10202 nats/base. The associated correlations were 0.99999957 and 0.99999954, q99 drifts were 0.21149 and 0.17508 nats, and aggregate drifts were -7.43e-06 and -7.08e-06 nats/token. Lower-case maxima were 0.00448 and 0.00862 nats/base. In both cases the maximum came from a sparse window rather than a distributed mapping error.
- Decision: use a 0.25-nat/base supplemental maximum while retaining exact IDs/counts, q99 <= 0.55 nats, correlation >= 0.99999, and aggregate drift <= 2e-05 nats/token. This leaves 2.45x headroom over the observed sparse-window maximum; a focused regression test verifies that a 0.3-nat drift on a one-base window is still rejected. Regenerate all reports under one self-describing schema after scoring.
- Operational finding: a reused Sky workdir retained a stale root-level `src/marin_dna_evals` tree, which preceded the synced project-local `src` on `sys.path`. Workdir sync alone did not update that shadow copy. Both Sky tasks now prepend the owning project's `src` through `PYTHONPATH`; the remote signature check resolves the synced normalized gate.
- Result: inference outputs completed before each failed CPU gate were uploaded and reused, including 2B CDS. Estimated task-cluster cost was $1.79 at the last audit.
- Next action: finish the ten remaining model-region cells, regenerate all 24 validation reports, run the controlled analysis, and inspect the decision figure.

### 2026-08-20 03:54 UTC - CRP full ladder complete

- Hypothesis: same-corpus 46M-to-4B loss reduction remains positively associated with conservation after repeat, GC, local 7-mer predictability, and position controls, making it a candidate training weight.
- Commit Hash: working tree atop `14cfc93316d72593c3b0bc271e969ab190e2ca56`.
- Command: resumable full `predictability_478` Snakemake target on AWS `g5.xlarge`; schema audit of all regression JSON; controlled-summary inspection; rendered-figure review.
- Config: eight fixed-token checkpoints from 46M through 4B; three pinned RefSeq datasets; FWD/RC per-base average; primary positions `[32, 223)`; all-255 sensitivity; 10-Mb blocks; 1,000 bootstrap replicates; seed 478; controls `conserved * repeat + GC quadratic + 7-mer quadratic + position cubic`.
- Validation: all 24 issue #274 regression reports passed under schema 2. Worst upper/lower correlation was 0.99999520/0.99999924, q99 drift 0.48036/0.28254 nats per window, aggregate drift 1.4564e-05/1.2148e-05 nats per token, and maximum 0.10202/0.02294 nats per labeled base.
- Result: primary endpoint mean loss reductions for conserved nonrepeat sequence were 0.727 CDS, 0.351 upstream, and 0.267 downstream nats/base, versus 0.328, 0.056, and 0.025 for nonconserved nonrepeat sequence. Adjusted conserved-nonrepeat coefficients were 0.364 (95% CI 0.356–0.372), 0.292 (0.283–0.302), and 0.242 (0.232–0.252), respectively.
- Result: adjusted repeat coefficients were -0.196 CDS, -0.006 upstream, and -0.004 downstream; conservation-by-repeat interactions were -0.018, -0.206, and -0.177. Repeats therefore improved less with scale, while conservation remained positively associated with scale gain inside repeat strata.
- Result: 46M absolute NLL and predictive entropy both tracked conserved nonrepeat sequence after controls. They advance as cheaper baseline weighting candidates, while scale-differential loss advances as the primary candidate. Raw repeat NLL was sometimes lower, but the broad intrinsically-easier interpretation did not survive controls.
- Sensitivity: all-255 endpoints preserved the primary ordering and direction. All-rung mean curves were monotone except tiny 46M-to-76M decreases for nonconserved repeats upstream (-0.00064) and downstream (-0.00056 nats/base); every later adjacent mean and every endpoint gain was positive.
- CDS-only secondary result: codon positions 1 and 2 had endpoint gains of about 0.66–0.68 nats/base on both strands versus about 0.52–0.53 at position 3, passing the positive control. Donor/acceptor strand differences replicated descriptively and remain secondary.
- Artifacts: `s3://oa-bolinas/snakemake/analysis/evals_v2/results/predictability_478/v1/` contains 48 per-base score atoms, 24 regression reports, `analysis/summary.parquet` (528 rows), `analysis/controlled.parquet` (132 rows), `analysis/manifest.json`, and `figure/predictability.png`.
- Cost: approximately $4.08 for the successful DLAMI cluster plus $0.10 for the earlier spot attempt, or $4.18 total. The cluster auto-stopped and was terminated.
- Interpretation: accept same-corpus scale-differential loss for a causal weighting test and carry absolute NLL/entropy as baselines. This audit does not establish out-of-distribution functional discovery, causal training benefit, Rho-1 reducible loss, or the value of the current repeat downweighting; exact exposure and homology density remain alternative explanations.
- Next action: at matched training compute, compare a frozen scale-differential weight with uniform loss, current repeat weighting, 46M absolute-NLL weighting, and 46M entropy weighting.

### 2026-08-20 12:00 UTC - CRP orientation and half-compute sensitivity

- Hypothesis: omitting the FWD/RC average preserves both the conservation-by-repeat scaling pattern and enough of the per-base candidate ranking to support a half-inference-compute alternative.
- Commit Hash: working tree atop `14cfc93316d72593c3b0bc271e969ab190e2ca56`.
- Command: CPU-only forced rerun of `analyze_predictability_478_orientations` and `plot_predictability_478_orientations` on AWS `m7i.2xlarge` Spot, reusing all 48 existing score atoms; focused tests and rendered-figure review.
- Config: FWD-only and genomically realigned RC-only sensitivities; the same central `[32, 223)` span, all-255 check, strata, controls, 1,000 block-bootstrap replicates, and CDS-only secondary features as the primary analysis; 100,000-base sampled Spearman and 10% tail overlap for absolute 46M NLL, 46M entropy, and 46M-to-4B endpoint gain.
- Result: the group-level conclusions were unchanged. All endpoint cell means were positive in both orientations. Every adjacent-rung cell mean was positive except the same tiny first-rung decreases for nonconserved repeats upstream and downstream. Adjusted endpoint conservation coefficients were FWD/RC 0.365/0.364 CDS, 0.292/0.293 upstream, and 0.244/0.241 downstream, nearly identical to the averaged 0.364/0.292/0.242.
- Result: FWD and RC themselves had modest per-base endpoint agreement (Pearson 0.35–0.39; sampled Spearman 0.09 downstream, 0.15 upstream, and 0.37 CDS). Each single orientation was substantially closer to the FWD/RC mean: endpoint Pearson 0.82–0.83, sampled Spearman 0.69–0.81, top-decile overlap 0.58–0.72, and gain-sign agreement 0.76–0.86. Absolute-NLL single-versus-mean Spearman was 0.77–0.82 with 0.56–0.64 top-decile overlap; entropy was 0.70–0.77 with 0.42–0.45 overlap. Conservation-by-repeat cell minima remained 0.66 Spearman and 0.55 top-decile overlap for endpoint gain.
- Result: FWD and RC substitution metrics were nearly symmetric and neither direction was consistently superior. The revised durable outputs contain 1,056 orientation summary rows, 264 controlled rows, 135 agreement rows, a self-describing manifest, and `figure/orientation_sensitivity.png` under the existing `v1` prefix.
- Cost: four short CPU spot lifetimes, including one unavailable-instance attempt and one forced-rerun iteration, were approximately $0.03 total; no inference or GPU was used. Combined issue spend is approximately $4.21.
- Interpretation: one-orientation scoring is adequate for aggregate discovery and a cheap pilot, cutting inference compute in half. It is not an exact replacement for the averaged per-base weight: the top-decile membership and endpoint sign losses are large enough that the causal training experiment should either retain the average or include single-orientation scoring as an explicit compute-quality ablation. If only one pass is affordable, choose orientation deterministically without claiming FWD or RC is biologically preferred.
- Next action: carry both averaged and half-compute single-orientation score construction into the matched-compute weighting experiment; decide whether the downstream performance difference justifies the second pass.

### 2026-08-20 12:18 UTC - Rebase and interpretation reporting

- Hypothesis: the completed issue #478 evidence has a valid bounded interpretation that should be promoted under the experiment-page guidance introduced by #476.
- Commit Hash: `5b7b1298` after rebasing the two issue #478 commits onto `origin/main` at `d40a56ac`.
- Command: `git fetch origin main`; `git rebase origin/main`; regenerate the primary and orientation figures as SVG from the durable compact Parquet outputs.
- Config: the accepted page uses the same primary central span, controlled coefficients, block-bootstrap intervals, orientation sensitivity, and limitations as the final logbook entries.
- Result: the rebase completed without conflicts. `docs/research/experiments/478-conservation-repeat-predictability.md` records the accepted findings, evidence, directions, limitations, question backlink, and canonical issue. Two reviewed SVG figures are under `docs/research/experiments/figures/478/`, and the training-regions question now links the experiment page.
- Validation: the rebased `evals_v2` suite passed with 379 tests and 5 skips; scoped pre-commit hooks and `snakemake -n predictability_478` passed. The full suite peaked at 1,226,072 kB RSS, above the pre-run sub-500-MiB estimate, so it must not be repeated on the shared node; use focused local checks or appropriately sized remote compute.
- Interpretation: valid observational claims should be promoted. The page scopes the result to same-corpus scale-dependent learnability and explicitly excludes causal training benefit, Rho-1 equivalence, and a conclusion about repeat downweighting.
- Disposition: `pending` until a clean interpretation pull request is opened and reviewed; the research issue must remain open until that disposition is final.
- Next action: verify the rebased code and SVG render paths, snapshot the interpretation update, then extract or open the required interpretation pull request after GitHub-write authorization.

### 2026-08-20 19:02 UTC - CRP-005 conservation-classification pilot design

- Hypothesis: model uncertainty or scale-dependent loss reduction can classify conserved versus non-conserved positions without using conservation as an input score.
- Commit Hash: e175657b.
- Population: central positions [32, 223), excluding RefSeq repeats and ambiguous bases; positive class is conserved.
- Scores: negative FWD/RC-mean NLL and negative four-nucleotide entropy at every available model size; smaller-model NLL minus larger-model NLL for every ordered model pair.
- Metric: exact pooled average precision, reported as AUPRC with the conserved prevalence baseline, globally and separately for CDS, upstream, and downstream.
- Variation diagnostic: within-10-Mb-block AUPRC is stored separately; it describes genomic variation and is not a confidence interval for pooled AUPRC.
- Pilot: 46M and 76M only, including their practical loss delta, before reading the other six checkpoints.
- Command: uv run --locked snakemake -n predictability_478_classification_pilot.
- Validation: 6 focused classifier/figure tests pass; scoped pre-commit hooks pass; the dry-run plans one CPU analysis job over three joined artifacts and 12 existing atom files with no inference.
- Result: the revised non-repeat loss plot uses 3-inch square facets, automatic logarithmic ticks, and compact horizontal spacing. The 2-by-2 composition panels put conserved before non-conserved and report exact central-span percentages globally and by region.
- Next action: launch the CPU-only pilot, verify exact eligible-position counts and finite metrics, and inspect global plus per-region AUPRC before the full 28-pair sweep.

#### Background research brief

- Effort: low. The stop rule was that additional sources no longer changed the user-specified score set or pilot.
- Current Marin context: issue #478 already found that lower loss, lower entropy, and larger 46M-to-4B loss reduction track conserved non-repeat sequence in mean and controlled analyses.
- Internal prior work: the existing issue #478 atom contract preserves per-base FWD and genomically realigned RC NLL and entropy for all eight model sizes; issue #175 and the orientation follow-up support FWD/RC averaging as the primary score.
- External prior art: scikit-learn defines average precision as the recall-increment-weighted mean of precision over the ranking and accepts non-thresholded decision scores: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html.
- External prior art: GPN-MSA uses evolutionary conservation during training and demonstrates that genomic model scores can carry functional information beyond a single conservation column: https://pmc.ncbi.nlm.nih.gov/articles/PMC10592768/.
- Negative lead: mean separation does not guarantee useful per-position ranking. Exact training exposure and homology density remain unavailable, so AUPRC cannot establish that a score is a causal or out-of-distribution functional proxy.
- Contradiction check: region-specific prevalence can make raw AUPRC incomparable across regions. Every result therefore carries its own prevalence and AUPRC-minus-prevalence value.
- Falsifier: the candidate is not useful for this proxy task if pooled AUPRC is at or below prevalence, or if any apparent global lift is absent within the three region-specific analyses.
- Cost and risk: the pilot reuses existing atoms on one CPU instance and performs no model inference. The full sweep is conditional on pilot correctness and signal.
- Source ledger: issue #478 and its permanent logbook (direct Marin evidence); scikit-learn average-precision documentation (metric definition); GPN-MSA (external genomic-model precedent).

### 2026-08-20 19:50 UTC - CRP-005 conservation classification complete

- Hypothesis: among non-repeat positions, model loss, predictive entropy, and model-to-model loss deltas can rank case-encoded conserved positions without using conservation as a score input.
- Commit Hashes: `92f06090` for the full averaged sweep and figures; `700e3c50` for the single-orientation comparisons.
- Commands: CPU-only SkyPilot execution of `predictability_478_classification_pilot` followed by `predictability_478_classification`; no model inference or GPU was used.
- Population: central positions `[32, 223)`, excluding RefSeq repeats and ambiguous bases; global 8,104,672 positions with 2,252,035 conserved, plus CDS 2,830,380/1,286,562, upstream 2,694,225/528,812, and downstream 2,580,067/436,661.
- Baselines: conserved prevalence was 0.277869 global, 0.454555 CDS, 0.196276 upstream, and 0.169244 downstream.
- Pilot result: 76M loss AUPRC was 0.535 global, 0.710 CDS, 0.336 upstream, and 0.240 downstream; the 46M-to-76M delta was above prevalence in every scope.
- Full absolute-loss result: AUPRC increased monotonically from 46M to 4B in every scope, reaching 0.723 global, 0.857 CDS, 0.522 upstream, and 0.428 downstream at 4B.
- Full entropy result: 4B AUPRC was 0.731 global, 0.867 CDS, 0.533 upstream, and 0.434 downstream; entropy was slightly better than loss except at the smallest downstream sizes.
- Full loss-delta result: the best pair was 46M-to-1B globally at 0.569, 46M-to-255M in CDS at 0.702, and 46M-to-4B upstream/downstream at 0.436/0.400. The practical 46M-to-76M AUPRC was 0.429/0.603/0.260/0.214 global/CDS/upstream/downstream and was weaker than either absolute small-model score.
- Orientation result: all 352 FWD-only and RC-only metrics were finite; FWD-versus-RC AUPRC differed by at most 0.0053 across scopes and scores, so neither direction has empirical priority.
- Half-compute result: the better single orientation retained 86–92% of the averaged 46M-loss lift over prevalence, 92–101% at 4B, and 71–75% for the 46M-to-76M delta.
- Genomic variation: 46M loss beat within-block prevalence in 99.7% of evaluable global/CDS 10-Mb blocks, 99.0% upstream, and 86.2% downstream; the 46M-to-76M delta did so in 100%, 100%, 98.1%, and 91.3%. These fractions describe variation and are not confidence intervals for pooled AUPRC.
- Durable token cache: `s3://oa-bolinas/snakemake/analysis/evals_v2/results/predictability_478/v1/atoms/{model}/{region}.{fwd,rc}.parquet` retains `window_id` plus the full per-token `nll` and `entropy_4nuc` vectors; `joined/{region}.parquet` retains the normalized labels and genomic metadata.
- Durable derived outputs: `classification/metrics.parquet` has 176 averaged rows, `orientation_metrics.parquet` has 352 single-orientation rows, separate block-metric Parquets preserve 10-Mb variation, manifests record the cache contract, and five SVGs are stored under `classification/`.
- Validation: 8 focused classifier/figure tests pass; scoped pre-commit hooks pass; the final dry-run schedules only the expected render rule; all figures were visually inspected.
- Cost: the on-demand `m7i.2xlarge` ran for about 42 minutes after Spot was unavailable, approximately $0.28. Combined issue cost is approximately $4.49 and remains below the $20 cap.
- Interpretation: loss and entropy are useful same-corpus conservation proxies among non-repeat positions, and larger models provide progressively stronger rankings. For a practical cheap proxy, one-orientation 46M loss or entropy is preferable to the two-model 46M-to-76M delta.
- Limitations: validation casing combines below-threshold with missing or unaligned positions; the result is in-corpus, cannot separate exposure or homology, and does not demonstrate out-of-distribution functional discovery or causal training benefit.
- Decision: promote CRP-005. Retain FWD/RC averaging for the highest-quality per-base weight, but use one deterministic orientation for half-compute pilots without claiming biological directionality.
- Next action: finish human plot iteration, then update issue #478 and interpretation PR #482 with the accepted classification result and durable links. CDS-only codon/splice classification remains optional secondary analysis.

### 2026-08-20 20:23 UTC - CRP-005 plot revision and compute comparison

- Feedback: rename the non-repeat loss figure, keep its legend outside the facets, shorten the loss-delta annotations, and use a symmetric zero-centered palette when lift is negative.
- Commit Hash: `d2b062ba`.
- Heatmap result: loss-delta AUPRC-minus-prevalence is displayed as integer percentage-point lift with a symmetric diverging scale; the categorical model labels remain legible without changing font sizes.
- Compute proxy: sum model parameter counts across the model and orientation passes required to score the same fixed 255-base window, normalized so one 46M single-orientation pass equals 1.
- Compute inputs: loss and entropy each require one model pass, loss delta requires both model passes, and FWD/RC averaging doubles the corresponding single-orientation cost.
- Scope: the comparison uses global AUPRC and shows every loss, entropy, and loss-delta candidate for one deterministic orientation and the FWD/RC mean.
- Limitation: relative parameter-passes are an estimated scoring-compute proxy, not measured FLOPs, runtime, or windows per hour.
- Durable output: `s3://oa-bolinas/snakemake/analysis/evals_v2/results/predictability_478/v1/classification/compute_efficiency_auprc.svg`.
- Validation: 8 focused tests pass, scoped pre-commit checks pass, the Snakemake dry-run schedules only the render rule, the render-only target uploaded all six SVGs, and the revised PNGs were visually inspected.
- Interpretation: absolute loss and entropy dominate the loss-delta cloud at comparable compute in the global plot; the cheap frontier begins with one-orientation small-model scores.
- Next action: obtain human feedback on the three revised figures before integrating the accepted plot order and classification interpretation into PR #482.

### 2026-08-20 20:28 UTC - CRP-005 FWD-only efficiency frontier

- Decision: use FWD-only scores for the throughput-versus-performance comparison because FWD and RC have nearly identical classification AUPRC and averaging doubles scoring cost.
- Commit Hash: `dc860b2b`.
- Result: the global scatter now contains the 44 FWD loss, entropy, and loss-delta candidates without the redundant FWD/RC-mean points or orientation legend.
- Compute proxy: one 46M FWD pass equals 1, each absolute score costs its model parameter count, and each loss delta costs the sum of its two model parameter counts.
- Durable output: `s3://oa-bolinas/snakemake/analysis/evals_v2/results/predictability_478/v1/classification/compute_efficiency_auprc.svg` was replaced with the FWD-only render.
- Validation: 8 focused tests and scoped pre-commit checks pass; the dry-run schedules only the render rule, the render completed, and the PNG was visually inspected.
- Interpretation: loss and entropy remain the practical frontier, while no loss-delta candidate improves AUPRC at comparable estimated scoring compute.
