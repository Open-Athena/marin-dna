---
topic: online-loss-selection
issue: https://github.com/Open-Athena/marin-dna/issues/515
description: Test online current-loss token selectors during a matched TSS-checkpoint continuation.
author: gonzalobenegas
---

# Online Loss Selection: Task Logbook

## Current TL;DR

Issue #515 is active on one Lambda A100; preprocessing, calibration, the 20-step canary, and the 100-step bridge are complete, and evaluation is resuming from the bridge after a device-placement repair.
The experiment will use the pinned glm-experiments code through a standalone Lambda/SkyPilot project because current Marin/Iris training does not provide the requested PyTorch selector and online evaluation path.
The primary endpoint is the pinned `marin-dna/evals_mendelian_traits` TSS-proximal train split on odd-numbered autosomes and chromosome X.
The per-species source-case audit passed without invoking the RefSeq fallback.

## Scope

- Goal: Test whether selecting the lowest-current-loss half of eligible nonrepeat targets improves promoter VEP learning speed at fixed processed-input compute.
- Primary metrics: TraitGym Mendelian promoter development-split AUPRC at the shared bridge, continuation midpoint, and endpoint; paired differences from uniform-100 conditional on the uniform sanity gate.
- Constraints: One paired seed, one Lambda A100, no more than $30 all-in or $28 GPU compute, 100 shared bridge steps, no more than 1,000 continuation steps per arm, no held-out even-autosome or chromosome-Y labels.
- Coordinating issue: https://github.com/Open-Athena/marin-dna/issues/515
- Experiment IDs: OLS-515-BRIDGE, OLS-515-U100, OLS-515-R50, OLS-515-L50, OLS-515-M50, and OLS-515-H50.
- Logging: CSV is authoritative; W&B is optional best effort and cannot stop the experiment.

## Baseline

- Date: 2026-08-23.
- Source checkpoint: gs://marin-us-east5/checkpoints/dna-exp232-zoonomia-v1-0p25b-v4_tss_region_and_utr5-v0.1-rerun-d08452/hf/step-2000.
- Source model contract: Qwen3 causal LM, 254,851,968 parameters, 12 layers, width 1,152, nine attention heads, seven-token vocabulary, one BOS plus 255 nucleotide tokens.
- Training corpus: marin-dna/zoonomia-v1-v4_tss_region_and_utr5 at 80b44bf6129d6ec7988f8cf1b706e4b1464ec9dc.
- Vendored upstream: Open-Athena/glm-experiments at b46cf87c2926201473797f9b00c13e1781c16403.
- Baseline numbers: Pending the shared bridge evaluation.

## Hypothesis Queue

### Active

- OLS-515-L50: Selecting the lowest-current-loss half improves promoter VEP learning speed relative to uniform loss and a random half-mask.
- OLS-515-M50: Selecting the centered half avoids both mastered and unstable targets and outperforms the low- and high-loss tails.
- OLS-515-H50: Current high loss identifies gradient-rich targets that outperform easy-token selection.
- OLS-515-R50: Any masked-arm gain is explained by sparse-gradient noise rather than the loss ranking.

### Blocked

- None.

### Falsified / Dead End

- None.

### Promoted

- None.

## Decision Log

- 2026-08-23: Use standalone Lambda/SkyPilot execution rather than Marin/Iris because issue #515 requires a pinned PyTorch Lightning and online TraitGym path, while issue #358 records that current Marin routes VEP offline.
- 2026-08-23: Use odd-numbered autosomes and chromosome X for every development VEP checkpoint and leave even-numbered autosomes and chromosome Y untouched.
- 2026-08-23: Use `marin-dna/evals_mendelian_traits` revision `4aed58e50c5dea0b878a665007af2ef9e5108e9f`, split `train`, subset `tss_proximal`, as the primary endpoint.
- 2026-08-23: The user explicitly authorized use of TraitGym test labels if needed; the newer pinned Mendelian endpoint makes that fallback unnecessary.
- 2026-08-23: Store retained experiment checkpoints and issue-scoped outputs under s3://oa-bolinas/issues/515/online-loss-selection/v1, authoritative dense metrics as CSV in the same run record, and the small audit summary on the permanent branch.
- 2026-08-24: The user authorized a single Lambda A100 and raised the all-in budget ceiling to $30; reserve $2 by stopping estimated GPU compute at $28.

## Negative Results Index

- None.

## Background Research Brief

- Effort: Low.
- Stop rule: Stopped after the current issue, the issue #479 standalone Lambda precedent, issue #358's current Marin launch direction, the pinned upstream code, source-checkpoint metadata, and dataset schemas fixed the implementation and launch path.
- Date: 2026-08-23.

### Current Marin Context

- Issue #479 established commit-pinned standalone PyTorch Lightning execution on one Lambda GH200 at $2.29/hour with SkyPilot --down, exact preflight gates, resume checkpoints, and runtime cost projections.
- Issue #358 routes current Marin VEP evaluation to offline evals_v2, which conflicts with #515's registered online checkpoint gate.
- The source checkpoint contains a Qwen3 model, tokenizer, and safetensors weights but no transferable Lightning optimizer state.

### External And Data Context

- The pinned Zoonomia TSS dataset has one train split, 11,281,780 255-base rows, a species field covering the frozen 108-species family cohort, a sequence field, and source-assembly soft masking preserved in sequence case.
- The pinned biofoundation CLM transform requires an even number of genomic bases, so the vendored project needs a local 255-base transform to produce the checkpoint's 256-token BOS-plus-sequence input without changing the dependency.
- The upstream glm-experiments data module writes lowercase flags at token position zero and therefore does not account for the checkpoint tokenizer's leading BOS.

### Recommended Next Experiment

- Minimum experiment: Complete the deterministic local selector and resume tests, then run the actual-checkpoint GH200 canary and 100-step shared bridge.
- Baseline/control: uniform-100 from the shared bridge checkpoint.
- Expected signal: uniform-100 endpoint promoter AUPRC exceeds bridge AUPRC before any masked arm launches.
- Falsifier: A non-finite canary, failed checkpoint/logit parity, or endpoint AUPRC no greater than bridge AUPRC.
- Cost/risk: Stop GPU compute at $18 and retain $2 for artifact transfer and storage uncertainty.

## Entry Log

### 2026-08-23 02:00 - Prologue and source verification

- Hypothesis: The issue can use the standalone Lightning/SkyPilot pattern from issue #479 while preserving the pinned glm-experiments model and evaluation interfaces.
- Commit Hash: `4c59ba060b575afd313b7df604da6ffbde1f3ca1`.
- Command: gh issue view 515; gcloud storage ls gs://marin-us-east5/checkpoints/.../hf/step-2000/**; sky check lambda; sky show-gpus --cloud lambda GH200.
- Config: Lambda GH200 at the listed $2.29/hour rate; source checkpoint and required local GCS, Hugging Face, and W&B credentials are present.
- Result: The checkpoint contains model config, safetensors, tokenizer, and special-token metadata; Lambda is enabled; no issue #515 cluster is running.
- Interpretation: No-cost implementation and validation can proceed, followed by a commit-pinned paid canary under the approved $20 cap.
- Next action: Implement the selector, BOS-aligned repeat eligibility, deterministic data and resume contracts, bounded species audit, cost guard, and development-only TraitGym evaluation.

### 2026-08-23 22:12 - Source-case audit and endpoint verification

- Hypothesis: Source-assembly case is intact enough across all represented species to use the same-data Zoonomia continuation without invoking the preregistered RefSeq fallback.
- Commit Hash: `4c59ba060b575afd313b7df604da6ffbde1f3ca1`.
- Command: `uv run --locked exp515-audit --output ../../.agents/artifacts/515-online-loss-selection/case-distribution-audit.json`.
- Config: Seed 1,515; 128 sequences per species; 108 expected species; pinned Zoonomia revision `80b44bf6129d6ec7988f8cf1b706e4b1464ec9dc`.
- Result: The audit covered all 108 species after 19,842 streamed rows, flagged no species, and set `fallback_required=false`.
- Result: Per-species lowercase-base fraction ranged from 0.64% to 19.72% with a 2.55% median, maximum all-lowercase-sequence fraction was 5.47%, and median eligible targets ranged from 213 to 255.
- Result: The current canonical `marin-dna/evals_mendelian_traits` revision is `4aed58e50c5dea0b878a665007af2ef9e5108e9f`; its TSS-proximal train endpoint contains 2,050 variants, 205 positives, and 205 match groups on odd autosomes and X.
- Interpretation: The primary Zoonomia pilot and the newer Mendelian development endpoint satisfy the registered pre-paid data gates.
- Next action: Commit and publish the implementation snapshot, dry-run the commit-pinned Sky task, then launch the approved one-GH200 canary.

### 2026-08-23 22:14 - Local validation boundary

- Hypothesis: The issue-specific selection, coordinate, storage, resume, and publication contracts pass independently of unrelated vendored test debt.
- Commit Hash: `4c59ba060b575afd313b7df604da6ffbde1f3ca1`.
- Command: `uv run --locked pytest tests/test_selection.py tests/test_exp515.py`.
- Result: All 17 issue-specific tests passed.
- Command: `uv run --locked pytest`.
- Result: The full vendored suite reported 145 passed, 17 failed, and 4 skipped.
- Interpretation: Remaining failures are outside the paid path: four CPU FlexAttention-backward tests unsupported by Torch, tests requiring the absent local `data/gpn-animal-promoter-dataset` fixture, Lightning-version configuration assertions, one stale default-evaluation assertion, and one internally contradictory stochastic weighted-loss assertion.
- Next action: Run the clean-commit launch preflight and remote issue-specific tests before the GPU canary.

### 2026-08-23 22:28 - First Lambda capacity attempt

- Hypothesis: The clean commit-pinned task can acquire one Lambda GH200 and begin the remote canary.
- Commit Hash: `b79b86375d30150112315d3cb6dafc0bf78bec10`.
- Command: `uv run --locked python launch.py --commit b79b86375d30150112315d3cb6dafc0bf78bec10 --run-id b79b86375d30-20260823t2229z --execute --retry-until-up`.
- Result: Three allocation attempts returned `insufficient-capacity`; no Lambda instance was created, the asynchronous request was canceled, and the terminated `INIT` record was removed.
- Interpretation: Estimated GPU spend remains $0, and later checks should use Lambda's read-only instance catalog at a multi-minute cadence.
- Next action: Launch the same experimental plan after `gpu_1x_gh200` advertises capacity.

### 2026-08-23 22:40 - Selector hot-path optimization

- Hypothesis: Vectorizing per-sequence ranking and threshold summaries preserves the registered masks while avoiding a host synchronization for every sequence.
- Commit Hash: `b076518b49035ea93dd7b00b29583ca86e256c39`.
- Command: `uv run --locked pytest tests/test_selection.py tests/test_exp515.py`; bounded 100-call CPU benchmark on 256 by 255 tensors.
- Result: All 18 issue-specific tests passed, Ruff check and format passed, and 100 full middle-rank microbatches took 0.305 seconds on CPU.
- Interpretation: Stable per-sequence ranking, lower-position tie breaks, empty rows, random-stream resume, and differentiable empty loss are preserved without the millions of per-row GPU synchronization points implied by the first implementation.
- Next action: Publish the optimization snapshot and restart the sparse capacity watcher on the new clean commit.

### 2026-08-24 00:22 - A100 protocol change

- Hypothesis: One available 40 GB Lambda A100 can run the registered matrix after automatic microbatch calibration while the larger budget preserves the paired design.
- Commit Hash: `410b16cc86a3587a6f0cf21ac95b67a11cfa30ec`.
- Command: Lambda read-only catalog query; `uv run --locked pytest tests/test_selection.py tests/test_exp515.py`; Ruff check and format.
- Config: One `A100:1` at $1.99/hour; $28 GPU compute stop; $30 all-in cap; largest power-of-two microbatch below 85% HBM; effective batch 2,048.
- Result: `gpu_1x_a100_sxm4` advertised capacity in `us-east-1`, `us-west-2`, and `asia-south-1`.
- Result: All 20 issue-specific tests passed in 3.86 seconds with 1,035,992 KiB peak RSS; Ruff check and format passed after import sorting.
- Interpretation: The available A100 removes the GH200 capacity blocker without changing the data, optimizer, selector, evaluation, gate, or paired-arm protocol.
- Next action: Commit and push the logbook snapshot, update issue #515, and launch the exact clean commit.

### 2026-08-24 01:31 - A100 bridge complete and evaluator repair

- Hypothesis: The completed preprocessing and step-100 bridge can be reused exactly while moving Lightning-detached models back to CUDA for each Mendelian checkpoint evaluation.
- Commit Hash: `f07aea7c880a7cad994e5601b58a6b45eb01922f`.
- Config: Lambda `gpu_1x_a100_sxm4` in `us-east-1`, 40 GB A100, selected microbatch 128, effective batch 2,048, W&B disabled, CSV authoritative, run `f34f5e359311-20260824t0025z`.
- Result: Remote issue-specific tests passed; the fixed sequence plan contains 2,252,800 rows; source checkpoint download and exact smoke tests passed; the 20-step canary sustained about 60,111 input tokens/second at 79.85% peak HBM; the 100-step bridge sustained about 59,817 input tokens/second and saved a valid full checkpoint.
- Result: The pinned Mendelian TSS-proximal frame and GRCh38 reference were materialized once and cached before 01:08 UTC.
- Result: Evaluation then consumed CPU because Lightning detached the trained module to CPU after `Trainer.fit`; this was an evaluator device-placement defect, not additional preprocessing.
- Result: The repair explicitly moves every evaluation model to CUDA, validates the passing smoke record, canary/bridge metadata, microbatch, and step-100 checkpoint before a bridge resume, and preserves the original instance start time in the hard budget guard.
- Validation: All 22 issue-specific local tests passed in 3.89 seconds with 1,033,324 KiB peak RSS; Ruff check, format, and `git diff --check` passed.
- Interpretation: No preprocessing, calibration, canary, or bridge work needs to be repeated; only the interrupted bridge evaluation and downstream gated arms resume on the same A100.
- Next action: Publish the repair snapshot, stop the CPU-bound evaluator without terminating the instance, and relaunch the same run ID from the validated step-100 bridge.

### 2026-08-24 13:08 - Exp58 CDS seven-arm gate preregistration

- Hypothesis: The qualitative ranking of current-loss selectors can be tested with higher AUPRC precision on a CDS-specialized model, while frozen final-checkpoint teacher arms distinguish hard token selection from self-distillation.
- Source: `exp58-animals-r01-1e3682` HF step 1,000; Qwen3 0.6B-class configuration, 256 raw nucleotide tokens, vocabulary size 6, and no BOS or EOS.
- Teacher: The compatible final HF export at step 16,999 from the same run and tokenizer.
- Data: `marin-dna/genomes-v4-genome_set-animals-intervals-v5_256_128` revision `04d374450a0f78f0ab5e17a8bc7b7c4baeb8295c`; original exp58 animal-CDS corpus; lowercase targets excluded in every arm.
- Schedule: Fresh AdamW; one shared 100-step uniform warmup from 1e-5 to 1e-3; then constant 1e-3 for exactly 100 steps in every arm; effective batch 2,048; all arms consume the same 204,800 post-bridge rows.
- Arms: uniform 100%; random 50%; current-student low, middle, and high 50%; pure `KL(teacher || student)` over every eligible target at temperature 1; and hard CE on the frozen teacher's lowest-NLL 50% targets.
- Evaluation: Pinned Mendelian train split, pooled missense plus splicing AUPRC as primary; 8,990 rows, 899 positive match groups, one positive and nine negatives per group; report missense and splicing separately.
- Decision: Synonymous variants are excluded by user direction because their AUPRC is too noisy.
- Stop: Evaluate the shared bridge and all seven step-100 arm checkpoints, compute paired match-group comparisons to bridge with Holm adjustment across the six nonuniform arms, and return for a continuation decision without running further steps.
- Validation: 30 focused issue tests passed in 4.06 seconds, including no-BOS alignment, pure teacher KL gradients, teacher-low selection, checkpoint compatibility, retained-cluster launch behavior, resume state, and statistical-gate contracts; peak local RSS was 1,037,220 KiB.
- Budget: Retain the existing Lambda A100 at $1.99/hour; preserve the original instance-start clock and $48 GPU stop under the authorized $50 all-in cap; CSV remains authoritative and W&B optional.
- Next action: Publish this amendment to issue #515, snapshot the clean implementation, run remote parity and HBM calibration, and launch only if the measured full gate remains below the compute stop.

### 2026-08-24 14:31 - Objective-transition amendment after bridge

- Observation: The user clarified that every arm is a distinct objective, so inheriting uniform-CE Adam moments and switching nonuniform objectives directly at 1e-3 would favor uniform and confound the shallow gate.
- Result: Run `364abd024f3c-20260824t1320z` completed its 409,600-row CDS plan, teacher-KL memory and timing canary, shared 100-step bridge, and bridge evaluation before being canceled during the partial uniform arm; no nonuniform arm had begun.
- Bridge: Microbatch 32; 2,234.36 seconds for 100 hard-CE steps; peak HBM 47.81%; pooled missense-plus-splicing AUPRC 0.156796; missense AUPRC 0.126489; splicing AUPRC 0.219649.
- Data: The exact plan contains 11.740% lowercase bases after skipping 1,103 fully lowercase windows; sequence checksum `d3df0a5bc8fd980d503bc48e2597efd586163c2803e0ea4724e6357fa56bcb0c`.
- Decision: Reuse the complete bridge model weights and post-bridge row offset, discard the partial uniform arm, reset AdamW and the scheduler independently for all seven arms including uniform, and apply the same 20-step warmup from 1e-5 to 1e-3 followed by 80 constant steps.
- Rationale: The original exp58 configuration used learning rate 1e-3 at effective batch 2,048, so the plateau remains lineage-consistent; the amendment removes stale objective-specific optimizer moments and the abrupt peak-LR transition.
- Next action: Validate, post the amendment, snapshot the code, archive the partial arm evidence, and resume from the retained bridge on the same A100.
