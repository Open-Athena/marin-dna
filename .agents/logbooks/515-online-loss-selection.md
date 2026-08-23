---
topic: online-loss-selection
issue: https://github.com/Open-Athena/marin-dna/issues/515
description: Test online current-loss token selectors during a matched TSS-checkpoint continuation.
author: gonzalobenegas
---

# Online Loss Selection: Task Logbook

## Current TL;DR

Issue #515 is in implementation and no paid run has started.
The experiment will use the pinned glm-experiments code through a standalone Lambda/SkyPilot project because current Marin/Iris training does not provide the requested PyTorch selector and online evaluation path.
Development VEP uses odd-numbered autosomes and chromosome X only.

## Scope

- Goal: Test whether selecting the lowest-current-loss half of eligible nonrepeat targets improves promoter VEP learning speed at fixed processed-input compute.
- Primary metrics: TraitGym Mendelian promoter development-split AUPRC at the shared bridge, continuation midpoint, and endpoint; paired differences from uniform-100 conditional on the uniform sanity gate.
- Constraints: One paired seed, one Lambda GH200, no more than $20 all-in or $18 GPU compute, 100 shared bridge steps, no more than 1,000 continuation steps per arm, no held-out even-autosome or chromosome-Y labels.
- Coordinating issue: https://github.com/Open-Athena/marin-dna/issues/515
- Experiment IDs: OLS-515-BRIDGE, OLS-515-U100, OLS-515-R50, OLS-515-L50, OLS-515-M50, and OLS-515-H50.
- Shared W&B tags: OLS-515, issue-515, online-loss-selection, and tss-continuation.

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
- 2026-08-23: Store full experiment checkpoints and issue-scoped outputs under s3://oa-bolinas/issues/515/online-loss-selection/v1, dense metrics in W&B project marin, and small code/audit summaries on the permanent branch.

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
- Commit Hash: Pending the first implementation snapshot.
- Command: gh issue view 515; gcloud storage ls gs://marin-us-east5/checkpoints/.../hf/step-2000/**; sky check lambda; sky show-gpus --cloud lambda GH200.
- Config: Lambda GH200 at the listed $2.29/hour rate; source checkpoint and required local GCS, Hugging Face, and W&B credentials are present.
- Result: The checkpoint contains model config, safetensors, tokenizer, and special-token metadata; Lambda is enabled; no issue #515 cluster is running.
- Interpretation: No-cost implementation and validation can proceed, followed by a commit-pinned paid canary under the approved $20 cap.
- Next action: Implement the selector, BOS-aligned repeat eligibility, deterministic data and resume contracts, bounded species audit, cost guard, and development-only TraitGym evaluation.
