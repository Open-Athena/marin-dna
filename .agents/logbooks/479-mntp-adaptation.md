---
topic: 479-mntp-adaptation
issue: https://github.com/Open-Athena/marin-dna/issues/479
description: Full-attention MNTP adaptation pilot from the released 1B m5.1 checkpoint
author: gonzalobenegas
---

# Exp479 MNTP adaptation: research logbook

## Scope

- Goal: Run the issue #479 transferred-MNTP pilot and matched controls on one Lambda GH200 under a $50 total compute cap.
- Primary metrics: pooled and single-mask MNTP validation cross-entropy; left- and right-context dependence; odd-autosome/X Mendelian, complex-trait, and SGE VEP; single-orientation versus FWD+RC nucleotide-dependency agreement.
- Constraints: 1,000 training steps per trained arm; no gradient accumulation; 100-step checkpoints; 10%/70%/20% WSD; one seed; no labeled access to even-numbered autosomes or chromosome Y.
- Coordinating issue: [#479](https://github.com/Open-Athena/marin-dna/issues/479)
- Experiment ID prefix: `MNTP-479`
- W&B project/group: `marin` / `dna-exp479`

## Current TL;DR

The one-seed pilot and integrity audit are complete and technically valid.
All three trained arms finished 1,000 finite steps on a standalone Lambda GH200 path with no Marin or Iris dependency.
Transferred MNTP narrowly beat scratch on pooled and single-mask validation loss and acquired bilateral context use, but it did not improve any primary VEP endpoint over source CLM and did not exceed the no-adaptation control on both flank probes.
The audited continued-CLM arm progressively damaged the source checkpoint under its fresh optimizer and high registered peak learning rates.
The investigation remains open for a short low-learning-rate AdamW calibration before treating causal continuation as a reasonable control.
Do not propose the 10,000-step MNTP extension.
The compact audited result bundle is at [`issue-479-mntp-pilot-audited-result`](https://github.com/Open-Athena/marin-dna/tree/issue-479-mntp-pilot-audited-result/.agents/artifacts/479-mntp-adaptation), and the dense record is in the [W&B report](https://wandb.ai/gonzalobenegas/marin/reports/Issue-479-1k-step-MNTP-adaptation-pilot--VmlldzoxNzc2ODgyOQ).

## Current baseline

- Source checkpoint: [`marin-dna/marin-dna-exp135-m5.1@a73a5dcf`](https://huggingface.co/marin-dna/marin-dna-exp135-m5.1/tree/a73a5dcfb3d64b8941e7e7596c6e88ef77db3e7a), step 59,158, 1,120,772,224 parameters.
- Architecture: Qwen3, 19 layers, hidden size 1,920, intermediate size 7,680, 15 attention/KV heads, 256-token context.
- Vocabulary: `[PAD]`, `[UNK]`, `[BOS]`, A, C, G, T. The tokenizer lowercases input.
- Current Lambda list price: $2.29/GH200-hour before applicable tax, checked 2026-08-19.
- Completed pilot and audit list-price estimate: $24.7340 of the $50 cap; final cluster confirmed terminated.
- Odd-autosome/X labeled diagnostics only; no even-autosome or Y labels, predictions, effect measurements, or aggregate metrics were accessed.

## Hypothesis queue

### Active

- `MNTP-479-H4`: full-parameter causal fine-tuning from the released source weights can preserve or lower fixed-plan causal validation loss when AdamW uses an appropriately small learning rate.
  Next test: one 200-step arm at `1e-6`, with validation at steps 0, 1, 10, 25, 50, 100, and 200.

### Blocked

- None.

### Falsified / dead end

- `MNTP-479-H2` (strict control criterion): transferred MNTP used both flanks, but its left response did not exceed the full-attention/no-adaptation control. Evidence: [result bundle](../artifacts/479-mntp-adaptation/README.md).
- `MNTP-479-H3` (downstream gate): no primary VEP endpoint improved over source CLM FWD+RC. Single-pass dependency structure remained similar to FWD+RC, but that scoped mechanism did not rescue the registered VEP/extension gate. Evidence: [W&B report](https://wandb.ai/gonzalobenegas/marin/reports/Issue-479-1k-step-MNTP-adaptation-pilot--VmlldzoxNzc2ODgyOQ).

### Promoted

- `MNTP-479-H1` (exploratory, one seed): transferred MNTP reached lower step-1,000 pooled loss (0.397270 versus 0.399543) and single-mask loss (0.310077 versus 0.313152) than scratch. Evidence: [validation figure](../artifacts/479-mntp-adaptation/figures/validation-loss.svg).

## Decision log

- 2026-08-19: Use a Lambda GH200 sidecar. Do not introduce Marin/Iris launch dependencies.
- 2026-08-19: Preserve the released checkpoint's untied input embedding and LM head. Initialize the new `[MASK]` input row and output row separately from their respective A/C/G/T means. Tying the matrices would alter the source model before adaptation and invalidate causal-logit parity.
- 2026-08-19: Do not add runtime reverse-complement augmentation. The three genomes-v5 datasets state that reverse complements are included, and the two pinned Zoonomia partitions inherit the m5.1 source construction. The experiment samples the pinned stored examples directly.
- 2026-08-20: Call the pilot technically valid: all registered smoke tests passed and every trained arm completed with finite loss, gradients, and optimizer state.
- 2026-08-20: Do not propose the 10,000-step extension. The strict bilateral-control and source-VEP criteria were not met.
- 2026-08-20: Do not claim single-orientation VEP support for any task. FWD stayed within one AUPRC point of transferred FWD+RC, but failed the required source-improvement gate.
- 2026-08-20: Keep final checkpoints and per-variant scores private; publish compact metrics, uncertainty, runtime, figures, and provenance on the permanent branch and W&B.
- 2026-08-20: Treat complete-flank ablation and ±64-base window shifts as post-hoc, non-gating diagnostics because their exact parameterization was fixed after primary evaluation.
- 2026-08-21: Keep the completed pilot result, but leave the investigation open for a short causal-continuation calibration.
- 2026-08-21: Treat the registered continued-CLM arm as evidence about its fresh high-learning-rate optimizer recipe, not as evidence that reasonable causal fine-tuning must degrade.
- 2026-08-21: Sweep full-parameter AdamW at `1e-6`, `3e-6`, `1e-5`, and `3e-5` for 200 steps before funding another 1,000-step arm.
- 2026-08-21: Sequence the calibration arms and start only `1e-6`; review its validation trajectory before choosing any other learning rate.

## Negative results index

- The strict bilateral-context criterion failed: transferred exceeded the no-adaptation control on the right but not the left.
- Transferred MNTP did not improve Mendelian macro, complex-trait global, or SGE accession/consequence macro AUPRC over source CLM FWD+RC.
- No VEP task passed the single-orientation gate because transferred FWD did not exceed source CLM FWD+RC.
- A 10,000-step extension is not proposed from this one-seed pilot.
- The registered continued-CLM recipe progressively increased fixed-plan loss from 0.23138 at step 0 to 0.35965 at step 800 before partial cooldown recovery to 0.35010 at step 1,000.
- No low-learning-rate causal fine-tuning control has run yet.

## Background research brief

- Effort: medium.
- Stop rule: Stop when the local research question, issue protocol, pinned source implementations, released model metadata, and an adversarial implementation check no longer change the first experiment.
- Date: 2026-08-19.

### Question

Can 1,000 full-parameter MNTP steps cheaply convert the released mature causal m5.1 checkpoint into a model that uses both flanks and improves a registered representation or VEP diagnostic?

### Current MarinDNA context

The accepted [bidirectional-models research question](../../docs/research/questions/bidirectional-models.md) identifies sequential MNTP as the leading first test. Issue #479 fixes the objective, controls, schedule, hardware, budget, and evaluation gate. No MarinDNA conversion run exists.

### Internal prior work

- The pinned m5.1 recipe uses five uniformly sampled components, 255 nucleotide bases plus BOS, lowercase loss weight 0.01, and the DNA-calibrated CompletedAdamH heuristic.
- The released Hugging Face config records `tie_word_embeddings=false`. The issue's instruction to resize a tied matrix does not match the released artifact.
- The three genomes-v5 dataset cards state that reverse complements are included. Adding random RC augmentation would change the source distribution.
- `glm-experiments` supplies the per-sequence `Uniform(0, 1)` mask-rate precedent. It does not exclude special tokens, guarantee a target, shift MNTP labels, or normalize loss per sequence; exp479 must add those contracts.

### External prior art

- [LLM2Vec](https://arxiv.org/abs/2404.05961) combines bidirectional attention with MNTP and shows that a short conversion phase can produce useful text encoders. It does not isolate MNTP from ordinary MLM in DNA or test full-parameter conversion of a mature genomic checkpoint.
- [Training Compute-Optimal Protein Language Models](https://arxiv.org/abs/2411.02142) reports successful sequential CLM-to-MLM transfer in protein models. Its masked phase consumed most of the transferred run's tokens, so it is weak evidence for a 1,000-step budget.
- Current [Transformers attention documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/attention_interface.md#bidirectional-attention) exposes `is_causal=False` for decoder models. A behavioral right-flank and gradient test remains necessary because backend or model integration bugs can silently preserve or remove causality.
- [Lambda pricing](https://lambda.ai/pricing) lists one 96 GB GH200 at $2.29 per GPU-hour before tax on 2026-08-19.

### Negative / failed leads

- Marin's current ArtifactStep/Iris migration is irrelevant to this sidecar experiment.
- The released checkpoint cannot be treated as tied without changing its stored model definition.
- No external result establishes that 1,000 full-parameter MNTP steps are sufficient for DNA, or that a bidirectional checkpoint should eliminate the RC ensemble.

### Evidence map

#### Claim: MNTP is the best first cheap-transfer objective

- Support: LLM2Vec preserves the decoder's next-token output alignment while enabling both-sided attention.
- Contradictions: the closest biological transfer result used ordinary MLM and a much larger masked-training budget.
- Directness to MarinDNA: low to moderate; neither result uses DNA or the m5.1 architecture/data regime.
- Confidence: exploratory.
- Action: run transferred and scratch MNTP with matched samples, corruptions, optimizer, batch, and schedule.

#### Claim: backend configuration is insufficient evidence of bidirectionality

- Support: Transformers exposes a generic `is_causal` control, while model- and backend-specific code paths can differ.
- Contradictions: none that remove the need for a behavioral test.
- Directness to MarinDNA: high because the pilot uses Qwen3 through Transformers.
- Confidence: stable as an implementation requirement.
- Action: gate every paid arm on paired-logit and input-embedding-gradient tests under both attention modes.

### Recommended next experiment

#### 1. `MNTP-479-001`: local tiny-model preflight

- Minimum experiment: exercise corruption, label shift, loss reduction, WSD, AdamH parity equations, grouping, attention behavior, checkpoint round-trip, and interrupted resumption on a tiny Qwen3 config.
- Baseline/control: explicit causal attention on the same weights and inputs.
- Expected signal: causal right-flank invariance/zero gradient; full-attention sensitivity/nonzero gradient; exact resumed-versus-uninterrupted state.
- Falsifier: any contract fails or depends on an unpinned backend behavior.
- Cost/risk: local CPU only; no labeled data and no cloud spend.

### Source ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Issue #479 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/479 | Registered protocol and decision criteria | High | Coordinating issue |
| Bidirectional-models question | Research synthesis | `docs/research/questions/bidirectional-models.md` | Current hypothesis and prior work | High | Accepted on `main` |
| m5.1 model release | Model card/artifact | https://huggingface.co/marin-dna/marin-dna-exp135-m5.1/tree/a73a5dcfb3d64b8941e7e7596c6e88ef77db3e7a | Architecture, tokenizer, untied matrices, source step | High | Commit-pinned |
| exp135 m5.1 recipe | Marin code | https://github.com/marin-community/marin/blob/a41a83fdddfdef85a75e39b56c32949518e3f578/experiments/dna/exp135_bolinas_mix_sweep.py | Mixture, source weighting, scaling anchor | High | Commit-pinned |
| Levanter AdamH | Marin code | https://github.com/marin-community/marin/blob/a41a83fdddfdef85a75e39b56c32949518e3f578/lib/levanter/src/levanter/optim/adamh.py | PyTorch update equations and grouping intent | High | Commit-pinned |
| glm-experiments | External code | https://github.com/Open-Athena/glm-experiments/blob/b46cf87c2926201473797f9b00c13e1781c16403/glm_experiments/data/lm_datamodule.py | Diffusion-style corruption precedent | High | Missing exp479-specific guards |
| LLM2Vec | Paper | https://arxiv.org/abs/2404.05961 | MNTP transfer rationale | Moderate | Text-domain result |
| Compute-optimal protein LMs | Paper | https://arxiv.org/abs/2411.02142 | Biological CLM-to-MLM transfer | Moderate | Masked phase is not cheap |
| Transformers attention interface | Official docs/code | https://github.com/huggingface/transformers/blob/main/docs/source/en/attention_interface.md#bidirectional-attention | Runtime causal/full-attention control | Moderate | Pin and behaviorally test the release used |
| Lambda pricing | Official pricing | https://lambda.ai/pricing | $2.29/GH200-hour budget conversion | High | Recheck before launch |

## Entry log

### 2026-08-19 22:33 - Protocol and source audit

- Hypothesis: The registered pilot can be implemented without changing the released model definition or source sequence-orientation distribution.
- Commit hash: `c9d330e43c6624d5134cd60570462e36f6794761`.
- Command: GitHub/Hugging Face source inspection recorded in the source ledger; no training command run.
- Config: `MNTP-479-001`, local CPU preflight planned; remote device fixed to one Lambda GH200.
- Result: The released config is untied and the source datasets already contain the intended orientations. The sidecar will initialize two `[MASK]` rows and will not add runtime RC augmentation.
- Interpretation: These are protocol clarifications required for causal parity and matched data, not optional implementation details.
- Next action: Implement the self-contained Lightning project and tiny-model tests.

### 2026-08-19 22:54 - Local preflight snapshot

- Hypothesis: The registered corruption, loss, attention, optimizer, schedule, and resume contracts can be made deterministic before loading the 1B checkpoint.
- Commit hash: `c9d330e43c6624d5134cd60570462e36f6794761`.
- Command: `uv sync --locked --group dev`; `uv run --locked pytest -q`; `uv run --locked ruff check .`; `uv run --locked ruff format --check .` from `experiments/exp479_mntp_adaptation/`.
- Config: Transformers 5.15.1, Torch 2.11.0+cu128, Lightning 2.6.5, TorchData 0.11.0; tiny Qwen3 causal/full-attention tests on CPU.
- Result: 22 tests passed in 3.21 seconds; Ruff check and format check passed. Peak local pytest RSS was 840,612 KiB. Interrupted training resumed from step 2 and matched the uninterrupted step-4 model state exactly. The causal readout was invariant to a changed right flank with zero position-specific gradient; full attention changed the readout and produced a nonzero right-flank gradient.
- Interpretation: The local contracts are ready for the actual-checkpoint GH200 gate. No 1B checkpoint, cloud resource, or labeled evaluation has run.
- Next action: Push the snapshot, dry-run the commit-pinned Sky plan, update issue #479, then obtain explicit paid-launch approval.

### 2026-08-19 23:49 - Restart-safe pilot and diagnostics snapshot

- Hypothesis: The complete three-arm pilot, odd/X VEP suite, context probes, and fixed nucleotide-dependency panel can run as one restart-safe, self-terminating Lambda task without Marin or Iris.
- Commit hash: `5b3634c8c5e6f9dfc77a62e7965642977b38d708`.
- Command: Resource-guarded `uv run --locked ruff check .`, `ruff format --check .`, `python -m compileall`, and `pytest`; then `uv run --locked python launch.py pilot --commit 5b3634c8c5e6f9dfc77a62e7965642977b38d708 --execute --dry-run`.
- Config: One Lambda `gpu_1x_gh200` in `us-east-1`, 96 GB GH200, 512 GB disk, listed at $2.29/hour; Sky `--down`; $50 hard cap; private `marin-dna/marin-dna-exp479-mntp-m5.1` checkpoint staging.
- Result: All 37 tests passed in 5.23 seconds; Ruff, format, compile, and whitespace checks passed. Peak local pytest RSS was 1,051,008 KiB. The no-provisioning Sky dry run selected the registered single GH200 and validated the exact pushed commit, secrets interface, repository access, and teardown configuration.
- Interpretation: The code path is ready for the actual-checkpoint smoke tests and capped pilot. Full Lightning checkpoints are uploaded every 100 steps; a fresh instance resumes the newest private checkpoint; all trained arms log fixed flank probes; evaluation emits explicit FWD and FWD+RC runtime rows and paired odd/X summaries.
- Data boundary: Only one streamed row from each pinned public odd/X `train` split was used to validate schemas. No even-autosome/Y labels, predictions, effects, or aggregate metrics were accessed. No cloud resource was provisioned and no charge was incurred.
- Next action: Obtain explicit approval for the $50 Lambda cap and the private-staging model-card draft, then launch the exact reviewed commit and monitor through self-termination.


### 2026-08-20 00:05 - Actual-checkpoint preflight and dataset-stream fix

- Hypothesis: The remote smoke suite and actual 1B checkpoint will pass before the three training arms start.
- Launch commit: `8a01ea37cfbbe32b7bec7c0c6e1391096b4b7b72`; fix commit: `0c63a8b552ab057ff522eefb6b6e6db69da1a46a`.
- Command: `uv run --locked python launch.py pilot --commit 8a01ea37cfbbe32b7bec7c0c6e1391096b4b7b72 --model-card-reviewed --execute`.
- Config: One Lambda GH200 in `us-east-3` at $2.29/hour after `us-east-1` reported insufficient capacity. The user confirmed the registered $50 cap and private staging card through the standing instruction to continue to the goal.
- Result: The remote locked suite passed 37 tests. The actual m5.1 preflight passed with batch 512, 47,510,808,576 peak allocated bytes of 101,468,602,368, 53.18% headroom, 31,633 model tokens/s, exact causal parity, zero causal right-flank effect, nonzero full-attention effect, and an $18.90 total-cost projection.
- Failure: Data-plan materialization stopped before training step 1 because the environment lacked the zstd decoder. A local real-source check then found that validation repositories require split `validation`, and the enhancer/ncRNA validation rows use field `seq`.
- Fix: Added the zstd dependency, explicit split routing, the observed validation field names, and four regression tests. One real record from each of the five train and five validation sources now has length 255; the locked suite passes 41 tests.
- Cost/data boundary: Sky terminated the first instance after about five minutes; the listed-price estimate is below $0.21 and must be reconciled with the provider bill. No training step or aggregate labeled evaluation ran.
- Next action: Push and dry-run the fix commit, then relaunch the standalone pilot.

### 2026-08-20 00:26 - Full-trainer batch fallback

- Hypothesis: The preflight-selected batch of 512 will fit the complete Lightning training path, including its full optimizer and callback state.
- Launch commit: `b2b5b7cf90d8271e83578a526e3e10b5c828b1a2`.
- Command: `uv run --locked python launch.py pilot --commit b2b5b7cf90d8271e83578a526e3e10b5c828b1a2 --model-card-reviewed --execute`.
- Result: The corrected remote suite passed 41 tests, all ten data sources materialized, and the actual-checkpoint preflight repeated successfully. The preflight measured 31,475 model tokens/s at batch 512, 53.18% memory headroom, and an $18.98 projection including the $10 evaluation reserve. The first full Lightning training step then OOMed with 1.70 GiB free while requesting 1.88 GiB. No optimizer step completed and no checkpoint was produced.
- Interpretation: Lightning's complete resident state uses materially more memory than the isolated preflight loop. This is exactly the registered condition for reducing the no-accumulation batch after the first optimizer step OOMs.
- Fix: Add an explicit, validated maximum-batch-size pilot option and cap training and VEP batches at 256. This preserves the preflight gate, five-component balance, 1,000 optimizer steps, no gradient accumulation, and matched exposure across all three trained arms.
- Verification: The locked local suite passes 41 tests in 6.83 seconds; Ruff, format, and whitespace checks pass. Peak pytest RSS was 1,051,312 KiB.
- Cost boundary: The second instance ran from about 00:12 to 00:26 UTC, at most about $0.54 at the listed price. Together with the first attempt, failed-launch listed exposure remains below about $0.75; final provider billing still requires reconciliation.
- Next action: Snapshot and relaunch at batch 256, confirm the first optimizer step and step-100 private checkpoint, then continue through all registered evaluations and teardown.

### 2026-08-20 00:50 - Deterministic-backend preflight mismatch

- Hypothesis: Halving the full-trainer batch from 512 to 256 will provide adequate memory headroom under the otherwise unchanged run configuration.
- Launch commit: `36a09c18743b3acd9ed1bc1a4ab37b0f2f36fd32`.
- Result: The data manifest confirms batch 256 and 256,000 training rows, but the first full-trainer step again OOMed. It had 93.22 GiB allocated before requesting another 480 MiB in rotary-position attention. No optimizer step completed and no checkpoint was produced.
- Diagnosis: The actual trainer sets `deterministic=True`, which makes Lightning call `torch.use_deterministic_algorithms(True)`, disables cuDNN benchmarking, and sets `CUBLAS_WORKSPACE_CONFIG`. The batch-selection preflight did not set those flags, so it measured a different CUDA-kernel regime. This explains why its apparent 53.18% headroom did not reproduce and why treating the second failure as ordinary activation scaling was incorrect.
- Fix: Make the actual-checkpoint preflight set the same deterministic backend flags before any CUDA work, record the flag in its result, and again use its dynamic selected batch for both training and VEP. Retain the explicit maximum-batch-size option as a tested emergency cap, but do not impose an unmeasured hardcoded batch.
- Verification: The locked local suite passes 42 tests in 5.25 seconds; Ruff, format, and whitespace checks pass. Peak pytest RSS was 1,051,184 KiB.
- Cost boundary: This attempt ran from about 00:27 to 00:36 UTC, at most about $0.35 at the listed price. All failed-launch listed exposure remains below about $1.10; final provider billing still requires reconciliation.
- Data boundary: No training step, checkpoint, aggregate labeled evaluation, even-autosome label, or Y label was produced or accessed.
- Next action: Relaunch the corrected preflight, require it to reject the memory-heavy batches under the real deterministic backend, and proceed only with a selection retaining at least 10% measured headroom.

### 2026-08-20 01:05 - Replace approximate memory gate with exact Lightning step

- Hypothesis: Matching Lightning's deterministic backend flags in the hand-written optimizer preflight will reproduce full-trainer memory and reject batch 512.
- Diagnostic launch commit: `06adaea8604e9a8e060cce78428a2b03e66d7c47`.
- Result: The corrected preflight still measured exactly 47,510,808,576 allocated bytes and 53.18% headroom at batch 512. The deterministic-backend hypothesis was falsified. The job was cancelled immediately after the preflight result, before data-plan materialization or any training step, rather than knowingly repeating the OOM.
- Interpretation: The approximation itself is insufficient: it does not exercise the complete Lightning Trainer, precision-plugin closure, module logging, gradient clipping, scheduler, and hook lifecycle. Inferring another batch from it would not satisfy the registered memory gate.
- Fix: Add an isolated one-step `trainer-preflight` command using the production `AdaptationModule`, `ExperimentDataModule`, bf16-mixed precision, deterministic Trainer, gradient checkpointing, AdamH, clipping, and scheduler. It records resident memory before the closure, complete-step peak allocation/reservation, elapsed time, and status. Candidate processes start at batch 128 because 512 and 256 are empirical failures; each failed process exits before the next candidate so CUDA allocations cannot leak. Training and VEP consume only the first result with at least 10% headroom, and the result is uploaded to private staging.
- Allocator: Set expandable CUDA segments for both the exact probe and production run to avoid treating allocator fragmentation as model memory.
- Verification: The locked local suite passes 43 tests in 5.23 seconds; Ruff, format, whitespace, compilation, and both changed CLI help surfaces pass. Peak pytest RSS was 1,051,204 KiB.
- Cost boundary: The cancelled diagnostic ran from about 00:52 to 00:58 UTC, at most about $0.23 at the listed price. Cumulative failed/diagnostic list-price exposure remains below about $1.33; final provider billing still requires reconciliation.
- Data boundary: No optimizer step from a registered arm, checkpoint, or aggregate labeled evaluation was produced or accessed.
- Next action: Run the exact Lightning gate, use its measured batch and headroom, and require the first registered arm to reach the step-100 restart checkpoint before treating training as established.

### 2026-08-20 01:18 - Exact batch selection and all-N source row

- Hypothesis: The exact Lightning gate will identify a safe production batch and allow transferred MNTP to reach its first restart checkpoint.
- Launch commit: `ae8fa63c25cc187bb82b5cbce1b75cb9ba53fab3`.
- Batch result: The exact Trainer step OOMed at batch 128. Batch 64 passed with 4,483,565,056 bytes allocated before the step, 54,238,396,416 peak allocated bytes of 101,468,602,368, 46.55% headroom, and 6.94 seconds for the complete first optimizer step. Batch 64 is therefore the largest production-path candidate not already falsified by 512, 256, or 128.
- Training result: Transferred MNTP then ran at about 1.87 steps/s and logged through `trainer/global_step=31` at [W&B run rtdk8zn1](https://wandb.ai/gonzalobenegas/marin/runs/rtdk8zn1). The latest logged train loss was 1.20538 and accuracy was 0.27121. No step-100 checkpoint was reached.
- Failure: DataLoader worker 0 rejected batch row 51 while fetching the next batch because it had no eligible A/C/G/T target. The corresponding deterministic plan position is sample 2,099, ncRNA component output index 419.
- Reproduction: The pinned unlabeled ncRNA stream at seed 4 and shuffle buffer 10,000 yields a 255-character sequence whose alphabet is exactly `N` at source output index 419. This is a source-data contract edge case, not stochastic masking failure.
- Fix: Skip source rows containing no uppercase or lowercase A/C/G/T and draw the next row from the same component. This preserves uniform component counts, total exposure, deterministic output sample IDs, and the requirement that every sequence contribute a defined loss. The corrected real ncRNA output index 419 is 255 bases and contains an eligible target.
- Verification: The locked suite passes 44 tests in 5.19 seconds; Ruff, format, and whitespace checks pass. Peak pytest RSS was 1,051,424 KiB. The workflow README now states the filter.
- Cost boundary: The instance ran from 01:06:41 to about 01:17 UTC, at most about $0.40 at the listed price. Cumulative failed/diagnostic list-price exposure remains below about $1.73; final provider billing still requires reconciliation.
- Data boundary: Only pinned unlabeled training sequence was inspected for reproduction. No aggregate labeled evaluation, even-autosome label, or Y label was accessed.
- Next action: Snapshot and relaunch. Require the transferred arm to publish step 100 before considering restart safety exercised remotely.

### 2026-08-20 01:50 - Learning established; private checkpoint quota recovery

- Launch commit: `379ceca120f434ba535140c4ef71bdd911138147`.
- Training result: Transferred MNTP completed validation boundaries through step 800 with finite state. Step-800 pre-cooldown metrics were diffusion loss 0.39797, single-mask loss 0.31697, diffusion accuracy 0.33200, single-mask accuracy 0.40781, left-flank L1 0.01498, and right-flank L1 0.01215. The context probes grew from 0.00226/0.00123 at step 100, so bidirectional context use is established during adaptation.
- Durable records: Full step-100 through step-700 Lightning checkpoints are present in the original private staging repository. W&B run [6iqcmdm7](https://wandb.ai/gonzalobenegas/marin/runs/6iqcmdm7) retains every completed validation boundary through step 800.
- Failure: The step-800 upload transferred its 13.4 GB payload, then the Hugging Face commit endpoint rejected it because the private owner storage limit had been reached. Training did not fail numerically; publication stopped the job at the registered pre-cooldown boundary before the checkpoint commit.
- Recovery decision: Do not delete or rewrite the seven existing private checkpoints. Resume read-only from durable step 700 into a second private repository owned by the authenticated user. Continue saving every 100-step checkpoint on the Lambda task disk, but copy only steps 400 and 800 for later arms and the required transferred step 800 to spillover staging. Final exports and evaluations also go to spillover.
- Budget: This run lasted about 24 minutes, at most about $0.92 at the listed price. Conservative cumulative prior cost for the next hard guard is $2.70, covering all failed, diagnostic, and current attempts.
- Implementation: Add separate resume/publication repositories, selectable checkpoint upload steps, cumulative prior-cost accounting, and exception preservation. Existing private artifacts remain untouched.
- Data boundary: Only unlabeled validation metrics were produced. No aggregate VEP evaluation, even-autosome label, or Y label was accessed.
- Next action: Verify the recovery snapshot, create private personal spillover staging, resume from step 700, and complete the remaining arms and registered diagnostics under the same $50 cap.

### 2026-08-20 02:30 - Transferred completion and sequential W&B isolation

- Recovery launch commit: `44b117549c371f98a56a97ac165cbc4a5d5b66ca`.
- Transferred result: The arm resumed read-only from original-private step 700, published a full step-800 restart checkpoint to private spillover, completed step 1,000, and published its cooled Hugging Face export, manifest, and runtime record. Original-private artifacts remain unchanged.
- Logging defect: After transferred publication, Lightning's pinned `WandbLogger.finalize()` left the process-global W&B run open. Scratch therefore reused transferred recovery run [oddka8kk](https://wandb.ai/gonzalobenegas/marin/runs/oddka8kk) instead of creating its registered distinct run.
- Containment: Cancelled the task around scratch step 170, before its first selected restart upload at step 400, and explicitly tore down the Lambda cluster. No scratch checkpoint or export was published; the completed transferred arm remains durable and will be skipped on relaunch.
- Fix: Explicitly finish each successful arm's W&B run after final private publication and regression-test that boundary. This permits the next arm in the same process to initialize a distinct run and preserves one history per comparison arm.
- Budget: This recovery instance ran for about 12 minutes, at most about $0.46 at the listed price. Conservative cumulative prior cost for the next hard guard is $3.25.
- Data boundary: Only unlabeled training and validation data were used. No labeled VEP evaluation, even-autosome label, or Y label was accessed.
- Next action: Verify and snapshot the isolation fix, dry-run the exact commit, relaunch on Lambda with completed transferred publication skipped, and require distinct scratch and CLM W&B runs before accepting their histories.

### 2026-08-20 03:14 - Three arms durable; BGZF evaluation dependency

- Launch commit: `6f4d8d69d0d79d409d327c327b24d951d400f050`; Lambda GH200 `us-east-3` at $2.29/hour after `us-east-1` lacked capacity.
- Training: Private spillover state skipped the completed transferred arm. Scratch MNTP completed 1,000 steps in 14:54 at [W&B 4nstge1d](https://wandb.ai/gonzalobenegas/marin/runs/4nstge1d); CLM continuation completed 1,000 steps in 13:35 at [W&B yod8l3mb](https://wandb.ai/gonzalobenegas/marin/runs/yod8l3mb).
- Durable records: Both new arms have independently verified private full checkpoints at steps 400 and 800 plus final step-1,000 Hugging Face exports and manifests. The transferred final export remains independently verified in spillover.
- Isolation result: Scratch and CLM each created and finished distinct W&B runs. Scratch ended with nonzero left/right context probes (0.01149/0.01423); CLM ended with nonzero left and exactly zero right context influence (0.15719/0), matching the registered attention controls.
- Evaluation failure: Before scoring any variants, `pyfaidx` rejected the BGZF GRCh38 reference because its optional Biopython runtime dependency was absent. The Lambda task failed and the cluster was explicitly torn down.
- Fix: Declare and lock Biopython, and exercise the production reference-window path against an actual temporary BGZF FASTA in the locked test suite.
- Budget: This instance ran for about 39 minutes, at most about $1.50 at the listed price. Conservative cumulative prior cost for the next hard guard is $4.80.
- Data boundary: The public odd/X train-split files were materialized, but no variant was scored and no aggregate labeled metric was computed. No even-autosome or Y labels, predictions, effects, or aggregates were accessed.
- Next action: Verify and snapshot the BGZF dependency fix, then relaunch on Lambda with all three trained arms skipped and run the registered evaluations and final publication.

### 2026-08-20 06:15 - Pilot complete; negative VEP and no extension

- Hypotheses: `MNTP-479-H1` predicts an early transfer advantage over scratch; `MNTP-479-H2` requires bilateral context beyond both controls; `MNTP-479-H3` requires source-relative VEP or a scoped mechanistic gain sufficient for continuation.
- Code/result snapshots: primary evaluation code `e10b3e772d90d7c126769765b90ba49f2e8588d2`; diagnostics code `97a6e3c50080005ad4f93f2206c4155b8f5cb7b9`; compact artifacts `cb0d37ffa97361947fc01c434f670c747ca94af4`.
- Commands: final primary evaluation ran the commit-pinned `sky/pilot.yaml` path with all completed arms skipped; follow-up diagnostics ran `uv run --locked python launch.py diagnostics --commit 97a6e3c50080005ad4f93f2206c4155b8f5cb7b9 --hf-repo-id gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover --prior-cost-usd 9.5601726286 --execute`. Both used `sky launch --down`; the final cluster was confirmed absent with `sky status -r dna-exp479-gh200`.
- Config: one Lambda GH200 96 GB at $2.29/hour; batch 64; one seed; 1,000 steps per trained arm; 16,384,000 model tokens and 16,320,000 bases per arm; odd-autosome/X labeled development data only. Context/window follow-up used complete left/right `N`/`[UNK]` flank ablation and fixed ±64-base window shifts.
- Technical validity: all remote and local checks passed; the final suite has 55 tests. Transferred, scratch, and causal continuation all completed with finite state. Transferred ran 300 recovery steps in 250.3 seconds; scratch ran 1,000 steps in 922.1 seconds; causal continuation ran 1,000 steps in 841.2 seconds.
- Validation result: step-1,000 transferred versus scratch pooled loss was 0.397270 versus 0.399543; single-mask loss was 0.310077 versus 0.313152. Pooled accuracy was 0.334408 versus 0.333749; single-mask accuracy was 0.418750 versus 0.396875. `H1` is supported exploratorily, with small one-seed margins.
- Context result: matched VEP probe left/right L1 was source CLM 0.16545/0, no adaptation 0.02581/0.01280, transferred MNTP 0.02007/0.01988, scratch MNTP 0.01216/0.01381, and continued CLM 0.13638/0. Transferred uses both flanks but exceeds no adaptation only on the right, so strict `H2` fails.
- VEP result:

  | Primary endpoint | Source CLM FWD+RC | Transferred FWD | Scratch FWD | Continued CLM FWD+RC | Transferred FWD+RC |
  |---|---:|---:|---:|---:|---:|
  | Mendelian macro AUPRC | 0.3951 | 0.1151 | 0.1112 | 0.3064 | 0.1152 |
  | Complex-trait global AUPRC | 0.1342 | 0.1003 | 0.1018 | 0.1188 | 0.0996 |
  | SGE accession/consequence macro AUPRC | 0.3577 | 0.1427 | 0.1378 | 0.3052 | 0.1429 |

- Paired uncertainty: transferred FWD minus source FWD+RC was -0.2982 (95% CI -0.3213 to -0.2759) for the Mendelian global endpoint and -0.0339 (-0.0460 to -0.0221) for complex traits. Transferred minus scratch was +0.0021 (-0.0008 to +0.0052) and -0.0015 (-0.0049 to +0.0021), respectively. No downstream source-relative gain was observed.
- Single-orientation decision: transferred FWD stayed within one AUPRC point of its own FWD+RC on all three tasks, but failed the required source-improvement gate on all three. No task is supported.
- Post-hoc context/window result: ablating either complete flank reduced transferred raw-score Spearman correlation with centered scoring to 0.717–0.747 across datasets. ±64-base window shifts retained 0.991–0.993. Primary AUPRC changes remained small and did not become positive relative to source.
- Dependency-map result: the visually reviewed LDLR, TH, GRIA4, HBA1, and tRNA-Arg-TCT maps retained structure in both triangles. Single-orientation versus FWD+RC off-diagonal Spearman was 0.9649–0.9738, mean 0.9692.
- Runtime: transferred MNTP processed all 51,623 variants at 520.0 variants/s FWD and 259.8 variants/s FWD+RC with 3.28 GB peak CUDA allocation. Source CLM processed 265.1/133.0 variants/s with 4.29 GB peak.
- Cost: final conservative list-price estimate $10.2326 of the $50 cap, including failures, recovery, training, primary evaluation, and follow-up diagnostics. Provider billing still requires external reconciliation.
- Boundaries: no even-autosome or Y labels, predictions, effect measurements, or aggregate metrics were accessed. HBA1 chromosome-16 use was limited to unlabeled reference sequence for its preregistered dependency map.
- Artifacts: [compact bundle](../artifacts/479-mntp-adaptation/README.md), [W&B report](https://wandb.ai/gonzalobenegas/marin/reports/Issue-479-1k-step-MNTP-adaptation-pilot--VmlldzoxNzc2ODgyOQ), [clean analysis run](https://wandb.ai/gonzalobenegas/marin/runs/xe7qj1c3), and private W&B evaluation artifact `dna-exp479-evaluation`.
- Interpretation: The objective conversion works behaviorally and transfers a small early optimization advantage, but this 1,000-step checkpoint is not a useful source-relative VEP model. Dependency similarity and window stability are scoped mechanisms, not evidence for extending training.
- Decision: pilot technically valid; no 10,000-step extension; no single-orientation VEP support; keep negative result and private checkpoints durable.
- Next action: update the active research question, final model card, coordinating issue body/comment, and seal an annotated result tag.

### 2026-08-20 08:15 - Checkpoint-integrity follow-up prepared

- Trigger: The continued-CLM VEP decline is surprising enough that the final endpoints alone do not distinguish progressive forgetting from serialization, resume, coordinate, tokenization, or readout bugs.
- Zero-update control: Load the pinned source checkpoint, save it in the Hugging Face format used by evaluation, reload it, and compare token-level contracts plus every odd/X per-variant score with direct source loading.
- Early replay: Rebuild the exact batch-64 train and validation plans, require their original hashes, and replay CLM steps 1, 5, 10, 25, 50, 100, 200, and 400 under the original 1,000-step schedule. Compare replayed step 400 row-for-row with the original full Lightning checkpoint.
- Alignment gates: Check PAD/UNK/BOS/EOS/MASK IDs, vocabulary sizes, 256-token length, attention masks, and the input-position-to-output-position shift at nucleotide indices 0, 63, 127, 191, and 254 in FWD and RC. Compare training-collator logits with inference LLRs and require invariance to the true nucleotide hidden under MASK.
- Coordinate gates: Independently reconstruct 0-based half-open windows at three positions per odd/X dataset and at variant indices 63, 127, and 191.
- Trajectories: Recompute fixed-plan loss from source, replayed early checkpoints, every retained Lightning checkpoint, and final exports. Score both orientations on all three primary odd/X endpoints and plot AUPRC versus optimizer steps.
- Dependency control: Plot the raw directed forward and aligned RC matrices for all five loci. Recompute tRNA-Arg-TCT under full attention and under a forced-causal negative control; the forbidden causal triangle must remain zero.
- Publication: Keep per-variant scores and numeric maps private. Publish compact tables and figures to W&B and GitHub. Upload failing parity diagnostics before exiting nonzero.
- Compute boundary: Use one self-terminating Lambda GH200 at $2.29/hour, carry forward the conservative $10.2326 prior cost, and retain the original $50 hard cap. No Iris or Marin runtime dependency is introduced.
- Verification before launch: the locked experiment suite passes 61 tests; Ruff, formatting, and whitespace checks pass. Peak pytest RSS was 1,074,164 KiB.
- Next action: snapshot the exact audit commit, dry-run Sky provisioning, launch, monitor through upload and teardown, inspect every rendered figure, and publish the evidence to issue #479.

### 2026-08-20 19:15 - Checkpoint-integrity audit completed

- Trigger: Continued-CLM VEP degraded unexpectedly, so the result required positive evidence against serialization, replay, coordinate, tokenization, readout-shift, loss-path, and optimization-instability bugs.
- Audit snapshots: checkpoint/stability preparation `0f780cf3a260e5ceadc43372bf3bf16284e313cc`; vectorized final-dependency implementation `173a95d82bb3e8b6ae25130c0f7ca404f68acce1`; capacity-retrying launch `872a68b997d45eceec96d0d56c705ee327428af7`.
- Commands: commit-pinned `audit`, `stability`, and focused `dependency` stages on self-terminating Lambda GH200 instances. The final focused launch used `uv run --locked python launch.py dependency --commit 872a68b997d45eceec96d0d56c705ee327428af7 --hf-repo-id gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover --prior-cost-usd 23.572794793431264 --retry-until-up --execute`.
- Checkpoint result: direct source scoring and zero-update Hugging Face save/reload were bit-exact across all 51,623 odd/X variants and both strands. Replayed CLM step 400 was bit-exact to the original Lightning checkpoint. Deterministic 400-step replays of transferred MNTP, scratch MNTP, and continued CLM matched every original W&B per-step loss exactly.
- Alignment result: one BOS plus 255 nucleotides, no EOS, PAD/UNK/BOS IDs 0/1/2, canonical IDs 3–6, and MNTP MASK ID 7. Training and inference both read nucleotide `i` from output `i - 1`; indices 0, 63, 127, 191, and 254 passed in FWD and RC with zero score error. MNTP logits were invariant to the hidden true base under MASK. Twenty-seven independent 0-based, half-open coordinate anchors passed.
- Validation/loss result: exactly five sources—CDS, downstream, enhancer, ncRNA, and upstream—contribute 128 fixed sequences each. Lowercase repeat bases have weight 0.01 versus 1.0 for uppercase in both training and validation. Recomputed original validation points matched W&B within 0.000319.
- Progressive CLM result: fixed-plan validation loss was 0.23138 at steps 0 and 1, 0.23131 at 10, 0.24005 at 50, 0.27310 at 100, 0.30652 at 200, 0.33297 at 400, 0.35965 at 800, and 0.35010 at 1,000. Mendelian FWD+RC AUPRC was 0.39553 at source, 0.39555 at step 1, 0.39289 at 10, 0.38241 at 100, 0.32686 at 400, 0.26384 at 800, and 0.30681 at 1,000. The decline is progressive under warmup/peak learning rate with partial cooldown recovery, not an immediate serialization failure.
- Stability result: continued-CLM gradient norm median/p95/max was 0.791/1.002/1.263 with 23/400 clipped steps and no post-warmup loss spikes. Transferred MNTP was 1.259/11.115/60.517 with 275/400 clipped; scratch MNTP was 2.118/28.263/84.999 with 296/400 clipped. Both MNTP transients decayed rapidly; none of the three arms showed sustained divergence.
- Diagnostic bug: the original dependency code compared a batch-one baseline with batch-1,020 substitutions, so BF16 batch-shape kernel differences contaminated the map. This bug affected the diagnostic only, not training or inference. The corrected code evaluates the baseline and substitutions in the same model call and has a causal future-context zero regression test.
- Final-checkpoint dependency result: on reference-orientation tRNA-Arg-TCT, transferred MNTP past/future mean was 0.05314/0.05334 with maxima 1.55953/1.60974; scratch MNTP mean was 0.03056/0.02917 with maxima 1.41963/1.41369; continued CLM mean was 0.12510/0 with maxima 2.92341/0. All five gates passed, including an exactly zero CLM future triangle.
- Inference recheck: transferred and scratch final raw VEP scores reproduced exactly. Independent causal runs were exact on complex-trait and SGE frames but had sparse Mendelian BF16 outliers; source/continued mean absolute error was 9.0e-6–1.7e-5 despite maxima 0.031–0.097. Aggregate AUPRC remained consistent. These independent-run numerics do not resemble a coordinate, tokenizer, or checkpoint shift.
- Direct records: [checkpoint audit](https://wandb.ai/gonzalobenegas/marin/runs/gavkgtmf), [stability audit](https://wandb.ai/gonzalobenegas/marin/runs/q67hbkp4), [final dependency](https://wandb.ai/gonzalobenegas/marin/runs/yl5sgffn), and [compact audit bundle](../artifacts/479-mntp-adaptation/audit/).
- Compute: final conservative listed-price estimate is $24.7340 of the $50 cap, including all failed, recovery, training, evaluation, audit, cancelled exhaustive, and focused attempts. The final Lambda cluster was confirmed absent after self-termination. No Iris or Marin runtime was used.
- Interpretation: no training/inference implementation bug was found. The surprising continued-CLM regression is best explained by destructive optimization from a fresh optimizer and high registered peak learning rates, with progressive damage and cooldown recovery. The one confirmed bug was isolated to the original dependency-analysis batching method and is corrected in the durable result.
- Decision: retain the original negative pilot decision—no 10,000-step MNTP extension and no single-orientation VEP recommendation—but qualify all dependency claims with the corrected same-call final-checkpoint maps.
- Next action: publish the audited snapshot/tag, update issue #479 and the bidirectional-models research synthesis, and retain the raw scores/maps only in private staging.

### 2026-08-21 15:02 - Causal-continuation calibration selected as the next gate

- Trigger: The integrity audit ruled out the tested serialization, replay, tokenization, coordinate, readout-shift, validation, and instability bugs, but the registered continued-CLM arm used a fresh optimizer with peak learning rates of 0.004396588845822712 for AdamH parameters and 0.02308025763094388 for ordinary Adam parameters.
- Hypothesis: Full-parameter causal fine-tuning from the released source weights can preserve or lower fixed-plan causal validation loss when a simpler AdamW optimizer uses a learning rate appropriate for a mature checkpoint.
- Config: Sweep `1e-6`, `3e-6`, `1e-5`, and `3e-5` for 200 steps on the same tokenizer, training stream, orientation policy, lowercase repeat weighting, and five-component fixed validation panel.
- Validation: Recompute pooled and per-component causal loss at steps 0, 1, 10, 25, 50, 100, and 200.
- Gate: Advance only a configuration whose step-200 pooled loss is no higher than its step-0 loss and whose component trajectories do not show progressive degradation.
- Downstream evaluation: Run AUPRC trajectories only for the best non-degrading configuration.
- Budget: Carry forward $24.7340 of the $50 cap and project the complete calibration cost before provisioning.
- Decision: Do not launch another 1,000-step arm until the causal fine-tuning gate passes.
- Next action: freeze the AdamW constants and short schedule in a reviewed run config, execute the bounded sweep, and update issue #479 with validation trajectories before interpreting the causal control.

### 2026-08-21 15:24 - Conservative learning rate selected for the first sanity arm

- Trigger: The human collaborator requested one conservative learning rate before any parallel or sequential sweep expansion.
- Change: Run only one 200-step full-parameter AdamW arm at `1e-6` from the released source checkpoint.
- Validation: Recompute pooled and per-component causal loss at steps 0, 1, 10, 25, 50, 100, and 200.
- Gate: Run checkpoint AUPRC trajectories only if step-200 pooled loss is no higher than step 0 and the five component trajectories do not progressively degrade.
- Expansion boundary: Review this result before choosing a larger learning rate, a learning rate below `1e-6`, a narrower parameter-update scope, or any longer run.
- Budget: Carry forward $24.7340 of the $50 cap and project this single arm before provisioning.
- Public record: [issue comment](https://github.com/Open-Athena/marin-dna/issues/479#issuecomment-5371856205).
- Next action: implement and test the single-arm run config, snapshot it, project cost, and launch the self-terminating Lambda task.
