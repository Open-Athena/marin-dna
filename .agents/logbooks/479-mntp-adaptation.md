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

The reported exp479 training and validation losses used an incorrect denominator and omitted the source training z-loss.
They multiplied token losses by the lowercase-repeat weights but divided by raw selected-token count; the pinned Marin source training reducer divides by the sum of effective weights and includes the small z-loss term.
On the fixed panel, the raw weight-sum/token-count ratio is 0.305 across all five probes and 0.239 across the three source validation datasets, explaining why the reported source loss was only 0.231.
The original tagged validation callback had a separate bug that multiplied repeat weights twice for mixed-case slices and omitted z-loss.
The corrected 128-row-per-dataset causal audit reproduces the invalid 0.23138 value, reports 0.76463 pure validation CE under the source-compatible three-dataset macro, and retains the small worsening direction through step 1,000.
The full 49,152-row audit reproduces the original nine-metric W&B macro as 0.861413936 versus 0.861344755, with maximum metric error 0.000168145, and reports a corrected single-weight macro of 0.875662646.
Every checkpoint remains retained in W&B, and knowledge-base interpretation remains paused.
The corrected 1,000-step causal replacement reports five-component macro validation CE of 0.769008732 at source and 0.773670488 at step 1,000.
Its training and gradient traces are finite, all 13 corrected checkpoints are retained, and the knowledge-base interpretation remains paused.

## Current baseline

- Source checkpoint: [`marin-dna/marin-dna-exp135-m5.1@a73a5dcf`](https://huggingface.co/marin-dna/marin-dna-exp135-m5.1/tree/a73a5dcfb3d64b8941e7e7596c6e88ef77db3e7a), step 59,158, 1,120,772,224 parameters.
- Architecture: Qwen3, 19 layers, hidden size 1,920, intermediate size 7,680, 15 attention/KV heads, 256-token context.
- Vocabulary: `[PAD]`, `[UNK]`, `[BOS]`, A, C, G, T. The tokenizer lowercases input.
- Current Lambda list price: $2.29/GH200-hour before applicable tax, checked 2026-08-19.
- Completed experiment list-price estimate: $28.3080 of the $50 cap; final cluster confirmed terminated.
- Odd-autosome/X labeled diagnostics only; no even-autosome or Y labels, predictions, effect measurements, or aggregate metrics were accessed.

## Hypothesis queue

### Active

- No further compute is selected while the corrected causal result awaits human review.

### Blocked

- None.

### Falsified / dead end

- `MNTP-479-H2` (strict control criterion): transferred MNTP used both flanks, but its left response did not exceed the full-attention/no-adaptation control. Evidence: [result bundle](../artifacts/479-mntp-adaptation/README.md).
- `MNTP-479-H3` (downstream gate): no primary VEP endpoint improved over source CLM FWD+RC. Single-pass dependency structure remained similar to FWD+RC, but that scoped mechanism did not rescue the registered VEP/extension gate. Evidence: [W&B report](https://wandb.ai/gonzalobenegas/marin/reports/Issue-479-1k-step-MNTP-adaptation-pilot--VmlldzoxNzc2ODgyOQ).
- `MNTP-479-H6` (objective-bug explanation): correcting the training denominator and source z-loss did not eliminate the small initial improvement followed by progressive validation worsening. Evidence: [corrected W&B run](https://wandb.ai/gonzalobenegas/marin/runs/f77ypos4).

### Promoted

- `MNTP-479-H1` (exploratory, one seed): transferred MNTP reached lower step-1,000 pooled loss (0.397270 versus 0.399543) and single-mask loss (0.310077 versus 0.313152) than scratch. Evidence: [validation figure](../artifacts/479-mntp-adaptation/figures/validation-loss.svg).

- `MNTP-479-H5` (diagnostic): the retained causal-checkpoint trajectory still improves slightly during warmup and then worsens after repeat-weight normalization is corrected and the macro is limited to the three source validation datasets. Evidence: [corrected loss audit](https://wandb.ai/gonzalobenegas/marin/runs/v6mo9gh3).

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
- 2026-08-21: Supersede every exp479 absolute-loss claim and the prior no-loss-bug conclusion because repeat-weighted losses were divided by raw token count instead of the effective weight sum.
- 2026-08-21: Preserve the retained checkpoints and recompute the causal trajectory before running more training or interpreting the prior loss direction.
- 2026-08-21: Keep the exp479 denominator bug in the experiment record because it is local to this one-off framework; track only Levanter's current shared double-weighting bug in #499.
- 2026-08-21: Replace the superseded 1,000-step causal control with one corrected run from the released source checkpoint before interpreting the causal trajectory.
- 2026-08-21: Keep the corrected causal result in experiment mode, select no further compute automatically, and leave knowledge-base interpretation paused for human review.

## Negative results index

- The strict bilateral-context criterion failed: transferred exceeded the no-adaptation control on the right but not the left.
- Transferred MNTP did not improve Mendelian macro, complex-trait global, or SGE accession/consequence macro AUPRC over source CLM FWD+RC.
- No VEP task passed the single-orientation gate because transferred FWD did not exceed source CLM FWD+RC.
- A 10,000-step extension is not proposed from this one-seed pilot.
- The registered continued-CLM recipe progressively increased fixed-plan loss from 0.23138 at step 0 to 0.35965 at step 800 before partial cooldown recovery to 0.35010 at step 1,000.
- The completed AdamW `1e-6` and `1e-5` controls optimized the invalid count-normalized objective and are not faithful causal-continuation controls.
- All prior exp479 absolute losses are invalid for comparison with the source W&B run; their denominator lowers the scale in proportion to each panel's uppercase/lowercase composition.
- The corrected AdamW `1e-5` replacement increased five-component macro validation CE from `0.769008732` at source to `0.773670488` at step 1,000 despite finite training and gradients.

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

### 2026-08-21 16:10 - AdamW 1e-6 causal-continuation sanity result

- Launch snapshot: `3b1fd1747c9d9e9ff35e8e19ea997247b3027dce`.
- Command: `uv run --locked python launch.py calibration --commit 3b1fd1747c9d9e9ff35e8e19ea997247b3027dce --hf-repo-id gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover --prior-cost-usd 24.7340 --retry-until-up --execute`.
- Config: One full-parameter causal arm, AdamW at `1e-6`, betas `(0.9, 0.95)`, epsilon `1e-8`, no weight decay, 10-step linear warmup from `1e-7`, batch 64, 200 steps, and gradient clipping at `1.0`.
- Validation: The exact fixed 640-row panel contains 128 rows from each of CDS, downstream, enhancer, ncRNA, and upstream, with lowercase repeat weight `0.01` in the shared causal loss path.
- Pooled result: Fixed-plan causal loss changed from `0.231380263` at step 0 to `0.231159212` at step 200, a decrease of `0.000221051`; the fitted slope was `-1.2567e-6` per step.
- Component result: CDS, enhancer, and ncRNA passed both strict checks. Upstream ended `0.000014797` below baseline but had a fitted slope of `+8.64e-8` per step. Downstream ended `0.000006404` above baseline with a fitted slope of `+2.37e-8` per step.
- Gate: The preregistered all-component gate is `false` because upstream and downstream missed a zero-tolerance sign check, even though the pooled and three other component trajectories improved and neither miss shows broad degradation.
- Stability: The 200-step training-loss trace was finite. Pre-clipping gradient norm ranged from `0.5176` to `0.8916`, averaged `0.6477`, and never reached the `1.0` clip threshold. No clipped step or gradient spike occurred.
- Runtime: Training processed 3,276,800 model tokens in 105.55 seconds at 31,044 tokens/s, with 67,360,480,768 peak allocated CUDA bytes.
- Publication: The compact W&B evidence completed at [run q09fcejx](https://wandb.ai/gonzalobenegas/marin/runs/q09fcejx). The final 2.2 GB BF16 checkpoint upload failed after evaluation because the Hugging Face account had reached its private-storage limit. No result metric depends on that failed publication.
- Compute: The Lambda instance ran from 15:52:01 to the 16:05:51 pre-autodown cost record. This arm cost an estimated `$0.5284`, bringing the conservative listed-price total to `$25.2624 / $50`; the cluster then self-terminated and was confirmed absent.
- Decision: Do not run AUPRC, another learning rate, or a longer causal arm automatically. First review the visually near-flat component misses and choose whether the next gate should use a lower learning rate, a parameter-efficient update, a larger fixed validation panel, or a tolerance justified before another run.
- Artifacts: [compact calibration bundle](../artifacts/479-mntp-adaptation/causal-calibration-lr1e-6/).

### 2026-08-21 16:46 - One-thousand-step causal trajectory selected

- Trigger: The human collaborator chose a single longer causal run after reviewing the near-flat 200-step AdamW `1e-6` pooled trajectory.
- Config: Run full-parameter AdamW at peak learning rate `1e-5` for 1,000 steps with betas `(0.9, 0.95)`, epsilon `1e-8`, zero weight decay, global gradient clipping at `1.0`, batch 64, and seed 0.
- Schedule: Linear warmup from zero through step 100, constant peak learning rate through step 800, and linear decay to zero at the step-1,000 boundary.
- Validation: Report only the pooled fixed-plan causal loss at steps 0, 25, 50, 100, every 100 steps through 800, 900, and 1,000.
- Macro equivalence: The pooled panel has exactly 128 sequences from each of CDS, downstream, enhancer, ncRNA, and upstream, so its sequence mean is also the equally weighted five-component macro.
- Retention: Copy every post-update trajectory export to a W&B model artifact as it is produced, then copy the complete step-1,000 Lightning model/optimizer/scheduler/loop checkpoint before validation.
- Publication boundary: Do not create or upload to a Hugging Face repository, and do not delete retained checkpoint artifacts during this investigation.
- Budget: Carry forward `$25.26241970350875`; a conservative two-hour Lambda GH200 reservation projects `$29.84241970350875 / $50`.
- Interpretation boundary: Remain in experiment mode and publish factual trajectories and stability checks without updating the research knowledge base.
- Next action: Implement and verify the locked long-run stage, snapshot the exact commit, update issue #479, launch one self-terminating Lambda GH200, and monitor through durable checkpoint retention and pooled validation.

### 2026-08-21 17:26 - AdamW 1e-5 causal trajectory completed

- Launch snapshot: `9fed5963d454167f9d1fbc74f91e87f1ecc6944b`.
- Command: `uv run --locked python launch.py longrun --commit 9fed5963d454167f9d1fbc74f91e87f1ecc6944b --prior-cost-usd 25.26241970350875 --retry-until-up --execute`.
- Placement: One Lambda GH200 in `us-east-3` at `$2.29/hour` after `us-east-1` reported insufficient capacity.
- Verification: The remote locked suite passed 93 tests, and the regenerated training/validation plan hashes exactly matched `9c715b08dad078c8ae5cf06325d4917051f52453f048674f6507ef6563130b91` and `35542611d71102479f3d07dc6565350120d1d89944e5a93f88efb641ece7e3ba`.
- Config: Full-parameter AdamW at peak learning rate `1e-5`, betas `(0.9, 0.95)`, epsilon `1e-8`, zero weight decay, clipping at `1.0`, batch 64, seed 0, 100-step warmup, stable through step 800, and linear decay to zero at step 1,000.
- Pooled validation: Loss was `0.231380263` at step 0, reached a minimum of `0.230961750` at step 100, crossed above baseline between steps 200 and 300, peaked at `0.233230022` at step 900, and ended at `0.233014855` at step 1,000.
- Gate: The step-1,000 versus step-0 delta was `+0.001634592`, and the fitted slope was `+2.1429e-6` per step, so the pooled validation gate is false.
- Stability: All 1,000 numeric trace rows were finite. Pre-clipping gradient norm minimum/median/p95/maximum was `0.5173/0.6599/0.7577/1.3261`; only steps 265 and 738 clipped, and neither coincided with a training-loss spike. Successive 100-step mean training losses stayed within `0.8696–0.8827`.
- Retention: Twelve trajectory exports and one full step-1,000 Lightning checkpoint containing optimizer, scheduler, and loop state were committed as 13 W&B model artifacts totaling 67.25 GB. The full artifact is `gonzalobenegas/marin/dna-exp479-causal-longrun-step-1000-full:v0`.
- Publication boundary: No output was uploaded to Hugging Face and no retained checkpoint was deleted.
- Runtime: Training executed 1,000 steps in 783.78 seconds, processed 16,384,000 model tokens at 20,904 tokens/s including synchronous artifact retention pauses, and peaked at 67,360,480,768 allocated CUDA bytes.
- Cost: The instance ran from 17:00:46 to the 17:23:19 pre-autodown record. This attempt cost an estimated `$0.8613`, bringing the conservative listed-price total to `$26.1237 / $50`.
- Teardown: The job succeeded and `sky status -r dna-exp479-gh200` confirmed the cluster absent.
- Direct record: [W&B run 5lbazal6](https://wandb.ai/gonzalobenegas/marin/runs/5lbazal6).
- Factual readout: The selected recipe improves pooled validation during warmup, then causes a small progressive degradation at constant peak learning rate with only partial cooldown recovery. The trace does not show numerical instability.
- Interpretation boundary: Do not run AUPRC or update the research knowledge base from this result until the experiment presentation is reviewed.
- Next action: Commit and tag the compact result bundle, update issue #479 body/comment with the validation and stability figures, and keep the W&B checkpoints retained.

### 2026-08-21 17:58 - Repeat-weight normalization bug found

- Trigger: The human collaborator noticed that the exp479 source validation loss of `0.231380263` was far below the original training run.
- Original W&B result: The released m5.1 run ended at `eval/val_cds/loss=0.633785069`, `eval/val_downstream/loss=0.620932400`, `eval/val_upstream/loss=0.782120407`, `eval/loss=0.997990429`, and `eval/macro_loss=0.861344755`.
- Loss bug: `per_sequence_weighted_loss` multiplied cross-entropy by weights 1.0 for uppercase and 0.01 for lowercase, then divided by raw selected-token count.
- Source contract: Pinned Haliax `maybe_reduce_loss` divides the weighted numerator by the sum of effective weights; the source recipe also uses z-loss weight `4.312883184368223e-6`.
- Panel composition: Effective-weight sum divided by selected-token count is `0.373182` for CDS, `0.160320` for downstream, `0.182522` for upstream, `0.408275` for enhancer, and `0.402785` for ncRNA.
- Aggregate shrinkage: The ratio is `0.305417` across all five fixed probes and `0.238675` across the three source validation datasets.
- Validation-scope bug: The original recipe validates CDS, upstream, and downstream. Its five-component mixture is a training mixture. Exp479 added enhancer and ncRNA validation probes and incorrectly described the resulting five-way mean as source-comparable.
- Macro definition: Original W&B `eval/macro_loss` is the mean of nine metrics: default, uppercase-only, and lowercase-only losses for each of the three source validation datasets. Exp479 did not recreate those nine slices.
- Impact: Absolute losses, loss gates, and the claim that the loss path was bug-free are invalid. Trajectory direction may survive because each checkpoint used the same fixed panel, but it must be recomputed from the retained models.
- Training impact: All completed exp479 arms optimized the count-normalized objective and omitted the source z-loss term. Corrected evaluation cannot make those checkpoints faithful continuations retroactively.
- Fix: Normalize sequence-balanced MNTP loss by each sequence's weight sum; use the global token-weighted mean for continued CLM; restore the pinned source z-loss term; retain the legacy reducer only in the audit output for diagnosis.
- Verification: Ruff lint and format pass. Local PyTorch tests are intentionally deferred because their measured working set exceeds the shared-node 500 MiB limit; the locked suite runs on the Lambda audit worker before evaluation.
- Next action: Snapshot the fix, warn issue #479, run the source plus all 12 retained causal checkpoints on one self-terminating Lambda GH200, and publish corrected three-source and five-probe macro trajectories without deleting checkpoints.

### 2026-08-21 18:12 - Retained causal trajectory re-evaluated

- Launch snapshot: `2605278a7760ee3eb474f678bfb3db2b10850de2`.
- Command: `uv run --locked python launch.py loss-normalization --commit 2605278a7760ee3eb474f678bfb3db2b10850de2 --prior-cost-usd 26.1237 --retry-until-up --execute`.
- Placement: One Lambda GH200 in `us-east-3` at `$2.29/hour` after `us-east-1` reported insufficient capacity.
- Verification: All 98 locked tests passed. The regenerated training and validation hashes matched `9c715b08dad078c8ae5cf06325d4917051f52453f048674f6507ef6563130b91` and `35542611d71102479f3d07dc6565350120d1d89944e5a93f88efb641ece7e3ba`.
- Legacy reproduction: The new evaluator reproduced the prior five-probe source value as `0.231380263`, confirming that the low scale comes from the registered count denominator rather than a changed checkpoint.
- Corrected source-three macro: Marin-weighted loss was `0.764665566` at step 0, reached `0.763741364` at step 100, and ended `0.767604491` at step 1,000, a final change of `+0.002938926`.
- Corrected five-probe macro: Loss was `0.769045854` at step 0, reached `0.767755466` at step 100, and ended `0.773838651` at step 1,000, a final change of `+0.004792797`.
- Direction: The small initial improvement and subsequent worsening survive corrected normalization and the three-source-dataset scope.
- Scale: The corrected fixed-panel source-three macro is above the original run's three default-region mean of `0.678945959`.
- Sampling diagnosis: Each original source validation dataset contains 16,384 rows and the original W&B config used all rows. The exp479 fixed panel uses only 128 shuffled rows per dataset, or 0.78%, so it cannot establish exact source-metric parity.
- Z-loss: The pinned source z-loss contributes only about `3e-5`; it does not explain either the original 3–4× scale error or the small-panel/full-validation gap.
- Retention: The source and all 12 numbered causal checkpoints were evaluated from the released source or version-pinned W&B artifacts. No checkpoint was deleted or modified.
- Evidence: [W&B run v6mo9gh3](https://wandb.ai/gonzalobenegas/marin/runs/v6mo9gh3) and compact artifact directory `loss-normalization-audit/`.
- Compute: The audit cost an estimated `$0.3450`, bringing the conservative listed-price total to `$26.4687 / $50`. The cluster self-terminated and is confirmed absent.
- Next action: Evaluate all 16,384 rows from each original CDS, upstream, and downstream validation dataset once, derive the default/uppercase-only/lowercase-only metrics from the same logits, and compare the nine slices plus macro directly with the original W&B run.

### 2026-08-21 18:44 - Full source-validation gate localized a second reporting bug

- Launch snapshot: `ec3c110a9897c793d364714fd805cafaa495f8f1`.
- Run: [W&B mwdtno1h](https://wandb.ai/gonzalobenegas/marin/runs/mwdtno1h) on one Lambda GH200 in `us-east-3` after `us-east-1` lacked capacity.
- Verification: All 102 locked tests passed before 49,152 source validation rows were evaluated.
- First gate result: The naively single-weighted macro was `0.875689735` versus original W&B `0.861344755`, so the hard gate failed with maximum metric delta `0.057035602`.
- Localization: All six binary uppercase-only/lowercase-only metrics matched within `0.000215`; only the three default 1.0/0.01 repeat-weighted metrics differed.
- Pinned-source diagnosis: Levanter's tagged evaluator obtains per-position loss through `compute_next_token_loss(reduction=None)`, which already applies `loss_weight`, and then multiplies that array by `loss_weight` again in `TaggedEvaluator.accum_for_batch`.
- Exact historical reproduction: Applying the resulting squared numerator and single-weight denominator gives default CDS/upstream/downstream losses `0.633996/0.782180/0.621024`, within `0.000211/0.000059/0.000091` of original W&B.
- Interpretation: Source training uses the normal one-weight Haliax mean and includes z-loss; the tagged validation callback separately double-weights repeat masks and omits z-loss.
- Cost: The attempt cost an estimated `$0.275383`, bringing the conservative listed-price total to `$26.744106 / $50`.
- Retention: The compact failed-gate CSV, JSON, SVG, and W&B run are retained; no checkpoint was deleted, modified, or uploaded.
- Teardown: Sky armed one-minute `down` autodown after the failed job; final absence confirmation is pending.
- Next action: Encode both historical double-weight reproduction and corrected single-weight validation CE, rerun the hard gate, and publish the correction without loosening tolerance.

### 2026-08-21 19:21 - Full source-validation gate passed

- Launch snapshot: `7801a14e5b5abb46cb8ca1aca7c289a99d8d3016`.
- Run: [W&B hfuhn3ta](https://wandb.ai/gonzalobenegas/marin/runs/hfuhn3ta) on one Lambda GH200 in `us-east-3`.
- Verification: All 103 locked tests passed before 49,152 source validation rows were evaluated.
- Full parity: The reproduced pinned-evaluator macro is `0.861413936` versus original W&B `0.861344755`, a delta of `0.000069181`.
- Metric gate: The maximum absolute delta among all nine component/slice metrics is `0.000168145`, inside the unchanged `0.002` tolerance.
- Corrected scale: Applying repeat weights once gives a nine-metric validation CE macro of `0.875662646`.
- Historical bias: The pinned tagged evaluator's second repeat-weight multiplication biased the macro downward by `0.014248710`.
- Integrity readout: Exact historical reproduction across the three full datasets and all three slice definitions is strong evidence against an off-by-one shift, tokenizer special-token mismatch, or different source checkpoint in this evaluation.
- Evidence: The passing CSV, JSON, SVG, PNG, and manifest are in `source-validation-reproduction/`; the first failed localization run remains in `source-validation-reproduction-v0-failed/`.
- Retention: No checkpoint was deleted, modified, or uploaded.
- Compute: This attempt cost an estimated `$0.289679`, bringing the conservative listed-price total to `$27.033785 / $50`.
- Teardown: The managed job succeeded, and `sky status -r` confirms no in-progress job or live service.
- Next action: Commit and snapshot the exact evidence, then update issue #479's body and add a concise correction comment while keeping research knowledge-base interpretation paused.

### 2026-08-21 19:51 - Corrected causal replacement preregistered

- Implementation commit: `69981456f2ef4adf4e9a90896a0d3f0a07e72cbf`.
- Trigger: The previously authorized AdamW `1e-5` run optimized the local count-normalized loss and omitted source z-loss, so its checkpoints cannot answer whether source-compatible causal continuation is stable.
- Command: `uv run --locked python launch.py longrun --commit <launch-snapshot> --prior-cost-usd 27.033784759 --retry-until-up --execute`.
- Initialization: Start again from released `marin-dna/marin-dna-exp135-m5.1@a73a5dcf`; do not resume the superseded optimizer state.
- Optimizer: Full-parameter AdamW at peak learning rate `1e-5`, betas `(0.9, 0.95)`, epsilon `1e-8`, zero weight decay, batch 64, seed 0, and global gradient clipping at `1.0`.
- Schedule: Linear warmup from zero through step 100, constant peak learning rate through step 800, and linear decay to zero at the step-1,000 boundary.
- Training objective: Apply each uppercase/lowercase repeat weight once, divide the global batch numerator by the effective-weight sum, and include source z-loss weight `4.312883184368223e-6`.
- Validation: At 13 retained checkpoints, report only pure causal cross-entropy computed as the equal macro of five exact global component reducers over 128 fixed rows each; exclude training-only z-loss.
- Verification: Run the full locked test suite on the Lambda worker before model loading and training; abort the task if it fails.
- Retention: Upload each of 12 post-update Hugging Face-format exports and the full step-1,000 Lightning checkpoint to a distinct W&B artifact namespace; do not upload to Hugging Face or delete a retained checkpoint.
- Budget: Carry forward `$27.033784759`; a conservative two-hour Lambda GH200 reservation projects `$31.613784759 / $50`.
- Interpretation boundary: Publish factual loss and stability trajectories while keeping the knowledge-base update paused.
- Next action: Commit this logbook entry, dry-run the exact launch snapshot, launch one self-terminating Lambda GH200, and monitor through checkpoint retention and validation publication.

### 2026-08-21 20:28 - Corrected AdamW 1e-5 causal replacement completed

- Launch snapshot: `42fc993e3245a0f6a1c1d77813b0665ef56e68e5`.
- Command: `uv run --locked python launch.py longrun --commit 42fc993e3245a0f6a1c1d77813b0665ef56e68e5 --prior-cost-usd 27.033784759 --retry-until-up --execute`.
- Placement: One Lambda GH200 in `us-east-3` after `us-east-1` reported insufficient capacity.
- Verification: All 104 locked tests passed before model loading, and regenerated plan hashes matched `9c715b08dad078c8ae5cf06325d4917051f52453f048674f6507ef6563130b91` for training and `35542611d71102479f3d07dc6565350120d1d89944e5a93f88efb641ece7e3ba` for validation.
- Objective: Full-parameter AdamW at `1e-5` used one repeat-weight application, global effective-weight normalization, source z-loss `4.312883184368223e-6`, and the preregistered 10%/70%/20% schedule.
- Validation: Five-component macro pure CE was `0.769008732` at step 0, `0.767801766` at step 100, `0.768950871` at step 200, `0.770591888` at step 300, `0.773703543` at step 800, `0.774135425` at step 900, and `0.773670488` at step 1,000.
- Gate: The final change was `+0.004661756` and the fitted slope was `+6.169216e-6` per step, so the no-increase macro gate is false.
- Cross-check: Step 0 differs by `-0.000006417` from the earlier independent evaluator, and the corrected trajectory differs from the same metric on the superseded checkpoints by at most `0.000136974` across all 13 points.
- Stability: All 1,000 numeric trace rows are present and finite; successive 100-step mean training losses stayed within `1.0216–1.0308`; gradient norm median/p95/maximum was `0.7674/0.8845/1.3722`; steps 238, 265, 646, 682, 738, and 867 clipped.
- Retention: Twelve numbered model exports and one full optimizer-bearing Lightning checkpoint are committed as 13 W&B model artifacts totaling 67.25 GB; the full artifact is `gonzalobenegas/marin/dna-exp479-causal-longrun-corrected-step-1000-full:v0`.
- Runtime: The 1,000 training steps took `1,306.77` seconds including synchronous artifact retention, processed 16,384,000 model tokens at 12,537.74 tokens/s, and peaked at 67,360,482,816 allocated CUDA bytes.
- Cost: The final teardown record estimates `$1.274169` for this attempt and `$28.307954 / $50` cumulative.
- Evidence: [W&B f77ypos4](https://wandb.ai/gonzalobenegas/marin/runs/f77ypos4) and compact artifact directory `causal-longrun-lr1e-5-corrected/`.
- Publication boundary: No Hugging Face upload, checkpoint deletion, AUPRC evaluation, held-out even-autosome/Y access, or knowledge-base update was performed.
- Teardown: The managed job succeeded, and `sky status -r` confirms no cluster, in-progress job, or live service.
- Next action: Commit and tag this factual result, update issue #479's body and comment with the figures, and select no additional compute automatically.
### 2026-08-21 21:15 - Corrected transferred-MNTP run preregistered

- Parent snapshot: `4d6dc5963257074753c18330ad911989322ce795`.
- Approval: The user approved one corrected 1,000-step transferred-MNTP run and no parallel sweep.
- Initialization: Start from the untouched released `marin-dna/marin-dna-exp135-m5.1@a73a5dcf`, add one MASK row to each untied vocabulary matrix from its A/C/G/T row mean, and use explicit full attention.
- Optimizer: Use a fresh full-parameter AdamW state at peak learning rate `1e-5`, betas `(0.9, 0.95)`, epsilon `1e-8`, zero weight decay, batch 64, seed 0, and global gradient clipping at `1.0`.
- Schedule: Warm linearly from zero through step 100, remain at peak through step 800, and decay linearly to zero at the step-1,000 boundary.
- Training objective: Apply each repeat weight once, divide each sequence by its effective target-weight sum, average sequences equally, and include source z-loss `4.312883184368223e-6`.
- Validation: Report equal five-component macro pure CE for deterministic diffusion and single-mask protocols at steps 0, 25, 50, 100, and every 100 steps thereafter.
- Downstream trajectory: Compute registered FWD+RC AUPRC on odd-numbered autosomes and chromosome X at step 0 and every 100 steps through 1,000.
- Dependency: Compute the directed tRNA-Arg-TCT nucleotide-dependency map only at the final checkpoint.
- Retention: Commit all 12 numbered model exports and the full step-1,000 optimizer-bearing Lightning checkpoint as W&B model artifacts; do not upload to Hugging Face and do not delete a retained checkpoint.
- Verification: The local locked suite passes 111 tests; the Lambda worker reruns the entire locked suite before model loading.
- Budget: Carry forward `$28.307954`; a conservative two-hour Lambda GH200 reservation projects `$32.887954 / $50`.
- Storage cleanup: At the user's explicit request, permanently delete private repository `gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover`, which held 175 files and 80,766,064,598 logical bytes at revision `59fd525018c545f81c4deccaf3800bef5a41886a`; authenticated verification reports it absent.
- Publication boundary: Do not update the research knowledge base while the experiment remains in progress.
- Next action: Snapshot and push the exact implementation, dry-run it, launch one self-terminating Lambda GH200, and monitor startup plus the first trajectory evidence.

### 2026-08-21 23:12 - LDLR final-checkpoint dependency replacement preregistered

- Trigger: The user rejected tRNA-Arg-TCT as the selected nucleotide-dependency example and requested the locus shown by default in the interactive browser.
- Locus: Use LDLR at chromosome 19 `[11089299, 11089425)` on the positive strand, matching the browser's first configured locus and the project's 0-based half-open coordinate convention.
- Checkpoint: Load only retained W&B artifact `gonzalobenegas/marin/dna-exp479-mntp-longrun-corrected-step-1000:v0`; do not retrain, modify, upload, or delete a checkpoint.
- Method: Compute a reference-orientation 255-by-255 directed map with full attention and a masked readout; pair each wild-type baseline with its substitutions in the same model call.
- Gates: Require the exact registered shape, finite values, an exactly zero diagonal, and nonzero past- and future-context maxima.
- Publication: Retain only the compact matrix, summary, invariant record, manifest, and SVG in W&B; do not upload to Hugging Face or update the research knowledge base.
- Compute: Use one self-terminating Lambda GH200, carry forward the conservative `$32.289179 / $50` cost, and run the complete locked suite before model loading.
- Next action: Commit and push the rebased implementation, dry-run the exact snapshot, launch the evaluation, inspect the rendered LDLR plot, and replace the tRNA example in the factual evidence.

### 2026-08-21 23:45 - Paired nucleotide information gate preregistered

- Trigger: The user made coherent nucleotide prediction from both flanks a prerequisite and paused VEP interpretation until that prerequisite passes.
- Existing-checkpoint panel: Evaluate the released causal model, released weights under full attention with the added MASK row, released weights under full attention with existing UNK as the mask, and the corrected step-1,000 MNTP checkpoint under both causal and full attention.
- Pairing: Use the exact registered validation plan hash `35542611d71102479f3d07dc6565350120d1d89944e5a93f88efb641ece7e3ba` and the same one deterministic A/C/G/T target in each of its 640 sequences for every readout.
- Primary metrics: Report unweighted four-way A/C/G/T-renormalized cross-entropy and top-1 accuracy so the added output row and repeat weights cannot alter the comparison.
- Secondary metrics: Retain full-vocabulary cross-entropy and accuracy as a training-contract audit.
- Gate: The adapted full-attention readout must have no higher paired mean four-way CE and no lower paired top-1 accuracy than the released causal readout, with both directions supported by paired 95% sequence-bootstrap intervals.
- Localization: Also compare step-0 full attention against source causal, existing UNK against the added MASK row, and adapted full against adapted causal; plot overall macros and position-binned trajectories.
- Compute: Use one Lambda A10 rather than GH200, download only the retained 4.28 GB model artifact plus the public source model, and query the five pinned validation datasets remotely without downloading a reference genome.
- Boundaries: Do not evaluate VEP, update the research knowledge base, upload to Hugging Face, delete a checkpoint, or run nucleotide-dependency analysis.
- LoRA next step: If the existing checkpoint fails, freeze the released base and test a rank-16 MNTP adapter against this identical paired gate before any VEP evaluation.
- Canceled LDLR attempt: The earlier Lambda GH200 dependency job was canceled and the cluster torn down after the user requested cheaper compute; it produced no replacement map and did not modify or delete the retained checkpoint.
- Next action: Commit and push the exact evaluator, dry-run the A10 launch, run the complete locked suite remotely, and inspect the paired figure before specifying the LoRA pilot.

### 2026-08-22 00:09 - Existing-token mask control amended

- User direction: Explore both existing `[UNK]` and `[PAD]` tokens early rather than assume one mask representation.
- Step-0 panel: Add full-attention `[UNK]` and `[PAD]` readouts beside the new `[MASK]` row on the identical paired targets.
- Sequential selection: Advance only the better existing-token choice to the first rank-16 LoRA pilot; if their paired intervals overlap, prefer `[UNK]` for its unknown-nucleotide semantics.
- Compute: Keep the readout on one Lambda A10 at the current listed price of `$1.29/hour` and record that rate explicitly in the cost estimator.
- Boundaries: Nucleotide-dependency and VEP evaluation remain paused.

### 2026-08-22 00:20 - Paired nucleotide information gate completed

- Launch snapshot: `605122f27f94c06231a864efedda2d3740221920`.
- Verification: All 118 locked tests passed on the Lambda A10 before data preparation or model loading.
- Identity: Every readout used the same 640 validation sequences and deterministic target nucleotide indices from plan hash `35542611d71102479f3d07dc6565350120d1d89944e5a93f88efb641ece7e3ba`.
- Source causal: Four-way nucleotide CE was `1.051060` and accuracy was `0.507812`.
- Step-0 full attention: New `[MASK]`, existing `[UNK]`, and existing `[PAD]` produced CE `1.412228`, `1.424901`, and `1.422409`, with accuracy `0.276563`, `0.271875`, and `0.273438`, respectively.
- Mask-token conclusion: Neither existing token improved the unadapted full-attention readout; the new vocabulary row is therefore not the leading cause of the bidirectional deficit.
- Adapted checkpoint: Step-1,000 causal CE/accuracy was `1.056157/0.512500`, while full-attention CE/accuracy was `1.260321/0.415625`.
- Paired adapted comparison: Full attention versus its own causal readout changed CE by `+0.204164` with 95% CI `[+0.158707, +0.250380]` and accuracy by `-0.096875` with 95% CI `[-0.140625, -0.053125]`.
- Primary gate: Adapted full attention versus released source causal changed CE by `+0.209261` with 95% CI `[+0.162877, +0.256399]` and accuracy by `-0.092188` with 95% CI `[-0.135938, -0.048438]`; both the point-estimate and confidence-supported gates failed.
- Evidence: [W&B u29jytbi](https://wandb.ai/gonzalobenegas/marin/runs/u29jytbi) and artifact `gonzalobenegas/marin/dna-exp479-paired-nucleotide-information-gate:paired-gate`.
- Cost: The self-terminating A10 attempt took `0.142880` instance-hours and added an estimated `$0.184315`, bringing the cumulative listed-price estimate to `$32.473494 / $50`.
- Boundaries: No VEP, nucleotide-dependency, knowledge-base update, Hugging Face upload, or checkpoint deletion was performed.
- Next action: Specify one frozen-base rank-16 LoRA pilot using `[UNK]`, fixed low-rate corruption, and the identical paired information gate; do not proceed to VEP unless the gate passes.

### 2026-08-22 00:45 - Frozen-base rank-16 LoRA pilot preregistered

- Trigger: The exact paired gate failed after full-parameter MNTP, the user selected LoRA as the next idea, and the LLM2Vec implementation provides a concrete adapter recipe.
- Initialization: Load the untouched released `marin-dna/marin-dna-exp135-m5.1@a73a5dcf`, keep every base-model parameter frozen, and add zero-effect PEFT LoRA adapters with rank 16, alpha 16, dropout 0.05, and no bias training.
- Target modules: Adapt `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`, matching the official LLM2Vec MNTP code.
- Mask token: Use existing `[UNK]`; `[UNK]` and `[PAD]` were practically indistinguishable in the step-0 control, so `[UNK]` is selected for its unknown-nucleotide semantics.
- Corruption: Select eligible A/C/G/T targets independently at fixed probability 0.20, matching LLM2Vec's configured rate, but replace 100% of selected targets with `[UNK]` so full attention cannot leak unchanged answers.
- Alignment and loss: Continue to supervise target nucleotide `i` from output `i - 1`, apply each repeat weight once with per-sequence effective-weight normalization, and include source z-loss during training.
- Optimizer: Train only adapter matrices with fresh AdamW at `1e-5`, betas `(0.9, 0.95)`, epsilon `1e-8`, zero weight decay, and global gradient clipping at `1.0`.
- Schedule: Warm linearly through step 100, stay constant through step 800, and decay linearly to zero through step 1,000.
- Exposure: Use microbatch 16 with four-step gradient accumulation on one A10, preserving effective batch 64 and the exact 64,000-sequence, 16,384,000-token exposure.
- Paired trajectory: At steps 0, 25, 50, 100, and every 100 steps through 1,000, score the identical 640 deterministic target nucleotides with active adapters and full attention against the frozen source causal baseline.
- Gate: Require final full-attention LoRA four-way CE no higher and accuracy no lower than source causal, with both paired 95% sequence-bootstrap intervals supporting those directions.
- Preservation: Disable adapters after training and require the causal per-target scores to be bit-exact to their step-0 values.
- Retention: Commit every selected adapter-only snapshot and one final adapter-plus-optimizer/RNG checkpoint to W&B; do not delete a checkpoint or upload to Hugging Face.
- Verification: Run the complete locked suite on the worker before model loading, and require finite 1,000-step loss and gradient traces.
- Compute: Use one self-terminating Lambda A10 at `$1.29/hour`; a four-hour bound projects at most `$37.633494 / $50` cumulative.
- Boundaries: Do not run VEP, nucleotide dependency, or knowledge-base interpretation unless the paired nucleotide information gate passes.

### 2026-08-22 01:20 - DiffuLLaMA reference reviewed and attention annealing added

- Reference: Gong et al., [Scaling Diffusion Language Models via Adaptation from Autoregressive Models](https://arxiv.org/abs/2410.17891), ICLR 2025, with released code at [HKUNLP/DiffuLLaMA](https://github.com/HKUNLP/DiffuLLaMA).
- Relevance: The paper identifies the same two discontinuities present here: replacing causal attention with bidirectional attention and replacing clean autoregressive inputs with noisy denoising inputs.
- Shift: Its adaptation keeps the autoregressive next-token shift, so output position `i - 1` predicts target token `i`; exp479 already satisfies this requirement.
- Annealing method: The released code opens every otherwise-forbidden future attention edge independently with probability `min(1, (global_step + 1) / anneal_steps)`, shares the sampled matrix across the batch, and retains all causal edges.
- Paper scale: DiffuGPT anneals for 10,000 steps and uses full-parameter training at much larger token exposure than this pilot.
- Limitation: DiffuLLaMA 7B skips annealing for FlashAttention efficiency, and the paper describes its measured ablation gain as minimal.
- Ablation evidence: On their GSM8K-symbolic proxy, discrete diffusion without annealing reaches 43.3/47.2 accuracy for GPT2-S/M versus 45.4/49.7 with the full recipe.
- Diffusion objective: The paper samples one continuous time `t` per sequence, masks each token with probability `t`, and weights masked-token cross-entropy by `1/t`.
- Tokenizer: The paper generally reuses a rare existing vocabulary token as `[MASK]`, consistent with the exp479 choice to avoid adding another embedding row after the existing-token controls.
- LoRA boundary: The paper's autoregressive-to-diffusion adaptation is full-parameter; LoRA is used only for downstream GSM8K-symbolic fine-tuning, so it does not directly validate frozen-base LoRA for this conversion.
- Experimental implication: Attention annealing directly targets exp479's measured step-0 full-attention collapse and is the only paper-derived change added to the first LoRA pilot.
- Revised attention schedule: During training, linearly open stochastic future edges over optimizer steps 0 through 99, reaching full attention with the last microbatches before optimizer step 100, then keep full attention through step 1,000.
- Rationale for 100 steps: This makes the transition occupy the preregistered 10% learning-rate warmup and avoids spending most of a 1,000-step sanity run in an intermediate attention regime.
- Controlled scope: Retain fixed 20% corruption and the existing exp479 loss for this run so attention annealing is not confounded with a simultaneous switch to variable `t` and `1/t` weighting.
- Determinism: Seed each sampled future-edge matrix from the experiment seed and first sample ID, share it across the microbatch as in the released implementation, and preserve the token padding mask.
- Instrumentation: Log the future-edge probability with every optimizer step and retain it in the loss and gradient trace.
- Behavioral verification: Require exact causal and full endpoints, deterministic sampled masks, shared batch masks, enforced key padding, and the existing causal/full right-flank tests before model loading.
- Follow-up if needed: If annealed LoRA fails the exact paired nucleotide gate, test the paper's continuous-time mask-rate and `1/t` weighting as a separately preregistered objective change rather than interpreting VEP.
- Boundaries: The run remains one A10 pilot with no VEP, nucleotide dependency, knowledge-base update, Hugging Face upload, or checkpoint deletion.

### 2026-08-22 01:45 - Frozen-source attention annealing sanity diagnostic preregistered

- Trigger: The user asked whether the source degradation from about 50% causal accuracy to about 27% full-attention accuracy appears gradually as the causal mask is removed, before any parameter update.
- Model: Load the untouched released `marin-dna/marin-dna-exp135-m5.1@a73a5dcf` in BF16 and perform zero optimizer steps.
- Targets: Reuse the exact 640 deterministic target nucleotides from validation plan hash `35542611d71102479f3d07dc6565350120d1d89944e5a93f88efb641ece7e3ba`.
- Mask token: Replace the selected target with existing `[UNK]`; causal scores cannot depend on that future embedding, and the earlier full-attention controls show no material advantage for `[MASK]` or `[PAD]`.
- Attention levels: Evaluate future-edge opening probabilities `0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1`.
- Nested masks: For each of five fixed mask seeds and each evaluation batch, reuse one uniform random matrix across all probabilities so increasing probability only adds future edges and never removes an already opened edge.
- Pairing: Score every probability, mask replicate, sample, and target under the same model weights and record both four-way A/C/G/T-renormalized and full-vocabulary CE and accuracy.
- Endpoint parity: Compare the custom 0% and 100% masks with the standard causal and full-attention code paths on the identical targets; require identical nucleotide correctness and maximum per-target four-way CE error below 0.002.
- Uncertainty: Plot every nested-mask replicate plus their mean and range, and compute paired 95% sequence-bootstrap intervals from replicate-mean per-target scores against the 0% causal endpoint.
- Interpretation: Report the fraction of adjacent intervals with increasing CE and decreasing accuracy and the normalized fraction of the total endpoint degradation reached at every attention level.
- Sequential decision: Pause the LoRA launch until this no-training curve is complete and inspected.
- Compute: Run the complete locked suite and then this source-only diagnostic on one self-terminating Lambda A10 at `$1.29/hour`; a two-hour bound projects at most `$35.053494 / $50` cumulative.
- Boundaries: Do not train, evaluate VEP, run nucleotide dependency, update the knowledge base, upload to Hugging Face, or delete any checkpoint.

### 2026-08-22 01:25 - First attention-annealing attempt rejected by endpoint audit

- Launch snapshot: `561c0a1fddefb903566f8e80cbdcfa8282f03309`.
- Verification: All 139 locked tests passed on the Lambda A10 before model loading.
- W&B: [Run 5yobgefq](https://wandb.ai/gonzalobenegas/marin/runs/5yobgefq).
- Audit failure: The stochastic 4D-mask path and Transformers' optimized standard paths were not sufficiently numerically equivalent in BF16 to satisfy the preregistered endpoint parity gate.
- Causal control: The 640 nucleotide correctness values were identical, but maximum absolute per-target four-way CE difference was `0.0859375`, above the registered `0.002` threshold.
- Full control: Maximum absolute per-target four-way CE difference was `0.03125`, and at least one correctness value differed.
- Disposition: Reject the completed sweep before aggregation or interpretation; these outputs do not answer whether degradation is gradual.
- Cost: Automatic teardown recorded `0.272573` instance-hours and `$0.351619`, bringing the cumulative listed-price estimate to `$32.825113 / $50`.
- Corrected numerical control: Force both standard endpoints and every stochastic level through PyTorch's math SDPA backend while retaining BF16 model weights.
- Fail-fast order: Evaluate and persist endpoint checks before any intermediate attention levels, retaining the unchanged exact-correctness and maximum-CE parity criteria.
- Diagnostics: Record maximum, mean absolute, and mean signed per-target CE differences plus correctness mismatch counts, and log progress after every attention level.
- Efficiency: Reuse deterministic endpoint scores across the five nominal mask replicates because 0% and 100% masks do not depend on the mask seed.
- Compute correction: Move the rerun to one AWS `g5.xlarge` A10G spot instance in `us-east-2` at the current Sky quote of approximately `$0.365/hour`; the two-hour bound projects `$33.555113 / $50` cumulative.
- Boundaries: Still perform zero optimizer steps and no VEP, nucleotide dependency, knowledge-base update, Hugging Face upload, or checkpoint deletion.

### 2026-08-22 01:30 - AWS spot capacity unavailable; use EC2 on-demand

- Provisioning result: AWS returned `InsufficientInstanceCapacity` for `g5.xlarge` spot in all three `us-east-2` availability zones before creating an instance.
- Cost: No GPU instance was created, so the failed spot provisioning added no compute charge.
- Cleanup: Cancel the asynchronous Sky retry request before changing resources.
- Fallback: Keep the same AWS `g5.xlarge` A10G and region but use on-demand capacity at the current `$1.006/hour` quote.
- Budget: The unchanged two-hour maximum projects `$34.837113 / $50` cumulative.

### 2026-08-22 01:34 - Real cost environment exposed a test-isolation bug

- Launch snapshot: `e911dc751a945a245cfeb14419464341c7dd631e`.
- Verification result: 139 of 140 locked tests passed before model loading.
- Failure: `test_observed_budget_projection_uses_completed_arm_runtime` inherited the launch's real `EXP479_PRIOR_COST_USD=32.825113` instead of testing its documented zero-cost default.
- Threshold effect: The inherited cost raised the synthetic projection to `$50.15`, so a fixture that previously passed only because the cumulative issue cost was lower now failed.
- Disposition: This is a unit-test environment-isolation failure; no model was loaded and no attention result was produced.
- Fix: Explicitly remove `EXP479_PRIOR_COST_USD` within that test before asserting the zero-prior default behavior.
- Cost: Automatic teardown recorded `0.047294` instance-hours and `$0.047578`, bringing the cumulative listed-price estimate to `$32.872691 / $50`.
- Boundaries: No training, VEP, nucleotide dependency, knowledge-base update, Hugging Face upload, or checkpoint deletion occurred.

### 2026-08-22 01:40 - EC2 root disk filled before model reconstruction

- Launch snapshot: `3168c2666e33fc69c0ce2245fcb065dd8f0586fc`.
- Verification: All 140 locked tests passed with the real cumulative-cost environment.
- Failure: The 32 GB EC2 root disk reached zero free space while the public Hugging Face model was being reconstructed after the locked CUDA environment installation.
- Disposition: No model forward or endpoint comparison occurred, so this run produced no scientific result.
- Fix: Increase the self-terminating task's ephemeral root disk to 80 GB and assert that capacity in the launch test.
- Cost: Automatic teardown recorded `0.072254` instance-hours and `$0.072688`, bringing the cumulative listed-price estimate to `$32.945379 / $50`.
- Boundaries: The enlarged disk is ephemeral and auto-deleted; no persistent model upload, training, VEP, nucleotide dependency, knowledge-base update, or checkpoint deletion occurred.

### 2026-08-22 01:56 - Frozen-source attention trajectory completed

- Producing snapshot: `c52d0e9e1d48fc2d29e46a754eb3c1b2405c852b`.
- Verification: All 140 locked tests passed on the AWS A10G before data preparation or model loading.
- Evidence: [W&B 0vvh4kcb](https://wandb.ai/gonzalobenegas/marin/runs/0vvh4kcb) and artifact `gonzalobenegas/marin/dna-exp479-source-attention-annealing-diagnostic:nested-attention`.
- Identity: Every readout used the same 640 deterministic targets from validation plan SHA-256 `35542611d71102479f3d07dc6565350120d1d89944e5a93f88efb641ece7e3ba` with `[UNK]` replacing the selected future token.
- Endpoint audit: Under the shared math-SDPA backend, custom 0% and 100% masks reproduced the standard causal and full paths with zero maximum, mean-absolute, and mean-signed per-target four-way CE difference and zero correctness mismatches.
- Causal endpoint: Four-way nucleotide CE was `1.051681` and accuracy was `0.510938`.
- One-percent future edges: CE was `1.174222` and accuracy was `0.456563`; versus causal, paired CE delta was `+0.122541` with 95% CI `[+0.095581, +0.149141]`, and accuracy delta was `-0.054375` with 95% CI `[-0.078125, -0.031867]`.
- Two-percent future edges: CE was `1.234798` and accuracy was `0.417813`.
- Five-percent future edges: CE was `1.302963` and accuracy was `0.379688`.
- Ten-percent future edges: CE was `1.341090` and accuracy was `0.346875`.
- Later trajectory: CE/accuracy were `1.354823/0.325000` at 20%, `1.370995/0.309688` at 30%, `1.382455/0.295000` at 40%, `1.389651/0.295000` at 50%, `1.399677/0.291875` at 60%, and `1.410491/0.286250` at 80%.
- Full endpoint: CE was `1.424632` and accuracy was `0.270313`.
- Monotonicity: CE increased in all 11 adjacent intervals; accuracy was non-increasing in all 11, decreasing in 10 and tying only from 40% to 50%.
- Front loading: One percent of future edges accounted for `32.86%` of total endpoint CE degradation, 2% for `49.10%`, 5% for `67.38%`, and 10% for `77.60%`.
- Answer: Degradation is gradual and monotone across the whole transition, but it is strongly front-loaded rather than approximately linear in the fraction of opened future edges.
- Training implication: The preregistered 100-step linear LoRA anneal would move through most of the unadapted functional collapse in its first ten optimizer steps, so revise it before launch rather than treating 100 steps as conservative.
- Runtime: The zero-update diagnostic itself took `653.61` seconds; the complete self-terminating instance took `0.244491` hours.
- Cost: This successful attempt added `$0.245958`, bringing the cumulative listed-price estimate to `$33.191337 / $50`.
- Boundaries: No parameter update, VEP, nucleotide dependency, knowledge-base update, Hugging Face upload, or checkpoint deletion occurred.

### 2026-08-22 02:19 - Damage-calibrated LoRA information gate preregistered

- Trigger: The frozen-source sweep showed that a 100-step linear attention transition would expose the unadapted model to 77.60% of its endpoint CE degradation within the first ten optimizer steps.
- Schedule: Replace the unlaunched linear transition with the piecewise-linear inverse of the measured frozen-source CE-degradation curve.
- Calibration: Pin the exact probabilities and normalized CE-damage fractions from W&B run `0vvh4kcb` under calibration tag `source-unk-zero-training-v1`.
- Milestones: Start at exactly 0% future edges and reach approximately 1%, 2%, 5%, and 10% at optimizer steps 263, 393, 539, and 621.
- Full-attention phase: Reach 100% future edges at step 800 and retain fully bidirectional attention for the final 200 optimizer steps.
- Adapter scope: Freeze the released source model and train rank-16, alpha-16 LoRA adapters with dropout 0.05 on q, k, v, o, gate, up, and down projections.
- Objective scope: Keep the existing fixed 20% selected-base rate, replace every selected base with existing `[UNK]`, and retain the sequence-balanced effective-repeat-weight MNTP CE plus source z-loss.
- Optimizer: Keep AdamW at peak learning rate `1e-5`, betas `(0.9, 0.95)`, zero weight decay, gradient clipping at 1.0, effective batch 64, seed 0, and the 100-step warmup, 700-step constant, 200-step linear-decay schedule.
- Pairing: Evaluate full-attention LoRA readouts at steps 0, 25, 50, 100, and every 100 steps thereafter on the exact 640 deterministic targets from validation plan SHA-256 `35542611d71102479f3d07dc6565350120d1d89944e5a93f88efb641ece7e3ba`.
- Source control: Disable the adapter and require the step-1,000 causal source readout to remain bit-exact to its step-0 baseline.
- Decision rule: Proceed beyond nucleotide prediction only if the step-1,000 full-attention LoRA readout has paired four-way CE no higher and accuracy no lower than the causal source, with both one-sided conclusions supported by the paired 95% sequence-bootstrap intervals.
- Development status: This odd-autosome/X gate is a mechanism-development check rather than a final generalization estimate.
- Verification: Run the complete locked test suite on the paid instance before data preparation or model loading.
- Compute: Use one self-terminating AWS `g5.xlarge` A10G on demand in `us-east-2` at `$1.006/hour` because spot capacity was unavailable and this configuration completed the source diagnostic.
- Budget: A four-hour automatic guard projects at most `$37.215337 / $50` cumulative from the current `$33.191337`.

### 2026-08-22 02:57 - Final-adapter reload parity audit preregistered

- Trigger: The user explicitly asked whether saving and reloading could worsen the model, while the training job retains PEFT adapters rather than full-model exports.
- Serialization check: Download the retained step-1,000 adapter and evaluation artifacts, attach the adapter to a freshly loaded pinned source model, and reproduce all 640 stored final full-attention scores.
- Frozen-source check: Disable the reloaded adapter and reproduce all 640 stored step-0 causal source scores to prove that the base did not change.
- Attention encoding check: Under one forced math-SDPA backend, compare standard full attention with the all-open additive mask used by the training path on all 640 targets.
- Identity contract: Pair every comparison on sample ID, target nucleotide index, and target base, rejecting missing, repeated, or changed targets.
- Serialization tolerance: Require maximum absolute per-target four-way and full-vocabulary CE differences no greater than `1e-6` and zero correctness mismatches.
- Attention tolerance: Require maximum absolute per-target CE differences no greater than `0.002` and zero correctness mismatches under the shared math backend.
- Verification: Run the complete locked test suite before downloading either final artifact or loading the model.
- Compute: Launch one self-terminating AWS `g5.xlarge` A10G only after the training run finishes and its observed cost updates the cumulative budget projection.
- Retention: Publish the parity scores, checks, manifest, cost, and artifact identities to W&B without deleting the source checkpoints.
- Boundaries: Do not evaluate VEP, run nucleotide dependency, update the knowledge base, upload to Hugging Face, or delete any checkpoint in this audit.
- Runtime guard: Terminate the audit command after two hours so the instance still reaches the automatic finalizer and shutdown path if an external download or evaluation hangs.

### 2026-08-22 03:04 - LoRA step-300 milestone

- Status: W&B run `w7c0q9qo` was healthy at optimizer step 346, with the planned step-300 adapter retained.
- Exact paired readout: Step 300 full-attention four-way CE was `1.4200915212` and accuracy was `0.2671875`, versus step-0 full-attention CE `1.4256775886` and accuracy `0.271875`.
- Schedule context: The latest logged optimizer step had future-edge probability `0.0159368254`; the step-300 readout is still in the deliberately slow opening phase.
- Interpretation boundary: The small CE improvement is an intermediate observation, not evidence that the final source-matching gate will pass.
- Issue record: Posted the milestone and commit-pinned reload-audit preregistration at https://github.com/Open-Athena/marin-dna/issues/479#issuecomment-5377521171.

### 2026-08-22 03:11 - LoRA step-400 milestone

- Exact paired readout: Step 400 full-attention four-way CE was `1.4161973607` and accuracy was `0.2765625`, improving CE by `0.0094802279` and gaining three correct targets relative to step 0.
- Gate context: Released source causal remained `1.051060/0.5078125`, so this intermediate readout remained far from passing.
- Stability: All 401 W&B optimizer rows through step 400 were present and finite; loss min/median/max was `1.272581/1.342372/1.589207`, latest-20 mean was `1.336021`, and the future-edge probability trace was monotone.
- Reload audit: The commit-pinned Sky configuration completed its no-cost dry run on AWS `g5.xlarge` at the quoted `$1.006/hour`.
- Issue record: Posted the milestone at https://github.com/Open-Athena/marin-dna/issues/479#issuecomment-5377553210 and updated the issue body from preregistration to live-run status.

### 2026-08-22 03:17 - Executable reference and reload-contract audit

- LLM2Vec match: The live run matches the executable reference on rank 16, dropout 0.05, and q/k/v/o plus gate/up/down target modules.
- Deliberate differences: LLM2Vec sets alpha to twice the rank (`32` here), inherits Transformers `5e-5` learning rate, beta2 `0.999`, no warmup, linear scheduling, and the default 80/10/10 MLM replacement, while exp479 uses alpha 16, `1e-5`, beta2 `0.95`, WSD, and 100% `[UNK]` replacement.
- Interpretation: These are preregistered conservative and anti-leak choices rather than runtime bugs, but they make the live adapter update materially smaller than the reference recipe.
- DiffuLLaMA boundary: The 7B conversion is full-parameter constant-`2e-5` training over 65B tokens and skips annealing; its exact variable-time loss samples one `t` per sequence and weights the masked-token sum by `1/t`.
- Sequential fallback: Do not modify the live run; use its final trajectory to choose between a reference-strength LoRA scaling test and a variable-`t` objective test.
- Issue record: Posted the audited implementation differences at https://github.com/Open-Athena/marin-dna/issues/479#issuecomment-5377568348.
- Reload strengthening: Extend the queued final audit to require the exact seven-token PAD/UNK/BOS contract, frozen rank/alpha/dropout/target-module contract, artifact metadata, and retained final-artifact identity before scoring parity.

### 2026-08-22 03:27 - LoRA step-500 milestone

- Exact paired readout: Step 500 full-attention four-way CE was `1.4151834456` and accuracy was `0.278125`, improving CE by `0.0104941430` and gaining four correct targets relative to step 0.
- Schedule context: Future-edge probability was `0.0419953950` at the step-500 training readout.
- Stability: All 554 W&B optimizer rows through step 553 were present and finite; loss min/median/max was `1.272340/1.338690/1.589207`, latest-20 mean was `1.326702`, and the future-edge probability trace was monotone.
- Gate context: The exact readout remained far behind released source causal `1.051060/0.5078125` and did not pass.
- Issue record: Posted the milestone at https://github.com/Open-Athena/marin-dna/issues/479#issuecomment-5377607698.

### 2026-08-22 03:39 - LoRA step-600 milestone

- Exact paired readout: Step 600 full-attention four-way CE was `1.4153311488` and accuracy was `0.284375`, improving CE by `0.0103464398` and gaining eight correct targets relative to step 0.
- Local trajectory: CE changed by `+0.0001477032` and accuracy gained four correct targets from step 500, so the last 100 steps traded a statistically tiny CE reversal for a small accuracy gain.
- Schedule context: Future-edge probability was `0.087284` at optimizer step 600 and `0.092175` at the latest observed step 608.
- Stability: All 611 W&B optimizer rows observed through step 610 were finite; loss min/median/max was `1.267362/1.338044/1.589207`, latest-20 mean was `1.327053`, and the future-edge probability trace was monotone.
- Gate context: The exact readout remained far behind released source causal `1.051060/0.5078125` and did not pass.
- Interpretation boundary: This remains an intermediate observation before the schedule reaches full attention at step 800 and trains there for 200 steps.
- Issue record: Posted the milestone at https://github.com/Open-Athena/marin-dna/issues/479#issuecomment-5377633272.

### 2026-08-22 03:47 - LoRA step-700 milestone and primary-code audit

- Exact paired readout: Step 700 full-attention four-way CE was `1.4032503088` and accuracy was `0.2875`, improving CE by `0.0224272798` and gaining ten correct targets relative to step 0.
- Local trajectory: Relative to step 600, CE improved by `0.0120808401` and accuracy gained two correct targets as the scheduled future-edge probability rose from `0.087284` to approximately `0.357170`.
- Stability: All 702 W&B optimizer rows observed through step 701 were finite; loss min/median/max was `1.267362/1.337512/1.589207`, latest-20 mean was `1.330645`, latest-100 mean was `1.333240`, and the attention trace remained monotone.
- Gate context: The exact readout remained far behind released source causal `1.051060/0.5078125` and did not pass.
- Primary-code pin: Audited HKUNLP/DiffuLLaMA at commit `c17e897f6476c174b4623da594e4c65554f1613d`.
- Attention match: Its annealing sampler independently opens upper-triangular edges with one shared sequence matrix and keeps the causal lower triangle, matching the live implementation structurally.
- Objective difference: Its trainer samples one continuous `t` per sequence, masks each eligible token with probability `t`, shifts logits/targets by one, and weights selected-token CE by `1/t`; the live LoRA run instead holds the corruption rate at 20% and therefore does not test that reference objective.
- Issue record: Posted the milestone and pinned primary-code audit at https://github.com/Open-Athena/marin-dna/issues/479#issuecomment-5377668557.

### 2026-08-22 04:00 - LoRA step-800 full-attention boundary

- Exact paired readout: Step 800 full-attention four-way CE was `1.3800283802` and accuracy was `0.2984375`, improving CE by `0.0456492084` and gaining 17 correct targets relative to step 0.
- Local trajectory: Relative to step 700, CE improved by `0.0232219285` and accuracy gained seven correct targets while the attention schedule completed its transition.
- Phase boundary: The checkpoint is evaluated at full attention after 800 optimizer updates; updates 801 through 1,000 use exactly full attention, leaving 200 full-attention updates before the final gate.
- Stability: All 809 W&B optimizer rows observed through step 808 were finite; overall loss median/max was `1.337762/1.589207`, latest-20 mean was `1.339518`, and the attention trace reached exactly 1.0 without a loss spike.
- Gate context: The checkpoint remained far behind released source causal `1.051060/0.5078125`; final interpretation remains deferred to step 1,000 and the paired intervals.
- Issue record: Posted the full-attention boundary milestone at https://github.com/Open-Athena/marin-dna/issues/479#issuecomment-5377714520.

### 2026-08-22 04:08 - LoRA step-900 milestone

- Exact paired readout: Step 900 full-attention four-way CE was `1.3684129107` and accuracy was `0.3109375`, improving CE by `0.0572646779` and gaining 25 correct targets relative to step 0.
- Full-attention phase: Relative to step 800, the first 100 exactly full-attention updates improved CE by `0.0116154695` and gained eight correct targets.
- Stability: All 902 W&B optimizer rows observed through step 901 were finite; the 102 rows at or after step 800 had mean/min/max loss `1.338354/1.300498/1.374053`.
- Gate context: Step 900 had 199/640 correct targets versus 325/640 for the released causal source, so the trajectory was healthy but the final source-matching gate was overwhelmingly unlikely to pass in the remaining 100 cooldown updates.
- Interpretation boundary: Complete the preregistered final checkpoint, bootstrap intervals, source preservation check, artifact publication, and independent reload audit before selecting the next sequential run.
- Issue record: Posted the milestone at https://github.com/Open-Athena/marin-dna/issues/479#issuecomment-5377753096.

### 2026-08-22 04:13 - First LoRA attempt failed during post-training finalization

- Training completion: All 1,000 optimizer updates completed, and the retained step-1,000 adapter recorded preliminary full-attention CE `1.3655851085` and accuracy `0.31875` on the paired panel.
- Failure point: After `Trainer.fit` returned, Lightning teardown had moved the model to CPU; the subsequent disabled-adapter source check passed CUDA inputs to the CPU embedding and raised a device-mismatch error.
- Impact boundary: Training, intermediate readouts, and the step-1,000 adapter artifact completed; the failure occurred before source-preservation verification, paired bootstrap/gate publication, gradient-trace export, optimizer-checkpoint publication, retention manifest, and evaluation artifact.
- Retention: The failed-attempt step-1,000 adapter remains W&B artifact `dna-exp479-lora-r16-mntp-unk-step-1000:v0` with artifact ID `QXJ0aWZhY3Q6MzM3MTUyMDg5MQ==`; no deletion occurred.
- Fix: Move the bundle explicitly to the requested evaluation device inside every paired evaluation helper call, preserve train/eval mode, add focused regression coverage, and stream pre-clipping norm, clipping indicator, and learning rate to W&B so a late finalizer failure cannot erase the stability trace again.
- Relaunch rule: Treat this as a failed attempt and repeat the identical deterministic 1,000-step run only after the full locked remote suite passes; do not use the preliminary endpoint as the registered gate result.
- Cost: The failed attempt added `$1.803845571`, bringing cumulative listed cost to `$34.995182571 / $50`.
- Teardown: The AWS cluster automatically terminated after failure.
- Issue record: Posted the failure, impact boundary, retained artifact, fix, and relaunch plan at https://github.com/Open-Athena/marin-dna/issues/479#issuecomment-5377826904.

### 2026-08-22 04:23 - Corrected deterministic LoRA replay launched

- Immutable source: Launched tag `exp479-lora-damage-calibrated-v2` at commit `b25d46b39b6d696bafdce09fa1dcb4603750c971` on one AWS `g5.xlarge` A10G.
- Verification: All 151 locked tests passed on the remote worker before training.
- Identity: The train plan retained SHA-256 `9c715b08dad078c8ae5cf06325d4917051f52453f048674f6507ef6563130b91`, the validation plan retained SHA-256 `35542611d71102479f3d07dc6565350120d1d89944e5a93f88efb641ece7e3ba`, and the seed remains zero.
- Change boundary: The optimizer, learning-rate schedule, masking policy, attention schedule, data, and 1,000-step design are unchanged; only post-training evaluation device placement and durable W&B gradient telemetry changed.
- W&B: The corrected run is https://wandb.ai/gonzalobenegas/marin/runs/tnkdn3v3.
- Issue record: Posted the corrected replay at https://github.com/Open-Athena/marin-dna/issues/479#issuecomment-5377852127.

### 2026-08-22 04:34 - Corrected warmup and executable LLM2Vec audit

- Deterministic replay: The corrected step-0, step-25, step-50, and step-100 paired readouts match the failed attempt exactly; step 100 is CE `1.4256078663` and accuracy `0.2765625`.
- Live stability: The first 92 streamed pre-clipping norms were all finite with median/p95/maximum `0.4983/0.5979/0.6941`, and none clipped while the learning rate approached `1e-5`.
- Loss and attention: The first 124 training losses were finite with min/median/max `1.3134/1.4464/1.5892` and latest-20 mean `1.3408`; the future-edge trace was monotone and in `[0, 1]`.
- Reference exposure: Pinned LLM2Vec uses 32 sequences of length 512 for 1,000 MNTP steps, matching this pilot's 64 sequences of length 256 at `16,384` model tokens per optimizer step.
- Reference update scale: Its executable code sets LoRA alpha to twice rank, so rank 16 means alpha 32, and its config inherits Transformers' `5e-5` learning rate and linear decay rather than this pilot's alpha 16, `1e-5`, and WSD.
- Anti-leak boundary: LLM2Vec provides an all-mask collator but its published configuration selects the default 80/10/10 MLM collator; retaining 100% `[UNK]` replacement is necessary here because an unchanged selected nucleotide is directly visible under full attention.
- Sequential fallback: If the registered gate fails after reload validation, the leading next test is a small reference-strength, full-attention LoRA probe with the anti-leak replacement retained; do not launch it before finalization.

### 2026-08-22 05:21 - Localized predictor-row attention control preregistered

- Literature trigger: PreDiff-LM reports that uniform full attention can perturb pretrained prompt computation and instead keeps a prompt prefix causal while opening attention within the target suffix.
- Applicability boundary: Exp479 masks arbitrary interior nucleotides and reads the shifted predictor row, so the paper's prompt-prefix/target-suffix mask cannot be copied literally.
- Diagnostic inference: Keep every query causal except the exact shifted predictor row `i-1`, which can attend to every non-padding key, including right context.
- Controls: Compare standard causal, an additive-mask causal parity control, localized predictor-row attention, and uniform full attention on the same 640 frozen-source paired targets under math SDPA.
- Encoding gate: Require zero top-1 prediction mismatches and maximum per-target four-way CE drift below `0.002` between standard and additive causal encodings.
- Information gate: Require localized attention to have paired four-way CE no higher and top-1 nucleotide accuracy no lower than causal with 95% sequence-bootstrap support.
- Immutable source: Commit `eb0db004b6418e9db1f7db71c8041983c4a90a9a`, tag `exp479-localized-predictor-attention-zero-training-v1`.
- Compute: The one-hour AWS `g5.xlarge` A10G configuration passed a no-cost SkyPilot dry run at the pinned commit.
- Sequence: Launch only after the corrected LoRA replay and its independent final-adapter reload audit complete.
- Boundaries: No training, VEP, nucleotide dependency, Hugging Face upload, checkpoint deletion, or knowledge-base update occurs in this diagnostic.
- Issue record: Posted the preregistration at https://github.com/Open-Athena/marin-dna/issues/479#issuecomment-5378162150.

### 2026-08-22 05:27 - Causal-path-preserving conversion methods ranked

- Dec2Enc: Retain causal attention, compute an additional right-side/full-attention output, and learn a zero-initialized gate so the first forward pass is algebraically causal rather than the damaged uniform-full endpoint.
- Executable check: The released Qwen2 implementation contains the zero-initialized gate and explicit causal/full attention mixture described by the paper.
- Bitune: Independently retains causal and bidirectional feature paths, gives them separate PEFT-adapted parameters, and learns their mixture.
- Two-Pass FCM: Duplicate the sequence under an unchanged causal mask so the second copy can use right context from the first, but its from-scratch training, new `[copy]` sentinel, and doubled length make it a secondary diagnostic here.
- Controlled CLM-to-MLM evidence: A 38-model study supports continued masked adaptation from causal initialization, but its 22,000-step full-model setting is much larger than the current 1,000-step LoRA sanity check.
- Ranked fallback: If the paired gate and localized zero-training control fail, prefer a zero-initialized gated right-attention residual with LoRA on the new path over another abrupt uniform-full transition.
- Boundary: Do not implement or launch the fallback until the running replay, reload audit, and localized frozen-source diagnostic resolve.
- Issue record: Posted the cited research synthesis at https://github.com/Open-Athena/marin-dna/issues/479#issuecomment-5378188024.

### 2026-08-22 06:07 - Corrected damage-calibrated LoRA replay completed and failed the paired gate

- Completion: W&B run `tnkdn3v3` completed all 1,000 optimizer steps and finalized the paired evaluation, gradient trace, retention manifest, adapter-plus-optimizer checkpoint, and result artifact.
- Determinism: Every retained paired checkpoint from step 0 through step 1,000 exactly matches the first attempt's preliminary trajectory.
- Source baseline: The disabled-adapter causal readout was four-way CE `1.0507945247` and accuracy `0.50625` on the exact 640 targets.
- Final readout: The step-1,000 full-attention LoRA readout was CE `1.3655851085` and accuracy `0.31875`.
- Paired CE gate: Candidate-minus-source CE was `+0.3147905837` with 95% sequence-bootstrap interval `[+0.2675381738, +0.3648440462]`.
- Paired accuracy gate: Candidate-minus-source accuracy was `-0.1875` with 95% interval `[-0.2359375, -0.1421875]`.
- Decision: Both point-estimate and confidence-supported information gates failed decisively.
- Source preservation: Disabling the adapter after step 1,000 reproduced the step-0 causal source scores bit-exactly.
- Stability: Exactly 1,000 trace rows cover steps 0 through 999; every loss and norm is finite, pre-clipping norm median/p95/maximum is `0.075796/0.485920/0.694055`, and zero steps clipped.
- Schedule audit: Future-edge probability is monotone from exactly zero to exactly one, reaches one at step 800, and remains there through step 999; learning rate follows the registered 100-step warmup, plateau through step 800, and cooldown.
- Retention: Thirteen corrected adapter artifacts plus the final adapter/optimizer/RNG artifact are retained; the final adapter is W&B artifact ID `QXJ0aWZhY3Q6MzM3MjYzMDc1OA==`, and the optimizer checkpoint is `QXJ0aWZhY3Q6MzM3MjYzNTU2Mw==`.
- Runtime: Training and registered evaluation took `6114.68` seconds at `2679.45` model tokens/second, with peak allocated/reserved CUDA memory `7.28/7.57` GB.
- Cost: The final pre-autodown listed-price estimate is `$36.803058 / $50` cumulative, and the AWS cluster was confirmed absent.
- Evidence: Compact files are in `.agents/artifacts/479-mntp-adaptation/lora-mntp/`; the W&B evaluation artifact ID is `QXJ0aWZhY3Q6MzM3MjYzNjYyMw==`.
- Boundaries: No VEP, nucleotide dependency, Hugging Face upload, checkpoint deletion, or knowledge-base update occurred.
- Next step: Run the preregistered fresh-process final-adapter reload parity audit, then the frozen localized predictor-row diagnostic.

### 2026-08-22 06:16 - Unmatched numeric-state reload audit rejected

- Initial audit: W&B run `0c10doln` reloaded the retained final adapter and source in a fresh process but omitted numeric controls active during the registered Lightning evaluation.
- Observed drift: Adapter four-way CE had maximum absolute drift `0.0010141134` and source CE had maximum absolute drift `0.0054775476`; neither path had a top-1 mismatch on 640 targets.
- Attention control: Forced-math standard full attention and the additive-mask training encoding matched exactly, excluding attention-mask encoding as the source of the discrepancy.
- Artifact identity: The failed-attempt and corrected final `adapter_model.safetensors` files have the same W&B digest `gsA7VpJe+pg3rLi4RXsktA==` and byte size `53,733,848`; only nondeterministic JSON target-module ordering differed.
- Decision: Reject the v2 audit as an unmatched execution-state comparison and rerun with the training-time numeric controls set before model loading.
- Cost: The audit added approximately `$0.108837`, bringing cumulative listed cost to `$36.911895 / $50`; the AWS cluster was confirmed absent.
- Issue record: Posted the failure diagnosis and immutable rerun plan at https://github.com/Open-Athena/marin-dna/issues/479#issuecomment-5378420589.

### 2026-08-22 06:29 - Matched numeric-state reload audit passed exactly

- Immutable source: W&B run `r9m9m9gj` used commit `6431bdc20c249f98ec7de32f0af11d0890b427c5`, tagged `exp479-lora-reload-audit-v3`.
- Verification: All 159 locked tests passed on the remote worker before the audit.
- Numeric controls: The fresh process matched `float32_matmul_precision=high`, deterministic algorithms, disabled cuDNN benchmarking, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before loading either model.
- Adapter parity: All 640 freshly reloaded final-adapter scores matched the retained training-process scores with maximum absolute four-way CE drift `4.44e-16` and zero top-1 mismatches.
- Source parity: All 640 freshly loaded frozen-source scores matched with maximum absolute four-way CE drift `4.44e-16` and zero top-1 mismatches.
- Attention parity: Forced-math standard full attention and the additive-mask training encoding were exactly equal with zero CE drift and zero top-1 mismatches.
- Contract audit: The source tokenizer is exactly vocabulary size 7 with PAD `0`, UNK `1`, BOS `2`, and no EOS; the adapter is rank 16, alpha 16, dropout 0.05 over q/k/v/o/gate/up/down projections, with no trainable parameters after reload.
- Conclusion: Saving and reloading does not degrade the model; the apparent v2 drift was entirely due to unmatched numeric execution state.
- Evidence: Compact files are in `.agents/artifacts/479-mntp-adaptation/lora-reload-audit/`; the W&B audit artifact ID is `QXJ0aWZhY3Q6MzM3Mjg2NjM0OA==`.
- Cost: The final pre-autodown cumulative listed-price estimate was `$37.010826 / $50`, and the AWS cluster was confirmed absent.
- Boundaries: No VEP, nucleotide dependency, Hugging Face upload, checkpoint deletion, or knowledge-base update occurred.
- Next step: Launch the preregistered frozen-source localized predictor-row attention diagnostic before choosing any further training design.

### 2026-08-22 06:44 - Frozen localized predictor-row attention failed the paired gate

- Immutable source: W&B run `dq67i9vi` used commit `eb0db004b6418e9db1f7db71c8041983c4a90a9a`, tagged `exp479-localized-predictor-attention-zero-training-v1`.
- Verification: All 158 locked tests passed on the remote worker, and the validation-plan SHA-256 remained `35542611d71102479f3d07dc6565350120d1d89944e5a93f88efb641ece7e3ba`.
- Encoding control: Standard causal attention and an explicit causal additive mask matched exactly on all 640 targets with zero CE drift and zero top-1 mismatches under forced math SDPA.
- Causal readout: Four-way nucleotide CE was `1.0516807109` and accuracy was `0.5109375`.
- Localized readout: Opening only the exact shifted predictor row produced CE `1.2071856022` and accuracy `0.4484375`.
- Uniform-full readout: Opening every query row produced CE `1.4246315002` and accuracy `0.2703125`.
- Paired localized effect: Candidate-minus-causal CE was `+0.1555048913` with 95% sequence-bootstrap interval `[+0.1007878925, +0.2076929440]`.
- Paired localized accuracy: Candidate-minus-causal accuracy was `-0.0625` with 95% interval `[-0.0953515625, -0.028125]`.
- Interpretation: Localization recovers about 58% of the uniform-full CE damage and 74% of its accuracy damage, but both registered non-inferiority criteria still fail decisively.
- Scope: The result rules out collateral perturbation of unrelated query rows as a sufficient explanation; an unadapted causal decoder also does not interpret right-context keys safely at the predictor row itself.
- Runtime and cost: Evaluation took `74.96` seconds; final pre-autodown cumulative listed cost was approximately `$37.091314 / $50`, and the AWS cluster was confirmed absent.
- Evidence: Compact files are in `.agents/artifacts/479-mntp-adaptation/localized-attention/`.
- Boundaries: Zero parameter updates occurred, and no VEP, nucleotide dependency, Hugging Face upload, checkpoint deletion, or knowledge-base update occurred.
- Next step: Preregister a frozen-causal, separate full-attention LoRA branch with an exactly zero-initialized learned gate and a right-context-removal control.

### 2026-08-22 07:14 - Causal-preserving gated dual-path LoRA preregistered

- Trigger: Frozen uniform-full attention lost `0.372951` CE and 24.06 accuracy points versus causal, while predictor-row localization still lost `0.155505` CE and 6.25 points; neither raw attention intervention is information-safe.
- Architecture: Run the unchanged adapter-disabled source causally, run a separate full-attention rank-16 LoRA branch on the same `[UNK]`-corrupted input, and mix logits as `causal + tanh(causal_logits @ gate) * (branch - causal)`.
- Exact initialization: The seven-value gate vector is initialized to exactly zero, so candidate logits are algebraically and runtime-checked bit-exact to the causal source at step 0.
- Gradient path: The gated candidate loss trains the gate, and an equally weighted auxiliary full-branch MNTP loss trains the LoRA matrices even while the gate is initially closed; the causal source is detached and frozen.
- Training contract: Use the human-selected `1e-5` AdamW rate, betas `(0.9, 0.95)`, epsilon `1e-8`, zero weight decay, gradient clipping at 1.0, 100-step warmup, constant rate through step 800, and cooldown through step 1,000.
- Data contract: Reuse effective batch 64, the exact registered training and 640-row validation plans, fixed 20% masking, 100% `[UNK]` replacement, shifted supervision, and one repeat-weight application with per-sequence effective-weight normalization.
- Source gate: Final paired four-way nucleotide CE must be no higher and accuracy no lower than the frozen causal source, with 95% sequence-bootstrap support.
- Use gate: The final full candidate must be non-inferior on both metrics and strictly improve at least one with 95% support versus the same trained candidate with its LoRA branch forced causal; equality from a closed gate explicitly fails.
- Trajectories: Retain and evaluate steps 0, 25, 50, 100, every 100 steps through 900, and 1,000; record gated and auxiliary losses, accuracy, gate coefficients, separated LoRA/gate gradient norms, clipping, and learning rate at every optimizer step.
- Literature boundary: This is a conservative output-level proxy inspired by Dec2Enc and Bitune, not a reproduction; Dec2Enc mixes causal and right-attention contributions inside layers, while this diagnostic mixes separately computed logits.
- Retention: Upload adapter-plus-gate snapshots and the final adapter/gate/optimizer/RNG checkpoint to W&B; perform no Hugging Face upload or checkpoint deletion.
- Budget: One self-terminating AWS `g5.xlarge` A10G has a four-hour ceiling at `$1.006/hour`, projecting at most `$41.115314 / $50` from the current `$37.091314` cumulative estimate.
- Decision boundary: Run no VEP, nucleotide dependency, or knowledge-base update unless both paired gates pass and fresh-process reload parity is subsequently established.
