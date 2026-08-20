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

Implementation is in progress. The execution target is a self-contained Hugging Face/PyTorch Lightning project on one Lambda GH200; Marin/Iris is not involved. Two source facts refine the issue text: the released checkpoint declares untied input and output matrices, and the pinned training datasets already encode the source orientation policy.

## Current baseline

- Source checkpoint: [`marin-dna/marin-dna-exp135-m5.1@a73a5dcf`](https://huggingface.co/marin-dna/marin-dna-exp135-m5.1/tree/a73a5dcfb3d64b8941e7e7596c6e88ef77db3e7a), step 59,158, 1,120,772,224 parameters.
- Architecture: Qwen3, 19 layers, hidden size 1,920, intermediate size 7,680, 15 attention/KV heads, 256-token context.
- Vocabulary: `[PAD]`, `[UNK]`, `[BOS]`, A, C, G, T. The tokenizer lowercases input.
- Current Lambda list price: $2.29/GH200-hour before applicable tax, checked 2026-08-19.
- No adaptation, behavioral smoke test, or labeled evaluation has run.

## Hypothesis queue

### Active

- `MNTP-479-H1`: Transferred MNTP reaches lower pooled and single-mask validation loss than scratch MNTP within 1,000 steps. Next test: matched transferred and scratch arms.
- `MNTP-479-H2`: Full-attention MNTP creates measurable dependence on both flanks beyond the causal and no-adaptation controls. Next test: behavioral preflight followed by fixed perturbation probes every 100 steps.
- `MNTP-479-H3`: The cooled transferred checkpoint improves at least one odd-autosome/X VEP endpoint or a scoped mechanistic diagnostic over source CLM FWD+RC. Next test: run the registered VEP and nucleotide-dependency panels after training.

### Blocked

- None.

### Falsified / dead end

- None.

### Promoted

- None.

## Decision log

- 2026-08-19: Use a Lambda GH200 sidecar. Do not introduce Marin/Iris launch dependencies.
- 2026-08-19: Preserve the released checkpoint's untied input embedding and LM head. Initialize the new `[MASK]` input row and output row separately from their respective A/C/G/T means. Tying the matrices would alter the source model before adaptation and invalidate causal-logit parity.
- 2026-08-19: Do not add runtime reverse-complement augmentation. The three genomes-v5 datasets state that reverse complements are included, and the two pinned Zoonomia partitions inherit the m5.1 source construction. The experiment samples the pinned stored examples directly.

## Negative results index

- No experiment results yet.

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
