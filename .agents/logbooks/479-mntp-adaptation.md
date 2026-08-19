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
