# Experiment 479: m5.1 MNTP adaptation pilot

This permanent experiment project implements [issue #479](https://github.com/Open-Athena/marin-dna/issues/479): three matched 1,000-step training arms and two no-training controls for converting the released 1B m5.1 causal checkpoint into a full-attention masked-next-token-prediction model.

The runtime is one Lambda Cloud GH200 managed through SkyPilot. Marin and Iris are not dependencies of this experiment. Every numbered checkpoint is saved on the task disk and every validation boundary is logged to W&B; selected restart milestones are copied to reviewed private Hugging Face staging before Sky tears the instance down.

## Current experiment status

The one-seed pilot and initial follow-up causal runs used an incorrect repeat-weight denominator.
The reducer divided weighted loss by selected-token count instead of effective-weight sum, shrinking the loss according to lowercase-repeat content.
The original source validation suite also contains three datasets, while the exp479 five-way panel added enhancer and ncRNA probes.
The corrected 128-row-per-dataset audit reproduces the invalid source value of `0.231380263`, reports validation CE of `0.764633691` on the three original datasets, and preserves a small worsening through step 1,000.
The full 49,152-row source audit also found that the original tagged evaluator applied repeat weights twice for its mixed-case slices.
It reproduced the original nine-metric W&B macro as `0.861413936` versus `0.861344755`, with maximum metric error `0.000168145`, while the corrected single-weight macro is `0.875662646`.
A source-compatible 1,000-step AdamW replacement is complete.
Its five-component macro validation CE improved from `0.769008732` to `0.767801766` by step 100, then rose to `0.773670488` at step 1,000 with finite loss and gradients.
The VEP, attention, coordinate, serialization, and final dependency evidence is unchanged by this discovery, but interpretation remains paused.
The current conservative list-price estimate is $28.3080 of the $50 cap, and every Lambda cluster was confirmed terminated.

See the [compact result bundle](../../.agents/artifacts/479-mntp-adaptation/README.md), [corrected causal run](https://wandb.ai/gonzalobenegas/marin/runs/f77ypos4), [full source-validation reproduction](https://wandb.ai/gonzalobenegas/marin/runs/hfuhn3ta), [checkpoint audit](https://wandb.ai/gonzalobenegas/marin/runs/gavkgtmf), [stability audit](https://wandb.ai/gonzalobenegas/marin/runs/q67hbkp4), and [final dependency run](https://wandb.ai/gonzalobenegas/marin/runs/yl5sgffn).
Final weights and per-variant scores remain private.

## Causal fine-tuning sanity gate

The audited continued-CLM arm used a fresh optimizer with peak learning rates of 0.00440 and 0.0231, and fixed-plan validation loss rose from 0.23138 to 0.35965 by step 800.
That arm does not establish how the mature checkpoint behaves under ordinary low-learning-rate fine-tuning.
The `calibration` stage runs one conservative full-parameter AdamW arm at `1e-6` for 200 steps before considering any other learning rate.

The stage fixes:

- AdamW betas `(0.9, 0.95)`, epsilon `1e-8`, zero weight decay, and global gradient clipping at 1.0;
- a 10-step linear warmup followed by constant `1e-6`;
- the original batch-64 tokenizer, first 200 training batches, orientation policy, lowercase repeat weight, and 640-row five-component validation plan; and
- post-hoc validation at steps 0, 1, 10, 25, 50, 100, and 200.

The arm passes only when step-200 macro and component losses are no higher than their step-0 values and each component trajectory has a non-positive fitted slope.
This stage does not run VEP or any other learning rate.
It uploads the final checkpoint and compact validation evidence to private staging, records dense trajectories in W&B, and terminates through `sky launch --down`.

```bash
uv run --locked python launch.py calibration \
  --commit "$(git rev-parse HEAD)" \
  --hf-repo-id gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover \
  --prior-cost-usd 24.7340 \
  --execute
```

## Selected 1,000-step causal trajectory

The `longrun` stage is the corrected replacement for the superseded 1,000-step AdamW run at the human-selected peak learning rate of `1e-5`.
It starts from the released source checkpoint and reuses the batch-64 training plan, tokenizer, orientation policy, and lowercase repeat weights.
Training applies each repeat weight once, divides by effective-weight sum, and includes the pinned source z-loss weight.
The full-parameter AdamW constants remain betas `(0.9, 0.95)`, epsilon `1e-8`, zero weight decay, and global gradient clipping at `1.0`.
The schedule warms linearly from zero through step 100, remains at `1e-5` through step 800, and decays linearly to zero at the step-1,000 boundary.

Validation reports only the equally weighted macro of the five fixed component losses.
Each component loss is a global effective-weight mean of pure cross-entropy with one repeat-weight application and no training-only z penalty.
The trajectory is evaluated at steps 0, 25, 50, 100, every 100 steps through 800, 900, and 1,000.

Every post-update Hugging Face-format trajectory export is retained immediately in the distinct W&B namespace `dna-exp479-causal-longrun-corrected-*`.
After training, the complete step-1,000 Lightning checkpoint containing model, optimizer, scheduler, and loop state is retained as a separate W&B model artifact before validation begins.
The Lambda task does not upload any output to Hugging Face and does not delete a retained checkpoint.

```bash
uv run --locked python launch.py longrun \
  --commit "$(git rev-parse HEAD)" \
  --prior-cost-usd 27.033784759 \
  --retry-until-up \
  --execute
```

### Corrected result

The corrected run completed from snapshot `42fc993e3245a0f6a1c1d77813b0665ef56e68e5` at [W&B f77ypos4](https://wandb.ai/gonzalobenegas/marin/runs/f77ypos4).
All 104 locked tests passed remotely before model loading, and the regenerated train and validation plan hashes matched the preregistered inputs.

Five-component macro validation CE was `0.769008732` at source, reached its minimum of `0.767801766` at step 100, crossed above source between steps 200 and 300, peaked at `0.774135425` at step 900, and ended at `0.773670488`.
The final increase was `+0.004661756`, and the fitted slope was `+6.1692e-6` per optimizer step.
Step 900 to 1,000 recovered `0.000464937` without returning to source.
The step-0 value differs by only `6.4e-6` from the earlier independent evaluator, and the corrected trajectory differs from the same metric on the superseded checkpoints by at most `0.000136974` across all 13 points.

All 1,000 training-loss, learning-rate, and pre-clipping gradient rows are present and finite.
Successive 100-step mean training losses stayed within `1.0216–1.0308`.
Pre-clipping gradient norm had median/p95/maximum `0.7674/0.8845/1.3722`, with six clipped steps: 238, 265, 646, 682, 738, and 867.

The run retained 12 Hugging Face-format trajectory exports and the full step-1,000 Lightning checkpoint as 13 committed W&B model artifacts totaling 67.25 GB.
The full checkpoint includes model, optimizer, scheduler, and loop state at `gonzalobenegas/marin/dna-exp479-causal-longrun-corrected-step-1000-full:v0`.
No checkpoint was uploaded to Hugging Face or deleted.

The 1,000 training steps took `1,306.77` seconds including synchronous artifact retention and processed 16,384,000 model tokens at 12,537.74 tokens/s.
The complete Lambda task cost an estimated `$1.274169`, bringing the conservative listed-price total to `$28.307954 / $50`.
The cluster self-terminated and is confirmed absent.

Exact CSV, JSON, SVG, and PNG outputs are in `causal-longrun-lr1e-5-corrected/` under the compact branch artifact bundle.
This is a factual experiment record, and research knowledge-base interpretation remains paused.

### Superseded count-normalized result

The exact run completed all 1,000 optimizer steps, but its reported validation gate used the invalid count-normalized loss.
Loss reached its minimum of `0.230961750` at the end of warmup, then rose progressively to `0.233014855` at step 1,000 versus `0.231380263` at step 0.
The final increase was `0.001634592`.
Cooldown produced a small recovery from `0.233230022` at step 900.

All numeric training and gradient values were finite.
Pre-clipping gradient norm had median/p95/maximum `0.6599/0.7577/1.3261`.
Only steps 265 and 738 exceeded the `1.0` clip threshold, and neither coincided with a training-loss spike.
Successive 100-step training-loss means stayed within `0.8696–0.8827`, so the validation increase was progressive rather than an optimization blow-up.

The W&B run retained 12 Hugging Face-format trajectory exports plus one full step-1,000 Lightning optimizer checkpoint as 13 model artifacts totaling 67.25 GB.
No checkpoint was deleted and no output was uploaded to Hugging Face.
The run cost an estimated `$0.8613`, bringing the conservative listed-price total to `$26.1237 / $50`.
The Lambda cluster self-terminated and was confirmed absent.

These values remain useful only for reproducing the bug and checking whether trajectory direction survives corrected evaluation.
Direct evidence is at [W&B run 5lbazal6](https://wandb.ai/gonzalobenegas/marin/runs/5lbazal6) and in the branch artifact directory `causal-longrun-lr1e-5/`.
This is a factual experiment record; research knowledge-base interpretation remains paused.

## Selected 1,000-step transferred-MNTP trajectory

The `mntp-longrun` stage applies the corrected causal fine-tuning setup to one transferred-MNTP arm.
It starts from the untouched released CLM checkpoint, adds `[MASK]` with independent input and output rows initialized from the corresponding A/C/G/T means, and uses explicit full attention.
It uses full-parameter AdamW at `1e-5`, betas `(0.9, 0.95)`, epsilon `1e-8`, zero weight decay, global gradient clipping at `1.0`, and the same 10% warmup, 70% constant, and 20% decay schedule.
Training applies each repeat weight once, normalizes each sequence by its effective-weight sum before averaging sequences, and includes the pinned source z-loss.

Post-hoc validation reports the equal five-component macro for both deterministic diffusion masks and deterministic single masks at steps 0, 25, 50, 100, and every 100 steps thereafter.
The task computes registered FWD+RC AUPRC on odd-numbered autosomes and chromosome X at step 0 and every 100 steps.
The selected final-checkpoint example is the LDLR promoter, the default locus in the interactive nucleotide-dependency browser.
A focused evaluation loads the retained step-1,000 checkpoint without retraining and computes one directed map with the same paired-baseline implementation.
No held-out even-autosome or chromosome-Y labels are accessed.

Every numbered Hugging Face-format export and the full step-1,000 Lightning checkpoint are retained as W&B model artifacts.
The task does not upload to Hugging Face and does not delete a retained checkpoint.
The deleted historical spillover repository is not an input to this stage.

```bash
uv run --locked python launch.py mntp-longrun \
  --commit "$(git rev-parse HEAD)" \
  --prior-cost-usd 28.307954 \
  --retry-until-up \
  --execute
```

## Loss-normalization and source-parity audits

The `loss-normalization` stage evaluates the released source plus every retained causal checkpoint with both the invalid count denominator and the corrected Marin reducer.
It uses the immutable 128-row-per-dataset panel, limits the source-comparable macro to CDS, upstream, and downstream, and uploads only compact tables and plots to W&B.
The completed audit reports corrected source-three validation CE of `0.764633691` at step 0, `0.763708129` at step 100, and `0.767572101` at step 1,000.
No checkpoint was deleted, modified, or uploaded elsewhere.
Direct evidence is at [W&B run v6mo9gh3](https://wandb.ai/gonzalobenegas/marin/runs/v6mo9gh3).

The `source-validation` stage evaluates all 16,384 rows in each of the three original source validation datasets.
One model forward supplies each dataset's repeat-weighted, uppercase-only, and lowercase-only CE.
The pinned tagged evaluator applied repeat weights inside the loss function and again in its accumulator, squaring them in the numerator but not the denominator.
The uppercase-only and lowercase-only metrics are unaffected because their weights are binary.
The stage compares the exact historical double-weight outputs with W&B at an absolute tolerance of `0.002` and separately reports corrected single-weight CE.
This is a hard gate for the source checkpoint's tokenization, next-token shift, special-token handling, repeat weights, and global reduction.
It uses public Hugging Face inputs, forwards only the W&B secret, uploads no checkpoint, and self-terminates.
The completed gate passed all nine metrics over 49,152 rows.
Its reproduced historical macro is `0.861413936` versus `0.861344755` in W&B, and its largest metric delta is `0.000168145`.
The corrected single-weight macro is `0.875662646`, so the historical evaluator biased the nine-metric macro downward by `0.014248710`.
Direct evidence is at [W&B run hfuhn3ta](https://wandb.ai/gonzalobenegas/marin/runs/hfuhn3ta) and in `source-validation-reproduction/` within the compact result bundle.

```bash
uv run --locked python launch.py source-validation \
  --commit "$(git rev-parse HEAD)" \
  --prior-cost-usd 26.744105814380116 \
  --retry-until-up \
  --execute
```

## Frozen reflected-RoPE BICO diagnostic

The `bico-attention-diagnostic` stage tests a positional-attention detail that every earlier full-attention arm omitted.
[BICO](https://aclanthology.org/2024.emnlp-main.754/) observes that opening future keys in a RoPE causal decoder exposes positive relative positions absent from causal pretraining.
For every future key, this diagnostic reflects the RoPE distance so both left and right keys use non-positive relative distances.
It also tests BICO's `[PAD]` replacement with the selected masked token excluded as an attention key.

The frozen source is evaluated on the same 640 deterministic single-mask targets under standard causal attention, standard full attention, reflected-RoPE full attention, `[UNK]`, attended `[PAD]`, and excluded `[PAD]` controls.
The patched attention must reproduce standard eager causal predictions with zero nucleotide-prediction mismatches and at most `0.002` maximum CE difference.
The mechanism contrast is reflected versus standard RoPE with the same excluded-`[PAD]` mask.
The single-forward-pass gate compares reflected-RoPE full attention with excluded `[PAD]` directly against the causal source.
The stage performs no training, VEP, nucleotide-dependency analysis, Hugging Face upload, or checkpoint deletion.
It runs on one Lambda GH200, the same device class selected for a successful mechanism's no-accumulation LoRA follow-up.

```bash
uv run --locked python launch.py bico-attention-diagnostic \
  --commit "$(git rev-parse HEAD)" \
  --prior-cost-usd 40.354899 \
  --retry-until-up \
  --execute
```

## Reflected-RoPE BICO LoRA

The `bico-lora-mntp` stage is the first trained follow-up to the BICO mechanism audit.
It runs the frozen diagnostic first, then selects the largest physical batch on one Lambda GH200 by completing two exact optimizer steps in a fresh process for every candidate.
The search brackets the feasible batch and binary-searches to the largest integer batch with finite loss and gradients, at least 10% CUDA memory headroom, and enough remaining issue budget for the projected run.
It reruns the selected batch immediately before training and uses no gradient accumulation.

The source weights remain frozen and rank-16 LoRA matrices are added to every registered attention and MLP projection.
Training uses BICO reflected future RoPE, fixed 15% MNTP corruption with the existing `[PAD]` token, and exclusion of every selected masked position as an attention key in every layer.
AdamW uses learning rate `1e-5`, 10% warmup, 70% constant rate, and 20% decay for 1,000 optimizer steps.
The run reports physical batch size, sequences, model tokens, exact supervised masked targets, validation CE and accuracy trajectories, pre-clipping gradient norms, and the paired final information gate.
Adapter checkpoints, final optimizer state, batch-selection preflights, and result tables are retained as W&B artifacts.
The stage performs no VEP, nucleotide-dependency analysis, Hugging Face upload, or checkpoint deletion.

```bash
uv run --locked python launch.py bico-lora-mntp \
  --commit "$(git rev-parse HEAD)" \
  --prior-cost-usd 40.354899 \
  --retry-until-up \
  --execute
```

The first combined launch selected batch 94 but exposed a temporary-hook lifetime bug during the first training backward pass, before any optimizer update.
The corrected `bico-lora-resume` stage keeps BICO attention installed across activation-checkpoint recomputation, reruns all locked tests, and rechecks only batch 94 before training.
It does not repeat the completed frozen diagnostic or maximum-batch search.

```bash
uv run --locked python launch.py bico-lora-resume \
  --commit "$(git rev-parse HEAD)" \
  --prior-cost-usd 40.85832745128643 \
  --retry-until-up \
  --execute
```

## Causal-preserving gated LoRA follow-up

The `gated-lora-mntp` stage is the sequential follow-up to the failed uniform-full LoRA and frozen localized-attention gates.
It leaves the released causal computation frozen and evaluates it with the adapter disabled on every forward pass.
A separate rank-16 LoRA branch uses full attention on the same `[UNK]`-corrupted input.
A seven-value zero-initialized projection of the causal source logits produces one token-wise `tanh` mixing coefficient.
The candidate is therefore algebraically identical to the causal source at step 0.

The gated candidate and full-attention branch receive equally weighted sequence-balanced, repeat-weighted MNTP losses.
The auxiliary branch loss gives the LoRA matrices a learning signal while the mixing gate is still exactly closed.
The causal source path remains detached and receives no updates.
Training uses AdamW at `1e-5`, 10% warmup, 70% constant rate, 20% decay, fixed 20% masking, and effective batch size 64 for 1,000 steps.

The final candidate must pass two gates on the same 640 deterministic single-mask targets.
First, it must have paired four-way nucleotide CE no higher and accuracy no lower than the frozen causal source, with 95% sequence-bootstrap support.
Second, full attention must be non-inferior on both metrics and strictly improve at least one with 95% support relative to the same trained candidate with its LoRA branch forced causal.
This second comparison prevents an exactly closed mixing gate from being counted as successful bidirectionality.

The design is a conservative output-level proxy inspired by [Dec2Enc](https://doi.org/10.1016/j.knosys.2024.112907) and [Bitune](https://arxiv.org/abs/2405.14862), not a reproduction of either architecture.
Dec2Enc mixes causal and right-attention contributions inside each layer, while this first gate mixes frozen-causal and separately adapted full-attention logits.
No VEP or nucleotide-dependency analysis runs until both paired gates pass.
Every adapter-plus-gate milestone and the final optimizer state are retained in W&B, with no Hugging Face upload or checkpoint deletion.

```bash
uv run --locked python launch.py gated-lora-mntp \
  --commit "$(git rev-parse HEAD)" \
  --prior-cost-usd 37.091314 \
  --retry-until-up \
  --execute
```

## Frozen two-causal-pass information gate

The `two-pass-information-gate` stage is a zero-training sequential diagnostic for use only after the gated LoRA run has a final disposition.
It leaves the released model unchanged and predicts each registered masked nucleotide once from its left context and once from its reverse-complemented right context.
The reverse pass maps target position `i` to `254 - i` and realigns its A/C/G/T columns with permutation `[3, 2, 1, 0]`.
Runtime assertions require reverse complementation to preserve the `[UNK]` mask and the alpha-zero readout to match the canonical causal evaluator bit-exactly.

The first 640 registered training-plan sequences form a calibration slice that is disjoint from the 640 validation sequences.
An empirical A/C/G/T prior uses a Jeffreys pseudocount of `0.5` per base.
The stage evaluates the fixed grid `alpha = 0, 0.001, ..., 1` for `left_logp + alpha * (right_logp - log_prior)` and selects the smallest alpha with minimum calibration CE.
The untouched validation panel remains the sole decision set.

The diagnostic passes only if the selected alpha is positive, the calibrated two-pass readout is confidence-supported non-inferior to left-causal prediction on both paired CE and accuracy, and at least one metric strictly improves with 95% support.
It publishes raw directional scores, the calibration curve, paired intervals, and one figure to W&B.
It performs no model update, VEP, nucleotide-dependency analysis, Hugging Face upload, or checkpoint deletion.

```bash
uv run --locked python launch.py two-pass-information-gate \
  --commit "$(git rev-parse HEAD)" \
  --prior-cost-usd <cumulative-cost-after-gated-run> \
  --retry-until-up \
  --execute
```

## Frozen two-pass VEP gate

The `two-pass-vep` stage runs only after the frozen paired nucleotide gate passes.
It performs zero training and scores the pinned odd-autosome/X development rows from the Mendelian, complex-trait, and SGE datasets.
It does not access even-autosome or chromosome-Y labels.

The primary conditional readout is strand-symmetric.
The same 640-row calibration slice selected `alpha = 0.408` before VEP labels are loaded by minimizing four-way nucleotide CE for the normalized geometric mean of the forward-anchored and reverse-anchored two-pass distributions.
On the already-registered nucleotide validation panel, that fixed readout reached CE `0.913447` and accuracy `0.625`.

For every central SNV, the stage masks the reference base with `[UNK]`, obtains one native causal distribution from the left context and one realigned native causal distribution from the reverse-complemented right context, and computes alternate-minus-reference log-probability ratios.
It also recomputes the original source CLM full-sequence FWD+RC score in the same process.
Runtime assertions cover the BOS/UNK/PAD IDs, input-to-output shift, central coordinate, `i -> 254 - i` reverse mapping, A/C/G/T realignment, reference-token identity, and exact double reverse complementation.

The gate uses the original Mendelian consequence macro, complex-trait global, and SGE accession-consequence macro AUPRC endpoints.
It passes only if paired primary AUPRC is confidence-supported non-inferior to source CLM FWD+RC on all three endpoints and strictly improves at least one.
It publishes raw development-split scores, metrics, paired comparisons, coordinate controls, runtimes, and one figure to W&B.
It performs no Hugging Face upload, nucleotide-dependency analysis, checkpoint deletion, or knowledge-base update.

```bash
uv run --locked python launch.py two-pass-vep \
  --commit "$(git rev-parse HEAD)" \
  --prior-cost-usd <cumulative-cost-after-two-pass-gate> \
  --retry-until-up \
  --execute
```

## Registered behavior

- Source checkpoint: `marin-dna/marin-dna-exp135-m5.1@a73a5dcfb3d64b8941e7e7596c6e88ef77db3e7a`.
- Context: one BOS plus 255 nucleotide bases.
- MNTP: sample one `Uniform(0, 1)` probability per sequence, select eligible A/C/G/T positions independently, resample zero-target rows, replace every target with `[MASK]`, and supervise target `i` from output `i - 1`.
- Training loss: normalize weighted cross-entropy by effective-weight sum; MNTP first normalizes within each sequence and then averages sequences, while continued CLM uses Marin's global token-weighted mean.
- Repeat weights: uppercase bases have weight 1 and lowercase bases have weight 0.01.
- Training regularization: include the pinned source z-loss weight `4.312883184368223e-6`.
- Validation loss: report pure CE without the training-only z penalty.
- Data: sample the five pinned m5.1 components uniformly. Deterministically skip any source row with no A/C/G/T base and draw the next row from that same component. One materialized plan fixes the underlying sequence and component order for every arm. Corruption is a stateless function of the plan sample ID.
- Optimizer: the pinned m5.1 DNA scaling heuristic supplies separate AdamH and Adam learning rates, betas, epsilon, and clipping. Linear weights use AdamH; embeddings, normalization weights, and biases use Adam. An actually tied embedding/head matrix would remain in the Adam group.
- Schedule: linear warmup through step 100, stable through step 800, and linear cooldown to zero at step 1,000.
- Checkpoints: full Lightning state every 100 steps. Step 800 is the only permitted source for a separately approved 10,000-step continuation.

The released checkpoint declares `tie_word_embeddings=false`. The new input embedding row and untied LM-head row are initialized separately from their respective A/C/G/T means. Tying the matrices would alter source logits before adaptation.

The pinned training datasets already encode the m5.1 orientation policy; the genomes-v5 cards state that reverse complements are included. This project does not add runtime reverse-complement augmentation.

## Environment and local checks

From this directory:

```bash
uv sync --locked --group dev
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
```

The tests cover target shifting, special-token exclusion, zero-target handling, stateless corruption, per-sequence reduction, WSD boundaries, AdamH equations and norm preservation, parameter grouping, optimizer/scheduler checkpoint state, causal/full-attention behavior, and exact interrupted resumption.

## GH200 preflight

Preflight loads the pinned released checkpoint and must pass before any training arm starts. It checks:

- default-versus-explicit causal-logit parity;
- causal right-flank invariance and zero position-specific gradient;
- full-attention right-flank sensitivity and nonzero gradient;
- finite MNTP loss and gradients;
- AdamH norm preservation;
- the largest no-accumulation batch with at least 10% CUDA memory headroom;
- measured step throughput and a conservative total-cost projection below $50.

The launch helper prints commands unless `--execute` is set. It requires a clean, pushed commit because Sky clones the exact SHA.

```bash
uv run --locked python launch.py preflight --commit "$(git rev-parse HEAD)"
```

Validate the exact pilot request without provisioning or requiring model-card approval:

```bash
uv run --locked python launch.py pilot \
  --commit "$(git rev-parse HEAD)" \
  --execute --dry-run
```

Paid execution requires explicit approval from the user coordinating issue #479. The preflight task uses `sky launch --down`, so the Lambda instance is terminated when the task finishes. Provisioning or setup failure must still be checked in `sky status`; Sky documents that `--down` cannot tear down an instance it failed to configure.

After explicit approval of both the $50 cap and `README.hf.md`, the `pilot` stage runs preflight, all three trained arms, odd/X-only VEP, and the fixed nucleotide-dependency panel in one self-terminating task:

```bash
uv run --locked python launch.py pilot \
  --commit "$(git rev-parse HEAD)" \
  --model-card-reviewed
```

Add `--execute` only for the approved paid run. `HF_TOKEN` and `WANDB_API_KEY` are forwarded as Sky secrets; the launcher can read the existing Hugging Face token file and W&B netrc entry into the Sky subprocess environment without printing either value. The Hugging Face repository is created private and is not made public by this workflow.

The evaluation-only `diagnostics` stage loads the final transferred checkpoint, does not repeat training, and runs the registered context-ablation and window-shift diagnostics on the same odd/X-only VEP frames. It masks an entire left or right flank with the tokenizer's unknown-base token and moves the 255-base reference window 64 bases upstream or downstream while tracking the variant's exact 0-based position. The exact perturbation sizes were fixed after the primary evaluation and are therefore recorded as post-hoc diagnostics, not model-selection gates.

```bash
uv run --locked python launch.py diagnostics \
  --commit "$(git rev-parse HEAD)" \
  --hf-repo-id <private-final-repository> \
  --prior-cost-usd <conservative-cumulative-cost> \
  --execute
```

This stage forwards only `HF_TOKEN`, applies the same $50 cost guard, uploads its compact tables under `evaluation/context-window`, and uses `sky launch --down`.

## Checkpoint and alignment audit

The `audit` stage rebuilds the original batch-64 plans and requires their SHA-256 hashes before loading a model. It then:

- saves and reloads the source CLM checkpoint without an optimizer update;
- replays CLM steps 1, 5, 10, 25, 50, 100, 200, and 400 with the original 1,000-step optimizer schedule;
- compares replayed step 400 with the original full Lightning checkpoint;
- recomputes fixed-plan validation loss at every available checkpoint and compares original points with their W&B values;
- computes odd-autosome/X AUPRC in both reference and reverse-complement orientations;
- checks coordinate slicing and training-label/inference-readout alignment at nucleotide indices 0, 63, 127, 191, and 254;
- checks PAD, UNK, BOS, EOS, MASK, vocabulary, attention-mask, and true-base-under-MASK contracts;
- renders raw forward, registered reverse-complement, and FWD+RC nucleotide-dependency maps; and
- recomputes the tRNA-Arg-TCT map with full attention and a forced-causal negative control.

Per-variant scores and numeric dependency arrays are uploaded only to private Hugging Face staging. Compact tables and figures are logged to W&B. The task uploads parity failures before exiting nonzero and always runs the pre-autodown cost recorder.

```bash
uv run --locked python launch.py audit \
  --commit "$(git rev-parse HEAD)" \
  --hf-repo-id gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover \
  --prior-cost-usd 10.232556777459978 \
  --execute
```

The task requests one Lambda GH200 with a 256 GB disk and uses `sky launch --down`.

## Training-stability audit

The original runs log every per-step loss but do not log gradient norms. The
`stability` stage deterministically replays the first 400 optimizer steps of all
three arms—transferred MNTP, scratch MNTP, and continued CLM—with the original
batch plan, corruption, objective, optimizer, and 1,000-step WSD schedule. It
requires each replayed loss to match the corresponding original W&B history,
then records the global gradient norm before clipping, the clipping decision,
and both parameter-group learning rates. It publishes one compact three-arm
table and a matched-scale loss/gradient figure; it does not publish model
weights or per-example scores.

```bash
uv run --locked python launch.py stability \
  --commit "$(git rev-parse HEAD)" \
  --hf-repo-id gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover \
  --prior-cost-usd <conservative-cumulative-cost> \
  --retry-until-up \
  --execute
```

The task requests one Lambda GH200, enforces the same cumulative $50 guard,
uploads its compact evidence before exit, and uses `sky launch --down`.

## Final-checkpoint nucleotide dependency

The focused `dependency` stage computes one directed 255-by-255 map for the
step-1,000 checkpoint of each trained arm: transferred MNTP, scratch MNTP, and
continued CLM. It uses the reference orientation of the preregistered
`tRNA_Arg_TCT` locus. Each readout's wild-type baseline and all substitutions
are evaluated in the same model call, preventing BF16 batch-shape numerics from
being misreported as nucleotide dependency. MNTP readouts are masked and use
full attention; CLM uses its ordinary causal next-token readout. The causal map
must have exactly zero right-context dependency, while both MNTP maps must use
context on both sides.

```bash
uv run --locked python launch.py dependency \
  --commit "$(git rev-parse HEAD)" \
  --hf-repo-id gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover \
  --prior-cost-usd <conservative-cumulative-cost> \
  --execute
```

The stage requests one Lambda GH200 with a 64 GB disk, publishes only compact
tables and figures to W&B, keeps the raw numeric maps in private Hugging Face
staging, and uses `sky launch --down`.

## Corrected BICO LoRA gate audit

The maximum-batch BICO LoRA run used physical batch 94 with `accumulate_grad_batches=1`, but its first retained report mislabeled an adapter-disabled full-attention readout as causal because the custom SDPA hook ignored `is_causal`.
The training path and full-attention candidate trajectory are unaffected.
The `bico-lora-gate-audit` stage fixes the hook, reloads the retained step-1,000 adapter in a fresh process, reproduces the mislabeled row under full attention, and recomputes the paired trajectory against a fresh standard causal source.
It requires the corrected causal hook, bug reproduction, and final-adapter reload to match their controls within `0.002` maximum per-target CE with zero correctness mismatches.

```bash
uv run --locked python launch.py bico-lora-gate-audit \
  --commit "$(git rev-parse HEAD)" \
  --prior-cost-usd 42.139827271281725 \
  --retry-until-up \
  --execute
```

The stage uses one self-terminating AWS A10G, forwards only the W&B secret, preserves every retained checkpoint, and performs no VEP, nucleotide-dependency, Hugging Face, or knowledge-base action.

## Data plans

After preflight selects the batch size, materialize the shared plans on the GH200:

```bash
uv run --locked exp479 prepare-data \
  --output-dir ~/exp479-artifacts/data \
  --batch-size <selected-batch-size> \
  --seed 0
```

`train.jsonl` contains exactly `1,000 × batch size` rows, uniformly interleaved across the five components. `validation.jsonl` contains 128 fixed rows per component. Every run records both SHA-256 hashes.

## Train one arm

```bash
uv run --locked exp479 train \
  --arm transferred_mntp \
  --batch-size <selected-batch-size> \
  --train-plan ~/exp479-artifacts/data/train.jsonl \
  --validation-plan ~/exp479-artifacts/data/validation.jsonl \
  --output-dir ~/exp479-artifacts/transferred_mntp
```

Valid arms are `transferred_mntp`, `scratch_mntp`, and `clm_continuation`. `accumulate_grad_batches` is fixed to 1. W&B runs use project `marin`, group `dna-exp479`, and names of the form `dna-exp479-<arm>-seed0`.

## Output contract

Each trained arm writes:

- `checkpoints/step-<N>.ckpt`: model, optimizer, scheduler, loop, RNG, and stateful DataLoader state;
- `checkpoints/last.ckpt`: latest resumable state;
- `hf/step-1000/`: cooled Hugging Face safetensors export and tokenizer;
- `runtime.json`: wall time, throughput, and peak CUDA allocation/reservation;
- `manifest.json`: arm, seed, batch, exposure, optimizer values, attention mode, plan paths, final checkpoint, and export path.

Dense scalar series live in W&B. GitHub issue #479 and [the logbook](../../.agents/logbooks/479-mntp-adaptation.md) remain the narrative record.

The private staging repository also receives the actual-checkpoint preflight, sequence-plan hashes, final safetensors exports, odd/X VEP scores and natural-unit bootstrap summaries, inference runtime, context-ablation/window-shift metrics and stability tables, numeric dependency maps, and matched-scale SVG comparisons. The VEP loader requests only each pinned public `train` split and rejects any chromosome outside odd autosomes/X before scoring. The dependency-map panel uses unlabeled reference sequence and may include its preregistered even-autosome locus.

Checkpoint publication defaults to every numbered checkpoint. A recovery launch may set
`--checkpoint-upload-steps` and a separate `--resume-hf-repo-id` when the original private
owner reaches its storage quota. This leaves existing checkpoints untouched while preserving
the specified restart milestones in a second private repository.

## Held-out evaluation boundary

Development, probes, VEP selection, and continuation decisions may use labeled data only from odd-numbered autosomes and chromosome X. Do not read labels, predictions, effect measurements, or aggregate metrics for even-numbered autosomes or chromosome Y without explicit permission for final evaluation.
