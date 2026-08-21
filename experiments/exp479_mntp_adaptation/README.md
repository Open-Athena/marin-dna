# Experiment 479: m5.1 MNTP adaptation pilot

This permanent experiment project implements [issue #479](https://github.com/Open-Athena/marin-dna/issues/479): three matched 1,000-step training arms and two no-training controls for converting the released 1B m5.1 causal checkpoint into a full-attention masked-next-token-prediction model.

The runtime is one Lambda Cloud GH200 managed through SkyPilot. Marin and Iris are not dependencies of this experiment. Every numbered checkpoint is saved on the task disk and every validation boundary is logged to W&B; selected restart milestones are copied to reviewed private Hugging Face staging before Sky tears the instance down.

## Current experiment status

The one-seed pilot and follow-up causal runs completed, but their absolute loss values used an incorrect repeat-weight denominator.
The reducer divided weighted loss by selected-token count instead of effective-weight sum, shrinking the loss according to lowercase-repeat content.
The original source validation suite also contains three datasets, while the exp479 five-way panel added enhancer and ncRNA probes.
The corrected 128-row-per-dataset audit reproduces the invalid source value of `0.231380263`, reports `0.764665566` with Marin's reducer on the three original datasets, and preserves a small worsening through step 1,000.
An all-16,384-row reproduction of each original validation dataset is the remaining source-parity gate.
The VEP, attention, coordinate, serialization, and final dependency evidence is unchanged by this discovery, but interpretation remains paused.
The current conservative list-price estimate is $26.4687 of the $50 cap, and every Lambda cluster was confirmed terminated.

See the [compact result bundle](../../.agents/artifacts/479-mntp-adaptation/README.md), [checkpoint audit](https://wandb.ai/gonzalobenegas/marin/runs/gavkgtmf), [stability audit](https://wandb.ai/gonzalobenegas/marin/runs/q67hbkp4), and [final dependency run](https://wandb.ai/gonzalobenegas/marin/runs/yl5sgffn). Final weights and per-variant scores remain private.

## Causal fine-tuning sanity gate

The audited continued-CLM arm used a fresh optimizer with peak learning rates of 0.00440 and 0.0231, and fixed-plan validation loss rose from 0.23138 to 0.35965 by step 800.
That arm does not establish how the mature checkpoint behaves under ordinary low-learning-rate fine-tuning.
The `calibration` stage runs one conservative full-parameter AdamW arm at `1e-6` for 200 steps before considering any other learning rate.

The stage fixes:

- AdamW betas `(0.9, 0.95)`, epsilon `1e-8`, zero weight decay, and global gradient clipping at 1.0;
- a 10-step linear warmup followed by constant `1e-6`;
- the original batch-64 tokenizer, first 200 training batches, orientation policy, lowercase repeat weight, and 640-row five-component validation plan; and
- post-hoc validation at steps 0, 1, 10, 25, 50, 100, and 200.

The arm passes only when step-200 pooled and component losses are no higher than their step-0 values and each component trajectory has a non-positive fitted slope.
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

The follow-up `longrun` stage tests the human-selected peak learning rate of `1e-5` for 1,000 causal optimizer steps.
It uses the same full-parameter AdamW constants, batch-64 training plan, tokenizer, orientation policy, lowercase repeat weighting, and fixed validation panel as the 200-step calibration.
The schedule warms linearly from zero through step 100, remains at `1e-5` through step 800, and decays linearly to zero at the step-1,000 boundary.

Only the pooled 640-sequence validation loss is reported for this stage.
Because the five components contribute 128 sequences each, this pooled value is also the equally weighted component macro.
The trajectory is evaluated at steps 0, 25, 50, 100, every 100 steps through 800, 900, and 1,000.

Every post-update Hugging Face-format trajectory export is retained immediately as a W&B model artifact.
After training, the complete step-1,000 Lightning checkpoint containing model, optimizer, scheduler, and loop state is retained as a separate W&B model artifact before validation begins.
The Lambda task does not upload any output to Hugging Face and does not delete a retained checkpoint.

```bash
uv run --locked python launch.py longrun \
  --commit "$(git rev-parse HEAD)" \
  --prior-cost-usd 25.26241970350875 \
  --retry-until-up \
  --execute
```

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

## Loss-normalization and source-parity audits

The `loss-normalization` stage evaluates the released source plus every retained causal checkpoint with both the invalid count denominator and the corrected Marin reducer.
It uses the immutable 128-row-per-dataset panel, limits the source-comparable macro to CDS, upstream, and downstream, and uploads only compact tables and plots to W&B.
The completed audit reports corrected source-three loss of `0.764665566` at step 0, `0.763741364` at step 100, and `0.767604491` at step 1,000.
No checkpoint was deleted, modified, or uploaded elsewhere.
Direct evidence is at [W&B run v6mo9gh3](https://wandb.ai/gonzalobenegas/marin/runs/v6mo9gh3).

The `source-validation` stage evaluates all 16,384 rows in each of the three original source validation datasets.
One model forward supplies each dataset's repeat-weighted, uppercase-only, and lowercase-only loss.
The stage compares those nine values and their macro directly with the original W&B run at an absolute tolerance of `0.002`.
This is a hard gate for the source checkpoint's tokenization, next-token shift, special-token handling, repeat weights, z-loss, and global reduction.
It uses public Hugging Face inputs, forwards only the W&B secret, uploads no checkpoint, and self-terminates.

```bash
uv run --locked python launch.py source-validation \
  --commit "$(git rev-parse HEAD)" \
  --prior-cost-usd 26.468723089202907 \
  --retry-until-up \
  --execute
```

## Registered behavior

- Source checkpoint: `marin-dna/marin-dna-exp135-m5.1@a73a5dcfb3d64b8941e7e7596c6e88ef77db3e7a`.
- Context: one BOS plus 255 nucleotide bases.
- MNTP: sample one `Uniform(0, 1)` probability per sequence, select eligible A/C/G/T positions independently, resample zero-target rows, replace every target with `[MASK]`, and supervise target `i` from output `i - 1`.
- Loss: normalize weighted cross-entropy by effective-weight sum. MNTP first normalizes within each sequence and then averages sequences; continued CLM uses Marin's global token-weighted mean. Uppercase bases have weight 1 and lowercase bases have weight 0.01. Include the pinned source z-loss weight `4.312883184368223e-6`.
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
