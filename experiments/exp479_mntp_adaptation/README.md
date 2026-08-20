# Experiment 479: m5.1 MNTP adaptation pilot

This permanent experiment project implements [issue #479](https://github.com/Open-Athena/marin-dna/issues/479): three matched 1,000-step training arms and two no-training controls for converting the released 1B m5.1 causal checkpoint into a full-attention masked-next-token-prediction model.

The runtime is one Lambda Cloud GH200 managed through SkyPilot. Marin and Iris are not dependencies of this experiment. Every numbered checkpoint is saved on the task disk and every validation boundary is logged to W&B; selected restart milestones are copied to reviewed private Hugging Face staging before Sky tears the instance down.

## Final pilot result

The one-seed pilot completed and is technically valid. Transferred MNTP narrowly beat scratch on step-1,000 pooled loss (0.397270 versus 0.399543) and single-mask loss (0.310077 versus 0.313152), and it acquired bilateral context use. It did not improve Mendelian, complex-trait, or SGE VEP over source CLM FWD+RC, so the 10,000-step extension and single-orientation VEP inference are not supported. The final conservative list-price estimate was $10.2326 of the $50 cap, and the cluster was confirmed terminated.

See the [compact result bundle](../../.agents/artifacts/479-mntp-adaptation/README.md) and [final W&B report](https://wandb.ai/gonzalobenegas/marin/reports/Issue-#479-%E2%80%94-1k-step-MNTP-adaptation-pilot--VmlldzoxNzc2ODgyOQ==). Final weights and per-variant scores remain private.

## Registered behavior

- Source checkpoint: `marin-dna/marin-dna-exp135-m5.1@a73a5dcfb3d64b8941e7e7596c6e88ef77db3e7a`.
- Context: one BOS plus 255 nucleotide bases.
- MNTP: sample one `Uniform(0, 1)` probability per sequence, select eligible A/C/G/T positions independently, resample zero-target rows, replace every target with `[MASK]`, and supervise target `i` from output `i - 1`.
- Loss: average weighted cross-entropy within each sequence, then average sequences. Uppercase bases have weight 1 and lowercase bases have weight 0.01; the denominator is the selected-target count.
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
