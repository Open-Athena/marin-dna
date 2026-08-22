# Damage-calibrated LoRA MNTP paired gate

This directory is the compact W&B evaluation artifact from the corrected deterministic replay at [run `tnkdn3v3`](https://wandb.ai/gonzalobenegas/marin/runs/tnkdn3v3).

The immutable producing code is commit [`b25d46b39b6d696bafdce09fa1dcb4603750c971`](https://github.com/Open-Athena/marin-dna/commit/b25d46b39b6d696bafdce09fa1dcb4603750c971), tagged `exp479-lora-damage-calibrated-v2`.

The frozen-base rank-16 LoRA run completed 1,000 optimizer steps and retained every registered adapter plus a final optimizer/RNG checkpoint in W&B.

The paired information gate failed decisively: final full-attention CE was `1.3655851085` versus source-causal `1.0507945247`, and final accuracy was `0.31875` versus `0.50625` on the same 640 targets.

The paired CE delta was `+0.3147905837` with 95% bootstrap interval `[+0.2675381738, +0.3648440462]`.

The paired accuracy delta was `-0.1875` with 95% bootstrap interval `[-0.2359375, -0.1421875]`.

Disabling the adapter at step 1,000 preserved the source-causal scores bit-exactly.

All 1,000 training losses and pre-clipping gradient norms are finite, and no step clipped.

No VEP, nucleotide-dependency, Hugging Face upload, checkpoint deletion, or research knowledge-base update was performed.

Use `paired-nucleotide-gate.json` for the decision, `paired-nucleotide-comparisons.csv` for paired intervals, `paired-nucleotide-summary.csv` and `paired-nucleotide-scores.csv` for trajectories and raw paired readouts, `gradient-norm-trace.csv` for stability, and `retention-manifest.json` for exact artifact identities.
