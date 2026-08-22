# Frozen-source attention annealing diagnostic

This directory is the compact permanent snapshot of the zero-training attention-mask trajectory requested in issue 479.

- Producing code: [`c52d0e9e`](https://github.com/Open-Athena/marin-dna/commit/c52d0e9e1d48fc2d29e46a754eb3c1b2405c852b).
- W&B run: [`0vvh4kcb`](https://wandb.ai/gonzalobenegas/marin/runs/0vvh4kcb).
- W&B artifact: `gonzalobenegas/marin/dna-exp479-source-attention-annealing-diagnostic:nested-attention`.
- Hardware: one AWS `g5.xlarge` A10G in `us-east-2`, using an 80 GB ephemeral root disk and automatic teardown.
- Model: `marin-dna/marin-dna-exp135-m5.1@a73a5dcfb3d64b8941e7e7596c6e88ef77db3e7a`, loaded in BF16.
- Validation: 640 exact deterministic targets from plan SHA-256 `35542611d71102479f3d07dc6565350120d1d89944e5a93f88efb641ece7e3ba`.
- Parameter updates: zero.
- Command: `uv run --locked python launch.py attention-anneal-diagnostic --commit c52d0e9e1d48fc2d29e46a754eb3c1b2405c852b --execute --prior-cost-usd 32.945379`.

The raw per-target scores and replicate-mean target table remain in the W&B artifact.
The committed files retain the aggregate trajectory, five-mask replicate summaries, paired bootstrap comparisons, exact endpoint controls, manifests, and rendered figure.
