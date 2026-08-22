# Frozen localized predictor-row attention diagnostic

This directory contains the compact evidence from [W&B run `dq67i9vi`](https://wandb.ai/gonzalobenegas/marin/runs/dq67i9vi).
The producing source is commit `eb0db004b6418e9db1f7db71c8041983c4a90a9a`, tagged `exp479-localized-predictor-attention-zero-training-v1`.

All four readouts use the same 640 deterministic single-`[UNK]` targets and forced math SDPA.
The standard causal mask and its explicit additive-mask encoding match exactly.

Opening only the shifted predictor row reduces most of the damage from uniform full attention but remains significantly worse than causal prediction.
Localized attention changes four-way CE from `1.051681` to `1.207186` and accuracy from `0.510938` to `0.448438`.
The paired localized-minus-causal CE delta is `+0.155505` with 95% interval `[+0.100788, +0.207693]`, and the accuracy delta is `-0.0625` with 95% interval `[-0.095352, -0.028125]`.

No parameters were updated.
No VEP, nucleotide-dependency analysis, Hugging Face upload, checkpoint deletion, or knowledge-base update was performed.
