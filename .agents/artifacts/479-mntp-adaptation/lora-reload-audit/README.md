# Final LoRA adapter reload parity

This directory contains the compact evidence from the successful matched-numeric-state reload audit in [W&B run `r9m9m9gj`](https://wandb.ai/gonzalobenegas/marin/runs/r9m9m9gj).
The producing source is commit `6431bdc20c249f98ec7de32f0af11d0890b427c5`, tagged `exp479-lora-reload-audit-v3`.

A fresh process reloaded the retained final LoRA adapter and reproduced all 640 paired adapter and frozen-source scores with maximum absolute four-way CE drift `4.44e-16` and zero top-1 mismatches.
The standard full-attention encoding and its additive-mask training encoding were exactly equal under forced math SDPA.

The preceding v2 audit was rejected because it omitted the numeric controls active during Lightning evaluation.
Matching `torch.set_float32_matmul_precision("high")`, deterministic algorithms, cuDNN benchmarking, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` eliminated that drift.
The failed-attempt and corrected final adapter weight files also have identical W&B digests and byte sizes.

No VEP, nucleotide-dependency analysis, Hugging Face upload, checkpoint deletion, or knowledge-base update was performed.
