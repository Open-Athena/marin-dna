---
tags:
  - biology
  - genomics
  - dna
library_name: transformers
---

# MarinDNA ortholog-RAG 104M — 2M-token batch, 30k steps

This is the 103.8M-parameter checkpoint from issue
[#402](https://github.com/Open-Athena/marin-dna/issues/402), trained from
scratch for 30,000 updates and 62,914,560,000 token presentations. Every
optimizer update contains 1,024 fixed-layout 2,048-token documents
(2,097,152 tokens/update), with seven projected mammalian ortholog windows
followed by the human window. The 104M run uses 128 documents per TPU chip and
two exact microsteps per optimizer update because the no-accumulation geometry
exceeded device HBM by 420 MiB.

The exact self-contained training launcher is
[`launch_100m_large_batch_30k.py`](https://github.com/Open-Athena/marin-dna/blob/f076021674dc32205bc1c79bf0d44f477c0fed52/experiments/exp402_rag_glm/launch_100m_large_batch_30k.py#L1-L124).
Training used the frozen
[`bolinas-dna/zoonomia-rag-v1-v1`](https://huggingface.co/datasets/bolinas-dna/zoonomia-rag-v1-v1)
dataset at revision `5e6b30cf878b61c99e6432ad8ab7865b18cbe0e7`.
The AdamH recipe was resolved for this exact batch and horizon; the learning-rate
schedule uses 3,000 warmup updates, 21,000 stable updates, and 6,000 decay
updates. Validation, permanent native optimizer checkpoints, and Transformers
exports are retained every 1,000 updates.

## Contents

The repository contains the final step-29,999 Transformers export:
`model.safetensors`, `config.json`, `tokenizer.json`, and
`tokenizer_config.json`. The eight-token vocabulary is `[PAD]`, `[UNK]`,
`[BOS]`, `[SEQ]`, `a`, `c`, `g`, `t`; `[SEQ]` separates the fixed 255-base
ortholog slots.

Levanter trains in Transformers 5's Qwen3 configuration schema. Before public
upload, metadata is normalized—without changing model weights—to preserve the
same Llama-3 RoPE (`theta=500000`) and atomic special-token contract in both
Transformers 4.57 and 5.12. The exact assertions and normalization are
[commit-pinned here](https://github.com/Open-Athena/marin-dna/blob/6c82d0fe4384d9906ec42e74de83462110616ee7/src/marin_dna/pipelines/rag_glm/hf_publication.py#L10-L118).

## Intended use

This is a research checkpoint for genomic language-model analysis and
retrieval-conditioned variant-effect prediction. It is not a clinical model
and must not be used for medical diagnosis or patient-level decisions.

The corresponding issue records the immutable evaluation datasets, exact
forward/reverse-complement likelihood convention, per-subset results,
conservation baselines, attention diagnostics, and known limitations.

This is one exploratory training seed and is not a matched human-only control.
