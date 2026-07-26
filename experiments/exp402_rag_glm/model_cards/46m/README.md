---
tags:
  - biology
  - genomics
  - dna
library_name: transformers
---

# MarinDNA ortholog-RAG 46M — 30k steps

This is the 45.9M-parameter checkpoint from issue
[#402](https://github.com/Open-Athena/marin-dna/issues/402), trained from
scratch for 30,000 updates (3,932,160,000 tokens) on fixed-layout
2,048-token documents containing seven projected mammalian ortholog windows
followed by the human window.

The exact self-contained training launcher is
[`launch_30k.py`](https://github.com/Open-Athena/marin-dna/blob/cd82495babc15fa6f189ad441f8211407943d84c/experiments/exp402_rag_glm/launch_30k.py).
Training used the frozen
[`bolinas-dna/zoonomia-rag-v1-v1`](https://huggingface.co/datasets/bolinas-dna/zoonomia-rag-v1-v1)
dataset at revision `5e6b30cf878b61c99e6432ad8ab7865b18cbe0e7`.

## Contents

The repository contains the final Transformers export:
`model.safetensors`, `config.json`, `tokenizer.json`, and
`tokenizer_config.json`. The eight-token vocabulary is `[PAD]`, `[UNK]`,
`[BOS]`, `[SEQ]`, `a`, `c`, `g`, `t`; `[SEQ]` separates the fixed 255-base
ortholog slots.

## Intended use

This is a research checkpoint for genomic language-model analysis and
retrieval-conditioned variant-effect prediction. It is not a clinical model
and must not be used for medical diagnosis or patient-level decisions.

The corresponding issue records the immutable evaluation datasets, exact
forward/reverse-complement likelihood convention, per-subset results,
conservation baselines, attention diagnostics, and known limitations.
