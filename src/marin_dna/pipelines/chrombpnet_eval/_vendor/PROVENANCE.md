# Vendored: chrombpnet_pytorch (arsenal-chrombpnet)

Source: https://github.com/amanpatel101/arsenal-chrombpnet
Commit: dcfa42b1786713e131bb113f4c6d20acc046185d (2026-02-08)

A PyTorch ChromBPNet (forked from jsxlei/chrombpnet-pytorch) extended by the
ARSENAL authors to train a ChromBPNet head on language-model embeddings.

Local modifications (see git history / issue #236, PR #239):
- Absolute intra-package imports (`from chrombpnet.X`) -> relative (`from .X`).
- ARSENAL `from modeling.model import *` (regulatory_lm) removed; the embedding
  input is supplied by `marin_dna.pipelines.chrombpnet_eval.embedding` (our HF
  causal adapter).
- interpret.py / snp_scoring.py / snp_utils.py are removed (they pull
  tensorflow / deeplift / tangermeme); we use our own variant scoring + the M1a
  metric harness instead.
- Heavy/irrelevant top-level imports (weasyprint, pooch, deepdish) made lazy or
  trimmed on the training path.
