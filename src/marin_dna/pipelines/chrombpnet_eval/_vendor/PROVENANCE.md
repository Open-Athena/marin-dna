# Vendored: chrombpnet_pytorch (arsenal-chrombpnet)

Source: https://github.com/amanpatel101/arsenal-chrombpnet
Commit: dcfa42b1786713e131bb113f4c6d20acc046185d (2026-02-08)

A PyTorch ChromBPNet (forked from jsxlei/chrombpnet-pytorch) extended by the
ARSENAL authors to train a ChromBPNet head on language-model embeddings.

Local modifications (see git history / issue #236, PR #239):
- Absolute intra-package imports (`from chrombpnet.X`) -> relative (`from .X`).
- ARSENAL `from modeling.model import *` (regulatory_lm) removed; the embedding
  input is supplied by `marin_dna.pipelines.chrombpnet_eval.embedding` (our HF
  causal adapter). The one name that wildcard provided —
  `TransformerROPEEncoderLayer`, referenced by `ArsenalChromBPNet.get_avg_embeddings`
  — is now an optional import with a sentinel fallback (`chrombpnet.py`) so the
  module loads without the regulatory_lm package.
- interpret.py / snp_scoring.py / snp_utils.py are removed (they pull
  tensorflow / deeplift / tangermeme); we use our own variant scoring + the M1a
  metric harness instead.
- Heavy/irrelevant top-level imports (weasyprint, pooch, deepdish) made lazy or
  trimmed on the training path.
- Unused modules trimmed (#252): `metrics.py`, `metrics_utils.py`, `logger.py`
  (nothing in our pipeline imports them — the M1a/M1b harness lives in
  `marin_dna.pipelines.chrombpnet_eval.metrics`), and the broken-and-unused
  `BPNet.to_keras` (a `@classmethod` that referenced `self` and a Keras export
  that pulled tensorflow).

## Lint / format / type standards (#252)

This vendor is held to the same `ruff format` + `ruff check` + `mypy` standards as
the rest of the repo — it is **not** in any `pyproject.toml` exclusion. The code was
auto-formatted and lint-fixed (unused imports, `lambda` → `def`, bare-`except` →
typed, `== None` → `is None`, multi-statement lines split) and given enough
annotations / targeted `# type: ignore` to pass mypy (e.g. the bogus copied
`-> Union["pl.LightningDataModule", "pl.Trainer"]` return hints became `-> Any`;
the commented-out `_ADD_ARGPARSE_RETURN` import in `parse_utils.py` was inlined as
`Union[_ArgumentGroup, ArgumentParser]`).

Trade-off: this increases divergence from upstream `dcfa42b`. Re-syncing to a newer
`arsenal-chrombpnet` is now a manual merge rather than a clean re-vendor. Accepted:
uniform quality over easy re-sync.
