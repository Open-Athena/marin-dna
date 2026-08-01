# Experiment 436: Mendelian label across SAE layers

This permanent, unmerged experiment implements the protocol in [issue #436](https://github.com/Open-Athena/marin-dna/issues/436) and informs [research question #288](https://github.com/Open-Athena/marin-dna/issues/288).

## Scientific design

The first/middle/final m5.1 panel is reported blocks 1, 10, and 19. Issue #426 already trained and verified block-10 and block-19 SAEs at exact 5,000,550- and 25,000,200-activation checkpoints. This experiment reuses those artifacts and trains only the missing block-1 trajectory under the same seed-288, 8× BatchTopK K=64, 50/50 FWD/RC, per-layer-normalized recipe.

The primary scientific work tests Mendelian `label` against every activation-supported SAE feature, then distinguishes marginal feature association from information distributed across the full sparse code and from signal lost during SAE reconstruction. Layers, budgets, FWD/RC orientations, focal/pooled responses, and statistical families remain separately reported.

## Block-1 training

The copied #426 training path preserves the exact data revisions, budgets, optimizer, normalization sample, bf16 language-model / fp32 SAE boundary, and two-batch LLM prefetch. The language model remains eager because the pinned Qwen/SAELens hook cache failed the real `torch.compile` smoke in #426; this is the known-correct comparable recipe. Later Mendelian extraction uses the standardized compiled prediction path when its hooks pass the real smoke test.

From this directory:

```bash
uv lock
uv sync --frozen
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python train.py --dry-run --no-compile
sky launch -d --dryrun sky.train.yaml
```

After pushing the exact experiment commit:

```bash
sky launch -d -c exp436-lambda \
  --env EXPERIMENT_COMMIT=<40-character-commit> \
  sky.train.yaml
```
 The managed task stages both inference exports and a hash-complete manifest under `retrieval/<run-id>/`. Retrieve and independently verify every listed byte count and SHA-256 before terminating `exp436-lambda`.

## Existing SAE inputs

The verified local #426 artifacts are intentionally not committed. Their expected weight hashes are:

| arm | SHA-256 |
|---|---|
| block10-5m | `dacde7e27d8ff20eb1ca52497f8b76494a4b92fc8ee607cfff8bcd38604267a0` |
| block10-25m | `606b81e2cc34ad7225de0fbaf5e673e688c4f990fc748cb59223316893e826b6` |
| block19-5m | `a35abcd7d8b9098b3574bff1270cd177117b687ade5845471403900b46f00971` |
| block19-25m | `e4f10ba59f10be943dbdc33f469f986f598c5e34fcba42577efad27717231533` |

Analysis results belong in issue #436; this README remains a reproduction runbook.
