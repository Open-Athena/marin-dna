# Train a tiny DNA model

This is the smallest complete MarinDNA training example: public enhancer DNA
tokenization and decoder-only model training on local CPU.
It is a tutorial, not a biologically useful model or recommended production
recipe.

## Run

From a fresh clone:

```bash
cd examples/train_tiny_dna
uv sync --locked
MARIN_PREFIX="$PWD/local_store" WANDB_MODE=disabled uv run python train.py
```

No private data, cloud credentials, GPU/TPU, or W&B account is required.
The tokenized dataset and trained checkpoint are written below `MARIN_PREFIX`.

## Outputs and reruns

The command writes two versioned artifacts:

```text
local_store/
  tokenized/zoonomia-ccre-non-promoter-tutorial/2026.07.29/
  checkpoints/tiny-dna-qwen3-cpu/2026.07.29/
```

Marin records the completed graph below `MARIN_PREFIX`. Running the same command
again recognizes those records and reuses the completed tokenization and
checkpoint artifacts.

To start cleanly without deleting prior outputs, choose a different local
prefix:

```bash
MARIN_PREFIX="$PWD/another_local_store" WANDB_MODE=disabled uv run python train.py
```
