# Experiment 426: SAE budget × layer sweep

This permanent experiment branch implements the preregistered comparison in [issue #426](https://github.com/Open-Athena/marin-dna/issues/426), informing the durable research question in [issue #288](https://github.com/Open-Athena/marin-dna/issues/288). It is intentionally self-contained and is not intended to merge into `main`.

## Scientific design

One shared m5.1 forward pass captures reported blocks 4, 10, 16, and 19. A separate BatchTopK SAE (`d_sae=15,360`, `K=64`) is trained at each block. The exact 5,000,550-activation checkpoint is exported and the same trajectory continues to 25,000,200 activations, yielding eight inference arms. Training examples are exactly balanced across the five pinned m5.1 source streams and forward/reverse-complement orientation.

The gLM runs in bfloat16; SAEs train in float32. `torch.compile(..., mode="reduce-overhead")` wraps the shared `run_with_cache` path, and two prefetched LLM batches overlap generation with SAE optimization. Each layer estimates its own `expected_average_only_in` normalization scalar on 100 shared batches (255,000 nucleotide activations per layer) before training; the scalar is folded into each inference export. SAELens caches these multi-hook batches simultaneously, so its 1,000-batch default would require about 78 GB for the normalization cache alone and is unsafe on an 80 GB H100.

All arms are then evaluated on the exact frozen chr21 panel from #422. Ref and alt sequences are extracted separately in FWD and RC orientation. The registered primary endpoint is mean held-out one-vs-rest AUPRC across the 35 `consequence_cre` classes using the signed mean of paired FWD/RC SAE deltas. FWD, RC, and max-absolute views are diagnostic. Discovery chooses candidates, validation selects among them, and test labels remain untouched until final scoring.

## Local checks

From this directory:

```bash
uv sync --frozen
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python train.py --dry-run
```

The dry run writes nothing and prints every pinned revision, exact batch boundary, normalization prefix, layer, budget, dtype, and performance setting.

## Launch

The Sky task uses one direct Lambda H100 and runs training, variant extraction, and analysis serially. The caller must set `EXPERIMENT_COMMIT` to the full pushed commit SHA:

```bash
sky launch -c exp426-lambda sky.yaml \
  --env EXPERIMENT_COMMIT="$(git rev-parse HEAD)" \
  --env RUN_ID=dna-exp426-layer-budget-seed288
```

The task uses autostop/down after 30 idle minutes. It stages a `retrieval/<run-id>/` tree containing the eight inference models, extraction, analysis, and hash-complete manifests while excluding large trainer checkpoints. Retrieve that tree before the idle window expires.

## Output contract

`artifacts/<run-id>/manifest.json` records the model/data revisions, exact training and normalization counts, hardware/runtime, per-arm health metrics, and hashes of every inference model file. `extraction/manifest.json` hashes all 16 paired activation parquets. `analysis/manifest.json` hashes the class-level metrics, arm summaries, continued-training comparisons, paired activation-state summaries, and both PNG/SVG figures.
