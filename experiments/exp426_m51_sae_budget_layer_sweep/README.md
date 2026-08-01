# Experiment 426: SAE budget × layer sweep

This permanent experiment branch implements the preregistered comparison in [issue #426](https://github.com/Open-Athena/marin-dna/issues/426), informing the durable research question in [issue #288](https://github.com/Open-Athena/marin-dna/issues/288). It is intentionally self-contained and is not intended to merge into `main`.

## Scientific design

One shared m5.1 forward pass captures reported blocks 4, 10, 16, and 19. A separate BatchTopK SAE (`d_sae=15,360`, `K=64`) is trained at each block. The exact 5,000,550-activation checkpoint is exported and the same trajectory continues to 25,000,200 activations, yielding eight inference arms. Training examples are exactly balanced across the five pinned m5.1 source streams and forward/reverse-complement orientation.

The gLM runs in bfloat16; SAEs train in float32. The default is the manifest-recorded eager gLM path because `torch.compile(..., mode="reduce-overhead")` does not preserve the dynamic multi-layer hook cache in the pinned Qwen/SAELens combination. `COMPILE_LLM=1` remains available for a future compatibility retest, but it must pass the real multi-hook smoke test before any registered arm starts. Two prefetched LLM batches overlap generation with SAE optimization. The Qwen generation cache is disabled because these are independent full-context forwards; this avoids needless dynamic-cache initialization guards and repeated graph specialization without changing logits or captured activations. The pinned SAELens multi-hook path returns the LM's bf16 tensors directly despite its configured float32 activation-store dtype, so this experiment applies and tests a narrow adapter at that boundary before normalization or SAE training. A cloned-SAE optimizer preflight verifies an actual float32 forward/backward/update without mutating any registered arm. Each layer estimates its own `expected_average_only_in` normalization scalar on 100 shared batches (255,000 nucleotide activations per layer) before training; the scalar is folded into each inference export. SAELens caches these multi-hook batches simultaneously, so its 1,000-batch default would require about 78 GB for the normalization cache alone and is unsafe on an 80 GB H100.

All arms are then evaluated on the exact frozen chr21 panel from #422. Ref and alt sequences are extracted separately in FWD and RC orientation. The registered primary endpoint is mean held-out one-vs-rest AUPRC across the 35 `consequence_cre` classes using the signed mean of paired FWD/RC SAE deltas. FWD, RC, and max-absolute views are diagnostic. Discovery chooses candidates, validation selects among them, and test labels remain untouched until final scoring. Confidence intervals use consequence-stratified genomic-block resampling so spatial dependence is retained without allowing clustered classes to disappear from a balanced-panel replicate; a consequence represented in only one observed test block is held fixed because its within-class spatial variance is not identifiable.

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
  --env RUN_ID=dna-exp426-layer-budget-seed288-r5
```

The task uses autostop/down after 180 idle minutes, providing a bounded post-job recovery window. Training, extraction, analysis, and any recovery rerun must remain Sky-managed (`sky launch` or `sky exec`); do not start a long command through raw SSH because Sky cannot see it as active work. The task stages a `retrieval/<run-id>/` tree containing the eight inference models, extraction, analysis, and hash-complete manifests while excluding large trainer checkpoints. Retrieve that tree after the managed job succeeds, validate the hashes locally, and then explicitly run `sky down exp426-lambda` rather than waiting for the idle deadline.

## Output contract

`artifacts/<run-id>/manifest.json` records the model/data revisions, exact training and normalization counts, hardware/runtime, per-arm health metrics, and hashes of every inference model file. `extraction/manifest.json` hashes all 16 paired activation parquets. `analysis/manifest.json` hashes the class-level metrics, arm summaries, continued-training comparisons, paired activation-state summaries, and both PNG/SVG figures.
