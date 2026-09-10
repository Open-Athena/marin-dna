# Five-region RAG training (issue 550)

This permanent experiment project trains one scratch 46M Qwen3 on a uniform mixture of CDS, TSS+UTR5, UTR3, ncRNA, and enhancer documents.
The [tracking issue](https://github.com/Open-Athena/marin-dna/issues/550) owns the scientific decisions and run results.
The maintained producer is `snakemake/vertebrate_projection_dataset`; it owns projection, chromosome splits, public datasets, and internal provenance.

## Frozen recipe

Each document contains one through 40 available species windows of 255 bases, separated by atomic `[SEQ]` tokens.
The tokenizer adds one BOS and right-pads to 10,240 positions using a separate PAD token.
Loss weights exclude padding targets, and causal attention prevents real tokens from seeing later padding.
Training uses 200 documents per update, 100,000 updates, train seed 0, and data seed 42.
The five region components each have sampling weight 0.2.
The resulting budget is 204.8B allocated positions, including padding; actual unpadded exposure is reported separately.
Expected region epochs are `100000 * 200 * 0.2 / train_rows`, using the public release's forward-plus-RC row count.
All chr18 anchors are excluded from training; validation samples come from chr18.

The model has 7 layers, hidden dimension 640, intermediate dimension 2560, 5 attention heads, 5 KV heads, head dimension 128, untied embeddings, and an 8-token vocabulary.
The AdamH transfer in `recipe.py` follows the Complete(d) reference with batch normalized to allocated tokens across context lengths.
This context-length transfer is an experimental extension of the reference heuristic.
The production schedule has 10,000 warmup updates, 70,000 stable updates, and 20,000 linear-decay updates.
The resolved projection LR is 0.004695897205758308, Adam LR is 0.00020258341260453682, epsilon is 5.990618799422977e-8, beta2 is 0.9992190160617281, beta1 is 0.9, gradient clipping is 0.1, and z-loss weight is 1e-7.

Native resumable checkpoints, Hugging Face exports, and full region LM validation run every 10,000 completed updates.
An experiment-local hook adapter aligns those three milestones with completed updates while retaining upstream metrics and checkpoint-state conventions.
Variant-effect evaluation and frozen probes begin with the final checkpoint on canonical development splits; earlier checkpoints are optional within the remaining evaluation budget.

## Reproduce

Use Python 3.12 and uv 0.11.31 on a suitably sized remote worker.
The lockfile pins the coherent Marin release `0.2.106.dev34338714012`, whose source is `efe79892065589b154d969effd49eee3bd286284`.
The release is also pinned for the source-only `marin-dupekit` dependency needed by Marin normalization imports.
The CPU extra is intended for numerical and cache-format tests; the TPU extra is required on the training workers.

```bash
uv sync --locked --extra cpu
JAX_PLATFORMS=cpu uv run --locked --extra cpu pytest
```

Production requires a JSON object keyed by `cds`, `tss_utr5`, `utr3`, `ncrna`, and `enhancer`.
Each entry contains `repo_id`, immutable 40-character `revision`, `train_rows`, `validation_rows`, and `anonymous_verified: true` from verified public Hub receipts.
Repository IDs are `marin-dna/rag-five-regions-v1-<region>`.
Set `MARIN_PREFIX=gs://marin-us-east1/MarinDNA/exp550_rag_five_regions`, `WANDB_ENTITY=gonzalobenegas`, and `WANDB_API_KEY` in the launch environment.
The launch program propagates credentials through runtime environment variables and excludes them from recorded training configuration.

```bash
uv run --locked --extra tpu python -m marin_dna_exp550.launch --pilot --per-device 5 --version <snapshot-version> --run
uv run --locked --extra tpu python -m marin_dna_exp550.launch --dataset-manifest <verified-releases.json> --per-device 5 --version <snapshot-version> --run
```

Submit the entrypoint through the current Iris client with the committed project bundled and the TPU extra available to dispatched workers.
The dispatched worker installs the pinned `uv` before the standard Iris dependency setup, since the shared image may contain an older version.
The launch uses free, preemptible `v6e-8` capacity and a microbatch of 5 per chip, accumulating to exactly 200 documents.
It defaults to `us-east1`; `--region us-east5` selects the verified capacity fallback and requires `MARIN_PREFIX=gs://marin-us-east5/MarinDNA/exp550_rag_five_regions` so caches and checkpoints stay in the compute region.
It requests 80 GiB of local scratch within the pool's 100 GiB per-VM limit; tokenized caches and checkpoints are written to GCS.
The host allocation reserves 48 GiB RAM and 16 CPU cores; tokenization streams through two workers with batches of 128 documents.
Microbatch 1 is the supported memory fallback and preserves the same effective batch.
Larger data-parallel meshes do not divide this batch evenly and must not silently change the recipe.
The synthetic pilot runs 20 updates with saves and validation every 5 completed updates while following the production optimizer schedule.
Its synthetic documents cover one, 20, and 40 species and do not read biological data.

Tokenization runs in a CPU-only child process on the allocated TPU host, using the standard Marin cache writer and a local Zephyr client with two threads.
The child removes `IRIS_TASK_ID` so cache preparation cannot dispatch additional CPU workers.
Production caches use immutable public dataset revisions; the training process then loads those prebuilt document caches.
An experiment-local data adapter reads each requested cache row once and restores repeated draws in their original order, preserving the mixture distribution while avoiding the pinned upstream reader's [repeated-row-zero truncation bug](https://github.com/Open-Athena/marin-dna/issues/557).
The fixed-horizon optimizer lives in an importable module so dispatched workers can serialize its registered configuration.
Monitor pilot loss, actual TPU count, throughput, native save/resume, HF export, and validation before launching the full run.
After the pilot completes, pass `--pilot --resume-pilot-from <pilot-output>/checkpoints/step-10` with a fresh calendar version to verify native resume through update 20 in a separate output directory.
This check requires loading the full intermediate trainer state and keeps the production scratch-start recipe unchanged.
The full run's completion time must be based on measured pilot throughput.

## Development evaluation

The maintained `snakemake/analysis/evals_v2` project owns combined RAG scoring, canonical benchmark metrics, and frozen probes.
Register the exact final checkpoint and combined-harness SHA-256 in its model registry and submit the registration PR before biological inference.
The registration must include Mendelian traits, complex traits, and SGE development cohorts and the corresponding probe cells.
Use the experiment runtime overlay `config/rag_issue550/fp32.yaml` from that pipeline's root.
It selects batch 2, two loader workers, compiled fp32, and an explicit TF32 disable, and records the precision decision in Snakemake provenance.
The [synthetic GPU evidence](https://github.com/Open-Athena/marin-dna/tree/1e4e59db/.agents/artifacts/issue-550/gpu) records the failed reduced-precision candidates and the passing strict-fp32 check.
Rerun a bounded synthetic parity check with the final checkpoint before its biological inference, since its weights differ from the pilot.
Build only the registered final model's three metric and three probe-metric targets; those targets share one combined score computation.
Retain the canonical `results/scores`, `results/metrics`, and `results/probe_metrics` output identities and the pipeline's existing probe and metric contracts.
Use the remaining cumulative budget only after this final-checkpoint evaluation completes.
