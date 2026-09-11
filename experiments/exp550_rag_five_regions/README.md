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
The European fallback uses `--region europe-west4` and `MARIN_PREFIX=gs://marin-eu-west4/MarinDNA/exp550_rag_five_regions`; the bucket name differs from the canonical GCP region.
Its synthetic pilots append `-europe-west4` to the run identity to preserve the earlier US pilot's W&B history.
It requests 80 GiB of local scratch within the pool's 100 GiB per-VM limit; tokenized caches and checkpoints are written to GCS.
Production reserves 256 GiB of host RAM after the original 48 GiB container exhausted its limit during training; synthetic pilots retain 48 GiB.
The experiment adapter limits training-loader buffering and fetch lookahead to eight batches each while retaining upstream batching, shuffling, and resume order.
Resume the same production output version to reuse its tokenized data and let the native loader select the newest complete checkpoint, excluding partial checkpoints without metadata.
The host allocation reserves 48 GiB RAM and 16 CPU cores; tokenization streams through two workers with batches of 128 documents.
Use `--tpu-variant v6e-4` when eight-chip capacity is unavailable, after validating a synthetic pilot on four chips.
It preserves the model, optimizer, 200-document effective batch, and schedule; four chips with microbatch 5 accumulate ten microbatches per update.
Four-chip pilots use a separate run name ending in `-v6e-4` so their metrics and checkpoints remain distinct.
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
Use EC2 A10G with BF16 and compilation, as explicitly requested on September 11.
The combined RAG adapter uses the existing `evals_v2` cached scoring kernel: one prefix forward, then one batched REF/ALT suffix forward.
Inputs are left-padded to 10,240 tokens with an attention mask and unpadded position IDs, placing every human variant at token 10,112.
Use ordinary batches without length grouping; pool the final 255 human bases for REF/ALT embeddings and retain FWD/RC averaging.
The earlier strict-FP32 pilot and H100 run are historical evidence; their numerical tolerance gate does not override this BF16 choice.
The H100 evaluation and its transfer watchers were stopped before restarting on A10G.

Measure the actual checkpoint with `.agents/artifacts/issue-550/evaluation/sweep-cached-bf16.py` on the GPU worker.
The synthetic sweep uses the maintained scorer, includes tokenization, loading, both strands and embeddings, and chooses the fastest measured stable batch with GPU memory headroom.
Use the selected batch as a per-model execution override and retain a bounded prediction-offload cadence.
For the 10k checkpoint, `run-10k-a10g.py` verifies the staged checkpoint bytes, creates that overlay, checks the runtime against the worker deadline, and dry-runs the three canonical metric targets.
Run it without `--execute` first and inspect the plan.
The companion `run-10k-a10g.sh` executes it, preserves logs and recovery outputs in S3, and terminates the worker on completion or failure.
These wrappers have the September 11 worker paths and deadline pinned; review and update them before reuse on another worker.
The existing Snakemake S3 profile publishes the score bundles and metrics directly to their canonical paths.
If execution review blocks S3 publication, invoke the Python driver directly with `--local-only` for planning and add `--execute` after inspecting its dry-run.
That option uses `--workflow-profile none`, keeps the score and metric outputs local, and retains the explicit S3 harness input as a read.
Do not use the uploading shell wrapper for this mode.
The September 11 worker uses this local mode and stops automatically when the systemd evaluation service exits; its verified deadline also stops it at 6 p.m. NYC time.
EBS survives a stop, so retrieve and verify outputs before terminating the instance, and include temporary disk storage in the cumulative budget.
Canonical S3 publication is a separate pending step while execution access remains blocked.
The 10k score bundles include embeddings; the final checkpoint additionally requires all three frozen-probe metric targets.
Retain the final evaluation reservation when choosing compute for this additional run.

The registered models are `dna-exp550-rag46m-five-regions-v1-step-10000` and `dna-exp550-rag46m-five-regions-v1-step-100000` in [PR #565](https://github.com/Open-Athena/marin-dna/pull/565).
The active source is the europe-west4 version-9 export; check the tracking issue before using it after a recovery.
The permanent branch includes the combined cached backend, tokenizer compatibility, execution controls, and registration together; none of their PRs needs to be merged to reproduce this experiment.

After the final export exists, stage the registered checkpoint on the shared VM before starting paid GPU time.
The lightweight experiment helper uses normal GCS and S3 credential providers, pins GCS generations, validates model geometry and byte checksums, and writes the canonical S3 checkpoint directory with conditional puts.
It acquires the shared heavy-work lock, checks memory and load, monitors pressure throughout the transfer, and records timing, status, and peak RSS in its receipt.
Existing S3 objects must match; it never replaces a different checkpoint.
Run these commands from the repository root and inspect the plan before applying it:

```bash
model=dna-exp550-rag46m-five-regions-v1-step-100000
uv run --locked --script .agents/artifacts/issue-550/evaluation/stage-final-checkpoint.py \
  --model "$model" --receipt /tmp/issue550-final-checkpoint-plan.json
uv run --locked --script .agents/artifacts/issue-550/evaluation/stage-final-checkpoint.py \
  --model "$model" --apply --receipt /tmp/issue550-final-checkpoint-stage.json
```

For the requested first checkpoint, select `dna-exp550-rag46m-five-regions-v1-step-10000` throughout, use its registered `hf/step-10000` URI, and give the receipts distinct filenames.
The helper avoids the evaluation workflow's up-front ML imports, which exceed this shared VM's 500 MiB working-set limit.
Its small contract tests run with `uv run --locked --script .agents/artifacts/issue-550/evaluation/test-stage-final-checkpoint.py` and perform no cloud writes.
Require a successful staging exit and a receipt with `exit_status: 0` and `applied: true` before starting the GPU worker or copying the checkpoint.
S3 prefix existence and `.snakemake_timestamp` alone are insufficient: Snakemake can recognize a directory while a failed upload has left only some of its files.
Retry a failed stage through the same helper so all retained objects are revalidated.
On the GPU worker, use the same consumer commit and copy the completed S3 checkpoint into the explicit storage cache before the synthetic batch sweep.
This uses the GPU worker's normal S3 access; GCP credentials stay on the staging host.
Run from `snakemake/analysis/evals_v2`:

```bash
model=dna-exp550-rag46m-five-regions-v1-step-100000
storage_prefix=/opt/issue550/storage
checkpoint_local="$storage_prefix/s3/oa-bolinas/snakemake/analysis/evals_v2/results/checkpoints/$model"
uv sync --locked --group genome-s3
aws s3 sync "s3://oa-bolinas/snakemake/analysis/evals_v2/results/checkpoints/$model/" "$checkpoint_local/" --only-show-errors
uv run --locked --group genome-s3 python ../../../.agents/artifacts/issue-550/evaluation/sweep-cached-bf16.py \
  --checkpoint "$checkpoint_local" --output /opt/issue550/final-batch-sweep.json
```

Inspect the measured throughput, finite outputs, and memory headroom before proceeding.
Set the selected model's batch size and `eval_accumulation_steps: 8` in an execution overlay, retaining global BF16, compilation, RC, and embeddings.
The estimated runtime must fit the cumulative budget and worker shutdown deadline, including metric computation and uploads.
The following target list includes all three zero-shot and frozen-probe metric outputs without widening the model registry:

```bash
targets=()
for dataset in mendelian_traits complex_traits sge; do
  targets+=("results/metrics/$model/$dataset.parquet" "results/probe_metrics/$model/$dataset.parquet")
done
uv run --locked --group genome-s3 snakemake -n "${targets[@]}" --cores 2 --configfiles config/config.yaml /opt/issue550/a10g-bf16.yaml --local-storage-prefix "$storage_prefix"
uv run --locked --group genome-s3 snakemake "${targets[@]}" --cores 2 --configfiles config/config.yaml /opt/issue550/a10g-bf16.yaml --local-storage-prefix "$storage_prefix"
```
