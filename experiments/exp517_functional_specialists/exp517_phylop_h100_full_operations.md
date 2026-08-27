# exp517 Full H100 CDS Operations

> This document governs the full one-arm H100 run managed with the `run-training-sweep-cw` skill.
> Read it in full at the start of every heartbeat before inspecting code, SQLite, W&B, Iris, or fleet utilization.

## Invariants

The full H100 run has a distinct W&B and checkpoint identity from every TPU and smoke run.
The six existing TPU workflows remain untouched until the H100 path is explicitly adopted.
The training child uses one H100, batch priority, preemptible capacity, CoreWeave-local S3, and no GCS reads.
The scientific recipe remains the strict phyloP CDS control with global batch 8,192, sequence length 256, seed 0, and 5,000 optimizer steps.
Per-device parallelism is fixed at the validated one-H100 value 2,048.

## Sweep Definition

The single full CDS trial is defined in [`phylop_uniform_h100_experiment.py`](src/exp517_functional_specialists/phylop_uniform_h100_experiment.py).
It tokenizes the complete immutable CDS dataset revision `452a5a3538f22630c3dea94d441ac30216bb28ea` into CoreWeave-local S3 before training.
Tokenization runs as 16 local Zephyr workers inside one explicitly sized preemptible CoreWeave CPU job because the pinned Fray CPU actor path does not attach a uv environment on the production peers.
The run ID is `dna-exp517-phylop-uniform-0p25b-cds-h100-pdp2048-v1`.

## Operator Choices

The absolute trial deadline is 2026-09-04 20:00 UTC.
The deadline stops the trial; recovery timing is controlled separately by `reslice_after` and `restart_after`.
The approved scope is one preemptible H100 at batch priority, with a one-GPU ceiling and Iris user `gonzalo`.
Use either production H100 cluster and select between them using valid live capacity.
Do not use GB200 or a multi-GPU gang.

The durable state owner is `gs://marin-us-east5/MarinDNA/exp517_phylop_uniform_specialists/sweep_state/coreweave-full-cds/`.
Upload each consistent SQLite backup to a new immutable timestamped key with `gcloud storage cp` and verify byte count and SHA-256.
Recover from the newest valid immutable backup and fall back newest-to-oldest if validation fails.

## Operating Policy

Use `heartbeat_every=30m`, `reslice_after=1h`, `restart_after=3h`, and `pending_target_limit=1`.
Keep one active dispatch and one checkpoint writer.
Resume the same W&B run and checkpoint root after preemption or recoverable failure.
Do not reduce per-device parallelism unless the full run produces a new verified H100 OOM at 2,048.
Completion requires terminal optimizer step 4,999, a finished W&B run with sane telemetry, and a reachable terminal checkpoint.

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| `cw-us-east-02a` | `H100` | 1 | 1 | `eligible` | — |
| `cw-rno2a` | `H100` | 1 | 1 | `eligible` | — |

## Change Record

2026-08-27: The 8,192 and 4,096 per-device smoke configurations failed with verified first-step OOMs.
The 2,048 configuration completed 3/3 steps and wrote a remotely verified S3 checkpoint, so it is the fixed full-run value.
2026-08-27: Keep the CPU coordinator below Iris's large-resource threshold with 9 GB ephemeral disk.
The H100 child retains its independently declared 128 GB disk request.
2026-08-27: Disable top-level Iris auto-sync, install uv 0.11.31 as an isolated tool, select the nested H100 project by its full bundled path, and run the coordinator from the experiment root.
2026-08-27: Override the training data tokenizer with the vendored absolute path so a reused tokenized-cache record cannot resolve relative to `/app`.
2026-08-27: Both production peers' bare CPU callable images lacked `cloudpickle`, so the nested remote tokenization wrapper failed before reading data or allocating an H100.
Keep the maintained Marin tokenization and cache format, but run its Zephyr pool locally inside one locked, explicitly sized CoreWeave CPU task; launch training separately after that immutable cache completes.
2026-08-27: Local Fray context variables do not propagate into actor-method threads, and `IRIS_TASK_ID` would make those threads rediscover the ambient Iris backend.
Hide `IRIS_TASK_ID` only for the duration of local tokenization and restore it afterward so every Zephyr coordinator and worker remains in the explicitly sized task.
