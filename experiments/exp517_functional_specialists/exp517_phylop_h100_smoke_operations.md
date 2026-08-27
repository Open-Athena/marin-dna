# exp517 H100 Smoke Operations

> This document governs the one-arm H100 validation managed with the `run-training-sweep-cw` skill.
> Read it in full at the start of every heartbeat before inspecting code, SQLite, W&B, Iris, or fleet utilization.

## Invariants

The smoke has a distinct W&B and checkpoint identity from every production phyloP and GPN-Star-P run.
The six queued TPU workflows remain untouched during validation.
The training child uses one H100, batch priority, preemptible capacity, CoreWeave-local S3, and no GCS reads.

## Sweep Definition

The single CDS validation trial is defined in [`phylop_uniform_h100_smoke.py`](src/exp517_functional_specialists/phylop_uniform_h100_smoke.py).
It starts with the full 8,192-sequence batch as one per-device microbatch and reduces that microbatch only after a verified H100 OOM.

## Operator Choices

The absolute smoke deadline is 2026-08-28 18:00 UTC.
The deadline stops the smoke; recovery timing is controlled separately by `reslice_after` and `restart_after`.
The approved scope is one preemptible H100 at batch priority, with a one-GPU ceiling and Iris user `gonzalo`.
Use either production H100 cluster and select among them using valid live capacity.
Do not use GB200 or a multi-GPU gang.

The durable state owner is `gs://marin-us-east5/MarinDNA/exp517_phylop_uniform_specialists/sweep_state/coreweave-smoke/`.
Upload each consistent SQLite backup to a new immutable timestamped key with `gcloud storage cp` and verify byte count and SHA-256.
Recover from the newest valid immutable backup and fall back newest-to-oldest if validation fails.

## Operating Policy

Use `heartbeat_every=30m`, `reslice_after=1h`, `restart_after=3h`, and `pending_target_limit=1`.
Keep one active dispatch and one checkpoint writer.
Treat a verified OOM at per-device parallelism 8,192 as permission to retry 4,096, then 2,048, then 1,024.
Completion requires at least one advancing W&B optimizer step with an H100 device and no OOM.

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| `cw-us-east-02a` | `H100` | 1 | 1 | `eligible` | — |
| `cw-rno2a` | `H100` | 1 | 1 | `eligible` | — |

## Change Record

None.
