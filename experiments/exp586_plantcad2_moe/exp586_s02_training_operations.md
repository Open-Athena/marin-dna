# exp586 Stage-2 CoreWeave Sweep Operations

> This document governs the pretrained 1x continuation managed with the `run-training-sweep-cw` skill.
> Read it in full at the start of every heartbeat before inspecting code, SQLite, W&B, or Iris.

## Invariants

Use one live writer, one W&B identity, and one checkpoint root for the continuation trial.
Use H100s only, whole eight-GPU nodes, batch priority, and the approved Iris user.
Keep global batch 64, EP8, the tokenizer, and the complete restored optimizer and data clocks unchanged across placement changes.
Do not move to another accelerator family.
Treat W&B as training-state truth and use Iris only for exact-job liveness, placement, and diagnosed startup failures.
Do not publish GitHub updates without an explicit user request.

## Sweep Definition

The one-trial entry point, continuation source binding, schedule, and checkpoint contract are in `src/exp586_moe/train.py`, `src/exp586_moe/config.py`, `src/exp586_moe/schedule.py`, and `src/exp586_moe/state.py`.
Only the pretrained 1x `s02` configuration is in scope.
Production may start only after a real H100 smoke verifies exact full-state restore, the first scheduled learning rate, a finite update, checkpoint save, and checkpoint readback.

## Operator Choices

The sweep deadline is `2026-11-02T15:00:00Z`; it stops the entire sweep and does not define recovery timing.
Use at most 64 H100s and prefer an eight-node gang when live batch-priority capacity can sustain it.
Both `cw-rno2a` and `cw-us-east-02a` are approved.
Cross-family reslicing is disabled.
Use `--priority batch` and the explicitly approved Iris user on every root submission.
As of `2026-10-20T14:47:21Z`, the operator has terminated the stage-2 continuation for scientific eval-loss regression.
Do not submit, restart, recover, or reslice another Iris job for this trial, and do not resume monitoring.

The local control database is `scratch/exp586_s02/exp586_s02.sqlite`.
Its durable immutable backups belong under `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/operations/training-s02/` with keys `<UTC>-<uuid>.sqlite`.
Create backups with the skill helper, upload through authenticated CoreWeave S3, verify size and SHA-256, and recover from the newest valid immutable backup when the local database is absent.

## Operating Policy

Use a 30-minute heartbeat after W&B proves the trial is advancing.
Use `reslice_after=1h`, `restart_after=3h`, and `pending_target_limit=1`.
Recheck live H100 utilization before every submission, stop, or reslice.
Do not replace the job merely because Iris reports preemption or confusing task state; require W&B evidence under the recovery policy.
Prefer the measured 64-H100 target while it is productive, and reslice only when the expected wall-clock benefit exceeds restore and queueing overhead.
Completion requires `run_progress >= 1` and a reachable permanent final checkpoint at the exact target clock.

Every heartbeat must report scientific performance as well as operational health.
Report the current `eval/loss` and its signed delta from the independently verified restored-source value `0.947515` near 216.159B input tokens and from the 216.159B comparator endpoints: pretrained 0.5x `0.946480`, pretrained 1x `0.947506`, scratch 1x `0.949141`, and pretrained 2x `0.975580`.
Describe continuation values around `1.074` through `1.077` as a scientific regression even when training is finite, MoE drops are zero, checkpoints are healthy, and the writer is advancing.
Scientific regression alone does not authorize stopping, reslicing, or changing the run; placement and recovery actions remain governed by the operational thresholds above.

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| `cw-rno2a` | H100 | 1 | 8 | unvalidated | — |
| `cw-rno2a` | H100 | 2 | 16 | eligible | The original pretrained run and continuation restore smoke use this placement contract. |
| `cw-rno2a` | H100 | 4 | 32 | eligible | The validated 16- and 64-H100 endpoints use the same EP8-per-node mesh and fixed global batch; this intermediate whole-node replica count lowers per-GPU batch and preserves optimizer semantics. |
| `cw-rno2a` | H100 | 8 | 64 | eligible | The completed pretrained sweep validated this placement with unchanged global batch and EP8. |
| `cw-us-east-02a` | H100 | 1 | 8 | unvalidated | — |
| `cw-us-east-02a` | H100 | 2 | 16 | eligible | The original pretrained sweep validated this placement contract. |
| `cw-us-east-02a` | H100 | 4 | 32 | eligible | The validated 16- and 64-H100 endpoints use the same EP8-per-node mesh and fixed global batch; this intermediate whole-node replica count lowers per-GPU batch and preserves optimizer semantics. |
| `cw-us-east-02a` | H100 | 8 | 64 | eligible | The completed pretrained sweep validated this placement with unchanged global batch and EP8. |

## Change Record

2026-10-19: The permanent LR-peak checkpoint at update 418,887 was restored and written to a fresh TTL prefix on 16 H100s, then reloaded with exact model, optimizer, router, and clock equality. Three later restores crashed inside TensorStore OCDBT when they reused the same incomplete update-418,912 prefix; direct object-store transport produced the same failure. Checkpoint retries therefore leave incomplete prefixes untouched and write an append-only retry prefix for the same metadata-bound update. Production then committed `step-418912-retry-2a1e02a49318`, verified its metadata and latest pointer directly from S3, and resumed training beyond it on 32 H100s.

2026-10-20: Operator review added a scientific-performance contract to every heartbeat. Heartbeats now quantify eval-loss regression against the verified restored source and four 216.159B comparator endpoints while leaving stop, reslice, and recovery decisions under the existing operational policy.

2026-10-20: The operator terminated `/eczech/exp586-s02-production-a6` after review of the scientific eval-loss regression. Monitoring ended with no active or pending dispatch, and future submission, restart, recovery, and reslice actions for this trial are prohibited.
