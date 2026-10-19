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

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| `cw-rno2a` | H100 | 1 | 8 | unvalidated | — |
| `cw-rno2a` | H100 | 2 | 16 | eligible | The original pretrained run and continuation restore smoke use this placement contract. |
| `cw-rno2a` | H100 | 4 | 32 | unvalidated | — |
| `cw-rno2a` | H100 | 8 | 64 | eligible | The completed pretrained sweep validated this placement with unchanged global batch and EP8. |
| `cw-us-east-02a` | H100 | 1 | 8 | unvalidated | — |
| `cw-us-east-02a` | H100 | 2 | 16 | eligible | The original pretrained sweep validated this placement contract. |
| `cw-us-east-02a` | H100 | 4 | 32 | unvalidated | — |
| `cw-us-east-02a` | H100 | 8 | 64 | eligible | The completed pretrained sweep validated this placement with unchanged global batch and EP8. |

## Change Record

None.
