# exp586 CoreWeave Sweep Operations

> This document governs the training sweep managed with the `run-training-sweep-cw` skill.
> Read it in full at the start of every heartbeat before inspecting code, SQLite, W&B, or Iris.

## Invariants

Use one live writer, one W&B identity, and one checkpoint root per logical trial.
Use H100s only, whole eight-GPU nodes, batch priority, and the approved Iris user.
Keep global batch 64 and EP8 unchanged across placement changes.
Do not move to another accelerator family.
Treat W&B as training-state truth and use Iris only for exact-job liveness, placement, and diagnosed startup failures.
Do not publish issue comments or other GitHub status updates without an explicit user request.

## Sweep Definition

The trial catalog and one-trial entry point are in `src/exp586_moe/train.py`.
The scientific constants are in `src/exp586_moe/config.py` and the Hero schedule is in `src/exp586_moe/schedule.py`.
Use the permanent source checkpoint pinned there and the unchanged experiment 582 corpus and tokenization contracts.

## Operator Choices

The user authorized the complete six-trial H100 sweep on 2026-10-08 with a maximum of 96 submitted H100s, allowing one 16-GPU placement per trial.
Both `cw-rno2a` and `cw-us-east-02a` are approved.
There is no user-requested sweep deadline; continue until all trials complete or the user stops the sweep.
Cross-family reslicing is disabled.
Use `--priority batch` and the explicitly approved Iris user on every root submission.

The local control database is `scratch/exp586/exp586_sweep.sqlite`.
Its durable immutable backups belong under `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/operations/training/` with keys `<UTC>-<uuid>.sqlite`.
Create backups with the skill helper, upload through authenticated CoreWeave S3, verify size and SHA-256, and recover from the newest valid immutable backup when the local database is absent.

## Operating Policy

Use a 30-minute heartbeat after W&B proves at least one trial per initialization is advancing.
Use `reslice_after=1h`, `restart_after=3h`, and `pending_target_limit=1`.
Before that point, monitor startup continuously enough to diagnose compilation, restore, data, dependency, or numerical failures promptly.
Recheck live H100 utilization before every submission, stop, or reslice.
Do not replace jobs merely because Iris reports preemption or confusing task state; require W&B evidence under the recovery policy.
Classify correlated failures before retrying them.
Completion requires `run_progress >= 1` and a reachable permanent final checkpoint.

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| `cw-rno2a` | H100 | 1 | 8 | unvalidated | — |
| `cw-rno2a` | H100 | 2 | 16 | eligible | Scratch EP8 control reached finite updates on 2026-10-08. |
| `cw-rno2a` | H100 | 4 | 32 | unvalidated | — |
| `cw-rno2a` | H100 | 8 | 64 | unvalidated | — |
| `cw-us-east-02a` | H100 | 1 | 8 | unvalidated | — |
| `cw-us-east-02a` | H100 | 2 | 16 | eligible | Pretrained EP8 smoke completed checkpoint and replay on 2026-10-08. |
| `cw-us-east-02a` | H100 | 4 | 32 | unvalidated | — |
| `cw-us-east-02a` | H100 | 8 | 64 | unvalidated | — |

## Change Record

2026-10-08: Two-node pretrained EP8 passed 25 finite, dropless updates, checkpointing, and three-way replay at approximately 297,000 input tokens per second.
2026-10-08: The scratch default compiler path did not reach an update within one hour; disabling all autotuning restored startup but reduced throughput to approximately 49,000 input tokens per second.
2026-10-08: Scratch factor-32 drops were receiver-only and transient over the first 25 updates; a 600-update factor-32 control determines whether they settle as they did in experiment 582.
2026-10-08: Receiver factor 64 is rejected after allocating 84.86 GB per H100 and failing first-step NCCL communicator creation with CUDA out of memory.
