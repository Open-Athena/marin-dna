# exp586 1T CoreWeave Sweep Operations

> This document governs the fresh pretrained 1T run managed with the `run-training-sweep-cw` skill.
> Read it in full at the start of every heartbeat before inspecting code, SQLite, W&B, or Iris.

## Invariants

Use one live writer, one W&B identity, and one checkpoint root for the trial.
Use H100s only, whole eight-GPU nodes, batch priority, and the approved Iris user.
Keep global batch 64, the full language tokenizer, and the fresh pretrained initialization unchanged across placement changes.
The approved production runtime uses fully replicated data parallelism on exactly 64 H100s and the `trajectory-combined-no-pdl` settings validated from perf4 source commit `34aa317617c3a0ede609c76b01e44312dbd864e2`.
Retain the pre-perf4 DP checkpoint as the immediate rollback path and the legacy EP checkpoint as the older rollback path.
Require physical fingerprints for every persistent array plus an independent FP32 output-projection hash and norm after restore and at every checkpoint boundary.
Treat W&B as training-state truth and use Iris only for exact-job liveness, placement, and diagnosed startup failures.
Do not publish GitHub updates without an explicit user request.

## Sweep Definition

The single-trial entry point, source-language-checkpoint binding, 1T Hero schedule, and 12-checkpoint retention contract are in `src/exp586_moe/train.py`, `src/exp586_moe/config.py`, `src/exp586_moe/schedule.py`, and `src/exp586_moe/state.py`.
Only the fresh pretrained 1x `t1` configuration is in scope.
Production may start only after a real H100 smoke verifies the exact source weights, zero DNA-token and optimizer clocks, first scheduled learning rate, a finite update, checkpoint save, and checkpoint readback.

## Operator Choices

The sweep deadline is `2026-12-05T00:00:00Z`; it stops the entire sweep and does not define recovery timing.
Use at most 64 H100s and prefer an eight-node gang when live batch-priority capacity can sustain it.
Both `cw-rno2a` and `cw-us-east-02a` are approved.
Cross-family reslicing is disabled.
Use `--priority batch` and the explicitly approved Iris user on every root submission.

The local control database is `scratch/exp586_1t/exp586_1t.sqlite`.
Its durable immutable backups belong under `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/operations/training-1t/` with keys `<UTC>-<uuid>.sqlite`.
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
| `cw-rno2a` | H100 | 2 | 16 | ineligible | DP production runtime validated only at 64 H100s |
| `cw-rno2a` | H100 | 4 | 32 | ineligible | DP production runtime validated only at 64 H100s |
| `cw-rno2a` | H100 | 8 | 64 | eligible | — |
| `cw-us-east-02a` | H100 | 1 | 8 | unvalidated | — |
| `cw-us-east-02a` | H100 | 2 | 16 | ineligible | DP production runtime validated only at 64 H100s |
| `cw-us-east-02a` | H100 | 4 | 32 | ineligible | DP production runtime validated only at 64 H100s |
| `cw-us-east-02a` | H100 | 8 | 64 | eligible | — |

## Change Record

- `2026-10-31T17:04:24Z`: The operator approved replacing the legacy EP runtime with the validated fully replicated DP runtime. Production execution now requires a 64-H100 target; the same scientific, W&B, checkpoint, batch, schedule, tokenizer, and data identities remain binding, and the latest committed EP checkpoint remains the rollback path.
- `2026-11-01T21:45:00Z`: The operator approved deploying the matched `trajectory-combined-no-pdl` runtime from perf4 source commit `34aa317617c3a0ede609c76b01e44312dbd864e2`. Production retains one 64-H100 writer and the existing scientific, W&B, checkpoint, batch, schedule, tokenizer, and data identities; the transition adds the validated eager reader cache, node-local Muon work sharing, disabled PDL launch flags, restore/checkpoint physical-replica checks, and a recorded pre-transition rollback checkpoint.
