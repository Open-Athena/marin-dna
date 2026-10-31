# exp586 isolated data-parallel experiments

This experiment uses `run-training-sweep-cw`; read this document before every decision pass.

## Invariants

Never modify, stop, resubmit, or write checkpoints or W&B history for the live 1T run.
Never modify the monitoring agent's checkout, ledger, allocation, or deployment work.
Use only this independent `codex/exp586-dp` checkout, separate `-dp-` W&B identities, and temporary S3 artifacts.
Do not publish GitHub posts, issues, pull requests, or branches for this experiment without further authorization.
Every GPU owns the entire model and optimizer state; no EP, FSDP, tensor parallelism, or host-state offload is permitted in a DP candidate.
Use existing local MoE kernels without capacity truncation.

## Definition

The entry point and bounded catalog are `src/exp586_moe/dp_profile.py`.
The model/mesh adapter is `src/exp586_moe/data_parallel.py`.
Paired comparisons use eight examples per update, with the same 64-example reference LR/data cursor spacing as round 2.
Standalone batch-64 validation uses the actual production global batch, cursor, and token count on the same single eight-GPU node.
The existing global histogram router preserves its mathematical thresholds across different partitions of the same batch; do not substitute per-device quantiles or change its bins.
The source is the same frozen, independently verified DNA checkpoint used for the previous optimization experiments; never read a changing latest pointer.
Compare exact initial values, masks and example order; mathematical local-expert outputs and gradients; replicated optimizer updates; router estimator grouping; full-vocabulary loss; numerical trajectories against unchanged EP controls; and temporary checkpoint restoration.
EP comparison is conditional on zero reference drops; the local implementation never drops assignments.
Do not require bitwise equality across distinct GPU kernels or collective orders.
Do not claim long-term scientific equivalence solely from a short loss trajectory.

## Operator choices

Keep the existing experiment deadline, 2026-10-31T18:07:14Z, as a conservative hard stop for this follow-up.
Use at most one whole eight-H100 node at a time in `cw-us-east-02a` or `cw-rno2a`, always batch priority with user `eczech`.
Do not run scaling tests or request a larger gang.
Prefer unused East capacity while the production writer uses Reno; capacity must be refreshed before each submission.
Each child job is bounded by 45 minutes and the global deadline.
No automatic retries; diagnose failures before selecting a fresh attempt.
The ledger is `scratch/exp586_dp/dp.sqlite`.
Immutable backups belong to `s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp586_plantcad2_moe/dp-20261031/operations/<UTC>-<uuid>.sqlite`.
Use the skill's transactional persistence helper, backup each mutation, upload through authenticated CW S3, and verify byte length and SHA-256 by reading back.
Restore the newest valid immutable backup if local state is lost.

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| cw-us-east-02a | H100 | 1 | 8 | eligible | Existing exp586 launches validate environment, whole-node resources, and shared storage; this experiment validates the new runtime |
| cw-rno2a | H100 | 1 | 8 | eligible | Existing exp586 launches validate environment, whole-node resources, and shared storage; use only genuinely free capacity |

## Operating policy

Use W&B for training progress, exact Iris jobs for liveness and concrete startup failures, and the skill's v3 utilization helper for capacity.
Use one dispatch at a time and pending-target limit one.
Check bounded jobs at useful startup/completion boundaries; do not perform production-monitoring heartbeats here.
Completion requires expected update history, a directly readable result artifact, any promised checkpoint-replay evidence, and a terminal Iris root.
Stop all isolated work at the deadline and leave the live run unchanged.

## Change record

2026-10-31: The first smoke retained temporary Muon work partitioning over replicas and six asynchronous all-gathers.
The strict candidate computes orthogonalization locally and checks asynchronous as well as synchronous collective names.
A startup contraction error on size-one model/data annotations was fixed with a fully replicated local computation scope.
All model and optimizer arrays remain replicated; no live experiment files changed.
