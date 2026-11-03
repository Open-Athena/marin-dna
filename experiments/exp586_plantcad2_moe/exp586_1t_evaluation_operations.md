# exp586 1T CoreWeave Evaluation Operations

Read this document before each evaluation heartbeat.
Use the repository's `run-training-sweep-cw` skill for execution and state persistence.

## Scope and authorization

The user authorized the full final-checkpoint evaluation on 2026-11-03.
Evaluate only the final pretrained 1T checkpoint bound in `final_1t_binding.json`.
Reuse historical comparison results; do not rerun other checkpoints.
Prepare a local draft for review and do not post results, comments, or updates to GitHub.
The intended eventual destination is a new standalone results comment on plantcad/plantcad2 PR 22.
Keep the existing results comments unchanged.

## Protocol

Reuse the prior exp586 input manifests, scoring implementation, sample memberships, and metric definitions.
Require final checkpoint integrity, native training-loss replay, and native-versus-Sonic probability and frozen-feature parity before full evaluation.
Preserve effective pending router bias, full language vocabulary, character-forced inputs, zero expert drops, and the production compiler settings with PDL disabled.
Use the existing bounded pilot, 16-worker full partition, two context lengths, inference methods, and eight frozen-probe fits.
Compute common-sample validation loss using the previously verified immutable inputs.

## Execution

Use H100s only on `cw-rno2a` and `cw-us-east-02a`, whole eight-GPU nodes, at most 128 GPUs concurrently.
Every Iris root and child uses batch priority; roots use user `eczech`.
Check skill utilization before every submission or placement change.
Treat W&B as execution-progress truth; use Iris for exact-job liveness and concrete startup failures.
Keep one active dispatch per logical trial.
Use 30-minute heartbeats after progress is established, reslice after one hour without progress, and restart after three hours without progress.
Identical full workers may be admitted concurrently on a cluster once its first worker proves progress.
Root timeout must exceed the child timeout; use 24-hour roots and 20-hour full workers.
The sweep hard deadline is 2026-11-17T23:59:59Z; recovery timing is independent of that deadline.

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| cw-rno2a | H100 | 1 | 8 | eligible | Previously validated exp586 evaluation placement |
| cw-us-east-02a | H100 | 1 | 8 | eligible | Previously validated exp586 evaluation placement |

## Durable state

The local control database is `scratch/exp586-1t-eval/evaluation.sqlite`.
Use the skill persistence helper for transactional state and consistent backups.
Publish immutable SQLite backups under `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/operations/evaluation-1t/` and verify uploaded size and SHA-256.
Recover the newest valid immutable backup if the local database is unavailable.
New artifacts use the existing evaluation owner with distinct model label `pretrained-lrm1-seed0-1t`, leaving prior artifacts unchanged.
The PlantCAD2 local draft is owned by branch `codex/exp586-1t-draft` and directory `zero-shot-leaderboard/marindna_exp586/runs/exp586-1t-final/`.

## Change record

None.
