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
Use 30-minute heartbeats after progress is established, report full-evaluation progress to the user at each heartbeat, reslice after one hour without progress, and restart after three hours without progress.
Identical full workers may be admitted concurrently on a cluster once its first worker proves progress.
Root timeout must exceed the child timeout; use 24-hour roots and the existing 18-hour full workers.
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

2026-11-03: The controller rejected the training runtime's 2026-10-18 Iris build before allocating a job.
The evaluation now pins the coherent Marin 0.2.224.dev40027069267 set previously validated by the common-loss study, preserving the checkpoint model source and the required ArrayStacked API.
Native final-loss replay remains a mandatory numerical gate for this runtime change.
2026-11-03: The first accepted GPU task inherited a revoked CoreWeave key and failed during compilation-cache initialization.
Submit roots with current Secret Manager credentials using Iris named-secret file delivery, inherited by children, and an explicit compilation-cache location in the shared `marin-us-east-02a` bucket.
2026-11-03: Native scatter uses BF16 atomic weighted sums and is not repeatable; its suffix log probabilities exceed the original numerical oracle on identical repeated requests.
An independently implemented ordered FP32 weighted sum with the same grouped expert projections agrees bitwise with Sonic on all 56 real, reverse-complement, alternate, and stress cases at both 2K and 8K, including suffix logs and frozen features, and their 512-window losses are identical.
The initial maximum-error check against atomic scatter failed at 0.098 despite p99 0.015, mean 0.00075, 99.64% argmax agreement, and aggregate loss difference 0.000022.
Retain those original failed diagnostics; require the stronger exact independent-reference agreement, exact batch/repeat oracles, unchanged native mean/p99/argmax/feature limits, and tighten the aggregate backend-loss limit to 0.00005 for admission.
Repeat all native parity cases to locate outliers and measure atomic noise before full evaluation.
