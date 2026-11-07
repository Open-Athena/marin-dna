# exp593 training operations

This document governs the single augmented d768 training run and its bounded validation jobs.
Read it before each placement or recovery decision.
The local detailed research plan remains untracked in scratch/exp593-augmented-d768-plan.md.

## Ownership and authorization

The user authorized data preparation, single-node H100 validation, and full training on November 7, 2026.
Train one model to the approved token horizon; there is no multi-configuration sweep.
Use at most one active training/validation gang and at most 64 H100s.
Initial smoke tests use one eight-H100 node.
Use batch priority, Iris user eczech, and an explicit target in cw-rno2a or cw-us-east-02a.
Other accelerator families are outside this experiment.
The model identity and global scientific batch remain fixed across node-count changes.
Keep the other exp586 session's jobs, deployment, checkpoint roots, and checkout untouched.

All experiment code and training artifacts belong to the single permanent MarinDNA PR #594.
Keep that branch unmerged.
Do not create a PlantCAD2 PR at this stage.

## Scientific bindings

The source model is the exact text-pretrained checkpoint defined by exp586 SOURCE_CHECKPOINT at runtime source commit de5f428fc7b2bac8b6e40697f905c8b99ca3580d.
Caller controls FIM completion lengths, including for prompts without a length hint.
Do not add supervised end-of-fill tokens.
Training DNA is restricted to the original pinned Angiosperm65 train rows.
Annotations supply sampling locations; answers always come from those existing rows.
Primary eval/loss must preserve the historical 512 examples and encoding.
Keep evaluation source data out of the training/annotation sampler.

## Storage and state

Native durable owner: s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/.
Checkpoint identities, source manifests, and annotation sidecars are immutable once used for training.
The local working ledger is scratch/exp593/operations.sqlite.
Use the existing transactional persistence helper; keep observations and attempts in SQLite.
Back up each mutation to a new operations/<UTC>-<uuid>.sqlite object and verify size and SHA-256 by reading it back.
Recover the newest valid immutable backup if local state is lost.
Never overwrite committed checkpoints or allow two writers for one run.
The production entry point is `exp593_dna.train`; its checkpoint manager is `exp593_dna.recovery`.
The final update remains a complete global batch, and the exact token overshoot is recorded rather than masking part of its objective.

Use documented CW credentials from marin.env or current Secret Manager entries as described in marin-agent-kb/docs/coreweave.md.
Do not print or commit credential values.
Forward current storage credentials explicitly to remote workers.

## Placement and monitoring

Use fresh Iris v3 backend availability before submissions and node-count changes.
Stale observations are inconclusive and cannot justify a placement.
Keep one pending training dispatch at most.
Treat a one-node result as evidence only for its tested shape, batches, and memory.
Use W&B for numerical progress and exact Iris root/task IDs for liveness and concrete failures.
Use eric-czech/marin at the CoreWeave W&B endpoint with an exp593 identity.
After startup, check progress on a 30-minute cadence.
Iris owns preemption recovery; do not replace a dispatch solely because it was preempted.
Reassess placement after one hour without progress and diagnose a restart after three hours.
Do not silently alter model/data/schedule semantics to obtain capacity.
Completion requires the full input-token horizon, a readable verified final checkpoint, promised diagnostics, and a terminal Iris root.

## Placement evidence

The November 7 single-node smoke validated all context buckets and global batch 64 on eight H100s, including full-state checkpoint restoration and one resumed update.
See the companion's `artifacts/smoke-v1/result.json` for numerical limits and source bindings.
The one-node profile is eligible for production once runtime evaluation and production-loop validation pass.
The two-, four- and eight-node profiles remain candidates for bounded validation; they have not yet been exercised by this experiment.
There is no active GPU gang after the completed smoke.
