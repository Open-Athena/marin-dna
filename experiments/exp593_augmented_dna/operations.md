# exp593 training operations

This document governs the two augmented d768 training runs and their bounded validation jobs.
Read it before each placement or recovery decision.
The local detailed research plan remains untracked in scratch/exp593-augmented-d768-plan.md.

## Ownership and authorization

The user authorized data preparation, single-node H100 validation, and full training on November 7, 2026.
Train each approved model to the token horizon; there is no open-ended configuration sweep.
The November 8 authorization permits overlapping preparation allocations for resizing while the current training gang continues; release overlap promptly after verified cutover or failed preparation.
The later November 8 authorization adds a second independent production run and raises the reasonable aggregate ceiling to 512 H100s across productive, validation and replacement allocations.
This supersedes the earlier single-gang / 64-H100 and temporary 128-H100 limits, while retaining exactly one writer per model identity and at most one pending or warming replacement per model.
Each validated gang still uses at most eight nodes / 64 H100s at fixed global batch 64; the larger aggregate ceiling does not justify unnecessary allocations or scientific batch changes.
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
That initialization remains bound to `dna-exp593-augmented-d768-pretrained-seed0-v1`.
The second identity, `dna-exp593-augmented-d768-dna216b-lrm0p5-seed0-v1`, imports only learned weights from the permanent exp586 language-initialized 216B `step-412290` checkpoint.
It starts a fresh optimizer and a new zero-based step/input/example clock, halves both peak learning rates, and retains the 1% warmup and additional 216,158,699,520-input budget.
The exact sources and canonical metadata digests are selected by `TrainingConfig` in `src/exp593_dna/train.py`.
Source router bias is part of the learned weights; importing it must not restore source optimizer history or erase its effective bias.
The native weight loader uses the canonical metadata digest, which differs from the raw-file digest for the 216B source.
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
When reporting capacity, also show the raw `ListPeers` free/total counts displayed by the Iris UI, with their measurement age, even when the strict helper rejects a stale measurement.
Report peer `last_contact_ms` separately from backend `availability.observation_epoch_ms`: a reachable peer can have recent contact while forwarding an older capacity observation.
The documented query is `iris --cluster marin rpc controller list-peers`; the strict utilization helper remains the placement preflight.
The MarinFold skill references the same query; its current helper expects availability v2, while this experiment's updated helper supports the live v3 priority-rank semantics.
Keep one pending replacement dispatch at most, in addition to the productive allocation.
Treat a one-node result as evidence only for its tested shape, batches, and memory.
Use W&B for numerical progress and exact Iris root/task IDs for liveness and concrete failures.
Use eric-czech/marin at the CoreWeave W&B endpoint with an exp593 identity.
After startup, check progress on a 30-minute cadence.
Iris owns preemption recovery; do not replace a dispatch solely because it was preempted.
Reassess placement after one hour without progress and diagnose a restart after three hours.
For a prolonged admission wait, compare the exact task with the fresh backend's pod status before classifying it as a software stall.
When both show an unassigned SchedulingGated pod, other work is blocked on node fit, and no superior eligible placement has fresh capacity, retain the queued dispatch rather than resetting its queue age without addressing the constraint.
Do not silently alter model/data/schedule semantics to obtain capacity.
Completion requires the full input-token horizon, a readable verified final checkpoint, promised diagnostics, and a terminal Iris root.

## Placement evidence

The November 7 single-node smoke validated all context buckets and global batch 64 on eight H100s, including full-state checkpoint restoration and one resumed update.
See the companion's `artifacts/smoke-v1/result.json` for numerical limits and source bindings.
The one-node profile is eligible for production once runtime evaluation and production-loop validation pass.
The four-node profile passed sixteen production-loop updates, both full validation events and native checkpoint discovery at global batch 64; see artifacts/loop-smoke-n4-v1/result.json.
It is eligible for production with its tested initialization and runtime.
The two-node profiles remain candidates for bounded validation; the later Reno eight-node resume result is recorded below.
The subsequent fixed-panel evaluator passed both causal panels, 1,024 FIM presentations and 36 greedy completions on the same one-node profile; see artifacts/eval-smoke-v2/result.json.
The four-node production-loop validation uses the same scientific batch and bindings, with its own validation namespace.

The completed accelerator checks establish eligibility for one/four East nodes and four/eight Reno nodes, with the eight-node evidence restricted to full-state resume.
The full candidate grid below remains within the approved H100 scope; other targets need bounded validation before their first production dispatch.

| H100 nodes | GPUs | cw-us-east-02a | cw-rno2a |
| --- | --- | --- | --- |
| 1 | 8 | Eligible | Unvalidated |
| 2 | 16 | Unvalidated | Unvalidated |
| 4 | 32 | Eligible | Eligible |
| 8 | 64 | Unvalidated | Eligible for full-state resume |

The local four-device CPU checkpoint regression covers exact full-state restoration across replica-group counts 1→2→1, including the FP32 expert-master/momentum partition and replicated BF16 weights.
It supports the global-array checkpoint mechanism; it does not establish target eligibility or replace numerical checks on a future GPU reslice.

The first Reno eight-node validation completed its initial causal/FIM/functional panels and update one, then stalled in XLA ConfigAssigner's distributed autotuning key-value exchange.
All eight nodes had idle GPUs and approximately idle Python CPUs; sampled native stacks identified the compiler wait after cache compaction had finished.
The first workaround applied `--xla_gpu_shard_autotuning=false` only to GPU workers, preserving the other optimized runtime flags.
That retry passed the initial panels but stalled inside its first training update, with sampled GPU streams waiting in ConditionalThunk execution.
Two NCCL RAS reads found all 64 ranks present and identical, nonadvancing collective counters; they did not identify the underlying cause.
At that point, the eight-node placement remained unvalidated.
The original optimized runtime settings were restored and four Reno nodes validated before production.
The November 8 authorization permits a bounded eight-node standby validation alongside productive training; investigate a concrete failure before retrying that target.

## Non-blocking preparation and handoff

Use `exp593_dna.train` standby arguments to acquire the replacement gang without starting the production W&B tracker or writing production checkpoints.
The standby restores an existing strictly bound full-state checkpoint, exercises every bucket with and without norm collection, and runs all fixed validation panels.
It retains its allocation, decoded corpus and compiled executables while waiting for an explicit promotion command under its unique `placements/` namespace.
The agent owns every placement and promotion decision; the resident gate never chooses a fleet action.
Readiness alone must never stop the current writer or promote the replacement.
Before promotion, verify the old root and its exact training child are terminal, verify the selected committed checkpoint and clock, and publish an immutable command binding those facts to the readiness receipt.
Restore the final selected state before initializing the existing personal W&B identity with `resume=must` and a server-derived history offset.
Discard all disposable standby updates and restart from the committed production data/token cursor.
Use the new allocation's checkpoint-and-exit control for subsequent resizing.
The original running allocation predates that control; coordinate its first cutover with a newly completed periodic checkpoint and report any replayed work.
Do not promote unless the larger target passes its checks and its measured benefit justifies handoff overhead.

The four-node Reno check subsequently passed all sixteen updates and all four sequence buckets with zero routing drops.
Both fixed causal panels, all 1,024 FIM presentations / 36 greedy completions, and both functional-development panels completed.
The native full-state checkpoint and recovery pointer were read back, hashed and discovered with strict scientific bindings; its exact clock matches the earlier four-node East validation.
See artifacts/loop-smoke-rno-n4-v1/result.json.

The November 8 eight-node standby restored the actual four-node production checkpoint and passed sixteen disposable updates with the original optimized runtime.
All four buckets and both norm branches completed, all metrics were finite, routing drops were zero, and the complete fixed and functional validation panels passed.
See artifacts/standby-rno-n8-v1/result.json and its immutable native readiness receipt.
This establishes eligibility for the tested full-state resume path; the underlying cause of the earlier fresh-state eight-node failures is not established.
The first handoff waits for a newly completed periodic checkpoint from the original writer while both allocations remain held.

The resident first standby uses immutable source bcb00881 and predates automatic recognition of a prior promotion on re-entry.
If that allocation later fails on its serialized standby source or duplicate readiness after an Iris retry, diagnose the concrete error and recover through the ordinary production entry point with the same W&B/checkpoint identity.
Later source recognizes the immutable promotion and chooses normal latest-checkpoint recovery; it does not require the old source or handoff checkpoint to survive retention.

The first production handoff completed from step-35749 onto the already-held eight-node allocation, preserving the single W&B identity and exact native data/optimizer clock.
See artifacts/production-reslice-rno-n8-v1/result.json for the measured update gap, replayed logged tokens, actual resumed metrics and bounded throughput comparison.
The historical metric rows show that Iris controller/task KILLED state preceded the last old optimizer update.
For a future forced cancellation, require a fresh backend pod observation after the stop request showing the old allocation absent or physically terminal before publishing promotion.
A successful graceful checkpoint-and-exit is preferable for future handoffs.
Do not infer that resources or writers have stopped merely from cancellation acknowledgement.
