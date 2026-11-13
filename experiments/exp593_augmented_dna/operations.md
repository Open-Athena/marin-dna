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
The operator prefers 64-H100 production gangs when available; prepare that target from the outset rather than defaulting to 32 GPUs.
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
Refetch the documented peer query immediately before every submission.
Prefer measurements at most 90 seconds old; for an additional batch request that does not stop a productive gang, allow up to five minutes only when peer contact is at most 30 seconds old and reported free GPUs are at least four times the requested gang.
This bounded admission exception accommodates the backend's observed multi-minute reporting cadence; it never permits reclaiming another job, stopping production or bypassing scheduler admission.
When reporting capacity, also show the raw `ListPeers` free/total counts displayed by the Iris UI, with their measurement age, even when the strict helper rejects a stale measurement.
Report peer `last_contact_ms` separately from backend `availability.observation_epoch_ms`: a reachable peer can have recent contact while forwarding an older capacity observation.
The documented query is `iris --cluster marin rpc controller list-peers`; the strict utilization helper remains the placement preflight.
The MarinFold skill references the same query; its current helper expects availability v2, while this experiment's updated helper supports the live v3 priority-rank semantics.
Keep one pending replacement dispatch at most, in addition to the productive allocation.
The operator's November 8 instruction to submit a larger allocation while the current run continues also authorizes one queued, qualified 64-H100 Reno resident preparation before sufficient GPUs are free.
This exception retains a fresh capacity query and records the request as queued capacity, never as allocated or productive GPUs.
Count the entire request toward the 512-H100 ceiling and keep at most one pending preparation per exact target as well as per model.
Use batch priority and user `eczech`; admission remains the scheduler's decision and no held capacity is available to reclaim.
The existing writer continues until the resident preparation passes all checks and an explicit, physically fenced handoff is performed.
Cancel an unused preparation when its production model completes; readiness never authorizes promotion by itself.
Treat a one-node result as evidence only for its tested shape, batches, and memory.
Use W&B for numerical progress and exact Iris root/task IDs for liveness and concrete failures.
If checkpointing or evaluation exceeds three minutes without new optimizer updates, inspect the current GPU task attempts and logs before reporting the allocation as active.
A stale W&B `running` state and Iris root `RUNNING` can coexist with every GPU attempt queued after eviction; these states alone do not prove that a save is still executing.
An interrupted milestone without committed metadata is unavailable for benchmark handoff, and its pre-save diagnostic result must not be paired with a later replayed checkpoint at the same clock.
An Iris response reporting `PENDING` / peer-synced and no backend worker can lag actual execution; check the corresponding W&B identity and committed execution receipts before concluding that a new job has never started.
The evaluation task's `exp593-eval-text-q1-gate-a2` executed and failed in environment setup at 00:45 UTC on November 9 while Iris still returned pending at 00:48; its independently read receipt is `evaluations/plantcad2-v1/validation/text-q1/a2/validation-receipt.json` under the native owner, SHA-256 `4f2c0dbbc6ddcc90b965e19d23f471ea227c842fc1ea8aea34a7703e7ba37eb5`.
This observation does not establish a cluster-wide admission failure or permit a duplicate dispatch; evaluation recovery remains owned by that task.
Use eric-czech/marin at the CoreWeave W&B endpoint with an exp593 identity.
After startup, check progress on a 30-minute cadence.
Choose the periodic interval from measured native-write cost and observed interruption windows.
The first two writes after the qualified eight-to-64-H100 handoff each took 81 seconds, so the initial two-minute policy spends approximately 40% of a repeated training/save cycle checkpointing.
For ordinary placements, use ten-minute periodic checkpoints and retain six temporary saves, giving each newly written temporary checkpoint at least an hour of lifetime with approximately 12% estimated checkpoint cost at that measured write time.
The [qualified policy handoff](artifacts/dna216b-checkpoint-policy-rno-n8-v1/result.json) verified a periodic save after 601 seconds of training and a 76-second native write, approximately 11% of that repeated cycle before pointer verification and retention pruning.
The two-minute / thirty-save setting remains an emergency option if repeated short interruptions prevent longer intervals from making progress durable.
Prepare an allocation with a changed policy without stopping the current writer, then use the ordinary verified handoff.
The default remains thirty minutes and two temporary saves for existing serialized jobs; changing policy requires a new allocation with full-state recovery.
Checkpoint timing and retention are operational settings excluded from scientific bindings.
Iris owns preemption recovery; do not replace a dispatch solely because it was preempted.
Reassess placement after one hour without progress and diagnose a restart after three hours.
For a prolonged admission wait, compare the exact task with the fresh backend's pod status before classifying it as a software stall.
When both show an unassigned SchedulingGated pod, other work is blocked on node fit, and no superior eligible placement has fresh capacity, retain the queued dispatch rather than resetting its queue age without addressing the constraint.
Do not silently alter model/data/schedule semantics to obtain capacity.
The ledger's unfiltered placement-rate projection mixes bounded validation completion with production progress, so it cannot rank production targets directly.
Compare production dispatches at their scientific token horizons and use measured production throughput, including queueing, startup, validation and checkpoint overhead; a sixteen-update smoke reaching normalized progress one does not establish production speed.
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

The November 9 East eight-node preparation did not finish its first disposable probe before the original Reno writer recovered.
Native stacks from all eight workers on each of two sampled nodes waited in XLA ConfigAssigner's distributed key-value exchange; sampled GPUs were idle, while one sampled rank-zero process still consumed CPU.
This identifies the observed wait, not its cause or a proven deadlock.
The unused no-op-tracker allocation was cancelled without promotion and physically released after terminal root/child and post-cancellation backend checks.
East eight-node full-state recovery remains unvalidated; do not infer eligibility, change production compiler flags or retry blindly from this result.
See artifacts/dna216b-recovery-20261109-v1/result.json for the preserved recovery state and preparation evidence.

The additional DNA216B weight-only initialization passed sixteen updates on one East node and four Reno nodes at the original global batch and token schedule.
Both profiles exercised every sequence bucket, completed all fixed and functional panels, had finite metrics and zero routing drops, and committed strictly verified native checkpoints with identical clocks.
Their initial primary loss matches the evaluated source model within 2e-5, and their new optimizer/model counters start at zero.
These two profiles are eligible for the DNA-initialized production model; other new-source targets require their own bounded validation.
The sixteen-update checks do not reach production's first norm event at update 1,000; verify all twenty norm scalars there.
See artifacts/dna216b-loop-smoke-v1/result.json.
The DNA216B eight-node Reno resident standby subsequently restored its own permanent warmup checkpoint and passed sixteen probes, all buckets and norm branches, and every fixed/functional panel.
Its full-state resume is eligible; the first 1,000 promoted updates have finite losses, zero routing drops, twenty finite norms and input clocks matching the prior four-node history.
See artifacts/dna216b-production-reslice-rno-n8-v1/result.json.
The DNA216B one-node East profile also restored a full-state checkpoint produced by its eight-node Reno production gang.
It passed sixteen disposable updates, every bucket with and without norms, twenty finite norm scalars, zero routing drops and all fixed/functional panels.
This qualifies the tested 64→8 H100 recovery path, including optimizer and input cursor; it does not establish eligibility for other untested placements.
The no-op-tracker validation wrote no production state, was never promoted, and its allocation was physically released after readiness verification.
See artifacts/dna216b-recovery-east-n1-v1/result.json.
The DNA216B four-node East profile subsequently restored its permanent half-budget state from eight Reno nodes and passed seventeen disposable updates, every bucket and norm branch, twenty finite norm scalars, zero routing drops and all fixed/functional panels.
This qualifies the tested 64→32 H100 full-state recovery path; it does not qualify eight East nodes.
The resident preparation was released without promotion after the original Reno gang recovered, with terminal root/child and a post-cancellation backend observation showing every old pod absent.
See artifacts/dna216b-recovery-east-n4-v1/result.json.

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
The November 8 operator instruction prioritizes moving the DNA216B run to 64 H100s as soon as its held target passes the restore and numerical checks.
Do not delay that handoff for another throughput comparison; preserve the current writer until readiness and require physical writer termination before promotion.
Use the latest committed checkpoint and report any replay caused by the initial allocation lacking checkpoint-and-exit control.

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
That old-launcher failure subsequently occurred after scheduler preemption.
The text-initialized model now uses the ordinary production entry point, with the same identity and full native state; see artifacts/production-recovery-rno-n8-v1/result.json.
Repeated Reno preemption also led to promotion of the qualified eight-H100 East fallback for the DNA-initialized model; see artifacts/dna216b-production-reslice-east-n1-v1/result.json.
Its serialized checkpoint policy remains thirty minutes until a later allocation explicitly adopts shorter saves.

The first production handoff completed from step-35749 onto the already-held eight-node allocation, preserving the single W&B identity and exact native data/optimizer clock.
See artifacts/production-reslice-rno-n8-v1/result.json for the measured update gap, replayed logged tokens, actual resumed metrics and bounded throughput comparison.
The historical metric rows show that Iris controller/task KILLED state preceded the last old optimizer update.
For a future forced cancellation, require a fresh backend pod observation after the stop request showing the old allocation absent or physically terminal before publishing promotion.
A successful graceful checkpoint-and-exit is preferable for future handoffs.
The [qualified eight-to-64-H100 graceful handoff](artifacts/dna216b-production-reslice-rno-n8-cp120-v1/result.json) verified the exact saved cursor, successful process exits for both old tasks, and the first 1,000 resumed optimizer updates without replay.
For a natural successful exit, require the immutable stopped receipt, its matching native checkpoint clock, and successful root/child task exits before promotion.
If every GPU task has succeeded with exit code zero after acknowledging the exact checkpoint, a CPU-only coordinator still waiting on that completed child may be cancelled.
Record all GPU process exits before cancelling that exact root, then recheck both old jobs are terminal before promotion.
The all-task success certificate proves physical writer termination; terminal root/child checks fence re-entry.
Inspect an idle coordinator promptly after GPU completion to avoid unnecessary cutover delay.
Do not infer that resources or writers have stopped merely from cancellation acknowledgement.
Once a post-cancellation backend observation proves every old pod absent, that termination fact remains valid while fresh controller checks confirm both unique old jobs are terminal.
Recheck those terminal states immediately before promotion; a capacity observation's age does not resurrect a physically terminated writer.

## Change record

November 9, 22:55 UTC: a priority eviction interrupted the DNA216B three-quarter save while W&B still reported checkpointing/running.
Added a three-minute task-attempt check for prolonged save/evaluation phases and explicit exclusion of interrupted milestone diagnostics from later checkpoint pairing.
See artifacts/production-recovery-east-n4-v1/result.json for the observed failure and native commit check.

November 10, 00:00 UTC: repeated priority evictions left Reno without free H100s while both East writers were productive.
Clarified the existing operator authorization for one queued nonblocking 64-H100 preparation, preserving the GPU ceiling, batch priority, single-writer gate and fresh capacity observations.
