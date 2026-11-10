# exp593 annotation-guided annealing operations

Read this document before every placement or recovery decision for the three additional annealing runs.
It supplements the historical operations document without reopening its completed runs or sealed ledger.
The full authorized research plan remains untracked at scratch/exp593-annealing-plan.md.

## Scientific ownership

The immutable trial catalog and LR/FIM schedules live in src/exp593_dna/anneal.py.
The entry point is exp593_dna.anneal_train, with data, native state, batching, audit, and compact validation modules prefixed anneal_.
These runs expand issue #593 and the existing permanent MarinDNA PR #594; keep the branch unmerged and do not create another issue or PR.
The user requires explicit approval before posting or editing any GitHub comment.
Keep operational notes local and report progress in chat; do not restore comments the user deletes.
The user's November 10 authorization permits implementation, bounded validation, and all three full training runs with new personal W&B identities.
Schedules are fixed; evaluation results must never trigger early stopping, LR changes, horizon changes, or enrichment of the training sampler.
Do not initialize from weights alone, reset optimizer/global counters, or infer repeat labels from uniformly lowercase inputs.
All FIM samples are annotation-guided original training rows; raw causal samples retain the original all-species stream.
Every source checkpoint remains read-only.

## Resource authorization

Use H100s on cw-rno2a and cw-us-east-02a, with whole eight-GPU nodes and at most eight nodes per branch at fixed global batch 64.
The approved aggregate ceiling is 512 GPUs, counting productive, validation, and replacement allocations.
Prefer 64-H100 production gangs from the outset when qualified and available.
Submit every root through the marin controller with explicit --target-cluster, --priority batch, and --user eczech.
The launcher explicitly assigns batch priority to child gangs and forwards storage/W&B credentials without printing them.
Other accelerator families and changes to scientific global batch remain outside scope.
There is no experiment-level calendar deadline and the user explicitly prohibited early stopping; allocation timeouts remain recoverable placement events.

## Durable identities and state

The new local ledger is scratch/exp593-anneal/operations.sqlite.
Never mutate scratch/exp593/operations.sqlite or either completed #593 run.
Native checkpoints use the per-run annealing namespace selected by TrainingConfig in src/exp593_dna/anneal_train.py.
Immutable ledger backups belong to s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/operations/.
Use the repository transactional persistence helper and exp593_dna.publish_inputs.put_immutable for fresh timestamp/UUID-named backups, verifying full size and SHA-256 readback after every mutation and before the next external action.
On local state loss, restore the newest complete, checksum-valid immutable backup and reconcile every unresolved exact dispatch intent before submission.
Keep exactly one production writer per W&B/checkpoint identity; preserve it across retries or reslicing.
Disposable optimizer updates require separate validation identities and roots.

## Placement validation

All three source native manifests have identical 128-array path/shape/dtype schemas; live metadata hashes and source clocks were reverified before implementation.
Existing optimized runs establish hardware fit for one/four/eight H100-node execution, but the new full-state imports and sampler require bounded loop verification.
Targets marked eligible below permit bounded validation only until the source-specific train/eval/save/restore checks pass.

| Cluster | GPU | Nodes | GPUs | State | Scope |
| --- | --- | ---: | ---: | --- | --- |
| cw-rno2a | H100 | 1 | 8 | eligible | Production for the qualified 181b and 545b recipes; bounded validation for other sources |
| cw-rno2a | H100 | 2 | 16 | unvalidated | — |
| cw-rno2a | H100 | 4 | 32 | eligible | Bounded validation only |
| cw-rno2a | H100 | 8 | 64 | eligible | Bounded validation only |
| cw-us-east-02a | H100 | 1 | 8 | eligible | Production for the qualified 181b and 364b recipes; bounded validation for other sources |
| cw-us-east-02a | H100 | 2 | 16 | unvalidated | — |
| cw-us-east-02a | H100 | 4 | 32 | eligible | Bounded validation only |
| cw-us-east-02a | H100 | 8 | 64 | eligible | Bounded validation only; static H100 fit established, distributed runtime remains unqualified |

## Operating policy

Use W&B numerical progress first, exact Iris root/child liveness second, and freshly refetched ListPeers availability before admission or resizing.
Apply the existing v3 capacity helper and historical bounded observation-age rules; higher-priority holds are not free capacity.
Use one pending admission per exact target, except for an already approved resident replacement queued while productive training continues.
Run a complete agent decision pass every 30 minutes after startup; use one-hour reslice and three-hour restart thresholds unless a concrete failure warrants earlier diagnosis.
Helpers may observe, validate, and execute an exact agent-selected action; no unattended helper may decide placement, recovery, or training termination.
After ambiguous submission, reconcile only its exact immutable Iris name before replacement.
Physically fence the old writer before its replacement restores and writes the same identity.
Completion requires the exact fixed-stage token horizon, reachable permanent checkpoint, native clocks/state integrity, and terminal GPU writers.

## Change record

The 364b recipe passed its complete source-specific task/train/eval/save/restore loop on one East node, including all four buckets, 42 finite real-task pilot scores, and an exact native checkpoint roundtrip.
Its proven placement can advance production while larger gangs are validated in parallel; preserve the same production W&B/checkpoint identity when scaling up.
The first new eight-node Reno check completed the task pilot and one optimizer update but did not advance further; retain bounded-only qualification until that wait is diagnosed and the complete loop passes.
The 545b legacy output-head/master discrepancy is preserved under the user's explicit approval, with strict source/stage and discrepancy guards in anneal_state.py.
Its corrected one-node Reno check passed all 42 task pilot scores, 64 updates, all four context buckets, and a zero-error full-state checkpoint roundtrip; the first normal update reconciled the BF16 head and FP32 master.
The 181b source passed the complete one-node native loop and the real-task pilot on the larger gang; those combined checks qualify its single-node production fallback, not the stalled larger placement.
Eight-node East is eligible for bounded testing because the same H100 per-device model/optimizer layout fits and executes on Reno, while East's single-node inputs and runtime are verified.
An explicit empty --xla_gpu_per_fusion_autotune_cache_dir flag can isolate the bounded East test from Iris's shared FineStore autotune mirror; this changes cache I/O only and keeps GPU autotuning, the full executable cache, kernels, optimizer, and objective intact.
The same cache-isolation check is permitted on Reno after a later bounded attempt exited with a native process fault during source restoration; that fault does not establish the cause of the earlier post-update wait.
