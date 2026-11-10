# exp593 annotation-guided annealing operations

Read this document before every placement or recovery decision for the three additional annealing runs.
It supplements the historical operations document without reopening its completed runs or sealed ledger.
The full authorized research plan remains untracked at scratch/exp593-annealing-plan.md.

## Scientific ownership

The immutable trial catalog and LR/FIM schedules live in src/exp593_dna/anneal.py.
The entry point is exp593_dna.anneal_train, with data, native state, batching, audit, and compact validation modules prefixed anneal_.
These runs expand issue #593 and the existing permanent MarinDNA PR #594; keep the branch unmerged and do not create another issue or PR.
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
| cw-rno2a | H100 | 1 | 8 | eligible | Bounded validation only |
| cw-rno2a | H100 | 2 | 16 | unvalidated | — |
| cw-rno2a | H100 | 4 | 32 | eligible | Bounded validation only |
| cw-rno2a | H100 | 8 | 64 | eligible | Bounded validation only |
| cw-us-east-02a | H100 | 1 | 8 | eligible | Bounded validation only |
| cw-us-east-02a | H100 | 2 | 16 | unvalidated | — |
| cw-us-east-02a | H100 | 4 | 32 | eligible | Bounded validation only |
| cw-us-east-02a | H100 | 8 | 64 | unvalidated | Prior distributed compilation wait unresolved |

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

None.
