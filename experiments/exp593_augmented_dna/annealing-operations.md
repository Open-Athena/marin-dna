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
On November 11 the user superseded the three v1 jobs with an equal-scored-base restart: each exact old root and child was cancelled and physically fenced, and its latest checkpoint pointer preserved in the operations ledger. The v1 ledger trials are abandoned, not resumed. The replacement v2 trials restore the same original 1T full native source states and retain the fixed source-specific horizons and LR/FIM schedules; their checkpoint and personal W&B identities contain `equalbase-seed0-v2`.
The new encoder gives every scored causal or FIM base unit weight. The weighted-mean training reduction divides by all scored targets in the batch; it does not normalize within examples. New monitoring includes held-out token-weighted `eval/fim_loss` and exp586-style analytic padded-bucket MFU. W&B chart series are aggregate; cumulative species counts and individual zero-shot task scores live in summary records.
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
| cw-rno2a | H100 | 1 | 8 | eligible | Bounded v2 validation; production requires a v2 source-specific full-loop result |
| cw-rno2a | H100 | 2 | 16 | unvalidated | — |
| cw-rno2a | H100 | 4 | 32 | eligible | Bounded v2 validation; production requires a v2 source-specific full-loop result |
| cw-rno2a | H100 | 8 | 64 | unqualified | New bounded diagnostic basis required; no production placement |
| cw-us-east-02a | H100 | 1 | 8 | eligible | Bounded v2 validation; production requires a v2 source-specific full-loop result |
| cw-us-east-02a | H100 | 2 | 16 | unvalidated | — |
| cw-us-east-02a | H100 | 4 | 32 | eligible | Bounded v2 validation; production requires a v2 source-specific full-loop result |
| cw-us-east-02a | H100 | 8 | 64 | eligible | Bounded validation only; static H100 fit established, distributed runtime remains unqualified |

## Operating policy

Use W&B numerical progress first, exact Iris root/child liveness second, and freshly refetched ListPeers availability before admission or resizing.
Apply the existing v3 capacity helper and historical bounded observation-age rules; higher-priority holds are not free capacity.
Use one pending admission per exact target, except for an already approved resident replacement queued while productive training continues.
The qualified 545b four-node resize may also overlap the pending 181b four-node validation when fresh capacity covers the additional 32 H100s within the aggregate limit; that production admission uses a proven target rather than another unqualified profile.
Run a complete agent decision pass every 30 minutes after startup; use one-hour reslice and three-hour restart thresholds unless a concrete failure warrants earlier diagnosis.
Helpers may observe, validate, and execute an exact agent-selected action; no unattended helper may decide placement, recovery, or training termination.
After ambiguous submission, reconcile only its exact immutable Iris name before replacement.
Physically fence the old writer before its replacement restores and writes the same identity.
Completion requires the exact fixed-stage token horizon, reachable permanent checkpoint, native clocks/state integrity, and terminal GPU writers.
The read-only operator audit in scratch/exp593-anneal/audit_production_checkpoint.py checks remote metadata, native manifest, source bindings, W&B results and exact Iris writers; it cannot certify tensor contents. After each successful final run, use exp593_dna.anneal_final_audit on an additional batch/eczech eight-H100 node with the selected checkpoint and metadata digest to restore and check every native model/optimizer array before recording completion. This audit performs no training updates and uses a separate Iris identity.
The first disposable native audit, on the 181b step-356535 intermediate checkpoint, succeeded on eight Reno H100s: all 128 arrays restored and were finite, compute weights matched FP32 master casts, the optimizer count matched the global step, and the clock matched metadata. Its immutable full-readback report and exact terminal root/child are recorded in the separate audit trial in the annealing ledger. This confirms the native audit procedure on a live production checkpoint; each final checkpoint still requires its own audit.

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
Both cache-isolated eight-node checks subsequently stopped numerical progress in native XLA ConfigAssigner/BlockingKeyValueGet waits, sampled on one node per placement with idle sampled-node GPUs.
Both exact root and child writers were cancelled and verified terminal; cache-mirror isolation alone did not resolve distributed compilation.
Further eight-node checks require a concrete new diagnostic basis rather than repeating either launch; the historical operations document also records an unsuccessful shard-autotuning override.
An additional bounded runtime diagnostic may set --xla_gpu_autotune_level=0 to bypass online autotuning and its distributed configuration exchange, based on the sampled waits and the OpenXLA implementation.
Any resulting qualification applies only to that runtime variant; retain the default production flags until full numerical/state checks and measured throughput support a placement change.
The 64-H100 Reno autotuning-off diagnostic completed its 64 updates and real-task pilot, but failed the native full-state checkpoint roundtrip and ran slower than the qualified 32-H100 default profile. Its exact Iris root/child and disposable W&B run failed; the ledger dispatch was closed and the validation trial abandoned. The committed checkpoint has 128 manifest arrays and the expected stage/global clock, but the current evidence does not distinguish a clock mismatch from a state-array difference in the roundtrip guard. Do not promote this runtime variant or retry it without a concrete new diagnostic basis.
Three bounded global-primary native faults coincided with the first in-process watchdog dump during input or restore work; production does not enable that timer.
New bounded checks use anneal_validation.py to isolate the timer while leaving the native scientifically bound trainer unchanged; diagnose waits with external Iris task profiles.
The 545b four-node Reno check passed all 42 finite task pilot scores, 64 updates, every context bucket, first-update head/master reconciliation, and an exact full-state checkpoint roundtrip.
That profile is eligible for production while other source-specific four-node checks proceed.
The timer-isolated 364b four-node East check passed all 42 finite task pilot scores, 64 updates, every context bucket, first-update reconciliation, and an exact full-state checkpoint roundtrip.
That default optimized profile is eligible for 364b production.
The timer-isolated 181b four-node Reno check passed the same complete validation contract and qualifies that default optimized profile for 181b production.
Iris terminal states precede physical container teardown; positively identify the old containers before cancellation and require their absence in a fresh post-cancellation backend inventory before replacement admission.
Nested W&B summary fields can retain values from the original source import after a resume; use the scalar resume step, appended numerical history, committed native clocks and enforced import guards for recovery verification.

The equal-scored-base v2b source at revision 860778c3 passed source-specific 64-update validation on 32 H100s for all three recipes, including finite FIM loss and analytic MFU, first-update model/master reconciliation, 42 finite pilot task scores, all four context buckets, and exact native checkpoint roundtrips. The qualified production placements are Reno four nodes for 181b and East four nodes for 364b and 545b. All three new personal W&B identities have advanced numerically on those placements. At a constant 30% FIM example rate, the bounded validators measured 0.33–0.35% FIM scored targets, as expected from unit weight per scored base. Keep the fixed curriculum and schedules unchanged.
The exact v2 step-zero validation and 42-task baselines were captured from each new run before annealing input, not inherited from v1. Their complete-readback immutable record is `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/evidence/task-baselines-v2-b95865250978a121f36495ba97ac2c0760bb2fff86ba209057a12174d2fa05d8.json`; final results should use this source comparison.
The first equal-base production checkpoint native audit succeeded on 545b step-1042670 while its writer continued. Its immutable report at `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/native_audits/dna-exp593-anneal-545b-guidedfim30-equalbase-seed0-v2/step-1042670-d2fb358be4313529f614d3fb546a7edca162887cffbefac5db6c9e342efff85f.json` verified all 128 arrays, finiteness, model/master consistency, and optimizer clock with complete readback. Both separate audit Iris writers succeeded. This interim check does not replace each branch's required final native audit.
W&B flattens the final native clock into a second per-species summary family. After a run is finished and audited, remove only `training/final_clock.mixture.species/*` from its W&B summary; retain the 65 canonical `data_mix/species_examples.*` counts and the complete mixture in checkpoint metadata. This procedure was verified on all three completed disposable validators without changing their scientific metrics; the immutable before-state/verification record is `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/evidence/wandb-validation-species-cleanup-v2b-5cadf379356ddf0b96347976620317b5a7ecd6820447e689944ad94d4c43f2bc.json`.
