# exp593 annotation-guided annealing operations

Read this document before every placement or recovery decision for the four additional annealing runs.
It supplements the historical operations document without reopening its completed runs or sealed ledger.
The full authorized research plan remains untracked at scratch/exp593-annealing-plan.md.

## Scientific ownership

The immutable trial catalog and LR/FIM schedules live in src/exp593_dna/anneal.py.
The three FIM runs use exp593_dna.anneal_train; the old-distribution control uses exp593_dna.anneal_control_train. Data, native state, batching, audit, and compact validation modules are prefixed anneal_.
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
Other accelerator families and changes to scientific global batch remain outside scope. The eight-node Reno qualification for the old-distribution control does not qualify the three separate FIM recipes for eight nodes.
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
The same independent native restore check also passed on 181b step-353106 and 364b step-700893 while their production writers continued. The readback-verified immutable reports are `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/native_audits/dna-exp593-anneal-181b-guidedfim30-equalbase-seed0-v2/step-353106-946d5ea56fe28476410668e9e88ff8b6741a93dbc043685b92292889164cc06a.json` and `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/native_audits/dna-exp593-anneal-364b-guidedfim30-equalbase-seed0-v2/step-700893-6adae22c440f86fade9a60de2449afa35a146e42c0a5e12afd06406e48daaac6.json`. All 128 arrays were finite, compute weights matched FP32 masters, optimizer clocks matched, and each separate batch/eczech Iris root and child succeeded. Final native audits are still required.
W&B flattens the final native clock into a second per-species summary family. After a run is finished and audited, remove only `training/final_clock.mixture.species/*` from its W&B summary; retain the 65 canonical `data_mix/species_examples.*` counts and the complete mixture in checkpoint metadata. This procedure was verified on all three completed disposable validators without changing their scientific metrics; the immutable before-state/verification record is `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/evidence/wandb-validation-species-cleanup-v2b-5cadf379356ddf0b96347976620317b5a7ecd6820447e689944ad94d4c43f2bc.json`.

## Old-distribution control added November 11

The user authorized a fourth branch from the same 545.455B permanent 1T source checkpoint and complete native model/FP32-master/optimizer/global-step state. Its separate entry point, `exp593_dna.anneal_control_train`, is at scientific revision `66c9c57e`, with new control-only files so the 19-file bindings of the live FIM branches remain unchanged. It uses the same optimized update, model, optimizer, LR program, validation/task panels and monitoring cadence as the 545b FIM branch. Relative to the old 1T executable, both new branches use local scatter MoE transport, device-resident FP32 masters and an inherited-LR hold for the first 20% of their stage followed by linear cooling to 5% of the old peak. The controlled variable is the training data presentation.

The control's training stream uses the exact old corpus inventory and seed-0 block/Feistel mixture shuffle; each original 8192-base window is lowercase-tokenized with the pinned exp586 DNA tokenizer, has the exp586 occurrence-seeded reverse-complement draw, and scores its 8191 next-base targets once each. It has no FIM, text, metadata, crops, packing, or partial windows. The nearest whole-batch horizon below 120B is `119999561728` input tokens (`228881` updates), 438272 tokens short of nominal 120B. Its W&B identity is `dna-exp593-anneal-545b-originaldata-equalbase-seed0-v1`, with an independent production checkpoint namespace under the same annealing owner.

The first disposable smoke failed before any update because its validation-only data clock reused the FIM ramp override. Its exact root and child reached terminal failed states; the ledger trial `smoke-545b-control-rno-n8-v1` is abandoned. The corrected 64-update eight-node Reno smoke, `smoke-545b-control-rno-n8-v2`, succeeded on both Iris writers: exact source optimizer count 1040373 and 54 optimizer arrays, approved historical BF16 head preserved until the first normal update, zero FIM/text examples, all 42 finite pilot task scores, finite FIM validation loss and MFU, and a zero-error full native checkpoint roundtrip. The verification receipt is `scratch/exp593-anneal/smoke-545b-control-rno-n8-v2-verification.json` and the ledger backup is immutable in the same operations prefix.

Production trial `v1-control-545b` was submitted on 64 Reno H100s as batch/eczech under `/eczech/exp593-anneal-v1-control-545b-rno-n8-a1`. The personal W&B run has advanced past 158M input tokens; `scratch/exp593-anneal/v1-control-545b-production-advancing.json` independently verifies numerical progress, exact root/child liveness, restored optimizer state, zero FIM/text data, original full-window target counts, and its new permanent checkpoint namespace. The read-only remote checkpoint audit is `scratch/exp593-anneal/audit_control_checkpoint.py`. Keep the four runs' identities independent; do not change their schedules based on losses. The control also needs a final full native tensor audit before completion.
At 14.949B additional input tokens, the live control still had exactly 8192 input bases and 8191 scored next-base targets per example, zero FIM/text examples or FIM targets, and 49.953% reverse-complemented examples. The asserted W&B snapshot is `scratch/exp593-anneal/control-live-original-data-invariant.json`.
The control's exact production step-zero loss and 42-score task baseline is frozen at `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/evidence/control-task-baseline-v1-d69134257b0906c8a6e731c0df861780682d01761c4facebc17c5aa659b0c7dd.json`, with complete readback. Its first post-update checkpoint at global step 1042947 passed the remote metadata/manifest audit and an independent eight-H100 native restore of all 128 state arrays. The immutable native report is `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/native_audits/dna-exp593-anneal-545b-originaldata-equalbase-seed0-v1/step-1042947-dc11e7c27bf5be4d37012266aa3971d5620c7b679d1a2e58bb5105e109c152a3.json`; both audit Iris writers succeeded and the audit ledger trial is closed. This interim check does not replace the required final control native audit.
The 545b FIM and control step-zero baselines bind the same source checkpoint and task-panel metric keys, but their evaluated values are not bit-identical: primary loss differs by 0.0000110 and the largest difference among 42 task scores is 0.00463. Matched comparisons therefore use changes from each arm's own frozen step-zero baseline.

The matched first fixed validation at approximately 5.243B additional input tokens is frozen with complete readback at `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/evidence/545b-fim-vs-control-first-eval-6d8e0348f1ebb1c4e3ae338ec4a0cc71bce9f6a8da63df8c0e3ef616d056f6b2.json`. Against each arm's own source baseline, the 545b guided-FIM branch changed primary `eval/loss` by +0.0195453 and FIM validation loss by -0.0254222; the original-data control changed them by -0.0005917 and -0.0009941. Thus the causal validation loss rose in the FIM branch while slightly improving in the control over this first interval. These are observations only; preserve both fixed curricula and LR schedules through their planned horizons.
The 545b FIM run's W&B history row immediately before that validation (`_step=15216`) records 32,280 FIM examples, 3.315% of examples seen, but only 1,424,298 FIM scored targets among 5,070,294,866 total scored targets (0.0281%). The instantaneous FIM selection probability was 6.553% at that point in the ramp. Its causal examples also use variable-length crops of original train windows, whereas the control retains complete original 8192-base windows. Therefore this comparison distinguishes the whole augmented presentation from the original distribution; it does not isolate the small FIM gradient contribution from changes to causal-window presentation.
The second matched fixed validation at approximately 10.486B added tokens is frozen with complete readback at `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/evidence/545b-fim-vs-control-second-eval-c71259e9bf15d985f4f4c6a54dd0b7ee6fe57065dadd21a03b4ce2343d3df24b.json`. Primary `eval/loss` changed by +0.0234759 from source in the guided-FIM branch and -0.0006322 in the original-data control; FIM validation loss changed by -0.0215960 and +0.0035280, respectively. The primary-loss separation therefore persisted through a second interval. Both fixed training schedules continue unchanged.
The third matched fixed validation at approximately 15.729B added tokens is frozen with complete readback at `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/evidence/545b-fim-vs-control-third-eval-ae45de94c558b7fa5a294530c712d125328996ec434606e9f4fcfb35c94a0bdc.json`. Relative to each arm's source baseline, primary `eval/loss` changed by +0.0279458 in the guided-FIM branch and -0.0008251 in the original-data control; FIM validation loss changed by -0.0437819 and +0.0060603. Both fixed schedules continue unchanged.
The nearest training-history rows to that third validation (`_step` 45538 for FIM and 30003 for control, within 23768 input tokens of each other) record identical MuON LR `0.0009499295373467996` and Adam LR `0.00021921450861849223`, confirming that the primary-loss separation at this point is not from different LR values.

The first fixed 21-task panels, with 2048 examples per task in each raw and FIM presentation, are frozen at `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/evidence/364b-first-task-panel-5d1842334c985ee335d0e36f3722842f37dc23a62d704ac0e5d2cd0fcd164b68.json` and `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/evidence/545b-first-task-panel-63836f23640511edb2d6b322ad8405b4f9ed5b859e251e5061442b3a494d6433.json`, both with complete readback. Relative to each run's own source panel, the 364b branch improved 21/21 FIM task scores and 3/21 raw task scores; the 545b branch improved 21/21 FIM and 7/21 raw. For example, maize acceptor motif accuracy in the 364b branch changed from 0.000 to 0.946 under FIM presentation and from 0.801 to 0.761 under raw presentation. These are descriptive first-panel results; the fixed schedules continue regardless of task scores.
The 181b first fixed panel is also frozen with complete readback at `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/annealing/evidence/181b-first-task-panel-dc870bb2e39e4560fd5586e5c061529da863177aaebcf2139410ff3e42f0942c.json`. It improved 21/21 FIM task scores and 3/21 raw scores from its own source. Maize acceptor motif accuracy changed from 0.0005 to 0.9468 in FIM presentation and from 0.7666 to 0.7231 in raw presentation. The fixed schedule continues unchanged.

The data-path audit found no route from the fixed evaluation task pack or its labels into training: `anneal_train.py` opens only the original `train` corpus split and `annotations/train.parquet`, and `GuidedAugmenter` rejects non-train rows and verifies each annotation against its source DNA. `anneal_task_data.py` builds the 21-task panel separately for evaluation. This establishes code-path separation, not genomic-locus disjointness: task-panel rows contain 8192-base sequences but no genomic coordinates, and no full sequence-overlap audit against the train corpus has been done. The large FIM task gains should be interpreted with that limit.
