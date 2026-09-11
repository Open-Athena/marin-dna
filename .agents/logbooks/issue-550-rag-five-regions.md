---
topic: issue-550-rag-five-regions
issue: https://github.com/Open-Athena/marin-dna/issues/550
description: Five-region 46M RAG training with order-deduplicated vertebrates.
author: gonzalobenegas
---

# Issue 550: Task Logbook

## Scope

Train the agreed 46M model for 100,000 updates with 200 documents per update and a uniform five-region mixture.
Use the coordinating issue as the frozen scientific specification.
Training uses free Iris TPUs; data construction and development VEP share a hard $30 cap including pilots and retries.
No held-out labeled evaluation is authorized.
Keep experimental artifacts on `codex/issue-550-rag-five-regions`; extract reusable changes and registrations through reviewed pull requests.
Do not merge a pull request without explicit user approval.

## Baseline

The starting maintained code is `922e4114` with chain projection merged in #549.
Historical RAG #402 uses fixed seven-mammal context, 2,048 tokens, and 30,000 updates.
The new recipe changes multiple factors together and cannot isolate their causal contributions.

## Hypothesis Queue

- RAG-550-001: The agreed combined recipe improves development VEP performance; evaluate each benchmark independently with canonical metrics and uncertainty.
- RAG-550-002: One combined inference job reduces checkpoint setup and warmup costs; test joint versus separate processing on the same development fixture before budgeting the full schedule.

## Entry Log

### 2026-09-09 — RAG-550-000: Execution preflight

The user requested autonomous work on #550 and asked for foreseeable questions at the start.
The issue already authorizes the resource plan, publication, seeds, data geometry, split, and full development evaluation schedule.
No further scope permission was requested; paid execution must fit the cap and PR merges remain a human decision.
`aws sts get-caller-identity` and listing the pinned chain release succeeded.
`sky api info` reports no connected server; direct AWS access is available.
No paid worker has been started and no biological data have been processed.
Background research uses high effort for independent launch-compatibility and asset-provenance source checks before expensive execution.
The two example current-consumer dependency endpoints in the Marin skill returned 404; current upstream Marin source is the fallback.
The initial implementation will preserve all legacy projection rules and add an explicit RAG namespace.

### 2026-09-09 — RAG-550-001: User decisions and first implementation checks

Gonzalo replaced the random cross-chromosome validation split with a complete chr18 training holdout.
Sample up to 400 original-orientation chr18 validation documents per region using seed 42, and use all available if fewer exist.
No additional genome-wide overlap exclusion is required.
Region memberships outside chr18 remain intact; exact duplicate requests can share projection work.
Gonzalo also requested final-checkpoint VEP and frozen probes first, with earlier-checkpoint evaluations conditional on the remaining $30 budget.
The 100,000-update training horizon, checkpoint/export cadence, and language-model validation cadence are unchanged.
The coordinating issue body and comments record both decisions.
The user confirmed use of the scaling heuristic; its original reference units are being checked before resolving the optimizer.

The shared development VM's heavy-work lock was occupied, so the attempted locked local environment sync aborted immediately.
No lock polling or local biological work was performed.
A dedicated r6i.2xlarge CPU spot pilot was launched at 21:54:56 UTC: `i-058752e5529fb337b`, with a $0.20/hour bid ceiling and automatic termination after 110 minutes.
Its 200-GiB root volume deletes on termination.
The instance is isolated to issue 550 and uses the established S3 access profile.
Current quote at launch was $0.1707/hour; initial compute/disk allowance is below $1, counted against the $30 ceiling.

Command: from the owning projection project, `uv sync --locked --group dev` then `uv run --locked pytest`.
The initial run passed 276 tests but failed an existing storage integration test because Conda was missing.
After installing Conda on the pilot and applying Ruff formatting, all 277 tests passed and one Kent integration test skipped; wall time 8.61 seconds and peak RSS 295,768 KiB.
These tests cover the initial document, chr18 split, and exact request-reuse modules; later file adapters still require validation.
The source inventory confirms 39 target species and about 37 GiB of archived chain/genome inputs.
Staging streams bytes through SHA-256, uses S3 If-Match against the inspected object ETags, and checks byte counts; ETags are not treated as SHA-256.
No biological projection, Hugging Face publication, training, or VEP inference has yet run.

### 2026-09-09 22:38 UTC — RAG-550-002: Pinned inputs and additive workflow

All source archives finished staging and hashing in 7m33.72s with peak RSS 125,376 KiB.
All 18 mammalian chain hashes match their archived generation receipts.
The pinned inputs contain 39 target species and 565,959 source anchors across the five regions.
Chr18 availability is 4,933 CDS, 1,048 TSS/UTR5, 1,380 UTR3, 617 ncRNA, and 2,978 enhancer anchors; all regions support 400 validation documents.
Expected forward-plus-RC training row counts are 581,256; 112,740; 131,550; 56,396; and 228,064 respectively.
At four million expected draws per region, these correspond to 6.88, 35.48, 30.41, 70.93, and 17.54 effective epochs.
The three pinned official development splits contain 16,140 Mendelian, 11,630 complex-trait, and 23,853 SGE rows.
No held-out labeled file was downloaded.

The complete synthetic RAG integration test passes with real Kent 482 executables: three unique requests retain 24 source memberships and produce five datasets and three benchmark cohorts.
The existing chain workflow integration also passes with Kent installed.
The biological dry-run plans 365 jobs using registered anchor/benchmark inputs, chain validation/projection, genome validation/extraction, and RAG assembly.
No HAL generation, MultiZ scanning, or upstream anchor construction is scheduled.
Commands and immutable configuration are in the workflow README.

The training project resolves on Python 3.12 because the required Marin Resiliparse fork lacks a Python 3.13 wheel.
It imports no root MarinDNA symbols, so the unused root dependency was removed to avoid conflicting Python/Transformers constraints.
The Marin family is pinned to `0.2.106.dev34338714012`, release source `efe79892065589b154d969effd49eee3bd286284` published 2026-09-09.
Import testing exposed a missing transitive `dupekit` package; its Python wrapper is being pinned to that same source release.
The token-normalized AdamH heuristic uses projection LR 0.004695897205758308, Adam LR 0.00020258341260453682, epsilon 5.990618799422977e-8, beta2 0.9992190160617281, beta1 0.9, max gradient norm 0.1, and 10% warmup / 20% linear decay.
Reference batch size is 64 x 4096 positions; using allocated-token units across context lengths is an explicit experimental extension of the fixed-context heuristic.
Training and evaluation have not started.

### 2026-09-09 23:27 UTC — RAG-550-003: Producer progress, publication review, and training preflight

The biological workflow runs from immutable snapshot `6b1593c274a886d20f5c0ddf3712916d446f5fed` in a separate worker worktree.
Its request audit contains 593,468 exact unique projection requests and 617,582 source memberships, including the 51,623 canonical development benchmark rows.
At 23:12 UTC it had completed 283 of 365 steps; the remaining elephant liftOver process was actively using one CPU at the 23:20 check.
The worker's termination deadline was extended at 22:44 UTC to 2026-09-10 01:44:46 UTC, retaining a total compute/disk allowance below $1 and the overall $30 cap.
No source credentials were printed or committed.

Reusable dataset code was extracted into issue #551 and draft PR #552.
An independent published-diff review identified the inherited legacy anchor checksum and an experiment-specific epoch estimate in generic provenance.
The additive RAG requests rule now validates its own source contracts and shared dictionaries without applying a legacy catalog checksum; exposure estimates belong to this experiment consumer.
Publication provenance now records source producer and publisher separately, including the actual active storage prefix for row mappings and release hashes.
The integration test publishes from an older producing snapshot under a newer publisher snapshot and verifies every reconstructed public row.
The complete suite passed 282 tests and failed one existing storage test because the retest shell omitted Conda from PATH; both storage tests passed after restoring the known Conda PATH.
Thus all 283 tests passed across that run and the targeted environment-only retry, including both Kent integration workflows.
Ruff and Snakefmt passed.

The self-contained training project resolves the coherent current Marin release `0.2.106.dev34338714012` from upstream source `efe79892065589b154d969effd49eee3bd286284`, with Python 3.12 and uv 0.11.31.
The missing published dependency `marin-dupekit` is pinned to the same source commit.
All 13 training tests pass, including real-logit/loss/gradient invariance to padding, native-JAX versus exported-Hugging-Face logits, the fixed recipe, and completed-update milestone selection.
The standard Marin tokenization path successfully generated all five synthetic train/validation caches using a local Zephyr child on the paid data worker, without dispatching CPU jobs.
The launch graph constructs correctly for a synthetic 20-update pilot on one free preemptible v6e-8 slice in us-east1.
Microbatch 5 accumulates to exactly 200 documents; fallback 1 preserves the same batch.
The resolved scaling heuristic and its allocated-token context-transfer assumption are recorded in the experiment README and source.
No TPU has been submitted at this snapshot; actual native save/resume, milestone files, W&B progress, and throughput remain pilot checks.

### 2026-09-09 23:55 UTC — RAG-550-004: Published framework, compute gates, and Spot interruption

Dataset-framework PR #552 passed all CI checks and independent follow-up review and is ready for human review at `0a5e31300fd94f179e6478e376604c7b18b08194`.
No PR was merged.
The experiment and synthetic pilot were committed and pushed at `4c00035108d0fcd64790a51aea3fee88a0108323`.
Automatic approval review rejected the pilot submission because it does not accept the linked issue's compute agreement as trusted authorization for TPU resource use and the W&B credential side effect.
A direct approval request for the free TPU pilot/full run and W&B logging is pending; no TPU was submitted.

AWS Spot request `sir-ww3fkg5m` reports `instance-terminated-no-capacity` at 23:32:27 UTC; instance `i-058752e5529fb337b` terminated at 23:34:30 UTC.
The actual lifetime was approximately 99.6 minutes, before the extended shutdown deadline.
The producing RAG namespace contains 212 durable S3 objects totaling 3,038,871,673 bytes.
Final document assembly had not started; the last observed long-running process was the elephant liftOver query.
An on-demand r6i.2xlarge replacement was prepared with automatic termination after four hours and delete-on-termination disk, estimated at approximately $2.20 in compute/disk costs.
Automatic approval review also rejected that replacement because it requires the $30 paid-resource cap to be explicitly approved in the task rather than only recorded in the linked issue.
That approval request is pending; no replacement was launched.
Do not retry either blocked action without a user reply.

Before interruption, the human-window audit matched all 15,990 unique Mendelian loci, 11,629 complex-trait loci, and 9,276 SGE loci against the producer's human sequences.
All human benchmark windows contain only ACGT, with no Ns on either flank.
This preserves the maintained four-nucleotide LLR/JSD contract without removing canonical variants; genuine Ns in retrieval context remain permitted.
The audit covered sequence/reference properties only and computed no predictions or label aggregates.

The combined evals_v2 backend and tests are drafted on this research branch and extracted through issue #553.
It validates canonical source rows before loading a checkpoint, uses fixed-shape variable-human-position scoring through the existing Trainer prediction loop, pools only 255 human tokens, and routes outputs to the three canonical score files.
Ruff 0.16.2, Snakefmt 2.0.3, and Python syntax checks pass.
The new runtime tests did not run on the interrupted worker; repository CI is the next validation step.
No exact model–dataset cell has been registered, no biological model evaluation has run, and no HF dataset or model has been uploaded.
The final-checkpoint-first evaluation policy remains in force.

### 2026-09-09 23:54 UTC — RAG-550-005: CI validation of the combined scorer

Timestamp correction: the preceding entry was committed at 23:49:38 UTC; its 23:55 heading was entered incorrectly.
Repository CI ran the evals_v2 locked suite for PR #554: 416 passed, 5 skipped, and one workflow-fixture test failed.
All four new numerical, padding, pooling, and joint-versus-separate routing tests passed through the real HF prediction loop on a tiny model.
The workflow fixture failed because it replaced the model registry without clearing existing optional consumers.
Independent review also found eager RAG benchmark lookup would break reference-only subset configurations.
The follow-up clears fixture-only consumer lists and resolves RAG benchmark revisions only when a RAG job is constructed, with a new legacy subset dry-run test.
Those changes are published in PR #554 for CI and independent follow-up review.
No runtime GPU/bf16/compilation measurement has been performed for the new backend.

### 2026-09-09 23:58 UTC — RAG-550-006: Both framework PRs ready; execution approval pending

Combined-inference PR #554 passed its final CI suite at `221f799432bd5f010599338a228457a3d4506da4`: 418 passed, 5 skipped, 51.18 seconds for `uv run --locked pytest -m "not slow"`.
The combined-target and reference-only subset dry-runs both pass, and independent follow-up review reports no remaining findings.
PRs #552 and #554 are ready for human review; neither was merged.
The next execution steps are to resume the immutable 6b1593c data producer from S3, assemble and publish the five sequence-only datasets, submit the synthetic TPU pilot from the pinned training project, validate native resume/export/cadence and throughput, then launch production with verified HF revisions.
The prepared pilot submission script expects snapshot 4c000351 and must run from a worktree at that revision (the research branch has advanced with evaluation code and records).
The replacement worker must first dry-run the original producer and confirm cached artifacts are consumed; fresh checkout mtimes must not cause unnecessary recomputation of immutable pinned outputs.
Before biological VEP, register the exact checkpoint and harness hash in a small evals_v2 PR and perform GPU precision/compilation and throughput pilot checks.
The free-TPU/W&B approval and the $30 paid CPU/GPU budget approval remain pending in the task.
No blocked compute action was retried after the direct approval requests, and no training or biological evaluation has run.

### 2026-09-10 00:11 UTC — RAG-550-007: Direct authorization and execution recovery

The user directly approved the pending combined request in this task on September 10: free Iris TPU training, W&B logging with the existing credential, and a $30 cumulative CPU/GPU budget.
This covers the synthetic pilot and full run, CPU coordination, and replacement/retries within the cap.
The earlier issue-only approval blocker is resolved; no further approval is needed for these actions within their limits.
Scientific scope remains the five-region recipe, whole-chr18 holdout, and final-checkpoint evaluation first.
PR merges remain unauthorized.

The old Spot worker terminated after 99.6 minutes; estimated compute plus root-disk cost was approximately $0.32.
Launched on-demand recovery instance `i-00f345f512d4a8956` in us-east-2 at 00:05:31 UTC, r6i.2xlarge with 200 GiB gp3 and automatic termination after four hours.
Its four-hour estimate is about $2.2; all attempts remain within the cumulative $30 cap.
Recovery restores producer commit `6b1593c274a886d20f5c0ddf3712916d446f5fed`, identical pinned source assets, and the original durable S3 namespace, with a dry-run before resuming incomplete work.

Submitted the pilot from `4c00035108d0fcd64790a51aea3fee88a0108323` at 00:06:03 UTC: [Iris job](https://iris.oa.dev/#/job/%2Fgonzalo%2Fdna-exp550-rag46m-pilot-mb5-20260909).
The coordinator installed the locked runtime and constructed the graph, then submission of its TPU worker failed because the request specified 200 GiB local disk but the v6e-8 us-east1 pool provides 100 GiB per VM.
No training updates ran.
The next retry requests 80 GiB scratch; tokenized outputs and checkpoints use GCS.
The model, data, optimizer, and effective batch are unchanged.

The user also requested durable guidance to prevent repeated permission questions.
[Issue #555](https://github.com/Open-Athena/marin-dna/issues/555) and [PR #556](https://github.com/Open-Athena/marin-dna/pull/556) add initial approval consolidation and a persistent authorization record to AGENTS.md and the research/launch skills.
Independent review is in progress; the guidance explicitly cannot override platform approval review.

### 2026-09-10 00:25 UTC — RAG-550-008: Cached recovery and explicit worker setup

Approval-persistence guidance passed all CI jobs and independent review with no findings in PR #556, which is ready for human review and remains unmerged.
The same guidance is applied to this active branch at `c8b73e0f`.

The recovery dry-run initially scheduled 361 jobs because the freshly checked-out manifest timestamps were newer than the durable provenance outputs.
Verified the three manifest files byte-for-byte against producer commit `6b1593c274a886d20f5c0ddf3712916d446f5fed`, then restored their timestamps to that commit time.
The refined dry-run reused 206 completed jobs and scheduled 159 missing jobs, including 18 liftOver jobs and 20 species sequence outputs.
It preserves the original source hashes and S3 output namespace.
The producer resumed at 00:20 UTC using `--cores 4 --resources mem_mb=56000 --rerun-triggers code params input --set-resources rag_chain_liftover:mem_mb=18000`.
Completed mammal liftOver benchmarks peak at 12,801 MiB RSS, with all remaining compressed chains smaller than the measured mouse chain; 18 GB reservations allow at most three concurrent queries with headroom on the 64 GiB worker.
The elephant query actually completed and uploaded its mapped/accepted outputs just before the old worker terminated; it is reused.

The 80 GiB pilot retry reached an allocated TPU worker, but the shared image's uv 0.10.3 failed the project's required uv 0.11.31 check during child dependency setup.
The coordinator's installed uv was not inherited by the worker.
Snapshot `54b6f936467bbc657ae88753a6f24b768bd3363d` submits the child through Fray with an explicit pinned-uv bootstrap followed by the standard Iris TPU dependency setup.
All 14 locked training tests passed on the recovery worker in 24.04 seconds, including child environment propagation through the actual Fray-to-Iris converter.
A prior preflight helper used a nonexistent `--plan` flag; rerunning with the documented default plan mode passed.

The next pilot submission uses version `2026.09.10.uv` and job name `dna-exp550-rag46m-pilot-mb5-20260910-uv`.
No training updates or biological VEP have completed yet.

### 2026-09-10 00:44 UTC — RAG-550-009: Resume verification ready; publication review gate

Snapshot `62cdd734` adds `--resume-pilot-from` for an intermediate native synthetic-pilot checkpoint, with required full-state loading and a separate run/output identity.
The flag rejects production runs and final/non-native checkpoints.
All 15 locked training tests passed on the recovery worker in 17.58 seconds.
The current active pilot remains the tested `54b6f936` snapshot at [Iris job](https://iris.oa.dev/#/job/%2Fgonzalo%2Fdna-exp550-rag46m-pilot-mb5-20260910-uv1), version `2026.09.10.1`.
The attempted `2026.09.10.uv` version was rejected by the CLI before worker dispatch; the valid numeric calendar version passed a separate exact-command plan check before submission.

The pilot's child is pending because the controller reports `tier_blocked: 1 matching group(s) blocked by quota-pool tier monotonicity`.
At 00:35 UTC, the lower-tier v6e-4 us-east1 group had degraded health, blocking new v6e-8 allocations despite the latter group's available status.
The checked v6e-4/v6e-8 pools in other regions also had degraded capacity or occupied ready slices.
No cluster policy or budget was modified.
A task-log follower is active; no training update has run.

The existing Hugging Face credential identifies the user account as an administrator of `marin-dna`.
Authenticated metadata checks returned 404 for all five intended dataset destinations, so none would overwrite an existing repository.
Automatic approval review nevertheless rejected queuing their future public upload: it classified the operation as sensitive genomic-data egress without specific user authorization for the payload and destination.
The rejected command did not create or run its publication helper and did not transfer the credential.

The safer preparation-only continuation is queued on the existing CPU worker, gated on producer exit status zero and a successful Snakemake dry-run.
It runs `rag_all_publication_files` from publisher snapshot `54b6f936`, writes the schema-constrained sequence-only shards and release manifests under the workflow owner, and performs no Hugging Face upload or credential transfer.
After the exact files exist, inspect their public-reference provenance and manifests before reconsidering the rejected action; ask the user only if the remaining authority gap cannot be resolved with that evidence.

### 2026-09-10 00:59 UTC — RAG-550-010: Explicit public Hub authorization and live TPU preflight

The user explicitly stated in this task: “btw, I approve HF upload, if it wasn't clear”.
This authorizes public publication of the five prepared `marin-dna/rag-five-regions-v1-{cds,tss_utr5,utr3,ncrna,enhancer}` datasets using the existing Hub credential.
The prior automatic-review authority gap is resolved by this direct task instruction.
Preparation and the queued payload audit still precede upload; no additional user approval is needed within that exact scope.

All 16 training tests passed at `e98980ba08c35b805024fe0718397e429646a161` on the authorized CPU worker in 21.15 seconds.
The exact us-east5 plan passed, and the replacement [Iris pilot](https://iris.oa.dev/#/job/%2Fgonzalo%2Fdna-exp550-rag46m-pilot-mb5-20260910-east5) launched at 00:53 UTC with version `2026.09.10.2`.
Cancelled only the superseded pending us-east1 pilot and its descendants.
The replacement reached its TPU worker, installed uv 0.11.31, built all ten synthetic caches, and authenticated W&B.
Before its first training update, the adapter rejected a 10,176-token example against the required 10,240-token axis.
Investigation found that upstream jagged-array batch reads special-case only the last occurrence of row zero, potentially truncating earlier repeated reads by the shard row count (64 in this pilot).
A reproduction and a local adapter fix are in progress.
The CLI-defined optimizer class also failed configuration serialization under `__main__`; it will move to an importable module before retrying.

The upload command was accepted after the user's explicit authorization and is queued on the recovery worker (PID 14373).
It waits for successful release preparation and the complete payload audit (PID 13203), then dry-runs and publishes the five exact destinations serially.
The existing HF token is supplied only through encrypted SSH stdin into process memory/environment; no credential file is copied or written.

The cache bug is confirmed with three synthetic rows: `get_batch([0,1,0])` returned lengths `[10237,10240,10240]`.
[Issue #557](https://github.com/Open-Athena/marin-dna/issues/557) records the reusable upstream defect.
The experiment-local adapter reads unique component indices before shuffle/mixing and restores every sampled occurrence in order.
All 18 locked training tests passed in 16.79 seconds, including real TreeCache-to-NamedLmDataset repeated-read parity and optimizer configuration serialization after cloudpickle round-trip.

### 2026-09-10 01:24 UTC — RAG-550-011: TPU pilot and exact native-resume parity passed

The c0585d0e synthetic pilot completed 20 updates on eight TPU v6e chips with microbatch 5 and 200 documents per update.
The last steady update took 1.552648577 seconds, or 1,319,036 allocated tokens/second.
The 100,000-update compute-only projection is 43.13 hours; cache construction, checkpoint/validation overhead, and interruptions add time.
This exceeds one night, and the user was informed; the agreed 100,000-update scope is unchanged.

Native and HF milestones exist at completed updates 5, 10, 15, and 20.
Synthetic validation loss at those milestones was 2.286274, 2.222548, 2.121411, and 1.989978.
These values validate execution and are not biological results.
The full model export is 183,596,040 bytes and retains the 640/2560/7-layer/5-head architecture, vocabulary 8, and maximum context 10,240.

The separate native-resume job loaded update 10 and completed update 20.
Its final model has the same GCS MD5 and CRC32C as the uninterrupted pilot (`IpYBgDMQ2sZlO8knwEsuZQ==`, `Cyx4BA==`), and both update-15 and update-20 validation losses match exactly.
The resumed native metadata records step 20; full optimizer state was required on load.
The TPU backend took about three minutes to initialize on the resume worker, then proceeded successfully without a retry.

Small machine-readable evidence is in `.agents/artifacts/issue-550/pilot/`.
Independent review found no material issues in c0585d0e.
The training preflight is complete; production awaits the prepared datasets and verified immutable public revisions.

Release preparation is queued with the documented `--keep-storage-local-copies` option so the downstream audit can inspect its exact files.
The current preparation PID is 25210; audit PID 13203 and explicitly authorized upload PID 14373 remain gated in sequence.
At 01:22 UTC the biological producer had completed 153 of its 159 recovery jobs.

The historical 46M final-checkpoint score files were audited against the current pinned development datasets on the CPU worker.
They match all canonical variant and metric-membership tuples exactly: Mendelian 16,140 rows, Complex Traits 11,630, and SGE 23,853.
The preserved zero-shot macro AUPRC values are 0.395455, 0.184042, and 0.476728; frozen-probe macro AUPRC values are 0.408816, 0.297643, and 0.418497.
The compact artifact `.agents/artifacts/issue-550/baseline/exp402-exact-development-cohort-audit.json` records source URIs, canonical revisions, metric hashes, support counts, and standard errors.
No new baseline inference, probe fitting, or held-out evaluation was run.

### 2026-09-10 02:28 UTC — RAG-550-012: Public-release assembly and GPU precision validation

The last horse liftOver query finished after about 58 minutes at full CPU.
All 39 non-human chain queries are complete, and the producer reached `rag_documents` at 157 of 159 recovery jobs.
Final assembly is active with about 1 GiB RSS; release preparation, payload audit, authorized public upload, and immutable training-manifest verification remain queued sequentially.
The biological producer stays pinned to `6b1593c274a886d20f5c0ddf3712916d446f5fed` and the publisher to `54b6f936467bbc657ae88753a6f24b768bd3363d`.

Launched one Spot A10G worker (`i-0ca9594bb60eaf836`, us-east-2c) at 01:46:26 UTC for a one-hour synthetic inference check within the existing $30 cumulative CPU/GPU cap.
The initial us-east-2a request returned insufficient capacity without creating a worker.
SkyPilot had no connected API server, so the existing authorized direct-EC2 launch path was used.
The pinned runtime smoke check passed: DLAMI `ami-0324f0ad73bdcd087`, driver 595.71.05, PyTorch 2.13.0/CUDA 13.0, Transformers 4.57.6, NVIDIA A10G.
Only synthetic DNA documents and the completed synthetic TPU pilot's update-20 export were used.
The model weights were streamed over SSH without copying GCP credentials; their MD5 and 183,596,040-byte size matched the original export.
The shared-VM transfer held the nonblocking heavy-work lock, took 13.59 seconds, and peaked at 95,696 KiB RSS.

The GPU check discovered a Transformers 5 tokenizer export incompatibility: `extra_special_tokens` is a list, but the pinned Transformers 4 consumer expects a named-token dictionary.
[Issue #558](https://github.com/Open-Athena/marin-dna/issues/558) and [PR #559](https://github.com/Open-Athena/marin-dna/pull/559) add an in-memory translation that preserves tokenizer bytes, token IDs, and special-token behavior.
The regression was reproduced before the fix; all 424 evaluation tests passed with CUDA disabled, the workflow dry-run passed, and CI plus independent review passed.
An initial GPU-visible test run exposed five existing CPU-fixture device mismatches; the new tokenizer tests passed there too.
PR #559 is ready and unmerged; the fix is applied to the experiment branch at `614e18e105849736ee5c278ca5d9fd38aa9d191d`.

The existing Trainer wrapper also flattens bare tensor outputs at batch size one; eight RAG records became a length-10,256 vector instead of an 8-by-1,282 matrix.
[Issue #560](https://github.com/Open-Athena/marin-dna/issues/560) tracks that separate defect.
The synthetic fp32 reference and all intended production inference batches use at least two records.

Full bfloat16 parameter casting failed the predeclared numerical gate: reverse-strand LLR differed from eager fp32 by up to 0.19475, exceeding the maintained 0.15 tolerance.
Bfloat16 autocast with fp32 weights also failed the unchanged gates.
Both were rejected before biological benchmark inference.
The fp32 fallback exposed a second framework issue: Accelerate re-enables TF32 during Trainer construction, even after `tf32=False` was explicitly requested.
[Issue #561](https://github.com/Open-Athena/marin-dna/issues/561) and draft [PR #562](https://github.com/Open-Athena/marin-dna/pull/562) restore the explicit setting after construction and support a documented workflow fp32 fallback.
Independent review caught unconditional new provenance parameters that would invalidate existing default results; snapshot `9b0c69232683a65f70d88481cf6a0510e97beb85` adds those parameters only for explicit precision overrides.
Updated CI, dry-run, and independent review are pending.

With TF32 actually disabled, compiled fp32 matches eager fp32: maximum LLR difference 2.84e-5, JSD difference 4.81e-9, and pooled-embedding difference 9.54e-7.
Warmed batch-two inference reaches 0.89759 variants/second, including REF/ALT, both strands, tokenization, data loading, and prediction collection, at 7.32 GB peak GPU allocation.
Batch four reaches 0.89245 variants/second at 14.66 GB, while batch eight exceeds A10G memory.
Batch two is selected; this projects about 16 hours for all 51,623 development variants before probe fitting and setup.
The final combined-versus-separate synthetic timing comparison is still running.
Small failed-gate and passing-fallback reports are retained under `.agents/artifacts/issue-550/gpu/`; no biological scores have been computed.

## RAG-550-013 — 2026-09-10 02:49 UTC — Precision validation complete

The final synthetic A10G pilot completed successfully with strict fp32, compilation, batch 2, and explicit post-Trainer TF32 disabling.
The maximum compiled-versus-eager LLR difference was 2.83718e-5; pooled-embedding difference was 9.53674e-7.
Measured throughput was 0.898592 variants/second, projecting 15.958 hours for all 51,623 development variants before setup and frozen probes.
The warmed 96-row comparison took 108.871 seconds with three model loads and 107.309 seconds with one, saving 1.562 seconds.
Full bfloat16, bfloat16 autocast, and TF32 compiler candidates failed their predefined parity gates; the JSON evidence retains those failures.
The GPU Spot instance was reclaimed at 02:43:37 UTC after about 57.2 minutes, during a subsequent CPU-only test rerun; completed pilot reports had already been retrieved.

PR #562 at 92bc85b30944a44c5959d4caf8a320eb2b7a9cb6 passed 426 tests with five skips, the default 806-job workflow dry-run, and quality checks.
Independent review found no remaining material issues after restoring all three legacy rule bodies and their default parameter identities.
A Snakemake 9.25.1 completed-output fixture retained the prior output under the new default and scheduled a params-triggered rerun only for the documented fp32 fallback.
The reusable precision fix is ready for review and has been applied to the permanent experiment branch; no PR has been merged.
The experiment's new combined RAG rule now records and forwards the same precision setting.

At 02:47 UTC, the biological producer remained at 157/159 recovery jobs, ingesting Tolypeutes_matacus into the final document store.
HF payload preparation, audit, explicitly authorized upload, and immutable input-manifest checks remain queued in order.
An optional EBS IOPS increase was rejected by automatic approval review as additional disk spend outside the clearly approved CPU/GPU scope; no disk change occurred and processing continues on the original disk.
The explicit HF approval persists for the five registered sequence-only datasets.

### 2026-09-10 02:55 UTC — Reporting and quality follow-up

Repository-wide pre-commit checks passed after formatting the one-off evidence scripts and adding terminal newlines to JSON evidence copies.
The immutable original evidence remains available in its earlier snapshot.
Automatic approval review rejected optional copying of the already-published synthetic GPU report to W&B, including after supplying the earlier direct W&B authorization record.
No W&B report upload occurred; the pinned GitHub artifact remains the numerical reporting source.
This optional reporting step does not block dataset assembly or the approved HF publication.

### 2026-09-10 03:03 UTC — Integrated validation and evaluation runtime

The integrated permanent branch at eeb877f9 passed all five project test jobs and their configured dry-runs in GitHub Actions run 34431400058.
Independent review found no integration issue in combined RAG precision forwarding at 1e4e59db.
The experiment fp32 runtime overlay validates against the full existing inference policy and records its failed-candidate evidence.
Exact model/harness registration still awaits the final dataset artifact and production output identity; no biological inference is scheduled yet.
The assembly progress estimate was corrected: the uppercase mammalian species sort before the other vertebrates, so the two remaining jobs still contain substantial work.

### 2026-09-10 03:12 UTC — Platform approval blocker on worker extension

Reusable SQLite write-amplification follow-up is tracked in [issue #563](https://github.com/Open-Athena/marin-dna/issues/563).
At 03:09 UTC the original immutable producer was ingesting anaPla1, the twentieth of 39 sorted non-human species; recovery job count remained 157/159.
The existing CPU worker's shutdown deadline remains approximately 04:05 UTC.
Automatic approval review rejected extending it to roughly 06:10 UTC, including after checking the original task permission questions and the user's direct affirmative response.
No shutdown timer or disk-performance setting was changed.
A consolidated user confirmation is pending for the concrete worker extension, cumulative $30 CPU/GPU scope including retries/extensions, free TPU training, and synthetic/training W&B logging to gonzalobenegas/marin with the existing credential.
The approval question is required by that platform rejection; the earlier task authorization remains recorded and the repository guidance itself does not request a new approval.

The separately and explicitly approved HF upload remains queued behind producer completion, payload preparation, and audit.
The audit's existing bounded pass now also records exact per-split species histograms and padding totals, with expected training exposure explicitly distinguished from observed sampled-token counts.
The updated audit is /opt/issue550/audit-release.py on the worker; its local copy is /tmp/issue550-audit-release.py.
The registration worktree /tmp/issue550-registration exists on codex/issue-550-register-final-rag based on PR #554, with no registration edits yet.
Training still awaits actual immutable public dataset revisions, and no biological VEP inference has run.

### 2026-09-10 13:33 UTC — Unattended access guidance and durable recovery state

At 13:11 UTC, the previous CPU worker was unreachable after its approximately 04:05 UTC shutdown deadline; the task's prior CPU and GPU instance IDs no longer appeared in EC2 describe results.
The immutable biological S3 prefix retained 399 objects: five anchor artifacts, 351 chain artifacts, 40 species sequence Parquets, and three source artifacts.
No final RAG datasets or combined evaluation harness were present, the publication prefix was empty, and the anonymous Hugging Face dataset listing contained no five-region releases.
The queued continuation on the expired worker did not complete.
Production training and biological VEP have not started.
The validated TPU pilot, resume check, and synthetic GPU precision evidence remain durable.

The existing task authority remains free Iris TPU training and a cumulative $30 CPU/GPU budget, including failed attempts and recovery, with explicit approval for the five public HF datasets.
The original direct approval is recorded in the September 10 00:11 entry; its private message references remain in task handoff context.
The expired-worker extension question is obsolete; prior consent does not need to be repeated to resume within the authorized scope.
Reconcile spending and outstanding commitments before selecting a replacement worker.
The current session still has restricted workspace execution and automatic approval review; repository edits do not change those platform settings.

[PR #556](https://github.com/Open-Athena/marin-dna/pull/556) now adds effective-permission preflight, supported persistent Full Access setup, approval references in handoffs, and worker lifetime/recovery planning for future sessions.
Its published commit is 2299ad04ec6a2230b31ecf942b25859616a1c522, applied here at 31606d8d.
Repository quality checks, four skill validators, relative links, and the TOML example pass.
Independent review found no issues and checked resumed-budget, execution-mismatch, denied-upload, and missing-budget scenarios; it did not exercise actual permission changes or external actions.

Resume data construction from the 40 saved sequence Parquets and other immutable cached inputs, then audit and publish the five sequence-only datasets before production training.
Retain the producer identity and additive pipeline contracts; any implementation change must receive a distinct producing identity.
Use the latest checkpoint first for development VEP after training and spend on earlier checkpoints only if the cumulative budget permits.

### 2026-09-10 14:12 UTC — Final assembly recovery for production launch

The user directly requested launching the production training run in the resumed task.
Existing authorization covers the required data recovery, five public HF datasets, free Iris training, W&B logging, and the cumulative $30 CPU/GPU limit.
Current AWS public pricing confirms r6i.4xlarge at $1.008/hour and g5.xlarge at $1.006/hour in us-east-2.
The conservative plan reserves $4 for prior attempts, $4.182 for a four-hour recovery worker and its disk, $18.108 for 18 GPU hours, and $3.71 for remaining overhead/recovery.
The budget remains cumulative; these are upper allowances and reservations rather than an AWS billing settlement.

CPU worker i-03039e0a95792b4e0 launched at 14:04:10 UTC with 128 GiB RAM, a 200-GiB delete-on-termination gp3 disk, and automatic shutdown at approximately 18:04 UTC.
The producer and publisher remain pinned to 6b1593c274a886d20f5c0ddf3712916d446f5fed and 54b6f936467bbc657ae88753a6f24b768bd3363d.
Final assembly uses a 96-GiB tmpfs for its temporary SQLite database and local dataset outputs; successful outputs are uploaded through the existing S3 workflow.
This changes the execution storage location without changing the biological producer code or artifact identity.
The dry-run schedules only rag_documents and rag_all_documents, reusing all 363 completed upstream jobs.
Initial setup needed Conda installed before the dry-run could inspect the workflow; the corrected bootstrap includes that dependency.

The recovery driver started at 14:08 UTC and sequences assembly, public release preparation, bounded payload validation, approved HF publication, and verification of immutable input receipts.
The HF token is provided only through the remote process environment and is absent from committed scripts and logs.
Initial process inspection showed about 950 MiB RSS, 712 MB of SQLite write calls, and only 70 KB of physical writes, confirming that the previous disk bottleneck is avoided.
The worker has over 120 GiB available at the start of ingestion.

The prepared production submission retains the tested code and lockfile from c0585d0e117075026b4384d552d59b52009d257e.
It requires the exact five verified public revisions committed inside the experiment project, selects eight free preemptible v6e chips in us-east5, and preserves 100,000 updates, 200 documents per update, and the AdamH scaling heuristic.
Training has not been submitted while the final datasets are still being assembled.

### 2026-09-10 14:53 UTC — Public training inputs verified

Final biological assembly completed and uploaded all ten train/validation Parquets, the split summary, and the combined development harness by 14:38 UTC.
The aggregate target initially lacked a local copy of the already-durable producer manifest.
Staging that exact S3 manifest restored the aggregate; the subsequent dry-run reported no work, so the expensive assembly was not repeated.
The immutable producer identity and configuration remain unchanged.
The completed data contain 1,110,006 training documents and 2,000 validation documents, with all chr18 excluded from training.
The 51,623-row development harness has SHA-256 6631d35f9ae0afc754c623a2e3960682e8a834a28001906279745882f99cb73b.

All five release payloads passed the bounded schema, sequence-geometry, row-count, size, and checksum audit, then published under the existing explicit HF authorization.
The publisher verified public access without credentials and recorded immutable revisions and release hashes.
The production input manifest is experiments/exp550_rag_five_regions/config/verified-public-datasets.json; assembly and release evidence are under .agents/artifacts/issue-550/publication/.
The biological producer, publisher, and tested training code retain their earlier immutable identities.
This snapshot records the verified public inputs before production submission.

The user also identified an unrelated idle #517 EC2 worker during this launch.
Postmortem #564 tracks its approximately 400 idle hours and the cleanup investigation.
The user explicitly authorized termination and reported that no local output needed retention; AWS confirms i-0b417bcfc77ecc94e terminated at 14:50:46 UTC and its root volume is absent.
That historical worker's cost is separate from the approved #550 budget.

### 2026-09-10 15:09 UTC — Production submitted and CPU recovery worker reclaimed

Iris accepted /gonzalo/dna-exp550-rag46m-five-regions-v1-20260910 at 14:54:10.128 UTC from commit 41eaed8cb31d4a53a201497fd9ba1b5ac1e8e187.
The coordinator successfully installed the locked environment and dispatched train-worker at 14:55:23.566 UTC.
It retains the tested v6e-8 configuration, us-east5 placement, 100,000 updates, 200 documents per update, and version 2026.09.10.5.
The checkpoint root is gs://marin-us-east5/MarinDNA/exp550_rag_five_regions/checkpoints/dna-exp550-rag46m-five-regions-v1/2026.09.10.5.
The worker initially remained pending because quota-pool tier monotonicity blocked matching capacity.
At 15:06 UTC the controller reported 13 matching v6e-8 slices booting in us-east5; the submitted job was retained without a duplicate launch or region change.
Optimizer progress is not yet verified.

The CPU recovery and publication driver exited successfully at 14:49:27 UTC.
All biological outputs and publication receipts are stored in their existing S3 workflow namespaces, public HF revisions are pinned in the committed manifest, and 464,519 bytes of recovery logs and status markers are retained locally in /tmp/issue550-recovery-retained-logs.
Automatic approval review rejected an optional S3 recovery-log archive because its sensitive-content and destination approval checks were unresolved; no archive was uploaded.
Automatic approval review also initially rejected cleanup of the CPU worker under an incorrect active-NVMe-state premise.
New read-only evidence established successful completion, durable data, retained logs, and AWS InstanceStorageSupported=false for r6i.4xlarge; review then accepted normal cleanup under the existing task scope.
AWS confirms i-03039e0a95792b4e0 terminated at 15:03:40 UTC after 59.5 minutes, costing $0.9996 in compute before disk charges.
No #550 EC2 worker remains running.

The public release audit estimates 126,343,840,095.63 unpadded positions across training under four million draws per region, compared with 204.8 billion allocated positions.
Those are expected exposures from the exact dataset histograms; observed training exposure remains to be measured.
Postmortem #564 is delegated to a separate investigation agent at the user's request while this task monitors training startup.

### 2026-09-10 15:33 UTC — Production worker allocated in us-east1

The original us-east5 request and its child were canceled at 15:12:54 UTC after the matching pool degraded without providing a worker.
The us-east1 replacement was accepted at 15:13:06.954 UTC from 872c5e68f11cc9067bb0bce741782e187594e6e8, using checkpoint version 2026.09.10.6.
Its child initially lacked the requested 128 GiB RAM on the available v6e-8 host.
The reviewed resource-only change bd948cf275c4bf23d733df34401c40707b8c55a3 reduces host RAM to 48 GiB while preserving 16 CPU cores, 80 GiB disk, eight v6e chips, and the entire scientific recipe.
Independent review found no blocking issue; actual production memory usage still requires observation because the expanded token cache is approximately 91 GB and must stream to GCS.
The two tokenization workers process regions sequentially with bounded batches, and the tokenization subprocess exits before model training starts.

The version-6 child acquired capacity and began CDS tokenization while the RAM revision was being prepared.
It was canceled at 15:26 UTC, after approximately 52,000 initial CDS records had been processed and before any optimizer steps.
Both parent and child were confirmed killed before replacement submission.
The first replacement command failed local CLI validation because a 4-GiB coordinator requires --enable-extra-resources; adding that documented flag allowed submission without another resource allocation or permission request.

Iris accepted /gonzalo/dna-exp550-rag46m-five-regions-v1-20260910-east1-ram48 at 15:27:42.302 UTC from bd948cf275c4bf23d733df34401c40707b8c55a3.
The 4-GiB CPU coordinator ran JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --locked pytest before dispatch: all 18 tests passed in 52.81 seconds, with one upstream deprecation warning.
The child was submitted at 15:29:53.819 UTC and started at 15:30:13.117 UTC on marin-tpu-v6e-preemptible-8-us-east1-d-20260910-1516-7f7b3e67-worker-0.
Its controller resource receipt confirms v6e-8, count 8, 48 GiB RAM, 16 CPU cores, and 80 GiB disk.
Production tokenization began at 15:30:39 UTC and is writing CDS caches to gs://marin-us-east1/MarinDNA/exp550_rag_five_regions/checkpoints/dna-exp550-rag46m-five-regions-v1/2026.09.10.7/tokenized.
Both local tokenization workers are alive and document counts are advancing; optimizer steps and W&B training metrics are not yet available.
Retain this active job while it builds all five caches, then verify compilation, finite advancing optimizer steps, observed memory, and checkpoint creation.

The existing free-Iris authority and cumulative $30 CPU/GPU cap remain unchanged.
No paid EC2 worker is active for this task, and no biological VEP inference has run.
The user requested that all further postmortem #564 findings update its body directly, without new comments.

### 2026-09-10 16:07 UTC — Worker failures and four-chip pilot

The version-7 production child lost three workers between 15:30 and 15:49 UTC.
Each controller receipt reports worker reconcile failure threshold exceeded; none reports an optimizer step or an out-of-memory failure.
The last task status at 16:05 UTC is pending retry 3, although the aggregate job list labels the child running.
Tokenization completed at least one CDS shard before the last failure, so surviving GCS cache outputs can be reused.
A 15:42 resource snapshot measured approximately 4.7 GiB current and 4.9 GiB peak task memory.
At 15:50 the controller showed no ready or booting v6e-8 slices, while v6e-4 capacity remained available.
Do not confuse aggregate job state with an allocated, advancing training process.

Snapshot 57c4124e24b313215f9d7662cf3eb9faac3f9200 adds an explicit v6e-4 option and a distinct synthetic pilot identity.
The model, data, optimizer, schedule, and effective batch of 200 remain unchanged; microbatch 5 uses ten accumulation steps across four chips.
Independent review found no blocking issues, and the 19-test suite runs remotely before pilot dispatch.
The shared-node heavy-work lock prevented local packaging, so the existing Iris CPU coordinator downloaded and packaged the exact pushed snapshot.
Its initial CLI submission selected the public IAP URL and returned Forbidden; using the runtime-provided controller address, as the Iris worker client normally does, succeeded without credential extraction or another user permission request.
Iris accepted the pilot at 16:06:03 UTC as /gonzalo/dna-exp550-rag46m-five-regions-v1-20260910-east1-ram48/dna-exp550-rag46m-pilot-v6e4-20260910.
Its artifacts use run ID dna-exp550-rag46m-five-regions-v1-pilot-mb5-v6e-4 and version 2026.09.10.1 under the us-east1 checkpoint root.
Keep the original parent alive while this nested pilot runs; canceling the parent also cancels its descendants.
The pilot must verify four accelerators, finite advancing updates, checkpoint/export files, and throughput before production moves to four chips.

Draft PR #565 registers the final checkpoint and three development probe cells on top of #554.
Commit b08391e9f6965a0ed6eae64de814bd8d8d631d91 also fixes the credential-free CI dry-run by substituting temporary local harness inputs while preserving the full registry and production checksum/storage behavior.
All CI checks pass, and independent review found no blocking issues.
The registration currently points to version 2026.09.10.7; update it before evaluation if a replacement uses another version.
No biological VEP inference has run, and no additional paid worker has been launched.

The user approved closing postmortem #564; its body records the completed investigation, evidence limits, cost estimate, and unimplemented recommendations.
GitHub confirms closure at 15:38:01 UTC.

### 2026-09-10 16:28 UTC — Four-chip pilot passed; production replaced

The us-east1 four-chip pilot passed all 19 locked tests in 44.47 seconds but never acquired a TPU.
Its parent and child were canceled at 16:13 UTC after the regional pool lost all ready workers and reported quota-pool tier blocking.
Inspection of the pinned autoscaler code showed that failed allocation tiers cause that restriction; the pending eight-chip request was not itself the cause.
Submitted the same tested snapshot in us-east5 at 16:14:01.994 UTC with Iris's normal interactive priority and a separate root job, /gonzalo/dna-exp550-rag46m-pilot-v6e4-20260910-east5.
The child began setup at 16:18:26 UTC, completed synthetic caches, and ran all 20 training updates on four TPU v6 lite devices.
Native and HF milestones at completed updates 5, 10, 15, and 20 are present in GCS, and the final W&B run state is finished.
Final training loss is 1.5804566144943235; aggregate validation loss decreases from 2.2870240211486816 at update 5 to 1.9907548427581787 at update 20.
The last steady update took 2.705842138 seconds, projecting 75.1623 compute hours for 100,000 updates before tokenization, validation, checkpoint overhead, and interruptions.
The verification receipt is .agents/artifacts/issue-550/pilot/completed-v6e4-pilot.json.
W&B global_step remains zero-based and ends at 19; the final evaluation uses _step 20 and native metadata confirms 20 completed updates.
The first verification helper incorrectly expected global_step 20; correcting that helper to the documented convention verified all milestones without changing or rerunning training.

Production was accepted at 16:25:14 UTC as /gonzalo/dna-exp550-rag46m-five-regions-v1-20260910-east5-v6e4, from the exact pilot snapshot 57c4124e24b313215f9d7662cf3eb9faac3f9200.
It uses v6e-4, microbatch 5, batch 200, 100,000 updates, the five verified HF revisions, and version 2026.09.10.8 under gs://marin-us-east5/MarinDNA/exp550_rag_five_regions/checkpoints/dna-exp550-rag46m-five-regions-v1.
The coordinator uses the normal interactive band, two coordinator retries, and a seven-day job timeout; native rolling checkpoints and permanent 10,000-update checkpoints retain the tested recovery behavior.
No TPU or biological optimizer progress is claimed from coordinator submission alone.

Automatic approval review initially rejected cancellation of the old version-7 parent because it inferred active training and missing authority for that job.
Fresh task and descendant checks at 16:26 UTC proved that its sole live child was pending without a worker after three failures, its other descendants were killed, and the replacement was an independent root job.
The pilot receipt was retained locally before cleanup.
Review accepted cancellation under the existing autonomous execution/recovery scope after that evidence was supplied; no additional user permission was requested.
Iris confirmed the old parent canceled at 16:27 UTC; its GCS artifacts were not deleted.
The exact evaluation registration must now use the us-east5 version-8 export path.

### 2026-09-10 16:46 UTC — Production tokenization and final-evaluation preparation

The four-chip production worker acquired its TPU at 16:32:40.978 UTC and began tokenization at approximately 16:33:42 UTC.
At 16:45, get-task-status confirmed attempt 0 running on marin-tpu-v6e-preemptible-4-us-east5-b-20260910-1627-e8662a69-worker-0 with four v6e chips, 16 CPUs, 48 GiB RAM, and 80 GiB disk.
At 16:43, two of sixteen CDS training shards had completed and the next two were advancing.
Queue-full messages reflect writer backpressure; submitted-record counts alone do not prove durable shard completion.
Production has not yet reached optimizer updates.
The event-driven child log follower writes /tmp/issue550-production-v6e4-live.log; authoritative task status comes from `iris --cluster marin rpc controller get-task-status --task-id /gonzalo/dna-exp550-rag46m-five-regions-v1-20260910-east5-v6e4/train-worker/0`.

PR #565 now pins the exact us-east5 version-8 final export, passes all CI checks at ef4c34399f6c6bf1ecee8c618e05f65e27353559, and passed independent review.
It was marked ready without merging.
The permanent experiment branch incorporates the registration and credential-free CI overlay together with the existing combined backend, tokenizer compatibility, and strict-fp32 fixes.
Snapshot c5978e8fd2251dd7b36998afbd3e6c93d462adc8 adds the final-checkpoint synthetic parity recheck and exact checkpoint-download, inference, and probe commands to the experiment README.
The recheck retains the pilot's predeclared tolerances, verifies the requested URI against the registered model, records the actual consumer commit, and uses only synthetic sequences.
Its syntax check passes; execution with final weights remains pending their production.
GitHub workflow run 34503945833 validates the integrated consumer branch before final evaluation.
Normal authenticated `gcloud storage ls` verified access to the four-chip pilot's final HF export without copying or printing credentials.

The paid-budget reconciliation retains a $4 upper allowance for earlier CPU/GPU attempts and disks, $0.9996 for the completed recovery worker, and its original $0.15 disk allowance.
An 18-hour on-demand g5.xlarge compute reservation costs $18.108 at the verified $1.006/hour upper bound, leaving $6.7424 for further storage, setup, and permitted recovery within the $30 cap.
This is an allowance-based reconciliation, not an AWS billing settlement.
No new paid worker is active.
Next: verify completed production caches and finite optimizer progress, then measure actual training throughput and retain the final-checkpoint-first evaluation policy.

### 2026-09-10 16:51 UTC — Evaluation review complete; latent writer failure reproduced

GitHub workflow run 34503945833 passed all five project test jobs and the applicable offline Snakemake dry-runs for integrated consumer c5978e8fd2251dd7b36998afbd3e6c93d462adc8.
Independent review found that the checkpoint download commands retain files under Snakemake's S3 storage cache, whereas the initial synthetic command addressed results/checkpoints directly.
The README now declares /opt/issue550/storage as the local storage prefix and passes its exact s3/oa-bolinas/snakemake/analysis/evals_v2/results/checkpoints/<model> path to the recheck.
The same prefix is used for subsequent inference to reuse the cache.
Review of the corrected commands, fixed synthetic tolerances, integrated tokenizer/precision fixes, registry, and six final metric targets found no remaining actionable issues.
The correction changes documentation only; the integrated runtime's completed CI remains applicable.

The tokenizer source review verified the hashes and sizes of the five exact locked Marin wheels before reading their implementation.
Two writer queues can retain approximately 2.5 GiB of array payload at the current 128-document batch and 128-batch queue depth; queue-full warnings are compatible with ordinary backpressure.
Shard completion drains outstanding batches and renames the temporary GCS prefix, so progress counters may pause after the last submitted row while durable writes finish.
The review also reproduced a latent error-handling bug: ThreadedBatchWriter.close blocks while inserting a sentinel into a full queue after its background writer has failed, before checking the recorded exception.
The tiny standard-library reproducer completes in under a second and reports writer_failed=true, queued_batches=1, and close_blocked_in_sentinel_put=true.
Its source, immutable wheel metadata, and detailed review are retained under .agents/artifacts/issue-550/tokenizer.
Current completed shards and ongoing progress do not establish that this bug occurred in production or explain the earlier worker reconcile failures.
Track the upstream failure mode separately; do not modify an advancing production job based on this unobserved hypothesis.

### 2026-09-10 17:25 UTC — European eight-chip pilot and exact native resume passed

The latent writer-closure failure is tracked in #566 with its bounded reproducer; it remains unobserved in production.
The active four-chip production request confirms 100 automatic preemption retries, no hard-task retries, and no separate child timeout.
Its parent has two failure retries and a 604,800-second timeout.
The pinned trainer automatically searches its permanent and temporary checkpoint roots when restarted with the same output identity.

At 16:56, the US v6e-8 pools still had no ready or booting slices, while europe-west4 had five ready slices and 22 booting.
The active regional storage mapping uses marin-eu-west4, not marin-europe-west4; normal gcloud bucket metadata confirmed its EUROPE-WEST4 location.
Snapshot ddff6e1fa59dcc7dc386fe08c954cdec69224a2e adds only the explicit European region-to-bucket mapping and distinct European synthetic pilot identity, plus tests and documentation.
Model, dataset, optimizer, effective batch, tokenization, and schedule remain unchanged from the tested source.
Independent review found no actionable issues, and all 20 locked project tests passed remotely in 55.14 seconds before dispatch.

The independent pilot root /gonzalo/dna-exp550-rag46m-pilot-v6e8-20260910-europe was accepted at 17:02:06 UTC and completed at approximately 17:08:26 UTC.
Eight TPU devices, finite losses, and native/HF milestones 5, 10, 15, and 20 were verified through W&B and GCS.
The last steady update took 1.551344731 seconds, projecting 43.0929 compute hours for 100,000 updates before preprocessing, validation, checkpoint overhead, and interruptions.
Final training loss is 1.5800892114639282, exactly matching the earlier US eight-chip pilot.
Its receipt is .agents/artifacts/issue-550/pilot/completed-europe-pilot.json.

The separate native-resume root /gonzalo/dna-exp550-rag46m-pilot-v6e8-20260910-europe-resume was accepted at 17:10:25 UTC and completed at approximately 17:16:46 UTC.
It loaded the full native update-10 checkpoint and resumed through update 20 on eight chips.
The final HF model MD5, final training loss, and every region/aggregate validation loss at updates 15 and 20 match the uninterrupted run exactly.
The verification receipt is .agents/artifacts/issue-550/pilot/completed-europe-resume.json.

Both verified receipts gated submission of the independent production root /gonzalo/dna-exp550-rag46m-five-regions-v1-20260910-europe-v6e8 at 17:19:04 UTC.
It uses the exact ddff6e1f source, v6e-8, microbatch 5, batch 200, the same five public revisions, and 100,000 updates.
Artifacts use gs://marin-eu-west4/MarinDNA/exp550_rag_five_regions/checkpoints/dna-exp550-rag46m-five-regions-v1/2026.09.10.9.
The coordinator has a seven-day timeout and normal interactive priority; no paid resource was launched.
At 17:25, its first two CDS training shards were advancing, while the US four-chip fallback had completed 12 of 16 CDS shards.
Neither production run has reached optimizer updates.
Retain the US fallback until the European worker completes at least four full-data CDS shards without a worker failure, then release the fallback before either trainer starts using their common production W&B identity.
If the European replacement fails repeatedly during this overlap, retain the active four-chip fallback and cancel the replacement.
Do not run two production optimizers concurrently under the same W&B identity.

Final evaluation staging now explicitly runs the native checkpoint-download rule on a host with GCS and S3 access before paid GPU time.
The AWS GPU worker copies the resulting canonical S3 checkpoint into the correct Snakemake cache before the synthetic parity recheck, so it requires no copied GCP credential.
Independent review of these corrected cross-host commands found no actionable issues; the model registry remains on US version 8 until the production replacement is adopted.

## 2026-09-10 17:56 UTC — European production adopted and final staging prepared

The European worker completed four full CDS shards by 17:39 without a failure, meeting the recorded replacement gate.
The US four-chip fallback was canceled at 17:42:51 and its child reached TASK_STATE_KILLED at 17:43:07.818 with reason "Parent job terminated".
The European child remained TASK_STATE_RUNNING on attempt 0 at 17:49; eight of sixteen CDS shards were complete by 17:54.
Neither run had begun optimizer updates when the fallback was retired; its partial version-8 GCS caches remain intact.
No paid resource was launched.

PR #565 now registers the exact European version-9 final export, in commit 51fd75ad and experiment-branch cherry-pick abecf0d2.
All CI checks passed, including all five project test jobs and the credential-free evaluation dry-run.
Independent review of the published delta found no code issues; the PR description was corrected to the European source and the PR is ready and unmerged.
The final numerical-parity receipt now records training source ddff6e1fa59dcc7dc386fe08c954cdec69224a2e.
Issue #550's body was updated and read back, preserving the planned final-checkpoint-first evaluation and cumulative $30 cap.

The shared VM cannot run the full evaluation/Snakemake imports within its 500 MiB working-set limit.
A lightweight one-off stage-final-checkpoint.py therefore prepares the canonical S3 checkpoint using normal GCS and S3 credential providers, without copying credentials or importing the ML workflow.
It pins GCS object generations, verifies size and MD5, checks model geometry, validates all existing S3 objects before writes, uses conditional puts, rereads uploaded bytes, and creates the directory completion marker last.
It holds the nonblocking shared heavy-work lock, enforces the headroom and load thresholds, monitors resource pressure, and records start/end time, exit status, and a conservative peak-RSS upper bound, including failures.
Its transport working-set estimate is 400 MiB; total checkpoint payload is limited to 200 MB and processed through disk and 1 MiB hashing buffers.
Eight small mocked contract tests and pinned Ruff checks pass; no final checkpoint objects were uploaded because those weights do not yet exist.
A generation-pinned read of the existing 1,429-byte European pilot config succeeded and matched MD5 HnYMvWUzjZBiXxGYiAGtbw==.
The README documents plan/apply staging before paid GPU time, followed by the existing canonical S3 download, synthetic parity gates, and six development metric/probe targets.
The newly published helper still requires independent review before actual final-checkpoint staging.

## 2026-09-10 18:02 UTC — Final staging review completed

Independent review of published helper commit 94dc07ed found a race between the pressure monitor reading ACTIVE and the transport clearing that global when a gcloud child exits.
The corrected interrupt function captures the child once and sends the main-process interrupt in a finally block, including when child termination itself fails.
Two deterministic regression tests cover concurrent child cleanup and termination failure; all ten tiny transport tests and pinned Ruff checks pass.
Review confirmed the correction and found no additional blocking issues in generation pinning, checksum validation, conditional S3 writes, resource bounds, or checkpoint compatibility.
The README now explicitly requires a successful staging exit and a receipt with exit_status 0 and applied true before paid GPU launch or checkpoint copying: Snakemake can recognize a partial S3 directory, and its timestamp file is not a completion gate.
No checkpoint was uploaded, and final model weights are still pending.

The reconciled budget snapshot is .agents/artifacts/issue-550/recovery/budget-reconciled-20260910.json.
It retains the $4 prior-attempt allowance, $0.9996 completed recovery compute, $0.15 recovery disk allowance, and $18.108 reserved final GPU compute, leaving $6.7424 for further storage, setup, and permitted recovery under the cumulative $30 cap.
This is a conservative reservation ledger, not a settled AWS bill.
The original four-hour recovery reservation remains preserved in the earlier budget.json snapshot.

At 18:03, the European production child was still TASK_STATE_RUNNING on attempt 0 with no error, and ten of sixteen CDS shards were complete.
Production optimizer updates and final biological evaluation remain pending; the 43.09-hour pilot projection excludes tokenization and other overhead.
The reviewed staging helper is published in fde7b9c1; the canonical model registration PR remains ready, fully checked, and unmerged at 51fd75ad.

## 2026-09-10 19:49 UTC — Production training and recovery checkpoints verified

The live controller reports the European production child TASK_STATE_RUNNING on attempt 0, with no error or worker restart.
The prior local log follower stopped receiving output at 18:13, so this status uses a fresh controller log read rather than the stale local tail.
Production completed tokenization and the training-loop elapsed time places its start at approximately 19:24:14 UTC.
At 19:48:34, progress reached 900 of 100,000 updates, generally around 1.6 seconds per update, with reported training loss 0.864 versus approximately 1.10 at update 327.
The logs show occasional ten-second data-loading waits followed by continued updates; no active failure is reported.
Current progress implies about 43–45 training hours remaining before additional validation, checkpoint, and interruption overhead.

The step-360 temporary recovery checkpoint committed at 19:34:21, and the step-737 checkpoint committed at 19:44:26.
The older temporary checkpoint was removed only after the replacement committed.
A bounded GCS listing independently confirmed step-737 metadata.json, manifest.json, manifest.ocdbt, and data under gs://marin-eu-west4/tmp/ttl=14d/checkpoints-temp/marin-eu-west4/MarinDNA/exp550_rag_five_regions/checkpoints/dna-exp550-rag46m-five-regions-v1/2026.09.10.9/checkpoints/step-737/.
The first permanent checkpoint and full chr18 LM validation are scheduled at 10,000 completed updates.
Final-checkpoint biological evaluation is still pending; no paid worker was launched or budget reservation changed during this status check.

## 2026-09-11 13:17 UTC — Host-memory failure recovered; first-checkpoint VEP added

The European production coordinator exhausted its retries overnight.
Its final worker exited at 01:13:02 UTC with code 137 and the explicit controller error "OOM killed (container exceeded memory limit)" under a 48-GiB host-memory limit.
This differs from the earlier US reconciliation failures.
The permanent update-10,000 native checkpoint, HF export, and full chr18 LM validation all completed around 00:22 UTC.
Aggregate validation loss is 0.56209385 and the equal-region macro loss is 0.55885416; these are language-model losses, not VEP results.
The last complete rolling native checkpoint is step 11501; the partially written step 11876 has no metadata.json and is excluded by native checkpoint discovery.
The sanitized failure, recovery configuration, and region validation values are in .agents/artifacts/issue-550/recovery/20260911-oom-recovery.json.

Recovery source 51d5c1fbd0636c6a953eec4997cfa45fa87c8b1e raises production host memory to 256 GiB and bounds training-loader prefetch to eight buffered/eight fetched batches, preserving data order, model, optimizer, and effective batch.
All 22 locked training tests passed remotely in 53.63 seconds, and independent review found no material defects.
The separate /gonzalo/dna-exp550-recovery-check-v6e8-20260911 pilot resumed the old native step-10 state through update 20 and produced byte-identical HF weights and exactly matching region/aggregate validation losses.
Its comparison receipt is .agents/artifacts/issue-550/recovery/20260911-memory-resume-parity.json.
The pilot reused an existing W&B run ID whose step was already ahead; verification therefore uses the durable GCS bytes and eval_metrics.jsonl, not newly logged W&B metrics.

The replacement production root /gonzalo/dna-exp550-rag46m-five-regions-v1-20260911-europe-memory was accepted at 13:01:51 UTC from eddd3a6e1f4d379b9d2aeb9a8745e60ea87221bf, whose training source and lock match 51d5c1fb.
It reuses production version 2026.09.10.9, the existing tokenized inputs, eight European v6e chips, and the canonical production W&B run ID.
Logs confirm native resume from step 11501 at 13:06:21 and continued optimizer updates at about 1.6 seconds each by 13:10, projecting roughly 39 remaining compute hours before interruption/validation overhead.
The coordinator has four failure retries and a seven-day timeout; completed native checkpoints remain the recovery authority.
W&B temporarily ignores replayed steps below the previous attempt's logged step 11867, after which normal logging can resume.
No paid AWS worker was launched.

The user now explicitly requests development VEP at the first 10,000-update checkpoint as well as the final checkpoint.
PR #565 adds the canonical dna-exp550-rag46m-five-regions-v1-step-10000 model and its three development/probe cells while preserving the final registration.
Its published head 29978e0d passed all CI checks and independent review and is ready, open, and unmerged.
The generalized checkpoint staging helper passed 12 bounded mock tests and staged all four generation-pinned 10k files into the canonical evals_v2 S3 checkpoint prefix.
The successful receipt .agents/artifacts/issue-550/evaluation/step-10000-staged.json records SHA-256 and reread verification, a 40-second transfer, and peak RSS below 158 MB.
The 10k weights have SHA-256 a4fd7d562c61aade4f86d7a3349c5894d3a18c93357d797b6a1e3cd8171d3dad.

The original $30 cumulative paid cap remains in force; the conservative ledger still reserves $18.108 for final-checkpoint GPU inference and leaves $6.7424 for other work after prior allowances.
Two full on-demand AWS evaluations project approximately $42 including previous allowances; a decision on a $45 cap versus retaining $30 and using shared capacity is pending.
No budget increase is inferred from elapsed time or the added evaluation request.
Shared H100 capacity on the existing Iris cw-us-east-02a federation is being checked to avoid additional AWS compute.
The initial synthetic-only job failed because the workspace bundle lacked Git metadata; its replacement fetched and verified the exact public source commit before inference.
Automatic approval review initially rejected that retry on checkpoint-egress grounds, then accepted it after checking the public training-input provenance, the organization's configured Iris destination, exact four-file payload, and scoped one-hour read URLs.
No new user consent was requested for that resolved restriction, and no reusable AWS credentials were copied.
The second attempt exposed a missing Python.h in the image's system Python before scoring; the next attempt uses a separate uv-managed Python 3.13 environment with the same locked dependencies.
All these synthetic checks use one H100, a 30-minute timeout, and zero automatic retries; biological VEP has not yet run.

## 2026-09-11 13:49 UTC — First-checkpoint VEP running on shared H100

The actual 10k checkpoint passed all predefined strict-fp32 eager/repeat/compiler gates on an H100.
Measured throughput was 3.2789 variants per second, projecting 4.37 hours for the 51,623 development variants before metrics.
The already-published parity receipt records the numerical checks and runtime.
A storage-addressing preflight failure was corrected and independently reviewed; the replacement passed input checks, durable-storage checks, and the maintained Snakemake dry-run before starting combined inference at 13:33 UTC.
It computes the three development score bundles with embeddings and zero-shot metrics; final-checkpoint frozen probes remain in the original scope.
Predictions are saved durably before the metrics phase.
Production training separately reached approximately 13,000 updates at 13:44 UTC, with roughly 38–39 compute hours remaining.
The shared GPU route preserves the existing paid cap.

The completion-transfer helper passed independent review after fixes for submission-error redaction and two completion races.
Execution review still blocks canonical result transfer despite the existing task authorization, so an owner-controlled permissions-mode change is pending.
A read-only watcher mode records completion locally without AWS calls or further job submissions; eleven bounded contract tests and Ruff checks pass.
Detailed operational recovery and transfer records remain in private task context while the concise scientific issue update is public.
Training and VEP continue, and the resulting files remain in durable shared storage.
