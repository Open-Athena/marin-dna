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
