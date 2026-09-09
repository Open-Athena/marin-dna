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
