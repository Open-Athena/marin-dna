---
topic: hal-derived-whole-genome-chains
issue: https://github.com/Open-Athena/marin-dna/issues/523
description: Build and validate reusable whole-genome human-to-mammal chains from the Zoonomia HAL.
author: gonzalobenegas
---

# Issue #523 HAL-derived whole-genome chains: Research Logbook

## Scope

- Goal: Materialize and pin one reusable whole-genome human-to-species UCSC chain for each of the 107 family-deduplicated Zoonomia mammals.
- Primary metrics: Chain-generation wall time, peak process and node memory, page-cache pressure, local disk, compressed chain bytes, exact coordinate parity with direct `halLiftover --noDupes`, and all-grid `liftOver` time.
- Constraints: Use EC2 for all data-scale work, preserve 0-based half-open coordinates, keep completed issue #517 projection artifacts immutable, and do not scale to all 107 species until the three-species semantic and resource gate passes.
- Coordinating issue: https://github.com/Open-Athena/marin-dna/issues/523
- Parent experiment: https://github.com/Open-Athena/marin-dna/issues/517

## Current TL;DR

- Status: The real EC2 pilot is running; HAL staging, validation, the full-grid BED, and all four genome-asset builds are complete, and the first two chain conversions are active.
- Selected artifact: Whole-genome human-to-species chains, because later tilings, window lengths, anchor positions, and arbitrary annotations must not require another HAL traversal.
- Pilot: Build supported-default and `--noDupes` candidates for `Papio_anubis`, `Mus_musculus`, and `Loxodonta_africana` from one NVMe-staged HAL, with at most two pair pipelines running concurrently.
- Gate: Compare all 1,136,854 strict-phyloP center mappings with their immutable direct-HAL outputs before accepting either chain recipe.

## Baseline

- The immutable HAL is `s3://oa-bolinas/staging/447-mammalian-2022v1.hal`, with 1,262,706,573,453 bytes.
- The issue #517 GPN build projected 1,627,410 centers through exactly 107 `halLiftover --noDupes` calls in 41 minutes 30 seconds on a `c6id.12xlarge`.
- The strict-phyloP control projected 1,136,854 centers through exactly 107 calls in 46 minutes 37 seconds on the same instance type.
- The non-monotonic timings show that the existing 14.10-fold all-grid linear extrapolation is not a validated scaling law.
- The strict-phyloP input BED and direct raw outputs for all three pilot species are present in the workflow-owned S3 namespace and will be reused as the parity baseline.

## Hypothesis Queue

### Active

- `HALC-523-H1`: A whole-genome chain generated from `halLiftover --outPSL --noDupes` reproduces the current direct center-1 mappings closely enough to become the reusable scientific backend.
  Next test: Exact three-species parity over 1,136,854 queries per species.
- `HALC-523-H2`: Two concurrent chain pipelines can safely share one NVMe HAL copy and OS page cache on an `r6id.12xlarge` with 384 GiB RAM.
  Next test: Measure GNU-time RSS and five-second node memory, cache, dirty-page, and free-disk samples during the first concurrent pair.
- `HALC-523-H3`: Once generated, a chain can project all 22,948,560 uniform-grid centers fast enough that future selector experiments should filter after projection.
  Next test: Time UCSC `liftOver` over the complete center BED for each strict chain candidate.

### Blocked

- None.

### Falsified / Dead End

- A fixed-grid coordinate cache was rejected as the primary artifact because it cannot serve future tilings or arbitrary annotations.
- HAL's `hal2chain` binary was rejected because its source labels it unfinished and untested.
- MAF and TAF exports were rejected because the released direct chain pipeline does not require them.

### Promoted

- None.

## Decision Log

- 2026-08-28: Select whole-genome chains instead of a fixed-grid mapping cache.
- 2026-08-28: Use the released Cactus 3.3.0 HAL→PSL→chain pipeline in an isolated additive workflow.
- 2026-08-28: Compare the supported default chain with a strict candidate that adds `--noDupes`.
- 2026-08-28: Stage the HAL once on on-demand EC2 and run at most two pair pipelines concurrently.

## Entry Log

### 2026-08-28 16:42 UTC - `HALC-523-001` authorize and design the three-species pilot

- Human decision: Start the whole-genome chain work and use the batched shared-HAL approach in the pilot.
- Implementation boundary: Add `workflow/hal_chains.Snakefile`, task-specific rules, configuration, tests, and `sky/hal_chain_pilot.yaml` without editing existing projection rules or namespaces.
- Storage: The Snakemake profile owns durable chains and audits under `hal-chains-pilot-v1/<producer-commit>/<config-sha256>/full`.
  The HAL, derived 2bits, whole-genome BEDs, and raw all-grid liftOver files remain local to instance-store NVMe.
- Compute: Use on-demand AWS `r6id.12xlarge` in `us-east-2` with 48 vCPUs, 384 GiB RAM, and 2×1,425-GB NVMe in RAID0.
  On-demand placement avoids losing the 1.26-TB staged HAL to a spot preemption.
- Batch design: Six chain pipelines cover three species by two recipes.
  Snakemake permits two concurrently, each reserving 170 GiB within a 360-GiB node budget.
- Chain direction: Run each destination mammal as the HAL query and `Homo_sapiens` as the HAL target so the resulting chain has human on the UCSC `tName` source side.
- Strict baseline: Reuse the exact strict-phyloP `hal/input.bed` and direct `hal/raw/{species}.bed` objects rather than paying to repeat their `halLiftover --noDupes` calls.
- Full-grid timing: Generate the exact 22,948,560 0-based center-1 BED from the immutable strict-phyloP chromosome sizes and undefined-region BED, then run only the strict chain through `liftOver`.
- Verification plan: Run all locked project tests and a remote Snakemake dry-run before starting HAL staging or chain generation.
- Next action: Snapshot and push the pilot, launch the EC2 test target, inspect the dry-run DAG and resource schedule, then start the real target.

### 2026-08-28 17:23 UTC - `HALC-523-002` stage once and start the first shared-HAL pair

- Hypothesis: `HALC-523-H2`; two concurrent whole-genome chain conversions can safely reuse one NVMe-staged HAL on an `r6id.12xlarge`.
- Commit hash: `9627087eef9e4b1057a3b6f448771c0a17580ff0`.
- Command: `sky exec issue-523-hal-chains snakemake/vertebrate_projection_dataset/sky/hal_chain_pilot.yaml --env TARGET=all --env DRY_RUN=0 --env PIPELINE_COMMIT_SHA=9627087eef9e4b1057a3b6f448771c0a17580ff0`.
- Config: Sky job 7; on-demand AWS `r6id.12xlarge` in `us-east-2`; two-chain concurrency; Cactus 3.3.0; UCSC Kent 482.
- Result: The 1,262,706,573,453-byte HAL staged from 16:18:21 to 17:02:02 UTC in 43 minutes 41 seconds, an end-to-end average of 481.8 MB/s.
- Result: HAL validation finished in two seconds.
  All four genome-asset jobs finished by 17:03:16 UTC, 72 seconds after asset generation began.
- Result: The first pair started at 17:03:16 UTC: `Papio_anubis/no_dupes` and `Loxodonta_africana/default`.
  Both `halLiftover` processes remained CPU-active after 20 minutes 22 seconds with empty stderr.
  Their observed RSS was 8.7 GB and 0.94 GB, respectively.
- Result: Node `MemAvailable` was 371,145,776 KiB (353.9 GiB), kernel cache was 352,300,960 KiB (336.0 GiB), and the RAID had about 1.3 TiB free.
- Interpretation: The shared-HAL batch is safe at this checkpoint and the kernel has cached a large fraction of the staged HAL.
  Chain-generation wall time and parity remain unknown because neither first-pair chain has completed.
- Next action: Record the first completed chain's resource metrics, inspect direction validation and S3 upload, then allow the DAG to continue through parity and full-grid liftOver.
