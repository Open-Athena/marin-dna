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

- Status: A 100-kb TP53 regional preflight completes in 1.32 seconds or less per chain recipe, but exact parity is 91.2% for the strict `--noDupes` recipe and 97.1% for the default recipe with `liftOver -multiple`.
  No fixed-producer whole-genome rerun will start until the remaining directional and chain-conversion differences are understood.
  The old elephant producer remains active and its output is protected by same-filesystem recovery links.
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
  Regional evidence: The reverse-direction `--noDupes` TP53 chain reproduced 712 of 781 queries exactly.
  Next test: Generate the PSL in the human-to-baboon direction, swap it into human-source chain orientation, and repeat the regional audit before an expensive whole-genome rerun.
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
- 2026-08-29: Require a 100-kb TP53 parity preflight before starting another whole-genome chain producer.
  Allow the already-running elephant producer to finish for whole-genome evidence.

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

### 2026-08-28 20:44 UTC - `HALC-523-003` first HAL traversal reaches chain construction

- Hypothesis: `HALC-523-H2`; two concurrent whole-genome conversions can share the staged HAL without memory pressure.
- Commit hash: `9627087eef9e4b1057a3b6f448771c0a17580ff0`.
- Command: `sky queue issue-523-hal-chains`; `sky logs issue-523-hal-chains 7 --no-follow --tail 180`; remote process, `/proc/<pid>/io`, memory, and artifact inspection over SSH.
- Config: Sky job 7; on-demand AWS `r6id.12xlarge` in `us-east-2`; first pair `Papio_anubis/no_dupes` and `Loxodonta_africana/default` started at 17:03:16 UTC.
- Result: After 3 hours 41 minutes, the `Papio_anubis/no_dupes` `halLiftover` process had exited and its downstream `axtChain` process remained active.
  The `axtChain` process had read 247,028,577,523 bytes from the pipeline and used about 1.32 GiB RSS.
- Result: The `Loxodonta_africana/default` `halLiftover` process remained active at 99.3% CPU with about 2.31 GiB RSS.
  Its downstream `axtChain` process used about 1.57 GiB RSS and had read 1,207,380,544 bytes.
- Result: No final chain or generation JSON existed yet.
  Both compressed-chain partial files remained zero bytes because `axtChain` had not closed its output stream.
  Validation liftOver and parity audits therefore had not started; Snakemake remained at 8 of 30 completed steps.
- Result: The node retained about 358 GiB available memory and 1.3 TiB free NVMe space.
- Interpretation: One of the first two HAL traversals has moved into chain construction without resource pressure or observed stderr.
  Completion and success remain unconfirmed until the pipeline closes, writes the chain and generation JSON, and uploads both to S3.
- Next action: Inspect the first finalized chain, resource metrics, S3 objects, direction audit, and strict-phyloP parity before estimating the remaining runtime.

### 2026-08-28 23:46 UTC - `HALC-523-004` repair chain-comment validation and protect the elephant output

- Hypothesis: The generated UCSC chain is structurally valid and the observed failure is confined to the post-generation direction validator.
- Failed producer: `9627087eef9e4b1057a3b6f448771c0a17580ff0`, Sky job 7.
- Fixed producer: `983c9959a3073a49bfb26afd0d5391481050a97d`.
- Result: `Papio_anubis/no_dupes` reached `validate_chain_direction` at 21:50:21 UTC, 4 hours 47 minutes 5 seconds after the pair started.
  The conversion subprocess had returned successfully, but the validator attempted to parse the valid `##matrix=axtChain` metadata line as an aligned-block integer and raised `ValueError`.
- Result: Atomic failure cleanup removed the baboon chain partial and its GNU-time file before upload.
  No baboon chain object was installed locally or sent to S3, so that candidate must be regenerated.
- Fix: `validate_chain_direction` now skips non-empty chain comment lines beginning with `#`.
  The regression fixture includes both `##matrix=axtChain` and a normal comment before the first chain header.
- Verification: Sky job 8 ran `uv run --locked pytest` on the existing EC2 worker; all 262 tests passed in 13.08 seconds.
- Verification: Sky job 9 constructed the fixed producer's 28-job recovery DAG in dry-run mode.
  The expected shared Snakemake-lock warning was present because the old elephant rule was still active; the dry-run itself exited successfully.
- Recovery: At 23:41 UTC, `Loxodonta_africana/default` remained in `halLiftover` at 99.6% CPU after 6 hours 38 minutes.
  Same-filesystem hard links now preserve its compressed-chain partial, GNU-time file, and stderr file if the old validator unlinks their workflow paths.
- Resource check: The node retained about 359 GiB available memory and 1.3 TiB free NVMe space.
- Interpretation: The failure does not reject the HAL-to-chain method; it exposes a missing chain-format case in our validator.
  Baboon must be regenerated, while elephant's completed chain bytes and runtime evidence should survive the same validator failure.
- Next action: Wait for the old elephant process to finish, validate and account for the preserved chain, then launch the fixed producer without restaging the HAL.

### 2026-08-29 00:03 UTC - `HALC-523-005` add a 100-kb TP53 regional parity preflight

- Hypothesis: A single-gene regional chain can test the HAL→PSL→chain format and mapping semantics in seconds before committing hours to whole-genome generation.
- Commit hash: `9f048d8763e69faa8dce6978f669b86839fcccf3`.
- Region: Ensembl release 115 TP53 gene interval `chr17:7,661,778-7,687,546`, converted from GTF 1-based closed coordinates to 0-based half-open coordinates.
  The test uses the centered 100-kb human interval `chr17:7,624,662-7,724,662`, its main baboon ortholog span `CM001506.2:7,324,887-7,432,036`, and 781 one-base queries aligned to the production 128-bp tiling stride.
- Config: Existing staged Zoonomia HAL; `Homo_sapiens` source; `Papio_anubis` destination; Cactus 3.3.0; UCSC Kent 482; `axtChain -linearGap=medium`; `liftOver -minMatch=0.95`.
- Strict result: The regional `--noDupes` chain finished in 0.909 seconds with 58 chains, 91,147 aligned block bases, 3,717 compressed bytes, and 321,856 KiB maximum RSS.
  Direct HAL mapping took 0.043 seconds and chain `liftOver` took 0.008 seconds.
  Exact parity was 712/781 (91.17%): 629 exact mapped, 83 exact unmapped, 50 direct-only, 14 chain-only, and 5 coordinate conflicts.
- Default result: The regional default chain finished in 1.318 seconds with 75 chains, 94,211 aligned block bases, 4,242 compressed bytes, and 393,504 KiB maximum RSS.
  Ordinary `liftOver` reproduced 728/781 queries exactly (93.21%).
  Direct default HAL produced multiple mappings for 47 queries, while ordinary `liftOver` emitted no multiple mappings.
- Multiple-mapping control: `liftOver -multiple` increased default exact parity to 758/781 (97.06%): 679 exact mapped, 79 exact unmapped, 14 mapping conflicts, and 9 direct-only.
  It recovered multiple mappings for 34 queries, versus 47 from direct HAL.
- Diagnostic: The strict discrepancies span the locus instead of clustering at the test interval boundaries.
  The direct baseline mapped 683 queries to `CM001506.2` and one to `CM001495.2`; the strict chain mapped 648 queries only to `CM001506.2`.
- Interpretation (`exploratory`): The regional harness is fast enough to become a mandatory preflight.
  The current reverse-direction `--noDupes` recipe does not meet an exact-parity gate, and the default recipe needs `liftOver -multiple` to represent duplicated mappings.
  One species and one regional chain do not establish the eventual whole-genome parity rate because `axtChain` decisions can depend on wider context.
- Concurrent producer: Sky job 7 remains active on `Loxodonta_africana/default` after 6 hours 59 minutes at 99.6% CPU.
  Its protected chain, timing, and stderr hard links are still present; all three files remain empty while the pipeline is open.
- Decision: Do not launch the fixed whole-genome rerun yet.
  Let the existing elephant producer finish, and use the regional harness to test a direction-matched human-to-baboon PSL recipe first.
- Next action: Add the missing pinned PSL/chain orientation utility, test the direction-matched strict recipe on TP53, then decide which recipe deserves a whole-genome rerun.
