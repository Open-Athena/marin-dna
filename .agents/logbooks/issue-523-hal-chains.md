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

- Status: Paused at the user's request after the exact TP53 regional result.
  Sky job 7 was cancelled and the `issue-523-hal-chains` EC2 cluster was terminated, so compute billing has stopped.
  The direction-matched strict TP53 chain reproduces all 781 direct `halLiftover --noDupes` outcomes exactly, but this regional result still needs whole-genome validation.
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
  The direction-matched strict recipe with `axtChain -minScore=-1000000` reproduced 781 of 781 exactly.
  Next test: Generate a whole-genome baboon candidate with the direction-matched strict recipe and measure parity, chain size, and runtime.
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
- 2026-08-29: Promote the direction-matched strict recipe with a negative `axtChain` minimum score to the next whole-genome candidate after it achieved 781/781 exact TP53 parity.
- 2026-08-29: Cancel Sky job 7 and terminate the EC2 cluster to stop spend while the work is unsupervised.
  Preserve the 129,112-byte TP53 smoke archive in the current workspace pending authorization for a durable genomic-data destination; restage the source HAL from S3 when work resumes.

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

### 2026-08-29 00:14 UTC - `HALC-523-006` achieve exact regional parity with a direction-matched strict chain

- Hypothesis: The strict mismatch comes from applying `--noDupes` in the reverse direction and from `axtChain` dropping low-scoring PSL alignments.
- Base commit: `647c9c108a775d6c21b6eae23b67023a57c7ba24`.
- Orientation utility: Bioconda `ucsc-pslswap=482`, package SHA-256 `6c4c21b969e8794c5fa5a0e4ae6c85defdf7d85fa50d29e19eff9594544ce955`, executable SHA-256 `20e81ed41e19e9baa78dbfaa866343b3f037d30faf417a40eb5436e9e42fd6f7`.
  Its checksum-pinned `libiconv=1.18` and `mysql-connector-c=6.1.11` runtime packages were staged only on the existing EC2 worker.
- Direction-matched command: `halLiftover --noDupes --outPSL HAL Homo_sapiens TP53.bed Papio_anubis stdout | pslSwap | pslPosTarget | axtChain -psl -linearGap=medium | gzip`.
  This applies `--noDupes` in the same human-to-baboon direction as the direct-HAL baseline, then swaps the PSL so the chain still has human on the UCSC target/source side.
- Default-score result: The direction-matched chain took 1.011 seconds and reproduced 764/781 queries exactly (97.82%).
  All 17 discrepancies were direct-only; there were no chain-only mappings or coordinate conflicts.
- Zero-score result: Adding `axtChain -minScore=0` took 1.028 seconds and reproduced 777/781 exactly (99.49%).
  All four discrepancies remained direct-only.
- Diagnostic: The four remaining direct mappings sit in PSL records containing short, mismatch-heavy blocks; two records have zero matching bases.
  These alignments have negative chain scores and are discarded even at a zero minimum score.
- Negative-score result: `axtChain -minScore=-1000000` took 1.000 seconds, produced a 9,702-byte chain with 401 chains and 87,961 aligned block bases, and reproduced 781/781 queries exactly.
  It mapped the same 684 queries as direct HAL and left the same 97 queries unmapped, with no multiple mappings, chain-only mappings, direct-only mappings, or coordinate conflicts.
  `liftOver` took 0.010 seconds.
- Interpretation (`exploratory`): Direction matching and preservation of negative-scoring chains are both required for exact TP53 parity.
  Exact regional parity supports this as the next whole-genome candidate, but it does not establish whole-genome parity or bound chain-size inflation from retaining low-scoring chains.
- Decision: Replace the reverse-direction strict candidate in the next whole-genome run with the direction-matched strict recipe and measure output-size inflation explicitly.
  Keep the completed reverse-direction regional results as negative controls.
- Next action: Add pinned `pslSwap` support and an explicit minimum-score parameter to the experimental workflow, run locked tests and a remote dry-run, then generate one whole-genome baboon candidate from the staged HAL.

### 2026-08-29 01:11 UTC - `HALC-523-007` cancel the producer and terminate EC2

- Human decision: Pause the experiment during an unsupervised period and stop further cloud spend.
- Cancellation: `sky cancel issue-523-hal-chains 7 -y` changed Sky job 7 from `RUNNING` to `CANCELLED`.
  The cancelled process was still in the reverse-direction `Loxodonta_africana/default` `halLiftover` after about eight hours, so it produced no finalized chain.
- Preservation: The TP53 smoke directory contained 50 files and 409,498 bytes, including a manifest covering 49 payload files and 399,967 payload bytes.
  It was archived as `.agents/artifacts/issue-523-hal-chains/issue523-tp53-regional-smoke-v1.tar.gz` with 129,112 bytes and SHA-256 `f7a3b9aff52f0778907f6f389a3a494afd3da2ced66c4f896118c684b6b9495e`.
- Storage decision: The intended `s3://oa-bolinas/issues/523/tp53-regional-smoke/2026-08-29-direction-matched-v1/` upload was not performed because that genomic payload and destination required separate explicit authorization.
  Publishing the archive to GitHub was also not authorized.
  The small archive remains checksummed in the current workspace at `.agents/artifacts/issue-523-hal-chains/issue523-tp53-regional-smoke-v1.tar.gz` and is intentionally untracked.
- Termination: `sky down issue-523-hal-chains -y` completed successfully.
  A refreshed Sky status reports that `issue-523-hal-chains` is not found, confirming that EC2 compute billing has stopped.
- Ephemeral data: Termination discarded the 1.2627-TB staged HAL, derived 2bit/genome assets, unfinished elephant files, and other NVMe-only workflow state.
  The immutable source HAL remains in S3, the workflow and logbook are committed, and the TP53 result archive is preserved in the current workspace pending an authorized durable destination.
- Next action on resume: Rebase onto current `origin/main`, implement the direction-matched strict recipe with pinned Kent-482 `pslSwap` and an explicit negative minimum score, validate a diverse regional panel, then restage the HAL only when ready to launch the whole-genome baboon candidate.

### 2026-08-29 18:07 UTC - `HALC-523-008` pass the regional gate and launch the adaptive 107-species ramp

- Human decision: Authorize durable smoke-test artifacts, the complete 107-target whole-genome chain build, and an adaptive concurrency ramp on the existing EC2 node.
- Gate producer: `11322912b8564427ed8aa428f9beb87de3574308` with Cactus 3.3.0, Kent 482, human-to-target `--noDupes`, `pslSwap`, `pslPosTarget`, and `axtChain -minScore=-1000000`.
- Staging result: The 1,262,706,573,453-byte HAL transferred to local NVMe in 40 minutes 53 seconds and passed size and genome validation.
- Exact gate: All 9,374 regional queries matched direct HAL outcomes for each representative target with zero multiple mappings.
  Baboon had 6,846 exact mapped and 2,528 exact unmapped queries, mouse had 1,529 and 7,845, and elephant had 2,270 and 7,104.
- Full baboon: Sky job 6 started the whole-genome direction-matched chain at 17:26:39 UTC and remained active through this checkpoint.
- Ramp snapshot: `fe1817117b6a460a7066785b97daafd2e05d3785` adds the exact 107-target cohort, resumable per-species workers, verified durable uploads, an externally managed baboon requirement, and controller state publication.
- Verification: The EC2 node passed all 269 project tests, the default 79-job Snakemake DAG dry-run, and the controller CLI check.
  Follow-up commit `08c7543e` passed the same 269 tests and Ruff check and format validation in a separate EC2 worktree; its changes are formatting-only and do not alter the pinned running controller.
- Launch: At 18:00:47 UTC, the controller launched `Mus_musculus` and `Loxodonta_africana` at concurrency 2 beside the active baboon process.
  One whole-genome `halLiftover` invocation is planned for each of the 107 targets.
- First controller sample: 356 GiB available RAM, 1.3 TiB free NVMe, 6.8% CPU busy, 0.09% iowait, load/vCPU 0.067, and zero failures.
- Ramp policy: Evaluate every ten minutes and double 2→4→8→16→32→40 only while all memory, disk, CPU, iowait, and load gates pass.
- Lifecycle: Sky job 8 runs a one-CPU sentinel tied to the controller PID so cluster autostop remains aware of the externally supervised work after job 6 finishes.
- Durable record: The issue body and launch status are current at https://github.com/Open-Athena/marin-dna/issues/523#issuecomment-5463995670.
- Next action: Observe the first concurrency decision, verify the first completed whole-genome chain and metrics objects, and continue reporting retries, resource pressure, wall time, and cost through terminal cohort status.

### 2026-08-29 20:05 UTC - `HALC-523-009` recover the HAL and restart at eight workers

- Failure: The v1 controller increased to 32 workers and exhausted the 384-GiB node before any whole-genome chain finalized.
  Its last durable healthy sample retained about 60 GiB available RAM, followed by an observed collapse to about 1.8 GiB and loss of SSH and Ray responsiveness.
- Recovery: Rebooting EC2 instance `i-0323e820685d4e2b1` restored SSH without erasing instance-store data.
  Both NVMe members assembled as the original clean RAID0 filesystem, and the staged HAL remains intact at 1,262,706,573,453 bytes.
- Lifecycle protection: SkyPilot retained a stale 30-minute autodown policy but could not cancel it while the rebooted cluster was in `INIT` state.
  EC2 API stop and termination protection are enabled until intentional retirement so that the preserved HAL cannot be deleted by that stale policy.
- Publication search: The official CGL 447-way release lists HAL, MAF, tree, checksums, and construction notes but no chain archive or chain directory.
  The Cactus documentation instructs users to generate chains with `cactus-hal2chains`, and an independent Cactus issue records another attempt to generate human-query chains from the same 447-way HAL.
  UCSC's selected `hg38` liftOver chains use mixed target assemblies and alignment recipes, so they do not provide the strict exact-HAL control required here.
- Recovery snapshot: `b86897b7050bc9fdf397dd6abfb3af11fc876f86` creates the separate `hal-chains-directional-ramp-v2` namespace, fixes concurrency at eight, reserves 32 GiB for every admitted worker, and retains a projected 96-GiB memory floor.
- Verification: All 270 locked project tests passed remotely on the recovered EC2 node.
  The changed controller and test passed Ruff check and format validation, and the controller CLI check passed.
- Provenance correction: A first metadata-only start used a nonexistent full commit hash and was stopped before any `halLiftover` worker launched.
  Exactly three small metadata objects under that incorrect S3 prefix were deleted, and the prefix was verified empty before the corrected launch.
- Launch: The corrected controller started at 20:03:31 UTC with the verified recovery commit and configuration SHA-256 `d035c2561f6be3b11449647adfe9ce865884aef7da8b7e06b817d8a75c7f37f9`.
  Its first durable state has eight active workers, 99 queued species, zero completed species, and zero failures.
- First resources: About 365 GiB available RAM, 1.27 TiB free NVMe, 17.1% aggregate CPU busy, 1.3% iowait, and load/vCPU 0.048.
- Durable status: https://github.com/Open-Athena/marin-dna/issues/523#issuecomment-5464610246.
- Next action: Inspect the first finalized chain and metrics object, confirm sustained memory behavior at eight workers, and continue the cohort without increasing concurrency.

### 2026-09-08 19:27 UTC - `HALC-523-010` extract an additive reader and bound the adoption gate

- Recovered completion record: Issue comments 5533414047 and 5533449692 record all 107 chains plus 107 generation records verified, zero terminal failures, 90,636,189,891 compressed bytes, controller completion September 3 19:37 UTC, and worker termination September 3 23:26:27 UTC.
  The S3 chain namespace is the recovery snapshot above; live reads confirmed the baboon chain-generation SHA-256 and three strict-phyloP raw BED baselines.
- Human decision: Reuse existing chains and saved direct-HAL projections for a cheap sampled check, with no HAL restaging, new halLiftover computation, chain regeneration, full-grid projection, or training.
- Infrastructure snapshots: `25312a5a41aa5148792b2c79b939d4b57710ec7e` then `bc65d5b98b6a73100bf3686efa91d8710ace07ee`, draft PR #549.
  New Snakefile.chains and projection/chains.py accept a pinned anchor catalog and chain/dictionary manifest, preserve center-1 identities, reject ambiguity, apply the shared 255-bp contract, and account for every requested query.
  Existing HAL/MultiZ rules and shared code are unchanged.
- Review: Independent review found numeric-looking TSV IDs could be inferred as integers; the additive reader now types identifiers before inference and tests `001` versus `1`.
  CI passed after formatting and fixture checksum updates.
- Compute: The shared exe.dev VM had only approximately 2.2 GiB MemAvailable, below the local execution gate; no local data processing or test suite was run.
  `sky launch --dryrun -y .../sky/chains-smoke.yaml` selected r7i.xlarge spot in ap-northeast-2d at approximately $0.05/hour, under the $0.25/hour resource ceiling.
  Cluster chain-reader-523 launched for bounded validation with 4 vCPUs and 32 GiB; initial remote headroom exceeded 29 GiB.
  Autodown was temporarily extended from 10 to 60 idle minutes during setup/debugging because direct SSH commands do not count as Sky jobs; intentional termination is required at the end of this pass.
- Test environment: Initial remote run had 234 passed and one existing storage-integration failure because the Sky image shipped Conda 23.11.0, below Snakemake's 24.7.1 requirement.
  Sky job 2 upgraded Conda to 25.11.1; the launch template records this prerequisite.
  A dry-run-only retry used a mistyped full commit override; no real workflow ran under that identity, and the authoritative rerun uses verified `bc65d5b98b6a73100bf3686efa91d8710ace07ee`.
- Baseline availability: The old regional prefix contains metrics and chain files but no source-center or direct-HAL BED payloads, and the expected issue-owned prefix is empty.
  Do not claim to repeat the 9,374-query direct comparison from those metrics alone.
  The full strict-phyloP baseline contains the 1,136,854-query BED, request Parquet, and direct raw BEDs for baboon, mouse, and elephant.
- Sample design: Deterministic approximately equal allocation across chromosome × region × direct-mapped strata, then SHA-256-ranked fill to 10,000 per species.
  The resulting agreement rate describes this diagnostic sample, not an unbiased genome-wide fraction.
  Start with baboon to measure full-chain load/query resources before admitting mouse and elephant.
- Next: Complete remote tests and dry-runs, execute the synthetic chain workflow, then run the bounded saved-baseline sample and publish raw-coordinate parity and resource metrics.

### 2026-09-08 19:52 UTC - `HALC-523-011` validate the reader and recover from spot preemption

- Remote verification at `bc65d5b98b6a73100bf3686efa91d8710ace07ee`: all 236 locked project tests passed in 8.06 seconds; legacy dry-run had 79 jobs and the additive fixture DAG had six.
  A real six-job synthetic workflow using Kent 482 and S3 storage completed with two accepted plus/minus-strand projections and one explicit unmapped query.
  Its durable result prefix is `s3://oa-bolinas/snakemake/vertebrate_projection_dataset/results/chain-projection-v1/bc65d5b98b6a73100bf3686efa91d8710ace07ee/038b498d9a6bda979d32929240aa1fcb06c9d07342d1be94f97bb7e6fca65eb4/`.
- Environment correction: Upgrading the first worker's base Conda also changed its Python patch version, invalidating Sky's live Ray driver's version check.
  No biological workflow ran in the two failed Ray submissions.
  The maintained template now creates a separate Conda prefix and uses `jobs_and_ssh` for autodown; independent review found no additional issues through `0b68066478f079f1d81ab49a46014d176c9779ed`.
- Sample preservation: The baboon sample contains exactly 10,000 queries across 288 chromosome × region × direct-mapped strata, with 5,324 direct-mapped queries.
  Seven input artifacts are durable under `s3://oa-bolinas/issues/523/chain-reader-sampled-validation/aeb016efbda0f46c84a4672d6a3b0803bd2e90d2/Papio_anubis/`.
  Preparation uses research snapshot `aeb016efbda0f46c84a4672d6a3b0803bd2e90d2`; the audit script at `95424cdcdcd3f40725f13577796835e56d99a192` adds payload re-read and SHA-256 verification.
- Interruption: EC2 `i-05a2e9027e0823a18` was terminated by AWS spot capacity loss, not by autodown.
  Spot request `sir-wqxqh3vn` reports `instance-terminated-no-capacity` at 19:35:56 UTC.
  No biological mapped output finalized; only the three small request/producer artifacts exist in the interrupted real-data result prefix.
- Bounded retry: Launched `chain-reader-523b`, again optimizer-selected r7i.xlarge spot, 4 vCPUs / 32 GiB and approximately $0.05/hour.
  The isolated Conda setup completed without disrupting Ray; all 236 tests passed in 11.31 seconds and both dry-runs passed.
  Restored the seven saved baboon sample files without downloading the full baseline or HAL.
  The inspected real-data DAG contains six jobs and exactly one `liftOver -minMatch=0.95 -multiple` invocation; Sky job 3 runs it with a 900-second timeout.
- Next: Audit the baboon result before admitting the mouse and elephant checks, preserve verification metrics, and terminate the retry worker at the end of this bounded pass.

### 2026-09-08 20:17 UTC - `HALC-523-012` pass the three-species sampled adoption gate

- Source dictionary correction: The real baboon workflow rejected `KI270721.1` before projection because its source dictionary used UCSC scaffold names.
  The chain has 95 source contigs: 24 primary chromosomes already matched, while 70 GenBank scaffold accessions and `chrMT` needed explicit aliases.
  Research snapshot `f3c1de027c3660a17d9cb8bec9297154fd6d2328` checks all 455 original UCSC contigs against the original NCBI GRCh38 assembly report, then adds documented GenBank and assembled-molecule aliases to a 911-entry dictionary.
  Report SHA-256: `aa733ce92719f6339b3a4540137d4322894434363314c4b7704ba6691e9eab66`; dictionary SHA-256: `56706a5eeaaa368bfcab17bba94437b5a09885fe41494735a42bbb9118bac214`.
  No sizes are inferred from the candidate chain, and baboon's selected anchor Parquet plus input/direct BEDs are byte-identical before and after this correction.
- Tested reader: `0b68066478f079f1d81ab49a46014d176c9779ed`.
  Final PR #549 head `086f74ed31b4c471f538b2f9a466c866ee4e964c` adds only the full-contig alias requirement to the README.
  Independent reviews of both the published mainline diff and research alias correction found no remaining actionable issue; CI is green.
- Biological results: Exactly 10,000/10,000 raw query outcomes matched saved direct HAL for each of baboon, mouse, and elephant.
  Exact mapped/unmapped counts were 5,324/4,676; 5,170/4,830; and 5,190/4,810 respectively.
  There were no direct-only, chain-only, coordinate, or multiplicity discrepancies.
  All three deterministic samples span 288 chromosome × region × direct-mapped strata; the equal-allocation design does not estimate an unbiased genome-wide agreement rate.
- Contract accounting: Accepted 255-bp target windows numbered 5,323 / 5,170 / 5,190.
  One baboon mapped center was correctly rejected because expansion would leave its target contig; every other rejection was unmapped.
- Query timing: One Kent-482 `liftOver -minMatch=0.95 -multiple` invocation per species took 23.89 / 38.42 / 35.44 seconds, including whole-chain loading but excluding other pipeline stages.
  Sampled liftOver RSS was 1,783.75 / 6,252.93 / 6,047.54 MiB.
  Whole Snakemake command time was 104.91 / 202.02 / 167.66 seconds, and GNU-time peak RSS was 1,823,616 / 13,105,920 / 9,451,776 KiB.
  The larger whole-workflow peaks should inform memory reservations; these runs do not isolate cold-cache performance or predict all-grid throughput.
- Durable research prefix: `s3://oa-bolinas/issues/523/chain-reader-sampled-validation/f3c1de027c3660a17d9cb8bec9297154fd6d2328/<species>/`.
  All 11 payloads in each final manifest were re-read and SHA-256 verified from S3; resource-summary uploads were re-read separately.
  Manifest SHA-256 values in species order are `68ac25763504c515078f5df83ef1a3b02ed605b44750a79cb1399f52ac6f7061`, `88f72a2c301a4524a835408d4b2b0fbed2c248674d6c2e49986181eafcd104e1`, and `86107facc2de1a556cd08cb1f8639fd05468d355389f45ebf18cdcf4470765a3`.
  Each parity JSON identifies its workflow-native prefix, pinned chain, source/target dictionaries, species manifest, and audit code.
- Lifecycle: Retry EC2 `i-0c42f772833fbe270` (`sir-y81qjgyn`) launched at 19:47:13 UTC; Sky jobs 5, 9, and 11 completed the biological workflows and jobs 6, 10, and 12 completed their audits.
  `sky down chain-reader-523b -y` completed after artifact verification.
  EC2 subsequently confirmed state `terminated`, with user-initiated transition at 20:15:43 UTC.
  Only ephemeral worker copies were discarded; the original chains, saved baselines, and finalized validation artifacts remain in S3.
- Decision: The sampled gate supports opt-in center-1 projection with these pinned assets and the unchanged acceptance rules.
  It does not establish equivalence for all species or genomic coordinates, validate sequence extraction, authorize full-grid execution, or assess another chain-scoring threshold.
  No HAL staging, new halLiftover, chain regeneration, dataset replacement, or training occurred.
- Publication: https://github.com/Open-Athena/marin-dna/issues/523#issuecomment-5591278401 records the result and qualified go decision.
  PR #549 is ready for human review, not merged; the #517 interpretation remains separately reviewable in PR #548.
