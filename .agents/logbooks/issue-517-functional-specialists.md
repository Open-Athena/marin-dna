---
topic: issue-517-functional-specialists
issue: https://github.com/Open-Athena/marin-dna/issues/517
description: Annotation-first human anchors, vertebrate projection, and five matched region-specialist models.
author: gonzalobenegas
---

# Issue 517 Functional Specialists: Task Logbook

## Scope

- Goal: Test whether annotation-first functional anchors projected across the production vertebrate cohort recover the five-arm region-specialist diagonal.
- Primary metrics: Development Mendelian AUPRC by specialist subset and paired joint-bootstrap probability that the mapped home arm ranks first.
- Constraints: Use 0-based half-open coordinates internally, preserve the production uniform-anchor v2 artifacts, and do not start paid projection, publication, training, or held-out evaluation without explicit approval.
- Coordinating issue: [#517](https://github.com/Open-Athena/marin-dna/issues/517)
- Experiment ID prefix: `FAS-517`.
- Shared tags: `FAS-517`, `issue-517`, `functional-specialists`.

## Current TL;DR

The exact producer `d06519ab` reproduced the approved complete Ensembl release 115 audit and remains the direct-MAF smoke baseline.
The issue-specific non-mammal backend now uses 28 checksum-pinned UCSC hg38-to-target liftOver chains at snapshot `17ec5ddd`; the original uniform-anchor MultiZ path is unchanged.
The 250-cell chain-versus-MAF smoke is not strictly identical, but its discrepancies are bounded and fully reconciled; standard single-best liftOver is accepted as a deliberate experimental backend change rather than an equivalent optimization.
The full chain projection is the current gate.
PR #518 remains a draft record for a long-lived experiment branch rather than a merge proposal.
Paid projection, public Hugging Face publication, five-arm training, and development-only evaluation are approved; held-out even-autosome/Y evaluation remains unapproved.
Training is Hugging Face-only at immutable public dataset revisions; S3 is workflow-owned producer storage only.

## Baseline

- Date: 2026-08-24.
- Code ref: [`4ee29ac6`](https://github.com/Open-Athena/marin-dna/tree/4ee29ac67c4b5399e75b4d66724941dc3e3d3434).
- Baseline result: 221 tests passed in 11.60 seconds with 289,196 KiB peak RSS.

## Hypothesis Queue

### Active

- `FAS-517-P1`: At step 4,999, each mapped home arm ranks first on its eight development Mendelian subsets.
  Next test: run the full chain projection, publish its five immutable public Hugging Face datasets, then run the CDS training canary.
- `FAS-517-P2`: The mapped home arm reaches the #459 persistence threshold during training.
  Next test: retain every 500-step checkpoint and run the preregistered development-only trajectory evaluation.

### Blocked

None.

### Falsified / Dead End

- `FAS-517-B1`: Direct MultiZ MAF scanning for the issue-specific full projection.
  Why stopped: The 24 compressed MAFs total 74,694,939,245 bytes, while complete version-matched UCSC chains for the same 28 targets total 279,346,460 bytes.
  Evidence: `FAS-517-007` and `FAS-517-008`.

### Promoted

- `FAS-517-H1`: The additive Ensembl builder reconciles feature extraction, priority ownership, tiling, stable identity, conservation subsets, and review artifacts before projection.
  Evidence: commit `731807af`, 236 locked tests, and the 17-job preprojection DAG check.

## Decision Log

- 2026-08-24: Extend `snakemake/vertebrate_projection_dataset` additively and leave the production uniform-anchor v2 path unchanged.
- 2026-08-24: Reuse the maintained center-1 projection contract downstream of a new five-arm projection catalog.
- 2026-08-24: Use the complete Ensembl GRCh38 release 115 GTF and all qualifying transcripts; do not use RefSeq or an Ensembl_canonical-only filter.
- 2026-08-24: Keep the existing approval gates for projection, publication, training, and held-out evaluation.
- 2026-08-24: Keep PR #518 draft and treat this branch as the permanent experiment record; decide whether to extract reusable mainline changes only after the end-to-end results are available.
- 2026-08-24: Proceed with paid projection, five-arm training, and development-only evaluation.
  Keep public publication and held-out even-autosome/Y evaluation gated separately.
- 2026-08-24: Replace the issue-specific direct-MAF non-mammal projection with 28 batched UCSC liftOver calls after smoke parity review.
  Keep the production uniform-anchor MultiZ workflow unchanged.
- 2026-08-24: Accept standard single-best liftOver for the full issue #517 experiment after a non-strict 250-cell parity smoke.
  Record it as an intentional experimental backend change, not as output-equivalent to direct MultiZ MAF projection.

## Background Research Brief

- Effort: Low.
- Stop rule: Stop when the current issue, current code, durable Marin experiment records, and authoritative input-format sources no longer change the implementation hypothesis.
- Date: 2026-08-24.

### Question

Which existing contracts can #517 reuse, and which behavior must be implemented in a new additive path?

### Current Marin Context

The production vertebrate workflow already bundles human anchors into one center-1 request set and applies one shared HAL/MultiZ acceptance, orientation, extraction, and rejection contract.
Its current anchor path instead creates uniform conservation-selected windows and assigns functional labels after tiling, so it does not satisfy #517's annotation-first source identity or ownership gate.

### Internal Prior Work

- [#326](https://github.com/Open-Athena/marin-dna/issues/326) found that removing exon-overlap contamination moved distal AUPRC from 0.127 to 0.299 and collapsed off-diagonal splicing skill from 0.238 to 0.095.
- [#351](https://github.com/Open-Athena/marin-dna/issues/351) established one 255 bp centered window per dELS/pELS cCRE with all annotated exons excluded, while showing that centered-versus-tiled accuracy remained confounded by unequal epochs.
- [#417](https://github.com/Open-Athena/marin-dna/issues/417) and [#473](https://github.com/Open-Athena/marin-dna/issues/473) established the vertebrate cohort and center-1 projection contract reused by #517.
- The current [`anchors.smk`](https://github.com/Open-Athena/marin-dna/blob/4ee29ac67c4b5399e75b4d66724941dc3e3d3434/snakemake/vertebrate_projection_dataset/workflow/rules/anchors.smk) is the incompatible uniform-anchor path that must remain intact.
- The current [`validation.py`](https://github.com/Open-Athena/marin-dna/blob/4ee29ac67c4b5399e75b4d66724941dc3e3d3434/snakemake/vertebrate_projection_dataset/src/marin_dna_vertebrate_projection/validation.py) contains reusable Ensembl-flavored UTR and ncRNA extraction precedents, but its canonical-transcript filtering and validation-window semantics are not the #517 training-anchor contract.

### External Prior Art

- [Ensembl release 115 GTF README](https://ftp.ensembl.org/pub/release-115/gtf/homo_sapiens/README) confirms 1-based inclusive feature coordinates, bare chromosome names, transcript and gene biotype attributes, and transcript-stable identifiers.
- [ENCODE SCREEN cCRE classification](https://screen.encodeproject.org/index/about) defines pELS and dELS as the enhancer-like signature subclasses used by the enhancer arm.

### Negative / Failed Leads

- Reusing the existing uniform-anchor labels would invert #517's construction order and lose source-feature identity.
- Reusing the validation recipes directly would incorrectly restrict training anchors to canonical transcripts and would not apply the five-way base-priority ownership contract.
- Subtracting exon bases from enhancer windows is insufficient because it can fragment a window; #517 requires rejecting every centered enhancer window with any annotated-exon overlap.

### Evidence Map

#### Claim: The projection backend can be reused downstream of a new anchor catalog

- Support:
  - [Current projection requests](https://github.com/Open-Athena/marin-dna/blob/4ee29ac67c4b5399e75b4d66724941dc3e3d3434/snakemake/vertebrate_projection_dataset/src/marin_dna_vertebrate_projection/projection/requests.py) preserve the 255 bp source anchor and submit only its center base.
  - [#473](https://github.com/Open-Athena/marin-dna/issues/473) selected center-1 as the production default.
- Contradictions:
  - The current anchor reader retains only the five projection identity fields, so functional audit metadata must be joined back after projection instead of silently assuming it propagates through the shared contract.
- Directness to Marin: Exact maintained workflow and experiment contract.
- Confidence: Stable.
- Action: Keep the shared projection implementation unchanged and add a functional catalog plus downstream metadata join.

#### Claim: Enhancer candidates must be rejected on any exon overlap

- Support:
  - [#326](https://github.com/Open-Athena/marin-dna/issues/326) directly attributes the old arm's off-diagonal coding skill to exon contamination.
  - [#351](https://github.com/Open-Athena/marin-dna/issues/351) uses centered dELS/pELS windows with exon-overlap exclusion.
- Contradictions:
  - Centered windows had more effective epochs than tiled windows in #351, so that issue does not isolate an accuracy benefit from centering.
- Directness to Marin: Exact historical specialist setup and target arm.
- Confidence: Replicated for contamination removal; exploratory for any independent centering benefit.
- Action: Reuse centering as the anchor definition and treat zero exon overlap as a correctness invariant, not a model-selection result.

### Recommended Next Experiments

#### 1. `FAS-517-H1`: Synthetic human-anchor contract

- Minimum experiment: Unit tests for coordinate conversion, transcript duplication, CDS/UTR overlap, TSS/ncRNA priority, long tiling, short expansion, exact ownership ties, bounds, enhancer exon rejection, stable IDs, and 10%/20% catalog nesting.
- Baseline/control: The current locked 221-test project baseline.
- Expected signal: Exact reconciliation of disjoint owned bases and deterministic 255 bp candidates with no duplicate IDs.
- Falsifier: Any base with multiple owners, any retained source-arm loser, or any training anchor absent from the projection catalog.
- Cost/risk: Local and bounded.
- Sources: [#517](https://github.com/Open-Athena/marin-dna/issues/517), [#228](https://github.com/Open-Athena/marin-dna/pull/228), and the current pipeline code.

### Hypothesis Queue Update

- Add: `FAS-517-H1` as the current implementation gate.
- Revise: None.
- Falsify / stop: Do not attempt to adapt the production uniform-anchor rule in place.
- Promote: None.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Issue #517 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/517 | Fixed design and correctness gates | High | Current coordinating record |
| Issue #326 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/326 | Exon contamination and distal recovery | High | Direct experiment |
| Issue #351 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/351 | Centered enhancer recipe and epoch caveat | High | Direct experiment |
| Current pipeline | Marin code | https://github.com/Open-Athena/marin-dna/tree/4ee29ac67c4b5399e75b4d66724941dc3e3d3434/snakemake/vertebrate_projection_dataset | Reusable projection and incompatible anchor path | High | Exact baseline |
| Ensembl release 115 README | Official docs | https://ftp.ensembl.org/pub/release-115/gtf/homo_sapiens/README | GTF coordinate and attribute boundary | High | Authoritative input format |
| ENCODE SCREEN | Official docs | https://screen.encodeproject.org/index/about | pELS/dELS definitions | Medium | SCREEN currently exposes Registry V3 prose while #517 pins the V4 file |

### Handoff

- Suggested issue `Prior work` block: The current issue already contains a more complete prior-work section; no replacement is needed.
- Suggested logbook entry: This brief and the baseline test entry below.
- Open questions: The exact Registry V4 column schema and release checksum should be pinned in the functional recipe before a full audit execution.
- Stop reason: Additional sources did not change the additive builder, enhancer rejection, or projection-reuse decisions.

## Entry Log

### 2026-08-24 16:16 UTC - `FAS-517-001` baseline and implementation gate

- Hypothesis: The current vertebrate projection project is green before adding the annotation-first path.
- Commit hash: `4ee29ac67c4b5399e75b4d66724941dc3e3d3434`.
- Command: `flock -n /tmp/marin-dna-local-heavy.lock env POLARS_MAX_THREADS=2 RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /usr/bin/time -v nice -n 10 ionice -c 2 -n 7 uv run --locked pytest`.
- Config: Python 3.13.13, uv 0.11.31 locked environment, start 2026-08-24T16:16:05Z, end 2026-08-24T16:16:31Z.
- Result: 221 tests passed in 11.60 seconds; exit status 0; peak RSS 289,196 KiB.
- Interpretation: The branch starts from a green pipeline, so new failures can be attributed to the additive functional-anchor work.
- Next action: Implement the pipeline-local functional-anchor module and its synthetic contract tests.


### 2026-08-24 16:48 UTC - `FAS-517-002` Ensembl functional-anchor implementation

- Hypothesis: The additive builder can satisfy the fixed five-arm construction, ownership, provenance, conservation, and review contracts without changing the production uniform-anchor v2 workflow.
- Commit hash: `731807af`.
- Annotation decision: Complete Ensembl GRCh38 release 115 GTF, all qualifying transcripts, no RefSeq, and no Ensembl_canonical-only filter.
- Implementation: Added `workflow/functional.Snakefile`, a pinned issue-specific config, five-arm construction and audit libraries, projection/dataset/publication targets, review reports, and a reusable runbook.
- Validation: 236 locked tests passed in 9.33 seconds with exit status 0 and 271,448 KiB peak RSS; all configured pre-commit hooks passed.
- DAG checks: The unchanged production smoke DAG resolved 79 jobs; the functional preprojection DAG resolved 17 jobs; the opt-in functional smoke projection DAG resolved 94 jobs. All checks used `--default-storage-provider none` and executed no jobs.
- Interpretation: `FAS-517-H1` passes locally. The implementation now reaches the intended human-audit gate with Ensembl as the explicit annotation source.
- Next action: Publish the branch and draft PR, review the published diff, then request human review of the preprojection artifacts before any paid projection.

### 2026-08-24 17:42 UTC - `FAS-517-003` experiment execution preflight

- Hypothesis: The complete Ensembl audit and real cross-backend smoke can run on one retained projection worker without changing the production launcher or publishing datasets.
- Commit hash: `9bcb3decfa38f6e848dafd34ec9458bcaece11a1`.
- Commands: `uv run --locked pytest`; `uv run --locked snakemake all -n --snakefile workflow/functional.Snakefile --default-storage-provider none --cores 2 --config tier=full`; `uv run --locked snakemake all_projection -n --snakefile workflow/functional.Snakefile --default-storage-provider none --cores 2 --config tier=smoke`; `sky launch --dryrun -y -c issue-517-functional-project snakemake/vertebrate_projection_dataset/sky/functional_project.yaml --env TIER=full --env TARGET=all --env PIPELINE_COMMIT_SHA=a035631076a1381c0484c19e0d9280e719c1ce22`.
- Result: 236 tests passed in 9.52 seconds with 272,320 KiB peak RSS; the full preprojection audit resolved 61 jobs; the functional projection smoke resolved 94 jobs; the Sky dry-run selected one on-demand AWS `c6id.12xlarge` in `us-east-2` at $2.42/hour and provisioned nothing.
- Cost gate: The audit plus smoke is estimated at two to three worker-hours, or about $5–8 before storage and transfer charges.
  The comparable issue #473 producer staged the exact 1,262,706,573,453-byte HAL in about 49 minutes and completed a 634,926-anchor bundled projection phase in roughly five hours.
- Interpretation: The isolated launcher is ready for the approved Ensembl audit and projection smoke.
  PR #518 is draft, public dataset publication remains gated, and held-out VEP data remain untouched.
- Next action: Push this snapshot, materialize the full audit, inspect every reconciliation and manual sample, then run the smoke only if the audit passes.

### 2026-08-24 18:09 UTC - `FAS-517-004` preliminary Ensembl audit and completed review gate

- Hypothesis: The complete Ensembl release 115 anchor catalog reconciles biologically and the preprojection target materializes every issue-mandated review surface before projection.
- Preliminary producer commit: `d9af68a35ef3133e74c39f7cb1c7f44dc6b4167d`.
- Corrected audit-gate commit: `713a653873511138389133319910259133b84785`.
- Execution: Sky job 1 failed during setup because `uv --version` included an architecture suffix; it read no scientific input.
  Sky job 2 passed setup and completed the 61-job preliminary full-tier audit in 5 minutes 5 seconds on one AWS `c6id.12xlarge` at $2.42/hour.
- Preliminary Ensembl counts: 229,962 retained CDS candidates, 278,932 3′ UTR, 169,444 TSS-region, 397,639 ncRNA, and 1,428,246 enhancer.
  The ≥10% projection catalog contains 205,131 CDS, 83,766 3′ UTR, 64,349 TSS-region, 48,982 ncRNA, and 202,452 enhancer anchors.
  The ≥20% training catalog contains 188,830 CDS, 52,099 3′ UTR, 37,608 TSS-region, 28,815 ncRNA, and 117,010 enhancer anchors.
- Gate finding: The preliminary target did not materialize the required development-locus overlap, preprojection human GC/repeat/ambiguity distributions, chromosome summary, or explicit construction and ownership loss summaries.
  The preliminary result is therefore not an approved projection input even though its existing rules succeeded.
- Correction: Commit `713a6538` adds pinned development-only overlap against `marin-dna/evals_mendelian_traits` revision `4aed58e50c5dea0b878a665007af2ef9e5108e9f`, rejects any chromosome outside odd autosomes/X, removes complete mature-miRNA match groups, and converts 1-based variant positions to 0-based half-open points at the boundary.
  It also moves human sequence composition into the preprojection gate and adds arm-wise quantiles, chromosome counts, and explicit construction and ownership loss tables.
- Real-data development check: Four complete mature-miRNA match groups were removed.
  In the ≥20% training catalog, home-arm positive-locus coverage was 85.7% for CDS/missense, 69.9% for CDS/splicing, 76.1% for CDS/synonymous, 36.4% for 3′ UTR, 43.3% for TSS/5′ UTR, 41.0% for TSS/promoter, 65.2% for ncRNA, and 34.5% for enhancer/distal.
  The home arm had the highest coverage on every mapped subset in both conservation catalogs.
  This is an anchor-composition sanity check, not a model result.
- Local resource note: The first real-catalog overlap check ran from 2026-08-24T18:06:38Z to 18:06:40Z and unexpectedly reached 563,148 KiB peak RSS, above the 500 MiB local planning bound.
  The corrected implementation now reads only the five required anchor columns, and every subsequent full-data audit runs on the retained remote worker.
- Validation: 238 locked tests passed in 9.45 seconds with 278,696 KiB peak RSS; all applicable pre-commit hooks passed; the expanded full-tier preprojection dry-run resolved 68 jobs.
- Scope: The annotation remains the complete Ensembl GRCh38 release 115 GTF with all qualifying transcripts.
  RefSeq is not used.
  Public publication and held-out even-autosome/Y access remain gated and untouched.
- Interpretation: The preliminary catalog is biologically plausible, but the corrected 68-job audit must be materialized and reviewed before the real projection smoke.
- Next action: Push the corrected snapshot, rerun the complete remote audit, inspect every new table and deterministic sample, then launch the real cross-backend smoke only if all reconciliation gates pass.

### 2026-08-24 18:22 UTC - `FAS-517-005` corrected Ensembl audit passes

- Hypothesis: The complete Ensembl release 115 human-anchor catalog passes the corrected preprojection gate without coordinate, purity, conservation, or development-overlap contradictions.
- Producer: Commit `713a653873511138389133319910259133b84785`, config SHA-256 `ae0a77f634c651f189281faedc50fd5b4cee1f6ce6d0d9bd14dbda76f85d0b55`, pipeline `functional-v1`, tier `full`.
- Execution: Sky job 3 completed all 68 jobs in 4 minutes 58 seconds on the retained AWS `c6id.12xlarge` worker.
- Annotation: The complete Ensembl GRCh38 release 115 GTF with all qualifying transcripts remains the sole gene annotation source.
  RefSeq and canonical-transcript-only filtering are not used.
- Catalog reconciliation: Counts exactly match the preliminary run.
  The projection catalog contains 205,131 CDS, 83,766 3′ UTR, 64,349 TSS-region, 48,982 ncRNA, and 202,452 enhancer anchors.
  The nested training catalog contains 188,830 CDS, 52,099 3′ UTR, 37,608 TSS-region, 28,815 ncRNA, and 117,010 enhancer anchors.
- Sequence and purity gates: All projection and training anchors have zero ambiguous bases.
  Retained enhancer exon fraction is identically zero.
  The complete 2,566,810-row ownership audit has no duplicate `(arm, chrom, start, end)` coordinates and no ownership-gate inconsistencies.
- Loss audit: Defined-sequence loss is at most 0.0023% in any non-enhancer arm and 0.0016% for enhancers.
  The expected exon-overlap exclusion removes 285,288 enhancer candidates, or 16.60%, before ownership review.
  The ownership gate retains 86.21% of CDS candidates, 99.20% of 3′ UTR, 95.48% of TSS-region, 97.45% of ncRNA, and 99.64% of enhancer candidates.
  CDS losses are 6.80% to enhancer, 4.49% to TSS-region, 2.10% to 3′ UTR, and 0.39% to ncRNA.
  Lost CDS windows have only 59–81 median CDS-owned bases depending on the winner, versus 170 median bases among retained CDS windows, which is consistent with the fixed window-majority gate.
  The CDS loss is diffuse rather than locus-concentrated: chromosome loss fractions range from 5.56% on Y to 17.91% on chromosome 22, the largest chromosome enrichment is 1.30×, and the largest one-megabase winner bin contributes 0.26% of all CDS losses.
- Distribution audit: No arm places more than 10.45% of either catalog on one chromosome.
  Human GC, repeat masking, source ownership, union-functional coverage, exon coverage, and conservation quantiles are finite and biologically ordered across the five arms.
- Development-only gate: Four complete mature-miRNA match groups were removed, only odd autosomes and X were present, and the 1-based VEP position was converted to the 0-based half-open point `[pos-1,pos)` at the boundary.
  The biologically mapped home arm has the highest positive-locus overlap for all eight subsets in both the ≥10% projection and ≥20% training catalogs.
- Publication decision: The user explicitly approved publishing development artifacts to public Hugging Face and does not want training to read from S3.
  S3 remains the authoritative workflow owner, while validated arm datasets will be published under `marin-dna`, verified without credentials, and consumed by training at immutable Hub revisions.
- Interpretation: The 13.79% CDS ownership loss is material but exactly reconciled, low-purity by construction, and broadly distributed rather than biologically concentrated.
  The corrected preprojection audit passes without weakening the fixed gate.
- Next action: Run the real cross-backend smoke, then the full bundled projection if smoke accounting and sequence invariants pass.

### 2026-08-24 19:45 UTC - `FAS-517-006` exact publication producer and cross-backend smoke pass

- Hypothesis: The final publication producer reproduces the reviewed complete Ensembl audit and exercises both real projection backends before the full bundled projection.
- SHA correction: The exact corrected-audit commit is `713a6538f798761c5186520bc7c6823cae73c8bc`.
  Entries `FAS-517-004` and `FAS-517-005` and the corresponding progress comment expanded the abbreviated `713a6538` incorrectly as `713a653873511138389133319910259133b84785`.
  That incorrect string was used only as the old smoke's isolated producer namespace; the underlying checked-out code was the expected experiment branch, and the final audit and final smoke below use an exact locally resolved commit.
- Final producer: `d06519abc5dc2c6c14d4c9765057a6a363305ee5`.
  The full-tier config SHA-256 is `b7e3206e01f39ef85277e690ea1c82525e2e9a16d286cf662d810b7525c5908f` and the smoke-tier config SHA-256 is `a903698a0564180f24a7e52d3d03da2b748a90a11a7d77312b77b9e4c07adb40`.
- Final audit: Sky job 5 completed all 68 jobs in 4 minutes 57 seconds.
  It exactly reproduced the reviewed projection counts of 205,131 CDS, 83,766 3′ UTR, 64,349 TSS-region, 48,982 ncRNA, and 202,452 enhancer anchors and the nested training counts of 188,830, 52,099, 37,608, 28,815, and 117,010.
- Final smoke: Sky job 6 completed all 90 jobs in 17 minutes 52 seconds on the retained AWS `c6id.12xlarge`.
  It exercised Ensembl 115 and cCRE V4 anchor construction on chromosomes 7 and 18, center-1 HAL projection to mouse and elephant, MultiZ projection to five non-mammal clades, target twoBit compatibility, both strands, sequence extraction, QC, five arm splits, and final public-Hub cards.
- Smoke reconciliation: Each arm contributes 10 projection anchors and eight ≥20% training anchors.
  The 350 requested anchor-species cells reconcile exactly as accepted plus no-mapping, with no other rejection reason.
  The five arm acceptance totals are 52 CDS, 24 3′ UTR, 32 TSS-region, 33 ncRNA, and 25 enhancer projections.
- Sequence contracts: All 216 emitted rows are valid 255-base IUPAC sequences with exact in-bounds 255-base target spans and human-anchor orientation; 135 are target-forward and 81 are target-reverse.
  Seven projected rows contain 200 `N` bases in total, or 0.363% of 55,080 emitted bases, with a maximum single-row ambiguous fraction of 75.69%.
  This is preserved source assembly content and maps through the maintained tokenizer unknown token.
- Training and split contracts: The ≥20% catalog yields 180 original-orientation training-eligible source rows.
  Each arm selects one validation row before augmentation; every remaining source row has exactly one original and one correct reverse-complement training row, with no validation reverse complement in training.
- Reproducibility check: Every final-producer smoke QC table, anchor catalog, sequence table, split summary, and train/validation Parquet is byte-identical to the row-validated old smoke.
  Representative SHA-256 values are `11052c6b4c73637a414080d651ee5e991bad59c81398aac688d8a522da37ebf9` for `qc/per_anchor.parquet`, `ae5f0133f65af13372f2a6fb82358ec3a5b7f17337acd64b141a7daf3fb667ec` for `sequences/all_sources.parquet`, and `7c588a0cdeee48152a49dd72f9be6ed267bfad1f3bda78f9b58c0ef39a74082b` for `sequences/training_eligible.parquet`.
- Training boundary: The isolated trainer at `9e6e708526785aefe8abe5488f55d519959033fd` accepts only public `marin-dna/functional-*` Hugging Face datasets pinned to 40-character hexadecimal Hub commit hashes.
  Tests reject any S3 URI in the tokenization or training dependency graph, and a real Iris child-worker tokenizer preflight passed.
- Interpretation: The exact final producer passes both the complete human audit and real cross-backend smoke without a biological or pipeline-contract discrepancy.
  The full bundled projection may proceed; public publication remains downstream of its complete QC, and training will consume only the resulting immutable Hugging Face revisions.
- Next action: Launch the full 22-chromosome, 107-mammal-family, 28-non-mammal-family projection, review its complete QC, prepare and publish the five public datasets, and source-pin their exact Hub revisions before the CDS canary.

### 2026-08-24 20:27 UTC - `FAS-517-007` switch non-mammals to UCSC liftOver chains

- Hypothesis: Version-matched UCSC hg38-to-target chains can preserve or explicitly reconcile the reviewed center-1 non-mammal mappings while avoiding full MultiZ MAF scans.
- Commit hash: `07bb722328479c039bb8a988a14ff09f51c80e02`.
- Input comparison: Official chains exist for all 28 selected UCSC assemblies and total 279,346,460 compressed bytes.
  The 24 pinned MultiZ chromosome MAFs total 74,694,939,245 compressed bytes, or 267.39 times the chain volume.
- Cancelled execution: Sky job 7, the direct-MAF full projection, was cancelled after 24 minutes 25 seconds at 328 of 1,654 rules.
  The retained worker, staged HAL, earlier audit/smoke outputs, and completed durable S3 objects were preserved.
- Implementation: Added an issue-specific chain manifest with official UCSC MD5s and sizes, atomic verified download, one batched BED6 liftOver per target, mapped/unmapped partition reconciliation, exact target-2bit chromosome-size joins, and a chain adapter into the unchanged shared acceptance and sequence contracts.
  The default uniform-anchor workflow retains the direct MultiZ MAF rules.
- Coordinate contract: Human center landmarks and mapped target loci are 0-based and half-open one-base intervals.
  The default command is `liftOver -minMatch=0.95` without `-multiple`.
- Validation: All 243 locked pipeline tests passed in 10.75 seconds and every repository hook passed.
  The functional default audit resolves 24 rules, the chain smoke resolves 89 rules, and the full chain projection resolves 1,052 rules with exactly 28 `run_non_mammal_liftover` jobs and no MAF rules.
- Interpretation: The chain path removes the known full-MAF I/O cost and preserves every downstream invariant in unit and DAG checks.
  Coordinate parity with the final direct-MAF smoke remains unmeasured, so the new backend is not yet an approved full-projection producer.
- Next action: Run the real chain smoke on the retained worker and compare every non-mammal anchor/species outcome with the `d06519ab` direct-MAF smoke before restarting full projection.

### 2026-08-24 20:50 UTC - `FAS-517-008` accept bounded non-strict liftOver parity

- Hypothesis: Standard single-best UCSC liftOver can replace direct MultiZ MAF scanning for this experiment if every smoke discrepancy is bounded, sequence-consistent, and recorded rather than claimed to be equivalent.
- Candidate snapshot: `17ec5dddf7805cc76384add4d6b2877ff80917df` with configuration SHA-256 `0c99b26d54726d20e16e7496843c50c7a0f044d2072e8bfc7c7914d280a3c50b`.
- Execution: Sky dry-run job 8 succeeded with 88 rules, correcting the prelaunch 89-rule estimate in `FAS-517-007`.
  Sky job 9 then completed all 88 rules in 5 minutes 12 seconds with no projection-contract or sequence-extraction failure.
  The final direct-MAF smoke took 17 minutes 56 seconds, so the complete liftOver smoke was 3.45 times faster even though both runs also constructed Ensembl/ENCODE anchors, scored phyloP, projected two HAL mammals, extracted sequences, ran QC, and materialized dataset files.
- Request invariant: The old and new `projection_requests.parquet` tables are exactly identical across 50 anchors and five non-mammal targets, for a complete 250-cell comparison.
- Cell outcomes: 62 exact accepted outputs, 162 exact no-mappings, 13 common accepted mappings with different target windows, 11 liftOver-only mappings, and 2 direct-MAF-only mappings.
  Neither backend produced a contract rejection in these five species.
- Conflict reconciliation: All 13 non-identical common mappings remain on the same target chromosome and strand.
  Their absolute 255 bp window shift is 1 to 46 bp with a median of 4 bp; every pair overlaps by at least 209 bp, and all 13 overlapping sequence segments are exactly identical.
- Backend-only reconciliation: The pairwise chains recover 11 mappings absent from the chromosome MultiZ MAF results.
  UCSC liftOver reports both direct-MAF-only cells as `Deleted in new` under the corresponding pairwise chain.
- Multiple-mapping check: Ad hoc Sky job 12 failed before downloading or running liftOver because its shell omitted the Sky runtime AWS path.
  Corrected job 13 succeeded.
  `liftOver -multiple` returned the same query partitions and target coordinates for all five species, with zero duplicated queries; only the BED score field changed from `0` to `1`.
- Durable artifacts: The 250-row detail table, aggregate summary, and JSON disposition are under `s3://oa-bolinas/snakemake/vertebrate_projection_dataset/results/functional-v1/17ec5dddf7805cc76384add4d6b2877ff80917df/0c99b26d54726d20e16e7496843c50c7a0f044d2072e8bfc7c7914d280a3c50b/smoke/parity/maf-vs-liftover/`.
  Their SHA-256 digests are `0b9e4d2a62ca7536323a2e1a3f956230a439e9e78fd10c9b6e48c64e18ab21ac` for `details.tsv`, `9187f00684919e66f09c85dee8b408584a1c4021925bf57c4fbc3b3858064a18` for `summary.tsv`, and `0049c9debb7cae75a4402f8cb6b6659d7ef2e9af925823daf0ecebd381fcb348` for `report.json`.
- Interpretation: Strict parity is false.
  The discrepancies are internally sequence-consistent and reflect bounded differences between pairwise-chain and multiple-alignment representations rather than assembly, chromosome, strand, coordinate-boundary, or extraction corruption.
  Standard single-best liftOver is accepted for the full issue #517 experiment because it preserves the reviewed contract, recovers more smoke mappings, avoids the 267.39-fold compressed-input penalty, and has no smoke benefit from `-multiple`.
- Next action: Snapshot this interpretation, dry-run the exact full producer, then restart the 22-chromosome, 107-HAL-species, 28-liftOver-species projection before any Hugging Face publication or training.

### 2026-08-24 22:06 UTC - `FAS-517-009` full projection and publication package pass

- Hypothesis: The complete five-arm Ensembl catalog can be projected through 107 HAL mammals and 28 pairwise-chain non-mammals, split reproducibly, and packaged for Hugging Face without violating the shared coordinate, sequence, or storage contracts.
- Producer: Commit `e42a4ea1eca760219e0add91004b45cac59b19c9`, full-tier config SHA-256 `a104a2756f538a1993405165fb5b50b6d4aeaf0a32810d8bb72907916eef1beb`, pipeline `functional-v1`.
- Execution: Sky job 15 completed all 1,051 full-projection rules in 43 minutes 20 seconds on the retained AWS `c6id.12xlarge` worker.
  The graph contained 107 batched `halLiftover` jobs, 28 batched UCSC `liftOver` jobs, and zero direct-MAF jobs.
- Annotation: Human anchors use the complete Ensembl GRCh38 release 115 GTF with all qualifying transcripts.
  RefSeq and canonical-transcript-only filtering are not used, and the enhancer source remains ENCODE SCREEN Registry V4 dELS/pELS.
- Accepted projections: CDS emitted 24,922,190 accepted rows from 205,131 anchors; enhancer emitted 20,912,797 from 202,452; ncRNA emitted 4,972,825 from 48,982; TSS-region emitted 6,180,915 from 64,349; and 3′ UTR emitted 8,691,627 from 83,766.
  Mean accepted non-human species per anchor was 121.49 for CDS, 103.30 for enhancer, 101.52 for ncRNA, 96.05 for TSS-region, and 103.76 for 3′ UTR.
- Reconciliation: Every anchor requested all 135 non-human targets.
  Rejections partition into no-mapping, out-of-bounds centered windows, and target chromosomes shorter than 255 bases, with no unclassified cell.
  Only 342 of 604,680 anchors had zero accepted non-human projections.
- Dataset splits: Each arm selected exactly 16,384 unaugmented validation rows before reverse-complement augmentation with seed 517.
  Training contains 46,882,278 CDS rows, 25,364,652 enhancer rows, 6,209,692 ncRNA rows, 7,577,794 TSS-region rows, and 11,364,040 3′ UTR rows.
- Sequence audit: All 81,920 uniformly selected validation sequences are exactly 255 bases, use only the configured DNA alphabet, have in-bounds 255-base source and target intervals, and preserve the projection strand contract.
  There are 173 validation rows with at least 50% `N` and 43 all-`N` rows, or 0.211% and 0.0525% respectively.
  Every high-ambiguity row comes from an older Zoonomia HAL mammal assembly; the pairwise-chain non-mammals have no validation row at or above 50% `N`.
- Manual review: All 15 deterministic accepted examples and all four deterministic rejections were inspected.
  The rejection examples correctly represent two target chromosomes shorter than the requested window and two negative centered-window starts.
- Publication package: Sky job 16 built all 325 compressed JSONL shards but exposed a storage-path bug before manifest creation.
  Commits `e3079582eeb0552512927c75a63a9aa4cf7fc595` and `e220485133861eded90a251c314b496ae9de51aa` keep cards local and make the validator consume Snakemake's explicit staged producer paths.
  The fixed project passes 244 locked tests, all repository hooks, and the exact functional publication DAG dry-run.
  Sky job 24 then validated the exact release trees, source reconciliation, shard checksums, schemas, and split reconstruction and stored the publication manifest successfully.
- Evaluation boundary: No held-out even-autosome or chromosome-Y VEP record was read.
  The observed ambiguity rate is recorded as a low-rate source-assembly limitation for this development experiment, not silently filtered.
- Interpretation: The complete producer and its Hugging Face release package pass the fixed correctness gates.
  Public development-dataset upload is authorized and in progress; model training remains gated on anonymous verification and exact Hub revision pinning.
- Next action: Verify all five public datasets without credentials, pin their immutable Hub revisions in the HF-only trainer, then run the CDS canary and development-only evaluation before launching the other four arms.

### 2026-08-24 22:21 UTC - `FAS-517-010` public Hugging Face release

- Hypothesis: The validated five-arm release trees can be published as public, ungated Hugging Face datasets and consumed by the trainer only through immutable Hub revisions.
- Execution: Sky job 25 published all five development datasets and completed all six publication-DAG steps in 11 minutes 5 seconds.
- Immutable releases:
  - `marin-dna/functional-cds` at `eb6bc7737c7f546870020a4e3d4c7a2a20d4c92c`.
  - `marin-dna/functional-utr3` at `790ec0ade6df6dce8e597058fc819dcf13f2eed1`.
  - `marin-dna/functional-tss` at `90f596e35b9d0a79e3f7a7c889581158472694eb`.
  - `marin-dna/functional-ncrna` at `ecb7e9480be5e2c18db59b3544a0c61e23fc2a2f`.
  - `marin-dna/functional-enhancer` at `07fac22abf6d158b8a155150d8aa49e813e6125e`.
- Anonymous verification: A token-free Hub client resolved every exact revision as public and ungated.
  Each repository contains exactly 64 train shards, one validation shard, the generated card, and `.gitattributes`; every data shard has a positive size and a 64-character LFS SHA-256 identifier.
  The five data trees total 15,440,515,617 compressed bytes.
- Training gate: The HF-only launcher now pins those five revisions and records the immutable full-data producer commit `e42a4ea1eca760219e0add91004b45cac59b19c9`.
  The launch documentation targets the user-authorized `open-athena` W&B entity and a dedicated `marin-dna` project.
- Verification: All nine locked experiment-project tests passed after pinning, with peak local RSS 481,024 KiB under the shared-node resource gate.
- Evaluation boundary: No held-out even-autosome or chromosome-Y VEP record was read.
- Interpretation: Public dataset publication is complete and the training input contract no longer has an unpublished or mutable dependency.
- Next action: Commit the immutable pins, launch CDS as the canary, and require successful Hub download, tokenization, W&B telemetry, and the step-500 checkpoint before starting the other four arms.

### 2026-08-24 23:22 UTC - `FAS-517-011` launch all five HF-only training arms

- Hypothesis: The public immutable datasets can initialize the matched five-arm training experiment without any S3 training input and with healthy TPU and W&B telemetry.
- CDS canary: The complete CDS train and validation token caches were built from `marin-dna/functional-cds@eb6bc7737c7f546870020a4e3d4c7a2a20d4c92c` and stored in the training-owned GCS cache.
  A real `v5p-8` trainer completed JIT compilation and optimizer steps, reached step 224 with loss 1.3284 and approximately 634 thousand tokens per second, and remained healthy.
- W&B namespace: The authenticated account can read the `open-athena` entity but W&B rejected model-run writes there after the organization migration.
  The experiment therefore uses the existing writable `gonzalobenegas/marin` project and the shared `dna-exp517-functional-specialists` group.
  The rejected attempts consumed no optimizer step.
- Protocol deviation: The original plan gated the remaining launches on a verified CDS step-500 checkpoint.
  After the CDS first-step, sustained-training, HF-cache, and W&B gates passed, the user explicitly directed launching the other four arms before step 500.
- Launches: Iris accepted `/ubuntu/exp517-utr3-personal`, `/ubuntu/exp517-tss_region-personal`, `/ubuntu/exp517-ncrna-personal`, and `/ubuntu/exp517-enhancer-personal` between 23:22:09 and 23:22:28 UTC.
  Every launcher pins its arm's public 40-character Hugging Face revision, forwards `gonzalobenegas/marin`, and allows either `v5p-8` or `v6e-4` in `us-east5`.
- Initial status: All four parents spawned their arm-specific Hugging Face tokenization child.
  UTR3 workers read `marin-dna/functional-utr3@790ec0ade6df6dce8e597058fc819dcf13f2eed1` directly from Hub and completed their parallel train-shard pass; TSS-region, ncRNA, and enhancer workers are active on their corresponding Hub repositories.
- Fixed exposure: At 5,000 steps by 8,192 sequences, the augmented training-table exposures are 0.8737 epochs for CDS, 3.6044 for 3′ UTR, 5.4053 for TSS-region, 6.5961 for ncRNA, and 1.6148 for enhancer.
- Evaluation preparation: The development-only registry contains 50 Mendelian cells, 50 Complex Traits cells, and the 10 CDS-only SGE cells required by the preregistration.
  It contains no held-out dataset and removes complete mature-miRNA groups before Mendelian metrics.
- Evaluation boundary: No held-out even-autosome or chromosome-Y VEP record was read.
- Next action: Verify immutable Hub reads and W&B initialization for all four new arms, monitor all five through the terminal step 4,999 exports, then run the 110-cell development evaluation DAG.

### 2026-08-24 23:50 UTC - `FAS-517-012` multi-arm startup validation

- Tokenization: The UTR3, TSS-region, ncRNA, and enhancer tokenization parents all succeeded after complete train and validation passes against their pinned public Hugging Face revisions.
  Zephyr terminated the completed shard workers and cache probes normally after their work drained.
- TPU allocation: UTR3, TSS-region, and ncRNA received four-chip TPU v6e workers and initialized the intended 0.25B, 8,192-sequence training configuration.
  All three registered their expected W&B runs; UTR3 and TSS-region began loading their first cached training batches while ncRNA completed runtime setup.
  The independent enhancer trainer remains accepted but queued on explicit insufficient-TPU capacity feedback.
- CDS recovery: The CDS worker was preempted after W&B step 236 and resumed from its latest temporary checkpoint at step 143.
  It retrained the discarded interval, surpassed the old W&B step, and was advancing again at step 257 with loss 1.4312.
- Evaluation implementation: Commit `b8ac67fb` adds the preregistered five-way paired match-group bootstrap.
  The optional rule produces 80 checkpoint-by-subset trajectory rows with the complete five-arm AUPRC matrix and `P(home ranks first)`, plus eight persistence rows using the first of two consecutive checkpoints at or above 95%.
  Exact ties count as ranking first, complete mature-miRNA groups are excluded before analysis, and no specialist macro or global score is emitted.
- Verification: The focused home-rank tests passed, the complete locked evals_v2 suite passed with 416 tests and five skips, repository hooks passed, and the exact issue-517 development DAG dry-run resolves 272 jobs including the single joint aggregation.
- Evaluation boundary: No held-out even-autosome or chromosome-Y VEP record was read.
- Next action: Confirm first optimizer steps for UTR3 and TSS-region, allow ncRNA and enhancer to acquire capacity, and monitor all five through their retained step-4,999 exports.

### 2026-08-25 13:05 UTC - `FAS-517-013` terminal CDS regression, anchor-first diagnosis, and ncRNA contrast

- Evaluation status: The terminal step-4,999 CDS and ncRNA exports completed their authorized development-only evaluations.
  No held-out even-autosome or chromosome-Y VEP record was read.
- CDS regression: Relative to the corrected issue-473 center-one CDS endpoint, issue 517 loses 0.0476 Mendelian missense AUPRC, 0.1326 Mendelian splicing, 0.1420 Mendelian synonymous, 0.0262 Complex Traits missense, 0.0557 SGE missense, and 0.1657 SGE splicing.
  The issue-517 endpoint is lower on all six matched rows.
- Shared annotation source: Both CDS datasets use Ensembl GRCh38 release 115 rather than RefSeq.
  The leading difference is anchor construction, not annotation provider or release.
- Historical anchors: Issue 473 inherits the issue-417 complete 255 bp genome-wide grid at 128 bp stride, filters it by phyloP, and then assigns functional labels by base-priority and window majority.
  Its CDS training catalog has 295,561 anchors and excludes all chromosome-18 source anchors from training.
- Current anchors: Issue 517 starts from disjoint owned Ensembl CDS intervals, adds 20 bp splice context, expands intervals shorter than 255 bp, merges, tiles each resulting interval from its own start, and retains a candidate only if CDS wins the 255 bp ownership vote.
  The current catalog has 188,830 training anchors, 106,731 fewer than issue 473, for a 36.11% reduction.
- Ownership loss: Of 266,735 construction-valid CDS-origin candidates, 36,773, or 13.79%, lose the ownership vote.
  Enhancer wins 18,143, TSS-region 11,964, 3-prime UTR 5,613, and ncRNA 1,053.
  These lost candidates are CDS-sparse but functional-rich: mean CDS fraction 0.2902 and mean union-functional fraction 0.8748.
- Geometric turnover: Only 1,296 coordinates are exact matches because the tiling origin changed.
  Nevertheless, 92.58% of the current union base footprint is covered by the historical catalog, while the current catalog covers only 70.42% of the historical footprint; base-level Jaccard is 0.6666.
  This supports a smaller, shifted subset of mostly the same biological loci rather than a different annotation universe.
- Distribution shift: Current CDS anchors are more conserved than historical anchors, with mean human phyloP-covered fraction 0.4542 versus 0.4120, and more CDS-dense, with mean CDS fraction 0.6895 versus 0.6063.
  The issue-517 random row-level validation split also leaves chromosome 18 in training, unlike issue 473.
- ncRNA result: Issue 517 reaches 0.551602 Mendelian non-coding-transcript-exon AUPRC with SE 0.040600 on 115 match groups and 1,150 rows, versus 0.366257 with SE 0.037725 for the exp232 terminal ncRNA specialist on the exact same rows.
  The paired 1,000-resample match-group bootstrap gives a delta of +0.185345, SE 0.034926, 95% interval [0.115002, 0.251740], and two-sided p at most 0.001.
  Issue 517 Complex Traits ncRNA AUPRC is 0.224272 with SE 0.054646 on 37 groups and 370 rows; no exp232 Complex Traits metric artifact is available for a matched comparison.
- ncRNA attribution caveat: This is not a clean anchor-only comparison.
  Exp232 trained for about 2.51 row epochs on 16,319,886 rows, whereas issue 517 trains for about 6.60 row epochs on 6,209,692 rows, so the new arm sees roughly 2.63 times as many dataset passes under the same fixed token schedule.
- Interpretation: Anchor construction is now the primary CDS regression hypothesis, especially the loss of mixed boundary windows and the 29.58% of the historical CDS footprint absent from the current catalog.
  The contrasting ncRNA gain shows that the five-arm redesign is not uniformly worse, but it cannot yet distinguish a better ncRNA anchor definition from the substantially higher ncRNA row exposure.
- Next action: Relate old-only and ownership-lost CDS anchors to the development VEP loci, especially splicing and synonymous groups, then run bounded anchor-policy ablations before attributing the regression to liftOver or training dynamics.

### 2026-08-25 13:50 UTC - `FAS-517-014` UTR3 and TSS terminal evaluation plus VEP anchor coverage

- Terminal exports: The 3-prime UTR and TSS-region arms each produced a complete step-4,999 HF-format export in the experiment-owned GCS checkpoint tree.
  Enhancer has not produced its terminal export yet.
- Evaluation execution: Managed jobs 34 through 37 completed the Mendelian and Complex Traits development evaluations for both new arms.
  A delayed first submission made a sequential retry unnecessary; retry job 38 failed on the duplicate target race, and duplicate jobs 39 through 41 were cancelled while the original four remained healthy.
  All four intended metric artifacts landed successfully.
- UTR3 home result: Issue 517 reaches Mendelian 3-prime UTR AUPRC 0.156987 with SE 0.025669 on 77 groups and 770 rows, versus exp232 at 0.216933 with SE 0.034839 on the identical support.
  The paired 1,000-resample match-group bootstrap gives delta -0.059946, SE 0.025113, 95% interval [-0.109282, -0.006289], and two-sided p 0.022.
  Issue 517 Complex Traits 3-prime UTR AUPRC is 0.171884 with SE 0.038236 on 49 groups and 490 rows; exp232 has no stored Complex Traits metric for a paired comparison.
- TSS home results: Issue 517 reaches Mendelian 5-prime UTR AUPRC 0.189121 with SE 0.020174, versus exp232 at 0.288954 with SE 0.031648 on 210 groups and 2,100 rows.
  The paired delta is -0.099833 with SE 0.018897, 95% interval [-0.139274, -0.065439], and two-sided p at most 0.001.
  On TSS-proximal variants, issue 517 reaches 0.234706 with SE 0.024475 versus exp232 at 0.258743 with SE 0.028388 on 205 groups and 2,050 rows; paired delta -0.024038, SE 0.017297, 95% interval [-0.060142, 0.007030], p 0.124.
  The TSS-proximal change is inconclusive.
- Current four-arm diagonal: UTR3 ranks first among the four completed issue-517 arms on 3-prime UTR.
  TSS-region ranks first on TSS-proximal, but ncRNA ranks first on 5-prime UTR at 0.268308 versus TSS-region at 0.189121.
  NcRNA also ranks first on its own non-coding-transcript-exon subset at 0.551602.
  The enhancer arm is still missing, so this is not the final five-way home-rank result.
- Exposure context: Issue 517 sees about 3.60 row epochs for UTR3 versus exp232's 3.25, and about 5.41 row epochs for TSS-region versus exp232's 3.63.
  The UTR3 and 5-prime UTR regressions therefore do not have an underexposure explanation under the fixed token schedule.
- Development VEP anchor coverage: On pathogenic variants, historical versus current CDS training-anchor coverage is 88.45% versus 85.69% for missense, 71.79% versus 69.91% for splicing, and 82.61% versus 76.09% for synonymous.
  Historical-only positive loci number 31 missense, 30 splicing, and 4 synonymous.
- Historical-only cause partition: For missense positives, the 31 historical-only loci partition into 4 removed by the current conservation gate, 10 by ownership, and 17 by changed construction geometry.
  For splicing, 30 partition into 4 conservation, 11 ownership, and 15 construction.
  For synonymous, 4 partition into 2 conservation, 1 ownership, and 1 construction.
- Ownership exposure: Current ownership-lost CDS candidate windows touch 9.66% of pathogenic missense, 17.55% of pathogenic splicing, and 15.22% of pathogenic synonymous loci.
  Across the full matched sets, current-versus-historical coverage is 72.31% versus 75.07% for missense, 53.01% versus 56.96% for splicing, and 78.48% versus 83.04% for synonymous.
- Interpretation: The direct locus audit strengthens the anchor-construction hypothesis.
  Construction and ownership explain most historical-only pathogenic missense and splicing coverage, while synonymous is additionally sensitive to the conservation gate.
  The cross-arm results also show that the redesign concentrates substantial 5-prime UTR signal in the ncRNA arm rather than producing a clean TSS-region diagonal.
- Evaluation boundary: No held-out even-autosome or chromosome-Y VEP record was read.
- Next action: Audit ncRNA-versus-TSS ownership at the 5-prime UTR development loci, then evaluate enhancer immediately after its terminal export and compute the preregistered five-way home-rank result.

### 2026-08-25 14:20 UTC - `FAS-517-015` off-diagonal transfer and anchor-stage localization

- Completed-arm matrix: The full four-specialist by eight-subset Mendelian matrix was compared with the matching exp232 terminal specialists on the same development support, excluding mature miRNA.
  The only material off-diagonal gain is the issue-517 ncRNA specialist on 5-prime UTR, at 0.268308 versus 0.200047 for exp232, delta +0.068261.
  A paired 1,000-resample match-group bootstrap gives SE 0.017303, 95% interval [0.031221, 0.099889], and two-sided p 0.001.
- Asymmetry: The ncRNA specialist changes by only +0.003518 on TSS-proximal variants, with 95% interval [-0.005539, 0.014910] and p 0.444.
  The TSS specialist changes by only +0.002405 on ncRNA-exon variants, with 95% interval [-0.021228, 0.025555] and p 0.828.
  The 5-prime UTR transfer is therefore specific rather than a broad promoter/ncRNA interchange.
- Direct leakage audit: Current ncRNA training anchors cover zero of 210 pathogenic 5-prime UTR loci and zero of all 2,100 matched 5-prime UTR rows.
  The significant ncRNA-to-5-prime-UTR gain is not explained by those benchmark loci being reassigned into, or directly contained by, ncRNA training anchors.
- TSS anchor-stage localization: Historical TSS anchors cover 147 of 210 pathogenic 5-prime UTR loci, whereas current final TSS anchors cover 91.
  Current TSS candidates that pass ownership cover 178, while ownership-lost TSS candidates touch only two and both lose to enhancer rather than ncRNA.
  The missing current coverage therefore appears after ownership, at the shared conservation threshold combined with annotation-first tiling geometry.
- UTR3 anchor-stage localization: Historical 3-prime UTR anchors cover 49 of 77 pathogenic 3-prime UTR loci, whereas current final UTR3 anchors cover 28.
  Current ownership-passing UTR3 candidates cover 71, and only two ownership-lost candidates touch positives; both lose to CDS.
  This independently points to post-ownership conservation and tiling rather than cross-arm reassignment.
- ncRNA selectivity: The current ncRNA catalog has 28,815 anchors spanning 5.770 million union bases, versus 98,630 historical anchors spanning 18.598 million bases.
  It retains direct coverage of 75 of 115 pathogenic ncRNA loci versus 81 historically, while matched-negative coverage falls to 40 of 1,035 from 142 historically.
  This much stronger positive-to-negative coverage enrichment is consistent with a more selective functional-biotype and conservation-filtered anchor set, not raw region leakage.
- Interpretation: The anchor evidence supports two different mechanisms.
  UTR3 and TSS regressions localize primarily to how annotation-first candidates interact with the conservation gate and tiling phase, while the ncRNA home gain is plausibly helped by aggressive anchor-set purification and its 2.63-fold greater dataset-pass exposure.
  The ncRNA-to-5-prime-UTR gain is real transfer, but direct coordinate leakage is falsified by zero current ncRNA-anchor coverage of the 5-prime UTR benchmark rows.
- Enhancer status: The enhancer trainer remains healthy and resumable after 14 preemptions.
  It saved the durable step-3,000 native and HF-compatible GCS checkpoints and was advancing near step 3,130; no terminal enhancer evaluation has been launched.
- Next action: Run bounded TSS and UTR3 anchor-policy ablations that preserve ownership while varying conservation and tiling, then separate the ncRNA functional-biotype narrowing from exposure with a matched-token comparison.

### 2026-08-25 15:07 UTC - `FAS-517-016` exact anchor-stage diagnosis and additive grid ablations

- Exact-stage result: Historical functional anchors all begin on the chromosome-wide `start % 128 == 0` grid, while each current arm occupies all 128 start residues because it tiles from each merged owned interval's start.
  Of 420,134 historical CDS, UTR3, and TSS-region anchors, 418,127, or 99.52%, were never emitted at their exact coordinate by the current candidate builder.
  Among the 2,007 exact historical coordinates that were emitted, 1,993 remain in the current at-least-20-percent training catalog, 14 CDS coordinates lose the window ownership vote, and zero fall below either conservation threshold.
- Construction-validity check: The three arms have only six construction-invalid windows genome-wide, all due to undefined sequence, and none matches a historical functional anchor.
  Exact historical anchors therefore disappear at interval-origin tiling rather than at bounds, defined-sequence, ownership, or conservation checks.
- Pathogenic anchor trace: The 156 historical-only pathogenic locus/subset cases are covered by 106 unique historical anchors, and all 106 exact anchors are absent because their coordinates were never constructed.
  At the shifted current-window level, 77 cases have an ownership-passing same-arm candidate below the 20% conservation threshold, 23 have a candidate that loses ownership, and 56 have no current candidate covering the pathogenic base.
- Annotation-versus-edge refinement: For 55 of those 56 no-candidate cases, the historical anchor still overlaps ownership-passing current same-arm candidate territory.
  Only one missense case has no current same-arm candidate overlapping the historical anchor.
  The dominant no-candidate mechanism is therefore interval-edge or terminal coverage created by the shifted lattice, not absent Ensembl annotation.
- Prior interpretation correction: The `FAS-517-014` phrase "removed by the current conservation gate" applies to shifted current candidates at the locus, not to the exact historical anchor.
  The exact historical window is never scored in the current pipeline.
- Additive-rule contract: Every ablation retains the current at-least-20-percent catalog and adds only historical chromosome-grid anchors that themselves satisfy the same at-least-20-percent phyloP-covered-base threshold.
  Development labels are used only for the post-construction coverage readout, mature-miRNA groups are removed completely, and no held-out even-autosome or chromosome-Y VEP record is read.
- Upper bound: Adding every historical same-arm grid anchor restores all historical-only pathogenic coverage on top of the current catalog, moving missense from 497 to 528 of 580 positives, splicing from 223 to 253 of 319, synonymous from 35 to 39 of 46, UTR3 from 28 to 49 of 77, 5-prime UTR from 91 to 149 of 210, and TSS-proximal from 84 to 96 of 205.
  This requires 294,265 additional CDS anchors, 66,695 UTR3 anchors, and 57,181 TSS-region anchors and is an intentionally broad diagnostic upper bound rather than a selected recipe.
- Novel-base ablations: For UTR3, adding 9,252 grid anchors that overlap any current same-arm candidate and contribute at least 128 bases outside the current training union recovers all 21 historical-only pathogenic loci, while adding coverage of 31 matched negatives instead of the upper bound's 45.
  For TSS-region, adding 14,510 anchors that overlap ownership-passing territory and contribute at least 128 novel bases recovers 55 of 58 historical-only 5-prime UTR positives and all 12 historical-only TSS-proximal positives, with 177 and 139 newly covered matched negatives respectively.
  For CDS, the analogous all-candidate 128-base rule adds 29,560 anchors and recovers 15 of 31 missense, 23 of 30 splicing, and 3 of 4 synonymous positives; relaxing to 64 novel bases adds 115,139 anchors and recovers 28, 27, and 3 respectively.
- Cross-region coverage: The compact 128-base CDS rule also newly covers 2 pathogenic 3-prime UTR, 2 pathogenic 5-prime UTR, 1 pathogenic ncRNA-exon, and 1 pathogenic TSS-proximal locus.
  The compact UTR3 rule newly covers 10 pathogenic missense, 3 pathogenic splicing, and 1 pathogenic synonymous locus, while the compact TSS rule newly covers 21 missense, 20 splicing, 2 synonymous, and 1 ncRNA-exon locus.
  These are coordinate-overlap warnings, not model-performance results, and argue against choosing an additive rule from home coverage alone.
- Interpretation: The material code difference is ordering rather than annotation source, window size, stride, or conservation threshold.
  A chromosome-global grid was previously scored and then labeled, whereas issue 517 resolves base ownership first and tiles each resulting interval from its own origin.
  Compact novel-base grid additions can recover most UTR3 and TSS pathogenic coverage, but the CDS missense/splicing tradeoff and cross-region overlaps require an exposure-aware training ablation before any rule is accepted.
- Next action: Publish this diagnosis and policy table to issue 517, then turn the shortlisted per-arm policies into isolated, reproducible experiment outputs without replacing the current catalog or launching training yet.

### 2026-08-25 18:25 UTC - `FAS-517-017` terminal enhancer evaluation

- Terminal export: The enhancer arm produced a complete step-4,999 HF-format export in the experiment-owned GCS checkpoint tree.
- Evaluation result: The canonical development-only evaluation completed all five intended jobs and stored both score files and both metric files under the standard evals_v2 S3 prefix.
- Enhancer home result: Issue 517 reaches Mendelian distal AUPRC 0.322690 with SE 0.055261 on 58 match groups and 580 rows.
  The exp232 terminal enhancer specialist reaches 0.126778 with SE 0.025444 on identical support, for an issue-517 absolute gain of 0.195912.
- Complex-trait result: Issue 517 reaches Complex Traits distal AUPRC 0.127846 with SE 0.008923 on 616 match groups and 6,160 rows.
  No exp232 Complex Traits metric artifact is stored for a matched terminal comparison.
- Five-arm distal comparison: Enhancer ranks first on the Mendelian distal subset at 0.322690, followed by UTR3 at 0.115520, CDS at 0.110572, ncRNA at 0.103049, and TSS-region at 0.096410.
  Enhancer also ranks first on Complex Traits distal at 0.127846, followed by UTR3 at 0.102944, CDS at 0.099658, TSS-region at 0.098834, and ncRNA at 0.098485.
- Interpretation: The enhancer specialist has a large, region-specific diagonal advantage on both distal endpoints rather than an off-diagonal specialist winning through apparent region leakage.
  The result contrasts with the CDS, UTR3, and 5-prime UTR regressions and with the ncRNA home gain, reinforcing that the anchor redesign has strongly region-dependent effects.
- Analysis defect: The registered joint home-rank bootstrap retrieved all five canonical score inputs but failed because the rule opened unstaged literal `results/scores/...` paths instead of Snakemake storage-backed inputs.
  The failure did not alter the completed score or metric artifacts, and the exact five-arm AUPRC ordering above comes directly from the canonical metric files.
- Launch recovery: A stale local SkyPilot API daemon retained a deleted worktree current directory after the desktop/worktree restart and broke runtime-file synchronization before job submission.
  The evaluation was completed on an isolated A10G VM by running the exact commit-pinned workflow directly, and no model was published to Hugging Face Hub.
- Evaluation boundary: No held-out even-autosome or chromosome-Y VEP record was read.
- Next action: Post the terminal enhancer result and five-arm distal ordering to issue 517, repair the home-rank storage-input integration separately, and continue the anchor-policy investigation without selecting the fixed-grid architecture yet.

### 2026-08-25 20:58 UTC - `FAS-517-018` fixed-grid Arm A and background assignment decision

- Decision: Project every uniform 255 bp / 128 bp-stride anchor with at least 51 bases satisfying the strict GPN-Star-P criterion `entropy_calibrated < 0.081001`, independent of region assignment.
- Four-arm assignment: Reuse the #232 v4 base-priority and window-majority labels for CDS, 3-prime UTR, protein-coding TSS/5-prime UTR, and ncRNA exon.
- Enhancer assignment: Use #326 Arm A, `v4_ccre_noexon`, as the initial fifth arm.
  A window must receive the v4 `ccre_non_promoter` label and have exactly zero CDS, 3-prime UTR, TSS/5-prime UTR, and ncRNA-exon coverage.
  No dELS+pELS dominance filter is applied.
- Background assignment: Assign every GPN-selected window outside the other five arms to the sixth arm.
  This complement includes #232 v4-background windows and cCRE-labeled windows that fail Arm A.
  The six arms therefore form an exhaustive, mutually exclusive partition of the GPN-selected catalog.
- Evidence: Arm A reached development Mendelian distal AUPRC 0.299 versus 0.272 for the narrower enhancer-dominant Arm B and 0.127 for the broad #232 cCRE arm.
  These point estimates do not establish a statistically powered Arm A versus Arm B difference, but zero-other-functional curation has the clearest empirical support and Arm A retains broader non-promoter cCRE coverage.
- Architecture consequence: Background windows and broad-cCRE windows excluded from Arm A remain in the projected substrate and enter the complement background arm.
  Arm B and other assignment recipes can therefore be evaluated later without repeating cross-species projection.
- Interpretation caveat: The complement arm is not the #232 negative-control background.
  It is a heterogeneous `GPN-constrained but unassigned` arm that may contain unannotated functional sequence and rejected regulatory windows.
- Status: Human decision recorded in issue #517.
- Next action: Implement a versioned exhaustive six-arm assignment table, then report the six arm counts, exact catalog reconciliation, disjointness, and chromosome composition on EC2 before projection.

### 2026-08-25 21:44 UTC - `FAS-517-019` GPN-Star-P full catalog and six-arm assignment

- Execution: The additive `workflow/gpn_star.Snakefile` full catalog completed on the retained `c6id.12xlarge` EC2 worker under producer commit `65b7806ea56a270124c9973af0366f5ab412c665` and config SHA-256 `28cb7786197945ef1798c3581873e4b3d68b7bf91189a59585b0dcabcad7a5e4`.
  The locked EC2 test suite passed 251 tests, and the resumed execution graph contained no scoring jobs after reusing the 24 checksum-validated score shards whose producer code and immutable inputs were unchanged by the audit fix.
- Threshold audit: The strict full-genome results reproduce 22,948,560 uniform windows, 2,421,580 windows with at least 26 passing bases, and 1,627,410 windows with at least 51 passing bases.
  The projection universe is therefore exactly the windows passing the agreed 20% GPN conservation filter.
- Source-count clarification: The strict cutoff selects 109,564,133 unique GPN source bases.
  Those bases contribute 218,273,080 observations when summed over the overlapping 255 bp / 128 bp-stride grid windows.
  The previous aggregation failure came from treating the observation count as a unique-position assertion, and the corrected workflow now gates both quantities independently.
- Exhaustive arm counts: The catalog assigns 306,369 CDS, 99,802 3-prime UTR, 74,691 protein-coding TSS/5-prime UTR, 98,789 ncRNA exon, 653,017 issue-326 Arm A enhancer, and 394,742 background windows.
  All six arms are nonempty, mutually exclusive, and sum exactly to 1,627,410.
- Background decomposition: The background complement contains 321,123 v4-background windows and 73,619 cCRE-labelled windows rejected by Arm A.
  No window outside the 20%-filtered universe is assigned.
- Durable storage: The commit/config-keyed catalog, assignments, threshold audit, assignment audit, and producer manifest are stored below `s3://oa-bolinas/snakemake/vertebrate_projection_dataset/results/gpn-star-p-uniform-v1/65b7806ea56a270124c9973af0366f5ab412c665/28cb7786197945ef1798c3581873e4b3d68b7bf91189a59585b0dcabcad7a5e4/full/`.
- Reporting: The result and next execution gate were posted at https://github.com/Open-Athena/marin-dna/issues/517#issuecomment-5417208531.
- Next action: Complete the chr18 four-windows-per-arm HAL/MultiZ smoke projection and inspect its QC before launching the full 135-target projection.

### 2026-08-25 22:38 UTC - `FAS-517-020` chr18 projection smoke gate and full launch

- Smoke completion: The chr18 `all_projection` smoke workflow completed all 76 jobs under producer commit `65b7806ea56a270124c9973af0366f5ab412c665` and smoke config SHA-256 `722734369b5b4eb6e362472fe28bc82da878e379a3c32fdcdb1d7b7f224bf24d`.
  Its 24 anchors contain exactly four windows from each of the six arms, and every anchor satisfies the agreed at-least-51-of-255 GPN gate; the minimum observed count was 53 selected bases.
- Smoke projection contract: The smoke requested two Zoonomia/HAL mammals and five UCSC MultiZ non-mammals in addition to the human reference.
  The combined artifact has 93 unique query/species rows: 24 human-reference rows and 69 accepted non-human projections from 168 requests.
  All sequences and target spans are 255 bp, all coordinates are 0-based half-open and in bounds, all strings pass IUPAC validation, and there are no duplicate query/species pairs.
- Smoke recovery: The other 99 requests are `no_mapping`; no projection-contract or sequence-extraction rejection was observed.
  Accepted non-human projections by arm were 9 background, 20 CDS, 8 enhancer, 14 ncRNA exon, 7 TSS/5-prime UTR, and 11 3-prime UTR.
  Every arm recovered at least three of four anchors in a non-human species, and all four background anchors recovered.
- Inspection: The deterministic manual-inspection sidecar contains three rows per arm and includes both strands and both projection backends.
  Its sequence, coordinate, IUPAC, and extraction-orientation prechecks passed.
  The common 1-of-255 alignment-coverage display is expected because the contract uniquely projects the central human nucleotide and then extracts the centered target-genome window.
- HAL staging: The immutable 1,262,706,573,453-byte Zoonomia HAL is resident on the EC2 NVMe array and passed the workflow's size and genome-compatibility checks.
  It is reused by the full run without another S3 transfer.
- Reporting: The smoke gate was posted at https://github.com/Open-Athena/marin-dna/issues/517#issuecomment-5417760838.
- Full dry-run: The full `all_projection` DAG passed with 1,645 jobs and schedules no GPN scoring, tiling, catalog aggregation, or arm assignment.
  It consumes the durable 1,627,410-window catalog and expands to 107 HAL mammals plus 28 MultiZ non-mammals, or 219,700,350 non-human projection requests.
- Full launch: SkyPilot job 13 started successfully on the retained `c6id.12xlarge` worker, using full config SHA-256 `28cb7786197945ef1798c3581873e4b3d68b7bf91189a59585b0dcabcad7a5e4` and the existing canonical S3 result prefix.
  The launch was posted at https://github.com/Open-Athena/marin-dna/issues/517#issuecomment-5417805367.
- Next action: Monitor the full backend/species milestones, preserve resumability and NVMe capacity, then inspect the final combined sequence and QC artifacts before stopping the worker.

### 2026-08-26 00:39 UTC - `FAS-517-021` full projection completion and exhaustive audit

- Completion: SkyPilot job 13 completed all 1,645 jobs successfully under producer commit `65b7806ea56a270124c9973af0366f5ab412c665` and full config SHA-256 `28cb7786197945ef1798c3581873e4b3d68b7bf91189a59585b0dcabcad7a5e4`.
  End-to-end runtime was 1 hour 45 minutes 17 seconds, from 2026-08-25 22:36:02 UTC to 2026-08-26 00:21:19 UTC.
- HAL measurement: The full run made exactly 107 batched `halLiftover` invocations with no retries.
  The first invocation began at 22:37:29 UTC and the last completed at 23:18:59 UTC, a 41-minute-30-second HAL interval.
  Linear scaling by the 14.10-fold unfiltered-to-filtered window ratio gives a 9-hour-45-minute and $23.60 point estimate on the same node, with an 8–12-hour / $19–29 planning range.
- Output: The combined table contains 167,607,189 rows: 1,627,410 human references, 154,654,979 accepted HAL projections, and 11,324,800 accepted MultiZ projections.
  Accepted non-human output is 165,979,779 rows across all 135 target species.
- Exact accounting: `165,979,779 accepted + 52,947,632 no mapping + 726,471 target window out of bounds + 46,468 target chromosome too short = 219,700,350 requests`.
- Exhaustive audit: A bounded-memory EC2 scan covered all 1,970 row groups and all 167,607,189 rows.
  It confirmed unique query/species keys, the exact 136-species set including human, 255 bp source and output spans, target bounds, valid IUPAC strings, valid strands, catalog/assignment/request parity, and every per-anchor and aggregate reconciliation.
  The 18-row deterministic inspection sample contains three rows per arm and covers both backends and both strands; all automated prechecks pass, while biological/browser review remains pending human review.
- ZRS caveat: The conservation-filtered GPN catalog intentionally excludes the two ZRS anchors, and neither the GPN full namespace nor the chr18 smoke namespace contains the separate ZRS sidecar referenced by the generated inspection report.
  This is recorded as an outstanding QC-coverage/reporting caveat rather than a projection-accounting failure.
- Durable audit: The exact audit summary is stored at `qc/final_audit_summary.json` under the canonical full result prefix with SHA-256 `246cc85fdba7bfbc7ca13b7fd99f79f677cfc7b7f4d45429215a0e3d21790a34`.
- Follow-up: Issue https://github.com/Open-Athena/marin-dna/issues/523 tracks a separate benchmark of projecting all 22,948,560 uniform windows before conservation filtering, including format, compression, streaming, disk, and downstream-stage optimization before a larger-run approval request.
- Reporting: The completed run and final audit were posted at https://github.com/Open-Athena/marin-dna/issues/517#issuecomment-5418953835.
- Next action: Commit and publish this append-only result record, then terminate the retained EC2 worker after verifying the durable S3 audit object.
