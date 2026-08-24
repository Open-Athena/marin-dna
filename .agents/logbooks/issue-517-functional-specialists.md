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

The exact publication producer `d06519ab` reproduced the approved complete Ensembl release 115 audit and passed the real cross-backend smoke.
The 90-job final smoke completed both HAL and MultiZ paths, and every QC, anchor, sequence, and split artifact is byte-identical to the row-validated smoke.
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
  Next test: run the complete Ensembl audit, projection smoke, full projection, and training canary in sequence.
- `FAS-517-P2`: The mapped home arm reaches the #459 persistence threshold during training.
  Next test: retain every 500-step checkpoint and run the preregistered development-only trajectory evaluation.

### Blocked

None.

### Falsified / Dead End

None.

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
