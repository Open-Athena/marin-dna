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

The annotation-first five-arm experiment produced a 6/8 Mendelian diagonal, with large ncRNA and enhancer gains but lost synonymous and 5-prime-UTR ownership.
The current center-1 uniform-grid experiments produced 7/8 row wins under GPN-Star-P selection and 8/8 under the strict historical phyloP selector.
The strict phyloP matrix recovers #232's canonical 8/8 pattern with nearly the same mean home AUPRC and mean home-versus-best-nonhome margin; no paired strict-phyloP versus #232 home endpoint survives an eight-endpoint Bonferroni threshold.
Every compared arm retained 40,960,000 sequence presentations, but effective exposure ranges from 0.312 to 9.666 row epochs because the post-augmentation datasets differ greatly in size; selector and historical comparisons are fixed-compute, not epoch-matched.
Across the four 0.25B full diagonals, home-specialist epoch/AUPRC correlations are exploratory at only four experiments per endpoint; distal is monotonically positive and synonymous monotonically negative, but neither association is robust to the eight-endpoint multiplicity burden.
The strict uniform-grid enhancer remains far below the targeted #326 and #351 enhancer specialists, while the current unassigned-background arm has much more splicing signal than #232's background and should not be interpreted as the same negative control.
All six strict-control runs and development-only VEP evaluations are complete.
Held-out even-autosome/Y evaluation remains unapproved and untouched.
The current follow-up tests whether a strict-phyloP Arm A enhancer corpus with exactly one sequence source per represented vertebrate order improves distal VEP through greater effective exposure.
Human occupies the sole Primates slot; the post-hoc source subset therefore retains 39 non-human targets across 18 mammalian and 21 non-mammalian orders.

## Baseline

- Date: 2026-08-24.
- Code ref: [`4ee29ac6`](https://github.com/Open-Athena/marin-dna/tree/4ee29ac67c4b5399e75b4d66724941dc3e3d3434).
- Baseline result: 221 tests passed in 11.60 seconds with 289,196 KiB peak RSS.

## Hypothesis Queue

### Active

- `FAS-517-H2`: Repeating the strict-phyloP uniform Arm A enhancer corpus more often by retaining one sequence source per vertebrate order will improve development distal AUPRC over the family-deduplicated strict baseline of 0.135.
  Next test: audit and publish the 40-source corpus, train the matched 0.25B enhancer arm for 5,000 steps, and compare every 500-step checkpoint with the strict family baseline.
- `FAS-517-P2`: The mapped home arm reaches the #459 persistence threshold during training.
  Next test: apply the two-consecutive-checkpoint `P(home ranks first) >= 95%` readout to the order-control enhancer trajectory.

### Blocked

None.

### Falsified / Dead End

- `FAS-517-B1`: Direct MultiZ MAF scanning for the issue-specific full projection.
  Why stopped: The 24 compressed MAFs total 74,694,939,245 bytes, while complete version-matched UCSC chains for the same 28 targets total 279,346,460 bytes.
  Evidence: `FAS-517-007` and `FAS-517-008`.

### Promoted

- `FAS-517-P1`: The terminal strict-phyloP matrix achieved 8/8 mapped home-arm wins on the development Mendelian subsets.
  Evidence: `FAS-517-055`.
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
- 2026-09-04: Test enhancer repetition with one sequence source per NCBI order across the entire dataset.
  Human is the sole Primates source; retain 39 non-human projection targets spanning 18 mammalian and 21 non-mammalian orders.
- 2026-09-04: Derive the order control post hoc from the immutable strict-phyloP Arm A center-1 projection table.
  Do not recompute scoring or projection.
  Preserve the 0.25B model, seed 0, global batch 8,192, 5,000 steps, and 500-step checkpoint cadence.

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

### 2026-08-26 00:56 UTC - `FAS-517-022` six-arm GPN training authorization and publication design

- Human decision: Proceed with training the new uniform-grid GPN-Star-P experiment.
  The experiment contains the six exhaustive arms selected in `FAS-517-018`: CDS, 3-prime UTR, protein-coding TSS/5-prime UTR, ncRNA exon, issue-326 Arm A enhancer, and the GPN-constrained remainder background.
- Fixed training contract: Reuse the issue-232-compatible Qwen3-like 0.25B model, character-plus-BOS tokenizer, source-case-aware loss, optimizer, seed 0, 5,000 steps, batch 8,192, and 500-step native/Hugging Face checkpoint cadence.
  Each arm receives 40,960,000 sequence presentations, or 10,485,760,000 tokens including BOS.
- Input boundary: Training remains public-Hugging-Face-only at immutable revisions.
  The authoritative source stays at the audited workflow-owned S3 projection prefix produced by `65b7806ea56a270124c9973af0366f5ab412c665` with config SHA-256 `28cb7786197945ef1798c3581873e4b3d68b7bf91189a59585b0dcabcad7a5e4`.
- Publication design: A separate additive `gpn_star_publication` workflow reads that immutable 167,607,189-row projection table without scheduling projection rules.
  It creates one dataset per arm, selects 16,384 original-orientation validation rows with seed 517 before augmentation, reverse-complements training rows only, builds 64 train shards and one validation shard, validates every file and split manifest, then uploads through an explicit gated target.
- Expected exposure after the deterministic split and reverse-complement augmentation:

  | arm | source rows | training rows | effective row epochs |
  | --- | ---: | ---: | ---: |
  | CDS | 35,517,702 | 71,002,636 | 0.577 |
  | 3-prime UTR | 10,285,758 | 20,538,748 | 1.994 |
  | TSS/5-prime UTR | 7,341,817 | 14,650,866 | 2.796 |
  | ncRNA exon | 10,029,795 | 20,026,822 | 2.045 |
  | enhancer Arm A | 65,750,304 | 131,467,840 | 0.312 |
  | background | 38,681,813 | 77,330,858 | 0.530 |

- Interpretation constraint: Compute remains fixed across arms, so dataset exposure differs by almost ninefold between enhancer and TSS/5-prime UTR.
  Every training and evaluation summary must report these effective epochs; this experiment does not isolate region identity from exposure.
- Launch sequence: Commit the additive publication workflow, run its locked tests and dry-run on a dedicated EC2 worker, materialize and validate all six public artifacts, publish and verify immutable Hub revisions, then launch one CDS canary on Iris before the other five independently resumable arms.
- Next action: Snapshot the publication workflow and start the EC2 test/dry-run gate.

### 2026-08-26 02:02 UTC - `FAS-517-023` six public GPN datasets and training launch gate

- Materialization: The additive publication workflow completed all 417 EC2 jobs under publication producer commit `1e057b7ee3a22ffe4c07df948fed9a440994b756` and config SHA-256 `6d8c12803f95d6a8c1fb577518119de2ff6081ed6b5ab11a643f921de444873d`.
  Runtime was 37 minutes 57 seconds, from 2026-08-26 01:04:42 UTC to 01:42:39 UTC on one `r6i.8xlarge` in `us-east-2`.
  The execution built the six deterministic splits, 390 compressed shards, six dataset cards, and one cross-arm integrity manifest without scheduling projection, GPN scoring, or `halLiftOver` rules.
- Publication: The explicitly gated upload target completed all six repositories and its aggregate target in 6 minutes 27 seconds, from 01:48:30 UTC to 01:54:57 UTC.
  Every arm has 64 compressed train shards, one compressed validation shard, one dataset card, and an upload receipt.
  The rule checked the local manifest before upload and then anonymously verified the exact Hub revision, file inventory, sizes, LFS SHA-256 values, and dataset-card hash.
- Immutable public inputs:

  | arm | public Hugging Face dataset | immutable revision |
  | --- | --- | --- |
  | CDS | `marin-dna/gpn-star-p-uniform-v1-cds` | `4c722c74e4616d8cbf8bce55844ec26da7fc516f` |
  | 3-prime UTR | `marin-dna/gpn-star-p-uniform-v1-utr3` | `42ac7aed4565d0ec2800c9d8e2b1829daec274bd` |
  | TSS/5-prime UTR | `marin-dna/gpn-star-p-uniform-v1-tss-utr5` | `c2fdcf05d24856f004be303470183e5fc39188b9` |
  | ncRNA exon | `marin-dna/gpn-star-p-uniform-v1-ncrna-exon` | `c5cea96abe3ae84dafdb52967b1168a269e01f43` |
  | enhancer Arm A | `marin-dna/gpn-star-p-uniform-v1-enhancer-arm-a` | `243210a0d93d93423b42e817d82d0abc3de37ef8` |
  | background | `marin-dna/gpn-star-p-uniform-v1-background` | `24f9ccb7cdc7c242d2ce88783e25db5597466543` |

- Training implementation: A separate `gpn_uniform_experiment` entry point pins these six public revisions and reuses the fixed issue-517 model, tokenizer, optimizer, seed, batch, sequence length, 5,000-step schedule, and 500-step checkpoint cadence.
  It rejects mutable or malformed revisions, non-MarinDNA repositories, and any S3 training dependency before graph construction.
  Each arm has its own token cache, W&B run, run ID, and checkpoint root.
- Validation: The publication workflow passed all 253 locked tests on EC2 under its configured Miniforge environment.
  The independently locked Python 3.12 training project passed all 12 tests on EC2, including all six pinned inputs, public-only graph provenance, fixed recipe, W&B tags, TPU bounds, and preemptible configuration.
- Launch decision: The human explicitly authorized public publication, preemptible TPU v5p-8 workers, and simultaneous launch of all six arms without a CDS canary gate.
  Checkpoint/resume remains enabled, and the almost ninefold cross-arm effective-epoch disparity from `FAS-517-022` remains a mandatory interpretation caveat.
- Evaluation boundary: No held-out VEP data was read or registered for this launch.
- Next action: Snapshot the tested pinned launcher, run its plan preflight, submit all six independent Iris jobs, and confirm that each coordinator accepts the immutable input graph.

### 2026-08-26 02:24 UTC - `FAS-517-024` graph preflight credential finding

- Preflight: The commit-pinned CDS plan resolved both expected artifacts and the exact immutable Hub revision without starting either step.
- Finding: Marin's `train_lm(..., env_vars=...)` includes that mapping in the durable artifact fingerprint.
  Passing `WANDB_API_KEY` through this mapping therefore serialized the credential into plan output, even though Sky had delivered it to the coordinator through a redacted secret channel.
- Containment: No Iris training job was submitted and no TPU was allocated.
  The launcher now requires the credential in the coordinator process but excludes its value from the artifact configuration; Fray's environment constructor inherits it only when the coordinator submits the TPU worker.
  A regression assertion rejects the credential name and test value in the resolved training pod.
- Authentication gate: The first credential-free Iris presence probe reached the Marin controller but was rejected because the EC2 host lacked application-default IAP credentials.
  Resolve the already-authorized unattended Iris authentication path before rerunning the sanitized graph preflight.

### 2026-08-26 11:48 UTC - `FAS-517-025` six-arm Iris launch

- Credential decision: The human explicitly authorized the established launch path in which `WANDB_API_KEY` is passed to Iris and may be visible in Iris job metadata.
  The credential remains excluded from the `train_lm` artifact configuration and fingerprint, and it was not printed in the launcher output.
- Authentication: The previously authorized cached Marin IAP credential was copied from the `aws-claude-code` SSH host to the dedicated EC2 launcher with mode `0600`.
  The launcher then reached the Marin controller successfully.
- Snapshot: All launches use tested and pushed MarinDNA commit `c2bfe6e53d6e1b792683371d83a8ca75da2df81b` and the six immutable public Hugging Face revisions recorded in `FAS-517-023`.
- Launches: Six independent Iris coordinators were accepted between 11:42:48 and 11:43:05 UTC:
  `/ubuntu/exp517-gpn-uniform-cds`, `/ubuntu/exp517-gpn-uniform-utr3`, `/ubuntu/exp517-gpn-uniform-tss-utr5`, `/ubuntu/exp517-gpn-uniform-ncrna-exon`, `/ubuntu/exp517-gpn-uniform-enhancer-arm-a`, and `/ubuntu/exp517-gpn-uniform-background`.
- Runtime contract: Each arm retains the Qwen3-like 0.25B model, sequence length 256, batch 8,192, 5,000 steps, seed 0, and 500-step checkpoint cadence.
  Training workers request preemptible TPU v5p-8 capacity in `us-east5`; no CPU training fallback is configured.
- Initial health: At 11:47 UTC all six coordinators and all six tokenization subtrees were running with no reported failures or preemptions.
  The CDS tokenization worker group had 64 tasks, with 59 running and five building at the sampled instant.
  This confirms graph construction and child scheduling, but TPU training and W&B telemetry had not yet started because tokenization was still in progress.
- Interpretation and evaluation: Each arm still receives approximately 10.486 billion token presentations, while effective row exposure differs by almost ninefold across arms.
  No held-out VEP data was read or registered.
- Next action: Monitor tokenization completion, verify each preemptible v5p-8 training descendant and its W&B telemetry, publish the launch status to issue #517, and terminate the temporary EC2 launcher once the independent Iris jobs no longer depend on it.

### 2026-08-26 12:25 UTC - `FAS-517-026` flexible v5p-8/v6e-4 relaunch

- Capacity finding: All six initial graphs completed and validated their train and validation token caches, then submitted v5p-8 training descendants.
  The `tpu_v5p-preemptible_8-us-east5-a` scale group entered degraded backoff with no ready slices and repeated provisioning failures, leaving all six training tasks pending without W&B runs.
- Prior-work check: [Issue #303](https://github.com/Open-Athena/marin-dna/issues/303#issuecomment-4653647945) records successful 0.25B, batch-8,192, sequence-length-256 training on `v6e-4` with PDP 1,024 in `us-east5`, including live TPU loops and W&B for both arms.
  [Issue #351](https://github.com/Open-Athena/marin-dna/issues/351#issuecomment-4871479878) records the same v6e-4/PDP-1,024 route for two 0.25B enhancer runs, and the issue's final record reports both completed all 5,000 steps.
- Human decision: Allow the historically used `v6e-4` TPU in addition to the original `v5p-8` request.
  The exact ordered alternatives are `v5p-8,v6e-4` in `us-east5`, with preemptible capacity retained.
- Validation: The existing EC2-side `test_tpu_resource_overrides_are_bounded` test passed for this exact variant pair.
  The model, optimizer, seed, batch, sequence length, step budget, checkpoint cadence, artifact paths, and W&B run IDs are unchanged.
- Replacement: After an exact six-job cancellation dry-run, the six v5p-only coordinators were deliberately cancelled before any optimizer step.
  Their durable token caches were preserved.
  Six replacements named `/ubuntu/exp517-gpn-uniform-{cds,utr3,tss-utr5,ncrna-exon,enhancer-arm-a,background}-flex` were accepted between 12:18:11 and 12:18:29 UTC and reused the completed caches without new tokenization.
- Initial flexible scheduling: By 12:24 UTC, CDS, 3-prime UTR, TSS/5-prime UTR, and ncRNA exon were running on four distinct physical `tpu_v6e-preemptible_4-us-east5-b` workers.
  Their Levanter loops reached `Progress on:train -/5000`, and four W&B runs were created in the intended group; no metric step had been logged yet during first-step compilation.
  Enhancer Arm A and background remained pending for available slices.
- Next action: Confirm real optimizer steps and metrics on the four active arms, verify enhancer Arm A and background allocation, publish the scheduling change and prior-work links to issue #517, then stop the temporary EC2 launcher.

### 2026-08-26 12:35 UTC - `FAS-517-027` first optimizer steps on v6e-4

- Runtime verification: The four allocated arms crossed compilation into real optimizer work on physical `tpu_v6e-preemptible_4-us-east5-b` workers.
  At 12:33 UTC, W&B reported CDS at step 58, 3-prime UTR at step 60, TSS/5-prime UTR at step 59, and ncRNA exon at step 55, with all four runs in `running` state.
- Queue state: Enhancer Arm A and background remain accepted and pending rather than failed.
  Their scheduler reason is insufficient immediately available TPUs; at 12:34 UTC, the v6e-4 `us-east5-b` scale group reported 17 ready and nine booting workers, while the v5p-8 `us-east5-a` group reported two booting and none ready.
- Interpretation: The previous issue records correctly predicted compatibility for this exact training geometry, and the physical worker assignments demonstrate that Iris selected the ordered v6e-4 alternative while v5p capacity was unavailable.
- Next action: Publish the verified first-step status and queue state to issue #517, leave all six independently resumable training graphs active, and stop the temporary EC2 launcher.

### 2026-08-26 13:44 UTC - `FAS-517-028` all six arms allocated

- Allocation: All six flexible training children are now in Iris `running` state on TPU v6e workers.
  Enhancer Arm A and background left the capacity queue after the 12:35 UTC snapshot.
- W&B snapshot: CDS was at step 712, 3-prime UTR at 733, TSS/5-prime UTR at 726, ncRNA exon at 719, background at step 3, and enhancer Arm A was still compiling its first step after rescheduling.
  Across the four established loops, current training loss was 1.286-1.323, evaluation loss was 1.299-1.354, and throughput was approximately 452,000-459,000 tokens per second.
- Durability: CDS, 3-prime UTR, TSS/5-prime UTR, and ncRNA exon each have a complete native step-500 manifest and exported Hugging Face checkpoint in the configured GCS root.
  Enhancer Arm A and background have not yet reached the first 500-step checkpoint boundary.
- Reliability: The CDS, 3-prime UTR, TSS/5-prime UTR, ncRNA exon, and background training children report zero failures and zero preemptions.
  Enhancer Arm A reports zero failures and one preemption by a higher-priority Iris job; Iris rescheduled it automatically and its replacement worker is running.
- Confidence: The five arms with metrics have finite losses and advancing global steps, and all six W&B runs identify their device as `TPU v6 lite`.
  Enhancer Arm A still needs an advancing W&B step after its rescheduled compilation before it has the same end-to-end verification.
- Next action: Confirm enhancer Arm A's first optimizer step, continue coarse monitoring through the 5,000-step boundary, and validate final native and Hugging Face checkpoints for every arm.

### 2026-08-26 13:46 UTC - `FAS-517-029` all six optimizer loops advancing

- First-step gate: Enhancer Arm A recovered from its single preemption and reached W&B step 5 at approximately 446,000 tokens per second.
  Background reached step 50 at approximately 453,000 tokens per second.
- Concurrent snapshot: CDS was at step 756, 3-prime UTR at 776, TSS/5-prime UTR at 770, and ncRNA exon at 767.
  All six W&B runs and all six Iris training children remained in `running` state.
- Interpretation: Every authorized arm now satisfies the end-to-end launch gate of a real TPU allocation, W&B initialization, and an advancing optimizer step.
  The enhancer loss at step 5 is too early to interpret; its first evaluation occurs later in the schedule.
- Next action: Monitor at coarse intervals, verify step-500 checkpoints for enhancer Arm A and background, and validate all six terminal step-4,999 exports before evaluation.

### 2026-08-26 17:15 UTC - `FAS-517-030` preemptions and automatic recovery queue

- W&B interruption snapshot: CDS last reported step 2,476; 3-prime UTR 2,459; TSS/5-prime UTR 2,436; ncRNA exon 2,441; enhancer Arm A 1,636; and background 1,584.
  W&B marked five runs `crashed` and left CDS `running`, but the Iris controller is authoritative for scheduling state.
- Iris state: All six training jobs remain active with zero failures and pending replacement workers.
  3-prime UTR, TSS/5-prime UTR, and ncRNA exon have one preemption each; CDS, enhancer Arm A, and background have two each.
- Recovery points: Native manifests and Hugging Face exports are complete at step 2,000 for CDS, 3-prime UTR, TSS/5-prime UTR, and ncRNA exon, and at step 1,500 for enhancer Arm A and background.
  The maximum replay from the last W&B step is therefore 476, 459, 436, 441, 136, and 84 steps respectively.
- Capacity: At 17:15 UTC, the controller was healthy with 738 of 738 workers healthy.
  The `tpu_v6e-preemptible_4-us-east5-b` group reported 16 booting and two ready workers; the `tpu_v5p-preemptible_8-us-east5-a` group reported two booting and none ready.
- Action: Do not submit duplicate jobs.
  The existing independently resumable Iris children are queued to restore their verified checkpoints when replacement TPU capacity becomes available.
- Monitoring path: No new EC2 launcher was created.
  Read-only W&B queries ran from the current VM, and Iris state was queried through authenticated GCP access to the controller.
  The former `aws-claude-code` IP presented a changed SSH host key and no longer resolved to an instance in the authorized AWS account, so the connection was rejected instead of disabling host verification.
- Next action: Confirm checkpoint restoration and advancing W&B steps after workers allocate, then revise the completion estimate from post-resume throughput.

### 2026-08-26 17:50 UTC - `FAS-517-031` first replacement worker and CDS restore

- Allocation: CDS moved from pending through building to running on replacement attempt 2 with zero failures and two recorded preemptions.
  The other five training tasks remain pending with their previous zero-failure preemption counts.
- Restore verification: CDS completed both native and Hugging Face step-2,500 checkpoints before its previous worker loss.
  The replacement worker restored 2.85 GiB across 48 arrays from TensorStore in 1.8 seconds and entered the 5,000-step training loop.
- First-step state: The CDS log showed `Progress on:train -/5000`, so checkpoint restoration is complete and first-step compilation is still in progress.
  W&B has not yet logged a post-restore step; its latest durable history is step 2,531 and currently displays `crashed` from the prior worker exit.
- Capacity: At 17:48 UTC, Iris reported 807 of 807 workers healthy.
  The v6e-4 `us-east5-b` group had 11 ready and seven booting workers against demand of 59, so allocation order and wait time for the remaining five are uncertain.
- ETA: CDS has approximately 2,500 optimizer steps remaining after restore, or about 3.2 hours at its prior steady-state throughput plus compilation and evaluation overhead.
  The other arms retain the previous estimate of approximately 4-5 hours after allocation; their wall-clock completion remains capacity-dependent.
- Next action: Confirm a post-restore CDS W&B step, continue waiting on the existing five queued jobs, and avoid duplicate submissions.

### 2026-08-26 17:51 UTC - `FAS-517-032` CDS W&B reconnect

- Telemetry: The existing CDS W&B run changed back to `running` under the same run ID after the replacement worker restored step 2,500.
  Its latest global step remains 2,531, so resumed telemetry is connected but a post-restore optimizer step has not yet been observed.
- Next action: Treat the restore as operationally healthy after W&B advances beyond step 2,531.

### 2026-08-26 19:16 UTC - `FAS-517-033` durable progress through repeated preemptions

- Current allocation: Background is running at W&B step 2,307.
  CDS, 3-prime UTR, TSS/5-prime UTR, ncRNA exon, and enhancer Arm A are pending replacement workers after additional short allocations.
- Latest W&B steps: CDS reached 2,581, 3-prime UTR 2,639, TSS/5-prime UTR 2,940, ncRNA exon 2,716, enhancer Arm A 1,915, and background 2,307.
  Current train and evaluation losses remain finite for all arms with evaluation history.
- Reliability: Every training child still reports zero failures.
  Preemption counts are four for CDS, three for enhancer Arm A, and two for each of 3-prime UTR, TSS/5-prime UTR, ncRNA exon, and background.
- Durable progress: Native and Hugging Face checkpoints are complete at step 2,500 for CDS, 3-prime UTR, TSS/5-prime UTR, and ncRNA exon; step 2,000 for background; and step 1,500 for enhancer Arm A.
  The leading four therefore gained a durable 500-step checkpoint since `FAS-517-030`, and background gained the same.
- Capacity: Iris is healthy with 670 of 670 workers healthy, but the v6e-4 `us-east5-b` pool reports one ready and 28 booting workers against demand of 69.
  The v5p-8 `us-east5-a` pool reports two booting and none ready.
- Interpretation: Preemptible capacity churn is delaying wall-clock completion, while checkpointing continues to preserve progress.
  There is no evidence of a model, data, checkpoint, or launcher failure.
- ETA: From durable restore points, the four leading arms need about 3.2 hours of optimizer work, background about 3.9 hours, and enhancer Arm A about 4.5 hours at prior steady-state throughput.
  Wall-clock completion remains those runtimes plus an uncertain capacity wait and any further replay.
- Next action: Keep the existing jobs queued, verify background's next step-2,500 checkpoint if its worker survives, and avoid duplicate submissions.

### 2026-08-26 20:20 UTC - `FAS-517-034` TPU request rationale and continued churn

- Current state: CDS is running at W&B step 2,854 with zero failures and six preemptions.
  3-prime UTR, TSS/5-prime UTR, ncRNA exon, enhancer Arm A, and background are pending replacement workers; all retain zero failures.
  Background last reached step 2,337 before its third preemption.
- Durable state: Restore checkpoints remain step 2,500 for CDS, 3-prime UTR, TSS/5-prime UTR, and ncRNA exon; step 2,000 for background; and step 1,500 for enhancer Arm A.
- Request contract: The jobs request preemptible TPU capacity in `us-east5` with ordered variants `v5p-8,v6e-4`, 56 GiB host RAM, fixed global batch 8,192, sequence length 256, and per-device parallelism 1,024.
  This is an enumerated compatibility set rather than a single v6e-only request.
- Compatibility evidence: Every observed allocation has selected `v6e-4` and exposed four `TPU v6 lite` devices in a `2x2x1` topology.
  The recipe compiles without OOM, restores its sharded checkpoints, sustains roughly 450,000 tokens per second outside transient evaluation samples, and reports mean MFU around 18%.
  Issues #303 and #351 previously completed the same 0.25B, batch-8,192, sequence-256 geometry on v6e-4 with per-device parallelism 1,024.
- Why enumerate variants: Iris capacity is organized by exact TPU generation, slice size, region, and preemptibility, while JAX compilation and batch partitioning depend on the resulting device mesh.
  Allowing an unvalidated arbitrary slice could change device count, per-device batch, compilation, cost, and checkpoint-restore behavior.
  Keeping the artifact bucket and TPU in `us-east5` also avoids cross-region checkpoint and input traffic.
- Optimality caveat: The evidence establishes that v6e-4 is compatible and reproducible; it does not establish that v6e-4 minimizes wall-clock time under current fleet contention.
  The supplied fleet overview shows every displayed TPU pool at 100% utilization, including v6e-4, v6e-8, v6e-16, v5p-8, and the v5e pools, so widening the request does not imply immediate capacity.
- Decision: Keep the six existing resumable jobs unchanged during this run.
  Before a future launch, test an expanded same-region alternative set such as v6e-8 or v6e-16 with an explicit device-mesh, global-batch, checkpoint-restore, throughput, and cost parity gate.
- Next action: Continue monitoring the current jobs and treat broader TPU-shape support as a separately validated execution optimization.

### 2026-08-26 22:55 UTC - `FAS-517-035` CDS passes step 4,000 during continued capacity churn

- Commands: Query the six live children with `iris job describe`; query the six deterministic W&B run IDs in `gonzalobenegas/marin`; list the native and HF step directories under `gs://marin-us-east5/MarinDNA/exp517_gpn_uniform_specialists/checkpoints/dna-exp517-gpn-uniform-0p25b-*/2026.08.26/{checkpoints,hf}/`.
- Current allocation: CDS is running on a v6e-4 worker at W&B step 4,164.
  3-prime UTR, TSS/5-prime UTR, ncRNA exon, enhancer Arm A, and background are pending replacement workers.
  All six Iris children still report zero failures.
- Reliability: Preemption counts are seven for CDS, three each for 3-prime UTR, TSS/5-prime UTR, and ncRNA exon, and four each for enhancer Arm A and background.
  The pending diagnostics cite TPU preemption or worker-reconcile failure thresholds, with no model or data error.
- Latest W&B steps: CDS 4,164; 3-prime UTR 3,005; TSS/5-prime UTR 2,940; ncRNA exon 2,720; enhancer Arm A 1,915; background 2,337.
  CDS reports train loss 1.1627, evaluation loss 1.1827, and approximately 451,000 tokens per second on `TPU v6 lite`.
- Durable progress: Matching native and HF-compatible checkpoints are complete at step 4,000 for CDS, step 3,000 for 3-prime UTR, step 2,500 for TSS/5-prime UTR and ncRNA exon, step 1,500 for enhancer Arm A, and step 2,000 for background.
- Capacity: The Iris controller is healthy with 71 of 71 workers healthy.
  Its sampled status lists two ready v6e-4 workers and no v5p-8 workers in `us-east5`; ready workers may already be occupied, and five issue 517 children remain pending in the authoritative task view.
- ETA: CDS needs about one hour of uninterrupted optimizer work to reach step 5,000 at its observed throughput.
  The other arms need approximately 2.6, 3.2, 3.2, 4.5, and 3.9 hours of optimizer work from their durable restore points, respectively, after allocation.
  Wall-clock completion remains capacity- and preemption-dependent.
- Next action: Keep the existing resumable jobs active, verify the CDS terminal native and HF exports, and continue coarse monitoring of the five queued workers without duplicate submissions.

### 2026-08-26 22:58 UTC - `FAS-517-036` Iris dashboard attribution correction

- Finding: The active issue 517 job paths begin with `/ubuntu/` because the EC2 launcher ran as the Linux user `ubuntu` and did not set Iris `--user` or `IRIS_USER`.
  Iris resolves a new top-level job user from an explicit override, then `IRIS_USER`, then the enclosing job, and then the operating-system user.
- Dashboard reconciliation: The `ubuntu` row's 12 active jobs are the six experiment coordinators and six training children.
  Its seven running and five pending tasks match six live coordinators, one allocated CDS worker, and five queued training workers at the sampled instant.
- Separate identity: `gonzalobenegas` is the W&B entity and does not control Iris quota attribution.
- Decision: Leave the active resumable jobs unchanged because Iris does not support in-place owner renaming.
  Set `--user gonzalo` or `IRIS_USER=gonzalo` explicitly on future top-level launches.

### 2026-08-27 02:27 UTC - `FAS-517-037` terminal-only VEP evaluation launch

- Training completion: The CDS Iris training child succeeded with zero failures and seven preemptions.
  Its terminal native and Hugging Face exports are complete at step 4,999; the Hugging Face export contains `config.json`, `model.safetensors`, `tokenizer.json`, and `tokenizer_config.json` under the configured GCS root.
- Evaluation registration: Commit `858f70bc` and draft PR [#529](https://github.com/Open-Athena/marin-dna/pull/529) register the six GPN-Star-P-filtered uniform-grid specialists at terminal step 4,999.
  The dedicated issue config reads only the pinned development `train` splits.
  Every specialist runs Mendelian Traits and Complex Traits, and CDS also runs the biologically scoped SGE evaluation.
- Validation: The focused workflow-config, model-registry, Sky-task, and GPU-runtime suite passed all 31 tests.
  The exact CDS dry-run resolved one terminal-checkpoint download, three score jobs, and three metric jobs.
  A broader project test run was stopped at 80% after slow failures across unrelated model tests; no focused launch-gate test failed.
- Launch: Three independent Sky clusters started for CDS Mendelian Traits, Complex Traits, and SGE.
  AWS spot A10G capacity was unavailable in all `us-east-2` zones, so the configured fallback allocated on-demand A10G workers.
  Every worker passed the pinned runtime smoke with NVIDIA A10G, driver 595.71.05, PyTorch 2.13.0, compiled CUDA 13.0, and executable bf16.
  The first cluster copied only the step-4,999 checkpoint into the workflow-owned S3 cache, and all three cells entered scoring.
- Remaining-arm trigger: A lightweight five-minute watcher on the current VM requires the complete four-file Hugging Face step-4,999 export before scheduling an arm.
  It launches Mendelian Traits immediately and Complex Traits six minutes later so the first cell can populate the shared model cache without a concurrent checkpoint write.
  Failed launches are retried, and each successful Sky job tears down its worker after completion.
- Evaluation boundary: No intermediate checkpoint and no held-out even-autosome or chromosome-Y VEP split is registered or read.
- Next action: Let the watcher launch the ten remaining development cells as their five training arms export step 4,999, monitor the three active CDS cells, and report verified metrics after all artifacts are durable.

### 2026-08-27 02:29 UTC - `FAS-517-038` CDS terminal VEP cells complete

- Completion: All three CDS Sky jobs returned `SUCCEEDED` and uploaded their score and metric Parquets to workflow-owned S3 storage.
- Mendelian Traits: Scoring covered 16,140 development records and completed at 02:16:43 UTC.
  Metric aggregation produced 66 rows and completed at 02:18:21 UTC.
- Complex Traits: Scoring covered 11,630 development records and completed at 02:20:01 UTC.
  Metric aggregation produced 60 rows and completed at 02:21:27 UTC.
- SGE: Scoring covered 23,853 development records and completed at 02:21:59 UTC.
  Metric aggregation produced 216 rows and completed at 02:25:47 UTC.
- Interpretation: This entry verifies execution and durable artifact production only.
  Cross-arm model interpretation waits for the remaining ten terminal-checkpoint cells.
- Next action: Keep the five-minute terminal-export watcher active and launch each remaining arm's two development cells when its complete step-4,999 export appears.

### 2026-08-27 05:24 UTC - `FAS-517-039` all terminal VEP cells complete

- Completion: The five remaining training arms exported complete terminal Hugging Face checkpoints at step 4,999.
  The five-minute watcher launched their Mendelian Traits and Complex Traits cells and exited after all ten evaluations completed.
  Together with CDS, all 13 authorized development cells now have durable score and metric Parquets under `s3://oa-bolinas/snakemake/analysis/evals_v2/results/`.
- Configuration: All values use the pinned `train` development splits, terminal step 4,999, strand-averaged LLR scores, and 1,000 bootstrap iterations with seed 0.
  No held-out VEP labels, predictions, or metrics were accessed.
- Target-aligned matched-trait AUPRCs (point estimate ± bootstrap SE):

  | Training arm | Mendelian Traits | Complex Traits |
  | --- | --- | --- |
  | CDS | Missense 0.339 ± 0.017; Splicing 0.400 ± 0.028; Synonymous 0.299 ± 0.061 | Missense 0.159 ± 0.015 |
  | 3′ UTR | 3′ UTR 0.185 ± 0.035 | 3′ UTR 0.195 ± 0.058 |
  | TSS / 5′ UTR | 5′ UTR 0.249 ± 0.023; Promoter 0.260 ± 0.027 | Promoter 0.149 ± 0.022; 5′ UTR 0.251 ± 0.064 |
  | ncRNA exon | ncRNA 0.384 ± 0.035 | ncRNA 0.162 ± 0.044 |
  | Enhancer arm A | Distal 0.119 ± 0.018 | Distal 0.104 ± 0.005 |
  | Background | Macro avg 0.133 ± 0.006 | Macro avg 0.113 ± 0.008 |

- SGE: The CDS accession-macro rows are 0.264 ± 0.008 for Missense across eight eligible accessions and 0.454 ± 0.021 for Splicing across six eligible accessions.
  `Macro`, `Both`, and other specialist-wide aggregates are omitted under the standalone reporting contract.
- Interpretation: The matched datasets use 1:9 positives:controls, so chance AUPRC is 0.10.
  CDS, ncRNA exon, and TSS/5′ UTR show the clearest target-aligned Mendelian signal.
  Enhancer arm A is close to chance on Distal in both matched benchmarks and needs a direct comparison with the prior uniform-grid enhancer baseline.
- Mature-miRNA check: The dedicated YAML declared `exclude_complete_match_groups_with_subsets`, but the current workflow does not consume that key and the raw Mendelian metric Parquets still contain the four mature-miRNA groups.
  None of the specialist rows above uses that subset.
  The background macro was independently checked over supported per-subset rows after excluding mature miRNA; its value and SE are unchanged because four groups do not pass the 30-group macro support gate.
- Public record: [Issue #517 update](https://github.com/Open-Athena/marin-dna/issues/517#issuecomment-5438753599).
- Next action: Compare these target-aligned cells with the established arm-specific baselines and fix or reject unsupported dataset-filter keys in the maintained evals_v2 configuration path.

### 2026-08-27 12:19 UTC - `FAS-517-040` terminal Mendelian diagonal

- Request: Reproduce the issue #232 final-step diagonal heatmap for the six GPN-Star-P uniform-grid arms.
- Commit Hash: `a83fa00c`.
- Inputs: The six terminal step-4,999 Mendelian metric Parquets under `s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics/exp517-gpn-uniform-*-step-4999/mendelian_traits.parquet`.
  The plot uses the development `train` split and `minus_llr_avg` with the existing 1,000-replicate match-group bootstrap SE.
- Figure: The eight Mendelian consequence subsets form rows and the six training arms form columns.
  Black outlines mark the prespecified arm-to-subset assignment, and green stars mark nominal row-max point estimates.
  Each cell displays AUPRC ± bootstrap SE.
- Result: The nominal diagonal is 7/8.
  CDS wins Missense, Synonymous, and Splicing; 3′ UTR wins 3′ UTR; ncRNA exon wins ncRNA; and TSS/5′ UTR wins 5′ UTR and Promoter.
  Distal is the exception: ncRNA exon is the nominal maximum at 0.138 ± 0.033, while Enhancer arm A is 0.119 ± 0.018.
- Interpretation: The Distal difference is small relative to the displayed uncertainty and has not been pairwise-tested.
  The figure supports a clear specialist diagonal for seven consequence rows and no clear Enhancer-arm advantage on Distal.
- Artifacts: `.agents/artifacts/issue-517/evaluation/issue517_mendelian_diagonal.svg`, its exact source-data CSV, and the reproducible plotting script live in commit `a83fa00c`.
- Next action: Publish the commit-pinned figure to issue #517 and compare the Distal row directly with issue #232 and the selected arm-A predecessor.

### 2026-08-27 13:02 UTC - `FAS-517-041` same-size regression audit

- Request: Compare the six terminal GPN-Star-P uniform-grid specialists with previous Qwen3-0.25B runs at the same 5,000-step by 8,192-token training budget and flag regressions.
- Research effort: Medium.
  The internal search covered issues #232, #326, #351, #473, the earlier annotation-first issue #517 runs, their logged configurations, and the registered development score and metric artifacts.
  No external search was needed because this is an internal experiment comparison.
- Evaluation boundary: Every paired result uses the pinned Mendelian Traits `train` development split, `minus_llr_avg`, and the biologically assigned home subset.
  Mature-miRNA rows, specialist-wide aggregates, background, and held-out labels are excluded.
  Background is not compared because the current GPN-constrained unassigned complement is not definition-compatible with the earlier background arms.
- Primary control: Issue #232 is the canonical six-arm uniform-grid control with the same Qwen3-0.25B geometry and terminal compute budget.
  A 1,000-replicate paired match-group bootstrap with seed 517 was run on an AWS `c6i.2xlarge` spot instance and the EC2 cluster was torn down after completion.

  | Home endpoint | Current | #232 | Delta | Paired 95% CI | Two-sided p | Verdict |
  | --- | ---: | ---: | ---: | ---: | ---: | --- |
  | CDS Missense | 0.339 | 0.309 | +0.029 | [+0.010, +0.049] | 0.002 | Improvement |
  | CDS Splicing | 0.400 | 0.395 | +0.005 | [-0.027, +0.039] | 0.716 | Inconclusive |
  | CDS Synonymous | 0.299 | 0.281 | +0.018 | [-0.065, +0.086] | 0.684 | Inconclusive |
  | 3′ UTR | 0.185 | 0.217 | -0.032 | [-0.075, +0.027] | 0.348 | Inconclusive |
  | ncRNA | 0.384 | 0.366 | +0.018 | [-0.031, +0.060] | 0.464 | Inconclusive |
  | 5′ UTR | 0.249 | 0.289 | -0.040 | [-0.074, -0.010] | 0.010 | Nominal regression |
  | Promoter | 0.260 | 0.259 | +0.001 | [-0.029, +0.028] | 0.970 | Inconclusive |
  | Distal | 0.119 | 0.127 | -0.008 | [-0.043, +0.012] | 0.508 | Inconclusive; both near chance |

- Multiplicity: The intervals and p-values are unadjusted across eight primary home endpoints.
  The 5′ UTR result is a regression candidate but does not pass a Bonferroni threshold of 0.00625; the CDS Missense improvement does.
- Earlier annotation-first issue #517 control: The current grid significantly improves all three CDS endpoints and 5′ UTR, is inconclusive on 3′ UTR and Promoter, and regresses on ncRNA by -0.168 AUPRC with paired 95% CI [-0.243, -0.095] and Distal by -0.204 with paired 95% CI [-0.307, -0.099].
  Both regressions have bootstrap p = 0.001 and survive an eight-endpoint Bonferroni threshold.
- Enhancer cross-checks: Current Distal AUPRC is 0.119 versus the issue #326 Arm A point estimate of 0.299, issue #351 tiled point estimate of 0.308, and issue #351 centered point estimate of 0.366.
  Those older values came from online `lm_eval` rather than registered offline score Parquets, so they are strong regression signals but not paired inferential comparisons.
- CDS cross-check: Against the issue #473 full-window CDS run, the current point estimates are higher on Mendelian Missense and Splicing and SGE Missense, and lower on Mendelian Synonymous, Complex Traits Missense, and SGE Splicing.
  None of the lower point differences is clearly separated using the reported marginal standard errors, and no paired #473 test was run.
- Exposure caveat: Current effective epochs are lower than issue #232 for every comparable arm, including TSS/5′ UTR at 2.80 versus 3.63 and Enhancer at 0.31 versus 0.47.
  The current Enhancer arm also uses 653,017 GPN-Star-P-filtered human anchors, while issue #326 Arm A used the same assignment logic with a phyloP-filtered cohort of roughly 370,000 windows and at least about 0.5 effective epoch.
  These exposure and selection differences are plausible explanations, not identified causes.
- Interpretation: The current experiment is not a uniform regression.
  It recovers CDS performance lost in the earlier annotation-first setup, retains the #232-level ncRNA and Promoter results, has a nominal 5′ UTR regression against #232, and loses the earlier ncRNA and Enhancer gains.
  Enhancer is the highest-priority regression because it is near chance, fails to own the Distal row, and is far below every same-size targeted predecessor.
- Negative result: No registered offline score Parquets were found for the issue #326 or #351 Enhancer runs, so a paired comparison with those predecessors is not currently possible.
- Artifacts: The exact paired JSON, 25-row comparison CSV, and reproducible EC2 Sky configuration are stored under `.agents/artifacts/issue-517/evaluation/`.
- Next action: Inspect the terminal Enhancer and TSS learning trajectories, then run the smallest controlled continuation or matched-exposure comparison that separates underexposure from the GPN-versus-phyloP window-selection change.

### 2026-08-27 13:21 UTC - `FAS-517-042` strict phyloP selector-control launch

- Human decision: Run a strict selector-only control of the six GPN-Star-P uniform-grid specialists using the historical phyloP window gate.
  Everything downstream of window eligibility stays fixed.
- Selector: Uniform GRCh38 255 bp windows at 128 bp stride are scored with `hg38.phyloP447way`; a base is selected at phyloP `>= 2.2162`, and a window is retained with at least 51 of 255 selected bases.
  The expected full catalog is 1,136,854 human windows.
- Fixed factors: The control uses the current center-1 projector, the same 107 non-human Zoonomia HAL mammals, 28 non-mammal MultiZ targets, human reference stream, six-arm assignment recipe `exp232-v4-plus-exp326-arm-a-remainder-v1`, Qwen3-like 0.25B recipe, seed 0, 5,000 by 8,192 schedule, tokenizer, optimizer, checkpoint cadence, and development VEP cells.
- Projection decision: Reproject mammals rather than reuse issue #232/#326 outputs.
  Those historical phyloP datasets used the old full-window projection contract, whereas the GPN experiment used the current center-1 contract; reuse would therefore confound selector and projector.
- Snapshot: Commit `2162b6aa8299a9748eeb8031318b49072bb8c3fc` adds the source workflow without modifying the existing GPN workflow.
- Remote validation: On AWS `c6id.12xlarge`, all 256 locked project tests passed; pinned Ruff, Snakefmt, YAML, and file checks passed; and smoke/full Snakemake dry-runs passed.
  The full DAG contains 1,708 rules, exactly 107 batched `halLiftOver` invocations, and all 28 non-mammal targets.
- Execution: Sky job 12 on cluster `issue517-phylop-control` started the real smoke tier from the immutable producer snapshot.
  Its outputs are under `s3://oa-bolinas/snakemake/vertebrate_projection_dataset/results/phylop-uniform-v1/2162b6aa8299a9748eeb8031318b49072bb8c3fc/5d9c798e656218acf8ea203023d71252223022c6010f70e508b9c43df83ef62e/smoke/`.
  The retained worker stages the 1.26 TB HAL once onto local instance-store RAID0 before the full tier.
- Public record: [Issue #517 strict-control update](https://github.com/Open-Athena/marin-dna/issues/517#issuecomment-5439739545).
- Next action: Audit the six-arm smoke projection, run the full projection on the retained worker, publish six immutable datasets, and launch all six matched preemptible TPU runs.

### 2026-08-27 15:56 UTC - `FAS-517-043` strict phyloP projection complete

- Execution: Sky job 13 completed the full source DAG on the retained AWS `c6id.12xlarge` worker: 1,705/1,705 execution steps with no errors.
  All 107 actual mammal `halLiftOver` invocations succeeded without retry; the measured HAL-only wall time was 46m37s from 14:20:17 to 15:06:54 UTC.
- Projection scope: 1,136,854 unique 255 bp human anchors across all 24 primary chromosomes, 107 non-human mammals, and 28 non-mammal MultiZ targets.
  Mammal and non-mammal fragments, contracts, and sequences all completed under the current center-1 projection contract.
- Assignment audit: The six arms are exhaustive and sum exactly to the anchor catalog.
  Anchor counts are 295,561 CDS; 67,155 3-prime UTR; 57,418 TSS/5-prime UTR; 98,630 ncRNA exon; 369,860 enhancer Arm A; and 248,230 background.
  Background consists of 57,220 cCRE windows rejected by Arm A plus 191,010 v4-background windows.
- Publication row counts: Including one human reference row per anchor, the combined source has 124,196,403 rows.
  Per-arm totals are 34,758,271 CDS; 7,264,712 3-prime UTR; 5,806,425 TSS/5-prime UTR; 10,411,649 ncRNA exon; 39,879,096 enhancer; and 26,076,250 background.
- Contract audit: For every anchor, accepted plus rejected projections equals all 135 requested non-human species; mammal plus non-mammal accepted counts equal the accepted total; recovered fractions agree to floating-point precision; and source spans are uniformly 255 bp.
  Rejections are limited to 29,839,169 `no_mapping`, 541,280 `target_window_out_of_bounds`, and 35,292 `target_chromosome_too_short` events.
- Manual sample: The deterministic 18-row sample covers all six arms and both projection backends, includes 11 reverse-strand rows, and has uniformly 255 bp target spans and sequences with valid IUPAC DNA.
  The reported 1/255 alignment coverage is expected because the projector maps the center nucleotide and extracts 127 bp of target context on each side.
- Durable source: `s3://oa-bolinas/snakemake/vertebrate_projection_dataset/results/phylop-uniform-v1/2162b6aa8299a9748eeb8031318b49072bb8c3fc/94d512050de327f96fda1105ce9c6ae5562944e402802516c7cde54795d8cdd1/full/`.
- Next action: Snapshot the source-pinned publication workflow, build and anonymously verify six immutable Hugging Face datasets, then launch all six matched preemptible TPU runs as Iris user `gonzalo`.

### 2026-08-27 17:00 UTC - `FAS-517-044` strict phyloP datasets published

- Publication snapshot: Commit `fbc8968b14415b2722e7bcc4afaf95051acd6638` pins the completed source producer, adds the six-arm publication workflow, and passed all 257 locked source-project tests plus a 417-job publication dry-run.
- Remote build: A dedicated AWS `r6i.8xlarge` completed all 417 build jobs and then all seven upload targets without error.
  Each arm contains 64 train shards and one validation shard, and the uploader verified every remote object against its local source.
- Anonymous audit: After removing the temporary EC2 Hugging Face token, an unauthenticated API read confirmed that all six repositories are public and each has the exact 67-file inventory: `.gitattributes`, `README.md`, 64 train shards, and one validation shard.
- Immutable inputs:
  - CDS: `marin-dna/phylop-uniform-v1-cds` at `452a5a3538f22630c3dea94d441ac30216bb28ea`.
  - 3-prime UTR: `marin-dna/phylop-uniform-v1-utr3` at `2b73d5d9ebda34a361536db5e3d2697b6a1b1d6c`.
  - TSS/5-prime UTR: `marin-dna/phylop-uniform-v1-tss-utr5` at `5134205d86cd03e7833843d99e947e43e7aa11ac`.
  - ncRNA exon: `marin-dna/phylop-uniform-v1-ncrna-exon` at `54667e7bb49505f463afc147676e880a30c11d89`.
  - Enhancer Arm A: `marin-dna/phylop-uniform-v1-enhancer-arm-a` at `6f879b3747330e2c92e1402ead55cda6621f50ff`.
  - Background: `marin-dna/phylop-uniform-v1-background` at `7d84519dccb4286622a14642a82a4f045d93a42c`.
- Training gate: The six-run launcher now pins the publication producer and exact Hub revisions while retaining the GPN experiment's model, optimizer, tokenizer, seed, batch, schedule, checkpoint cadence, ordered `v5p-8,v6e-4` preemptible request, and six region assignments.
- Next action: Validate and snapshot the pinned launcher, launch all six independent jobs as Iris user `gonzalo`, and verify coordinator acceptance, immutable input download, and child scheduling.

### 2026-08-27 17:25 UTC - `FAS-517-045` capacity-aware strict-control launch

- Rebase: At the human's request, the experiment branch was rebased cleanly onto `origin/main` commit `7cf936d97b7a92baccb82b147cb66dadf6d48503` before any training submission.
  The already-published pre-rebase producer is preserved by annotated tag `issue517-phylop-publication-v1`.
- Updated guidance: Rebased commit `cf41be91` applies the new TRC sweep operating contract, records the exact target grid and recovery policy, adds structured `tpu_region=us-east5` W&B lineage, and passed all 15 locked project tests on EC2.
- Durable state: Six logical trials and six regional runs are registered in the sweep SQLite database.
  Every dispatch intent and confirmation has an immutable, size- and SHA-256-verified backup under `gs://marin-us-east5/MarinDNA/exp517_phylop_uniform_specialists/sweep_state/`.
- Launches: Six independent coordinators were accepted between 17:17:21 and 17:23:05 UTC as `/gonzalo/exp517-phylop-uniform-{cds,utr3,tss-utr5,ncrna-exon,enhancer-arm-a,background}-alt-d001`.
  They use interactive priority, one regional replica, `us-east5`, and ordered preemptible child alternatives `v5p-8,v6e-4`; the 48-chip registry total is a submitted ceiling rather than current allocation.
- Initial health: At 17:24:31 UTC all six coordinators were running.
  CDS, 3-prime UTR, TSS/5-prime UTR, ncRNA exon, and enhancer Arm A had running tokenization subtrees; the background tokenizer was newly pending behind its running coordinator.
  No W&B run had registered and no TPU was yet allocated, which is expected before tokenization completes.
- Cleanup: The temporary EC2 W&B credential was removed and the completed publication cluster `issue517-phylop-hf` was terminated.
- Next action: Observe W&B first, verify each immutable token cache, actual TPU family, first optimizer progress, and sane telemetry, then keep the 30-minute capacity-aware heartbeat until all six terminal step-4,999 checkpoints exist.

### 2026-08-27 18:23 UTC - `FAS-517-046` single-H100 validation snapshot

- Human decision: Keep all six existing TPU workflows alive while validating one CDS arm on a single preemptible CoreWeave H100.
  Start with global batch 8,192 as one per-device microbatch and reduce to 4,096, 2,048, or 1,024 only after a verified H100 OOM.
- Commit Hash: `2f698bf9`.
- Isolation: The established TPU project and lock remain unchanged.
  A thin GPU-only `h100_smoke` launch environment pins the same Marin commit and resolves `torch==2.11.0+cu128`, CUDA JAX, and the Marin GPU dependency set without mixing the TPU/CPU PyTorch index into the H100 worker.
- Smoke scope: Sample 16,384 rows from the immutable CDS dataset revision `452a5a3538f22630c3dea94d441ac30216bb28ea`, tokenize to cluster-local S3, and run three optimizer steps at sequence length 256 with one H100, seed 0, and per-device parallelism 8,192.
- Scheduling: CPU tokenization and the one-H100 training child are both preemptible, batch-priority work pinned to the selected production peer `cw-us-east-02a` with `regions=[ANY_REGION]`.
  The latest capacity snapshot showed 175 of 256 H100s free there; `cw-rno2a` showed no free H100s.
- Validation: The pinned `uv 0.11.31` lock resolved 265 packages in 1.46 seconds.
  The CUDA selection dry-run chose `torch==2.11.0+cu128` and `torchvision==0.26.0+cu128` without installing CUDA packages locally.
  All 18 experiment tests passed in 4.27 seconds with 477,740 KiB peak RSS.
- Negative result: Adding a CUDA index directly to the universal TPU project lock caused `uv` to reject conflicting CPU and CUDA indexes for transitive `torch` requirements.
  The first remote preparation attempt also failed before testing because its image had `uv 0.12.6` instead of the repository-pinned `0.11.31`; the temporary AWS spot node was then terminated.
- Next action: Rebase the validated snapshot onto current `origin/main`, publish the branch, dispatch the CDS smoke directly from exe-codex, and verify actual H100 allocation plus advancing W&B optimizer steps before changing any TPU workflow.

### 2026-08-27 19:22 UTC - `FAS-517-047` H100 batch calibration

- Outcome at 8,192: The first real one-H100 train step failed with JAX `RESOURCE_EXHAUSTED`; XLA requested 203.65 GiB after rematerialization could not reduce the program below the H100 budget.
  This is the required verified OOM and authorizes the planned reduction to per-device parallelism 4,096 while global batch remains 8,192.
- Placement: Live capacity moved from full `cw-us-east-02a` to `cw-rno2a`, where repeated version-3 Iris snapshots reported at least 72 free H100s and as many as 192 during calibration.
  A CPU probe confirmed that RNO injects the same `s3://marin-us-east-02a/marin` artifact prefix, so the completed tokenization remains reusable.
- Launcher calibration: RNO's task image has uv 0.10.3, while the repository pins 0.11.31.
  CPU-only preflights established the reproducible path: disable Iris auto-sync, install uv 0.11.31 as an isolated tool, address the nested project at its full bundled path, and run from the experiment root.
- GPU validation: Dispatch `cds-h100-pdp4096-smoke-d005` allocated one preemptible batch H100 and installed the CUDA environment, but failed before compilation because the reused tokenized-cache record serialized its tokenizer as the relative string `tokenizer`.
  The training child therefore looked for `/app/tokenizer` and fell back to a nonexistent Hugging Face repository.
- Fix and validation: The H100 wrapper now overrides the data tokenizer with the already-proven absolute vendored tokenizer path.
  All 18 project tests pass, including an assertion that the child training config receives an absolute tokenizer path.
- Preservation: All six TPU workflows remain untouched throughout H100 calibration.
- Next action: Snapshot and publish the absolute-tokenizer fix, rerun the 4,096 three-step smoke, and require advancing W&B optimizer progress plus a reachable checkpoint before considering the H100 path validated.

### 2026-08-27 19:53 UTC - `FAS-517-048` single-H100 path validated

- 4,096 result: Dispatch `cds-h100-pdp4096-smoke-d006` reached the first compiled `jit__train_step` on one preemptible H100 and failed with JAX `RESOURCE_EXHAUSTED` while requesting 94.29 GiB.
  No optimizer step completed, so this is the second verified memory limit and authorizes the planned reduction to per-device parallelism 2,048.
- 2,048 placement: A fresh Iris version-3 capacity snapshot showed 208 of 512 H100s free on `cw-rno2a` and none free on `cw-us-east-02a`.
  Dispatch `cds-h100-pdp2048-smoke-d001` therefore used one batch-priority preemptible H100 on RNO while retaining global batch 8,192, sequence length 256, seed 0, the immutable CDS dataset revision, and commit `103990e6`.
- Success: The H100 child completed all 3/3 optimizer steps and both the child and coordinator exited successfully.
  The child ran for 7m22s; the first batch took 145.7s to load, and the three-step training progress reported 93.8s per step over 4m41s including initial compilation.
- Telemetry: W&B finalized normally with finite evaluation loss 7.61084 and bits per byte 2.20068.
  The run is `dna-exp517-phylop-uniform-0p25b-cds-h100-pdp2048-smoke-v1`.
- Checkpoint: Levanter saved native and Hugging Face step-2 checkpoints.
  A separate preemptible RNO CPU probe used virtual-host S3 addressing to HEAD `hf/step-2/model.safetensors`, confirming a 1,019,422,904-byte object written at 19:47:30 UTC.
- Persistence: The 2,048 trial is marked complete with a verified checkpoint in the sweep database, and each observation, terminal transition, and completion transaction has an immutable downloaded-and-SHA-256-verified GCS backup.
- Preservation: All six TPU workflows remain untouched.
- Next action: Use per-device parallelism 2,048 for any full one-H100 control launch; decide whether to run the full CDS arm first or expand immediately to the six-arm H100 sweep while retaining the TPU controls.

### 2026-08-27 20:49 UTC - `FAS-517-049` full CDS H100 handoff armed

- Human decision: Proceed from the validated three-step smoke to the complete strict-phyloP CDS arm on one preemptible H100 while leaving all six TPU workflows untouched.
  The full scientific contract remains global batch 8,192, per-device parallelism 2,048, sequence length 256, seed 0, and 5,000 optimizer steps over immutable dataset revision `452a5a3538f22630c3dea94d441ac30216bb28ea`.
- Pre-GPU failures: Full dispatches `d001` and `d002` failed before any H100 allocation, W&B registration, or optimizer step because the bare nested CPU tokenizer image lacked `cloudpickle`.
  They are execution-setup failures rather than scientific training failures.
- Tokenizer correction: Commit `7e7f78ee` runs the maintained Marin tokenizer with 16 local Zephyr workers inside one explicitly sized preemptible CoreWeave CPU task and hides `IRIS_TASK_ID` only while that local pool runs.
  All 22 locked project tests pass, and live job `/gonzalo/exp517-phylop-cds-h100-full-tokenize-d002` shows 16/16 workers alive, zero dead workers, and no nested Iris worker jobs.
- Cache scale and ETA: The source is 10.73 GB across 64 train shards and one validation shard.
  The immutable training split contains 71,002,636 rows after reverse-complement augmentation, so the measured aggregate throughput implies approximately 1h50m for this one-time cache build and a current completion window near 22:15-22:30 UTC.
- Durable control state: Every preprocessing intent, accepted submission, terminal retry, and current active state has a new immutable SQLite backup under `gs://marin-us-east5/MarinDNA/exp517_phylop_uniform_specialists/sweep_state/coreweave-full-cds/`, downloaded and SHA-256 verified after upload.
- Capacity compatibility: Iris production peers now advertise availability schema version 3 while the pinned strict helper only accepts version 2.
  A direct v3 audit reconciled free plus held H100s to each fleet total and showed 232 of 512 free on `cw-rno2a` versus 4 of 256 on `cw-us-east-02a` at the latest sample.
- Event-triggered handoff: A no-polling watcher is armed on exe-codex.
  It requires preprocessing success, rejects a duplicate `d003`, validates a fresh internally consistent v3 snapshot for both eligible peers, selects the peer with more free H100s, persists and verifies each state transition, and then submits `/gonzalo/exp517-phylop-cds-h100-pdp2048-full-d003`.
- Public record: The issue records the successful 2,048 calibration, the healthy full-cache build, and a correction from the pre-augmentation to the post-augmentation row-count ETA.
- Next action: Confirm cache completion and accepted `d003` submission, then verify the actual one-H100 child, W&B progress, sustained throughput, and first full-run checkpoint before considering any TPU cancellation or additional H100 arm.

### 2026-08-28 00:22 UTC - `FAS-517-050` full CDS H100 retry at per-device parallelism 1,024

- Human decision: Retry the complete strict-phyloP CDS arm on one preemptible H100 with per-device parallelism 1,024, while preserving global batch 8,192, sequence length 256, seed 0, 5,000 optimizer steps, and all six existing TPU workflows.
- Prior full-run result: The three-step 2,048 smoke was not predictive of sustained memory use.
  The complete 2,048 run failed at optimizer step 5 when XLA requested another 52.65 GiB, so the H100 calibration moved to 1,024 without changing the scientific batch contract.
- Reproducible snapshot: Commit `55fef9e1` records the 1,024 configuration, distinct W&B and checkpoint identities, and updated operating notes.
  All 22 locked experiment tests passed in 4.17 seconds with 466,652 KiB peak RSS.
- Dispatch: Iris accepted `/gonzalo/exp517-phylop-cds-h100-pdp1024-full-d001` on `cw-rno2a` at batch priority with one preemptible H100.
  The training child is `/gonzalo/exp517-phylop-cds-h100-pdp1024-full-d001/run_levanter_train_lm-5adc1d44`.
- Data reuse: The run reused the completed immutable token cache at `s3://marin-us-east-02a/marin/h100/inputs/phylop-uniform-cds-char-bos/2026.08.27`; no repeat tokenization was required.
- Startup validation: The child reports exactly one H100, global batch 8,192, and per-device parallelism 1,024.
  W&B registered `dna-exp517-phylop-uniform-0p25b-cds-h100-pdp1024-v1`, the first batch loaded in 206.5 seconds, and the first compiled train step took 83.7 seconds.
- Progress: The run passed the prior step-5 failure point and reached step 30 with finite loss, zero Iris failures, and zero preemptions.
  Steps 9 through 20 took 131 seconds, or 11.9 seconds per optimizer step after startup.
- Preliminary ETA: Extrapolating that short steady-state interval gives approximately 16.5 hours for the remaining 4,980 steps.
  This is an early uninterrupted-runtime estimate and excludes checkpoint, evaluation, data-stall, and preemption overhead.
- Durable state: Dispatch intent, acceptance, and step-21 observation were persisted in the sweep database and uploaded as immutable, downloaded-and-SHA-256-verified snapshots under `gs://marin-us-east5/MarinDNA/exp517_phylop_uniform_specialists/sweep_state/coreweave-full-cds-pdp1024/`.
- Next action: Verify continued progress after the startup window and validate the first durable full-run checkpoint near step 500 before expanding the H100 path to another arm or changing any TPU workflow.

### 2026-08-28 00:39 UTC - `FAS-517-051` single-H100 1,024 memory limit

- Terminal result: W&B reached global step 59 with run progress 0.012, then the next `jit__train_step` failed with `RESOURCE_EXHAUSTED` while trying to allocate 27.36 GiB.
  Iris reports one failed task attempt, zero preemptions, and an 18m30s child duration, confirming a training OOM rather than lost capacity or scheduler failure.
- Throughput result: The final successful steps processed approximately 188,887-195,559 tokens per second and 738-764 examples per second.
  With global batch 8,192, that corresponds to approximately 10.9-11.1 seconds per optimizer step near steps 48-59.
- Interpretation: A single H100 is not sufficient for this exact full-run implementation at per-device parallelism 1,024, despite making substantially more progress than the 2,048 run.
  The earlier approximately 16.5-hour uninterrupted ETA described speed only and is invalidated by the delayed OOM.
- Checkpoint: The run stopped before the first planned step-500 checkpoint, so there is no durable training state to resume.
  The complete token cache remains reusable.
- Durable state: The final W&B observation, exact Iris failure, dispatch termination, and possible 512 follow-up were persisted transactionally.
  The terminal 73,728-byte SQLite snapshot was uploaded to `gs://marin-us-east5/MarinDNA/exp517_phylop_uniform_specialists/sweep_state/coreweave-full-cds-pdp1024/exp517_h100_full_cds_pdp1024_20260828T003833Z_terminal.sqlite`, downloaded, and verified at SHA-256 `3e0f6341a38ffe589b387b9f7f34b3ca9797deb28c276d6b558f0c5a741a07d8`.
- Preservation: All six TPU workflows remain untouched.
- Next action: Keep the single-H100 path paused pending a decision between a 512 memory-fit test and returning focus to the TPU controls.

### 2026-08-28 12:35 UTC - `FAS-517-052` four terminal controls and immediate VEP

- Training completion: CDS, 3-prime UTR, TSS/5-prime UTR, and enhancer Arm A reached optimizer step 4,999 and have reachable terminal Hugging Face `model.safetensors` checkpoints.
  Their final W&B evaluation losses are 1.08367, 0.92517, 0.85712, and 1.27926, respectively.
- Host-memory failures: ncRNA exon reached W&B step 4,984 and background reached step 4,998 before their coordinators exhausted host RAM while writing late checkpoints.
  Both retained durable step-4,500 checkpoints, so this is an execution failure rather than a change to the scientific control.
- Recovery: The corrected recovery roots are `/gonzalo/exp517-phylop-uniform-ncrna-exon-v6e-96g-d005` and `/gonzalo/exp517-phylop-uniform-background-v6e-96g-d003`.
  They preserve the exact dataset revision, training configuration, checkpoint root, and W&B identity while requesting preemptible `v6e-4` children with 96 GiB host RAM.
  At 12:35 UTC both coordinators were healthy and both children were pending because Iris reported zero of the required four TPU chips immediately available.
- VEP registration: PR [#536](https://github.com/Open-Athena/marin-dna/pull/536) registers the six strict-control terminal cells and a train-only issue configuration at commit `7eff7a121bd9ddf015a155863df72e04f43c1558`.
  The established evaluation metrics remain unchanged and exclude miRNA records internally.
- Immediate evaluation: SkyPilot job `exp517-phylop-vep-4arms` job 1 is running the nine available development-only cells on one AWS A10G.
  Spot A10G capacity was unavailable in the eligible `us-east-2` zones, so the checked-in fallback provisioned an on-demand `g5.xlarge` at $1.01 per hour; the runtime gate confirmed an NVIDIA A10G with bf16 support before scoring.
- Durable state: The current two-recovery sweep database was uploaded to `gs://marin-us-east5/MarinDNA/exp517_phylop_uniform_specialists/sweep_state/exp517_sweep_20260828T123057Z_two_recoveries_pending.sqlite`, downloaded, and verified at SHA-256 `a5ff50140fa8c826b3135e70752d5f740a2198eb69cabbf0eae09335aa3ddc0a`.
- Next action: Publish the nine available AUPRC results after completion, verify each recovered step-4,999 checkpoint, and trigger the ncRNA-exon and background VEP cells as soon as their terminal artifacts appear.

### 2026-08-28 13:00 UTC - `FAS-517-053` recovery tokenizer correction and paired audit

- Recovery diagnosis: ncRNA recovery child `d005` obtained a TPU slice but failed before checkpoint restore because its reused cache metadata exposed the historical relative tokenizer string `tokenizer` under the no-sync coordinator.
  The background `d003` child was cancelled while still pending, before it could consume a TPU slice and reproduce the same deterministic failure.
- Recovery correction: Commit `17af2bad` makes the TPU training wrapper replace the cache metadata with the vendored absolute tokenizer path before child submission.
  The full locked experiment suite passes on EC2 with 22 tests, including an assertion on the child training configuration.
- Current recovery: Corrected ncRNA `d006` and background `d004` coordinators are healthy, and both preemptible `v6e-4` children are pending in `us-east5` because Iris reports zero of four required TPU chips available.
  Both runs retain their original dataset revisions, checkpoint roots, W&B identities, optimizer state, and scientific configuration.
- Evaluation: The four completed-arm VEP workflow has produced all inputs needed for the same-row phyloP-versus-GPN audit while its remaining metrics finish.
  A 1,000-bootstrap paired comparison is running on a preemptible EC2 `c6i.2xlarge` in `us-east-2c`; the established metric implementation excludes miRNA internally.
- Next action: Publish the completed four-arm AUPRC and paired deltas, then evaluate ncRNA exon and background immediately after their terminal step-4,999 checkpoints are verified.

### 2026-08-28 13:29 UTC - `FAS-517-054` all six strict controls terminal

- Recovery completion: Corrected ncRNA `d006` and background `d004` children both used the absolute vendored tokenizer path, restored optimizer state at steps 4,886 and 4,908, and completed successfully on preemptible `v6e-4` slices.
- Terminal verification: W&B reports both runs `finished` at global step 4,999 with `run_progress=1`.
  Their evaluation losses are 0.92461 for ncRNA exon and 1.22600 for background.
  Each terminal Hugging Face export contains a 1,019,422,904-byte `model.safetensors` plus reachable configuration and tokenizer files.
- Sweep completion: All six logical trials and regional runs are now complete, with zero active dispatches, zero submitted TPU chips, and no unresolved persistence conditions.
- Durable state: The 118,784-byte terminal SQLite snapshot was uploaded to `gs://marin-us-east5/MarinDNA/exp517_phylop_uniform_specialists/sweep_state/exp517_sweep_20260828T132900Z_all_six_terminal.sqlite`, downloaded, and verified at SHA-256 `e0a9a51b0a4fe94d9c2264a3f9b348e6ea1b18bf93d7900da82e27df831029ec`.
- Immediate evaluation: Background VEP job 1 and ncRNA VEP job 2 were submitted on the same reviewed-commit A10G evaluator after artifact verification.
  The cluster obtained the approved on-demand fallback after spot capacity was unavailable, and the pinned A10G runtime parity gate passed before scoring.
- Next action: Complete the four remaining VEP cells, extract the full six-arm metrics, produce the issue-232-style diagonal, and publish the final selector-control comparison.

### 2026-08-28 13:46 UTC - `FAS-517-055` full six-arm VEP and selector-control audit

- Evaluation completion: The background and ncRNA-exon terminal checkpoints completed all five `evals_v2` steps on the reviewed evaluator commit `7eff7a121bd9ddf015a155863df72e04f43c1558`.
  The standard metric implementation excludes miRNA records internally; no additional post-hoc filtering was applied.
- ncRNA exact results: On Mendelian non-coding-transcript-exon variants, strict phyloP achieved AUPRC `0.330531 ± 0.034943` versus `0.383780 ± 0.034972` for the same-size GPN-selected specialist.
  The same-row paired bootstrap delta is `-0.053249` with 95% CI `[-0.098419, -0.009662]` and two-sided `p=0.012` across 115 variant groups.
  On the complex-trait counterpart, phyloP achieved `0.171269 ± 0.042174` versus GPN `0.161625 ± 0.044437`, with paired delta `+0.009644`, 95% CI `[-0.060828, 0.073639]`, and `p=0.712` across 37 groups.
- Full selector-control comparison: Across eight Mendelian and six complex-trait home-scope endpoints, the only nominal two-sided `p<0.05` differences are the ncRNA Mendelian decrease (`p=0.012`) and the enhancer Mendelian increase (`+0.016069`, `p=0.032`).
  Neither remains below a simple Bonferroni threshold for 14 exploratory endpoints, so the result does not support a broad selector-induced regression.
- Specialization sanity check: In the six-arm strict-phyloP Mendelian matrix, all eight rows are maximized by the matching established specialist: CDS for missense, synonymous, and splicing; 3-prime UTR for its row; ncRNA exon for its row; TSS/5-prime UTR for both 5-prime UTR and promoter; and enhancer Arm A for distal variants.
  The unassigned background arm is included as a control and wins no row.
- Artifacts: Exact same-size phyloP-versus-GPN metrics are in `issue517_phylop_vs_gpn_metrics.csv`; paired Mendelian and complex-trait results are in the corresponding JSON files; and the issue-232-style strict-phyloP diagonal is accompanied by its source-data CSV under `.agents/artifacts/issue-517/evaluation/`.
- Next action: Publish the exact artifacts and interpretation to issue #517, then terminate the completed evaluation clusters.

### 2026-08-28 14:11 UTC - `FAS-517-056` historical diagonal comparison

- Request: Compare the strict phyloP result with the earlier arm-diagonal experiments, in addition to the GPN-Star-P selector control.
- Research effort: Low.
  The internal pass covered issues #187, #232, #326, #351, the earlier annotation-first issue #517 matrix, their immutable evaluator registrations, and the canonical development metric and score artifacts.
  The stop rule was reached after the two exact full-matrix predecessors and the two targeted enhancer follow-ups were identified; external literature would not change this internal experiment comparison.
- Evaluation boundary: The exact matrices use the Mendelian Traits `train` development split, terminal step 4,999, `minus_llr_avg`, the same eight supported consequence subsets, and the established complete-group miRNA exclusion.
  No held-out labels, predictions, or metrics were read.
- Full-matrix result:

  | Experiment | Diagonal wins | Mean home AUPRC | Mean home margin | Minimum margin | Background wins |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | #232 uniform grid, phyloP, old full-window projector | 8/8 | 0.280374 | 0.118512 | +0.004817 | 0 |
  | Earlier #517 annotation-first, phyloP, center-1 | 6/8 | 0.271671 | 0.121861 | -0.079187 | no background arm |
  | #517 uniform grid, GPN-Star-P, center-1 | 7/8 | 0.279436 | 0.124137 | -0.018729 | 0 |
  | #517 uniform grid, phyloP, center-1 | 8/8 | 0.272529 | 0.117997 | +0.012588 | 0 |

- Pattern interpretation: Strict phyloP recovers the canonical #232 8/8 diagonal rather than merely improving on GPN's 7/8.
  Its mean home AUPRC is 0.007845 below #232 and its mean diagonal margin is only 0.000515 lower, so the complete diagonal is a structural recovery rather than a broad AUPRC improvement.
  Distal remains the weakest strict-phyloP margin at +0.012588, compared with +0.004817 in #232 and -0.018729 under GPN.
- Paired strict-phyloP versus #232: Missense is the only nominal difference, at +0.022380 with 95% CI [+0.002365, +0.040664] and `p=0.022`.
  The other seven home endpoints have `p>=0.076`; none of the eight comparisons passes the Bonferroni threshold of 0.00625.
- Paired strict-phyloP versus annotation-first: Strict uniform tiling improves splicing by +0.116277 (`p=0.001`), synonymous by +0.143021 (`p=0.004`), and 5-prime UTR by +0.082420 (`p=0.001`).
  It reduces ncRNA by -0.221071 (`p=0.001`) and distal by -0.187706 (`p=0.001`).
  These five differences pass the eight-endpoint Bonferroni threshold and show a real design tradeoff rather than a uniform ordering.
- Background readout: #232 background has mean AUPRC 0.109 across the eight rows and splicing AUPRC 0.099.
  The current GPN and strict-phyloP complements have mean row AUPRCs 0.133 and 0.140, with splicing at 0.244 and 0.230.
  Both still win zero rows, but the present complement includes cCRE-labelled windows rejected by Arm A and is not definition-compatible with #232's v4 background negative control.
- Earlier experiments: #187 reported 5/8 diagonal wins but used a 1B model, v3 labels, and PairwiseAccuracy, so only its qualitative win pattern is comparable.
  The targeted enhancer specialists reached distal point estimates of 0.299 for #326 Arm A, 0.272 for #326 Arm B, 0.308 for #351 tiled, and 0.366 for #351 centered, versus 0.135 for the strict-phyloP uniform Arm A.
  #326 and #351 used the historical in-training metric and have no registered offline score artifacts for a paired comparison; #351 also has the documented centered-versus-tiled epoch confound.
- Artifacts: The shared-scale four-panel SVG and PNG, exact 184-row source table, home-row margins, diagonal summary, paired JSON, plotting script, and remote Sky configuration are stored under `.agents/artifacts/issue-517/evaluation/`.
- Source ledger: [#187](https://github.com/Open-Athena/marin-dna/issues/187) supplies the qualitative v3/1B predecessor; [#232](https://github.com/Open-Athena/marin-dna/issues/232) supplies the canonical registered six-arm matrix; [#326](https://github.com/Open-Athena/marin-dna/issues/326) and [#351](https://github.com/Open-Athena/marin-dna/issues/351) supply targeted enhancer point references; issue #517 and canonical `evals_v2` artifacts supply both current matrices and the annotation-first predecessor.
- Next action: Publish the historical comparison and its definition caveats to issue #517, then terminate the completed audit node.

### 2026-08-28 15:24 UTC - `FAS-517-057` effective-epoch audit

- Request: Calculate the effective epochs for every full diagonal and targeted enhancer comparison in `FAS-517-056`.
- Definition: All retained terminal checkpoints represent 5,000 optimizer updates with global batch 8,192, or 40,960,000 sequence presentations per arm.
  Effective row epochs are `40,960,000 / post-augmentation training rows`.
  Sequence length is fixed at 256 tokens including BOS, so the row-epoch and token-epoch ratios are identical.
  Replay done by failed or preempted attempts is excluded because it is not retained in the terminal optimizer state.
- Full diagonal exposures:

  | Experiment | CDS | 3-prime UTR | ncRNA exon | TSS / 5-prime UTR | Enhancer | Background |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | #187 v3, 1B | 0.523 | 3.975 | 2.681 | 5.042 | 0.424 | 2.704 |
  | #232 v4 uniform | 0.711 | 3.254 | 2.510 | 3.631 | 0.465 | 1.076 |
  | Earlier #517 annotation-first | 0.874 | 3.604 | 6.596 | 5.405 | 1.615 | no arm |
  | #517 GPN-Star-P uniform | 0.577 | 1.994 | 2.045 | 2.796 | 0.312 | 0.530 |
  | #517 strict phyloP uniform | 0.589 | 2.825 | 1.970 | 3.537 | 0.514 | 0.786 |

- Targeted enhancer exposures:

  | Experiment | Arm | Training rows | Effective row epochs |
  | --- | --- | ---: | ---: |
  | #326 | Arm A, no exon overlap | 76,710,830 | 0.534 |
  | #326 | Arm B, no exon overlap plus enhancer-dominant | 62,734,288 | 0.653 |
  | #351 | Tiled, one-per-order | 10,992,626 | 3.726 |
  | #351 | Centered, one-per-order | 4,237,620 | 9.666 |

- Selector interpretation: Strict phyloP versus GPN is fixed-compute but not epoch-matched.
  Strict phyloP receives 2% more CDS epochs, 42% more 3-prime-UTR epochs, 4% fewer ncRNA epochs, 27% more TSS/5-prime-UTR epochs, 65% more enhancer epochs, and 48% more background epochs.
  The strict enhancer's nominal Mendelian improvement therefore coincides with substantially more repetition, whereas its ncRNA decline coincides with only 4% lower repetition.
- Historical interpretation: The earlier annotation-first ncRNA and enhancer arms received 3.35 and 3.14 times the strict-control epochs, respectively.
  Their much higher ncRNA and distal AUPRCs cannot be read as assignment-only effects.
  In contrast, #326 Arm A is close to the strict enhancer exposure at 0.534 versus 0.514 epochs, so its large distal gain is not explained by a large epoch advantage.
  #351 remains deliberately confounded at 3.726 versus 9.666 epochs.
- Correction: `FAS-517-049` and [issue comment 5445009834](https://github.com/Open-Athena/marin-dna/issues/517#issuecomment-5445009834) attributed 71,002,636 training rows to the strict-phyloP CDS H100 cache.
  That is the GPN-Star-P CDS count.
  The exact pinned strict-phyloP CDS dataset contains 69,483,774 training rows, yielding 0.589490 epochs rather than 0.576880.
  The H100 and TPU launch code pin `marin-dna/phylop-uniform-v1-cds@452a5a3538f22630c3dea94d441ac30216bb28ea`, so this was a reporting and ETA-denominator error, not a wrong-dataset training error.
- Artifacts: The exact 33-row audit is `issue517_historical_effective_epochs.csv`, generated by `compute_historical_effective_epochs.py` under `.agents/artifacts/issue-517/evaluation/`.
- Next action: Publish the epoch matrices, fixed-compute interpretation, and row-count correction to issue #517.

### 2026-08-28 15:34 UTC - `FAS-517-058` same-size epoch/performance correlation

- Human decision: Do not compare against issue #187 because its 1B model scale is incompatible with the current 0.25B experiments.
  The #187 rows were removed from the active epoch artifact and from both published comparison comments.
- Correlation scope: For each of the eight Mendelian variant types, pair the effective epochs of its matched home specialist with terminal home AUPRC across exactly four Qwen3-like 0.25B full diagonals: #232 v4, earlier issue #517 annotation-first, issue #517 GPN-Star-P uniform, and issue #517 strict-phyloP uniform.
  #326 and #351 remain enhancer point references but are excluded because they only cover distal, use the historical in-training metric rather than the registered offline score Parquets, and change projection/species scope.
- Statistics: Report Pearson `r`, Spearman `rho`, and two-sided exact permutation p-values over all 24 label permutations.
  With only four experiments per endpoint, p-values have coarse support and all correlations are exploratory.
  Leave-one-experiment-out Pearson signs are also recorded as a fragility check.
- Result:

  | Variant type | Home arm | Pearson r | Exact p | Spearman rho | Exact p | Leave-one-out sign stable |
  | --- | --- | ---: | ---: | ---: | ---: | --- |
  | Missense | CDS | -0.868 | 0.125 | -1.000 | 0.083 | yes |
  | Synonymous | CDS | -0.944 | 0.042 | -1.000 | 0.083 | yes |
  | Splicing | CDS | -0.861 | 0.125 | -0.800 | 0.333 | no |
  | 3-prime UTR | 3-prime UTR | -0.192 | 0.833 | -0.200 | 0.917 | no |
  | ncRNA | ncRNA exon | +0.976 | 0.125 | +0.800 | 0.333 | yes |
  | 5-prime UTR | TSS / 5-prime UTR | -0.755 | 0.333 | -0.200 | 0.917 | no |
  | Promoter | TSS / 5-prime UTR | -0.783 | 0.125 | -0.800 | 0.333 | yes |
  | Distal | Enhancer | +0.997 | 0.042 | +1.000 | 0.083 | yes |

- Interpretation: Distal has a perfectly monotonic positive rank ordering across the four experiments, and synonymous has a perfectly monotonic negative ordering.
  Both have nominal Pearson exact `p=0.0417`, but neither has Spearman exact `p<0.05`, and no endpoint passes the eight-test Bonferroni threshold of 0.00625.
  NcRNA is also strongly positive by point estimate, while most coding and promoter endpoints are negative and 3-prime UTR is near zero.
  The heterogeneous signs falsify a simple claim that more passes generally improve VEP and instead show that epoch count is entangled with selector, anchor construction, projection, and arm definition.
  In particular, the monotonic distal relation is hypothesis-generating rather than causal because the high-epoch annotation-first design is also the highest-AUPRC design.
- Artifacts: The 32 exact experiment-by-variant points are in `issue517_epoch_performance_home_points.csv`; the eight correlations, exact p-values, slopes, and leave-one-out ranges are in `issue517_epoch_performance_correlations.csv`; and `correlate_epochs_with_home_performance.py` reproduces the audit without external dependencies.
- Next action: Publish the same-size-only correlation table and its small-`n` caveat to issue #517.

### 2026-08-28 16:18 UTC - `FAS-517-059` direct HAL-to-chain investigation

- Question: Determine whether the 107 human-to-family-deduplicated-mammal coordinate mappings can be materialized from HAL and reused without exporting MAF or TAF.
- Human decision: Build reusable whole-genome human-to-species chains rather than a coordinate cache specialized to the current uniform grid.
  The ability to project arbitrary future tilings, anchor positions, window lengths, and annotations is the primary objective.
- Research effort: Medium.
  The search covered the official HAL representation, CLI, iterator, and unfinished chain sources; released Cactus 3.1.4 and 3.3.0 chain exporters; the UCSC chain specification; CAT's chaining precedent; the current issue #517 projection code and timing records; and issue #523.
- Result: Released Cactus provides `cactus-hal2chains`, whose MAF/TAF-free pipeline streams `halLiftover --outPSL` through `pslPosTarget`, `axtChain`, and gzip.
  This is a one-time whole-genome materialization step rather than a direct extraction of an already stored chain.
- Representation: HAL is a hierarchical alignment graph with ancestral and paralogy edges.
  UCSC chain is a scored pairwise block representation, so `axtChain` can introduce selection and grouping semantics beyond raw coordinate traversal.
- Version finding: The currently pinned Cactus 3.1.4 contains the converter but schedules a separate Toil job and full HAL copy per pair.
  Released Cactus 3.3.0 adds batch-level HAL sharing and concurrent pair pipelines, so any pilot should isolate and test 3.3.0 instead of silently changing the completed strict-control environment.
- Direction caveat: Cactus writes `target_vs_query.chain.gz`, with `target` on the UCSC `tName` or source side.
  A chain consumed as human-to-species liftOver must therefore put `Homo_sapiens` in `--targetGenomes`, put the destination species in `--queryGenomes`, and pass an explicit header-direction check.
- Scientific caveat: The production mammal projector uses `halLiftover --noDupes`, while the official chain converter does not pass `--noDupes`.
  Default chains are not assumed to be strict replacements because paralogous candidates and `axtChain` scoring may change mapping multiplicity or locus choice.
- Scaling finding: Direct HAL is already batched into one BED and one invocation per species.
  GPN projected 1,627,410 windows in 41m30s, whereas strict phyloP projected only 1,136,854 windows in 46m37s, so the prior 14.10-fold linear all-grid extrapolation is not empirically validated.
- Design ranking: Whole-genome pairwise chains are the selected reusable artifact for arbitrary future intervals.
  A coordinate-only cache of the 2,455,495,920 human-center/species requests could be more compact for the fixed grid but would lose the requested flexibility and is not the target deliverable.
- Recommended pilot: Generate official-default and `--noDupes` chain variants for `Papio_anubis`, `Mus_musculus`, and `Loxodonta_africana`; benchmark generation resources and chain bytes; compare every center's mapping state, target coordinate, strand, and multiplicity with direct HAL; time all-grid chain liftOver; and calculate the reuse break-even point.
- Negative results: MAF and TAF are unnecessary for chain production; HAL's apparent direct `hal2chain` binary is explicitly unfinished and untested; and the native multi-target column iterator is documented as an inefficient traversal, making a custom mapper a higher-risk follow-up rather than the first experiment.
- Artifact: `.agents/artifacts/issue-517/projection/hal_to_chain_investigation.md` records the complete rationale, source ledger, and bounded pilot.
- Public record: [Issue #517 decision and investigation](https://github.com/Open-Athena/marin-dna/issues/517#issuecomment-5454658109).
- Execution boundary: No cloud job was launched and no projection backend was changed.
- Next action: After launch authorization, implement the three-species EC2 parity and resource pilot under issue #523 as the gate to generating and pinning all 107 whole-genome chains.

### 2026-09-04 14:25 UTC - `FAS-517-060` whole-dataset order-control gate

- Hypothesis: Increasing effective enhancer exposure with one sequence source per represented vertebrate order may recover part of the gap between the strict uniform Arm A distal AUPRC of 0.135 and the targeted enhancer experiments.
- Human correction: The order constraint applies to the complete training dataset.
  Human occupies Primates, so every non-human primate projection is excluded.
  The pinned manifest contains 39 projection targets across 18 mammalian and 21 non-mammalian orders; adding human yields 40 sources and 40 orders.
- Source isolation: The new workflow reads the immutable strict-phyloP Arm A center-1 projection at source commit `2162b6aa8299a9748eeb8031318b49072bb8c3fc` and source config SHA-256 `94d512050de327f96fda1105ce9c6ae5562944e402802516c7cde54795d8cdd1`.
  Its DAG contains no scoring, HAL, chain, or MultiZ projection rule.
- Snapshot: `2cb84accd18a2e5934c88fb3828c2de6ecfd975a` adds the committed order manifest, isolated audit/split/publication path, EC2 runbook, and contract tests.
- Tests: `flock -n /tmp/exe-codex-local-heavy.lock env UV_CACHE_DIR=/tmp/issue517-uv-cache XDG_CACHE_HOME=/tmp/issue517-xdg-cache TMPDIR=/tmp POLARS_MAX_THREADS=2 RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 10 ionice -c 2 -n 7 /usr/bin/time -v /tmp/issue517-uv/uv run --locked pytest` from `snakemake/vertebrate_projection_dataset` passed 277 tests in 9.35 seconds with 301,852 KiB peak RSS.
- Dry-run: `/tmp/issue517-uv/uv run --locked snakemake -n all_phylop_order_hf_files --snakefile workflow/phylop_uniform_enhancer_order_publication.Snakefile --profile workflow/profiles/default --cores 32 --resources mem_mb=250000 final_large_scan=1 hf_uploads=1` resolved 73 jobs: one source audit, one split, one train-shard preparation, one validation-shard preparation, 65 compressions, one card, one producer manifest, one release-manifest validation, and the aggregate target.
- Publication gate: `source_rows` remains `-1` until the remote audit reports the exact eligible row count.
  Split, artifact validation, and upload rules fail closed while the count is unpinned.
- Evaluation boundary: Development VEP only.
  Held-out even-autosome/Y data remain untouched.
- Next action: Push the audit snapshot, post the corrected cohort decision to issue #517, run the one-rule EC2 source audit, and pin its exact count before publication.

### 2026-09-04 14:33 UTC - `FAS-517-061` order-control source audit

- Audit result: The immutable strict-phyloP enhancer table contributes 7,876,044 selected source rows under the whole-dataset one-per-order contract.
  Human contributes 369,860 rows and is the sole Primates source; the 39 non-human representatives contribute 7,506,184 rows across 18 mammalian and 21 non-mammalian orders.
- Exposure: Removing the fixed 16,384-row validation holdout and adding reverse complements yields 15,719,320 training rows.
  The fixed 40,960,000-sequence schedule therefore corresponds to 2.606 effective row epochs, 5.07 times the strict family-deduplicated enhancer baseline's 0.514 epochs.
- Durable evidence: The audit JSON is stored at `s3://oa-bolinas/snakemake/vertebrate_projection_dataset/results/phylop-uniform-enhancer-vertebrate-order-publication-v1/f00530cb4c045d66e681a92bb7169d23950731f8/1afb3dec505111775b91c2d75d5cbe0458eb050ec889ad25ee33935501149707/full/metadata/enhancer_order_source_audit.json`.
- Cost audit: The source Parquet is 9.90 GB, and the actual row audit took only seconds after download.
  The initial 256-GiB `r6i.8xlarge` was an inherited conservative publication envelope rather than an audit requirement, so it was terminated after the result reached S3.
  The pinned build recipe forces the existing bounded deterministic hash-sort path at 10 million rows and uses a 64-GiB `r6i.2xlarge`, eight cores, and 500-GB disk.
  This changes deterministic row order relative to the family baseline but preserves membership, sampling distribution, and exposure while reducing the worker's compute price to about one quarter of the initial audit node.
- Validation: `/tmp/issue517-uv/uv run --locked pytest` passed all 278 project tests in 9.07 seconds with 288,104 KiB peak RSS under the shared-node safety envelope.
  The S3-aware dry-run with eight cores, 60,000 MB, and one `final_large_scan` slot resolved exactly 73 intended jobs under configuration SHA-256 `a5d7ff16ecc2b4574e4803e4858392ffa00aefba17da1e155c4859355ad7b437`; it includes no scoring or projection rule.
- Next action: Validate and snapshot the pinned publication recipe, build and anonymously verify the public dataset, then rebase on `origin/main` before adding and launching the training configuration.

### 2026-09-04 14:50 UTC - `FAS-517-062` validated order-control publication build

- Execution: SkyPilot job 1 completed all 73 publication-build jobs on one AWS `r6i.2xlarge` in `us-east-2` under producer commit `90b86f6426c919470f0eb26e1b1aa2cab6a261ed` and configuration SHA-256 `a5d7ff16ecc2b4574e4803e4858392ffa00aefba17da1e155c4859355ad7b437`.
  Snakemake ran from 14:43:59 to 14:48:01 UTC, or 4 minutes 2 seconds.
- Hash-shuffle result: The 15,719,320-row bounded shuffle completed in 69 seconds.
  A live sample during the sort showed 4.5 GiB used and 56 GiB available out of 61 GiB RAM, with 465 GB of 485 GB disk free.
- Release manifest: The validated artifact contains 64 train shards with 15,719,320 rows and 2,416,541,878 compressed bytes, one validation shard with 16,384 rows and 2,505,322 compressed bytes, plus the 4,188-byte dataset card.
  Every shard has a recorded SHA-256 digest and reconciles to the source Parquet row counts.
- Durable evidence: The split summary and release manifest are stored under `s3://oa-bolinas/snakemake/vertebrate_projection_dataset/results/phylop-uniform-enhancer-vertebrate-order-publication-v1/90b86f6426c919470f0eb26e1b1aa2cab6a261ed/a5d7ff16ecc2b4574e4803e4858392ffa00aefba17da1e155c4859355ad7b437/full/`.
- Upload gate: The remote upload dry-run resolved exactly `phylop_order_hf_upload_dataset` plus its aggregate target.
  The real public upload was rejected before command submission because publishing this specific 2.42-GB payload to `marin-dna/phylop-uniform-v1-enhancer-arm-a-vertebrate-order` requires explicit user authorization.
  The worker is stopped, so compute charges have ended and the validated local shards remain on its persistent disk.
- Next action: Resume the stopped worker and upload only after explicit authorization for this public Hugging Face destination; then verify anonymously and rebase on `origin/main` before training configuration work.

### 2026-09-04 14:56 UTC - `FAS-517-063` public order-control dataset release

- Authorization: The user explicitly approved public publication of the validated 2.42-GB enhancer dataset to Hugging Face.
- Upload: SkyPilot job 3 ran only `all_phylop_order_hf` with `ALLOW_HF_UPLOAD=1` on the existing AWS publisher and succeeded at 14:55:25 UTC.
  The Hub committed all 66 release files at immutable revision `6a592fffcdd155d19e6c8e0986eab606aab19606`.
- Public artifact: [`marin-dna/phylop-uniform-v1-enhancer-arm-a-vertebrate-order`](https://huggingface.co/datasets/marin-dna/phylop-uniform-v1-enhancer-arm-a-vertebrate-order/tree/6a592fffcdd155d19e6c8e0986eab606aab19606).
  The repository is public and ungated under OpenMDW 1.1.
- Anonymous verification: Unauthenticated Hub API and download requests resolved the exact immutable revision.
  The remote tree contains all 65 expected data shards with 2,419,047,200 compressed bytes, no missing or extra data paths, and no size or LFS SHA-256 mismatches against the producer release manifest.
  The downloaded 4,188-byte README matches its manifest SHA-256, and the downloaded validation shard matches SHA-256 `d8765eb82cf546c38fa940187d503f55de8ee3dc44e03ec4ea97c12cd241f7b0`.
  Decompression yielded exactly 16,384 rows; a representative row had the expected schema, `enhancer` label, and 255-base sequence.
- Cost control: The temporary `issue-517-phylop-enhancer-order-hf` AWS cluster was terminated immediately after verification.
- Next action: Rebase the research branch onto `origin/main`, validate the rebased workflow, pin this Hugging Face revision in an isolated order-control training configuration, and launch the authorized preemptible 0.25B TPU run.

### 2026-09-04 16:31 UTC - `FAS-517-064` order-control training launch

- Hypothesis: Raising strict phyloP Arm A enhancer exposure from about 0.514 to 2.606 effective row epochs may recover part of the distal-enhancer AUPRC gap without changing the selected human windows, projection recipe, model size, batch size, or optimization schedule.
- Training snapshot: `bd8191b7305bcaa86fe1514a1a130e5e783c7792` adds the isolated order-control training configuration and pins public dataset revision `6a592fffcdd155d19e6c8e0986eab606aab19606`.
  `origin/main` was already an ancestor when the branch was rebased before this configuration was committed.
- Validation: `uv run --locked pytest` from `experiments/exp517_functional_specialists` passed all 25 tests with 465,580 KiB peak RSS.
  The plan resolved the intended Qwen-like 0.25B model, seed 0, global batch 8,192, per-device parallelism 1,024, 5,000 steps, and checkpoints every 500 steps.
- Launch: Iris accepted `/gonzalo/exp517-phylop-enhancer-order-d002` at 16:19:35 UTC for one `us-east5` preemptible TPU slice with ordered fallback `v5p-8,v6e-4`.
  The one-CPU coordinator is non-preemptible by design; the trainer child request is preemptible.
- Tokenization: The workflow found the exact 64 training shards and one validation shard from the immutable Hugging Face revision.
  It tokenized 15,719,320 training documents into 32 cache shards in 320.9 seconds, including 241.5 seconds of writes and 79.1 seconds of consolidation.
  Individual shard workers sustained about 0.79 million tokens per second, and all 32 shards completed without a recorded failure.
- Initial training state: The W&B run [`dna-exp517-phylop-uniform-0p25b-enhancer-order-v1`](https://wandb.ai/gonzalobenegas/marin/runs/dna-exp517-phylop-uniform-0p25b-enhancer-order-v1) entered `running` by 16:31 UTC.
  No optimizer step or loss had been reported at this observation, so TPU initialization remained in progress.
  Iris reported zero failures and zero preemptions.
- Durable monitor state: Immutable SQLite snapshots are stored under `gs://marin-us-east5/MarinDNA/exp517_phylop_enhancer_order/sweep_state/`.
  The 16:31:26 UTC snapshot has SHA-256 `75b333282b3d6e282b3ce5e6dfc7177f8240004be5dd035cdc267fc66822b52b` and was independently downloaded and verified.
- Next action: Confirm the allocated TPU family and first finite training metrics, then monitor W&B at 30-minute intervals.
  Run development VEP from the terminal checkpoint and compare it with GPN and prior same-size Arm A diagonal experiments; exclude 1B models.
