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

The additive Ensembl release 115 functional-anchor workflow passes all 235 locked pipeline tests and all three credential-free DAG checks.
The default target stops at a pending human preprojection audit; paid projection, publication, training, and held-out evaluation remain approval-gated.

## Baseline

- Date: 2026-08-24.
- Code ref: [`4ee29ac6`](https://github.com/Open-Athena/marin-dna/tree/4ee29ac67c4b5399e75b4d66724941dc3e3d3434).
- Baseline result: 221 tests passed in 11.60 seconds with 289,196 KiB peak RSS.

## Hypothesis Queue

### Active

- `FAS-517-P1`: At step 4,999, each mapped home arm ranks first on its eight development Mendelian subsets.
  Next test: blocked on approved projection, publication, and training after the human audit passes.
- `FAS-517-P2`: The mapped home arm reaches the #459 persistence threshold during training.
  Next test: blocked on approved projection, publication, and training after the human audit passes.

### Blocked

- `FAS-517-PROJECTION`: Full HAL and MultiZ projection requires explicit paid-compute approval.
  Resume when: the five human catalogs and pre-projection audit pass human review and projection is approved.
- `FAS-517-PUBLISH`: Hugging Face publication requires explicit approval after projection QC and card review.
  Resume when: projection QC passes and immutable cards are ready.
- `FAS-517-TRAIN`: The canary and remaining four training jobs require explicit paid-compute approval.
  Resume when: immutable datasets are published and launch accounting is reviewed.

### Falsified / Dead End

None.

### Promoted

- `FAS-517-H1`: The additive Ensembl builder reconciles feature extraction, priority ownership, tiling, stable identity, conservation subsets, and review artifacts before projection.
  Evidence: commit `731807af`, 235 locked tests, and the 17-job preprojection DAG check.

## Decision Log

- 2026-08-24: Extend `snakemake/vertebrate_projection_dataset` additively and leave the production uniform-anchor v2 path unchanged.
- 2026-08-24: Reuse the maintained center-1 projection contract downstream of a new five-arm projection catalog.
- 2026-08-24: Use the complete Ensembl GRCh38 release 115 GTF and all qualifying transcripts; do not use RefSeq or an Ensembl_canonical-only filter.
- 2026-08-24: Keep the existing approval gates for projection, publication, training, and held-out evaluation.

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
- Validation: 235 locked tests passed in 9.75 seconds with exit status 0 and 273,620 KiB peak RSS; all configured pre-commit hooks passed.
- DAG checks: The unchanged production smoke DAG resolved 79 jobs; the functional preprojection DAG resolved 17 jobs; the opt-in functional smoke projection DAG resolved 94 jobs. All checks used `--default-storage-provider none` and executed no jobs.
- Interpretation: `FAS-517-H1` passes locally. The implementation now reaches the intended human-audit gate with Ensembl as the explicit annotation source.
- Next action: Publish the branch and draft PR, review the published diff, then request human review of the preprojection artifacts before any paid projection.
