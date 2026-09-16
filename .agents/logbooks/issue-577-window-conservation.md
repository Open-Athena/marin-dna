---
topic: unary-window-conservation
issue: https://github.com/Open-Athena/marin-dna/issues/577
description: Species-prevalence scores for conserved-region selection with linear expected work.
author: user
---

# Unary window conservation: task logbook

## Scope

- Goal: one score per genomic window that enriches selected bases for conservation annotations while approaching O(species × windows) time and storage at fixed window length.
- User correction: homology pairs and clusters are not the target output or success metric.
- Primary endpoint: chromosome-2 phastCons100way annotated-base enrichment at a 5% selected-base budget, versus random and GC/repeat/entropy-matched selection.
- Boundaries: three real reference genomes; synthetic fixtures test computational scaling only; no billion-window production run or training.
- Authority: user's original $30 EC2 allowance plus explicit request to try the follow-up in this session.
- Prior cost: conservative estimate below $1.34, not an invoiced bill.

## Hypothesis queue

### Active

- PREVALENCE-001: species-deduplicated sampled k-mer support ranks conserved windows beyond composition and repeats.
- PREVALENCE-002: copy-number suppression improves specificity without erasing the conservation signal.
- PREVALENCE-003: aggregate counting and one scoring pass give expected linear work in total sequence, without pair enumeration or per-window global scans.

## Background research brief

- Effort: low.
- Date: 2026-09-16.
- Stop rule: the prior experiment, aggregate-counting complexity, and an external window-label source define a bounded falsifiable test.
- Internal evidence: #568 recovers known homologs with full sets but has no validated conservation-selector endpoint; its fixture is biased toward known-partner availability.
- Negative lead: the dense output arrays in prior exact/LSH implementations introduce quadratic all-window overhead; pair recall does not answer this follow-up.
- External context: Linclust's representative strategy can bound comparisons but had poor recall in #568; Progressive Cactus demonstrates hierarchical multi-genome alignment at 605 vertebrates but is a different, much larger compute project.
- Selected intervention: aggregate distinct-species counts of deterministic sampled canonical words, then score each window independently of labels.
- Evaluation source: UCSC hg38 phastConsElements100way, an alignment-derived annotation of conserved regions.
- Limitation: agreement with those annotations is not independent biological proof of constraint; equal species counts are not phylogenetic calibration.

| Source | Type | Claim used |
| --- | --- | --- |
| https://github.com/Open-Athena/marin-dna/issues/568 | Marin experiment | Prior signal, failed scaling path, and fixture limitations |
| https://www.nature.com/articles/s41467-018-04964-5 | Paper | Representative-based bounded-comparison precedent; protein evidence |
| https://www.nature.com/articles/s41586-020-2871-y | Paper | Hierarchical genome alignment at hundreds of species |
| https://genome.ucsc.edu/cgi-bin/hgTrackUi?c=chr21&db=hg38&g=cons100way | Official docs | Conservation label definition and scope |

## Entry log

### 2026-09-16 — PREVALENCE-001 protocol and worker

- User requested a bounded experiment, then explicitly clarified that conservation is scored per window.
- New coordinating issue #577 separates this intervention and endpoint from #568; PR #569 remains unchanged and open.
- Source project: `experiments/window_conservation/`; frozen protocol: `config/protocol.json`.
- Genomes: unchanged #568 hg38, GCF_000001635.26 mouse, GCF_000208655.1 armadillo 2bit assets with ETag/size/SHA checks.
- Sample canonical k17/k21/k25 at 1/64; use 4,096 bp output windows, chromosome 1 development and chromosome 2 held-out evaluation.
- All labels are excluded from construction and scoring.
- Worker: `i-0f0be03bac311291c`, c7i.4xlarge, us-east-2a, launched 2026-09-16T15:19:55Z.
- Root volume: `vol-0f242c96029f9f3b7`, encrypted 80 GiB gp3, DeleteOnTermination.
- Instance-initiated shutdown terminates the worker; confirmed scheduled shutdown 2026-09-16T23:20:12Z.
- Eight-hour compute estimate at the prior verified $0.714/hour is $5.712, plus approximately $0.12 disk/public IPv4; current price verification remains pending.
- Shared VM load was 0.00 with approximately 5.7 GiB available before work; all compilation, tests, dependency installs, and data processing run remotely in one foreground command at a time.
- Initial six native counting/scoring tests pass, including strand, ambiguous bases, species deduplication, copy counts, and independent enumeration checks.
- Genome preparation completed for 3,209,286,105 human, 2,818,974,548 mouse, and 3,631,505,655 armadillo bases; annotation download from the first UCSC host timed out.
- Review during implementation caught and removed an avoidable species-times-index-loading cost: batch scoring loads the aggregate index once for all species.
- Next: freeze source, complete annotation download, run all-window scores, select on chr1, publish selection, and evaluate chr2 once.

### 2026-09-16 — reuse the established phyloP bigWig before evaluation

- User directed reuse of conservation bigWigs already used by Marin pipelines.
- Replace the proposed phastCons-elements endpoint with the existing vertebrate-projection `phyloP_447m` definition, before inspecting any development or held-out conservation result.
- Source: `s3://oa-bolinas/staging/vertebrate_projection_dataset/v1/06549d8f7f3ba76151b9c54a5e52d3e3f4402a2d/full/anchors/phyloP_447m.bw`.
- Verified object: 10,022,801,463 bytes; ETag `43926355ad35c1a32f4238c7f4b394f4-1195`; last modified 2026-08-01T17:05:34Z.
- Existing definition: phyloP >= 2.2162, inclusive; NaN contributes zero to the conserved-base numerator and remains in the full-window denominator; retain finite coverage and mean phyloP separately.
- Primary metric remains conserved-base enrichment at the same fixed selection budget with the same covariate controls.
  Report the fraction of windows with at least 20% conserved bases as a secondary endpoint, using the pipeline's fraction cutoff at our larger window length.
- The earlier phastCons table was downloaded but never used for metrics or parameter selection; it is superseded and excluded from the final data owner.
- The complete-genome scoring run is independent of labels and continues unchanged.

### 2026-09-16 — local resolution correction before held-out evaluation

- User clarified that the target is conserved stretches at approximately 100 bp resolution, with short conserved exons/enhancers embedded in mostly neutral 4 kb windows.
- Supersede the 4 kb protocol with nonoverlapping 100 bp bins and deterministic 1/4 sampling (roughly 25 sampled word positions per complete valid bin, before deduplication).
- Only the old chr1 development result was inspected: k25/any selected, 1.81245× random and 2.22192× composition-matched conserved-base enrichment at 5%; this is exploratory history, not the local-resolution endpoint.
- Chr2 conservation labels remain uninspected.
- Start with the same three complete genomes for a fast iteration; the user explicitly permits deciding whether additional species are warranted.
- Query index retains words from complete human chr1/chr2 only, then counts species support and maximum within-species copies across all complete genomes.
  This reduces memory while preserving exactly the scores from a global index for those query chromosomes; an independent-enumeration test asserts equivalence.
- Keep the k17/k21/k25 and six-score grid, covariate controls, held-out split, and gate; use 200 genomic-block bootstrap replicates at the much larger bin count.
- Primary spatial artifact: 0-based half-open BED stretches formed by merging only immediately adjacent selected 100 bp bins at each fixed base budget.
- Expected cost for B total genome bases and Q query bases is O(B + Q), with O(U_query) working space; all-genome scoring uses O(U_global) space and is tested separately on synthetic data.
- Validation: remote locked pytest, 13 passed; ruff check and format passed; C++ warning-as-error build passed.
- All tests and analysis remain on the authorized EC2 worker; no child agents or concurrent analysis workers launched.

### 2026-09-16 — exact membership acceleration and validation

- The first dense query-counting attempt was interrupted after 549.429 seconds with exit 130; only this task's native process received SIGINT.
  Its command, timing, and stderr remain under `slow-query-prototype/` on the worker.
- Add a blocked Bloom membership prefilter before the exact query dictionary lookup.
  Every candidate that survives is still checked against the exact dictionary; false positives can add work but cannot add a word or alter a score.
- Avoid sorting the ordinary fixed-budget result: partition the score and deterministic tie key, then merge selected bins in genomic order.
  Bootstrap uncertainty still uses a cached ranking for weighted resampling; this is evaluation overhead, not the selector's algorithmic bound.
- Published implementation: `895163e9` on the permanent research branch.
- Validation on EC2: 15 locked tests passed in 1.66 seconds, ruff check/format passed, and the native C++ warning-as-error build passed.
- Public AWS Ohio price list effective 2026-09-01 confirms Linux shared c7i.4xlarge at $0.714/hour; the exact source row is retained in `report/aws-price.json`.
- First-minute startup check: approximately 3.7 GB resident memory and 26.9 GB available; one foreground analysis process, no added workers.
- No 100 bp development or held-out conservation metrics have yet been inspected.

### 2026-09-16 — freeze the 100 bp development selection

- Exact run: `uv run --locked python -m window_conservation.local_run --root /data/issue577`, followed by `uv run --locked python -m window_conservation.evaluate --root /data/issue577 --split dev`.
- Generation source: `895163e9`; final output-schema and archive changes are included in this selection snapshot.
- Three full-genome support scans completed for each k, covering 9,659,766,308 input bases; each setting scored 4,911,499 complete 100 bp query intervals across human chr1/chr2.
- Construction/scoring wall seconds: k17 628.739/59.749, k21 431.596/64.375, k25 421.364/65.351.
- On 2,304,742 eligible chromosome-1 intervals, the declared selection rule chose k25, `any`, at the 5% budget.
- Selected bases: 11,523,700; conserved-base fraction 0.13783585 versus 0.04000620 population and 0.02549140 matched control.
- Development enrichment: 3.44536× random and 5.40715× matched; annotated-base recall 0.17226798.
- Selected soft-mask fraction is 0.71975, so repeat-family confounding remains a material limitation despite composition matching.
- Three genomes show enough development signal to run the prespecified held-out test; additional genomes are deferred to keep this iteration bounded, not ruled out as useful.
- Final locked validation: 16 passed in 1.53 seconds, including localization of a planted 150 bp shared tract inside independent 5 kb backgrounds, exact query-index counts, partition selection, bigWig boundaries, and BED6+2 stretch merging.
- The selected k/score, complete development matrix, unchanged protocol hash, and input-manifest hash are committed before any chromosome-2 conservation labels are inspected.
- Next: one frozen chromosome-2 evaluation, the declared global-index synthetic resource matrix, output-contract audits, durable archive, and worker termination.

### 2026-09-16 — secondary density comparison declared before held-out labels

- The development matrix exposes a tradeoff in the original selection objective: k25/any maximizes matched enrichment, whereas k25/any_copy4 maximizes the absolute conserved-base fraction among selected bases.
- Declare the latter as a secondary, development-selected comparison before reading any chr2 labels; retain the original primary setting and gate.
- Development k25/any_copy4: selected fraction conserved 0.23662487, 5.91471× random, 5.26465× matched, and 0.29573502 annotated-base recall.
- Its selected soft-mask fraction is 0.49365, versus 0.71975 for the primary setting.
- The amended protocol explicitly records that this secondary comparison was added after inspecting development results.
- Both settings and all output budgets are frozen together; the held-out evaluation will produce both in one invocation without retuning.
- Validation still passes all 16 locked tests; report/audit tooling checks BED6+2 outputs for both settings.
