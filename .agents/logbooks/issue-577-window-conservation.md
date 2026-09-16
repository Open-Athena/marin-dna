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
- Endpoint: existing pipeline phyloP447way >= 2.2162 annotated-base density at a 5% selected-base budget, with random and GC/repeat/entropy-matched comparisons.
  The original chr2 pilot and the extension on fresh chr3 are frozen separately; chr1 supplies development selection.
- Boundaries: biological comparisons use 3/6/10 complete mammalian genomes; the global all-window trial uses the original three genomes.
  Synthetic fixtures test computational scaling only; no 1,000-complete-genome production run or training.
- Authority: user's original $30 EC2 allowance plus explicit request to try the follow-up in this session.
- Prior cost: conservative estimate below $1.34, not an invoiced bill.

## Hypothesis queue

### Supported within the tested scope

- PREVALENCE-001: species-deduplicated word support enriches conserved bases on held-out human chromosomes, including after coarse composition matching.
- PREVALENCE-002: copy suppression improves absolute density in the original three-genome panel; it does not eliminate all repeats or establish functional constraint.
- PREVALENCE-003: aggregate counting avoids pair enumeration; synthetic timing follows approximately linear input growth, with global real-genome resource checks pending.
- PREVALENCE-004: exact compact storage preserves all 4,911,499 pilot query scores and lowers peak RSS from 5.77 to 3.01 GiB.
- PREVALENCE-005: ten-species development selection improves fresh-chr3 density from 20.14% to 27.69%, with a paired gain interval of +7.11 to +8.04 percentage points.

### Active

- PREVALENCE-006: thinner sampling trades lower density for smaller indexes; standalone footprints and disk-partitioned global measurements will determine practical tradeoffs.
- PREVALENCE-007: describe fixed selections by functional annotations and repeat class/family without retuning them.
- PREVALENCE-008: measure local evidence under planted tract length, divergence, indels, offset, duplication, and shuffled negatives.

The scope and queue above are living summaries; the entry log below preserves the chronological record.

## Initial background research brief (superseded label proposal)

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

### 2026-09-16 — frozen chromosome-2 conservation result

- Evaluation: `uv run --locked python -m window_conservation.evaluate --root /data/issue577 --split heldout --freeze-sha a9a641a1ae8fbc126e4d8be12889ff562fc5f734`.
- One invocation evaluated both frozen choices; runtime 75.75 seconds and peak RSS 692,228 KiB.
- Eligible chr2 intervals: 2,405,464; 5% selection contains 120,273 intervals / 12,027,300 bases.
- Primary k25/any: 14.5902% conserved bases, 3.53310× random (95% CI 3.26899–3.82294), 5.68280× matched (5.42107–5.87283), and 17.6655% conserved-base recall.
- Secondary k25/any_copy4: 23.3865% conserved bases, 5.66321× random (5.39075–5.94940), 5.53423× matched (5.34515–5.66324), and 28.3160% conserved-base recall.
- Population conserved-base fraction: 4.12956%; simple low-repeat, high-entropy, and high-GC selectors obtain 8.15555%, 3.61662%, and 5.41432% at the same 5% budget.
- At the prespecified secondary 1% budget, copy-filtered selection contains 60.0786% conserved bases and recovers 14.5480% of all annotated conserved bases.
  This is a precision–coverage tradeoff, not a new held-out tuning decision.
- The original biological gate passes.
- Interpretation: three genomes suffice for measurable local conservation enrichment; copy filtering improves absolute conserved-base density in this panel.
  This does not establish a calibrated conservation probability, training benefit, or generalization to 1,000 real genomes.
- Remaining work: finish the 21 synthetic timing runs, inspect figures and output contracts, archive, terminate the worker, and deliver the interpretation for review.

### 2026-09-16 — resource scaling and spatial-output audit complete

- Scaling command: `uv run --locked python -m window_conservation.scaling --root /data/issue577 --k 25`.
- All 21 runs completed: 7 shapes, 3 repetitions, with species counts 125/250/500/1,000 at 2,048 intervals each and interval counts 1,024/2,048/4,096/8,192 at 250 species.
- Eightfold species growth: 9.83567× median runtime and 7.99907× peak RSS; fitted log-log time slope 1.09710.
- Eightfold interval-count growth: 9.54999× median runtime and 7.99966× peak RSS; fitted slope 1.08328.
- The 1,000-species shape processes 204,800,000 bases and 2,048,000 intervals in median 75.5867 seconds (range 75.5176–75.7324), with 2,982,628 KiB peak RSS and 819,243,920 serialized index bytes.
- These are uniform synthetic genomes and measure global-index construction plus scoring, not biological accuracy or full production throughput.
- Naive linear extrapolation to one billion 100 bp intervals: 10.252 single-process hours, 400.0 GB index, and 1.491 TB peak RSS.
  This extrapolates constant throughput and distinct-word rate far beyond measurement; it is a planning warning, not a validated estimate for real genomes.
- Approximately 1,000 human-sized genomes contain 30 billion 100 bp intervals, rather than one billion.
- Algorithmic bound at fixed interval length: expected O(B + M) work and O(U + M) memory for B bases, M interval scores, and U sampled distinct words, including exact-budget partition selection.
  The query-only biological index and global synthetic index have different U; repeatedly rescanning the whole corpus for query batches adds a batch factor.
- Spatial audit passed all three budgets for both frozen settings, chromosome bounds, 100 bp grid alignment, absence of bridged gaps, conserved/covered-base ranges, and all 21 processed-base/window-count receipts.
- At 5%, primary output has 93,374 stretches and copy-filtered output has 88,952, each totaling 12,027,300 bp; both have median length 100 bp, with maxima 2,400 and 2,700 bp respectively.
- All 16 locked tests still pass; figures were rendered and inspected, and the conservation legend was moved outside the plot to avoid obscuring a data point.
- Conclusion: measurable local conservation signal from three genomes, with copy suppression improving conserved-base density; the present global in-memory index is not a demonstrated 1,000-real-genome solution.

### 2026-09-16 — archive, interpretation review, and authorized extension

- Pilot archive is complete: `s3://oa-bolinas/issues/577/local100-v1/bd3b6efd289507eac5151951042561e4ae51b9a4/`, 274 payload files, 311,956,628 bytes, all round-trip SHA-256 verified.
- Manifest SHA-256: `86fdcfec1b8d968f1a7debb1e0b5abf1a95eab713bbc4ccbcaa584565650e454`.
- Ownership checks confirmed the bucket belongs to authenticated account 836683583872, has only owner FULL_CONTROL ACL, and enables all public-access blocks.
  The initial automatic upload review needed this evidence; retry succeeded after read-only proof checks.
- Interpretation PR #578 passed all required checks and is ready for human review after independent GitHub Claude review.
  Addressed secondary-selection provenance and experiment-link wording; CI required removing trailing SVG whitespace.
  The independent review could check internal arithmetic and framing but lacked network permission to fetch raw artifacts; the separate experiment audit verifies source output contracts.
- Updated both #577 and #568 with the pilot result, archive, interpretation, and follow-up pointer.
- User explicitly requested keeping the worker for further analysis and pursuing all proposed axes, plus additional useful checks.
  This supersedes the earlier planned immediate worker termination; automatic termination at 23:20 UTC remains and cumulative EC2 costs must stay within $30.
- Extension protocol: `config/extension.json`; development remains chr1, fresh validation is chr3, and the completed chr2 pilot remains unchanged.
- Test nested panels of 3, 6, and 10 real mammals, and consistent sampling at 1/4, 1/8, and 1/16, preserving k25 and 100 bp resolution.
  Choose scores by absolute development conserved-base density; freeze all nine cell winners and the overall winner before opening chr3 labels.
- Add exact compact counting, biological/repeat-class characterization, and synthetic boundary/divergence/indel checks.
- Compact implementation uses contiguous 16-byte slots and the same exact species/copy semantics; all 22 tests pass, including byte-identical results against the baseline across sampling rates, reverse complements, ambiguous sequence, repeated words, incremental panels, and global/query-restricted modes.
- Full original-query byte-parity and resource measurement are running before any extension biological evaluation.

### 2026-09-16 — compact prior-work pass for memory and sampling

- Effort: low; stop after confirming the relevant sampling and external-memory design patterns.
- [Sourmash practical guide](https://sourmash.readthedocs.io/en/latest/using-sourmash-a-guide.html) distinguishes fixed-size sketches from fixed-rate scaled sketches.
  Our modulo-hash sample follows the latter pattern; the biological question is how far thinning can go at 100 bp resolution.
- [KMC 2](https://arxiv.org/abs/1407.1507) demonstrates disk-based k-mer counting with reduced RAM and I/O-aware compression.
  It supports partitioned counting as a design precedent, not a measured performance prediction for species-deduplicated scoring here.
- Negative lead: pairwise window MinHash does not remove the aggregate index requirement for this unary objective, and sparse whole-genome sketches may miss short local features.
- Active hypotheses: PREVALENCE-004 compact storage preserves scores; PREVALENCE-005 more diverse species improve density; PREVALENCE-006 thinner sampling preserves useful ranking; PREVALENCE-007 the selected signal remains after repeat/functional stratification; PREVALENCE-008 local boundaries tolerate realistic divergence and bin offsets.

### 2026-09-16 — fixed-size MinHash comparison and exact compact parity

- User asked specifically why not retain minimum hashes across all words in each window.
  Add bottom-16 and bottom-32 sketches of all valid canonical words, using the same full-genome species/copy counts; these do not compare window pairs.
- Final extension grid has 15 panel/sampling cells, each comparing the same six scores.
  Nominal seed count is the prespecified memory-proxy tiebreak, followed by panel size and score name.
- Bottom-16 is evaluated from the bottom-32 query universe; selected words are counted across every full support genome, so other windows' sketch choices do not remove matches.
- Exact compact pilot parity passed with `cmp`: all 4,911,499 rows and scores are identical to the original k25 score table.
  Full counted bases 9,659,766,308; distinct retained keys 100,973,493; allocated table 2,147,483,648 bytes.
  Build 278.454 seconds, score 24.6012 seconds, measured command 303.125 seconds, peak RSS 3,151,476 KiB (3.01 GiB).
  Original build-plus-score was 486.715 seconds and 5.77 GiB peak RSS; the compact run avoids serializing/reloading an index, so time includes that execution improvement.
- Seven additional full genomes are prepared and checksum-pinned: cow, elephant, rabbit, African wild dog, baboon, horse, and little brown bat.
  The elephant asset is identified by its exact HAL mirror object and hash, not a newer taxonomy-proxy assembly accession.
- Existing pipeline GTF and cCRE annotations were retrieved for descriptive biological overlap; UCSC hg38 RepeatMasker table was retrieved and SHA-pinned.
  GTF and cCRE bare chromosome names are explicitly converted for matching hg38 primary chromosomes at the annotation boundary.
- All 24 locked tests, ruff checks/format, and warning-as-error C++ builds pass.
  No chromosome-3 conservation labels have been inspected; expanded scoring and chromosome-1 development evaluation are next.

### 2026-09-16 — external-memory implementation and full-corpus check

- Fixed-rate panel counting/scoring completed successfully for all 3/6/10-species checkpoints, totaling 28,145,946,367 support bases at the largest panel.
  An idle SSH connection ended after the completed rate checkpoint; the exit-zero receipt and all outputs remain intact, no analysis process was left running, and the bottom-hash arm resumes separately with SSH keepalives.
- Added disk-partitioned global counting with one sequence scan, exact per-partition species/copy counting, and bounded chunks for assembling window scores.
  No partition triggers a rescan of the complete source genomes.
- Added exact per-species budget selection using score histograms and eight bounded-memory radix passes over only threshold-tied hashes.
  This avoids an O(all-windows) resident score array for final BED generation.
- All 30 locked project tests pass, including global score parity across partition counts, reverse complements, repeated words, partial contigs, and exact streaming-budget selection with ties and invalid windows.
- Planned resource matrix: 125 versus 1,000 species at 2,048 intervals each and 1,024 versus 8,192 intervals at 250 species; compact versus 32-partition counting at 1/4 and 1/16 sampling, three repetitions per cell.
  Original 1/4 synthetic tables supply full byte-parity references.
- After synthetic parity/resource checks, run the same 1/4 sampled partitioned algorithm over all three original full genomes and produce exact 5% per-species BED selections.
  Verify every original human chr1/chr2 score against the preserved pilot table.
  This is an all-window feasibility check at three real genomes, not a 1,000-genome production run or new biological endpoint.
- Expanded the sole worker's task-owned, DeleteOnTermination gp3 root volume from 80 to 200 GiB to hold this trial's temporary word partitions.
  No IOPS or throughput increase and no additional worker; the original automatic termination and $30 cumulative ceiling remain.
  At the verified $0.714/hour instance rate, even the full eight-hour worker lifetime plus prior $1.34 and a conservative disk/IPv4 allowance stays below $8, excluding a final invoice reconciliation.

### 2026-09-16 — extension development selection frozen

- Completed the 15 panel/sampling cells and all six scoring choices per cell on chromosome 1.
  The original chr2 pilot remains unchanged; no chr3 conservation labels have been created or inspected.
- Development winner: ten species, bottom 32 hashes across all words per 100 bp bin, copy-filtered mean other-species breadth.
  Selected conserved-base fraction is 31.4818%, versus 23.6625% for the three-species 1/4-sampled any-copy4 baseline, at the fixed 5% budget.
- Fixed-rate 1/4 density at 3/6/10 species is 23.6625%/28.6348%/31.0824%; bottom-32 is 24.0965%/29.1000%/31.4818%.
  These are development comparisons, not evidence from the fresh validation chromosome.
- The selection artifact fixes all 15 cell winners, the primary comparison, and hashes of the input manifest and protocol.
  Commit and publish this artifact before running `extension_evaluate --split validation --freeze-sha COMMIT`.
- Updated the resource matrix to include contemporaneous baseline reruns, giving 60 measurements in total; all 1/4 results must reproduce the original complete synthetic score tables.
- Streaming selection now uses weighted radix threshold selection instead of sorting distinct score values and keeps one species histogram resident at a time.
  Added standalone per-scheme query-index profiles and full-genome/extension artifact audits.
- All 31 locked tests pass in 11.71 seconds; ruff formatting/checks and three C++ warning-as-error builds pass on the worker.

### 2026-09-16 — fresh chromosome-3 validation complete

- Selection/code freeze: `aef30904792d0a26c5547dcdb885b1bf4665fcdc`, published in issue comment 5702602314 before opening chr3 labels.
- The frozen ten-species bottom-32 breadth-copy4 primary selects 5% of eligible chromosome-3 bases containing 27.6930% conserved bases, versus 20.1387% for the three-species rate-4 any-copy4 baseline.
  The paired 200-block-bootstrap density difference interval is +7.1074 to +8.0403 percentage points, around a +7.5543-point estimate.
- Primary random enrichment is 7.1974×, matched enrichment 4.9758×, and annotated-base recall 35.9869%.
  Baseline recall is 26.1702%; composition-matched enrichment is 5.0863×, so the extension improves the declared absolute-density endpoint rather than every possible enrichment statistic.
- At the prespecified 1% budget the primary contains 64.2041% conserved bases and recovers 16.6859%; at 10%, density is 17.2218% and recall 44.7593%.
- At ten species, fixed-rate 1/4 reaches 27.3423% density, bottom-16 26.7229%, and bottom-32 27.6930%.
  MinHash's incremental gain over the dense fixed-rate sample is only 0.3508 percentage points and is not a matched-memory comparison.
  The larger panel supplies most of the combined gain; per-scheme index footprints remain to be measured.
- All fifteen cell winners were fixed using chr1 and reported on chr3 without retuning.
  The resource matrix is now running, followed by full three-genome scoring/selection and descriptive controls.

### 2026-09-16 — repeated global synthetic resource matrix complete

- Completed all 60 runs: four shapes, five storage/sampling arms, three repetitions.
  All 36 quarter-rate comparisons reproduce the full original score tables, totaling 41,472,000 audited rows across repetitions and methods.
- Largest shape (1,000 species × 2,048 intervals, 204.8 Mb), median wall time / peak RSS:
  original quarter-rate index 76.8726 s / 2.84449 GiB;
  compact quarter-rate 22.0165 s / 1.50389 GiB;
  32-partition quarter-rate 32.7861 s / 0.0919571 GiB;
  compact sixteenth-rate 9.02176 s / 0.378784 GiB;
  32-partition sixteenth-rate 7.92341 s / 0.0252953 GiB.
- Negative scaling detail: eightfold input growth takes approximately 15× wall time in quarter-rate disk partitioning on both axes, versus approximately 10× for the original/compact indexes and 9.2× for sixteenth-rate partitioning.
  The expected hash-work bound is linear, but empirical cache and I/O effects prevent a constant-throughput production extrapolation.
- The real three-genome global run has passed its disk-space preflight and started; expected output is 96,574,572 complete intervals over 9,659,766,308 bases.
