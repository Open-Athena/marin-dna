---
topic: issue-489-likelihood-dynamics
issue: https://github.com/Open-Athena/marin-dna/issues/489
description: Track likelihood-derived token rankings through the m1 to m1.3 training lineage.
author: Codex
---

# Issue 489 likelihood dynamics: Task Logbook

## Scope

- Goal: Measure how per-token forward NLL, four-nucleotide entropy, conservation ranking, and easy-token membership change through five checkpoints on the m1 to m1.3 lineage.
- Primary metrics: Conservation AUPRC, low/high trajectory classes, lowest-loss 10% Jaccard, and subsequent loss reduction by current-loss decile.
- Constraints: Use the five complete validation probes released with the MarinDNA blog, score one forward orientation, retain unfiltered per-token atoms, and keep the primary nonrepeat central span at `[32, 223)`.
- Coordinating issue: https://github.com/Open-Athena/marin-dna/issues/489
- Experiment prefix: `LD489`
- Shared tags: `LD489`, `issue-489`, `m1-lineage`

## Current TL;DR

- The five-panel blog collection manifest and all dataset revisions are pinned.
- The complete cache contains 104,448,000 unfiltered token rows across five checkpoints and five regions, and every manifest and parquet-footer audit passes.
- The primary population contains 14,002,032 scorable nonrepeat central-span positions, including 4,792,703 case-encoded conserved positions.
- Global loss AUPRC for conservation rises from 0.502 at 21.0B tokens to 0.601 at 173.7B tokens; entropy rises from 0.492 to 0.599.
- Ranking is already above prevalence in every region at 21.0B tokens, so emergence before that checkpoint remains unresolved.
- ncRNA is the exception to continued improvement: its loss AUPRC peaks at 0.651 at 146.8B tokens and ends at 0.642.
- The global lowest-loss-decile Jaccard rises from 0.425 for the first adjacent pair to 0.604 for the final adjacent pair, but endpoint Jaccard is only 0.299.
- Enhancer has the most mask churn, with loss Jaccard 0.266 early, 0.484 late, and 0.225 end to end.
- The highest current-loss decile improves by 0.521 nats/base from 21.0B to terminal, compared with 0.047 for the lowest decile.
- Conservation remains positively associated with negative loss and entropy in every region and checkpoint after GC, 7-mer, and target-position controls.
- Use a frozen sufficiently trained teacher for the primary causal selector; require warm-up, smoothing, and a background floor for any online student-derived diagnostic arm.
- The reviewed tables and figures are under `.agents/artifacts/issue-489-likelihood-dynamics/`; estimated total SkyPilot cost is $1.79.

## Baseline

- Date: 2026-08-21
- Code refs: issue #478 permanent branch `79db49cedcb9c8bb9c45bd35d7d6f8fd078e91b2`; blog lineage scaffold commit `40fcee94`.
- Baseline numbers: issue #478 scored 16,384 windows in each of CDS, upstream, and downstream over the central span `[32, 223)`.

## Decision Log

- 2026-08-21: Use the five validation probes listed in the blog collection.
- 2026-08-21: Pin every Hugging Face dataset to its immutable current revision.
- 2026-08-21: Treat the blog collection as the experiment manifest even though `val_ncrna` and `val_enhancer` are clean validation recipes rather than exact partitions of the v3 training shards.
- 2026-08-21: Preserve separate RefSeq and Ensembl reference assets and fail on any sequence mismatch.

## Hypothesis Queue

### Active

- None.

### Blocked

- None.

### Falsified / Dead End

- The five validation datasets are not the five `zoonomia-v1-val_*` recipes considered initially.
- The blog collection uses three `genomes-v5-validation-*` probes and only two `zoonomia-v1-val_*` probes.
- The `LD489-H1` prediction that ncRNA and enhancer would have the largest training-time AUPRC gains is false; ncRNA is nearly flat and CDS has the largest gain.

### Promoted

- `LD489-H1`: Conservation ranking is already present at 21.0B tokens and strengthens globally through 173.7B tokens, with ncRNA as the nonmonotonic exception.
- `LD489-H2`: Adjacent lowest-decile stability increases late in training, but 0.299 global endpoint loss Jaccard is not a fixed online mask.
- `LD489-H3`: High-current-loss positions receive substantially more next and terminal loss reduction than low-current-loss positions.
- `LD489-H4`: The conservation contrast survives the prespecified GC, held-out 7-mer, and target-position controls in every region and checkpoint.

## Background Research Brief

- Effort: Low.
- Stop rule: Stop after the blog collection, dataset cards, immutable Hub revisions, issue #478 inputs, and blog lineage configuration agree on the input manifest.
- Date: 2026-08-21.

### Question

Which five complete validation datasets should issue #489 score so the probes match the training data represented in the MarinDNA blog?

### Current Marin Context

Issue #489 specifies five regions and asks to reuse complete existing validation sets.
Issue #478 established the CDS, upstream, and downstream probes and the `[32, 223)` primary span.
The original m1.3 training definition names five training sources but only three validation sources, so the training script alone does not identify the requested five-probe evaluation panel.

### Internal Prior Work

- Issue #478 used the three `genomes-v5-validation-*` probes pinned below.
- The blog lineage configuration records the available on-path m1, m1.1, m1.2, and m1.3 checkpoints and their GCS sources.
- All four stages share 2,097,152 training tokens per optimizer step.

### External Prior Art

- The MarinDNA Hugging Face collection describes five training datasets and five training-validation probes.
- The collection explicitly pairs CDS, upstream, downstream, enhancer, and ncRNA labels with the five repositories pinned below.
- The Zoonomia dataset cards define 255 bp Ensembl release 115 human windows, conservation prefiltering, case-encoded phyloP labels, and no reverse-complement augmentation.

### Negative / Failed Leads

- The full seven-recipe `zoonomia-v1-val_*` family does not match the collection manifest.
- The v3 ncRNA and cCRE training repositories expose a training split and do not themselves provide the complete human validation probes requested by issue #489.

### Evidence Map

#### Claim: The blog collection is the five-panel manifest for issue #489

- Support: https://huggingface.co/collections/marin-dna/a-1b-standard-transformer-rivals-evo-2-40b-on-vep lists the five datasets below as training-validation probes.
- Contradictions: The original m1.3 training script configures only three validation datasets.
- Directness to Marin: Exact release collection for the blog and model lineage under study.
- Confidence: High.
- Action: Pin these five revisions in the #489 workflow.

#### Claim: The two validation families require different reference assets

- Support: The three genomes-v5 cards use RefSeq `GCF_000001405.40` identifiers, while the two Zoonomia cards use bare Ensembl release 115 sequence names.
- Contradictions: None found.
- Directness to Marin: Exact input cards and schemas.
- Confidence: High.
- Action: Join repeat masking separately and assert uppercase sequence equality for every window.

### Recommended Next Experiments

#### 1. Earliest-versus-terminal pilot

- Minimum experiment: Score a fixed small prefix of every validation probe at the earliest and terminal selected checkpoints.
- Baseline/control: Reconstruct aggregate forward loss from per-token NLL and compare finite counts, IDs, shapes, and case totals across checkpoints.
- Expected signal: Identical row and token identities with finite NLL and entropy for every scorable A/C/G/T target.
- Falsifier: Any row-order drift, coordinate mismatch, missing token, non-finite atom, or disagreement with the aggregate loss kernel beyond the runtime tolerance.
- Cost/risk: Two 1B checkpoint loads across five small panels on one paid GPU.
- Sources: Issue #489 and issue #478.

### Hypothesis Queue Update

- Add: `LD489-H1` through `LD489-H4`.
- Revise: None.
- Falsify / stop: Do not use a five-member Zoonomia validation family as the manifest.
- Promote: The blog collection manifest into workflow configuration.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
| --- | --- | --- | --- | --- | --- |
| Issue #489 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/489 | Scope, metrics, checkpoint count, output namespace | High | Current coordinating issue |
| Blog collection | Official release collection | https://huggingface.co/collections/marin-dna/a-1b-standard-transformer-rivals-evo-2-40b-on-vep | Five validation probes | High | Collection labels each probe |
| Issue #478 branch | Marin code | `79db49cedcb9c8bb9c45bd35d7d6f8fd078e91b2` | Three probe pins, per-token scorer, central span | High | Permanent research branch |
| Blog lineage scaffold | Marin code | `40fcee94` | Checkpoint paths and available steps | High | Pinned historical config |
| Zoonomia validation cards | Official dataset docs | Hugging Face dataset pages | Coordinate assembly, recipes, conservation encoding | High | Exact datasets in collection |

### Handoff

- Suggested issue `Prior work` block: The blog Hugging Face collection resolves the five-panel manifest as the three issue #478 probes plus `zoonomia-v1-val_enhancer` and `zoonomia-v1-val_ncrna`.
- Suggested logbook entry: This entry.
- Open questions: Whether the primary analysis should include chromosome Y from these unlabeled validation probes; the project held-out-label restriction does not apply to unlabeled reference and functional-genomics inputs.
- Stop reason: The exact input manifest and reference split are resolved.

## Input Manifest

| Region | Hugging Face repository | Revision | Rows | Coordinate/reference family |
| --- | --- | --- | ---: | --- |
| CDS | `marin-dna/genomes-v5-validation-intervals-v5_255_255` | `daff592f213aaa1cab1711d477a79ff6b1bc4ef4` | 16,384 | RefSeq `GCF_000001405.40` |
| upstream | `marin-dna/genomes-v5-validation-intervals-v1_255_255` | `a761bc0b663a9827303f3112e4667d53d5326fac` | 16,384 | RefSeq `GCF_000001405.40` |
| downstream | `marin-dna/genomes-v5-validation-intervals-v15_255_255` | `d7b27eecd68453934ebb3e7e6e78d5401789faa5` | 16,384 | RefSeq `GCF_000001405.40` |
| enhancer | `marin-dna/zoonomia-v1-val_enhancer` | `d40d1e067b2a56ac812af122de029eb79cab1106` | 16,384 | Ensembl release 115 GRCh38 |
| ncRNA | `marin-dna/zoonomia-v1-val_ncrna` | `76a18c1bbf07ac9bd064722431bbdab894b9e6c6` | 16,384 | Ensembl release 115 GRCh38 |

## Entry Log

### 2026-08-21 14:47 UTC - `LD489-001` resolve the validation manifest

- Hypothesis: The blog release collection identifies the five complete validation probes intended by issue #489.
- Commit Hash: Pending.
- Command: Queried the public Hugging Face collection and dataset APIs, inspected issue #478 at `79db49cedcb9c8bb9c45bd35d7d6f8fd078e91b2`, and inspected the historical lineage config at `40fcee94`.
- Config: Five 16,384-row, 255 bp validation probes at the revisions in the input manifest.
- Result: The collection lists the three issue #478 probes plus the enhancer and ncRNA Zoonomia probes.
- Interpretation: The collection is the authoritative experiment manifest.
- Next action: Add an isolated, versioned #489 scoring path and an earliest-versus-terminal pilot target.

### 2026-08-21 15:08 UTC - `LD489-002` scaffold the versioned atom cache and pilot

- Hypothesis: The issue #478 per-token kernel can be isolated for forward-only #489 scoring while preserving aggregate LL behavior.
- Commit Hash: Pending.
- Command: `uv run --locked pytest tests/test_likelihood_dynamics_489.py`.
- Config: Eight focused unit tests; shared-node thread limits; nonblocking `/tmp/marin-dna-local-heavy.lock`; `nice -n 10`; `ionice -c 2 -n 7`.
- Result: Eight tests passed in 11.89 seconds.
- Result: Peak RSS was 1,018,944 KiB, dominated by importing the locked Torch stack.
- Interpretation: The reference mismatch guard, independent conservation/repeat case, token alignment, four-nucleotide entropy, stable identity, provenance, and pilot comparison contracts pass.
- Interpretation: Even a focused evals_v2 test process exceeds the 500 MiB shared-node working-set ceiling.
- Next action: Do not repeat local pytest for this task.
- Next action: Run subsequent parser and dry-run checks on remote compute after approval.

### 2026-08-21 15:08 UTC - `LD489-003` first pilot dry-run

- Hypothesis: The additive pilot target resolves to ten scoring cells and five reusable metadata cells without changing `rule all`.
- Commit Hash: Pending.
- Command: `uv run --locked snakemake -n --profile workflow/profiles/default likelihood_dynamics_489_pilot`.
- Config: Shared-node thread limits; nonblocking heavy-work lock; no jobs executed because this was a dry-run.
- Result: Snakemake stopped during parsing because `checkpoint` is a reserved Snakefile keyword in a top-level loop.
- Result: Peak RSS was 1,014,996 KiB even though no DAG jobs ran.
- Interpretation: Rename the loop variable to `checkpoint_cfg`.
- Interpretation: The dry-run did not reach DAG construction, so remote input and job counts remain unverified.
- Next action: The reserved keyword is renamed.
- Next action: Do not repeat the memory-heavy parser locally.
- Next action: Run the corrected dry-run on the remote execution node before launching the pilot.

### 2026-08-21 15:19 UTC - `LD489-004` corrected remote dry-run

- Hypothesis: The corrected target resolves only the approved earliest-versus-terminal pilot.
- Commit Hash: `e3879d3a80ee09006b964994e0254fd2f5a7ff27`.
- Command: `sky launch snakemake/analysis/evals_v2/sky/run.yaml -c evals-v2-ld489-pilot --env SNAKEMAKE_ARGS="-n --resources gpu=1 -- likelihood_dynamics_489_pilot" -y`.
- Config: AWS `g5.xlarge` with one NVIDIA A10G in `us-east-2`; all spot zones lacked capacity, so Sky used the configured $1.01/hour on-demand fallback.
- Result: The pinned AMI passed the PyTorch 2.13.0, CUDA 13.0, A10G, and bf16 runtime gate.
- Result: The DAG contained five metadata jobs, ten scoring jobs, one validator, and one named-target wrapper.
- Result: The two checkpoint conversions were already present in the S3 cache.
- Interpretation: The DAG was pilot-sized and contained no full 16,384-window scoring or unrelated targets.
- Next action: Launch the approved 128-window pilot.

### 2026-08-21 15:20 UTC - `LD489-005` stop unintended remote FASTA indexing

- Hypothesis: The Ensembl soft-masked FASTA reader will use its companion index for random access.
- Commit Hash: `e3879d3a80ee09006b964994e0254fd2f5a7ff27`.
- Command: Started the pilot, inspected remote processes, and canceled Sky job 2 before GPU scoring.
- Config: The ncRNA and enhancer metadata jobs each requested 128 intervals.
- Result: Both jobs instead held about 1.2 GiB RSS and one CPU at 99% for several minutes.
- Result: `pyfaidx` had opened the remote FASTA without consuming the existing companion `.fai` and was scanning the full FASTA to reconstruct indexing.
- Interpretation: The `subset_chroms` argument only filters exposed sequence names; it does not build chromosome subsets and did not prevent the indexing scan.
- Next action: Query the repeat-mask reference through direct `.fai`-derived S3 byte ranges.

### 2026-08-21 15:45 UTC - `LD489-006` pass the earliest-versus-terminal pilot

- Hypothesis: Direct indexed FASTA range requests preserve reference identity while avoiding a full genome scan, and the per-token scorer preserves aggregate likelihood behavior.
- Commit Hash: Pending.
- Command: Ran nine focused tests on the remote node, a live indexed S3 query, a corrected dry-run, and `likelihood_dynamics_489_pilot`.
- Config: First 128 windows from each of five 255 bp probes; m1 step 10,000 and m1.3 step 82,823; forward strand only; GPU jobs serialized with `--resources gpu=1`.
- Result: Nine focused tests passed in 4.63 seconds.
- Result: The 6,406-byte `.fai` exists beside the Ensembl release 115 FASTA and the live S3 byte-range query returned immediately.
- Result: The ncRNA and enhancer metadata joins both completed and uploaded in about 20 seconds total.
- Result: Peak observed GPU allocation for the first scoring cell was 4,942 MiB.
- Result: All ten scoring cells produced 32,640 rows and 32,640 scorable rows.
- Result: Every cell passed the aggregate likelihood parity gate, with maximum absolute differences from `3.30507755279541e-05` to `4.571676254272461e-05`.
- Result: The final validator passed token identity, row count, finiteness, and per-cell aggregate checks across 326,400 total atom rows.
- Result: The report is `s3://oa-bolinas/snakemake/analysis/evals_v2/results/m13_likelihood_dynamics_489/v1/pilot/validation_report.json`.
- Interpretation: The atom-cache producer is ready for review before expanding to five checkpoints and the full 16,384-window probes.
- Next action: Do not launch full scoring without a separate review and approval.
- Next action: The paid `evals-v2-ld489-pilot` cluster was terminated after the report was secured.

### 2026-08-21 17:25 UTC - `LD489-007` complete the full atom cache

- Hypothesis: The pilot-validated producer can score the complete five-checkpoint by five-probe matrix while preserving stable token identities and bounded remote reference access.
- Commit Hash: `c91b51c5f5dee922f85f5aca3f81b5068c84cbf4`.
- Dry-run command: `sky launch snakemake/analysis/evals_v2/sky/run.yaml -c evals-v2-ld489-full --memory=30+ --env SNAKEMAKE_ARGS="-n --resources gpu=1 -- likelihood_dynamics_489_atoms" -y`.
- Run command: `sky exec evals-v2-ld489-full snakemake/analysis/evals_v2/sky/run.yaml --env SNAKEMAKE_ARGS="--resources gpu=1 -- likelihood_dynamics_489_atoms"`.
- Config: AWS `g5.2xlarge[Spot]` in `us-east-2a` with one NVIDIA A10G, 8 vCPUs, and 32 GB RAM at a displayed spot price of $0.70/hour.
- Result: The dry-run contained five metadata jobs, 25 scoring jobs, and one named-target wrapper, with no unrelated jobs or model conversions.
- Result: The live run completed 31 of 31 jobs with exit code 0 from 16:00:07 UTC through 17:25:23 UTC.
- Result: GPU inference stabilized at 1.58 batches per second for 256 batches per scoring cell, or about 161 seconds per forward pass.
- Result: All five metadata parquet/manifest pairs were uploaded under `s3://oa-bolinas/snakemake/analysis/evals_v2/results/m13_likelihood_dynamics_489/v1/metadata/full/`.
- Result: All 25 score parquet/manifest pairs were uploaded under `s3://oa-bolinas/snakemake/analysis/evals_v2/results/m13_likelihood_dynamics_489/v1/scoring/full/atoms/`.
- Result: The score inventory contains 50 objects totaling 2,412,324,903 bytes.
- Result: Every scoring manifest reports scope `full`, 16,384 windows, 4,177,920 positions, 4,177,920 scorable positions, the expected stable identity `(region, row_index, target_pos)`, and a null aggregate gate as configured for the one-pass full run.
- Result: The 25 manifest checkpoint/region pairs exactly cover all five checkpoint orders and all five regions.
- Result: Direct parquet-footer validation found 25 parquets, four row groups per parquet, 4,177,920 rows per parquet, the required atom columns, and 104,448,000 rows total.
- Interpretation: The complete unfiltered atom cache is valid and ready for bounded-memory statistical summarization.
- Next action: Implement and run the primary-span nonrepeat summaries, conservation AUPRC, lowest-loss 10% Jaccard, loss-reduction deciles, covariate controls, and research plots.
- Next action: The paid `evals-v2-ld489-full` cluster was terminated after the inventory and manifest/footer audits passed.
### 2026-08-21 18:51 UTC - `LD489-008` complete the prespecified analysis

- Hypothesis: Conservation ranking strengthens during fixed-architecture training, the lowest-score decile becomes increasingly stable, and high-current-loss positions have more remaining loss reduction than low-current-loss positions.
- Commit Hash: Analysis `b71867b6b35d49286c75f1bd9fa44c74b44b56ef`; reviewed figure code `fb0ffda0`.
- Test command: `sky exec evals-v2-ld489-analysis snakemake/analysis/evals_v2/sky/analysis_489.yaml --env ANALYSIS_MODE=test`.
- Analysis command: `sky exec evals-v2-ld489-analysis snakemake/analysis/evals_v2/sky/analysis_489.yaml --env ANALYSIS_MODE=analysis`.
- Plot refresh command: `sky exec evals-v2-ld489-analysis snakemake/analysis/evals_v2/sky/analysis_489.yaml --env ANALYSIS_MODE=plot`.
- Config: Primary target positions `[32, 223)`; scorable, nonambiguous, nonrepeat positions; exact pooled AUPRC; region-specific lowest-score 10%; ten equal-count current-loss bins; 500 genomic-block bootstrap replicates at 10 Mb with seed 489; GC, GC-squared, held-out 7-mer NLL, 7-mer-NLL-squared, and cubic target-position controls.
- Result: The remote dry-run contained exactly the reducer, figure rule, and named-target wrapper.
- Result: Three focused tests passed in 4.08 seconds with 286,856 KiB maximum RSS.
- Result: The reducer and six-figure build completed in 3m34s with 5,723,164 KiB maximum RSS and no swaps.
- Result: The primary population contains 14,002,032 positions, of which 4,792,703 are conserved.
- Result: Global loss AUPRC rises from 0.502 at 21.0B tokens to 0.601 at 173.7B tokens, while entropy AUPRC rises from 0.492 to 0.599.
- Result: Loss AUPRC changes from first to terminal by +0.151 in CDS, +0.129 upstream, +0.141 downstream, +0.001 in ncRNA, and +0.114 in enhancer.
- Result: Global lowest-loss 10% Jaccard is 0.425, 0.549, 0.560, and 0.604 across adjacent pairs and 0.299 end to end.
- Result: Global lowest-entropy 10% Jaccard is 0.435, 0.550, 0.560, and 0.603 across adjacent pairs and 0.307 end to end.
- Result: The earliest-to-terminal global mean loss reduction is 0.047 nats/base for the lowest current-loss decile and 0.521 nats/base for the highest.
- Result: The conserved coefficient for negative loss and entropy is positive with a block-bootstrap interval above zero in every region and checkpoint after the prespecified controls.
- Result: All seven Parquet tables, the manifest, and six SVGs are under `.agents/artifacts/issue-489-likelihood-dynamics/` and the versioned S3 root.
- Result: Visual review found and corrected shared-title/legend overlap and crowded labels; all six final renderings pass visual inspection.
- Result: SkyPilot estimates $0.51 for the pilot GPU cluster, $1.07 for full GPU scoring, and $0.21 for CPU analysis, or $1.79 total.
- Interpretation: Ranking is already present by the earliest observed checkpoint and strengthens in four of five regions, but the online lowest-loss mask remains materially time- and region-dependent.
- Interpretation: A frozen sufficiently trained teacher should define the primary likelihood-derived selector; an online student-derived diagnostic arm requires warm-up, temporal smoothing, and a nonzero background floor.
- Interpretation: Low absolute loss predicts conservation but not remaining optimization opportunity, and it must not be called Rho-1 reducible loss.
- Next action: Publish the accepted interpretation through the knowledge-base pull-request gate and update issue #489 with immutable code, data, figure, and cost links.
- Next action: The paid `evals-v2-ld489-analysis` cluster was terminated after the final SVGs were secured.
