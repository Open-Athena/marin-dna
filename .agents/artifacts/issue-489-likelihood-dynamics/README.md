# Issue 489 likelihood dynamics results

This directory preserves the compact statistical outputs and reviewed figures for [issue #489](https://github.com/Open-Athena/marin-dna/issues/489).

## Design

Five checkpoints on the fixed 1B m1 to m1.3 lineage were scored at 21.0B, 62.9B, 104.9B, 146.8B, and 173.7B cumulative training tokens.
Each checkpoint was evaluated on the complete 16,384-window CDS, upstream, downstream, ncRNA, and enhancer validation probes from the MarinDNA blog Hugging Face collection.
The durable unfiltered cache contains 104,448,000 token rows across 25 checkpoint-region cells.
The primary population contains 14,002,032 scorable, nonambiguous, nonrepeat positions in target positions `[32, 223)`, including 4,792,703 case-encoded conserved positions.
All confidence intervals are 500-replicate bootstraps over region-specific 10 Mb genomic blocks.
The controlled contrasts regress negative loss or negative entropy on conservation, GC and GC squared, held-out 7-mer NLL and its square, and a cubic target-position basis.

## Findings

### Conservation ranking

Lower loss ranked conservation at the first observed checkpoint and continued to improve globally.

| Scope | Prevalence | Loss AUPRC at 21.0B | Loss AUPRC at 173.7B | Change |
| --- | ---: | ---: | ---: | ---: |
| Global | 0.342 | 0.502 | 0.601 | +0.099 |
| CDS | 0.455 | 0.535 | 0.686 | +0.151 |
| Upstream | 0.196 | 0.360 | 0.489 | +0.129 |
| Downstream | 0.169 | 0.362 | 0.503 | +0.141 |
| ncRNA | 0.429 | 0.641 | 0.642 | +0.001 |
| Enhancer | 0.433 | 0.555 | 0.669 | +0.114 |

Entropy gives the same conclusion, moving from 0.492 to 0.599 globally.
The earliest available checkpoint is already above prevalence in every region, so this design does not locate the initial emergence before 21.0B tokens.
ncRNA is the exception to continued improvement: loss AUPRC peaks at 0.651 at 146.8B tokens and ends at 0.642.

### Lowest-score set stability

The region-specific lowest-loss 10% becomes progressively more stable, but it does not define a fixed population across training.

| Checkpoint pair | Global loss Jaccard | Global entropy Jaccard |
| --- | ---: | ---: |
| 21.0B to 62.9B | 0.425 | 0.435 |
| 62.9B to 104.9B | 0.549 | 0.550 |
| 104.9B to 146.8B | 0.560 | 0.560 |
| 146.8B to 173.7B | 0.604 | 0.603 |
| 21.0B to 173.7B | 0.299 | 0.307 |

Enhancer has the most churn, with loss Jaccard 0.266 for the first adjacent pair, 0.484 for the final adjacent pair, and 0.225 end to end.
Upstream has the most stable late loss mask at 0.712 Jaccard.
Across region-specific mean-loss trajectory groups, 36.7% of positions are low to low, 37.8% are high to high, 11.7% are high to low, and 13.8% are low to high.
The two state-changing groups therefore contain 25.5% of the primary population.

### Current-loss change scores

High-current-loss deciles have larger observed current-to-later NLL reductions than low-current-loss deciles.
From 21.0B to 173.7B tokens, the global lowest current-loss decile improves by 0.047 nats/base, while the highest current-loss decile improves by 0.521 nats/base.
For the next checkpoint, the lowest decile changes by -0.028, -0.106, -0.088, and +0.002 nats/base from the four successive current checkpoints.
The highest decile improves by +0.304, +0.319, +0.343, and +0.255 nats/base over those same next-checkpoint intervals.
Current NLL both defines the decile bins and enters the reduction outcome with the same sign.
The association can therefore include regression to the mean and does not establish differences in remaining optimization opportunity.

### Covariate controls

The conserved coefficient for negative loss remains positive in every region and checkpoint after controlling for GC, held-out 7-mer predictability, and target position.
At the terminal checkpoint, the adjusted negative-loss contrast ranges from 0.297 nats in ncRNA to 0.548 nats downstream, with every 95% block-bootstrap interval above zero.
The contrast grows with training in CDS, upstream, downstream, and enhancer, but is nearly flat to declining in ncRNA.
These controls support a conservation association beyond the specified sequence-composition covariates, not a causal benefit from changing the training objective.

## Interpretation

Do not use the student's instantaneous lowest-loss 10% as the primary selector in a causal training experiment.
The mask stabilizes with training but still has only 0.299 endpoint Jaccard globally and materially worse stability in enhancer sequence.
A frozen, sufficiently trained teacher is the clean primary design because it pins the selection function before student training.
If an online student-derived selector is retained as a diagnostic arm, it should use a uniform warm-up, temporal smoothing, and a nonzero background floor.
A low-loss selector is also a self-paced objective that preferentially retains already-easy tokens; it is not Rho-1 reducible loss.

This experiment is observational and does not test whether frozen likelihood weights improve any downstream task.

## Artifacts and provenance

- Analysis implementation: `b71867b6b35d49286c75f1bd9fa44c74b44b56ef`.
- Reviewed figure implementation: `fb0ffda0`.
- S3 root: `s3://oa-bolinas/snakemake/analysis/evals_v2/results/m13_likelihood_dynamics_489/v1/`.
- Analysis manifest: [manifest.json](manifest.json).
- Exact tables: the seven Parquet files in this directory.
- Reviewed plots: the six SVG files under [figures](figures).
- Full reducer: 3m34s wall time and 5,723,164 KiB maximum RSS on an AWS r7i.2xlarge.
- Estimated SkyPilot cost: $1.79 total, comprising $1.58 for the pilot and full GPU scoring and $0.21 for CPU analysis and figure generation.
