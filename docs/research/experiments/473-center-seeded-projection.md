# Center-seeded projection for vertebrate training data

> [!NOTE]
> **TL;DR:** Projecting only the center nucleotide is a strong replacement for full-window projection for CDS anchors, but not for enhancer-centered cCRE anchors. At matched training tokens, center-1 improved every presented CDS Mendelian endpoint and the missense/splicing Complex and SGE endpoints; for enhancers it reduced distal Mendelian performance and was effectively tied on distal Complex. Projection policy should therefore be region-specific.

## Findings

Center-seeded projection does not create a universal recovery advantage.
Relative to full-window projection, it recovered 2.679 percentage points more CDS species-anchor pairs and 10.208 points more ncRNA-exon pairs, but 0.892 points fewer enhancer-centered cCRE pairs and 3.135 points fewer TSS/5′-UTR pairs.
CDS recovery improved in every audited clade, whereas enhancer recovery declined in every clade.
Both policies still reached jawless vertebrates.

The sampled reverse-trace audit did not support the proposed mechanism that center-1 generally sacrifices aligned anchor coverage or adds external flank.
Every region's paired aligned-coverage interval included zero.
Only UTR3 had a nonzero external-flank difference, with center-1 adding 1.143 bases on average.

Downstream results separate CDS from enhancers more sharply than projection yield does.

<p align="center">
  <img src="figures/473/cds_auprc_paired_delta_trajectories.svg" alt="Three panels showing center-1 minus full-window paired Mendelian AUPRC trajectories for CDS missense, splicing, and synonymous variants" />
</p>

_Positive differences favor center-1; bands are paired 95% match-group bootstrap intervals on the development split._

At step 4,999, center-1 improved CDS Mendelian AUPRC by 0.249 for missense, 0.237 for splicing, and 0.185 for synonymous variants.
All three paired intervals excluded zero.
The corresponding Group SMD differences were also positive, and CDS center-1 had lower case-weighted NLL on the exact shared chromosome-18 projection rows at every checkpoint.

<p align="center">
  <img src="figures/473/enhancer_auprc_paired_delta_trajectories.svg" alt="One panel showing the center-1 minus full-window paired Mendelian AUPRC trajectory for distal enhancer variants" />
</p>

_Positive differences favor center-1; bands are paired 95% match-group bootstrap intervals on the development split._

Enhancer center-1 did not show the same benefit.
Its terminal distal Mendelian AUPRC was 0.056 lower, with an interval excluding zero.
Distal Complex AUPRC differed by only +0.0015, much less than either policy's official bootstrap SE, and the shared-row loss trajectory changed sign across checkpoints.

The decision is to use center-1 for CDS projection and retain full-window projection for enhancer-centered cCREs.
This result does not justify one projection contract across every genomic region, nor does it justify a wider projection-policy pilot now.

## Projection evidence

The producer completed all 135 projection partitions and the deterministic sample trace completed all 645 trace jobs.
The accepted union contains 74,524,203 species-anchor rows.
All emitted intervals are 255 bp, all coordinates are in bounds, and chromosome/strand agreement is exact.
The median projected-center displacement is 3 bp.

| Region | Full window | Center 1 | Difference |
|---|---:|---:|---:|
| Enhancer-centered cCRE | 80.72% | 79.83% | -0.892 pp |
| CDS | 84.10% | 86.78% | +2.679 pp |
| ncRNA exon | 67.64% | 77.85% | +10.208 pp |
| TSS / 5′ UTR | 77.80% | 74.66% | -3.135 pp |
| 3′ UTR | 76.99% | 79.76% | +2.766 pp |

_Species-anchor recovery on the fixed human anchors. Across all five regions, recovery was 79.60% for full window and 82.28% for center-1._

The 5,328-row named-PSL sample reproduced every producer aligned-base count exactly.
Uncertainty used 1,000 bootstrap draws over human anchors.

| Region | Aligned-coverage difference | 95% interval | Anchors | External-flank difference | 95% interval | Anchors |
|---|---:|---:|---:|---:|---:|---:|
| Enhancer-centered cCRE | +0.00433 | [-0.00480, 0.01345] | 102 | -1.063 bp | [-3.465, 0.502] | 102 |
| CDS | +0.00040 | [-0.00312, 0.00393] | 214 | -0.163 bp | [-1.617, 1.010] | 209 |
| ncRNA exon | +0.00484 | [-0.00231, 0.01200] | 51 | -0.400 bp | [-3.952, 2.750] | 40 |
| TSS / 5′ UTR | +0.00580 | [-0.01104, 0.02264] | 46 | +0.889 bp | [-2.938, 4.245] | 45 |
| 3′ UTR | +0.00080 | [-0.01044, 0.01205] | 49 | +1.143 bp | [0.387, 2.204] | 49 |

_Paired center-1 minus full-window mean aligned-to-anchor fraction and total external-flank bases among the sampled traces._

Two examples make the policy difference concrete:

- CDS anchor `win_chr10_000116477` in *Dinomys branickii* maps both policies to `DinBra_scaffold_14185` on the plus strand. Full window uses four fragments and emits `[89487, 89742)`; center-1 uses one fragment and emits `[89485, 89740)`. Both retain 249 aligned anchor bases and three internal unaligned bases.
- Enhancer anchor `enh_001515628` in *Craseonycteris thonglongyai* is rejected by full window because 11 fragments span five scaffolds. Center-1 accepts the unique center locus at `CraTho_scaffold_257521:[922,1177)` on the plus strand, with 250 aligned anchor bases and an aligned center nucleotide.

## Matched training

All four arms used the same 40,960,000 training sequences and 10,485,760,000 tokens.
The exact issue-417 CDS full-window arm was reused; it was not retrained.
The other three arms were trained through step 4,999.

| Region | Policy | Training rows | Effective epochs | Run |
|---|---|---:|---:|---|
| CDS | Full window | 66,552,602 | 0.615 | Reused from #417 |
| CDS | Center 1 | 68,657,166 | 0.597 | New |
| Enhancer | Full window | 24,889,396 | 1.646 | New |
| Enhancer | Center 1 | 24,616,580 | 1.664 | New |

The final validation losses for the three new arms were 1.303 for CDS center-1, 1.3404 for enhancer full window, and 1.344 for enhancer center-1.
The public W&B runs are linked below.

## Development evaluation

Every arm was evaluated at steps 1,000 through 4,500 in 500-step increments and at step 4,999.
The complete matrix contains 90 score artifacts and 90 official metric artifacts.
Only region-relevant subsets are presented here: missense, splicing, and synonymous for CDS; distal for enhancers.

| Region | Subset | Metric | Full window | Center 1 | Difference | Paired 95% interval |
|---|---|---|---:|---:|---:|---:|
| CDS | Missense | AUPRC | 0.108 | 0.357 | +0.249 | [0.213, 0.285] |
| CDS | Missense | Group SMD | 0.035 | 0.792 | +0.756 | [0.653, 0.862] |
| CDS | Splicing | AUPRC | 0.151 | 0.388 | +0.237 | [0.184, 0.295] |
| CDS | Splicing | Group SMD | 0.384 | 0.765 | +0.381 | [0.266, 0.493] |
| CDS | Synonymous | AUPRC | 0.110 | 0.296 | +0.185 | [0.079, 0.308] |
| CDS | Synonymous | Group SMD | 0.041 | 0.635 | +0.594 | [0.299, 0.977] |
| Enhancer | Distal | AUPRC | 0.367 | 0.312 | -0.056 | [-0.100, -0.016] |
| Enhancer | Distal | Group SMD | 0.488 | 0.374 | -0.114 | [-0.186, -0.058] |

_Terminal Mendelian development metrics. Positive differences favor center-1; uncertainty uses 1,000 aligned match-group bootstrap draws._

<p align="center">
  <img src="figures/473/cds_group_smd_paired_delta_trajectories.svg" alt="Three panels showing center-1 minus full-window paired Mendelian Group SMD trajectories for CDS missense, splicing, and synonymous variants" />
</p>

<p align="center">
  <img src="figures/473/enhancer_group_smd_paired_delta_trajectories.svg" alt="One panel showing the center-1 minus full-window paired Mendelian Group SMD trajectory for distal enhancer variants" />
</p>

Complex uses the official `abs_llr_avg` AUPRC.
SGE uses official assay-macro `minus_llr_avg` AUPRC.
These official endpoints provide point estimates and bootstrap SEs rather than a paired policy-difference interval.

| Region | Dataset | Subset | Full window | Center 1 | Difference |
|---|---|---|---:|---:|---:|
| CDS | Complex | Missense | 0.1068 ± 0.0082 | 0.1721 ± 0.0155 | +0.0653 |
| CDS | Complex | Splicing | 0.0841 ± 0.0203 | 0.1186 ± 0.0495 | +0.0345 |
| CDS | Complex | Synonymous | 0.1841 ± 0.0753 | 0.1097 ± 0.0197 | -0.0744 |
| CDS | SGE | Missense | 0.1627 ± 0.0049 | 0.2757 ± 0.0086 | +0.1131 |
| CDS | SGE | Splicing | 0.1971 ± 0.0155 | 0.4308 ± 0.0234 | +0.2337 |
| Enhancer | Complex | Distal | 0.1205 ± 0.0084 | 0.1220 ± 0.0086 | +0.0015 |

On the exact shared chromosome-18 projection rows, terminal center-1 minus full-window case-weighted NLL was -0.04323 [-0.04614, -0.04038] for CDS.
The difference favored center-1 at every checkpoint.
For enhancers it was +0.002955 [0.002476, 0.003452] at step 4,999, but the trajectory changed sign across checkpoints.
This intersection analysis uses unlabeled projection sequence and is a fit diagnostic, not a held-out VEP result.

## Evidence and reproducibility

- Projection QC bundle: `s3://oa-bolinas/snakemake/analysis/issue473/results/ff8aaa8a8479e074751264c880d7034167a1654d/d0e5380a46cd66d4c42d763b3c42da1150c92073/bf8367c285f955407cfb2dba6102661b2e528261b64fe52095d52a688cd6d039/projection_report_v3/`
- Sample flank trace: `s3://oa-bolinas/snakemake/analysis/issue473/results/a2026df5c995d4e008884952511f2d344ee63130/d0e5380a46cd66d4c42d763b3c42da1150c92073/bf8367c285f955407cfb2dba6102661b2e528261b64fe52095d52a688cd6d039/trace_flanks_v1/`
- Final development analysis: `s3://oa-bolinas/snakemake/analysis/evals_v2/results/issue473/ae90f6d9e4b23ebe8fb1bd2314baa66cb82b37c1/analysis/89bcd07baf27d0326bf6efbbf101c6204fb0a7db/`
- Shared-row loss analysis: `s3://oa-bolinas/snakemake/analysis/evals_v2/results/issue473/ae90f6d9e4b23ebe8fb1bd2314baa66cb82b37c1/intersection_loss/`
- Public datasets: [CDS center-1](https://huggingface.co/datasets/marin-dna/vertebrate-v1-issue473-center1-cds/tree/4d9a04ab6c4a6e445345fe35fbe2be41b43e7938), [enhancer full window](https://huggingface.co/datasets/marin-dna/vertebrate-v1-issue473-fullwindow-ccre-enhancer-centered/tree/ffb9c63fae72311fb457640af9c8365b84f0edf8), and [enhancer center-1](https://huggingface.co/datasets/marin-dna/vertebrate-v1-issue473-center1-ccre-enhancer-centered/tree/23d1531f63998b5716e7895a74437e0568186bd1).
- Public W&B runs: [CDS center-1](https://wandb.ai/gonzalobenegas/marin/runs/dna-exp473-0p25b-cds_center_1-v1), [enhancer full window](https://wandb.ai/gonzalobenegas/marin/runs/dna-exp473-0p25b-enhancer_full_window-v1), and [enhancer center-1](https://wandb.ai/gonzalobenegas/marin/runs/dna-exp473-0p25b-enhancer_center_1-v1).
- Analysis snapshot: `89bcd07baf27d0326bf6efbbf101c6204fb0a7db`; experiment identity: `ae90f6d9e4b23ebe8fb1bd2314baa66cb82b37c1`.

All evaluation data came from the official public `evals_v2` `train.parquet` files.
No held-out even-autosome or chromosome-Y label, prediction, effect measurement, or aggregate metric was requested or read.
Enhancer checkpoints were not submitted to SGE.
All internal genomic coordinates are 0-based and half-open.

## Limitations

- This is one training seed. All eight relevant paired Mendelian endpoint cells met the recorded additional-seed evidence gate, but no extra seed was launched.
- Arms were matched on sequences and tokens, not epochs, because the policies materialize different row counts.
- Reverse-trace QC covers 214 retained named PSL files and 5,328 species-anchor rows rather than the full accepted corpus.
- Recovery is a quantity measure, not proof that every emitted window is the correct homologous locus.
- Center-1 and full window are the only policies compared; multiple-landmark and fragment-selection alternatives remain untested.
- The conclusion is specific to CDS and enhancer-centered cCRE training arms. Other region classes have projection QC but no matched downstream training comparison here.
- Held-out test labels remain untouched, so the accepted interpretation is development-only.

## Related questions

- [How should genomic anchors be selected and projected across species?](../questions/genomic-anchors.md)

## Research record

- [Experiment issue #473](https://github.com/Open-Athena/marin-dna/issues/473)
- [Projection/data PR #477](https://github.com/Open-Athena/marin-dna/pull/477)
