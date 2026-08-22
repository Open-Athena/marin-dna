# Center-seeded projection for vertebrate training data

> [!NOTE]
> **TL;DR:** One-seed development AUPRC trajectories were broadly similar for full-window and center-1-bp projection in CDS and enhancer specialists; center-1 is the default for new projection datasets because its contract is simpler and it increased aggregate species–anchor recovery by 2.68 percentage points, while existing full-window artifacts remain valid historical controls.

## Findings

Center-1-bp projection is a simpler contract than full-window projection.
It projects the central human nucleotide, requires one target locus, and extracts a fixed 255 bp target-genome window around the mapped nucleotide.
Full-window projection must instead reconcile all compatible fragments, enforce a 128–512 bp target-span gate, and resize around the accepted span midpoint.

Center-1 did not improve recovery in every region.
Across the five audited regions, it recovered 82.28% of requested species–anchor pairs, compared with 79.60% for full-window projection.
It increased CDS, ncRNA-exon, and 3′-UTR recovery, while enhancer-centered cCRE and TSS/5′-UTR recovery decreased.
Both policies reached jawless vertebrates.

The sampled alignment trace did not show a general increase in external flank under center-1.
Four of five paired external-flank intervals included zero.
The 3′-UTR estimate was 1.143 additional bases [0.387, 2.204].

At matched tokens, corrected one-seed development AUPRC trajectories showed no consistent advantage for either projection policy.
The two policies tracked closely for CDS Mendelian missense and splicing, Complex-trait missense, and SGE missense.
CDS synonymous, CDS SGE splicing, and enhancer Mendelian distal varied more, but their direction was not reproduced across the other relevant endpoints.

Based on these development results and the simpler projection contract, Gonzalo Benegas selected center-1 as the default for new projection datasets.
This is an operational choice from one seed, not a statistical-equivalence claim.
Existing full-window rules and artifacts remain available for reproducibility and historical comparisons.

## Evidence

### Projection QC

The producer completed all 135 projection partitions.
The accepted union contains 74,524,203 species–anchor rows.
Every emitted interval is 255 bp, coordinates are in bounds, and chromosome, strand, and sequence-orientation checks passed.
The median projected-center displacement between policies is 3 bp.

<p align="center">
  <img src="figures/473/projection_qc.svg" alt="Species-anchor recovery by projection policy and paired external-flank differences across five genomic regions" />
</p>

_The left panel shows exact recovery over every requested species–anchor pair; these are complete counts rather than sample estimates.
The right panel shows paired center-1-bp minus full-window external-flank differences in the named-PSL trace sample; error bars are 95% bootstrap intervals over human anchors._

| Region | Full window | Center 1 bp | Difference |
|---|---:|---:|---:|
| Enhancer-centered cCRE | 80.72% | 79.83% | -0.892 pp |
| CDS | 84.10% | 86.78% | +2.679 pp |
| ncRNA exon | 67.64% | 77.85% | +10.208 pp |
| TSS / 5′ UTR | 77.80% | 74.66% | -3.135 pp |
| 3′ UTR | 76.99% | 79.76% | +2.766 pp |
| All five regions | 79.60% | 82.28% | +2.68 pp |

The 5,328-row trace sample reproduced every producer aligned-base count exactly.
Its paired policy comparisons used 1,000 bootstrap draws over human anchors.
Recovery is a quantity measure; it does not by itself establish that every accepted target window is the correct homologous locus.

### Matched training and validation loss

All four arms used 40,960,000 training sequences, 10,485,760,000 tokens, the same 0.25B architecture, and one training seed.
The issue #417 CDS full-window arm was reused rather than retrained.

| Region | Policy | Available training rows | Effective epochs | Run |
|---|---|---:|---:|---|
| CDS | Full window | 66,552,602 | 0.615 | Reused from #417 |
| CDS | Center 1 bp | 68,657,166 | 0.597 | New |
| Enhancer | Full window | 24,889,396 | 1.646 | New |
| Enhancer | Center 1 bp | 24,616,580 | 1.664 | New |

A valid cross-policy validation-loss trajectory is unavailable.
Each native run used 16,384 chromosome-18 rows drawn from its own projection-policy dataset, so the rows differ between policies.
The reused issue #417 full-window run also lacks a W&B validation curve because logging was disabled during guarded resumptions.
An earlier shared-row replay used the incompatible legacy-RoPE loading path and was withdrawn.

Exact offline replays reproduced the modest late validation-loss increases in the new arms.
Similar non-monotonic behavior occurred in the corrected issue #417 full-window model, whose downstream VEP performance was successful.
These within-arm diagnostics do not isolate a projection-policy effect, so no validation-loss figure is used for the policy decision.

### Development evaluation

All results below use the official public `evals_v2` development `train` split containing odd-numbered autosomes and chromosome X.
Mature-miRNA groups were excluded before aggregation.
No even-autosome or chromosome-Y label, prediction, effect measurement, or aggregate metric was evaluated.

The CDS report is restricted to Mendelian missense, splicing, and synonymous variants; Complex-trait missense; and SGE missense and splicing.
The repaired full-window checkpoints were loaded through the maintained legacy-RoPE compatibility boundary.
The center-1 checkpoints already carried the intended dual-schema RoPE metadata.

<p align="center">
  <img src="figures/473/cds_auprc_by_projection.svg" alt="CDS development AUPRC trajectories for full-window and center-1-bp projection across six relevant benchmark subsets" />
</p>

_Actual AUPRC at all nine saved checkpoints from steps 1,000 through 4,999.
Vertical uncapped error bars are ±1 SE from the producing analysis artifact.
The gray dashed line is positive-label prevalence.
Each panel uses its own prevalence-anchored y-axis._

| Dataset | Subset | Full window | Center 1 bp | Center 1 bp − full window |
|---|---|---:|---:|---:|
| Mendelian | Missense | 0.3439 | 0.3571 | +0.0131 [-0.0081, 0.0343] |
| Mendelian | Splicing | 0.3916 | 0.3878 | -0.0037 [-0.0385, 0.0309] |
| Mendelian | Synonymous | 0.3465 | 0.2956 | -0.0509 [-0.1180, 0.0158] |
| Complex traits | Missense | 0.1737 ± 0.0153 | 0.1721 ± 0.0155 | -0.0016 |
| SGE | Missense | 0.2645 ± 0.0079 | 0.2757 ± 0.0086 | +0.0112 |
| SGE | Splicing | 0.4729 ± 0.0228 | 0.4308 ± 0.0234 | -0.0421 |

_Mendelian brackets are paired 95% match-group bootstrap intervals from 1,000 draws.
Complex-trait and SGE entries show per-policy ±1 SE; paired policy-difference intervals were not computed for those datasets._

For the secondary Mendelian Group SMD metric, all three corrected terminal CDS paired intervals included zero.
The center-1-bp minus full-window terminal differences were -0.006 for missense, -0.004 for splicing, and +0.027 for synonymous variants.

The enhancer report is restricted to Mendelian and Complex-trait distal variants.
SGE is omitted because it has no region-relevant enhancer endpoint.

<p align="center">
  <img src="figures/473/enhancer_auprc_by_projection.svg" alt="Enhancer development AUPRC trajectories for full-window and center-1-bp projection on Mendelian and Complex-trait distal variants" />
</p>

_Actual distal AUPRC at all nine saved checkpoints from steps 1,000 through 4,999.
Vertical uncapped error bars are ±1 SE from the producing analysis artifact.
The gray dashed line is positive-label prevalence.
Each panel uses its own prevalence-anchored y-axis._

| Dataset | Subset | Full window | Center 1 bp | Center 1 bp − full window |
|---|---|---:|---:|---:|
| Mendelian | Distal | 0.3675 | 0.3118 | -0.0557 [-0.100, -0.016] |
| Complex traits | Distal | 0.1205 ± 0.0084 | 0.1220 ± 0.0086 | +0.0015 |

_The Mendelian bracket is a paired 95% match-group bootstrap interval from 1,000 draws.
Complex-trait entries show per-policy ±1 SE; a paired policy-difference interval was not computed._

The terminal enhancer Mendelian contrast favored full-window projection, including on secondary Group SMD.
That result was not replicated across seeds and did not appear in distal Complex traits.
The accepted default therefore weighs the complete one-seed trajectory set, aggregate recovery, and implementation simplicity rather than treating the terminal enhancer Mendelian cell as a universal policy effect.

### Reproducibility

- Projection QC bundle: `s3://oa-bolinas/snakemake/analysis/issue473/results/ff8aaa8a8479e074751264c880d7034167a1654d/d0e5380a46cd66d4c42d763b3c42da1150c92073/bf8367c285f955407cfb2dba6102661b2e528261b64fe52095d52a688cd6d039/projection_report_v3/`
- Sample flank trace: `s3://oa-bolinas/snakemake/analysis/issue473/results/a2026df5c995d4e008884952511f2d344ee63130/d0e5380a46cd66d4c42d763b3c42da1150c92073/bf8367c285f955407cfb2dba6102661b2e528261b64fe52095d52a688cd6d039/trace_flanks_v1/`
- Corrected CDS development analysis: `s3://oa-bolinas/snakemake/analysis/evals_v2/results/issue473/ae90f6d9e4b23ebe8fb1bd2314baa66cb82b37c1/damage_control/cb2fe372485d8287fa72a4e0faeae1d80b830178/`
- Enhancer development analysis: `s3://oa-bolinas/snakemake/analysis/evals_v2/results/issue473/ae90f6d9e4b23ebe8fb1bd2314baa66cb82b37c1/analysis/89bcd07baf27d0326bf6efbbf101c6204fb0a7db/`
- Public datasets: [CDS center-1](https://huggingface.co/datasets/marin-dna/vertebrate-v1-issue473-center1-cds/tree/4d9a04ab6c4a6e445345fe35fbe2be41b43e7938), [enhancer full window](https://huggingface.co/datasets/marin-dna/vertebrate-v1-issue473-fullwindow-ccre-enhancer-centered/tree/ffb9c63fae72311fb457640af9c8365b84f0edf8), and [enhancer center-1](https://huggingface.co/datasets/marin-dna/vertebrate-v1-issue473-center1-ccre-enhancer-centered/tree/23d1531f63998b5716e7895a74437e0568186bd1).
- Public W&B runs: [CDS center-1](https://wandb.ai/gonzalobenegas/marin/runs/dna-exp473-0p25b-cds_center_1-v1), [enhancer full window](https://wandb.ai/gonzalobenegas/marin/runs/dna-exp473-0p25b-enhancer_full_window-v1), and [enhancer center-1](https://wandb.ai/gonzalobenegas/marin/runs/dna-exp473-0p25b-enhancer_center_1-v1).

All internal genomic coordinates are 0-based and half-open.

## Limitations

- This is one training seed and does not establish statistical equivalence.
- The terminal enhancer Mendelian AUPRC and Group SMD comparisons favored full-window projection; no additional seed tested whether that endpoint-level result replicates.
- The four arms were matched on presented sequences and tokens, not effective epochs, because the projection policies materialized different row counts.
- No valid shared-row validation-loss comparison remains after withdrawing the incompatible legacy-RoPE replay.
- The reverse trace is a deterministic sample rather than the full accepted corpus.
- Recovery and external-flank measurements do not prove that every emitted window is the correct homologous locus.
- Only full-window and center-1-bp policies were compared.
- CDS and enhancer-centered cCREs have matched downstream training comparisons; the other three regions have projection QC only.
- Held-out even-autosome and chromosome-Y VEP data remain untouched, so the interpretation is development-only.

## Related questions

- [How should genomic anchors be selected and projected across species?](../questions/genomic-anchors.md)

## Research record

- [Experiment issue #473](https://github.com/Open-Athena/marin-dna/issues/473)
