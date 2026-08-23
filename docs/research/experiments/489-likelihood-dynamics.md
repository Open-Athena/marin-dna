# Likelihood-derived token rankings through m1.3 training

> [!NOTE]
> **TL;DR:** In the fixed 1B m1-to-m1.3 lineage, conserved nonrepeat positions had lower loss throughout training and terminal loss ranked conservation better than the five-checkpoint loss slope; fitted trajectory groups differed in conservation prevalence and functional-region composition but remain descriptive.

## Findings

<p align="center">
  <img src="figures/489/token-composition.svg" alt="Six two-by-two heatmaps showing percentages of conserved and non-conserved positions crossed with repeat status globally and in five genomic regions" />
</p>

_Case-encoded conservation × reference-matched repeat composition in the central span; percentages sum to 100 within each scope._

Repeats accounted for 10.5% of positions globally, ranging from 4.8% in ncRNA to 17.6% downstream.

<p align="center">
  <img src="figures/489/nonrepeat-conservation-loss.svg" alt="Six square panels showing mean forward loss for conserved and non-conserved nonrepeat positions across cumulative training tokens globally and in five genomic regions" />
</p>

_Exact pooled forward loss in nats/base at nonrepeat central positions; each point is one checkpoint, and the y-axis range is specific to each panel._

Conserved positions had lower forward loss than nonconserved positions at every observed checkpoint in every scope.
The gap widened through training in the global, CDS, upstream, downstream, and enhancer panels.
ncRNA began with the largest gap and changed less than the other regions.

<p align="center">
  <img src="figures/489/conservation-classification-auprc.svg" alt="Six square panels showing exact pooled conservation-classification AUPRC for loss and entropy across cumulative training tokens globally and in five genomic regions" />
</p>

_Exact pooled AUPRC among nonrepeat central positions; the global panel combines the two conservation-label definitions documented below, dashed lines are scope-specific prevalence, and the y-axis range is specific to each panel._

Lower forward loss and four-nucleotide predictive entropy ranked case-encoded conservation by the earliest available checkpoint.
Global loss AUPRC increased from 0.502 at 21.0B cumulative tokens to 0.601 at 173.7B, while entropy AUPRC increased from 0.492 to 0.599.
The trend strengthened in CDS, upstream, downstream, and enhancer sequence, while ncRNA loss AUPRC was nearly flat and peaked before the terminal checkpoint.

<p align="center">
  <img src="figures/489/global-trajectory-loss.svg" alt="Single square panel showing mean loss across cumulative training tokens for four global high-to-high, low-to-high, high-to-low, and low-to-low fitted trajectory groups" />
</p>

_Mean forward loss for four global trajectory groups among nonrepeat central positions; H and L denote fitted endpoint loss above and below the fitted global population mean, curves are exact pooled checkpoint means, and bands are 95% genomic-block bootstrap intervals._

The high-to-low group fell from 1.514 to 0.278 nats/base, while the low-to-high group rose from 1.025 to 1.192 nats/base.
These groups describe fitted loss endpoints relative to global thresholds; they are not quantiles or biological classes.

<p align="center">
  <img src="figures/489/global-trajectory-conservation.svg" alt="Compact bar chart showing the proportion of conserved positions in four global fitted loss-trajectory groups" />
</p>

_Exact pooled conserved-position prevalence within each global fitted trajectory group among nonrepeat central positions; the global aggregate combines the two conservation-label definitions documented below._

Terminal-low groups were about twice as often conserved as terminal-high groups.

<p align="center">
  <img src="figures/489/trajectory-region-composition.svg" alt="Four horizontal stacked bars showing the CDS, upstream, downstream, ncRNA, and enhancer composition of each global fitted loss-trajectory group" />
</p>

_Exact functional-region composition within each global fitted trajectory group among nonrepeat central positions; every bar sums to 100%, and the region labels denote validation-panel membership._

High-to-low was enhancer-enriched, while low-to-low was ncRNA-enriched.
The regions are sampled validation panels rather than mutually exclusive genome annotations.

<p align="center">
  <img src="figures/489/loss-level-vs-slope-auprc.svg" alt="Dot plot comparing conservation AUPRC from prevalence, first-checkpoint loss, terminal loss, and five-checkpoint loss slope globally and in five functional regions" />
</p>

_Exact pooled conservation AUPRC among nonrepeat central positions; loss slope is the negative per-position ordinary-least-squares slope across all five checkpoints, and prevalence is the no-skill baseline._

Global loss-slope AUPRC was 0.448, below first-checkpoint loss at 0.502 and terminal loss at 0.601.
CDS was the only region where slope beat first-checkpoint loss, at 0.629 versus 0.535, while terminal loss remained higher at 0.686.
In ncRNA, slope AUPRC was 0.410, below its 0.429 prevalence.
The simple slope is retrospective, uses five model evaluations, and does not improve global conservation ranking over absolute loss.

## Evidence

The experiment followed one 1B causal lineage through five on-path checkpoints.

| Checkpoint | Cumulative training tokens |
| --- | ---: |
| m1 step 10,000 | 20,971,520,000 |
| m1.1 step 30,000 | 62,914,560,000 |
| m1.2 step 50,000 | 104,857,600,000 |
| m1.3 step 70,000 | 146,800,640,000 |
| m1.3 step 82,823 | 173,692,420,096 |

Each checkpoint scored one forward orientation over complete 16,384-window CDS, upstream, downstream, ncRNA, and enhancer probes.
The [CDS](https://huggingface.co/datasets/marin-dna/genomes-v5-validation-intervals-v5_255_255/tree/daff592f213aaa1cab1711d477a79ff6b1bc4ef4), [upstream](https://huggingface.co/datasets/marin-dna/genomes-v5-validation-intervals-v1_255_255/tree/a761bc0b663a9827303f3112e4667d53d5326fac), and [downstream](https://huggingface.co/datasets/marin-dna/genomes-v5-validation-intervals-v15_255_255/tree/d7b27eecd68453934ebb3e7e6e78d5401789faa5) probes use RefSeq `GCF_000001405.40`, `NC_*` sequence names, and 241-way phyloP >= 2.27.
The [ncRNA](https://huggingface.co/datasets/marin-dna/zoonomia-v1-val_ncrna/tree/76a18c1bbf07ac9bd064722431bbdab894b9e6c6) and [enhancer](https://huggingface.co/datasets/marin-dna/zoonomia-v1-val_enhancer/tree/d40d1e067b2a56ac812af122de029eb79cab1106) probes use the Ensembl release 115 GRCh38 soft-masked primary assembly, Ensembl sequence names, and 447-mammal phyloP >= 2.2162.
Repeat labels came from each probe's matching soft-masked reference, with uppercase sequence equality required before joining.
The primary analysis retained target positions `[32, 223)` in 0-based half-open coordinates and excluded repeats, ambiguous targets, and unscorable targets.

AUPRC is exact pooled average precision from negative loss or negative entropy.
The loss-slope score is the negative ordinary-least-squares coefficient from fitting each position's NLL against cumulative training tokens at all five checkpoints.

Each position's five losses were fit against cumulative tokens; fitted earliest and terminal losses were classified against global means of 1.153 and 0.969 nats/base.
The grouping used no region or conservation labels.
Trajectory intervals used 2,000 bootstrap replicates over region-specific 10 Mb genomic blocks with seed 489.

## Limitations

- The first observed checkpoint is already at 21.0B tokens.
  The experiment cannot identify when conservation ranking first emerged before that point.
- One model lineage and one checkpoint per training stage were measured.
  Seed variation and alternative mixtures are unobserved.
- Validation casing is an incomplete proxy for function.
  Lineage-specific, weakly conserved, or unaligned functional sequence can be labeled negative.
- The panels mix assemblies and quantile-matched cutoffs from different phyloP alignments.
  Global metrics are mixed-definition aggregates, and region comparisons are not matched causal contrasts.
- The trajectory groups use fitted loss endpoints relative to global means and are descriptive.
  The labels combine intercept and slope relative to those thresholds and do not define biological classes.
- The loss-slope score uses the same five evaluations summarized elsewhere on the page.
  Starting loss and fitted slope are statistically coupled, so the slope result can include regression-to-the-mean structure and is not an independent prospective score.
- The experiment is inference-only.
  No selector was used to train a student, so downstream benefit and likelihood calibration remain unknown.

## Related questions

- [Which genomic regions to train on, and how to find them?](../questions/training-regions.md)

## Research record

- [Experiment issue #489](https://github.com/Open-Athena/marin-dna/issues/489)
