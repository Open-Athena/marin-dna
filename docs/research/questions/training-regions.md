# Which genomic regions to train on, and how to find them?

> [!NOTE]
> **TL;DR:** Targeted or conservation-selected corpora often improve functional prediction at current scales, while loss or entropy from a trained model is a practical conservation proxy; a five-checkpoint loss slope did not improve conservation ranking, and no causal benefit from likelihood-derived weighting or repeat downweighting has been established.

## Question

Which regions of a genome should a gLM use for pretraining, and how should we obtain those regions when high-quality annotations or whole-genome alignments are unavailable?
For functional tasks such as variant-effect prediction, does enriching for constrained or annotated functional sequence outperform whole-genome training, or does that advantage disappear at sufficient model/data scale or with a different loss weighting or training objective?
Can cheap local alignments or a learned single-sequence predictor recover useful functional sequence without a full multiple-genome alignment?
Does repeat downweighting help, and how does its value change with model and data scale?
The answer may be task-dependent: sequence evolving approximately neutrally may be low-value for functional-constraint prediction but central to phylogenetic or mutation-process questions.

## Current answer

No experiment identifies one universally optimal genomic training footprint.
Current MarinDNA results and external ablations support task-aware enrichment at modest scale: region specialists beat mismatched specialists, clean enhancer curation fixes a large VEP failure, and the conservation-filtered footprint covers Mendelian positives much better than complex-trait positives.

The design has three separable controls.
Locus selection decides which sequence is present; sampling decides exposure frequency; per-base loss weights decide which observations shape the fitted distribution.
A hard filter, importance sampling, and loss weighting should not be treated as interchangeable because they change coverage, repetition, and likelihood semantics differently.

The current 100-fold repeat downweighting has not been ablated across scales.
We do not know whether it improves language modeling or downstream performance, or whether any benefit shrinks or grows with scale.
Uniform loss is the simpler prior.
If removing repeat-specific weights preserves performance at the scales and tasks we care about, we should remove them.
This would eliminate a special-case training heuristic and make likelihoods easier to interpret.

The [conservation and repeat predictability experiment](../experiments/478-conservation-repeat-predictability.md) found that, among nonrepeat human CDS, upstream, and downstream positions, absolute loss and entropy ranked conservation increasingly well with model scale and dominated same-corpus loss deltas at comparable FWD scoring compute.
Controlled scale-dependent loss reduction remained associated with conservation, but it is distinct from target-distribution reducible loss.
One orientation preserved the aggregate classification result at half the inference compute without preserving the exact per-base ranking.
The inference-only audit did not test whether any score improves training and could not separate training exposure or homology effects.
The [likelihood-dynamics experiment](../experiments/489-likelihood-dynamics.md) found conservation ranking by 21B tokens and terminal loss outperforming the five-checkpoint loss slope in every region.
Its trajectory groups remain descriptive, and neither score was tested as a training selector.

The leading hypothesis is that increasing the density of constrained or correctly annotated sequence improves functional-VEP sample efficiency at fixed compute.
Whole-genome data may become more useful at larger scale, under weighting that prevents easy background from dominating, or for mutation-process, repeat, phylogeny, and regional-context tasks.
Confidence is moderate for functional enrichment at current scales and low on how the optimum moves with model size and task.

The next decisive comparison should cross footprint choice with scale while matching training tokens and reporting unique loci, realized repetitions, contamination, and leakage.
It should retain a background arm so gains on functional VEP can be weighed against losses on outcomes that need neutral or repetitive sequence.

<details>
<summary>Related work</summary>

- [GPN-MSA](https://pmc.ncbi.nlm.nih.gov/articles/PMC10592768/) selected the top 5% of 128 bp windows by conservation, retained a small random background sample, upweighted conserved positions, and downweighted repeats.
  Its matched ablation favored conserved-region training for VEP.
  This supports functional enrichment at that model size and objective.
  It does not prove that neutral sequence is useless or determine how the result changes with scale.
- [Rho-1](https://arxiv.org/abs/2404.07965) prioritizes tokens by reducible loss: the online model's token loss minus the loss of a reference model trained on the target or high-quality distribution.
  The reference model may be smaller for efficiency, but its target-distribution training, not its size, defines the score.
  It also reports same-corpus self-reference selection using low frozen-reference loss and entropy.
  A loss difference between two same-corpus model sizes instead measures scale-dependent learnability and may rank tokens differently from Rho-1 reducible loss.
  In genomic data, repeats, local composition, and phylogenetic redundancy may all produce high predictability without functional constraint.
- [Self-paced learning](https://papers.nips.cc/paper_files/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html) introduces current small-loss examples first and anneals toward the full dataset, while [Selective Backprop](https://arxiv.org/abs/1910.00762) prioritizes current high-loss examples because low-loss gradients tend to be small.
  A permanent hard mask on the student's lowest-loss tokens changes the target distribution and can reinforce already-learned tokens; it is distinct from self-distillation and reducible-loss selection.

</details>

<details>
<summary>Related experiments</summary>

- [#8](https://github.com/Open-Athena/marin-dna/issues/8) established functional-versus-background likelihood gaps as a training diagnostic.
  It provides a readout for footprint experiments but also shows that likelihood gaps need not track VEP under contamination.
- [#87](https://github.com/Open-Athena/marin-dna/issues/87) scopes repeat and loss-weighting ablations across FLOP budgets.
  It is the direct scale-dependent weighting experiment, but no completed result is recorded.
- [#120](https://github.com/Open-Athena/marin-dna/issues/120) compared cheap pairwise methods for recovering cross-species enhancer sequence.
  It showed that local similarity search can acquire useful functional candidates at lower compute than sensitive whole-genome alignment, with a measurable recall frontier.
- [#177](https://github.com/Open-Athena/marin-dna/issues/177) audited N stretches and repeat masking in the Zoonomia corpus.
  Major corruption was not found, and the existing 100-fold repeat downweighting was documented; the biological value of repeat classes remains unresolved.
- [#213](https://github.com/Open-Athena/marin-dna/issues/213) measured conservation-footprint coverage.
  The production cutoff covers 83.9% of Mendelian positives and 42.6% of complex-trait positives, with especially low distal complex-trait coverage.
- [#232](https://github.com/Open-Athena/marin-dna/issues/232) trained matched region specialists and a background control.
  Every specialist won its home Mendelian class and the background model won none, showing that region-matched distributions matter at this scale.
- [#326](https://github.com/Open-Athena/marin-dna/issues/326) removed exon overlap and mixed cCRE classes from the enhancer arm.
  Distal AUPRC rose from 0.127 to about 0.30 while splicing leakage disappeared, showing that broad functional labels are insufficient without contamination control.
- [#351](https://github.com/Open-Athena/marin-dna/issues/351) compared enhancer-centered and clean tiled windows.
  Centering was suggestively better for distal VEP, but unequal epoch counts and non-converged curves confounded functional-base density, placement, and repetition.
- [#353](https://github.com/Open-Athena/marin-dna/issues/353) compared human-anchored CDS projection with native per-species annotation across vertebrate and animal scopes.
  Projection produced useful conserved-CDS data but lost distant species and did not dominate annotation on every endpoint.
- [Conservation and repeat predictability across model scale](../experiments/478-conservation-repeat-predictability.md) measured per-base loss and entropy across the fixed-token 46M–4B ladder in CDS, upstream, and downstream sequence.
  Conserved nonrepeat bases had positive composition-adjusted scale gains in all three regions, qualifying same-corpus scale-differential loss for a causal weighting test while leaving repeat downweighting unresolved.
  Absolute loss and entropy classified conservation increasingly well with scale and dominated every FWD loss delta at comparable estimated scoring compute.
  FWD-only and RC-only classification AUPRC was nearly identical, while their imperfect endpoint per-base agreement limits a single pass as an exact replacement for averaged weights.
- [Likelihood-derived token rankings through m1.3 training](../experiments/489-likelihood-dynamics.md) measured one 1B lineage at five cumulative-token checkpoints across five genomic regions, finding that loss and entropy ranked conservation by the earliest checkpoint while terminal loss outperformed a five-checkpoint loss slope globally and in every region.
  The trajectory groups differed in conservation and functional-region composition, but remain descriptive and do not establish a training benefit.

</details>

<details>
<summary>Possible directions</summary>

- Compare whole-genome, conservation-filtered, annotation-enriched, and functional-plus-background corpora at matched architecture, tokens, and evaluation, reporting unique loci and realized repetitions.
- Separate locus selection, exposure frequency, and per-base loss weighting, including data-size- and epoch-matched controls.
- Test terminal loss or entropy, target-distribution reducible loss, and same-corpus scale-differential loss as distinct selectors; control for repeats, GC, local predictability, training exposure, and homology density.
- Measure the footprint tradeoff across Mendelian and complex-trait VEP, region-matched likelihood gaps, frozen probes, and at least one outcome expected to benefit from neutral sequence.
- Ablate the current 100-fold repeat downweighting across model and token scales while holding footprint and sampling fixed.
- Compare conservation or whole-genome alignment with direct annotation, cheap local alignment, and learned single-sequence selection only after the target distribution and leakage contract are fixed.

</details>
