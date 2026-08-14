# Which genomic regions to train on, and how to find them?

## Metadata

| Field | Value |
|---|---|
| Question ID | `RQ-0395` |
| Status | `active` |
| Overall confidence | `medium` |
| Evidence considered through | `2026-08-13` |
| Predecessor issues | [#395](https://github.com/Open-Athena/marin-dna/issues/395) |

## Question and scope

Which regions of a genome should a gLM use for pretraining, and how should we obtain those regions when high-quality annotations or whole-genome alignments are unavailable? For functional tasks such as variant-effect prediction, does enriching for constrained or annotated functional sequence outperform whole-genome training, or does that advantage disappear at sufficient model/data scale or with a different loss weighting or training objective? Can cheap local alignments or a learned single-sequence predictor recover useful functional sequence without a full multiple-genome alignment, and how should repetitive elements be treated? The answer may be task-dependent: sequence evolving approximately neutrally may be low-value for functional-constraint prediction but central to phylogenetic or mutation-process questions.

## Current answer

Training-footprint choice materially affects functional prediction, and targeted or conservation-selected corpora often beat naive whole-genome sampling at current scales. No experiment establishes a universal region-selection rule across scale and tasks; confidence is moderate that task-aware enrichment helps current models, with unresolved tradeoffs around neutral sequence, repeats, and learned selection.

No experiment identifies one universally optimal genomic training footprint. Current MarinDNA results and external ablations support task-aware enrichment at modest scale: region specialists beat mismatched specialists, clean enhancer curation fixes a large VEP failure, and the conservation-filtered footprint covers Mendelian positives much better than complex-trait positives.

The design has three separable controls. Locus selection decides which sequence is present; sampling decides exposure frequency; per-base loss weights decide which observations shape the fitted distribution. A hard filter, importance sampling, and loss weighting should not be treated as interchangeable because they change coverage, repetition, and likelihood semantics differently.

The leading hypothesis is that increasing the density of constrained or correctly annotated sequence improves functional-VEP sample efficiency at fixed compute. Whole-genome data may become more useful at larger scale, under weighting that prevents easy background from dominating, or for mutation-process, repeat, phylogeny, and regional-context tasks. Confidence is moderate for functional enrichment at current scales and low on how the optimum moves with model size and task.

The next decisive comparison should cross footprint choice with scale while matching training tokens and reporting unique loci, realized repetitions, contamination, and leakage. It should retain a background arm so gains on functional VEP can be weighed against losses on outcomes that need neutral or repetitive sequence.

## Confidence and limitations

Training-footprint choice materially affects functional prediction, and targeted or conservation-selected corpora often beat naive whole-genome sampling at current scales. No experiment establishes a universal region-selection rule across scale and tasks; confidence is moderate that task-aware enrichment helps current models, with unresolved tradeoffs around neutral sequence, repeats, and learned selection.

The leading hypothesis is that increasing the density of constrained or correctly annotated sequence improves functional-VEP sample efficiency at fixed compute. Whole-genome data may become more useful at larger scale, under weighting that prevents easy background from dominating, or for mutation-process, repeat, phylogeny, and regional-context tasks. Confidence is moderate for functional enrichment at current scales and low on how the optimum moves with model size and task.

## Operational consequence

No experiment identifies one universally optimal genomic training footprint. Current MarinDNA results and external ablations support task-aware enrichment at modest scale: region specialists beat mismatched specialists, clean enhancer curation fixes a large VEP failure, and the conservation-filtered footprint covers Mendelian positives much better than complex-trait positives.

## Supporting evidence

- [GPN-MSA](https://pmc.ncbi.nlm.nih.gov/articles/PMC10592768/) selected the top 5% of 128 bp windows by conservation, retained a small random background sample, upweighted conserved positions, and downweighted repeats. Its matched ablation favored conserved-region training for VEP. This supports functional enrichment at that model size and objective. It does not prove that neutral sequence is useless or determine how the result changes with scale.
- Conservation is a proxy for purifying constraint rather than a complete definition of function. It misses lineage-specific and hard-to-align elements, while annotations miss unknown constrained sequence. A useful observable is the joint coverage of evaluation loci, annotated classes, conservation tiers, and repeats under each proposed footprint.
- [#391](https://github.com/Open-Athena/marin-dna/issues/391) synthesizes evidence that the current conservation-filtered footprint undercovers complex-trait positives, especially distal variants. This motivates weakly conserved/background arms but does not show that adding them improves VEP.
- [#392](https://github.com/Open-Athena/marin-dna/issues/392) identifies a long-context version of the same problem: uniformly weighted long windows contain much more background sequence than focal functional sequence. Any long-context pretraining study should report whether selection or weighting changes the distant-context result.

## Contradictory evidence

The predecessor issue did not maintain a separate contradictory-evidence section. Its caveats and negative results are preserved in Current answer and Supporting evidence.

## Related experiments

- [#8](https://github.com/Open-Athena/marin-dna/issues/8) established functional-versus-background likelihood gaps as a training diagnostic. It provides a readout for footprint experiments but also shows that likelihood gaps need not track VEP under contamination.
- [#87](https://github.com/Open-Athena/marin-dna/issues/87) scopes repeat and loss-weighting ablations across FLOP budgets. It is the direct scale-dependent weighting experiment, but no completed result is recorded.
- [#120](https://github.com/Open-Athena/marin-dna/issues/120) compared cheap pairwise methods for recovering cross-species enhancer sequence. It showed that local similarity search can acquire useful functional candidates at lower compute than sensitive whole-genome alignment, with a measurable recall frontier.
- [#177](https://github.com/Open-Athena/marin-dna/issues/177) audited N stretches and repeat masking in the Zoonomia corpus. Major corruption was not found, and the existing 100-fold repeat downweighting was documented; the biological value of repeat classes remains unresolved.
- [#213](https://github.com/Open-Athena/marin-dna/issues/213) measured conservation-footprint coverage. The production cutoff covers 83.9% of Mendelian positives and 42.6% of complex-trait positives, with especially low distal complex-trait coverage.
- [#232](https://github.com/Open-Athena/marin-dna/issues/232) trained matched region specialists and a background control. Every specialist won its home Mendelian class and the background model won none, showing that region-matched distributions matter at this scale.
- [#326](https://github.com/Open-Athena/marin-dna/issues/326) removed exon overlap and mixed cCRE classes from the enhancer arm. Distal AUPRC rose from 0.127 to about 0.30 while splicing leakage disappeared, showing that broad functional labels are insufficient without contamination control.
- [#351](https://github.com/Open-Athena/marin-dna/issues/351) compared enhancer-centered and clean tiled windows. Centering was suggestively better for distal VEP, but unequal epoch counts and non-converged curves confounded functional-base density, placement, and repetition.
- [#353](https://github.com/Open-Athena/marin-dna/issues/353) compared human-anchored CDS projection with native per-species annotation across vertebrate and animal scopes. Projection produced useful conserved-CDS data but lost distant species and did not dominate annotation on every endpoint.

## Open questions

### What outcome are we optimizing?

- Is there one corpus that transfers well across VEP, functional-region representations, sequence generation, genome annotation, and phylogenetic inference, or should MarinDNA maintain task-specific mixtures?
- For VEP, should the primary decision metric be zero-shot AUPRC, frozen-embedding probes, functional/non-functional likelihood gaps, or a combination? Improvements should be reported separately for Mendelian, complex-trait, coding, regulatory, conserved, and weakly conserved variants.
- What evaluation would expose the cost of removing neutral sequence? Candidate outcomes include phylogeny/species inference, mutation-spectrum prediction, regional composition, and held-out whole-genome likelihood, but these should not silently displace functional VEP as the primary task.

### What should count as functional training sequence?

- **Direct annotation:** genes, UTRs, ncRNAs, cCREs, experimentally measured regulatory elements. How should incomplete, tissue-specific, and species-biased annotations be combined without letting well-annotated organisms dominate?
- **Evolutionary constraint:** phyloP, phastCons, GERP, alignment depth, or a learned evolutionary-rate score. Which phylogenetic timescale and window statistic best match each downstream task? How much low-conservation sequence must remain to cover recent or rapidly evolving function?
- **Cheap local alignment:** can `mmseqs2` hits be converted into robust local statistics—hit depth, taxonomic breadth, identity/coverage profiles, or deviations from a neutral model—that select functional windows without a precomputed WGA? How should paralogs, synteny loss, fragmented hits, and repeats be handled?
- **Learned single-sequence selection:** once a strong model exists, can a classifier or probe trained on annotations and/or evolutionary labels identify functional windows in a new genome from sequence alone? What orthogonal evaluation prevents the selector from merely reproducing conservation or annotation bias?
- Should these sources be used as a union, an intersection, separate sampling strata, or continuous weights?

### Filter, sample, or weight?

- At equal training FLOPs, how do hard filtering, score-proportional sampling, per-base loss weighting, and a curriculum from enriched to whole-genome data compare?
- Can the whole genome remain in the corpus while constrained/annotated bases receive most of the gradient? Does this retain rare functional classes and useful neutral context without letting background dominate?
- Should a window be selected because it contains some functional sequence, while the loss is applied only or preferentially to the functional bases? How much flanking sequence is beneficial context rather than wasted loss?
- Does causal next-token prediction, masked modeling, or another objective change the optimal sampling/weighting policy?

### How do scale and data quantity change the answer?

- Does the benefit of enrichment shrink with parameter count, token budget, or context length?
- When a filtered corpus is smaller, are gains caused by better loci or simply by seeing the same loci for more epochs? Both compute-matched and exposure/epoch-matched comparisons are needed.
- Is there a curriculum in which constrained sequence is best early, but adding progressively broader genome sequence later improves generalization or prevents overfitting?
- What is the minimum useful amount of low-conservation/background data at each scale?

### How should repetitive elements be handled?

- Compare exclusion, deduplication, soft masking, family/age-aware sampling, and per-base loss downweighting rather than treating “repeat filtering” as one intervention.
- Which repeat classes mostly create redundant or ambiguous alignment signal, and which carry regulatory, structural, or lineage-specific information?
- Do repeats help species/phylogeny tasks while hurting functional VEP, and does that tradeoff change with scale?
- Are repeat effects actually about biology, or about duplicated windows, assembly/masking heterogeneity, and train/test similarity leakage?

### Candidate experiment ladder

1. **Small controlled baseline.** At one small MarinDNA scale and fixed context, compare:
   - uniform whole-genome sampling with uniform loss;
   - the current conservation-filtered footprint;
   - whole-genome sampling with conservation/annotation-aware loss weights; and
   - a matched mixture of annotated functional classes plus background.

   Hold architecture, optimizer, number of training tokens, splits, and evaluation fixed. Report unique loci, effective epochs, repeat content, region composition, and gradient/loss mass by stratum.

2. **Disentangle selection from repetition.** Add a data-size-matched whole-genome subset and an epoch/exposure-matched rerun of the enriched arm. This separates “better sequence” from “more passes over less sequence.”

3. **Task-stratified readout.** Evaluate Mendelian and complex-trait VEP by consequence and conservation stratum, region-matched likelihood gaps, frozen-embedding probes, and at least one outcome expected to benefit from neutral/background sequence.

4. **Scale and objective interaction.** Repeat the most informative contrast at a larger model/token budget and, if justified, under a second objective or weighting scheme. [#87](https://github.com/Open-Athena/marin-dna/issues/87) is the natural home for the repeat/loss-weighting × FLOPs slice.

5. **Acquisition-method comparison.** Once the target training distribution is clearer, compare WGA/conservation selection against direct annotation, `mmseqs2`-derived local evolutionary statistics, and a learned single-sequence selector on the same genomes and matched window budget.

6. **Mixture frontier.** Sweep the fraction of constrained/annotated, weakly conserved, neutral/background, and repetitive sequence rather than comparing only the endpoints. The useful output is a task-by-scale Pareto frontier and a reproducible default mixture, not a universal declaration that one class of sequence “matters.”

## History

- 2026-08-14 — Migrated from the predecessor research-question issue [#395](https://github.com/Open-Athena/marin-dna/issues/395). The issue remains the historical source for its original body and comments.
