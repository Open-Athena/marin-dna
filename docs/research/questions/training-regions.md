# Which genomic regions to train on, and how to find them?

> [!NOTE]
> **TL;DR:** Targeted or conservation-selected corpora often improve functional prediction at current scales, while frozen-model loss or entropy is a practical conservation proxy; no universal selection rule, causal benefit from likelihood-derived weighting, or repeat-downweighting benefit has been established.

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
- Conservation is a proxy for purifying constraint rather than a complete definition of function.
  It misses lineage-specific and hard-to-align elements, while annotations miss unknown constrained sequence.
  A useful observable is the joint coverage of evaluation loci, annotated classes, conservation tiers, and repeats under each proposed footprint.

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

</details>

<details>
<summary>Possible directions</summary>

- Audit token-loss dynamics across checkpoints of m1.3, the de novo 1B model trained on a uniform five-region mixture.
  On fixed nonrepeat validation tokens from CDS, upstream, downstream, ncRNA, and enhancer, track loss, entropy, conservation AUPRC where labels exist, selected-set overlap, and future loss reduction by current-loss quantile.
  This tests when functional ranking emerges and whether an online selection mask remains stable through training.
- Test whether annotation-free pretraining can recover functional sequence from likelihood.
  Train 46M, 76M, and 128M models for matched token budgets on the complete nonrepeat sequence of about 20 animal genomes sampled broadly across orders, with no annotation-based locus filtering.
  Use annotations and conservation only for held-out evaluation, measuring one-orientation loss and entropy AUPRC globally and for CDS, upstream, downstream, ncRNA, and enhancer against matched nonrepeat background.
  Held-out genomes and composition controls are needed to distinguish functional ranking from memorization and local sequence predictability.

### What outcome are we optimizing?

- Is there one corpus that transfers well across VEP, functional-region representations, sequence generation, genome annotation, and phylogenetic inference, or should MarinDNA maintain task-specific mixtures?
- For VEP, should the primary decision metric be zero-shot AUPRC, frozen-embedding probes, functional/non-functional likelihood gaps, or a combination?
  Improvements should be reported separately for Mendelian, complex-trait, coding, regulatory, conserved, and weakly conserved variants.
- What evaluation would expose the cost of removing neutral sequence?
  Candidate outcomes include phylogeny/species inference, mutation-spectrum prediction, regional composition, and held-out whole-genome likelihood, but these should not silently displace functional VEP as the primary task.

### What should count as functional training sequence?

- **Direct annotation:** genes, UTRs, ncRNAs, cCREs, experimentally measured regulatory elements.
  How should incomplete, tissue-specific, and species-biased annotations be combined without letting well-annotated organisms dominate?
- **Evolutionary constraint:** phyloP, phastCons, GERP, alignment depth, or a learned evolutionary-rate score.
  Which phylogenetic timescale and window statistic best match each downstream task?
  How much low-conservation sequence must remain to cover recent or rapidly evolving function?
- **Cheap local alignment:** can `mmseqs2` hits be converted into robust local statistics—hit depth, taxonomic breadth, identity/coverage profiles, or deviations from a neutral model—that select functional windows without a precomputed WGA?
  How should paralogs, synteny loss, fragmented hits, and repeats be handled?
- **Learned single-sequence selection:** once a strong model exists, can a classifier or probe trained on annotations and/or evolutionary labels identify functional windows in a new genome from sequence alone?
  What orthogonal evaluation prevents the selector from merely reproducing conservation or annotation bias?
- **Frozen likelihood-derived selection:** can per-base loss or entropy from a small frozen model identify useful training tokens without annotations?
  When a target or high-quality reference corpus exists, can a Rho-1-style reference model trained on that distribution provide a useful reducible-loss score?
  Separately, can loss reduction between two same-corpus frozen model sizes identify tokens with high scale-dependent learnability?
  Treat absolute predictability, target-distribution reducible loss, and same-corpus improvement with model scale as distinct signals.
  Repeat status, GC content, local k-mer predictability, taxon, and homology density are necessary controls because each can create predictable sequence without implying functional constraint.
- Should these sources be used as a union, an intersection, separate sampling strata, or continuous weights?

### Filter, sample, or weight?

- At equal training FLOPs, how do hard filtering, score-proportional sampling, per-base loss weighting, and a curriculum from enriched to whole-genome data compare?
- For likelihood-derived selection experiments, keep repeat downweighting fixed across every arm and compute selection thresholds only among nonrepeat tokens.
  This comparison tests how gradient mass is allocated within nonrepeat sequence, leaving repeat handling outside its scope.
- Can the whole genome remain in the corpus while constrained/annotated bases receive most of the gradient?
  Does this retain rare functional classes and useful neutral context without letting background dominate?
- Should a window be selected because it contains some functional sequence, while the loss is applied only or preferentially to the functional bases?
  How much flanking sequence is beneficial context rather than wasted loss?
- Does causal next-token prediction, masked modeling, or another objective change the optimal sampling/weighting policy?

### How do scale and data quantity change the answer?

- Does the benefit of enrichment shrink with parameter count, token budget, or context length?
- Run a small fixed-compute causal experiment comparing uniform nonrepeat loss, frozen FWD loss or entropy as a soft weight, excess loss against a frozen target-distribution reference, a scale-differential weight from frozen checkpoints, and the student's lowest-current-loss mask as a diagnostic control.
  Add direct conservation weighting on human sequence as an oracle positive control.
  Keep repeat downweighting identical in every arm.
  Match training compute and record the offline scoring cost separately.
  Treat same-corpus improvement as a learnability heuristic rather than a direct Rho-1 analogue.
  Treat lowest-current-loss selection as a self-paced objective that may reinforce already-learned, small-gradient tokens; use a uniform warm-up and a nonzero background floor if it advances beyond the diagnostic arm.
  Teacher nucleotide probabilities would define a separate self-distillation objective and should not be conflated with loss-based token selection.
  Use deterministic FWD scoring first and add an averaged-score sensitivity only if downstream outcomes justify the second inference pass.
- Add compatible training-corpus exposure and homology-density covariates when they become available.
  Test whether either explains the apparent association between likelihood-derived scores and conservation.
- When a filtered corpus is smaller, are gains caused by better loci or simply by seeing the same loci for more epochs?
  Both compute-matched and exposure/epoch-matched comparisons are needed.
- Is there a curriculum in which constrained sequence is best early, but adding progressively broader genome sequence later improves generalization or prevents overfitting?
- What is the minimum useful amount of low-conservation/background data at each scale?

### How should repetitive elements be handled?

- At several matched FLOP budgets, compare uniform loss with the current 100-fold repeat downweighting while holding footprint and sampling fixed.
  This should measure whether repeat downweighting helps and how its effect changes with scale.
- If uniform loss preserves language-modeling and downstream performance, remove repeat-specific loss weights.
- Compare exclusion, deduplication, soft masking, family/age-aware sampling, and per-base loss downweighting rather than treating “repeat filtering” as one intervention.
- Which repeat classes mostly create redundant or ambiguous alignment signal, and which carry regulatory, structural, or lineage-specific information?
- Do repeats help species/phylogeny tasks while hurting functional VEP, and does that tradeoff change with scale?
- Are repeat effects actually about biology, or about duplicated windows, assembly/masking heterogeneity, and train/test similarity leakage?

### Candidate experiment ladder

1. **Small controlled baseline.**
   At one small MarinDNA scale and fixed context, compare:
   - uniform whole-genome sampling with uniform loss;
   - the current conservation-filtered footprint;
   - whole-genome sampling with conservation/annotation-aware loss weights; and
   - a matched mixture of annotated functional classes plus background.

   Hold architecture, optimizer, number of training tokens, splits, and evaluation fixed.
   Report unique loci, effective epochs, repeat content, region composition, and gradient/loss mass by stratum.

2. **Disentangle selection from repetition.**
   Add a data-size-matched whole-genome subset and an epoch/exposure-matched rerun of the enriched arm.
   This separates “better sequence” from “more passes over less sequence.”

3. **Task-stratified readout.**
   Evaluate Mendelian and complex-trait VEP by consequence and conservation stratum, region-matched likelihood gaps, frozen-embedding probes, and at least one outcome expected to benefit from neutral/background sequence.

4. **Scale and repeat-weighting interaction.**
   Compare uniform loss with the current repeat downweighting at several model/token budgets while holding footprint, sampling, and objective fixed.
   If uniform loss preserves likelihood and downstream performance, remove repeat-specific weighting.
   [#87](https://github.com/Open-Athena/marin-dna/issues/87) first scoped this ablation.

5. **Acquisition-method comparison.**
   Once the target training distribution is clearer, compare WGA/conservation selection against direct annotation, `mmseqs2`-derived local evolutionary statistics, and a learned single-sequence selector on the same genomes and matched window budget.

6. **Mixture frontier.**
   Sweep the fraction of constrained/annotated, weakly conserved, neutral/background, and repetitive sequence rather than comparing only the endpoints.
   The useful output is a task-by-scale Pareto frontier and a reproducible default mixture, not a universal declaration that one class of sequence “matters.”

</details>
