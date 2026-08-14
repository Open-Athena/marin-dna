# How should evolutionary timescale shape training?

## TL;DR

Existing MarinDNA experiments show that useful evolutionary breadth depends on genomic region: promoter and CDS specialists prefer different timescales, and reducing mammalian species diversity hurts some regions more than others. Confidence is moderate that there is no universal best clade depth; the main gap is a matched factorial test of breadth, quantity, and downstream task.

## Question

For a target organism or clade, how should we choose the phylogenetic breadth and sampling density of genomic language model training data to balance raw sequence diversity against evolutionary relevance? Does the optimal timescale depend on genomic region and downstream task—for example, with deeply conserved coding biology benefiting from broader animal data while rapidly evolving regulatory sequence benefits from mammal- or primate-focused data? Can broad pretraining followed by adaptation to the target clade capture both advantages, and when is that curriculum better than training on a fixed mixture from scratch?

## Current answer

There is no universal best evolutionary timescale in the current evidence. Promoter, CDS, UTR, ncRNA, enhancer, and different evaluation endpoints respond differently to phylogenetic breadth. Broader data can improve diversity and expose conserved rules; closer species spend more tokens on lineage-relevant grammar; and each genomic feature evolves on a different timescale.

Promoter and 3′ UTR comparisons favor mammals for learning speed, while broad animal CDS can become stronger later for missense VEP. Reducing mammalian species density at matched compute is neutral for several regions but harmful for ncRNA and 5′ UTR/TSS. Animal-versus-vertebrate CDS results also change with evaluation and data-construction method. Nominal clade breadth can differ from effective breadth when a projection method recovers few distant species.

Confidence is moderate that timescale should be chosen by region and objective and low on the optimal schedule. Most experiments change unique data, epochs, species density, or sequence-selection method alongside clade depth. A matched factorial comparison is still needed.

The leading strategy is broad pretraining followed by target-clade adaptation or an explicit mixture, compared against broad-only and target-only training at matched total compute. This would separate early sample efficiency from final endpoint quality and test whether one static species mixture is unnecessarily restrictive.

<details>
<summary>Related work</summary>

- [#287](https://github.com/Open-Athena/marin-dna/issues/287) asks whether explicit species or clade conditioning can expose taxonomic context that sequence-only training must infer. Conditioning could reduce the need to choose one static timescale, but no matched MarinDNA conditioning ablation exists.
- [#391](https://github.com/Open-Athena/marin-dna/issues/391) asks whether shallower evolutionary context helps complex-trait VEP. GPN-Star’s mammal and primate models lead different complex-trait endpoints, so the relevant comparison is endpoint-specific rather than a generic “closer is better” claim.
- The distinction between nominal and effective breadth is methodological: annotation, whole-genome alignment, and human-anchored nucleotide projection admit different distant species even under the same taxonomic label. Future comparisons should report realized per-species tokens and locus coverage, not only the requested clade.

</details>

<details>
<summary>Related experiments</summary>

- [#6](https://github.com/Open-Athena/marin-dna/issues/6) established the initial evolutionary-timescale experiment series. It framed human, primate, mammal, vertebrate, and animal corpora as an empirical axis but did not by itself resolve region-specific optima.
- [#55](https://github.com/Open-Athena/marin-dna/issues/55) compared promoter training across those scopes. Mammalian data reached strong human Mendelian VEP sooner, while smaller human/primate corpora overfit; scope, unique-data quantity, and epochs remained confounded.
- [#58](https://github.com/Open-Athena/marin-dna/issues/58) compared CDS timescales. Shallower arms led early on missense VEP, while the animal arm later surpassed them, showing that learning speed and final quality can prefer different breadths.
- [#59](https://github.com/Open-Athena/marin-dna/issues/59) extended the comparison to downstream regions. Mammalian training learned 3′ UTR VEP faster, while vertebrate and animal arms later approached similar performance.
- [#142](https://github.com/Open-Athena/marin-dna/issues/142) compared primate-specific and mammal-wide conservation filters for projected enhancers. The mammal-wide filter was modestly better on Mendelian distal VEP, although source selection and projection coverage changed with the filter.
- [#255](https://github.com/Open-Athena/marin-dna/issues/255) compared 108-family and 19-order mammalian cohorts at matched compute. Coding, 3′ UTR, and enhancer VEP were similar; ncRNA and 5′ UTR/TSS worsened with fewer species, with one seed and different epoch counts.
- [#353](https://github.com/Open-Athena/marin-dna/issues/353) compared animal- and vertebrate-scale CDS built by native annotation and human-anchored projection. The preferred scope depended on evaluation and construction method, and low invertebrate recovery showed that nominal animal breadth did not imply broad nucleotide coverage.

</details>

## Possible directions

1. **What exactly should “evolutionary timescale” mean experimentally?** Phylogenetic breadth (primates versus mammals versus vertebrates versus animals), species density within a clade, total unique bases, and distance from the target organism are separate axes and should not be collapsed into one variable.

2. **What is the clean fixed-compute comparison?** Train matched primate-, mammal-, vertebrate-, and animal-scope arms with the same model, optimizer, total tokens, sequence-selection method, and evaluation checkpoints. Report both fixed-token and matched-epoch or matched-unique-locus views so that breadth is not mistaken for repetition or dataset size.

3. **Does broad-to-narrow training dominate a static mixture?** At matched total tokens, compare broad-only, target-clade-only, broad pretraining followed by target-clade adaptation, and an interleaved/reweighted mixture. Sweep the fraction of compute spent in each stage and measure both target-task gains and catastrophic forgetting of broader capabilities.

4. **How does the answer vary by genomic region?** Run the comparison separately for CDS, promoters/TSS, 5′ and 3′ UTRs, ncRNA exons, enhancers/distal regulatory sequence, and background. A region-specific species mixture or curriculum may be better than a single whole-genome recipe.

5. **How does the answer vary by biological endpoint?** Separate language-model loss and functional-versus-background LL gaps from human Mendelian VEP, complex-trait VEP, eQTL/regulatory effects, saturation assays, and cross-species transfer. Rare coding variants and common regulatory variants may reflect different evolutionary timescales.

6. **Does the optimum change with model scale, context length, or token budget?** A small or short-context model may benefit more from narrowly relevant data, whereas a larger model may have enough capacity and compute to absorb broad diversity without sacrificing target-clade performance.

7. **What result would support each training strategy?**
   - Broad-only winning at matched compute would support diversity and deeper conservation as the dominant source of signal.
   - Target-clade-only winning would support evolutionary relevance and efficient use of the token budget.
   - Broad-to-narrow beating both endpoints and the static mixture would support a curriculum in which general biological features transfer before lineage-specific adaptation.
   - Strong region-by-timescale interactions would support region-specific mixtures or curricula rather than one universal phylogenetic scope.
