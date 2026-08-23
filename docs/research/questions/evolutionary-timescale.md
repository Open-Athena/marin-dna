# How should evolutionary timescale shape training?

> [!NOTE]
> **TL;DR:** Useful evolutionary breadth depends on genomic region: promoter and CDS specialists prefer different timescales, and reducing mammalian species diversity hurts some regions more than others; confidence is moderate that there is no universal best clade depth, with a matched factorial test of breadth, quantity, and downstream task still needed.

## Question

For a target organism or clade, how should we choose the phylogenetic breadth and sampling density of genomic language model training data to balance raw sequence diversity against evolutionary relevance?
Does the optimal timescale depend on genomic region and downstream task—for example, with deeply conserved coding biology benefiting from broader animal data while rapidly evolving regulatory sequence benefits from mammal- or primate-focused data?
Can broad pretraining followed by adaptation to the target clade capture both advantages, and when is that curriculum better than training on a fixed mixture from scratch?

## Current answer

There is no universal best evolutionary timescale in the current evidence.
Promoter, CDS, UTR, ncRNA, enhancer, and different evaluation endpoints respond differently to phylogenetic breadth.
Broader data can improve diversity and expose conserved rules; closer species spend more tokens on lineage-relevant grammar; and each genomic feature evolves on a different timescale.

Promoter and 3′ UTR comparisons favor mammals for learning speed, while broad animal CDS can become stronger later for missense VEP.
Reducing mammalian species density at matched compute is neutral for several regions but harmful for ncRNA and 5′ UTR/TSS.
Animal-versus-vertebrate CDS results also change with evaluation and data-construction method.
Nominal clade breadth can differ from effective breadth when a projection method recovers few distant species.

Confidence is moderate that timescale should be chosen by region and objective and low on the optimal schedule.
Most experiments change unique data, epochs, species density, or sequence-selection method alongside clade depth.
A matched factorial comparison is still needed.

The leading strategy is broad pretraining followed by target-clade adaptation or an explicit mixture, compared against broad-only and target-only training at matched total compute.
This would separate early sample efficiency from final endpoint quality and test whether one static species mixture is unnecessarily restrictive.

<details>
<summary>Related work</summary>

- [Ye, Benegas et al., Predicting functional constraints across evolutionary timescales with phylogeny-informed genomic language models](https://www.biorxiv.org/content/10.1101/2025.09.21.677619v1) compares primate-, mammal-, and vertebrate-alignment models across coding, regulatory, Mendelian, and complex-trait endpoints.
  Different endpoints prefer different breadths, while alignment-conditioned architecture and training data remain confounded for comparison with MarinDNA.

</details>

<details>
<summary>Related experiments</summary>

- [#6](https://github.com/Open-Athena/marin-dna/issues/6) established the initial evolutionary-timescale experiment series.
  It framed human, primate, mammal, vertebrate, and animal corpora as an empirical axis but did not by itself resolve region-specific optima.
- [#55](https://github.com/Open-Athena/marin-dna/issues/55) compared promoter training across those scopes.
  Mammalian data reached strong human Mendelian VEP sooner, while smaller human/primate corpora overfit; scope, unique-data quantity, and epochs remained confounded.
- [#58](https://github.com/Open-Athena/marin-dna/issues/58) compared CDS timescales.
  Shallower arms led early on missense VEP, while the animal arm later surpassed them, showing that learning speed and final quality can prefer different breadths.
- [#59](https://github.com/Open-Athena/marin-dna/issues/59) extended the comparison to downstream regions.
  Mammalian training learned 3′ UTR VEP faster, while vertebrate and animal arms later approached similar performance.
- [#142](https://github.com/Open-Athena/marin-dna/issues/142) compared primate-specific and mammal-wide conservation filters for projected enhancers.
  The mammal-wide filter was modestly better on Mendelian distal VEP, although source selection and projection coverage changed with the filter.
- [#255](https://github.com/Open-Athena/marin-dna/issues/255) compared 108-family and 19-order mammalian cohorts at matched compute.
  Coding, 3′ UTR, and enhancer VEP were similar; ncRNA and 5′ UTR/TSS worsened with fewer species, with one seed and different epoch counts.
- [#353](https://github.com/Open-Athena/marin-dna/issues/353) compared animal- and vertebrate-scale CDS built by native annotation and human-anchored projection.
  The preferred scope depended on evaluation and construction method, and low invertebrate recovery showed that nominal animal breadth did not imply broad nucleotide coverage.

</details>

<details>
<summary>Possible directions</summary>

- Separate phylogenetic breadth, species density, unique bases, target distance, and effective projected coverage rather than treating them as one timescale variable.
- Compare primate, mammal, vertebrate, and animal scopes at matched model, optimizer, tokens, construction method, and checkpoints, reporting fixed-token and matched-exposure views.
- At matched total compute, compare broad-only, target-clade-only, broad-to-narrow adaptation, and an interleaved mixture.
- Measure region-by-timescale and endpoint-by-timescale interactions before selecting a whole-genome default or curriculum.

</details>
