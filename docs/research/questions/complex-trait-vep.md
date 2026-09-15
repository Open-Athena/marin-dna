# Why do MarinDNA models lag on complex-trait VEP?

> [!NOTE]
> **TL;DR:** MarinDNA's complex-trait VEP gap is real but has not been causally attributed; training-footprint undercoverage is the best-supported hypothesis, while shallower evolutionary timescales and retrieval remain plausible, and confidence is low on which intervention will close the gap.

## Question

Why do MarinDNA single-sequence models lag GPN-Star on complex-trait variant-effect prediction (VEP), even when region-specialized models can approach GPN-Star on the corresponding Mendelian subsets?
Is the main bottleneck (1) training-data selection that omits weakly conserved regions, (2) an evolutionary timescale that is too deep for recent, small-effect human variation, or (3) the absence of alignment-conditioned retrieval at inference?
The goal is to identify which factor causally explains the gap and which intervention is most likely to close it.

## Current answer

MarinDNA’s complex-trait VEP deficit is clearest in matched regulatory comparisons.
The enhancer specialist is within 0.057 AUPRC of the strongest GPN-Star arm on Mendelian distal variants but 0.084 behind on complex-trait distal variants.
A whole-genome 1B model also trails GPN-Star-M by 0.039 global and 0.097 macro AUPRC on complex traits.

Training-footprint undercoverage is the strongest supported explanation.
At the production conservation cutoff, kept windows cover 83.9% of Mendelian positives and 42.6% of complex-trait positives; only about 22% of distal complex-trait positives overlap a kept window.
This is direct evidence for distribution mismatch, but it is observational.
No fixed-compute experiment has shown that adding weakly conserved sequence closes the gap.

Scoring contributes but does not appear sufficient.
Richer embedding or downstream-effect scores and ensembling improve complex-trait readouts, while tested supervised probes and LoRA variants did not break the frozen-representation ceiling.
Shallower evolutionary timescales and alignment-conditioned retrieval remain plausible, but current comparisons confound them with data, architecture, calibration, and objective.

Confidence is moderate that conservation-focused coverage is one contributor and low on the intervention that will close the gap.
The decisive next evidence is a fixed-compute footprint ablation, followed by matched timescale and retrieval controls if undercoverage does not explain enough of the deficit.

<details>
<summary>Related work</summary>

- [Ye, Benegas et al., Predicting functional constraints across evolutionary timescales with phylogeny-informed genomic language models](https://www.biorxiv.org/content/10.1101/2025.09.21.677619v1) compares vertebrate-, mammal-, and primate-alignment models.
  Deeper evolution favors coding and rarer large-effect variants, while mammal and primate models lead different complex-trait endpoints.
  The study motivates an endpoint-specific timescale test but does not isolate timescale from alignment-conditioned architecture for MarinDNA.

</details>

<details>
<summary>Related experiments</summary>

- [#21](https://github.com/Open-Athena/marin-dna/issues/21) reported an early promoter model that generally approached Evo 2 but retained a particularly clear complex-trait promoter deficit.
  It established that the gap predates the current dashboard.
- [#55](https://github.com/Open-Athena/marin-dna/issues/55) compared promoter timescales on Mendelian VEP.
  Mammalian data learned faster than smaller human/primate corpora, but it did not evaluate a controlled complex-trait timescale effect.
- [#136](https://github.com/Open-Athena/marin-dna/issues/136) showed that human-anchored enhancer projection can produce strong home-domain Mendelian distal VEP.
  The remaining complex-trait gap shows that Mendelian success does not imply transfer to common-variant regulation.
- [#142](https://github.com/Open-Athena/marin-dna/issues/142) compared mammal-wide and primate-specific source filters for enhancer projection.
  Mammal-wide filtering was modestly better on Mendelian distal VEP, providing counterevidence to a generic “shallower is better” claim while leaving complex traits untested.
- [#175](https://github.com/Open-Athena/marin-dna/issues/175) showed that embedding-distance and downstream-effect scores, plus AlphaGenome ensembling, improve complex-trait VEP over absolute likelihood ratios.
  The remaining gap means scoring is one lever rather than a complete explanation.
- [#180](https://github.com/Open-Athena/marin-dna/issues/180) tested frozen supervised heads and LoRA variants on the whole-genome model.
  Neither broke the complex-trait frozen-embedding ceiling, supporting a data or representation bottleneck under the tested setup.
- [#213](https://github.com/Open-Athena/marin-dna/issues/213) measured evaluation-positive coverage by the conservation-filtered training footprint.
  Coverage was 83.9% for Mendelian positives, 42.6% for complex-trait positives, and about 22% for distal complex positives, providing the strongest direct evidence for distribution mismatch.

</details>

<details>
<summary>Possible directions</summary>

- Stratify current MarinDNA and GPN-Star predictions by kept-window membership, conserved fraction, consequence, and fine-mapping confidence.
- At fixed architecture, tokens, genome mixture, and window size, compare the current footprint with weaker-conservation and background arms; measure complex-trait gains and Mendelian tradeoffs.
- If coverage does not explain the gap, compare primate, mammal, and deeper-vertebrate training at fixed unique human loci and token exposure.
- Estimate the incremental value of ortholog retrieval with matched reader, loci, species, objective, and compute, using a fixed leakage-free scoring protocol.

</details>
