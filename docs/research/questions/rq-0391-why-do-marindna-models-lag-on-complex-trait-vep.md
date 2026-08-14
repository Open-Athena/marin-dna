# Why do MarinDNA models lag on complex-trait VEP?

## Metadata

| Field | Value |
|---|---|
| Question ID | `RQ-0391` |
| Status | `active` |
| Overall confidence | `unknown` |
| Evidence considered through | `2026-08-13` |
| Predecessor issues | [#391](https://github.com/Open-Athena/marin-dna/issues/391) |

## Question and scope

Why do MarinDNA single-sequence models lag GPN-Star on complex-trait variant-effect prediction (VEP), even when region-specialized models can approach GPN-Star on the corresponding Mendelian subsets? Is the main bottleneck (1) training-data selection that omits weakly conserved regions, (2) an evolutionary timescale that is too deep for recent, small-effect human variation, or (3) the absence of alignment-conditioned retrieval at inference? The goal is to identify which factor causally explains the gap and which intervention is most likely to close it.

## Current answer

MarinDNA's complex-trait VEP gap is real but has not been causally attributed. Training-footprint undercoverage has direct supporting evidence, while shallower evolutionary timescales and retrieval remain plausible; confidence is moderate on undercoverage as one contributor and low on which intervention will close the gap.

MarinDNA’s complex-trait VEP deficit is clearest in matched regulatory comparisons. The enhancer specialist is within 0.057 AUPRC of the strongest GPN-Star arm on Mendelian distal variants but 0.084 behind on complex-trait distal variants. A whole-genome 1B model also trails GPN-Star-M by 0.039 global and 0.097 macro AUPRC on complex traits.

Training-footprint undercoverage is the strongest supported explanation. At the production conservation cutoff, kept windows cover 83.9% of Mendelian positives and 42.6% of complex-trait positives; only about 22% of distal complex-trait positives overlap a kept window. This is direct evidence for distribution mismatch, but it is observational. No fixed-compute experiment has shown that adding weakly conserved sequence closes the gap.

Scoring contributes but does not appear sufficient. Richer embedding or downstream-effect scores and ensembling improve complex-trait readouts, while tested supervised probes and LoRA variants did not break the frozen-representation ceiling. Shallower evolutionary timescales and alignment-conditioned retrieval remain plausible, but current comparisons confound them with data, architecture, calibration, and objective.

Confidence is moderate that conservation-focused coverage is one contributor and low on the intervention that will close the gap. The decisive next evidence is a fixed-compute footprint ablation, followed by matched timescale and retrieval controls if undercoverage does not explain enough of the deficit.

## Confidence and limitations

MarinDNA's complex-trait VEP gap is real but has not been causally attributed. Training-footprint undercoverage has direct supporting evidence, while shallower evolutionary timescales and retrieval remain plausible; confidence is moderate on undercoverage as one contributor and low on which intervention will close the gap.

Scoring contributes but does not appear sufficient. Richer embedding or downstream-effect scores and ensembling improve complex-trait readouts, while tested supervised probes and LoRA variants did not break the frozen-representation ceiling. Shallower evolutionary timescales and alignment-conditioned retrieval remain plausible, but current comparisons confound them with data, architecture, calibration, and objective.

Confidence is moderate that conservation-focused coverage is one contributor and low on the intervention that will close the gap. The decisive next evidence is a fixed-compute footprint ablation, followed by matched timescale and retrieval controls if undercoverage does not explain enough of the deficit.

## Operational consequence

MarinDNA’s complex-trait VEP deficit is clearest in matched regulatory comparisons. The enhancer specialist is within 0.057 AUPRC of the strongest GPN-Star arm on Mendelian distal variants but 0.084 behind on complex-trait distal variants. A whole-genome 1B model also trails GPN-Star-M by 0.039 global and 0.097 macro AUPRC on complex traits.

## Supporting evidence

- The [Mendelian leaderboard](https://open-athena.github.io/marin-dna/leaderboards/mendelian), [complex-trait leaderboard](https://open-athena.github.io/marin-dna/leaderboards/complex), [#161](https://github.com/Open-Athena/marin-dna/issues/161), [#162](https://github.com/Open-Athena/marin-dna/issues/162), and [#145](https://github.com/Open-Athena/marin-dna/issues/145) define the current evaluation and GPN-Star baseline contract. They expose the Mendelian-versus-complex gap under consistent dashboard metrics. They do not identify whether data coverage, timescale, architecture, or scoring causes it.
- [Ye, Benegas et al., Predicting functional constraints across evolutionary timescales with phylogeny-informed genomic language models](https://www.biorxiv.org/content/10.1101/2025.09.21.677619v1) compares vertebrate-, mammal-, and primate-alignment models. Deeper evolution favors coding and rarer large-effect variants, while shallower models can favor non-coding or broader complex-trait endpoints; mammal and primate models lead different complex-trait readouts. This motivates a mammal-versus-primate test rather than assuming the closest clade wins. The paper does not isolate timescale from alignment-conditioned architecture for MarinDNA.
- [#394](https://github.com/Open-Athena/marin-dna/issues/394) maintains the broader evolutionary-timescale synthesis. Its implication here is that region and endpoint can prefer different breadths; it does not yet contain a matched complex-trait timescale result.
- [#395](https://github.com/Open-Athena/marin-dna/issues/395) maintains the training-footprint question. It connects conservation filtering, weakly conserved regulatory sequence, sampling, and loss weighting, but no completed experiment yet tests the causal coverage hypothesis.
- [#397](https://github.com/Open-Athena/marin-dna/issues/397) maintains the retrieval question. Alignment-conditioned context is a plausible explanation for GPN-Star’s advantage, but the current RAG evidence lacks a matched no-retrieval control.

## Contradictory evidence

The predecessor issue did not maintain a separate contradictory-evidence section. Its caveats and negative results are preserved in Current answer and Supporting evidence.

## Related experiments

- [#21](https://github.com/Open-Athena/marin-dna/issues/21) reported an early promoter model that generally approached Evo 2 but retained a particularly clear complex-trait promoter deficit. It established that the gap predates the current dashboard.
- [#55](https://github.com/Open-Athena/marin-dna/issues/55) compared promoter timescales on Mendelian VEP. Mammalian data learned faster than smaller human/primate corpora, but it did not evaluate a controlled complex-trait timescale effect.
- [#136](https://github.com/Open-Athena/marin-dna/issues/136) showed that human-anchored enhancer projection can produce strong home-domain Mendelian distal VEP. The remaining complex-trait gap shows that Mendelian success does not imply transfer to common-variant regulation.
- [#142](https://github.com/Open-Athena/marin-dna/issues/142) compared mammal-wide and primate-specific source filters for enhancer projection. Mammal-wide filtering was modestly better on Mendelian distal VEP, providing counterevidence to a generic “shallower is better” claim while leaving complex traits untested.
- [#175](https://github.com/Open-Athena/marin-dna/issues/175) showed that embedding-distance and downstream-effect scores, plus AlphaGenome ensembling, improve complex-trait VEP over absolute likelihood ratios. The remaining gap means scoring is one lever rather than a complete explanation.
- [#180](https://github.com/Open-Athena/marin-dna/issues/180) tested frozen supervised heads and LoRA variants on the whole-genome model. Neither broke the complex-trait frozen-embedding ceiling, supporting a data or representation bottleneck under the tested setup.
- [#213](https://github.com/Open-Athena/marin-dna/issues/213) measured evaluation-positive coverage by the conservation-filtered training footprint. Coverage was 83.9% for Mendelian positives, 42.6% for complex-trait positives, and about 22% for distal complex positives, providing the strongest direct evidence for distribution mismatch.

## Open questions

1. **Does training-footprint coverage explain the gap?** Stratify current predictions by [#213](https://github.com/Open-Athena/marin-dna/issues/213)'s variant-centered conserved fraction, kept-window membership, and consequence group. Compare MarinDNA with GPN-Star within each stratum.

2. **Does less-conserved training sequence causally help?** At fixed architecture, tokens, genome mixture, and window size, sweep the conservation cutoff and add a weakly conserved/background arm. Evaluate complex global/distal AUPRC and the Mendelian tradeoff; explicitly check coordinate and sequence-similarity leakage.

3. **What is the best timescale at fixed data quantity?** Train matched primate, mammal, and deeper-vertebrate arms while controlling tokens and unique human loci. Consider a deliberately reweighted mammal+primate mixture.

4. **Is retrieval the missing capability?** Compare one backbone with and without human-anchored ortholog retrieval using the same loci, species, objective, and compute. As a cheaper first test, ask whether MarinDNA's error relative to GPN-Star grows as conservation weakens or alignment evidence becomes more informative.

5. **How much is evaluation/readout?** Use a fixed, leakage-free scoring protocol informed by [#175](https://github.com/Open-Athena/marin-dna/issues/175) and cluster-bootstrap uncertainty. Stratify by PIP, consequence, MAF, and conservation to estimate any ceiling from fine-mapping ambiguity.

6. **What would distinguish the hypotheses?**
   - Improvement from a fixed-compute less-conserved-data arm, concentrated on low-conservation variants, supports hypothesis 1.
   - Improvement from a controlled mammal/primate arm, without merely adding tokens or loci, supports hypothesis 2.
   - Improvement from retrieval, largest on weakly conserved variants, supports hypothesis 3.
   - If none helps, benchmark noise or missing tissue/cell-state information becomes the leading explanation.

## History

- 2026-08-14 — Migrated from the predecessor research-question issue [#391](https://github.com/Open-Athena/marin-dna/issues/391). The issue remains the historical source for its original body and comments.
