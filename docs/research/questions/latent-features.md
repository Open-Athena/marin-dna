# What latent biological features do gLMs learn?

> [!NOTE]
> **TL;DR:** Sparse-autoencoder analyses identify reproducible splice, stop-codon, accessibility, and coding-context responses, while several broader semantic claims failed to replicate; the program is paused indefinitely, with moderate confidence in local causal features and low confidence in a complete biological inventory.

## Question

What biologically meaningful latent features do genomic language models learn, how are those features organized across layers, orientations, training stages, and model families, and what genomic computations do they support?

MarinDNA m5.1 is the first experimental system for this question, not its scope.
We will use sparse autoencoders as the best reasonable feature-discovery instrument available today.
The goal is not to establish that SAEs beat every raw-activation, probe, or sequence method.
Comparisons and controls should be included only when they rule out a concrete artifact or are necessary for a stronger robustness or causal claim; the priority is rapid, FDR-controlled association discovery, biological interpretation, and targeted follow-up.

**Primary discovery principle:** prioritize how features change between matched reference and alternate sequences at real or designed variants.
Paired Δ features are less studied than single-sequence annotation features, align directly with variant effect prediction, and provide a local counterfactual that removes much background-locus variation.
Absolute reference/alternate activations, human-reference inventories, and sequence logos remain essential supporting views for interpreting a response, but they are not the main endpoint.

## Current answer

The program is paused indefinitely as of 2026-08-10, and all scoped experiment issues are closed.
The strongest result is that m5.1 contains sparse features with reproducible local biological responses, but the inventory is selective and strongly dependent on layer, dictionary, orientation, and response definition.

Splice-acceptor, splice-donor, and stop-creation responses survive targeted perturbation, untouched-context replication, and transfer across independently trained dictionaries.
A broad synonymous/codon-degeneracy interpretation did not replicate.
A better-trained dictionary instead exposed a narrow leucine-codon-family response.
These results support local causal sequence grammars with moderate confidence and argue against assigning semantics from decoder similarity or a single association.

Layer choice is task-specific.
Middle layers performed best on the frozen broad-consequence endpoint, while final-layer features carried the strongest Mendelian-label, complex-trait-label, and accessibility-direction associations.
Reconstruction quality improved with more SAE training but did not reliably predict biological yield.
Feature counts also overstate biological multiplicity because correlated response families can contain many feature IDs.

The paired-variant protocol is part of the current answer: retain reference, alternate, signed change, and magnitude; report forward and reverse-complement orientations separately; predeclare any aggregate; test all eligible features with complete-family correction; and distinguish association, interpretation, and intervention.
Unsigned AlphaGenome associations currently identify broad GC/CpG-conditioned accessibility or promoter-effect magnitude rather than tissue-specific semantics.
Accessibility-direction discoveries are highly redundant and overlap broad consequence response, so accessibility causality remains untested.

Reference-sequence analyses show substantial repeat and annotation-state capacity.
Final-layer repeat information remains after strict composition matching, while promoter-like sequence is easier to identify than enhancer-like sequence.
Composition/repeat qualification and exact boundary localization are still needed before claiming genomic segmentation.
The next useful biological tiers are UTR, promoter/TSS-proximal, and annotated ncRNA variants, followed later by enhancer/cCRE variants.

<details>
<summary>Related work</summary>

- [Korsakova and Kelley, Learning monosemantic features in multitask DNA regulatory sequence models via sparse autoencoder decomposition](https://openreview.net/forum?id=AlLZnZX01x) trained TopK SAEs on early Borzoi layers and annotated features with repeats, motifs, and regulatory elements.
  Motifs specialized by depth, orientation, and flanking context, and larger dictionaries split concepts.
  This motivates layer panels, normalized activations, FWD/RC inspection, and context-aware interpretation.
  The remaining gap is transfer from a supervised regulatory model to paired variant effects in a causal gLM.
- [Evo 2 feature work](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1) trained a large BatchTopK SAE on a genomic foundation model and recovered biological sequence features at scale.
  It supports SAE feasibility but does not establish that individual features are monosemantic or causally relevant to variant effects.
- [Language Modeling Materializes a World Model of Protein Biology](https://www.biorxiv.org/content/10.1101/2024.12.18.629098v1) reports concept splitting, decoder neighborhoods, and feature combinations in protein models.
  It motivates analyzing feature families and co-activation systems alongside individual IDs.
  The open question is whether the same organizational principles hold for genomic sequence and across reverse-complement views.

</details>

<details>
<summary>Related experiments</summary>

- [#418](https://github.com/Open-Athena/marin-dna/issues/418) trained and validated the first production-shaped m5.1 block-10 BatchTopK SAE.
  It established workable reconstruction and sparsity and produced initial splice and nucleotide features, but did not by itself validate biological semantics.
- [#420](https://github.com/Open-Athena/marin-dna/issues/420) tested the first fixed feature panel on Mendelian variants.
  Pathogenic label was null overall and within subsets, while seven-way consequence/region prediction reached macro-AUPRC 0.2409 versus 0.1429 chance, showing encoded region information without pathogenicity separation.
- [#421](https://github.com/Open-Athena/marin-dna/issues/421) scanned unsigned SAE changes against AlphaGenome tracks.
  Leading block-10/19 signals were broad GC/CpG-conditioned accessibility or promoter-effect magnitude features; tissue-specific interpretations failed robustness checks, and one tail-sensitive lead instead connected to Mendelian label.
- [#422](https://github.com/Open-Athena/marin-dna/issues/422) ran the broad 35-consequence inventory across blocks 1, 10, and 19.
  Splice, stop, missense, miRNA, promoter-like, and 5′ UTR signals were selective rather than universal, and FWD/RC discoveries agreed in direction when shared despite incomplete ID overlap.
- [#424](https://github.com/Open-Athena/marin-dna/issues/424) froze the paired-variant and strand protocol.
  It retained ref, alt, signed and unsigned responses, required separate FWD/RC reporting, and selected signed mean as the then-best global reducer while documenting that same-ID strand invariance is uncommon.
- [#426](https://github.com/Open-Athena/marin-dna/issues/426) compared SAE layer and 5M versus 25M activation budgets.
  Block 10/5M led the frozen broad-consequence endpoint, later layers led some coding tasks, and better reconstruction at 25M did not consistently improve biological yield.
- [#428](https://github.com/Open-Athena/marin-dna/issues/428) replicated fixed coding-context features on a larger phase- and substitution-matched panel.
  Effects persisted but attenuated, and a local 31-bp sequence model was stronger, narrowing the interpretation to orientation-complementary local coding context.
- [#429](https://github.com/Open-Athena/marin-dna/issues/429) prospectively replicated causal acceptor, donor, stop-creation, and synonymous/codon-degeneracy perturbation effects on untouched contexts.
  This established the first local causal feature set for the program.
- [#431](https://github.com/Open-Athena/marin-dna/issues/431) transferred those semantic queries across independently trained dictionaries.
  Acceptor, donor, and stop creation replicated; synonymous/codon degeneracy did not, despite stable decoder-neighborhood geometry.
- [#432](https://github.com/Open-Athena/marin-dna/issues/432) repeated the causal protocol in the healthier 25M dictionary.
  Splice and stop semantics persisted; the broad synonymous interpretation remained null, while exhaustive search found a narrow leucine-codon-family response.
- [#434](https://github.com/Open-Athena/marin-dna/issues/434) scanned accessibility-QTL causality and direction.
  A final-layer signed-direction family appeared in the 559-positive dsQTL pilot, but it was highly redundant, overlapped broad consequence response, and did not test causality because negatives were absent.
- [#435](https://github.com/Open-Athena/marin-dna/issues/435) mapped repeat capacity across blocks 1, 10, and 19.
  Repeat hierarchy became more specific with depth; final-layer repeat information survived composition matching, and selected repeat grammars showed causal motif-loss responses, while many shallower signals were composition-linked.
- [#436](https://github.com/Open-Athena/marin-dna/issues/436) mapped Mendelian pathogenicity across layers.
  Individual-feature signal was weak in blocks 1/10 and stronger but distributed in block 19; focal associations do not yet bridge the full performance of official likelihood or whole-window embedding readouts.
- [#438](https://github.com/Open-Athena/marin-dna/issues/438) mapped complex-trait label signal across layers.
  Block 19 dominated the fixed panel, magnitude was more useful than signed change, and recurrent feature 1662 transferred to held-out data but currently supports local coding-impact/codon-context sensitivity rather than a broad causal mechanism.
- [#440](https://github.com/Open-Athena/marin-dna/issues/440) tested reference annotation states.
  The first pass found layer-specific gene-state inventories and strong promoter-like signals but little enhancer-specific signal; composition/repeat controls and transcript-boundary case studies remain unfinished.

</details>

<details>
<summary>Possible directions</summary>

- Extend the paired-variant inventory from CDS and splice variants to UTR, promoter/TSS-proximal, annotated ncRNA, and later enhancer/cCRE variants.
- Keep reference, alternate, signed change, magnitude, and FWD/RC views; predeclare any aggregate and correct over the complete eligible feature family.
- Test whether feature-family or set-level correspondence is more stable across orientations, SAE seeds, dictionaries, and checkpoints than same-ID matching.
- Compare SAE recipes by corrected biological yield, feature-family stability, and reconstructed-model degradation rather than reconstruction alone.
- Qualify reference-sequence annotation and boundary features against composition, repeat, and local-sequence controls.
- Test causal interventions only for preregistered features that survive independent contexts and dictionaries.

</details>
