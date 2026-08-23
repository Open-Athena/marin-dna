# Does conditioning on species/clade help?

> [!NOTE]
> **TL;DR:** Carbon-3B used species prefixes during human Mendelian variant scoring, but correct mammalian and far-wrong fungal prompts did not establish a macro-AUPRC difference from no tagging; whether tag-conditioned pretraining helps remains untested, so confidence is low.

## Question

Does explicitly conditioning a multi-species genomic language model on the source species or clade improve what it learns, relative to an otherwise identical sequence-only model that must infer taxonomic context from the DNA window?
The taxon is known for our reference-genome training and evaluation data; the question is whether exposing it improves conditional language modeling, zero-shot variant-effect prediction, or learned representations—not merely whether the model can classify species.
If conditioning helps, what taxonomic granularity and representation share information across related organisms while still generalizing to unseen species?
Can taxon-tag dropout preserve a useful unconditional mode?

## Current answer

MarinDNA has tested whether a metadata-trained checkpoint uses taxon prompts at inference.
[Experiment #486](../experiments/486-carbon-species-conditioning.md) held Carbon-3B, 8-kb human sequences, and development-set Mendelian variants fixed while comparing no tag, the correct mammalian tag, and a far-wrong fungal tag.
Correct conditioning changed macro AUPRC by -0.0030 [-0.0183, 0.0088] relative to no tagging, while fungal conditioning changed it by +0.0005 [-0.0137, 0.0124].
Neither prompt established an aggregate ranking difference.

Carbon-3B did use the prompt prefix.
Tagged-versus-no-tag score agreement varied by consequence subset and was lowest where no-tag LLR-derived scores had limited near-neutral dynamic range.
This establishes inference-time prompt sensitivity in one human setting without showing that the changes are useful.
It does not identify the causal value of metadata-conditioned pretraining, which requires otherwise-matched models trained with and without tags.

[Carbon](https://www.biorxiv.org/content/10.64898/2026.05.22.727119v1) supplies the most direct published evidence so far.
Its released 3B and 8B autoregressive models see species- and gene-type-tagged prompts on 50% of pretraining examples.
The reported tag ablation leaves overall sequence recovery essentially unchanged, while a correct species tag improves recovery for fungi, protozoa, and invertebrates and leaves high-resource groups unchanged.
Carbon evaluates ClinVar, BRCA2, and TraitGym VEP separately, but does not report how those results change with conditioning.
The released checkpoints support both conditional and unconditional prompts, which enabled the same-checkpoint intervention in experiment #486.

The combined evidence supports a conditional hypothesis with low confidence.
Taxon labels are most likely to help when the sequence window is too short to identify the relevant organismal regime and when the target feature varies across clades.
The benefit may shrink with longer context or larger models, and a label may expose compositional shortcuts instead of functional biology.

A leaf-species token is the simplest ablation, but a hierarchy could share information across related organisms and remain usable for unseen species.
Any positive result must survive shuffled-label controls, wrong-tag sensitivity, fixed data and compute, and downstream evaluation.
Validation-loss improvement alone would not show that conditioning learned useful biology.

<details>
<summary>Related work</summary>

- [Species-aware DNA language models](https://doi.org/10.1186/s13059-024-03221-x) prepends a learned species token to short fungal regulatory sequences and reports modest gains in motif reconstruction and representation tasks.
  Held-out species required a hand-chosen proxy token, so unseen-species behavior remains unresolved.
- [LOL-EVE](https://openreview.net/pdf?id=WxHbIY90IS) conditions a causal promoter model on clade, species, and proximal-gene embeddings with control-tag dropout.
  No matched ablation isolates the taxon tags from gene conditioning, data, and architecture.
- [Carbon](https://www.biorxiv.org/content/10.64898/2026.05.22.727119v1) supplies species and gene-type tags on half of pretraining examples.
  Correct species tags improve sequence recovery for several low-resource eukaryote groups without changing the overall result, but the paper does not report tag-conditioned VEP ablations.
- [Evo 2](https://doi.org/10.1038/s41586-026-10176-5) recovers species-shaped completion statistics from sequence-only long contexts, showing that explicit labels can become redundant.
  The result does not establish whether short-window models infer taxon efficiently.
- [SynCodonLM](https://academic.oup.com/nar/article/54/5/gkag166/8496864) finds similar average benchmark scores in small species-group ablations but lower robustness to synonymous perturbations without taxon context.
  Its evidence is restricted to coding sequence, codon tokens, and masked modeling.

</details>

<details>
<summary>Related experiments</summary>

- [Experiment #486](../experiments/486-carbon-species-conditioning.md) compared no tag, a correct mammalian tag, and a far-wrong fungal tag for one frozen Carbon-3B checkpoint on development-set human Mendelian variants.
  The tags changed per-variant scores but did not establish a macro-AUPRC difference; the intervention isolates inference-time prompt sensitivity rather than the effect of tagged pretraining.
- [#55](https://github.com/Open-Athena/marin-dna/issues/55) compared promoter training across human, primate, mammal, vertebrate, and animal corpora.
  Mammalian data learned human Mendelian VEP faster, but phylogenetic breadth, unique-data quantity, and epoch count changed together; the experiment motivates conditioning without testing it.
- [#58](https://github.com/Open-Athena/marin-dna/issues/58) repeated the timescale comparison for CDS and found a different trajectory: shallower arms led early, while the animal arm later became stronger for missense VEP.
  This suggests that useful taxonomic context depends on feature class and training stage.
- [#255](https://github.com/Open-Athena/marin-dna/issues/255) reduced the mammalian cohort from 108 family representatives to 19 order representatives at matched compute.
  Coding, 3′ UTR, and enhancer VEP were similar, while ncRNA and 5′ UTR/TSS VEP worsened, showing region-dependent value from species diversity.
- [#353](https://github.com/Open-Athena/marin-dna/issues/353) compared animal- and vertebrate-scale CDS corpora constructed by native annotation and human-anchored projection.
  The preferred breadth depended on evaluation and construction method, and nucleotide projection recovered little invertebrate sequence; nominal taxonomy and effective training context can differ.

</details>

<details>
<summary>Possible directions</summary>

- Compare otherwise-matched sequence-only and taxon-conditioned models on the same examples, order, optimizer, seed, and token budget, with shuffled-label and wrong-tag controls.
- Compare leaf-species tokens with hierarchical clade tokens or phylogenetic embeddings, including a defined policy for unseen species.
- Measure conditioning gains across context length and genomic region to test when sequence alone supplies enough taxonomic information.
- Use tag dropout to retain an unconditional mode and control fixed prompt positions so dropout does not change sequence alignment.
- Evaluate seen- and unseen-taxon language loss, functional-region gaps, VEP, and frozen probes after controlling for GC and repeat shortcuts.

</details>
