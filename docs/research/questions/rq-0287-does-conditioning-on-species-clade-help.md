# Does conditioning on species/clade help?

## Metadata

| Field | Value |
|---|---|
| Question ID | `RQ-0287` |
| Status | `active` |
| Overall confidence | `low` |
| Evidence considered through | `2026-08-14` |
| Predecessor issues | [#287](https://github.com/Open-Athena/marin-dna/issues/287) |

## Question and scope

Does explicitly conditioning a multi-species genomic language model on the source species or clade improve what it learns, relative to an otherwise identical sequence-only model that must infer taxonomic context from the DNA window? The taxon is known for our reference-genome training and evaluation data; the question is whether exposing it improves conditional language modeling, zero-shot variant-effect prediction, or learned representations—not merely whether the model can classify species. If conditioning helps, what taxonomic granularity and representation share information across related organisms while still generalizing to unseen species? Can taxon-tag dropout preserve a useful unconditional mode?

## Current answer

No matched MarinDNA ablation has tested taxon conditioning. Carbon directly ablates species and gene-type tags: overall sequence recovery is unchanged, while a correct species tag helps low-resource eukaryote groups; it does not report the conditioning effect on VEP. Other published gains remain small or task-specific. Confidence is low, and the cheapest next test is a frozen-Carbon tagged, untagged, and wrong-tag VEP comparison before matched retraining.

No MarinDNA experiment has compared identical examples with and without an explicit taxon input. The current model-facing path retains sequence but does not expose species identity, so every multispecies model must infer organismal context from the DNA window.

[Carbon](https://www.biorxiv.org/content/10.64898/2026.05.22.727119v1) supplies the most direct published evidence so far. Its released 3B and 8B autoregressive models see species- and gene-type-tagged prompts on 50% of pretraining examples. The reported tag ablation leaves overall sequence recovery essentially unchanged, while a correct species tag improves recovery for fungi, protozoa, and invertebrates and leaves high-resource groups unchanged. Carbon evaluates ClinVar, BRCA2, and TraitGym VEP separately, but does not report how those results change with conditioning. The released checkpoints support both conditional and unconditional prompts, enabling a same-checkpoint VEP intervention now; this measures inference-time use of metadata, not the causal effect of metadata-conditioned pretraining.

The evidence supports a conditional hypothesis with low confidence. Taxon labels are most likely to help when the sequence window is too short to identify the relevant organismal regime and when the target feature varies across clades. The benefit may shrink with longer context or larger models, and a label may expose compositional shortcuts instead of functional biology.

A leaf-species token is the simplest ablation, but a hierarchy could share information across related organisms and remain usable for unseen species. Any positive result must survive shuffled-label controls, wrong-tag sensitivity, fixed data and compute, and downstream evaluation. Validation-loss improvement alone would not show that conditioning learned useful biology.

## Confidence and limitations

No matched MarinDNA ablation has tested taxon conditioning. Carbon directly ablates species and gene-type tags: overall sequence recovery is unchanged, while a correct species tag helps low-resource eukaryote groups; it does not report the conditioning effect on VEP. Other published gains remain small or task-specific. Confidence is low, and the cheapest next test is a frozen-Carbon tagged, untagged, and wrong-tag VEP comparison before matched retraining.

The evidence supports a conditional hypothesis with low confidence. Taxon labels are most likely to help when the sequence window is too short to identify the relevant organismal regime and when the target feature varies across clades. The benefit may shrink with longer context or larger models, and a label may expose compositional shortcuts instead of functional biology.

## Operational consequence

Keep the production input sequence-only for now. Before matched retraining, compare frozen Carbon VEP with correct, absent, shuffled, and wrong taxon tags; add conditioning only if downstream gains survive those controls.

## Supporting evidence

- The [current MarinDNA preprocessor](https://github.com/Open-Athena/marin-dna/blob/cbb127b53b3d52421fae65cebedf2194c4ec84a3/src/marin_dna/levanter/batch_tokenizer.py#L63-L76) reads the sequence field and emits nucleotide IDs and loss weights without a taxon input. This establishes the sequence-only baseline. It does not determine whether a taxon input would improve representations or VEP.

| Work | Setup and finding | Implication and remaining gap |
|---|---|---|
| [Species-aware DNA language models](https://doi.org/10.1186/s13059-024-03221-x) | A learned species token was prepended to short fungal regulatory sequences from 806 species. Species-aware models modestly improved motif reconstruction and several representation tasks. | Explicit taxon context can help short regulatory models. Held-out species required a hand-chosen proxy token, so unseen-species behavior remains unresolved. |
| [LOL-EVE](https://openreview.net/pdf?id=WxHbIY90IS) | A causal promoter model conditions on clade, species, and proximal-gene embeddings and randomly drops control tags. It reports strong promoter-variant performance. | This supplies a practical hierarchy and dropout design. No matched ablation isolates the taxon tags from gene conditioning, data, and architecture. |
| [Carbon](https://www.biorxiv.org/content/10.64898/2026.05.22.727119v1) | Released 3B and 8B autoregressive models mix species and gene-type tags into 50% of pretraining prompts. Tags leave overall sequence recovery essentially unchanged; a correct species tag improves recovery for low-resource eukaryote groups. Its VEP suite includes ClinVar, BRCA2, and TraitGym Mendelian, but does not cross VEP with conditioning. | This is direct conditioning evidence and provides frozen checkpoints for a cheap tagged-versus-untagged VEP test. That test isolates inference-time dependence on the supplied tag; a tag-trained versus never-tag-trained model comparison is still needed to isolate the pretraining effect. |
| [Evo 2](https://doi.org/10.1038/s41586-026-10176-5) | A large sequence-only model uses long DNA prompts to recover alternative genetic codes and species-shaped completion statistics. | Long context can make explicit tags redundant. The result does not show that a short-window model can infer taxon efficiently. |
| [SynCodonLM](https://academic.oup.com/nar/article/54/5/gkag166/8496864) | Species-group embeddings produced similar average benchmark scores in small ablations, but the no-taxon model was less robust to synonymous perturbations and full-scale taxon IDs gave a small gain. | Taxon context may add coding priors while average gains remain small. The setup is restricted to CDS, codon tokens, and masked modeling. |
| [CodonTransformer](https://doi.org/10.1038/s41467-025-58588-7) | Organism embeddings enable host-specific codon optimization across 164 organisms. | Taxon conditioning supports controllable generation when host identity is part of the task. It does not isolate benefits for a general genomic LM. |
| [Nucleotide Transformer](https://doi.org/10.1038/s41592-024-02523-z), [DNABERT-2](https://openreview.net/forum?id=oMLQB4EZE1), and Evo 2 | These models pool multispecies sequence without explicit taxon tags and obtain useful representations or generation. | Sequence-only training is a strong baseline; a conditioning study must show added value at matched scale. |
| [DNABERT-S](https://arxiv.org/abs/2402.08777) and [ProGen](https://doi.org/10.1038/s41587-022-01618-2) | Species identity is recoverable from DNA embeddings, and taxonomic control tags are effective for protein generation. | The label is learnable and usable for control, but neither result shows that supplying it improves the biological signal sought here. |

## Contradictory evidence

The predecessor issue did not maintain a separate contradictory-evidence section. Its caveats and negative results are preserved in Current answer and Supporting evidence.

## Related experiments

- [#55](https://github.com/Open-Athena/marin-dna/issues/55) compared promoter training across human, primate, mammal, vertebrate, and animal corpora. Mammalian data learned human Mendelian VEP faster, but phylogenetic breadth, unique-data quantity, and epoch count changed together; the experiment motivates conditioning without testing it.
- [#58](https://github.com/Open-Athena/marin-dna/issues/58) repeated the timescale comparison for CDS and found a different trajectory: shallower arms led early, while the animal arm later became stronger for missense VEP. This suggests that useful taxonomic context depends on feature class and training stage.
- [#255](https://github.com/Open-Athena/marin-dna/issues/255) reduced the mammalian cohort from 108 family representatives to 19 order representatives at matched compute. Coding, 3′ UTR, and enhancer VEP were similar, while ncRNA and 5′ UTR/TSS VEP worsened, showing region-dependent value from species diversity.
- [#353](https://github.com/Open-Athena/marin-dna/issues/353) compared animal- and vertebrate-scale CDS corpora constructed by native annotation and human-anchored projection. The preferred breadth depended on evaluation and construction method, and nucleotide projection recovered little invertebrate sequence; nominal taxonomy and effective training context can differ.

## Open questions

- **Does it help at fixed compute and data?** Compare models trained on the same examples, order, optimizer, seed, and token budget. The primary contrast should be sequence-only versus correctly conditioned; a shuffled-label control can test whether any extra token/parameters alone explain the result.
- **Can the released Carbon checkpoints test inference-time VEP sensitivity now?** Start with Carbon-500M, then replicate any signal on 3B and 8B. Score the same development-safe human variants from odd-numbered autosomes and X under `<dna>` (untagged), `<vertebrate_mammalian><dna>` (correct), and equal-length wrong or shuffled taxon tags; optionally cross gene-type tags. Keep the checkpoint, window, strand handling, and scorer fixed. Compare paired per-variant score changes and consequence-stratified or macro AUPRC. Correct-versus-wrong tags control the semantics of the prefix more cleanly than correct-versus-untagged alone. This tests whether a trained Carbon model's VEP scores use metadata, not whether metadata-conditioned pretraining caused better VEP.
- **Species, clade, or hierarchy?** Compare a leaf-species token with hierarchical rank tokens (for example class/order/family/genus/species) or a shared phylogenetic embedding. A hierarchy may capture both universal and lineage-specific rules more efficiently than unrelated leaf embeddings.
- **How should unseen species work?** Candidate policies include an explicit unknown-taxon token, the nearest known clade, only the known prefix of the taxonomy path, or an embedding derived from phylogenetic distance. SpeciesLM's proxy-token workaround should not be assumed to generalize.
- **How redundant is the tag with sequence context?** Measure the conditioning gain across context lengths and genomic regions. Correct, null, and deliberately wrong tags at evaluation time would quantify how much the model actually uses the metadata.
- **Can one model remain unconditional?** Replace taxon tokens with fixed-position null tokens with probability $q$ during training, following the control-tag dropout idea used by LOL-EVE. Sweep $q$; evaluate both correct-tag and null-tag inference. Keeping fixed slots avoids conflating dropout with a positional shift.
- **Is the model learning useful biology or a shortcut?** Stratify language-modeling gains by coding/regulatory/background regions and by GC/repeat content. Test whether conditioning improves conserved-versus-background likelihood gaps and human VEP after controlling for simple composition.
- **Which outcomes decide the question?** At minimum: held-out language-model loss on seen and unseen taxa, region-relevant LL gaps, zero-shot Mendelian/complex-trait VEP AUPRC, and frozen-embedding probes. Correct-versus-wrong-tag sensitivity is a diagnostic, not the primary success metric.
- **Where should the first experiment live?** A short-window specialist with a known multi-species cohort is the cleanest first test; it avoids changing the data distribution and targets the regime in which taxon cannot always be inferred from long genomic context.

## History

- 2026-08-14 — Migrated from the predecessor research-question issue [#287](https://github.com/Open-Athena/marin-dna/issues/287). The issue remains the historical source for its original body and comments.
