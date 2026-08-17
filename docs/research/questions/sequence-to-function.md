# Can gLM pretraining improve human sequence-to-function modeling?

## TL;DR

Self-supervised gLM pretraining has not shown a consistent advantage for human sequence-to-function modeling under matched architectures and evaluation.
Positive results in other organisms and model families suggest the effect is regime-dependent; confidence is low, and the decisive gap is a scratch-versus-pretrained accessibility experiment with matched capacity and variant evaluation.

## Question

Can self-supervised genomic language model (gLM) pretraining improve human sequence-to-function models over the same downstream architecture trained from random initialization?
In which regimes—chromatin accessibility first, then gene expression—and through which transfer strategy (frozen features, partial fine-tuning, or full fine-tuning) does pretraining improve functional-track prediction and downstream variant-effect prediction?
All current positive precedents use bidirectional models; is bidirectionality important, and can it be obtained from existing causal MarinDNA checkpoints without retraining from scratch?
How do directionality and movement from one human reference genome to homologous functional sequence across wider evolutionary timescales affect whether gLM initialization improves full-data accuracy, sample efficiency, optimization, or generalization to unseen cell types, loci, and variant distributions?

## Current answer

Self-supervised gLM pretraining has not shown a consistent, controlled advantage for human sequence-to-function modeling.
Strong scratch-trained supervised models remain the main baseline, and the decisive MarinDNA frozen-versus-fine-tuned experiment has not been completed.

Published human evidence is mixed.
A short, regulatory-targeted masked model gives modest gains when its frozen per-base embeddings replace one-hot input to an accessibility model, while broader human DNALM benchmarks do not show a general advantage over strong ab-initio models.
Positive transfer is clearer in yeast and plants, where masked or bidirectional pretraining improves expression or functional-track prediction relative to random initialization.

Every positive precedent uses bidirectional representations.
MarinDNA’s causal checkpoints provide separate forward and reverse-complement views, but the two states do not interact inside the backbone.
Directionality, objective, training corpus, and downstream architecture are therefore confounded.

Confidence is low that generic causal pretraining will help without adaptation and moderate that a task-matched bidirectional or two-view representation can improve some accessibility settings.
The first decision gate is a matched accessibility experiment with identical downstream capacity, chromosome splits, labels, and optimization: random encoder, frozen pretrained encoder, end-to-end fine-tuning, one-hot baseline, and a bidirectional reference.
It should report full-data accuracy, label efficiency, unseen-cell/locus transfer, and caQTL/dsQTL effects separately.
Gene expression is a later long-context question.

<details>
<summary>Related work</summary>

- [DART-Eval](https://proceedings.neurips.cc/paper_files/paper/2024/hash/71998bfc3217ffe1cca1ee084dfadadd-Abstract-Datasets_and_Benchmarks_Track.html) compares human DNALMs on regulatory tasks and finds inconsistent gains over strong supervised baselines.
  It establishes the need for matched ab-initio controls.
  It does not test the exact MarinDNA architecture or every end-to-end fine-tuning regime.
- [AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0) learns thousands of human and mouse functional tracks from 1 Mb sequence and performs strongly across track and variant tasks without reported self-supervised gLM initialization.
  It defines a high supervised baseline and a long-context architecture precedent.
  It does not isolate whether self-supervised initialization would improve the same model.
- [ARSENAL](https://www.biorxiv.org/content/10.64898/2026.02.05.703637v3) pretrains a 350 bp masked model on human ENCODE cCREs and tiles frozen per-base embeddings into a ChromBPNet-style model.
  It reports accessibility gains across five cell lines and modest caQTL/dsQTL improvements.
  The setup is task-matched, human-only, short-context, and bidirectional, so corpus breadth and directionality remain confounded.
- [Shorkie](https://pubmed.ncbi.nlm.nih.gov/41282136/) initializes a yeast sequence-to-function model from masked pretraining on 165 fungi and improves expression and regulatory-variant prediction over random initialization.
  This is direct evidence for evolutionary pretraining transfer outside humans; the gap is transfer to stronger human supervised baselines.
- The [original GPN preprint](https://www.biorxiv.org/content/10.1101/2022.08.22.504706v1) fine-tunes a single-reference Arabidopsis model on 106 functional tracks and outperforms de novo DeepSEA and human-pretrained DNABERT controls.
  It shows that multispecies data are not required for transfer, while leaving the benefit of additional evolutionary breadth open.
- A [PlantCaduceus sequence-to-expression study](https://www.biorxiv.org/content/10.64898/2026.02.27.708524v2) and [PlantCAD2](https://pmc.ncbi.nlm.nih.gov/articles/PMC12425018/) report transfer across plant expression, accessibility, and abundance tasks.
  These broaden the positive precedent but do not isolate the architecture, objective, or data property that enables transfer.
- [#236](https://github.com/Open-Athena/marin-dna/issues/236) defines the ChromBPNet-on-gLM-embeddings evaluation harness.
  It is infrastructure rather than a completed experiment.
- [How should a short-context gLM acquire long-range context?](long-context.md) covers connecting a short local encoder to longer sequence-to-function models; [Can causal gLM checkpoints be cheaply adapted into bidirectional representation models?](bidirectional-models.md) covers causal-to-bidirectional conversion; [Which genomic regions to train on, and how to find them?](training-regions.md) covers the pretraining footprint.
  These are coupled design axes that must be controlled rather than attributed to pretraining as one bundle.

</details>

<details>
<summary>Related experiments</summary>

- [#243](https://github.com/Open-Athena/marin-dna/issues/243) scopes the direct ChromBPNet-on-MarinDNA comparison with frozen and fine-tuned encoders.
  Its body does not record a completed controlled result, so the main human transfer question remains open.
- [#314](https://github.com/Open-Athena/marin-dna/issues/314) evaluated frozen gLM embeddings for VEP across representation designs.
  Accessibility-QTL signal was weak, with best dsQTL AUPRC around 0.063; this argues against a shallow probe on generic frozen causal embeddings but does not test spatial heads, end-to-end fine-tuning, or bidirectional conversion.

</details>

## Possible directions

### What is the decisive accessibility experiment?

On identical chromosome splits and GM12878 functional data, compare at minimum:

1. the standard one-hot ChromBPNet baseline;
2. the same embedding-input architecture with a randomly initialized encoder;
3. a frozen pretrained MarinDNA encoder;
4. the same pretrained encoder fine-tuned end to end; and
5. ARSENAL under the same evaluation protocol.

- Is one-hot ChromBPNet the only relevant scratch baseline, or do we also need the exact gLM-plus-head architecture from random initialization to separate initialization from added capacity?
- Does pretraining improve held-out accessibility profiles, caQTL/dsQTL variant effects, or both?
  These are related but distinct outcomes and should not be collapsed into one verdict.
- Are gains consistent across cell types and assays, or restricted to GM12878 DNase data that closely matches the pretraining and benchmark distributions?

### When and why does transfer help?

- How does the pretrained-versus-random-init gap change with 1%, 10%, and 100% of the functional training data?
- Does pretraining primarily improve early optimization and convergence speed, or does it improve the final attainable solution after both arms are trained to convergence?
- Which layers should be frozen, gradually unfrozen, or fully fine-tuned?
  How much task-specific adaptation is needed before the original gLM representation is overwritten?
- Does the benefit come from self-supervised learning itself, from enrichment of the pretraining corpus for regulatory sequence, from multispecies evolutionary signal, or from the motif-scale inductive bias?
  Matched ablations should separate these factors rather than treating “pretraining” as one intervention.
- Are per-nucleotide embeddings necessary, or can a cheaper pooled or hierarchical interface preserve the benefit?

### Is bidirectionality important, and can we obtain it?

- Within the ChromBPNet experiment, compare causal-only embeddings, FWD+RC concatenation, and a genuinely bidirectional conversion of the same MarinDNA checkpoint.
  Keep the downstream head, functional labels, adaptation data, and compute matched.
- Is FWD+RC concatenation sufficient because the ChromBPNet head can learn cross-flank interactions, or does integration within every gLM layer provide additional signal?
- Can a completed causal checkpoint be cheaply adapted with full attention and masked next-token training, as proposed in [Can causal gLM checkpoints be cheaply adapted into bidirectional representation models?](bidirectional-models.md), or does representation transfer require a larger masked phase?
- How much masked adaptation is required before the converted model behaves bidirectionally?
  Compare no adaptation, parameter-efficient adaptation, and short full-parameter adaptation at the same sidecar budget.
- Separate bidirectional information flow from the masked objective and from extra optimization.
  Include a causal continued-pretraining arm and, if possible, a matched bidirectional architecture trained under alternative objectives.
- Does bidirectionality help accessibility-track prediction, caQTL/dsQTL effects, gene-expression prediction, and representation probes equally, or only tasks where a position's interpretation strongly depends on both flanks?
- Treat the source causal checkpoint and adapted bidirectional fork as separate artifacts.
  Autoregressive generation in the adapted fork is not a decision gate.
- Test directionality and evolutionary breadth factorially.
  A human-only bidirectional model versus a multispecies causal model cannot reveal which ingredient caused a gain.

### How much evolutionary breadth is useful?

- Compare ARSENAL-style human-reference-only pretraining with human population variation and progressively broader primate, mammal, vertebrate, and animal corpora.
  Which evolutionary timescale gives the strongest transfer to human accessibility and expression?
- Hold the number of training tokens, functional-region composition, model size, and compute fixed.
  Otherwise, a multispecies advantage could merely reflect more data or a denser functional training distribution.
- Should the comparison use orthologous projections of the same human regulatory elements, each species' native annotations, whole-genome sequence, or matched mixtures?
  These expose different biological information and should not be conflated.
- Do closer species provide the best balance between alignable regulatory grammar and informative variation, while distant species dilute lineage-specific human regulation?
  Is the optimal timescale different for promoters, enhancers, splice regions, and coding sequence?
- Does explicit alignment or conservation information add value beyond training a single-sequence gLM on the same multispecies corpus?
- Does evolutionary breadth improve full-data performance, label efficiency, cross-cell-type transfer, or variant-effect prediction most strongly?
  A benefit confined to one regime is still useful but should be stated narrowly.

### What constitutes fair evidence?

- Hold downstream architecture, labels, chromosome splits, optimizer search, stopping rule, and augmentation fixed.
  Train all arms to comparable convergence rather than giving the pretrained arm a privileged recipe.
- Report both downstream-only compute and total compute including gLM pretraining.
  The former measures amortized reuse; the latter measures whether pretraining is worthwhile for a single task.
- Match parameter count where possible and report trainable versus frozen parameters, examples seen, FLOPs, peak memory, and wall time.
- Test for sequence-similarity and locus leakage, especially when regulatory elements used for self-supervised pretraining overlap the genomic universe used for downstream supervision.
- Predefine whether success means higher full-data accuracy, improved label efficiency, better out-of-distribution transfer, lower compute, or some combination.
  A small gain on one metric should not silently become a general claim about transfer learning.

### How should we progress from accessibility to gene expression?

- What is the smallest gene-expression task with strong scratch baselines and enough long-range dependence to distinguish useful transfer from extra local capacity?
- Should the gLM replace the local convolutional encoder in an AlphaGenome- or Borzoi-style model, feed a hierarchical long-context model, or first undergo long-context language-model adaptation?
- Does accessibility pretraining provide an intermediate supervised stage between self-supervised gLM training and expression prediction, and if so, how do we distinguish the value of gLM initialization from ordinary supervised multitask transfer?
- Does one gLM initialization improve accessibility, histone marks, TF binding, and expression together, or are task-specific pretrained models more effective?
- For expression, should the primary readout be track correlation, gene-level expression, eQTL direction/causality, or perturbation response?
  Improvements in average coverage prediction may not translate to better variant effects.
