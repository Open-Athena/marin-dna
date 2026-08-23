# Can gLM pretraining improve human sequence-to-function modeling?

> [!NOTE]
> **TL;DR:** Self-supervised gLM pretraining has not shown a consistent advantage for human sequence-to-function modeling under matched architectures and evaluation; positive results elsewhere suggest the effect is regime-dependent, so confidence is low pending a scratch-versus-pretrained accessibility experiment with matched capacity and variant evaluation.

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

</details>

<details>
<summary>Related experiments</summary>

- [#243](https://github.com/Open-Athena/marin-dna/issues/243) scopes the direct ChromBPNet-on-MarinDNA comparison with frozen and fine-tuned encoders.
  Its body does not record a completed controlled result, so the main human transfer question remains open.
- [#314](https://github.com/Open-Athena/marin-dna/issues/314) evaluated frozen gLM embeddings for VEP across representation designs.
  Accessibility-QTL signal was weak, with best dsQTL AUPRC around 0.063; this argues against a shallow probe on generic frozen causal embeddings but does not test spatial heads, end-to-end fine-tuning, or bidirectional conversion.

</details>

<details>
<summary>Possible directions</summary>

- Run the matched accessibility comparison on identical chromosome splits and labels: one-hot ChromBPNet, the same encoder-and-head architecture from random initialization, frozen MarinDNA initialization, end-to-end MarinDNA fine-tuning, and ARSENAL.
- Report accessibility profiles and caQTL/dsQTL effects separately, including full-data quality, label efficiency, unseen-cell or locus transfer, and downstream-only versus total compute.
- Compare causal FWD/RC features with a genuinely bidirectional representation while holding the downstream head, adaptation data, and compute fixed.
- If pretraining helps, isolate regulatory enrichment and evolutionary breadth with matched corpus ablations rather than changing data and directionality together.
- Advance to gene expression only after the accessibility benchmark establishes a reproducible transfer gain and a long-range task with a strong scratch baseline.

</details>
