# Can autoregressive RAG gLMs be accurate and practical?

> [!NOTE]
> **TL;DR:** The fixed-ortholog 104M model reached VEP performance not seen in comparable small single-sequence models, and perturbations confirmed use of the context; Gonzalo Benegas attributes the unusually strong performance to retrieval; broader species coverage, longer training, and larger readers are the clearest next accuracy axes; the causal gain and serving practicality remain unquantified, while online retrieval and indel accuracy remain untested.

## Question

Can an autoregressive retrieval-augmented genomic language model (gLM), trained to model sets of evolutionarily related but unaligned DNA sequences, improve variant effect prediction—especially for indels—and learned representations relative to both single-sequence autoregressive models and alignment-based models such as GPN-Star?
Can it do so with a retrieval and inference system that is practical to deploy at genome scale, and how do model size, retrieval-corpus size, and the amount of retrieved context affect the accuracy–cost frontier?

## Current answer

Autoregressive retrieval-augmented genomic modeling is feasible.
Gonzalo Benegas's assessment, based on MarinDNA's experiments and his prior work, is that fixed ortholog retrieval materially improves the accuracy of small readers.
The experiment did not quantify the gain under matched training conditions, and serving practicality is not established.
The only MarinDNA experiment used a fixed prefix of seven precomputed mammalian ortholog windows.
Its 104M model exceeded the 1B single-sequence m5.1 reference on all three zero-shot development-cohort point estimates and on the Complex Traits and SGE frozen probes.
Context perturbations worsened validation loss and changed model outputs.
Gonzalo reports that no single-sequence model in the 45M- to 104M-parameter range across MarinDNA's experiments, his prior work, or the broader work known to him has achieved comparable VEP performance.
These measurements and the historical comparison are the basis for his assessment.

The current result identifies three direct scale opportunities.
The strongest near-term axis is expansion beyond the current seven-species mammalian subset to more mammals and non-mammalian vertebrates.
Longer optimization is promising because both validation losses were still falling at the final 30,000-update checkpoint.
Larger readers are a third axis because only models up to 104M parameters were tested.
These extensions build directly on the demonstrated recipe.
Their gains have not yet been measured.

External results make the hypothesis plausible.
Alignment-based genomic models show that ortholog context is highly informative; autoregressive protein models improve substitutions and indels with unaligned homologs; a DNA enhancer model generates conditioned on homolog sets; and learned protein retrievers can serve approximate-neighbor context quickly.
These setups differ from genome-wide DNA retrieval in corpus scale, repeats, ambiguous orthology, and query coordinates.

The likely accuracy benefit is non-monotonic.
More parameters, more training families, longer retrieved context, more homologs, and broader retrieval coverage are different scaling axes.
Protein results show gains from some of these axes and saturation from others.
A DNA model may benefit from a few informative orthologs while degrading when low-quality, repetitive, or redundant hits consume context.

Confidence is high that a reader can exploit curated ortholog context.
Gonzalo assesses that retrieval drove a material part of the current small-model performance.
The size and generality of that contribution remain uncertain because the experiment used one context construction, one species order, and one seed per model size.
Beyond the three direct scale opportunities, the next accuracy questions are which species and ordering matter, whether the result transfers to indels, and how benefit changes with retrieved context.
Online retrieval and index-cost work must then establish whether the approach is practical beyond fixed reference-genome lookups.

<details>
<summary>Related work</summary>

- [GPN-Star](https://pmc.ncbi.nlm.nih.gov/articles/PMC12458161/) uses whole-genome alignments and an explicit species tree for coding and non-coding VEP and functional-region embeddings.
  It establishes the value of structured ortholog context and a strong accuracy baseline.
  Its reference-coordinate alignment input does not support arbitrary queries, general unaligned retrieval, or a standard likelihood-based indel scorer.
- [PoET](https://papers.nips.cc/paper_files/paper/2023/hash/f4366126eba252699b280e8f93c0ab2f-Abstract-Conference.html) autoregressively models sets of unaligned protein homologs and improves substitution and indel fitness prediction.
  Its context ablation improves from 4K to 8K tokens and then saturates or worsens at 16K, suggesting that retrieval quantity is not monotonic.
  Protein families are cleaner and smaller than genome-wide DNA retrieval domains.
- [Tranception](https://proceedings.mlr.press/v162/notin22a.html) mixes autoregressive likelihoods with statistics from retrieved homologs and improves substitutions, multiple mutants, and indels.
  It is a lightweight inference-time precedent; it does not learn a general retrieval-aware genomic representation.
- [EnhancAR](https://www.biorxiv.org/content/10.64898/2026.04.13.718170v1) trains an autoregressive DNA model on 1.7 million sets of unaligned enhancer homologs from a 241-species alignment and preserves predicted activity, specificity, and motif properties in generation.
  It is the closest DNA precedent.
  Its homolog sets still come from an alignment, and it does not test general VEP, de novo retrieval, or serving cost.
- [PoET-2](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4d19160864e6a644496d61b21c7e015a-Abstract-Conference.html) combines retrieval-conditioned causal and masked objectives and improves protein VEP, indels, and supervised sequence-function embeddings.
  It suggests that retrieval can help representations as well as likelihoods.
  DNA transfer remains untested.
- [RAG-ESM](https://openreview.net/forum?id=i4vevaqugi) adds homolog cross-attention to pretrained protein encoders, while [Profluent-E1](https://www.biorxiv.org/content/10.1101/2025.11.12.688125v1) trains native retrieval-aware encoders over unaligned homologs.
  Both improve selected protein prediction tasks and show alignment-like or retrieval-aware attention.
  Neither is autoregressive or genome-scale.
- [Protriever](https://proceedings.mlr.press/v267/weitzman25a.html) jointly trains a protein retriever and autoregressive reader and reports approximately 4.6 ms retrieval with a 12.6 GB compressed UniRef50 index.
  This is evidence that learned dense retrieval can be practical in proteins.
  Genomes are larger, repetitive non-coding sequence is common, and local orthology can be ambiguous, so the speed and index size do not transfer directly.

</details>

<details>
<summary>Related experiments</summary>

- [#402: Fixed-ortholog retrieval prototype](../experiments/402-fixed-ortholog-rag.md) trained 46M and 104M causal models on seven fixed HAL-projected mammalian windows followed by the human window.
  The 104M arm exceeded the 1B m5.1 reference on all three zero-shot development-cohort point estimates and two of three frozen probes, while perturbations confirmed ortholog-context use.
  A matched arm did not quantify the gain.
  Broader species coverage, longer training, and larger readers are promising extensions; online retrieval, order ablation, and indel evaluation remain untested.

</details>

<details>
<summary>Possible directions</summary>

- Expand the fixed recipe beyond seven mammals, train longer, and test readers above 104M parameters as separate axes with repeated seeds.
- Add matched no-retrieval, wrong-context, and order-ablation arms to estimate the incremental benefit and distinguish orthology from extra tokens.
- Evaluate SNVs and indels across retrieval depth, evolutionary distance, region, repeat content, and reader size.
- Compare precomputed whole-genome-alignment lookup, local alignment, dense retrieval, and hybrid reranking at matched downstream accuracy.
- Measure retrieval recall, index build and memory cost, p50/p95 latency, reader throughput, and caching for fixed-reference and arbitrary-sequence use cases.
- Split loci, genomes, and homolog families to prevent near-duplicate, allele, and reference leakage.

</details>
