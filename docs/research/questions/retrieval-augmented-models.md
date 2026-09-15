# Can autoregressive RAG gLMs be accurate and practical?

> [!NOTE]
> **TL;DR:** Small fixed-ortholog RAG readers exploit retrieved context; a broader-context, longer-trained 46M recipe improved Mendelian, SGE, and frozen-probe point estimates over the earlier 46M prototype but did not close the Complex Traits zero-shot gap; causal retrieval gains and genome-scale serving practicality remain unquantified, and online retrieval and indel accuracy remain untested.

## Question

Can an autoregressive retrieval-augmented genomic language model (gLM), trained to model sets of evolutionarily related but unaligned DNA sequences, improve variant effect prediction—especially for indels—and learned representations relative to both single-sequence autoregressive models and alignment-based models such as GPN-Star?
Can it do so with a retrieval and inference system that is practical to deploy at genome scale, and how do model size, retrieval-corpus size, and the amount of retrieved context affect the accuracy–cost frontier?

## Current answer

Autoregressive retrieval-augmented genomic modeling is feasible, and small readers exploit curated ortholog context.
In the [fixed-ortholog prototype](../experiments/402-fixed-ortholog-rag.md), context perturbations worsened human-token validation loss and the 104M model exceeded the 1B single-sequence m5.1 reference on all three zero-shot development-cohort point estimates and two of three frozen probes.
Gonzalo Benegas attributes the unusually strong small-model performance to retrieval based on these measurements and his knowledge of MarinDNA's experiment history and prior work.
No otherwise-matched single-sequence training arm was run.

The [five-region RAG experiment](../experiments/550-five-region-rag.md) broadened the panel from seven mammals to 39 non-human vertebrate order representatives, changed document construction, and trained a 46M reader for 100,000 updates.
Compared with the earlier 46M prototype, final zero-shot macro AUPRC increased from 0.3955 to 0.4407 on Mendelian and from 0.4767 to 0.5093 on SGE, while Complex Traits fell from 0.1840 to 0.1578.
Frozen-probe point estimates increased on all three benchmarks.
The combined recipe therefore extends the small-reader result, but does not isolate the effects of broader species coverage, ordering, taxonomic deduplication, or longer training.
Within the new lineage, the final checkpoint exceeded 50k on Mendelian, SGE, and all three frozen probes, while Complex Traits zero-shot rose from 20k to 50k before falling again at 100k.
GPN-Star retained the strongest zero-shot macro point estimate on each benchmark.

External results make the hypothesis plausible.
Alignment-based genomic models show that ortholog context is highly informative; autoregressive protein models improve substitutions and indels with unaligned homologs; a DNA enhancer model generates conditioned on homolog sets; and learned protein retrievers can serve approximate-neighbor context quickly.
These setups differ from genome-wide DNA retrieval in corpus scale, repeats, ambiguous orthology, and query coordinates.

The likely accuracy benefit is non-monotonic.
More parameters, more training families, longer retrieved context, more homologs, and broader retrieval coverage are different scaling axes.
Protein results show gains from some of these axes and saturation from others.
A DNA model may benefit from a few informative orthologs while degrading when low-quality, repetitive, or redundant hits consume context.

Confidence is high that a reader can exploit curated ortholog context.
The accuracy contribution of individual recipe choices remains uncertain because the completed comparisons use few model sizes, one seed per size and recipe, and no matched no-retrieval arm.
The informative next accuracy tests separate species coverage, document order, optimization exposure, and reader size while retaining the benchmark-specific readouts.
Online retrieval and index-cost work must establish whether the approach is practical beyond fixed reference-genome lookups; indel accuracy is still untested.

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
  No otherwise-matched single-sequence training arm was run.
  Online retrieval, an isolated order ablation, and indel evaluation remain untested.
- [#550: Five-region RAG with order-level vertebrate representatives](../experiments/550-five-region-rag.md) combined broader species context, revised document construction, and longer training in one 46M reader.
  It improved Mendelian, SGE, and frozen-probe point estimates over the earlier 46M prototype but left the Complex Traits zero-shot gap unresolved; the experiment cannot assign those changes to a single intervention.

</details>

<details>
<summary>Possible directions</summary>

- Separate the broader-context recipe into matched species-coverage, ordering, missing-window, and training-exposure arms with repeated seeds; test readers above 104M parameters as a distinct axis.
- Add matched no-retrieval, wrong-context, and order-ablation arms to estimate the incremental benefit and distinguish orthology from extra tokens.
- Evaluate SNVs and indels across retrieval depth, evolutionary distance, region, repeat content, and reader size.
- Compare precomputed whole-genome-alignment lookup, local alignment, dense retrieval, and hybrid reranking at matched downstream accuracy.
- Measure retrieval recall, index build and memory cost, p50/p95 latency, reader throughput, and caching for fixed-reference and arbitrary-sequence use cases.
- Split loci, genomes, and homolog families to prevent near-duplicate, allele, and reference leakage.

</details>
