# Can autoregressive RAG gLMs be accurate and practical?

## Metadata

| Field | Value |
|---|---|
| Question ID | `RQ-0397` |
| Status | `active` |
| Overall confidence | `unknown` |
| Evidence considered through | `2026-08-13` |
| Predecessor issues | [#397](https://github.com/Open-Athena/marin-dna/issues/397) |

## Question and scope

Can an autoregressive retrieval-augmented genomic language model (gLM), trained to model sets of evolutionarily related but unaligned DNA sequences, improve variant effect prediction—especially for indels—and learned representations relative to both single-sequence autoregressive models and alignment-based models such as GPN-Star? Can it do so with a retrieval and inference system that is practical to deploy at genome scale, and how do model size, retrieval-corpus size, and the amount of retrieved context affect the accuracy–cost frontier?

## Current answer

A fixed-context mammalian-ortholog prototype produced promising VEP and representation results, but it lacks matched no-retrieval controls, online retrieval, and indel evaluation. Confidence is moderate that the model can use ortholog context and low that retrieval itself caused the gain or can be deployed efficiently.

Autoregressive retrieval-augmented genomic modeling is feasible, but neither its causal accuracy benefit nor its serving practicality is established. The only MarinDNA experiment used a fixed prefix of seven precomputed mammalian ortholog windows. Its 104M model produced promising zero-shot and frozen-probe point estimates and behaved as if it used ortholog context, but it lacked a matched human-only arm, online retrieval, species-order controls, and indel evaluation.

External results make the hypothesis plausible. Alignment-based genomic models show that ortholog context is highly informative; autoregressive protein models improve substitutions and indels with unaligned homologs; a DNA enhancer model generates conditioned on homolog sets; and learned protein retrievers can serve approximate-neighbor context quickly. These setups differ from genome-wide DNA retrieval in corpus scale, repeats, ambiguous orthology, and query coordinates.

The likely accuracy benefit is non-monotonic. More parameters, more training families, longer retrieved context, more homologs, and broader retrieval coverage are different scaling axes. Protein results show gains from some of these axes and saturation from others. A DNA model may benefit from a few informative orthologs while degrading when low-quality, repetitive, or redundant hits consume context.

Confidence is moderate that a reader can exploit curated ortholog context and low that retrieval itself caused the current MarinDNA gain or can be served economically genome-wide. The next decision gate is a matched human-only versus fixed-ortholog comparison with identical tokens and compute, followed by shuffled/wrong-species controls and indel scoring. Online retrieval and index-cost work should wait until the causal model benefit survives that gate.

## Confidence and limitations

A fixed-context mammalian-ortholog prototype produced promising VEP and representation results, but it lacks matched no-retrieval controls, online retrieval, and indel evaluation. Confidence is moderate that the model can use ortholog context and low that retrieval itself caused the gain or can be deployed efficiently.

Confidence is moderate that a reader can exploit curated ortholog context and low that retrieval itself caused the current MarinDNA gain or can be served economically genome-wide. The next decision gate is a matched human-only versus fixed-ortholog comparison with identical tokens and compute, followed by shuffled/wrong-species controls and indel scoring. Online retrieval and index-cost work should wait until the causal model benefit survives that gate.

## Operational consequence

Autoregressive retrieval-augmented genomic modeling is feasible, but neither its causal accuracy benefit nor its serving practicality is established. The only MarinDNA experiment used a fixed prefix of seven precomputed mammalian ortholog windows. Its 104M model produced promising zero-shot and frozen-probe point estimates and behaved as if it used ortholog context, but it lacked a matched human-only arm, online retrieval, species-order controls, and indel evaluation.

## Supporting evidence

- [GPN-Star](https://pmc.ncbi.nlm.nih.gov/articles/PMC12458161/) uses whole-genome alignments and an explicit species tree for coding and non-coding VEP and functional-region embeddings. It establishes the value of structured ortholog context and a strong accuracy baseline. Its reference-coordinate alignment input does not support arbitrary queries, general unaligned retrieval, or a standard likelihood-based indel scorer.
- [PoET](https://papers.nips.cc/paper_files/paper/2023/hash/f4366126eba252699b280e8f93c0ab2f-Abstract-Conference.html) autoregressively models sets of unaligned protein homologs and improves substitution and indel fitness prediction. Its context ablation improves from 4K to 8K tokens and then saturates or worsens at 16K, suggesting that retrieval quantity is not monotonic. Protein families are cleaner and smaller than genome-wide DNA retrieval domains.
- [Tranception](https://proceedings.mlr.press/v162/notin22a.html) mixes autoregressive likelihoods with statistics from retrieved homologs and improves substitutions, multiple mutants, and indels. It is a lightweight inference-time precedent; it does not learn a general retrieval-aware genomic representation.
- [EnhancAR](https://www.biorxiv.org/content/10.64898/2026.04.13.718170v1) trains an autoregressive DNA model on 1.7 million sets of unaligned enhancer homologs from a 241-species alignment and preserves predicted activity, specificity, and motif properties in generation. It is the closest DNA precedent. Its homolog sets still come from an alignment, and it does not test general VEP, de novo retrieval, or serving cost.
- [PoET-2](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4d19160864e6a644496d61b21c7e015a-Abstract-Conference.html) combines retrieval-conditioned causal and masked objectives and improves protein VEP, indels, and supervised sequence-function embeddings. It suggests that retrieval can help representations as well as likelihoods. DNA transfer remains untested.
- [RAG-ESM](https://openreview.net/forum?id=i4vevaqugi) adds homolog cross-attention to pretrained protein encoders, while [Profluent-E1](https://www.biorxiv.org/content/10.1101/2025.11.12.688125v1) trains native retrieval-aware encoders over unaligned homologs. Both improve selected protein prediction tasks and show alignment-like or retrieval-aware attention. Neither is autoregressive or genome-scale.
- [Protriever](https://proceedings.mlr.press/v267/weitzman25a.html) jointly trains a protein retriever and autoregressive reader and reports approximately 4.6 ms retrieval with a 12.6 GB compressed UniRef50 index. This is evidence that learned dense retrieval can be practical in proteins. Genomes are larger, repetitive non-coding sequence is common, and local orthology can be ambiguous, so the speed and index size do not transfer directly.
- The practical observables are retrieval recall for true orthologs, sensitivity to paralogs and repeats, accuracy versus homolog count and context length, per-query latency, index memory, preprocessing cost, and degradation under stale or incomplete corpora. None has been measured for MarinDNA.

## Contradictory evidence

The predecessor issue did not maintain a separate contradictory-evidence section. Its caveats and negative results are preserved in Current answer and Supporting evidence.

## Related experiments

- [#402](https://github.com/Open-Athena/marin-dna/issues/402) trained 46M and 104M causal models on seven fixed HAL-projected mammalian windows followed by the human window. The 104M arm produced promising official train-cohort zero-shot and frozen-probe point estimates and showed ortholog-context use, but no matched human-only arm, online retrieval, order ablation, convergence study, or indel evaluation was included.

## Open questions

- **Target and architecture.** Should the model directly generate a sequence of unaligned homologous sequences, as in PoET and EnhancAR, encode retrieved homologs separately and condition an autoregressive reader, or retrofit a strong pretrained single-sequence model with cross-attention, as in RAG-ESM? How should it represent species identity, phylogenetic distance, arbitrary homolog order, variable sequence length, and reverse complements?

- **What counts as useful retrieval?** Does the reader need strict orthologs, or can paralogs and more remote functional analogs help? How should we distinguish true evolutionary signal from repeats, low-complexity matches, assembly artifacts, and reference leakage?

- **Retrieval backend.** What is the best accuracy–cost tradeoff among direct lookup in a precomputed WGA, local sequence alignment against a genome corpus, dense embedding search, and a hybrid dense-retrieval-plus-alignment/reranking system? Can a retriever be trained jointly with the gLM, as in Protriever, and does task-aware retrieval outperform generic homology?

- **Embedding retrieval.** What training objective and segment granularity would make a DNA embedding index recover local orthology across substitutions, indels, rearrangements, and large evolutionary distances? How much recall against trusted WGA/orthology sets is required before dense retrieval improves the downstream reader?

- **VEP.** Does retrieval improve zero-shot performance on the existing Mendelian, complex-trait, saturation-genome-editing, and other SNV evaluations? More importantly, can full-sequence autoregressive likelihoods produce calibrated and length-robust scores for insertions and deletions that alignment-column models cannot naturally score?

- **Representations.** Do retrieval-conditioned embeddings improve functional-element separation, frozen-embedding linear probes, and other sequence-function tasks, as the protein results from E1, RAG-ESM, and PoET-2 suggest, or does retrieval mainly improve likelihood-based VEP in DNA? Which representation should be exported when the query is conditioned on a variable set of homologs, and is a causal objective sufficient or is a masked/dual objective needed?

- **Scaling.** Holding the retriever and data fixed, how does performance scale with model size? Holding the model fixed, how does it scale with training-family diversity, retrieval-corpus size, number and diversity of retrieved sequences, and total retrieved tokens? Where do these axes saturate, and which buys the most performance per unit of training and serving compute?

- **Deployment unit.** Is retrieval performed online per query window, amortized and cached per locus, or entirely offline for a fixed reference genome? The practical answer may differ for genome-wide precomputed VEP scores, interactive annotation of variants on a known reference, and scoring or generating arbitrary sequences.

- **Practicality criteria.** What index-build time, index size, memory footprint, p50/p95 retrieval latency, end-to-end throughput, cache hit rate, and dollar cost are acceptable? Retrieval and reader inference should be profiled separately, with WGA lookup, local alignment, and dense retrieval compared at matched downstream accuracy.

- **Evaluation hygiene.** How should genomes, loci, homolog families, and retrieval corpora be split to prevent near-duplicate or allele leakage? All comparisons need a no-retrieval ablation, matched reader capacity, fixed corpus versions, retrieval traces, and performance stratified by alignment depth, genomic region, repeat content, evolutionary distance, and indel length.

## History

- 2026-08-14 — Migrated from the predecessor research-question issue [#397](https://github.com/Open-Athena/marin-dna/issues/397). The issue remains the historical source for its original body and comments.
