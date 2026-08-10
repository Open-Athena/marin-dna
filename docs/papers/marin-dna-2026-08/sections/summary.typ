#import "../template.typ": paper-figure

MarinDNA applies the tools and #link("https://openathena.ai/blog/open-development-of-frontier-ai/")[open-development approach] of #link("https://github.com/marin-community/marin")[Marin] to genomic language modeling. This paper summarizes how data curation, hyperparameter transfer, scaling laws, and data-mixture experiments produced a 1B GPT-style model competitive with #link("https://doi.org/10.1038/s41586-026-10176-5")[Evo 2 40B], while using \~1,980× fewer training FLOPs and scoring variants \~2,330× faster.

#paper-figure(
  "figures/headline_cost_performance.svg",
  id: "fig-cost-performance",
  alt: "Zero-shot Mendelian VEP macro-average AUPRC versus variants scored per hour for MarinDNA 1B and Evo 2 1B, 7B, and 40B",
  caption: [*VEP performance versus inference throughput.* Zero-shot Mendelian VEP macro-average AUPRC versus variants scored per hour on a GH200.],
)
= Summary
<summary>
- #strong[Balanced data mixtures produce more even VEP performance across coding and non-coding regions.] Uniform weighting prevents the larger CDS dataset from dominating training. Adding ncRNA and enhancer data improves performance on ncRNA and distal variants.
- #strong[Optimization transfers, and loss scales predictably.] Hyperparameters tuned on \~25M-parameter reference models predicted the optimal learning rate for 255M, 476M, and 1B models; using that recipe, validation loss followed a clean scaling law through 4B parameters.
- #strong[Zero-shot VEP can regress with scale even as linear-probe VEP improves.] Zero-shot LLR improves for most variant classes, but deteriorates for Mendelian missense as model scale and validation log-likelihood increase.
- #strong[The resulting 1B model rivals Evo 2 40B on Mendelian VEP.] MarinDNA m5.1 slightly leads Evo 2 40B in zero-shot macro-average AUPRC while using \~1,980× fewer training FLOPs and scoring variants \~2,330× faster, although alignment-based and supervised models remain stronger overall.

#quote(block: true)[
#strong[A note on open development.] This paper turns a branching research process into a linear narrative; the #link("https://github.com/Open-Athena/marin-dna")[MarinDNA repository] preserves the underlying experiments, including unsuccessful and inconclusive directions.
]
