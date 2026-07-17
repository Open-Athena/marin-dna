---
title: "Genomic Language Model Optimization"
slug: "genomic-lm-optimization"
author: "Eric Czech & Gonzalo Benegas"
date: 2026-06-25
published: true
math: false
toc: true
tags:
  - Marin
summary: "How Marin can be used to train single-sequence, vanilla Transformer gLMs comparable to Evo 2 40B with ~1,980× fewer FLOPs, via hyperparameter transfer, scaling laws, and data-mixture experiments."
---

<style>
/* Math labels in the figures are positioned per-glyph from DejaVu Sans metrics,
   so they're left in DejaVu (the rest of the labels inherit the page font). Embed
   DejaVu so those positions render correctly instead of mis-spacing on a fallback. */
@font-face {
  font-family: 'DejaVu Sans';
  src: url('assets/fonts/DejaVuSans.woff2') format('woff2');
  font-weight: normal;
  font-style: normal;
  font-display: swap;
}
.blog-post-content figure img,
.blog-post-content figure svg.figure-svg {
  background: #ece3d5;
  padding: 1rem 1.25rem;
  border-radius: 10px;
  box-shadow: 0 1px 4px rgba(31, 30, 27, 0.10);
}
/* Inlined matplotlib figures: the viewBox carries the aspect ratio, so let the
   SVG fill the column and scale its height automatically. Authored with live
   <text> and currentColor (see site/build.py + utils.figure_theme), so labels
   render in the page font and follow the page ink. */
.blog-post-content figure svg.figure-svg {
  display: block;
  width: 100%;
  height: auto;
  box-sizing: border-box;
}
/* Nested lists inherit ul's 1rem margin-bottom, which stacks with the parent
   <li>'s margin to leave an oversized gap before the next top-level bullet.
   Zero it so nested bullets sit flush with the following item. */
.blog-post-content li > ul,
.blog-post-content li > ol {
  margin-bottom: 0;
}
</style>

How Marin can be used to train single-sequence, vanilla Transformer gLMs comparable to Evo 2 40B with ~1,980× fewer FLOPs. Covers DNA hyperparameter transfer, scaling law, and data mixture experiments.

## Introduction

Optimization of genomic language models (gLMs) has historically involved a lot of focus on model architecture. At a high level, the field has explored many architecture ideas borrowed from language and vision,[^glm-architecture] methods for making raw DNA usable at long context,[^glm-tokenization] and genomics-specific priors that encode biological symmetries, structure, or evolution.[^glm-biology] Our results here show that these inductive biases are not necessary for human variant effect prediction (VEP), arguably the most important near-term use case for gLMs. In the zero-shot setting, a standard GPT-style model can surpass Evo 2 40B when it is paired with careful data curation, the scaling practices we used previously in [Delphi](https://openathena.ai/blog/delphi/), and a set of less-principled ad hoc data mixture optimizations. The final model in this line of experiments does so with 1.8% as many training tokens (166B vs. 9.3T) and roughly 0.05% as many FLOPs (1.1e21 vs. 2.25e24).

### Why VEP evaluation?

VEP is arguably the most important application of gLMs. A useful VEP model can help scale clinical interpretation for rare disease, hereditary cancer, and variants of uncertain significance.[^vep-clinical] It can also help connect genetic association signals to disease mechanisms, target selection, and causal-variant prioritization in GWAS fine-mapping.[^vep-therapeutic] The same kind of evidence is relevant to clinical trial design when genetics can inform patient stratification, enrollment criteria, or mechanism-based cohort definition. Together, these are commonly used levers for improving the efficiency of pharmaceutical development, and it is uncommon for other gLM evaluations to have such a direct connection to commercially relevant research tasks. VEP is also one of the few evaluations backed by decades of costly clinical genetics curation, with resources such as ClinVar and OMIM providing a level of human variant evidence that has no real analogue in other species.[^human-variant-curation] That combination makes it a substantive test of whether a gLM has learned sequence constraints that actually matter for human biology. If a model has learned useful sequence-level constraints from DNA alone, it should help rank variants in places where direct experimental evidence is weak or nonexistent.

### Why Evo 2 40B baseline?

Evo 2 40B (published ~Feb. 2025) is still the most formidable relevant baseline among unsupervised, single-sequence DNA models. Within that same setting, we are not aware of another method with comparable performance across diverse, conserved genomic regions.[^evo2-regions] Stronger VEP models exist, but they use more information than unsupervised training on DNA sequence alone. The other reason Evo 2 40B matters is its training budget. Its reported 2.25e24 training FLOPs are unrivaled among gLMs, corresponding to roughly $2.5M of H100 time.[^evo2-cost] That budget is unusual in biology and comparable to major open-weight LLM training runs from recent model generations,[^evo2-llm-compute] e.g. just above Qwen2.5-14B and below Qwen2.5-32B, and roughly between DeepSeek-V2 and DeepSeek-V3. Since the literature has not really moved past this target yet, we believe it is the right baseline for asking whether a much simpler single-sequence gLM can be competitive.

### Why GPT-style architecture?

By GPT-style, we mean the dumb approach of training a stock causal, autoregressive, decoder-only language-model architecture on DNA that we pretend is text. In these experiments, that architecture is literally Qwen3 rather than a genomics-specific design. This approach is not new; a substantial line of prior gLM work has used causal language modeling with GPT- or Llama-like architectures.[^causal-glm-precedent] What is new here is the quality target. Even recent models in this family, such as Carbon, generally aim for non-inferiority to smaller Evo 2 checkpoints and still underperform Evo 2 40B on the broad zero-shot VEP setting we care about.[^carbon-eval] If the quality gap can be closed, GPT-style models have obvious advantages for deployment. They run through familiar training and inference stacks, move cleanly across hardware, and avoid model-specific kernels or bespoke architecture code, which matters a lot for cost, flexibility, and usability. E.g., the inference cost associated with the evaluations below is roughly $10 / billion tokens for our 1B model, compared with roughly $100 / billion tokens for Evo 2 40B (TODO: get real numbers).[^throughput-comparison]

[^glm-architecture]: Examples include long-convolution or hybrid long-context models such as [HyenaDNA](https://arxiv.org/abs/2306.15794) and [Evo 2](https://doi.org/10.1101/2025.02.18.638918); U-Net-like sequence-function models such as [NTv3](https://doi.org/10.64898/2025.12.22.695963); bidirectional models such as [DNABERT-2](https://arxiv.org/abs/2306.15006), [GenSLM](https://www.biorxiv.org/content/10.1101/2023.06.12.544594v3.full.pdf), [Caduceus](https://arxiv.org/abs/2403.03234), [PlantCAD2](https://doi.org/10.1101/2025.08.27.672609), and [TrinityDNA](https://arxiv.org/abs/2507.19229); state-space or hybrid state-space models such as [HybriDNA](https://arxiv.org/abs/2502.10807), [Caduceus](https://arxiv.org/abs/2403.03234), and [PlantCAD2](https://doi.org/10.1101/2025.08.27.672609); and early or less-established sparse-expert models such as [JanusDNA](https://arxiv.org/abs/2505.17257), [PlantBiMoE](https://arxiv.org/abs/2512.07113), and [MxDNA](https://arxiv.org/abs/2412.13716).

[^glm-tokenization]: Examples include learned or tokenizer-free approaches such as [dnaHNet](https://arxiv.org/abs/2602.10603) and [DNACHUNKER](https://arxiv.org/abs/2601.03019), multi-scale Transformers such as [MegaDNA](https://www.biorxiv.org/content/10.1101/2023.12.18.572218v3.full), and multi-scale attention in [TrinityDNA](https://arxiv.org/abs/2507.19229).

[^glm-biology]: Examples include reverse-complement equivariance in [Caduceus](https://arxiv.org/abs/2403.03234), double-helix groove fusion in [TrinityDNA](https://arxiv.org/abs/2507.19229), genomic loss weighting in [Evo 2](https://doi.org/10.1101/2025.02.18.638918) and [GPN](https://www.pnas.org/doi/10.1073/pnas.2311219120), factorized nucleotide supervision in [GENERATOR-v2](https://doi.org/10.64898/2026.01.27.702015) and related objective design in [Carbon](https://doi.org/10.64898/2026.05.22.727119). Outside unsupervised, single-sequence DNA language modeling, related architectural examples include the convolutional U-Net Transformer plus pairwise contact-map model in [AlphaGenome](https://doi.org/10.1101/2025.06.25.661532) and sequence-alignment plus phylogeny-aware attention in [GPN-Star](https://doi.org/10.1101/2025.09.21.677619).

[^vep-clinical]: Examples include zero-shot or disease-focused variant interpretation results in [Evo 2](https://doi.org/10.1101/2025.02.18.638918), [GPN-Star](https://doi.org/10.1101/2025.09.21.677619), [Carbon](https://doi.org/10.64898/2026.05.22.727119), and [EnTao-GPM](https://arxiv.org/abs/2507.21706).

[^vep-therapeutic]: Examples include fine-mapped GWAS and broader human-genetics results in [GPN-Star](https://doi.org/10.1101/2025.09.21.677619), regulatory variant-effect prediction in [AlphaGenome](https://doi.org/10.1101/2025.06.25.661532) and [ChromBPNet](https://www.biorxiv.org/content/10.1101/2024.12.25.630221v2), and the broader observation that human genetic evidence can support target-disease hypotheses in drug discovery in [Nelson et al.](https://doi.org/10.1038/ng.3314).

[^human-variant-curation]: [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/intro/) archives submitted reports relating human genomic variation to disease, cancer, drug response, and supporting evidence; [OMIM](https://doi.org/10.1093/nar/gku1205) is a curated catalog of human genes, genetic disorders, and gene-phenotype relationships. Nothing comparable exists for any other species: this depth reflects decades of clinical genetics effort directed specifically at human disease, an investment that has simply not been made for non-human genomes.

[^evo2-regions]: Here, diverse conserved genomic regions means variant effects spanning non-coding enhancers, promoters, UTRs, coding exons, and introns.

[^evo2-cost]: This estimate uses the [Evo 2](https://doi.org/10.1101/2025.02.18.638918) reported training compute of 2.25e24 FLOPs, 50% H100 model FLOP utilization following the costing convention in [Beyond Chinchilla](https://arxiv.org/abs/2401.00448), 989 TFLOP/s BF16 peak throughput for an H100 SXM, and $2 per H100-hour from [OLMo 3](https://arxiv.org/abs/2512.13961). The resulting accelerator requirement is about 1.26M H100-hours.

[^evo2-llm-compute]: Other over/under examples give the same intuition. The [AI2 OLMo 2 32B model card](https://huggingface.co/allenai/OLMo-2-0325-32B) places Evo 2 40B above Gemma 2 27B, OLMo 2 32B, and Llama 3.1 8B. A dense-accounting estimate from the [Llama 3.1 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) places it well below Llama 3.1 70B.

[^causal-glm-precedent]: GPT-style or otherwise causal genomic models include [GenSLM](https://pmc.ncbi.nlm.nih.gov/articles/PMC9709791/), [DNAGPT](https://arxiv.org/abs/2307.05628), [METAGENE-1](https://arxiv.org/abs/2501.02045), [GENERATOR](https://arxiv.org/abs/2502.07272), [GENERATOR-v2](https://doi.org/10.64898/2026.01.27.702015), [Gene42](https://arxiv.org/abs/2503.16565), and [Carbon](https://doi.org/10.64898/2026.05.22.727119). The closest human-DNA precedents are Carbon, GENERATOR, Gene42, and DNAGPT; several of the others are important causal gLM examples but are less directly relevant to human VEP.

[^carbon-eval]: The [Carbon-3B model card](https://huggingface.co/HuggingFaceBio/Carbon-3B) describes Carbon-3B as a 3B-parameter decoder-only autoregressive genomic model implemented as a stock `LlamaForCausalLM`, with 6-mer DNA tokenization, long-context support, and a two-stage training schedule that switches from a standard cross-entropy objective to a factorized nucleotide supervision loss, bridging its coarse 6-mer tokenization with single-nucleotide resolution. Its public zero-shot table compares to Evo 2 7B, not Evo 2 40B: Carbon-3B is slightly ahead on BRCA2 and ClinVar noncoding, but behind on ClinVar coding and TraitGym Mendelian.

[^throughput-comparison]: This calculation reuses the same $2 per H100-hour cost assumption from the Evo 2 training-cost estimate above, with draft throughput estimates of roughly 50k tokens / sec for our 1B model and 5k tokens / sec for Evo 2 40B, normalized to one H100 at peak BF16 throughput. The arithmetic is $2 / (3600 seconds x tokens / second) x 1B tokens, giving about $10 / billion tokens and $100 / billion tokens after rounding, respectively. The broader point is the order-of-magnitude usability comparison: standard GPT-style models can use common training and inference stacks such as Levanter, Hugging Face Transformers, vLLM, or SGLang, while large bespoke architectures are harder to serve and optimize.

## Results

### Hyperparameter transfer

The conserved metazoan DNA pool we can plausibly train on is only O(100B) tokens.[^metazoan-token-pool] That is large by genomics standards, but small relative to modern accelerator chips, making this a data-constrained problem in principle even though our practical constraints are messier. We are building on preemptible Google TPU Research Cloud resources, do not have consistent access to slices much larger than roughly 32 H100s worth of peak FLOPs, and want the recipe to remain reproducible at academic compute scale. O(100B) tokens therefore lands in an awkward middle ground where compute-constrained methods are still relevant, even though modest epoching is possible and likely breaks their assumptions at some unknown rate. The first thing to check is whether the learning-rate prediction survives that regime (Figure 1).

![Learning-rate transfer across model scales](/assets/images/blog/genomic-lm-optimization/figure1_lr_transfer.svg)

**Figure 1:** Learning-rate (LR) transfer across the 255M, 476M, and 1B validation scales. The `control` run type indicates final loss from the optimal configuration found in the initial smaller-scale reference sweep. The predicted, optimal LR results in a better loss than both this control and all other configurations at the same scale (`sweep` run type), for all model sizes.

We started with hyperparameter transfer for that reason. If a proven data-constrained transfer framework existed, we would use it. We do not know of one, so we followed the same basic pattern as [Delphi](https://openathena.ai/blog/delphi/), fitting a small reference sweep with the Vizier Bayesian optimization framework and then scaling the result using Complete(d).[^completed-framework] The reference sweep used ~25M-parameter models trained for 2.5B tokens with a 16k-token batch, or roughly 4e17 FLOPs per run. We then validated the transferred hyperparameters from 255M to 1B parameters, with 4x as many tokens, 1/4x the batch size, and roughly 170x the FLOPs per run. The less sensitive optimizer hyperparameters are shown separately in Figure 2.

![Adam beta2 and epsilon transfer across model scales](/assets/images/blog/genomic-lm-optimization/figure2_beta2_epsilon_transfer.svg)

**Figure 2:** Adam β₂ and ε transfer across the same scales as Figure 1.

That validation is a fairly unforgiving test. If the transferred learning rate were merely close by accident, it would be surprising for it to land correctly across all three validation scales, but the prediction remains well centered at each one. For DNA, that is a pretty cool result. Prior biology foundation-model work has used μP-style transfer, but we are not aware of a DNA result showing that a more inclusive framework like Complete(d) works across token horizon and batch size, which are the axes we keep leaning on later in ad hoc runs across epochs. The same is mostly true for the other optimizer hyperparameters too, although Adam β₂ shows some signs of being a bit aggressive at the largest scale. Figure 3 makes the same point across CDS, upstream, and downstream sequence, with no qualitative difference in transfer behavior across region types. That gives us enough confidence that the following parameter-scaling runs are at least close to optimally configured.

<details>
<summary>Transfer validation by region (Figure 3)</summary>

![Hyperparameter transfer validated per genomic region](/assets/images/blog/genomic-lm-optimization/figure3_region_hyper_transfer.svg)

**Figure 3:** Hyperparameter transfer validated separately for each genomic region (CDS, upstream, downstream).

</details>

[^metazoan-token-pool]: The proportional mix used here spans CDS, upstream, and downstream regions, with ~331M training examples and ~85B tokens from 366,411 RefSeq accessions across ~500 animal species. Each region is cycled only once, which is necessary to avoid invalidating assumptions of common hyperparameter transfer frameworks.

[^completed-framework]: Complete(d) refers to the compute-constrained hyperparameter-transfer framework described in [Complete(d): Data-Optimizing Hyperparameter Transfer](https://arxiv.org/abs/2512.22382).

### Parameter scaling

Before asking whether better validation loss translates into better VEP performance, we first needed to check whether validation loss scaled the way it should. The parameter sweep uses the same training recipe at each model size, with all hyperparameters set by the transfer heuristic above, and then asks whether the resulting losses fit a Kaplan-style scaling law well (they do).[^kaplan-scaling] Despite this being a simple experiment conceptually, actually getting there took months — fitting the hyperparameter transfer heuristic, running the validation experiments, and training the 4B model, which alone took about three weeks to finish. The final sweep spans 8 model sizes from 46M to 4B parameters, each trained on ~84B tokens, for ~4.3e21 FLOPs across the sweep. That puts it on par with canonical scaling-law studies in language modeling, e.g. its ~2.1e21 FLOP 4B run matches the compute Hugging Face used at that exact model scale in their data-constrained scaling work.[^muennighoff]

![Loss scaling across model sizes with Kaplan power-law fits](/assets/images/blog/genomic-lm-optimization/figure4_loss_scaling.svg)

**Figure 4:** Loss scaling across 8 model sizes (46M–4B params), with Kaplan power-law fit.

The result is about as tidy as we could hope for. Training is stable at every scale, and both training and validation loss decrease monotonically and predictably (Figure 4). We use WSD learning-rate schedules with 10% warmup and 20% decay, which causes the visible drop in both losses over the final 20% of tokens. Most importantly, the sweep gives a high-quality Kaplan scaling-law fit (R<sup>2</sup>=0.999), which makes the next question much better posed. Does lower validation loss actually correlate with better downstream VEP performance?

[^kaplan-scaling]: This follows the empirical scaling-law setup from [Kaplan et al.](https://arxiv.org/abs/2001.08361), where model loss is fit as a predictable function of model size, data, and compute.

[^muennighoff]: See Figure 4 of [Muennighoff et al.](https://arxiv.org/abs/2305.16264), "Scaling Data-Constrained Language Models" (NeurIPS 2023).

### Downstream performance

The relationship between validation loss and VEP performance is much less tidy. That is not a new or unexpected finding, but it was not obvious at the start whether better tuning and a more controlled parameter sweep would make the downstream picture less messy, and the sweep shows the same basic problem. VEP performance is not monotonic in parameter count for most variant types (Figure 5), nor does it correlate well with validation loss (Figure 6). CDS tasks peak around the middle of the sweep, upstream tasks improve more clearly with scale, and the remaining variant types are mixed.

![Composite VEP AUPRC vs parameter count](/assets/images/blog/genomic-lm-optimization/figure5_params_vs_vep_auprc.svg)

**Figure 5:** Composite VEP AUPRC vs parameter count.

![Composite VEP AUPRC vs validation loss](/assets/images/blog/genomic-lm-optimization/figure6_loss_vs_vep_auprc.svg)

**Figure 6:** Composite VEP AUPRC vs validation loss.

Token scaling at a fixed model size is not much cleaner. Within individual runs, VEP often improves early and then flattens or degrades, and the shape of that curve changes with model scale (Figure 7). The 128M model is especially prone to degradation, the 1B model continues to improve on several tasks, and the 4B model shows non-monotonic missense gains, which is especially discouraging given the direct relevance of coding amino-acid changes to protein-target drug development and the fact that this is our most prevalent class of variants to evaluate on.

![VEP AUPRC training curves by model scale](/assets/images/blog/genomic-lm-optimization/figure7_loss_vs_traitgym_curves.svg)

**Figure 7:** VEP AUPRC training curves by model scale.

Ultimately, the most useful finding is that monotonicity is scale-dependent (Figure 8). Mid-sized models are the most reliable by this measure, which gave us a practical target range for later experiments that train beyond one pass through the data.

![Loss vs VEP AUPRC correlation within model-size ranges](/assets/images/blog/genomic-lm-optimization/figure8_loss_vs_traitgym_correlation.svg)

**Figure 8:** Loss vs VEP AUPRC correlation during training. Bars show the mean Spearman ρ across variant classes for each model size; heatmap cells show the corresponding per-class correlations between validation loss and VEP AUPRC sampled over training.

### Mixture experiments

At this point we move away from theoretically-grounded, compute-constrained methods. The later experiments still rely on the transfer heuristics above, since we need learning rates and other hyperparameters for runs with very different token horizons, and on the parameter-scaling result that 1B is the largest scale with reasonably useful VEP monotonicity. But the actual optimization problem becomes much more ad hoc — we start changing mixture constituents, epoch them freely, and see whether in-flight changes can compensate for observed performance gaps.

The first clear gap we try to correct is in upstream performance. Promoter AUPRC from a model trained on all genomic regions lags one trained on upstream sequence alone by a substantial margin, roughly 20% vs. 33% in an earlier run.[^upstream-only-issue] A 1B model trained on a uniform mixture of the same 3-region animal sequences saturates by ~50B tokens on promoters and 5' UTRs, at levels below what upstream-only training can reach. Figure 9 shows why simply shifting weight upstream does not solve this problem. The gains are countered by losses in other genomic regions, and similar continuations from upstream-only or proportionally mixed checkpoints from the parameter-scaling sweep did not produce clear net wins.

![Macro average VEP AUPRC vs upstream mixture proportion](/assets/images/blog/genomic-lm-optimization/figure9_upstream_mix_auprc.svg)

**Figure 9:** Macro average VEP AUPRC vs upstream mixture proportion, against the uniform baseline (dotted). A 40% upstream continuation gives the best net gain in this sweep, but the improvement is small relative to the added mixture complexity.

A more productive strategy is to mix in new sequence types from species with less evolutionary divergence from humans, i.e. mammals rather than all animals. We expand the pool from CDS, upstream, and downstream sequence to a 5-region mixture with ncRNA exons and mostly mammalian enhancer sequence, then return to uniform weighting. The new mixture produced significant gains, improving promoter VEP from roughly 30% to 40%, ncRNA exon variants from 19% to 65%, and enhancer-like distal variants from 14% to 33%, while the other tasks mostly held. The best recipe trains on a uniformly-weighted 3-region mixture for ~104B tokens, then continues on the uniformly-weighted 5-region mixture for ~62B tokens (Figure 10). Importantly, this is a substantial improvement over de novo training on the 5-region mixture and indicates that order of exposure seems to matter. So mid-flight improvement is possible in the end, but in this sweep it comes from adding new, uniformly-weighted mixture components rather than reweighting the old ones.

![VEP AUPRC trajectories by mixture lineage](/assets/images/blog/genomic-lm-optimization/figure10_lineage_vep_trajectory.svg)

**Figure 10:** VEP AUPRC trajectories vs training tokens for three model-mixture lineages. The best model in this post is m5.1, shown in red, which shifts from a 3-region to a 5-region mixture at the dashed line. Curves for m1.3 and m3.3 are truncated at the m5.1 token horizon so the longer runs do not contribute extra evals. The macro average is highlighted in the top-left panel, and the distal and non-coding-exon panels show the clearest inflection after the mixture shift.

[^upstream-only-issue]: See [Open-Athena/marin-dna issue #55](https://github.com/Open-Athena/marin-dna/issues/55).

### Leaderboard scores

The result of the previous mixture experiments is the m5.1 model used for the headline comparison. Figure 11 is a snapshot of the Mendelian VEP leaderboard we host at [openathena.ai/marin-dna/leaderboards/mendelian](https://openathena.ai/marin-dna/leaderboards/mendelian), where we are continuing to add new experimental runs and baselines. In this snapshot, m5.1 is again just a 1B GPT-style model, but it comes out slightly ahead of Evo 2 40B on average across all variant classes.

![Mendelian VEP benchmark AUPRC heatmap across models](/assets/images/blog/genomic-lm-optimization/figure11_leaderboard_heatmap.svg)

**Figure 11:** Mendelian VEP benchmark — AUPRC (%) across models, with the Macro Avg column highlighted. This leaderboard is computed with a newer version of the TraitGym Mendelian eval, so its scores are not directly comparable to those in the earlier figures in the text (e.g. Figures 9 and 10); this is why m5.1's end-of-training score in Figure 10 does not match its current leaderboard score here.

## Conclusion

A fast, high-quality, easy-to-replicate gLM for human variant prioritization, with few restrictions on genomic context, would be a significant contribution to the field. The experiments above show how to check most of those boxes with a standard GPT-style model, though "easy-to-replicate" is still a work in progress. Many less successful attempts are not discussed here and are documented in [Open-Athena/marin-dna](https://github.com/Open-Athena/marin-dna).

There are also still important gaps. The largest technical omission is regularization, an obvious lever for data-constrained modeling. We are in an awkward regime between data-constrained and compute-constrained training, though the narrowed recipe and better infrastructure (TODO: link iris post) for using the Google TPU Research Cloud compute donated for these efforts should make that lever much easier to use. The other major gap is attribution. It is not yet clear exactly where the largest gains are coming from; data curation is almost certainly the biggest contributor, and we plan to explain those details separately.

There is plenty of work left to do, but we think these results clearly show the potential value of a general-purpose training platform like Marin for accelerating scientific foundation model development.
