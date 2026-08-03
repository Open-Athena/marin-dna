---
title: "A 1B standard Transformer rivals Evo 2 40B on variant effect prediction"
slug: "genomic-lm-optimization"
author: "Gonzalo Benegas & Eric Czech"
date: 2026-08-03
published: true
math: false
toc: true
tags:
  - Marin
  - DNA
summary: "How MarinDNA combined data curation, hyperparameter transfer, scaling, and data-mixture experiments to build an efficient 1B alignment-free genomic language model with balanced performance across genomic regions."
---

<style>
.blog-post-content figure img,
.blog-post-content figure svg.figure-svg {
  display: block;
  width: 100%;
  height: auto;
  margin-inline: auto;
  box-sizing: border-box;
  background: #ece3d5;
  padding: 1rem 1.25rem;
  border-radius: 10px;
  box-shadow: 0 1px 4px rgba(31, 30, 27, 0.10);
}
/* Each figure declares the narrowest comfortable media width for its content.
   The figure remains full-width so every caption uses the article measure;
   the SVG normalizer uses the media's inner width for its text hierarchy. */
.blog-post-content figure {
  display: block;
  width: 100%;
  max-width: 100%;
  margin-inline: auto;
}
/* The pinned site renderer emits SVGs as external images. The script below
   safely inlines this article's same-origin SVG assets so their live text can
   inherit Open Athena's loaded Lato webfont. */
.blog-post-content figure img,
.blog-post-content figure .figure-media {
  width: 100%;
  max-width: var(--figure-frame-width, 100%);
  margin-inline: auto;
  overflow-x: auto;
}
.blog-post-content figure svg.figure-svg {
  color: inherit;
  font-family: var(--font-body);
}
.blog-post-content figure svg.figure-svg text,
.blog-post-content figure svg.figure-svg tspan {
  font-family: var(--font-body) !important;
}
/* Fixed-width DNA and code strings are the sole semantic exception. */
.blog-post-content figure svg.figure-svg [font-family*="monospace" i],
.blog-post-content figure svg.figure-svg [font-family*="monospace" i] * {
  font-family: 'SF Mono', 'Menlo', 'Consolas', monospace !important;
}
@media (max-width: 600px) {
  .blog-post-content figure .figure-media-dense svg.figure-svg {
    min-width: var(--figure-frame-width, 740px);
  }
}
/* Nested lists inherit ul's 1rem margin-bottom, which stacks with the parent
   <li>'s margin to leave an oversized gap before the next top-level bullet.
   Zero it so nested bullets sit flush with the following item. */
.blog-post-content li > ul,
.blog-post-content li > ol {
  margin-bottom: 0;
}
</style>

<script>
/* Inline same-origin SVG figures so they share the page's Lato webfont. Prefix
   internal IDs first: independently authored SVGs commonly reuse IDs such as
   "title", "clip1", and "gradient", which would otherwise collide. */
window.addEventListener('DOMContentLoaded', function () {
  var images = document.querySelectorAll('.blog-post-content figure img');
  images.forEach(function (image, figureIndex) {
    var figure = image.closest('figure');
    var figureWidth = Number(figure && figure.dataset.figureWidth);
    if (figure && Number.isFinite(figureWidth)) {
      figure.style.setProperty('--figure-frame-width', figureWidth + 'px');
    }
    var source = new URL(image.getAttribute('src'), document.baseURI);
    if (source.origin !== window.location.origin || !source.pathname.endsWith('.svg')) return;

    fetch(source.href, { cache: 'no-store' })
      .then(function (response) {
        if (!response.ok) throw new Error('SVG request failed: ' + response.status);
        return response.text();
      })
      .then(function (sourceText) {
        var parsed = new DOMParser().parseFromString(sourceText, 'image/svg+xml');
        if (parsed.querySelector('parsererror')) throw new Error('Invalid SVG');
        var svg = parsed.documentElement;
        if (svg.namespaceURI !== 'http://www.w3.org/2000/svg') throw new Error('Not an SVG');

        svg.querySelectorAll('script, foreignObject').forEach(function (element) {
          element.remove();
        });
        Array.from(svg.querySelectorAll('*')).concat([svg]).forEach(function (element) {
          Array.from(element.attributes).forEach(function (attribute) {
            if (attribute.name.toLowerCase().startsWith('on')) element.removeAttribute(attribute.name);
          });
        });

        var prefix = 'blog-figure-' + figureIndex + '-';
        var ids = new Map();
        svg.querySelectorAll('[id]').forEach(function (element) {
          var previous = element.id;
          var next = prefix + previous;
          ids.set(previous, next);
          element.id = next;
        });
        Array.from(svg.querySelectorAll('*')).concat([svg]).forEach(function (element) {
          Array.from(element.attributes).forEach(function (attribute) {
            var value = attribute.value;
            if (attribute.name === 'aria-labelledby' || attribute.name === 'aria-describedby') {
              value = value.split(/\s+/).map(function (id) { return ids.get(id) || id; }).join(' ');
            } else {
              ids.forEach(function (next, previous) {
                value = value.split('url(#' + previous + ')').join('url(#' + next + ')');
                if ((attribute.localName === 'href') && value === '#' + previous) value = '#' + next;
              });
            }
            element.setAttributeNS(attribute.namespaceURI, attribute.name, value);
          });
        });

        svg.classList.add('figure-svg');
        svg.setAttribute('role', 'img');
        svg.setAttribute('aria-label', image.alt || '');
        svg.removeAttribute('aria-labelledby');
        svg.removeAttribute('aria-describedby');

        var frame = document.createElement('div');
        frame.className = 'figure-media';
        if (svg.querySelectorAll('text').length > 8) frame.classList.add('figure-media-dense');
        frame.appendChild(svg);
        image.replaceWith(frame);
      })
      .catch(function () {
        /* Preserve the original image as an accessible fallback. */
      });
  });
});
</script>

MarinDNA applies the tools and open-development approach of [Marin](https://github.com/marin-community/marin) to genomic language modeling.
This post summarizes how data curation, hyperparameter transfer, scaling laws, and data-mixture experiments produced a 1B GPT-style model competitive with [Evo 2 40B](https://doi.org/10.1038/s41586-026-10176-5), while using ~1,980× fewer training FLOPs and scoring variants ~2,330× faster.

<figure id="fig-cost-performance" data-figure-width="242.75">
<img src="/assets/images/blog/genomic-lm-optimization/headline_cost_performance.svg?v=throughput" alt="Zero-shot Mendelian VEP macro-average AUPRC versus variants scored per hour for MarinDNA 1B and Evo 2 1B, 7B, and 40B" />
<figcaption><strong>Figure 1: VEP performance versus inference throughput.</strong>
Zero-shot Mendelian VEP macro-average AUPRC versus variants scored per hour on a GH200.</figcaption>
</figure>

> **A note on open development.**
> This post turns a branching research process into a linear narrative; the [MarinDNA repository](https://github.com/Open-Athena/marin-dna) preserves the underlying experiments, including unsuccessful and inconclusive directions.

## Introduction

Optimization of genomic language models (gLMs) has historically involved a lot of focus on specialized architectures.[^glm-architecture]
We take a different approach: keep the architecture within the standard GPT family and focus on data curation and mixtures, training hyperparameters, and model scale.
The experiments below test how far that simpler, standardized recipe can go.

[^glm-architecture]: For a broader overview, see [“Genomic language models: opportunities and challenges”](https://doi.org/10.1016/j.tig.2024.11.013).
    Examples of specialized architectures include long-convolution or hybrid long-context models such as [HyenaDNA](https://arxiv.org/abs/2306.15794) and [Evo 2](https://doi.org/10.1101/2025.02.18.638918); U-Net-like models such as [NTv3](https://doi.org/10.64898/2025.12.22.695963); bidirectional models such as [DNABERT-2](https://arxiv.org/abs/2306.15006), [GENA-LM](https://www.biorxiv.org/content/10.1101/2023.06.12.544594v3.full.pdf), [PlantCAD2](https://doi.org/10.1101/2025.08.27.672609), and [TrinityDNA](https://arxiv.org/abs/2507.19229); state-space or hybrid state-space models such as [HybriDNA](https://arxiv.org/abs/2502.10807), [Caduceus](https://arxiv.org/abs/2403.03234), and [PlantCAD2](https://doi.org/10.1101/2025.08.27.672609); reverse-complement equivariance in [Caduceus](https://arxiv.org/abs/2403.03234); learned adaptive segmentation that replaces fixed tokenization in [dnaHNet](https://arxiv.org/abs/2602.10603) and [DNACHUNKER](https://arxiv.org/abs/2601.03019); the hierarchical, multiscale Transformer used by [megaDNA](https://www.biorxiv.org/content/10.1101/2023.12.18.572218v3.full); and early or less-established sparse-expert models such as [JanusDNA](https://arxiv.org/abs/2505.17257), [PlantBiMoE](https://arxiv.org/abs/2512.07113), and [MxDNA](https://arxiv.org/abs/2412.13716).

### Why alignment-free gLMs?

Many of the strongest genomic sequence models rely on whole-genome alignments, as in [GPN-Star](https://doi.org/10.1101/2025.09.21.677619),[^omnii-alignments] or functional-genomics measurements, as in [AlphaGenome](https://doi.org/10.1038/s41586-025-10014-0)—resources available for only a small subset of species.
Unlabeled DNA sequence, by contrast, is available for a rapidly growing number of species.
Alignment-free gLMs—also called single-sequence gLMs—can be trained directly on this growing collection of genomes and applied to evolutionary constraint prediction, sequence design, and transfer learning.

[^omnii-alignments]: Exact details are not yet available, but [Omnii](https://www.radicalnumerics.ai/blog/omnii-health-preview) also appears to benefit from alignment-derived information.

For humans and other well-studied species, single-sequence gLMs are still far from replacing alignment-based models or models supervised with functional-genomics data.
Our near-term goal is to build useful models for species that lack high-quality whole-genome alignments and functional-genomics data.
As a concrete example, we aim to enable accurate, cost-effective genome-wide mapping of evolutionary constraint for any mammalian species.[^constraint-map-every-genome]
Longer term, sequence-only models may learn from sequence in ways that complement alignments, conservation scores, and functional-genomics models even in data-rich species ([research question #396](https://github.com/Open-Athena/marin-dna/issues/396)).

[^constraint-map-every-genome]: Zoonomia illustrates the gap between having an alignment and having a ready-to-use constraint track for each mammalian genome.
    Producing these tracks requires running per-base scoring in each target genome's coordinate system, which can require substantial intermediate storage and compute.
    The project's [data page](https://zoonomiaproject.org/the-data/) links to a reference-free 241-way mammalian alignment, but the associated [CGL resource page](https://cglgenomics.ucsc.edu/november-2020-nature-mammalian-and-avian-alignments/) offers a single score download explicitly labeled “Human PhyloP scores.”
    The [expanded 447-way resource](https://cglgenomics.ucsc.edu/november-2023-nature-zoonomia-with-expanded-primates-alignment/) publishes alignment files but no phyloP score downloads; UCSC's standard [447-way phyloP track](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP447way/) is likewise exposed only under human hg38.
    This is part of a broader gap: as of July 2026, UCSC's standard [goldenPath](https://hgdownload.soe.ucsc.edu/goldenPath/) download tree contained phyloP tracks for only 20 species, nine of them mammals.

### Why data curation?

Many early gLMs were trained on the human genome alone, with little to no filtering.
Subsequent work made it clear that two factors were key drivers of model performance: including multiple species, and enriching for functional regions rather than sampling uniformly from the majority-neutral background of mammalian genomes.[^data-curation-evidence]

In this work, we follow up on two findings from [TraitGym](https://pmc.ncbi.nlm.nih.gov/articles/PMC11844472/).
First, 152M-parameter GPN-Promoter, trained only on animal promoters, performed comparably to Evo 2 40B on human promoter variants.
Second, Evo 2 improved substantially with scale overall but still struggled on distal variants.
Enhancers were the only region type not actively curated into its training data and were sparse among the intergenic regions it saw.

MarinDNA therefore treats dataset construction as a primary modeling lever: which species and evolutionary timescales to include ([research question #394](https://github.com/Open-Athena/marin-dna/issues/394)), which functional regions to sample and how to identify them ([research question #395](https://github.com/Open-Athena/marin-dna/issues/395)), how to weigh them, and when during training to introduce them.

[^data-curation-evidence]: See [*Genomic language models: opportunities and challenges*](https://doi.org/10.1016/j.tig.2024.11.013), [GPN](https://doi.org/10.1073/pnas.2311219120), [PlantCaduceus](https://doi.org/10.1073/pnas.2421738122), [GPN-MSA](https://doi.org/10.1038/s41587-024-02511-w), [Species-aware DNA language models](https://doi.org/10.1186/s13059-024-03221-x), [nucleotide-dependency analysis](https://doi.org/10.1038/s41588-025-02347-3), [Evo 2](https://doi.org/10.1038/s41586-026-10176-5), and [ARSENAL](https://doi.org/10.64898/2026.02.05.703637).

We began with annotation-derived datasets for coding, upstream, and downstream sequences.[^training-downstream]
Standard genome annotations make these regions relatively easy to identify and extract consistently across many species.

Later, we added ncRNA[^training-ncrna] and enhancers[^training-enhancer] built by alignment projection.
Because comparable annotations were not directly available across the target species, we projected human annotations through whole-genome alignments.

<figure id="fig-training-datasets" data-figure-width="660">
<img src="/assets/images/blog/genomic-lm-optimization/data_provenance_training_datasets.svg?v=compact-bars" alt="Token counts for annotation-derived CDS, upstream, and downstream datasets and alignment-projected enhancer and ncRNA datasets" />
<figcaption><strong>Figure 2: Training-dataset provenance and scale.</strong></figcaption>
</figure>

[^training-downstream]: “Downstream” denotes the 256 bp immediately downstream of each annotated CDS end, rather than annotated 3′ UTR intervals.
    In [experiment #53](https://github.com/Open-Athena/marin-dna/issues/53), this distance-based definition produced better 3′ UTR VEP performance than the annotation-derived baseline.

[^training-ncrna]: Most species had ncRNA annotations, but in [experiment #43](https://github.com/Open-Athena/marin-dna/issues/43), an annotation-derived ncRNA specialist showed little improvement on ncRNA variants.
    This motivated the later use of human annotations projected through whole-genome alignments.

[^training-enhancer]: “Enhancer” is shorthand for the ENCODE V4 non-promoter cCRE set.
    It includes enhancer-like signatures (dELS and pELS), but also classes such as CA, CA-CTCF, CA-TF, CA-H3K4me3, and TF.

### Why GPT-style architecture?

By GPT-style, we mean the deliberately simple approach of treating DNA as text and training a stock causal, autoregressive, decoder-only language model on it.
In these experiments, that architecture is literally Qwen3 rather than a genomics-specific design.
This approach is not new; a substantial line of prior gLM work has used causal language modeling with GPT- or Llama-like architectures.[^causal-glm-precedent]
What is new here is the quality target.
Even recent models in this family, such as Carbon, generally aim for non-inferiority to smaller Evo 2 checkpoints and still underperform Evo 2 40B on the broad zero-shot VEP setting we care about.[^carbon-eval]
If the quality gap can be closed, GPT-style models have obvious advantages for deployment.
They run through familiar training and inference stacks, move cleanly across hardware, and avoid model-specific kernels or bespoke architecture code, which matters a lot for cost, flexibility, and usability.

MarinDNA's first experiment compared three training objectives on promoter sequences: causal language modeling, masked language modeling, and masked diffusion ([experiment #3](https://github.com/Open-Athena/marin-dna/issues/3)).
Causal language modeling looked most promising in the initial training steps.
This was not a definitive matched-compute comparison of objectives, but it provided enough direction-setting evidence to pursue a simple causal architecture.
That choice also need not permanently constrain the model to left-to-right representations: decoder-only language models can be adapted into bidirectional encoders with further training ([research question #393](https://github.com/Open-Athena/marin-dna/issues/393)).

[^causal-glm-precedent]: GPT-style or otherwise causal genomic models include [GenSLM](https://pmc.ncbi.nlm.nih.gov/articles/PMC9709791/), [DNAGPT](https://arxiv.org/abs/2307.05628), [LOL-EVE](https://doi.org/10.1101/2024.11.11.623015), [METAGENE-1](https://arxiv.org/abs/2501.02045), [GENERATOR](https://arxiv.org/abs/2502.07272), [GENERATOR-v2](https://doi.org/10.64898/2026.01.27.702015), [Gene42](https://arxiv.org/abs/2503.16565), and [Carbon](https://doi.org/10.64898/2026.05.22.727119).
    The closest human-DNA precedents are Carbon, GENERATOR, Gene42, DNAGPT, and the promoter-focused LOL-EVE; several of the others are important causal gLM examples but are less directly relevant to human VEP.

[^carbon-eval]: The [Carbon-3B model card](https://huggingface.co/HuggingFaceBio/Carbon-3B) describes Carbon-3B as a 3B-parameter decoder-only autoregressive genomic model implemented as a stock `LlamaForCausalLM`, with 6-mer DNA tokenization, long-context support, and a two-stage training schedule that switches from a standard cross-entropy objective to a factorized nucleotide supervision loss, bridging its coarse 6-mer tokenization with single-nucleotide resolution.
    Its public zero-shot table compares to Evo 2 7B, not Evo 2 40B: Carbon-3B is slightly ahead on BRCA2 and ClinVar noncoding, but behind on ClinVar coding and TraitGym Mendelian.

Our initial goal was to model individual functional elements of the genome—such as an exon or an enhancer—well.
We therefore deliberately used a short 255-bp context: it was sufficient for the functional-element tasks we tested and made training and evaluation much faster.
Centering each example on an individual functional element also made the training data easier to construct, filter, audit, and interpret.
How best to extend context—either through additional long-context next-token pretraining or during downstream-task fine-tuning—remains future work ([research question #392](https://github.com/Open-Athena/marin-dna/issues/392)).

### Why VEP evaluation?

Variant effect prediction (VEP) is one of the most important applications of gLMs.
A useful VEP model can help scale clinical interpretation for rare disease, hereditary cancer, and variants of uncertain significance.[^vep-clinical]
It can also help connect genetic association signals to disease mechanisms, target selection, and causal-variant prioritization in GWAS fine-mapping.[^vep-therapeutic]
The same kind of evidence is relevant to clinical trial design when genetics can inform patient stratification, enrollment criteria, or mechanism-based cohort definition.
Together, these are commonly used levers for improving the efficiency of pharmaceutical development, and it is uncommon for other gLM evaluations to have such a direct connection to commercially relevant research tasks.
VEP is also one of the few evaluations backed by decades of costly clinical genetics curation, with resources such as ClinVar and OMIM providing a level of human variant evidence that has no real analogue in other species.[^human-variant-curation]
That combination makes it a substantive test of whether a gLM has learned sequence constraints that actually matter for human biology.
If a model has learned useful sequence-level constraints from DNA alone, it should help rank variants in places where direct experimental evidence is weak or nonexistent.

This creates a deliberate mismatch between evaluation and intended use: although we expect single-sequence gLMs to be especially valuable for non-model organisms in the near term, we evaluate them in humans because comparable variant-effect data are not yet available across species.
Human VEP is therefore the most rigorous available test of learned functional constraint, but it does not by itself establish transfer to other organisms; evaluating that transfer will require broader population-genetic or experimental datasets.

In this work, we focus on predicting deleteriousness, pathogenicity, or, more generally, functional constraint.
This task is the one most directly connected to the language-modeling training objective and is therefore easy to evaluate with zero-shot or linear-probing protocols.
We leave the prediction of changes in gene expression—the main application of sequence-to-function models—to follow-up work, as it requires more complex fine-tuning protocols and much larger context sizes.[^sequence-to-function-follow-up]

We use two complementary sources of evidence: clinically curated Mendelian variants[^mendelian-traitgym-differences] and saturation genome-editing (SGE) measurements.
The Mendelian benchmark compares pathogenic and putatively benign variants across broad coding and non-coding consequence types.
The SGE benchmark uses experimentally measured variant effects from a few genes in MaveDB, currently covering missense and splicing variants.

<figure id="fig-evaluation-datasets" data-figure-width="740">
<img src="/assets/images/blog/genomic-lm-optimization/eval_datasets.svg" alt="Clinical Mendelian and experimental SGE benchmarks, including labels and subset counts." />
<figcaption><strong>Figure 3: Variant effect prediction benchmarks.</strong></figcaption>
</figure>

We evaluate each frozen gLM with two readouts: a zero-shot sequence log-likelihood ratio and a linear probe trained on paired reference/alternate embeddings.
The zero-shot score tests whether the model's learned sequence likelihood reflects functional constraint: deleterious alternate alleles should incur larger likelihood penalties relative to the reference allele.
The probe instead asks what variant-relevant information is encoded in the model's learned representation, including information that may not be directly reflected in its sequence likelihoods.

<figure id="fig-evaluation-readouts" data-figure-width="680">
<img src="/assets/images/blog/genomic-lm-optimization/eval_apparatus.svg?v=standard-snowflake" alt="Reference and alternate sequences scored using likelihoods or frozen-model embeddings." />
<figcaption><strong>Figure 4: Variant effect prediction scoring approaches.</strong></figcaption>
</figure>

[^vep-clinical]: Examples include zero-shot or disease-focused variant interpretation results in [Evo 2](https://doi.org/10.1101/2025.02.18.638918), [GPN-Star](https://doi.org/10.1101/2025.09.21.677619), [Carbon](https://doi.org/10.64898/2026.05.22.727119), and [EnTao-GPM](https://arxiv.org/abs/2507.21706).

[^vep-therapeutic]: Examples include fine-mapped GWAS and broader human-genetics results in [GPN-Star](https://doi.org/10.1101/2025.09.21.677619), regulatory variant-effect prediction in [AlphaGenome](https://doi.org/10.1101/2025.06.25.661532) and [ChromBPNet](https://www.biorxiv.org/content/10.1101/2024.12.25.630221v2), and the broader observation that human genetic evidence can support target-disease hypotheses in drug discovery in [Nelson et al.](https://doi.org/10.1038/ng.3314).

[^human-variant-curation]: [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/intro/) archives submitted reports relating human genomic variation to disease, cancer, drug response, and supporting evidence; [OMIM](https://doi.org/10.1093/nar/gku1205) is a curated catalog of human genes, genetic disorders, and gene-phenotype relationships.
    Nothing comparable exists for any other species: this depth reflects decades of clinical genetics effort directed specifically at human disease, an investment that has simply not been made for non-human genomes.

[^sequence-to-function-follow-up]: One promising intermediate sequence-to-function target is chromatin accessibility.
    [ARSENAL](https://doi.org/10.64898/2026.02.05.703637) showed that embeddings from a short-context regulatory gLM improved supervised chromatin-accessibility prediction over strong ab initio baselines across multiple cell types, while also improving regulatory-variant scoring.

[^mendelian-traitgym-differences]: Our Mendelian benchmark is inspired by the published [TraitGym](https://doi.org/10.1101/2025.02.11.637758) benchmark.
    Relative to TraitGym, we broadened the gnomAD control set by lowering the minimum allele frequency from 5% to 0.1%; we refer to variants above this threshold as non-rare.
    The larger control pool allowed us to match potential confounders within each consequence class—including TSS distance and, for splicing variants, exon distance—so that these features are largely non-predictive of the label.
    We also expanded the benchmark to include missense and splicing variants, incorporated additional sources of pathogenic variants, and created chromosome-disjoint splits for development and final testing.
    See the [pinned dataset card](https://huggingface.co/datasets/bolinas-dna/evals_mendelian_traits/tree/4aed58e50c5dea0b878a665007af2ef9e5108e9f) for the full construction and matching diagnostics.

### Why compare with Evo 2 40B?

Evo 2 40B remains the strongest broadly evaluated baseline among single-sequence DNA models.
Despite substantial subsequent work on gLMs, we are not aware of a model in this setting that has clearly surpassed it across diverse genomic regions; many newer models do not attempt a direct comparison with Evo 2 40B.
The other reason Evo 2 40B matters is its training budget.
Its reported 2.25e24 training FLOPs are unrivaled among gLMs, corresponding to roughly $2.5M of H100 time.[^evo2-cost]
That budget is unusual in biology and comparable to major open-weight LLM training runs from recent model generations,[^evo2-llm-compute] e.g. just above Qwen2.5-14B and below Qwen2.5-32B, and roughly between DeepSeek-V2 and DeepSeek-V3.
Evo 2 40B is therefore the right baseline for asking whether a much simpler single-sequence gLM can be competitive.

[^evo2-cost]: This estimate uses the [Evo 2](https://doi.org/10.1101/2025.02.18.638918) reported training compute of 2.25e24 FLOPs, 50% H100 model FLOP utilization following the costing convention in [Beyond Chinchilla](https://arxiv.org/abs/2401.00448), 989 TFLOP/s BF16 peak throughput for an H100 SXM, and $2 per H100-hour from [OLMo 3](https://arxiv.org/abs/2512.13961).
    The resulting accelerator requirement is about 1.26M H100-hours.

[^evo2-llm-compute]: Other over/under examples give the same intuition.
    The [AI2 OLMo 2 32B model card](https://huggingface.co/allenai/OLMo-2-0325-32B) places Evo 2 40B above Gemma 2 27B, OLMo 2 32B, and Llama 3.1 8B.
    A dense-accounting estimate from the [Llama 3.1 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) places it well below Llama 3.1 70B.

## Results

### Early mixture experiments

We first trained a 1.7B upstream-region specialist, trying to replicate the success of GPN-Promoter ([experiment #21](https://github.com/Open-Athena/marin-dna/issues/21)).[^early-mixture-datasets]
Although we used reasonable defaults rather than the systematic hyperparameter-transfer recipe developed later, performance was broadly comparable to Evo 2 40B.
We saw a similar pattern when training a CDS specialist ([experiment #27](https://github.com/Open-Athena/marin-dna/issues/27)).
Overall, however, GPN-Star remained stronger.

[^early-mixture-datasets]: The upstream and CDS datasets used in these early experiments were earlier versions of those summarized in [Fig. 2](#fig-training-datasets): broadly comparable, but not identical.
    These models also used a 512-bp context without a BOS token, roughly twice the 255-bp context adopted for the later recipe.

<!-- Plot recipe: plots/blog/promoter_cds_specialists.py -->
<figure id="fig-upstream-cds-specialists" data-figure-width="477.357">
<img src="/assets/images/blog/genomic-lm-optimization/promoter_cds_specialists.svg" alt="Five independently scaled panels comparing upstream and CDS specialists with Evo 2 40B and GPN-Star on region-matched Mendelian variant classes" />
<figcaption><strong>Figure 5: Region-matched zero-shot Mendelian VEP: specialists versus generalists.</strong>
Error bars denote SE.</figcaption>
</figure>

After testing upstream and CDS specialists independently, the next experiment asked whether one model could retain both capabilities ([experiment #13](https://github.com/Open-Athena/marin-dna/issues/13)).
Sampling in proportion to dataset size—10% upstream and 90% CDS—is the naive default when the two datasets are simply pooled without reweighting.
In practice, it behaved similarly to CDS-only training.
Equal 50/50 upstream/CDS sampling produced balanced performance across both regions.
This made explicit mixture control a central axis of investigation.
Even the 50/50 mixture may not be optimal: regions can differ both in size and in the density of learnable biological signal.

<!-- Plot recipe: plots/upstream_cds_balance.py -->
<figure id="fig-upstream-cds-balance" data-figure-width="535.228">
<img src="/assets/images/blog/genomic-lm-optimization/upstream_cds_balance.svg" alt="Promoter and missense VEP AUPRC (%) trajectories for upstream-only, balanced, proportional, and CDS-only training mixtures" />
<figcaption><strong>Figure 6: Upstream/CDS training-mixture comparison.</strong>
Zero-shot results; the right panel is the unweighted mean of the promoter and missense AUPRC values.</figcaption>
</figure>

### Hyperparameter transfer

Our first attempt to scale manually lowered the learning rate as the model grew from 0.6B to 1.7B and 4B parameters ([experiment #57](https://github.com/Open-Athena/marin-dna/issues/57)).
The larger models did not improve over the 0.6B model, and the early 4B run became unstable.
That failure made systematic hyperparameter transfer a prerequisite: without a trustworthy optimization recipe, a model-size comparison would confound scale with tuning quality.

The annotation-derived DNA pool available at the time contained ~85B tokens.
[Fig. 7](#fig-annotation-derived-training-pool) shows its proportional CDS, upstream, and downstream composition.

<figure id="fig-annotation-derived-training-pool" data-figure-width="680">
<img src="/assets/images/blog/genomic-lm-optimization/annotation_derived_training_pool.svg" alt="Approximately 85 billion annotation-derived DNA tokens in a proportional CDS, upstream, and downstream mixture" />
<figcaption><strong>Figure 7: Available annotation-derived training pool.</strong></figcaption>
</figure>

That pool is large by genomics standards but small relative to modern accelerator-era training corpora.
This makes the project data-constrained in principle even though our practical constraints are messier.
We train on preemptible Google TPU Research Cloud resources, do not have consistent access to slices much larger than roughly 32 H100s worth of peak FLOPs, and want the recipe to remain reproducible at academic compute scale.
O(100B) tokens therefore lands in an awkward middle ground where compute-constrained methods are still relevant, even though modest epoching is possible and likely breaks their assumptions at some unknown rate.

We started with hyperparameter transfer for that reason.
If a proven data-constrained transfer framework existed, we would use it.
We do not know of one, so we followed the same basic pattern as [Delphi](https://openathena.ai/blog/delphi/), fitting a small reference sweep with the Vizier Bayesian optimization framework and then scaling the result using a Complete(d)-inspired AdamH heuristic.[^completed-framework]

<figure id="fig-hyperparameter-transfer-methodology" data-figure-width="700">
<img src="/assets/images/blog/genomic-lm-optimization/parameter_transfer_methodology_v1.svg" alt="Reference hyperparameter tuning and target hyperparameter transfer" />
<figcaption><strong>Figure 8: Hyperparameter calibration and transfer.</strong></figcaption>
</figure>

[Fig. 8](#fig-hyperparameter-transfer-methodology) separates reference calibration from target application.
Two heuristics sit behind that workflow and are fixed before tuning: an inherited rule maps hidden width D to model geometry,[^architecture-geometry] and a second rule maps reference optimizer settings to a new batch size B and token horizon T.
The reference sweep tunes initialization scale, the two learning rates, β₁, β₂, ε, gradient clipping, and z-loss.
Here, tuning a learning rate means tuning its peak value under a fixed fractional schedule; the target run reuses that schedule shape, so warmup and decay scale with the run length rather than keeping fixed absolute step counts.[^learning-rate-schedule]
At the target, the two learning rates, β₂, and ε are transformed with B and T, while initialization scale, β₁, gradient clipping, and z-loss are reused unchanged.[^optimizer-transfer-rule]

The reference sweep used ~25M-parameter models trained for 2.5B tokens with a 16k-token batch, or roughly 4e17 FLOPs per run.
We then validated the transferred hyperparameters across 255M–1B-parameter models, with 4x as many tokens, 1/4x the batch size, and roughly 170x the FLOPs per run.
The first test was whether the learning-rate prediction survived that regime.
[Fig. 9](#fig-learning-rate-transfer) shows that the transferred prediction lands exactly on the best observed learning-rate setting at all three validation scales, outperforming both the unchanged reference optimum and every other target-scale sweep setting; the less sensitive optimizer hyperparameters are shown separately in [Fig. 10](#fig-adam-transfer).

<!-- Plot recipe: plots/blog/genomic_lm_optimization/src/figures/figure1_lr_transfer.py -->
<figure id="fig-learning-rate-transfer" data-figure-width="410.702">
<img src="/assets/images/blog/genomic-lm-optimization/figure1_lr_transfer.svg" alt="Learning-rate transfer across model scales" />
<figcaption><strong>Figure 9: Learning-rate transfer across model scales.</strong>
Results across the 255M, 476M, and 1B validation scales.
The <code>control</code> run type indicates final loss from the optimal configuration found in the initial smaller-scale reference sweep.
The predicted, optimal LR results in a better loss than both this control and all other configurations at the same scale (<code>sweep</code> run type), for all model sizes.</figcaption>
</figure>

<!-- Plot recipe: plots/blog/genomic_lm_optimization/src/figures/figure2_beta2_epsilon_transfer.py -->
<figure id="fig-adam-transfer" data-figure-width="476.974">
<img src="/assets/images/blog/genomic-lm-optimization/figure2_beta2_epsilon_transfer.svg" alt="Adam beta2 and epsilon transfer across model scales" />
<figcaption><strong>Figure 10: Adam β₂ and ε transfer.</strong>
Results across the same scales as <a href="#fig-learning-rate-transfer">Fig. 9</a>.</figcaption>
</figure>

That validation is a fairly unforgiving test.
If the transferred learning rate were merely close by accident, it would be surprising for it to land correctly across all three validation scales, but the prediction remains well centered at each one.
For DNA, that is a pretty cool result.
Prior biology foundation-model work has used μP-style transfer, but we are not aware of a DNA result showing that a more inclusive framework like Complete(d) works across token horizon and batch size, which are the axes we keep leaning on later in ad hoc runs across epochs.
The same is mostly true for the other optimizer hyperparameters too, although Adam β₂ shows some signs of being a bit aggressive at the largest scale.
[Fig. 11](#fig-region-hyperparameter-transfer) makes the same point across CDS, upstream, and downstream sequence, with no qualitative difference in transfer behavior across region types.
That gives us enough confidence that the following parameter-scaling runs are at least close to optimally configured.

<details>
<summary>Fig. 11: Transfer validation by region</summary>

<!-- Plot recipe: plots/blog/genomic_lm_optimization/src/figures/figure3_region_hyper_transfer.py -->
<figure id="fig-region-hyperparameter-transfer" data-figure-width="662.7">
<img src="/assets/images/blog/genomic-lm-optimization/figure3_region_hyper_transfer.svg" alt="Hyperparameter transfer validated per genomic region" />
<figcaption><strong>Figure 11: Region-specific hyperparameter transfer.</strong>
Transfer validated separately for CDS, upstream, and downstream regions.</figcaption>
</figure>

</details>

[^completed-framework]: Complete(d) refers to the compute-constrained hyperparameter-transfer framework described in [Complete(d): Data-Optimizing Hyperparameter Transfer](https://arxiv.org/abs/2512.22382).

[^optimizer-transfer-rule]: Relative to the reference batch and token horizon (B₀, T₀), the fixed heuristic uses `AdamH LR ∝ √(B/B₀)·(T₀/T)^0.3`, `Adam LR ∝ √(r/r₀)`, `ε ∝ √(r₀/r)`, and `β₂ = clip(β₂,₀^(B/B₀))`, where `r/r₀ = (B·T₀)/(B₀·T)` and configured bounds still apply.
    The 0.3 exponent is inherited from Marin’s text recipe rather than fitted on the DNA sweep; the Complete(d) paper proposes a 0.5 token-horizon exponent, while a later [Marin text sweep](https://github.com/marin-community/marin/issues/4225) estimated ~0.28.
    See the [commit-pinned implementation](https://github.com/marin-community/marin/blob/a638849fa837f924aaac66ff3d0c1f581dfdd49e/experiments/scaling_law_sweeps/completed_adamh.py#L162-L209).

[^learning-rate-schedule]: Both learning rates use 10% linear warmup, remain at their peak through 80% of training, and then decay linearly to zero over the final 20%.
    These are fractions of the target run's total steps.
    See the [experiment configuration](https://github.com/marin-community/marin/blob/a638849fa837f924aaac66ff3d0c1f581dfdd49e/experiments/dna/exp109_bolinas_scaling_sweep.py#L269-L290) and [scheduler implementation](https://github.com/marin-community/marin/blob/a638849fa837f924aaac66ff3d0c1f581dfdd49e/lib/levanter/src/levanter/optim/config.py#L283-L376).

[^architecture-geometry]: The DNA experiment inherits its geometry rule from Marin’s text-model scaling heuristic rather than fitting it on DNA.
    For hidden width D, it sets `layers = round(D/(55 + 4·log₂D))`, `MLP width = 4D`, and both attention-head and KV-head counts to `D/128`; the sweep widths are selected manually.
    See the [commit-pinned geometry rule](https://github.com/marin-community/marin/blob/a638849fa837f924aaac66ff3d0c1f581dfdd49e/experiments/scaling_law_sweeps/completed_adamh.py#L133-L140) and [model builder](https://github.com/marin-community/marin/blob/a638849fa837f924aaac66ff3d0c1f581dfdd49e/experiments/scaling_law_sweeps/completed_adamh.py#L211-L244).

### Parameter scaling

Before asking whether better validation loss[^validation-loss] translates into better VEP performance, we first needed to check whether validation loss scaled the way it should.
The parameter sweep uses the same training recipe at each model size, with all hyperparameters set by the transfer heuristic above, and then asks whether the resulting losses fit a Kaplan-style scaling law well (they do).[^kaplan-scaling]
Despite this being a simple experiment conceptually, actually getting there took months — fitting the hyperparameter transfer heuristic, running the validation experiments, and training the 4B model, which alone took about three weeks to finish.
The final sweep spans 8 model sizes from 46M to 4B parameters, each trained on ~84B tokens, for ~4.3e21 FLOPs across the sweep.
That puts it on par with canonical scaling-law studies in language modeling, e.g. its ~2.1e21 FLOP 4B run matches the compute Hugging Face used at that exact model scale in their data-constrained scaling work.[^muennighoff]

<!-- Plot recipe: plots/blog/genomic_lm_optimization/src/figures/figure4_loss_scaling.py -->
<figure id="fig-loss-scaling" data-figure-width="693.082">
<img src="/assets/images/blog/genomic-lm-optimization/figure4_loss_scaling.svg" alt="Loss scaling across model sizes with Kaplan power-law fits" />
<figcaption><strong>Figure 12: Loss scaling across model sizes.</strong>
Eight models from 46M to 4B parameters, with a Kaplan power-law fit.</figcaption>
</figure>

The result is about as tidy as we could hope for.
Training is stable at every scale, and both training and validation loss decrease monotonically and predictably, as shown in [Fig. 12](#fig-loss-scaling).
We use WSD learning-rate schedules with 10% warmup and 20% decay, which causes the visible drop in both losses over the final 20% of tokens.
Most importantly, the sweep gives a high-quality Kaplan scaling-law fit (R<sup>2</sup>=0.999), which makes the next question much better posed.
Does lower validation loss actually correlate with better downstream VEP performance?

[^validation-loss]: Here, “validation loss” is best understood as a training-loss-like monitoring statistic computed on a fixed set of human training sequences, rather than a conventional estimate on held-out data.
    We have not yet found a satisfactory way to construct clean held-out genomic splits: genomes are phylogenetically correlated, and identifying orthologous non-coding regions by sequence alignment is difficult.
    See [issue #8](https://github.com/Open-Athena/marin-dna/issues/8) for split experiments and a broader discussion of why raw perplexity may not reliably track VEP performance.
    It also contains an incidental objective mismatch: during training, lowercase marks repetitive bases, which receive 1% of the standard loss weight; in these validation sets, lowercase instead marks non-conserved bases, so those positions also received 1% weight.
    That validation weighting was inherited from the training setup rather than chosen deliberately.
    We therefore use validation loss descriptively for like-for-like comparisons and trend analysis, not as an unbiased estimate of performance on unseen sequence; the biological conclusions rely primarily on downstream VEP evaluation.

[^kaplan-scaling]: This follows the empirical scaling-law setup from [Kaplan et al.](https://arxiv.org/abs/2001.08361), where model loss is fit as a predictable function of model size, data, and compute.

[^muennighoff]: See Fig. 4 of [Muennighoff et al.](https://arxiv.org/abs/2305.16264), "Scaling Data-Constrained Language Models" (NeurIPS 2023).

### Downstream performance

The final sweep shows a mostly consistent relationship between parameter count and downstream VEP performance.
When zero-shot LLR and linear probes are evaluated on identical variants, performance improves with scale for most variant types.
The clearest exception is Mendelian missense: zero-shot LLR peaks at 128M parameters and then deteriorates, even as linear-probe performance continues to improve.[^zero-shot-scaling]
This is not a general failure on missense variants—the SGE missense benchmark improves with scale under both scoring protocols.

<!-- Plot recipe: plots/blog/genomic_lm_optimization/src/figures/figure5_params_vs_vep_auprc.py -->
<figure id="fig-parameters-vs-vep" data-figure-width="554.969">
<img src="/assets/images/blog/genomic-lm-optimization/figure5_params_vs_vep_auprc.svg?v=auprc-percent" alt="VEP performance across model parameters for Mendelian and SGE consequences, comparing zero-shot LLR and linear probes" />
<figcaption><strong>Figure 13: VEP performance across model scale.</strong>
Zero-shot LLR and a linear probe are compared on identical variants.
Performance is measured as chromosome-weighted AUPRC; facet y-scales vary independently, and error bars denote ±1 chromosome-cluster bootstrap SE.</figcaption>
</figure>

Plotting the same results against matched-region validation log-likelihood gives the same picture.
Better validation log-likelihood is associated with better downstream performance across the other variant types and scoring protocols.
Zero-shot Mendelian missense again points in the opposite direction: performance declines even as validation log-likelihood improves.

<!-- Plot recipe: plots/blog/genomic_lm_optimization/src/figures/figure6_loss_vs_vep_auprc.py -->
<figure id="fig-loss-vs-vep" data-figure-width="562.454">
<img src="/assets/images/blog/genomic-lm-optimization/figure6_loss_vs_vep_auprc.svg?v=layout-20260803c" alt="VEP performance versus matched-region validation log-likelihood for Mendelian and SGE consequences, comparing zero-shot LLR and linear probes" />
<figcaption><strong>Figure 14: VEP performance versus validation log-likelihood.</strong>
Matched-region LL (shown as −loss) across the eight parameter-scaling endpoints.
Performance is measured as chromosome-weighted AUPRC.
Lines are least-squares fits, and <em>r</em> denotes Pearson correlation; facet axes vary independently, and error bars denote ±1 chromosome-cluster bootstrap SE.</figcaption>
</figure>

This divergence is not unique to MarinDNA.
On the same Mendelian missense benchmark, Evo 2 also shows improving linear-probe performance alongside declining zero-shot LLR performance as model size increases.
For now, this should be interpreted as a recurring pattern for Mendelian missense across these two model families—not as evidence that zero-shot readouts generally deteriorate with scale.
It also cautions against using zero-shot LLR alone to judge whether scaling has improved the learned representations for this task.

<!-- Plot recipe: plots/blog/genomic_lm_optimization/src/figures/figure6b_marin_evo2_missense.py -->
<figure id="fig-missense-readout-scaling" data-figure-width="417.506">
<img src="/assets/images/blog/genomic-lm-optimization/figure6b_marin_evo2_missense.svg?v=auprc-percent" alt="Missense VEP performance across MarinDNA and Evo 2 model scales, comparing zero-shot LLR and linear probes" />
<figcaption><strong>Figure 15: Missense readouts across model scale.</strong>
MarinDNA and Evo 2 are compared using zero-shot LLR and a linear probe on identical Mendelian variants.
Performance is measured as chromosome-weighted AUPRC; error bars denote ±1 chromosome-cluster bootstrap SE.</figcaption>
</figure>

[^zero-shot-scaling]: Non-monotonic likelihood-based zero-shot variant-effect performance with increasing model scale has previously been observed in other settings.
    [Gordon et al.](https://proceedings.iclr.cc/paper_files/paper/2025/hash/62cf81a87f367758cebabce08e8d40d8-Abstract-Conference.html) report that ESM-2 performance on protein deep-mutational-scanning benchmarks degrades beyond an intermediate model size and show that performance depends on the likelihood assigned to the wild-type sequence.
    [Pugh et al.](https://proceedings.neurips.cc/paper_files/paper/2025/hash/bdb30687f1c2255c29b11b0b45204ebe-Abstract-Conference.html) similarly report plateaus or regressions for larger protein language models under standard likelihood-based scoring.
    These studies concern protein models and different evaluation regimes; we do not know whether our Mendelian missense result has the same cause.

### Later mixture experiments

At this point we move away from theoretically grounded, compute-constrained methods.
The later experiments still rely on the transfer heuristics above, since we need learning rates and other hyperparameters for runs with very different token horizons.
But the actual optimization problem becomes much more ad hoc — we start changing mixture constituents, epoch them freely, and see whether in-flight changes can compensate for observed performance gaps.

We standardized these experiments on a 1B-parameter model.[^later-mixture-model-size]

Our starting point was a uniform mixture of CDS, upstream, and downstream sequence.
These were the three region datasets we could initially construct consistently across many species from standard genome annotations.
The scaling study had instead sampled these regions in proportion to dataset size.
Echoing the earlier mixture experiments, that recipe produced good CDS performance but left a large gap on the less abundant upstream tasks.
We therefore switched to uniform weighting so that each functional region received meaningful exposure.

By then, we had already trained m5.1 for ~104B tokens on the uniform three-region mixture.
Alignment projection then made it possible to turn human ncRNA and enhancer annotations into comparable multi-species training datasets.
Once those data became available, we added them to form a uniform five-region mixture and continued training the same model for another ~62B tokens.
We compare this staged history with two lineages trained on five-region mixtures from the beginning.

[^later-mixture-model-size]: At the time, 1B had reached a good level of zero-shot performance under our then-current evaluation.
    We had not established it as the optimal model size.
    The later linear-probe results change this judgment most: they continued to improve with scale even where zero-shot Mendelian missense did not, making a larger model a more compelling choice in retrospect.

<figure id="fig-five-region-lineage" data-figure-width="740">
<img src="/assets/images/blog/genomic-lm-optimization/continued_training_data_exposures.svg" alt="Training-data exposure histories for m5.1, m1.3, and m3.3 through a shared 166-billion-token horizon" />
<figcaption><strong>Figure 16: Training-data exposure histories.</strong>
Three recipes are compared.
m5.1 trains for approximately 104B tokens on a uniform three-region mixture before adding ncRNA and enhancer data for approximately 62B tokens.
The de novo m1.3 and m3.3 controls keep fixed five-region mixtures over the same displayed token horizon; m3.3 gives upstream sequence 25% weight and each other region 18.75%.</figcaption>
</figure>

For m5.1, the first evaluated checkpoint after adding ncRNA and enhancer data shows gains in variant-effect performance in the corresponding ncRNA and distal subsets.
Across all eight subsets, m5.1 ultimately finishes with the highest macro-average AUPRC under both zero-shot LLR and the linear probe, although the linear-probe trajectories are visibly noisier.[^mixture-probe-noise]
Its advantage is broad but not universal: the lineages trained on five regions from the beginning retain stronger endpoints for some distal and ncRNA subsets.
m5.1's strong endpoint raises the possibility that exposure order matters: learning first from the three-region mixture and introducing ncRNA and enhancer data later may be more effective than training on all five regions from the beginning, though this remains uncertain and requires further investigation.

[^mixture-probe-noise]: The curves shown here use a separate probe trained within each variant subset.
    In a [separate 255M analysis on one held-out chromosome](https://github.com/Open-Athena/marin-dna/issues/369#issuecomment-4936655473), training one probe across all subsets improved the AUPRC point estimate on several data-starved subsets, including ncRNA and distal, while hurting stronger or more specialized subsets.
    This makes limited labeled data one plausible contributor to the noise, but that analysis did not directly test the checkpoint-to-checkpoint variability in these 1B lineage curves.

<!-- Plot recipe: plots/blog/genomic_lm_optimization/src/figures/figure16_offline_lineage_prototype.py -->
<figure id="fig-mixture-lineage-trajectories" data-figure-width="567.68">
<img src="/assets/images/blog/genomic-lm-optimization/figure16_offline_lineage_llr_prototype.svg?v=offline-nine-panel-v6" alt="Nine-panel zero-shot Mendelian pooled AUPRC (%) trajectories with error bars along each mixture lineage" />
<figcaption><strong>Figure 17: Zero-shot mixture-lineage trajectories.</strong>
Mendelian AUPRC versus training tokens for three model-mixture lineages.
Within each subset, this is global (pooled) AUPRC across variants; the macro panel is the unweighted mean across subsets.
Error bars show ±1 SE from a matched-group cluster bootstrap.</figcaption>
</figure>

<details>
<summary>Show the linear-probe view</summary>
<figure id="fig-mixture-lineage-probe" data-figure-width="567.68">
<img src="/assets/images/blog/genomic-lm-optimization/figure16_offline_lineage_probe_prototype.svg?v=offline-nine-panel-v6" alt="Nine-panel linear-probe Mendelian chromosome-weighted AUPRC (%) trajectories with error bars along each mixture lineage" />
<figcaption><strong>Figure 18: Linear-probe mixture-lineage trajectories.</strong>
Linear-probe Mendelian AUPRC versus training tokens for three model-mixture lineages.
Within each subset, this is the sample-size-weighted mean of per-chromosome AUPRCs; the macro panel is the unweighted mean across subsets.
Error bars show ±1 SE from a chromosome-cluster bootstrap.</figcaption>
</figure>
</details>

### Leaderboard scores

The result of the previous mixture experiments is m5.1, a 1B GPT-style model evaluated alongside other models on our [live Mendelian VEP leaderboard](https://openathena.ai/marin-dna/leaderboards/mendelian), where we continue to add experimental runs and baselines.
In the zero-shot snapshot shown here, m5.1 comes out slightly ahead of Evo 2 40B.
Its advantage is considerably larger under linear probing.

m5.1 was trained on 166B tokens (~1.1e21 FLOPs), compared with 9.3T tokens (~2.25e24 FLOPs) for Evo 2 40B.
At their native context lengths on the same GH200, m5.1 scores one million variants in about 41 minutes, compared with roughly 66 days for Evo 2 40B—a roughly 2,330× throughput advantage.[^inference-throughput]

[^inference-throughput]: This benchmark measures steady-state scoring with forward and reverse-complement passes and embeddings enabled.
    m5.1 uses a 256-token context, while Evo 2 40B uses an 8,192-token context, so this measures as-deployed throughput rather than same-context or per-token efficiency.
    See [issue #354](https://github.com/Open-Athena/marin-dna/issues/354) for the full methodology and results.

Most notably, m5.1 closes Evo 2's main gap on distal variants, outperforming Evo 2 40B there under both readouts.
The improvement is not uniform, however: Evo 2 40B remains ahead on splicing under both readouts, as well as on the Promoter subset and synonymous variants in the zero-shot evaluation and missense variants under linear probing.

The [current leaderboard](https://openathena.ai/marin-dna/leaderboards/mendelian) also suggests there is considerable headroom: although m5.1 leads on Macro Avg among MarinDNA models, another MarinDNA run has a higher point estimate in seven of the eight displayed subsets, with m5.1 leading only on ncRNA.
Several of these winners are region specialists, suggesting that further mixture refinement could recover more of their complementary strengths.

On the broader zero-shot leaderboard, m5.1 remains slightly behind AlphaGenome and substantially behind GPN-Star.
These are different model families: AlphaGenome learns from functional-genomics supervision, while GPN-Star uses whole-genome alignments.
Further improvements to the alignment-free recipe may narrow the gap to GPN-Star, but it is not clear that they will eliminate it.
Some of the remaining difference may reflect information that an alignment-free model cannot recover from unaligned sequence alone ([research question #397](https://github.com/Open-Athena/marin-dna/issues/397)).

<!-- Plot recipe: plots/blog/figure11_leaderboard_heatmap.py -->
<figure id="fig-mendelian-leaderboard" data-figure-width="668.822">
<img src="/assets/images/blog/genomic-lm-optimization/figure11_leaderboard_heatmap__mendelian_llr.svg" alt="Mendelian VEP benchmark zero-shot AUPRC (%) heatmap across six headline models" />
<figcaption><strong>Figure 19: Zero-shot Mendelian leaderboard.</strong>
AUPRC across six headline models using each model family's canonical scoring protocol, with the Macro Avg column highlighted.</figcaption>
</figure>

<figure id="fig-mendelian-leaderboard-probe" data-figure-width="668.67">
<img src="/assets/images/blog/genomic-lm-optimization/figure11_leaderboard_heatmap__mendelian_probe.svg" alt="Mendelian VEP benchmark linear-probe AUPRC (%) heatmap across four overlapping models" />
<figcaption><strong>Figure 20: Linear-probe leaderboard.</strong>
AUPRC for the four models that also appear in the zero-shot leaderboard and have compatible probe metrics.
The two heatmaps are sorted and color-normalized independently, and their subset metrics use different aggregation: <a href="#fig-mendelian-leaderboard">Fig. 19</a> pools variants within each subset before computing AUPRC, whereas <a href="#fig-mendelian-leaderboard-probe">Fig. 20</a> computes a sample-size-weighted mean of per-chromosome AUPRCs; both Macro Avg columns are unweighted means across subsets.
Compare model names and within-panel values rather than row positions, colors, or absolute values across panels.
GPN-Star and AlphaGenome are absent because no compatible probe result is available here, not because of their performance.</figcaption>
</figure>

## Conclusion

These experiments show how far a simple and standardized genomic modeling recipe can go.
By keeping the architecture within the GPT family and iterating on data curation, training hyperparameters, model scale, and data mixtures, MarinDNA produced an alignment-free 1B model that is competitive with Evo 2 40B on Mendelian variant effect prediction while requiring far less training compute and offering much faster inference.
In terms of capabilities, we next want to improve our performance on complex-trait variant effect prediction ([research question #391](https://github.com/Open-Athena/marin-dna/issues/391)).

On the training side, neither scaling nor optimization appears exhausted.
Performance continued to improve through the largest model we tested, with no clear sign of saturation, making further scaling a natural next step.
Regularization also remains a largely unexplored part of the training recipe and may be especially valuable in the awkward regime between data- and compute-constrained training.
The narrowed recipe and [better infrastructure](https://openathena.ai/blog/cluster-scheduling-with-iris/) for using the Google TPU Research Cloud compute donated for this work should make both larger-scale runs and more systematic optimization easier to pursue.

There is plenty of work left to do, but we think these results clearly show the potential value of a general-purpose training platform like Marin for accelerating scientific foundation model development.

## Resources

- [🧪 MarinDNA repository](https://github.com/Open-Athena/marin-dna) (source code and experiments tracked through GitHub issues)
- [🤗 Hugging Face collection](https://huggingface.co/collections/marin-dna/marindna-genomic-language-model-optimization-blog-2026-07) (training datasets, benchmarks, and model)
- [🏆 MarinDNA leaderboard](https://openathena.ai/marin-dna/)
- [🧬 Interactive sequence explorer](https://molab.marimo.io/notebooks/nb_MrPpr5xYcN3HGt5tLY86bk/app)

## Acknowledgements

We thank Isaac Hodes, Yael Elmatad, David Hall, Will Held, and Jeff Hammerbacher, as well as others across the Open Athena and Marin communities, for thoughtful discussions.
We also thank the Google TPU Research Cloud (TRC) program for providing compute resources.
