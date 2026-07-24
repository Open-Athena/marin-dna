Title: Genomic Language Model Optimization
Description: How Marin can be used to train single-sequence, vanilla Transformer gLMs comparable to Evo2 40B with 1,980x fewer FLOPS. Covers hyperparameter transfer, scaling law and training mixture experiments.

## Introduction

- Optimizating of genomic language models (gLMs) historically involves a lot of focus on model architecture
  - Examples:
    - Long convolutions [hyenadna, evo2], U-Nets [ntv3], bi-directionality [dnabert2, genalm, caduceus, plantcad2, trinitydna], state-space layers [hybridna, caduceus, plantcad2], less proven adademic MoE models [janusdna, plantbimoe, mxdna]
    - Tokenizer-free methods [dnahnet, dnachunker], multi-scale Transformers [megadna], multi-scale attention [trinitydna]
    - reverse-complement equivariance [caduceus], double-helix groove fusion [trinitydna], genomic loss weighting [evo2, gpn] and factorization schemes [generatorv2, carbon]
    - Other notable architectural improvements outside of single-sequence DNA gLMs include the convolutional U-Net Transformer + pairwise contact map model of AlphaGenome [alphagenome] and sequence alignment + phylogeny-aware attention of GPN-Star [gpnstar]
  - Most of these innovations or adaptations, often from text and vision domains, are geared towards increasing efficiency of long context learning and grammatical structure inherent to reference genomes spanning many species
- In this work, we question the need for the vast majority of these inductive biases when focusing on a single high-impact downstream use case like human variant effect prediction (VEP)
  - Arguably, this is the most important application of gLMs likely to be successfully operationalized in the near future
  - It supports several commercially useful activities like target selection for new therapeutics, patient diagnostics, and clinical trial design (e.g. through enrollment critiera and risk stratification)
- We show this is the case by combining more principled, theoretically-grounded scaling practices we've used previously in language modeling [delphi / https://openathena.ai/blog/delphi/] with some ad hoc YOLO experiments to fill gaps where these methods break down on our task
  - This hybrid approach allowed us to prove that a difficult thing, i.e. surpassing Evo 2 40B with 1.8% as many training tokens (166B vs 9.3T) and ~0.05% as many FLOPs (1.1e21 vs 2.25e24) is at least possible

- Why are we defining the "hard thing" based on Evo 2 40B?
  - Evo 2 40B is the most formidable baseline among unsupervised, single-sequence DNA models
    - E.g. AlphaGenome is better, but it is not an unsupervised model and requires non-genomic data modalities
      - Similarly, GPN-Star is also better but requires input sequence alignment and a bespoke architecture
    - To our knowledge, no prior methods have demonstrated comparable performance to this baseline across diverse, conserved genomic regions, e.g. non-coding enhancers, promoters, UTRs and coding exons or introns.
  - Its 9.3T token, 2.25e24 FLOPs training budget is still unrivaled
    - Compare this training FLOP budget to leading LLMs
      - See @training-flops-vs-evo2.md
      - Describe how 2.25e24 is over/under model family generations, e.g.:
        - Over DeepSeek-V2 but not V3, Over Gemma 2 but not Gemma 3
    - Cost at 50% H100 MFU [beyondchinchilla / https://arxiv.org/abs/2401.00448] and $2 per H100-hour [olmo3 / https://arxiv.org/abs/2512.13961]


- We demonstrate that with standard autoregressive decoder models (Qwen3 in this case), tiny context windows (256bp @ single-bp resolution) and careful data curation alone, you can surpass Evo 2 40B on **zero-shot** VEP performance
  - Emphasis is on zero-shot here because supervised approaches introduce far more potential for leakage across splits
  - A lot of precedent actually exists for this "dumb" modeling approach, i.e. single-sequence, decoder-only, autogressive gLMs on stock architectures like Llama, Mistral or GPT
  - Notable well-resourced prior art from industry-academia collaborations or lone corporations includes Carbon [carbon] (Hugging Face), METAGENE-1 [metagene] (PrimeIntellect), GenSLM [genslm] (Cerebras), Gene42 [gene42] (Inception, M42), GENERator v1 [generator-v1] / v2 [generator-v2] (Alibaba), and DNAGPT [dnagpt] (Tencent)
  - Of these, only Carbon, GENERator, Gene42 and DNAGPT are relevant for human DNA
    - None of them report performance vs Evo 2 40B on zero-shot variant prioritization and instead don't compare to Evo directly or focus on trying to reach parity with smaller Evo 2 models like 7B or 20B
    - E.g. CARBON (3B and 8B) demonstrates lower performance on Mendelian (TraitGym) and coding ClinVar variant prioritization vs Evo 2 7B, and small gains on ClinVar non-coding variants

- Importantly, this enables construction of gLMs that can be deployed for inference cost-effectively and with good support across hardware and software stacks
   - E.g. just using raw PyTorch or JAX would be reasonable, as would using familiar training frameworks with inference support (Hugging Face Transformers, PyTorch Lightning, Levanter) or inference-optimized libraries (vllm, sglang)
   - For example, all of the evaluations run below on our 1B model are done with Levanter across various generations of TPUs with a throughput of approximately 50k tokens / sec / H100 equivalent; compare this to Evo 2 40B which runs at just 5k tokens / sec / H100 equivalent
     - TODO: get actual numbers for this (50k vs 5k is just a placeholder)

## Hyperparameter Transfer

- Following the Delphi [delphi] scaling recipe, we began by determining whether or not its possible to transfer hyperparameters across model size, token count and batch size
- Explain proportional data mix
  - Spans 3 genomic regions (CDS, upstream, downstream): ~331M training examples / ~85B tokens from 366,411 RefSeq accessions across ~500 species
  - All from animals dataset defined in https://github.com/Open-Athena/marin-dna/blob/a27534d104bc5c1505ab6c06e42157335ecf64ec/plots/exp58_evolutionary_timescales.py#L2
  - Proportional mix avoids conflation with epoching effects; each region is cycled only once
  - We do not yet have a mature, data-constrained hyperparameter transfer framework
- Explain reference Vizier sweep
  - ~25M params, 2.5B tokens, 16k batch (~4e17 FLOPs/run)
- Explain transfer validation
  - 10B tokens, 4k batch (~6.8e19 FLOPs/run); 76 runs across 255M/476M/1B scales (~2.9e21 FLOPs total)
  - Show `figure1_lr_transfer` (LR transfer) and `figure2_beta2_epsilon_transfer` (beta2 + epsilon)
  - LR is far more sensitive
    - Explain what this means for trying to run experiments w/o scaling LR based on tokens
  - Show `figure3_region_hyper_transfer` (validation by region)

## Parameter Scaling

- Hyperparameter transfer allows us to much more accurately estimate model scaling
- 8 model sizes (46M–4B params) at ~84B tokens each (~4.3e21 FLOPs across the sweep)
  - Show `figure4_loss_scaling`
- Loss scaling is smooth and fits to standard Kaplan laws well

## Downstream Performance

- As expected from prior art, parameter scaling does not yield monotonic improvements despite the tuning and scaling law results
  - Show `figure5_params_vs_vep_auprc`
- Loss correlation is weak
  - Show `figure6_loss_vs_vep_auprc`

 ## Mixture Experiments

- In an effort to further optimize our models, we move away from theoretically-grounded compute-constrained methods
  - Instead, we focus on mixture constituents (epoching them freely) and the extent to which they can be modified in-flight to compensate for observed performance gaps (YOLO)
- This still relies on two key results from the parameter scaling sweep:
  - Hyperparameter transfer scaling heuristics to configure training runs with very different token horizons
  - 1B target model scale since it was the largest model that still exhibited high VEP monotonicity
- We begin by training at 1B params on a uniform mixture of the same 3-region, animal sequences used previously
  - By ~50B tokens, this demonstrated saturation on upstream tasks (promoters and 5' UTRs) at significantly lower levels than models trained on upstream sequence alone
  - We then test shifts in mixture weights to identify whether or not upstream task performance can be improved without sacrificing performance on others
  - Show `figure9_upstream_mix_auprc`
  - Upstream task gains are easily undone by performance lost on other tasks
    - Similar experiments starting from models trained only on upstream data or from proportionally weighted checkpoints instead yielded no clear net-wins
  - Conclusion: improving zero-shot performance mid-flight is not really possible with non-uniform weighting of **existing** mixture components
- As an alternative strategy, we instead mix in new, distal sequence data largely from mammalian enhancer sequences with uniform weighting
  - This increases our mixture pool from 3 genomic regions (CDS, upstream, downstream) to 5 (+ncRNA exons and enhancers)
  - Surprisingly, this improves upstream task performance significantly (promoter VEP from 30%->40%) and very drastically improves distal task performance (ncRNA exon variants from 19%->65% and enhancer variants from 14%->33%) while mostly keeping performance on other tasks fixed
  - Our best recipe so far trains on a uniformly-weighted, 3-region mixture of sequence data proximal to genes (~104B tokens) followed by continued pretraining on a uniformly-weighted, 5-region mixture expanded to include distal sequences (~62B tokens)
    - This outperforms de novo training on the 5-region mixture
  - Show `figure10_lineage_vep_trajectory`
  - Conclusion: improving zero-shot performance mid-flight is possible by adding **new**, uniformly-weighted mixture components


## Conclusion

- We believe a fast, high-quality, easy-to-replicate gLM for human variant prioritization, with few restrictions on genomic context, would be a signifcant contribution to the field
  - We show here how to check most of these boxes; however, "easy-to-replicate" is still a work in progress
  - This model resulted from a messy, ad hoc process aided in unanticipated ways by hyperparameter transfer, scaling and mixture tools within Marin
  - Many less successful attempts are not mentioned here but documented in https://github.com/Open-Athena/marin-dna
- Ongoing work will hopefully yield a more consistent, effective training strategy and even greater quality gains
- There are several notable gaps or limitations in this research:
  - Regularization is entirely missing and that is a very important lever for data-constrained modeling
    - Again we are straddling data-constrained and compute-constrained regimes, but having narrowed the recipe now and with more effective infrastructure (TODO: link iris post) to better utilize the computing resources generously donated for these efforts by Google TPU Research Cloud, we believe we're likely to use this lever to significant effect in the future
  - It's not clear where exactly we're getting the most lift yet, but data curation is almost certainly the greatest contributor
    - We have not explained the details of this yet but plan to
- We have plenty of work left to do, but we think these results demonstrate the potential power that a general-purpose training platform like Marin can have for accelerating scientific foundation model development
