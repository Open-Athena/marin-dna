# Can causal gLM checkpoints be cheaply adapted into bidirectional representation models?

> [!NOTE]
> **TL;DR:** The tested 1,000-step LoRA conversion left one-pass nucleotide prediction below the causal source and Mendelian VEP near its 0.10 random-ranking baseline, lowering our expectation that a useful model can be obtained through similarly cheap adaptation.

## Question

Can the bulk of genomic pretraining remain causal/autoregressive in Marin, where it can be run at scale, followed by a relatively cheap sidecar adaptation that produces a useful bidirectional representation model?

The target is a separate encoder fork for position- and sequence-level embeddings and, secondarily, variant-effect prediction.
The original causal checkpoint remains the generation model.
Training the best possible MLM from scratch, preserving autoregressive generation in the adapted fork, and building a unified arbitrary-order generation model are outside the scope of this question.

## Current answer

The [m5.1 MNTP adaptation experiment](../experiments/479-mntp-adaptation.md) tested a 1,000-step rank-16 LoRA conversion using [BIdirectional Causal language model Optimization (BICO)](https://aclanthology.org/2024.emnlp-main.754/), which maps future-key RoPE offsets into the negative-offset range seen during causal pretraining.
Four-way nucleotide cross-entropy improved from 1.387 to 1.274 and accuracy from 33.0% to 41.1%.
The causal source reached 1.051 and 50.6% on the same targets, while a symmetric two-causal-pass control reached 0.913 and 62.5%.
The one-pass candidate therefore failed the nucleotide-information gate despite access to both directions.

Mendelian macro AUPRC stayed between 0.1048 and 0.1113 across checkpoints, close to the 0.10 random-ranking baseline from one-positive/nine-negative match groups.
The run did not establish better-than-random VEP, and all paired checkpoint intervals versus step 0 included zero.
Gonzalo Benegas interprets this near-random VEP result as evidence that the adapted representations are poor for the intended use.

Cheap causal-to-bidirectional conversion remains unproven for MarinDNA.
We now have low confidence that a useful one-pass model can be obtained through a similarly cheap adaptation.
Other objectives, attention parameterizations, and larger budgets remain untested.
The tested recipe should not receive a longer budget, and downstream VEP evaluation should wait until a one-pass candidate at least matches the causal source on paired nucleotide prediction.

<details>
<summary>Related work</summary>

| Route | Evidence | Remaining gap |
|---|---|---|
| Masked next-token adaptation | [LLM2Vec](https://arxiv.org/abs/2404.05961) removed causal attention and used 1,000 LoRA MNTP steps before contrastive training. | The evidence is from text embeddings and does not isolate MNTP for DNA. |
| RoPE-aware bidirectional attention | [BICO](https://aclanthology.org/2024.emnlp-main.754/) opens future attention while keeping relative-position offsets in the range seen during causal pretraining. | The original work used an autoregressive blank-infilling objective on text rather than MNTP on DNA. |
| Protein CLM-to-MLM transfer | [Training Compute-Optimal Protein Language Models](https://arxiv.org/abs/2411.02142) continued causal protein models with bidirectional MLM and improved two of three downstream results. | Masked training consumed most of the total tokens, so the result does not demonstrate cheap post-training of a mature checkpoint. |
| Decoder-to-embedding adaptation | [NV-Embed](https://arxiv.org/abs/2405.17428) removed causal attention and added learned latent-attention pooling. | Pooling and objective changes are confounded, and DNA transfer remains untested. |

</details>

<details>
<summary>Related experiments</summary>

- The [m5.1 MNTP adaptation experiment](../experiments/479-mntp-adaptation.md) found that a 1,000-step BICO LoRA run improved within-run nucleotide prediction but remained below the causal source; Mendelian VEP stayed near its random-ranking baseline and was statistically flat within the run.
- [#3](https://github.com/Open-Athena/marin-dna/issues/3) compared causal, masked, and diffusion objectives during early promoter training and proposed a causal-to-masked curriculum, but did not convert a mature checkpoint under a small budget.
- [#314](https://github.com/Open-Athena/marin-dna/issues/314) provides causal embedding and VEP baselines without testing jointly bidirectional states.

</details>

<details>
<summary>Possible directions</summary>

- Keep the m5.1 causal source and symmetric two-pass result as fixed information controls.
- Within the same 1,000-step budget, test whether a different attention parameterization or same-position MLM can close the one-pass nucleotide gap.
- Run representation, supervised sequence-to-function, and VEP evaluations only after the nucleotide gate passes.

</details>
