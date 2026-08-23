# Can causal gLM checkpoints be cheaply adapted into bidirectional representation models?

> [!NOTE]
> **TL;DR:** Several 1,000-step conversion routes failed to match the causal source in a single forward pass, and Mendelian VEP remained near its 0.10 random-ranking baseline, lowering our expectation that a useful model can be obtained through similarly cheap adaptation.

## Question

Can the bulk of genomic pretraining remain causal/autoregressive in Marin, where it can be run at scale, followed by a relatively cheap sidecar adaptation that produces a useful bidirectional representation model?

The target is a separate encoder fork for position- and sequence-level embeddings and, secondarily, variant-effect prediction.
The original causal checkpoint remains the generation model.
Training the best possible MLM from scratch, preserving autoregressive generation in the adapted fork, and building a unified arbitrary-order generation model are outside the scope of this question.

## Current answer

The [m5.1 MNTP adaptation experiment](../experiments/479-mntp-adaptation.md) tested full-parameter and LoRA conversion routes under budgets of at most 1,000 steps.
None matched the causal source in a single forward pass.
The best one-pass candidate had cross-entropy 1.058 versus 1.0508 for the source, while a symmetric two-causal-pass control reached 0.9134 and confirmed that the two directions contain complementary information.

Evaluated MNTP models remained close to the Mendelian VEP random-ranking baseline and far below the source CLM.
No checkpoint improvement was established within the final run.

Cheap causal-to-bidirectional conversion remains unproven for MarinDNA, and we have low confidence that a useful one-pass model can be obtained through a similarly cheap adaptation.
Other objectives, attention parameterizations, and larger budgets remain untested.
Downstream evaluation should wait until a one-pass candidate at least matches the causal source on paired nucleotide prediction.

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

- The [m5.1 MNTP adaptation experiment](../experiments/479-mntp-adaptation.md) tested full-parameter and LoRA conversion routes; none matched the causal source in one pass, while a two-pass information control succeeded outside the target constraint.
- [#3](https://github.com/Open-Athena/marin-dna/issues/3) compared causal, masked, and diffusion objectives during early promoter training and proposed a causal-to-masked curriculum, but did not convert a mature checkpoint under a small budget.
- [#314](https://github.com/Open-Athena/marin-dna/issues/314) provides causal embedding and VEP baselines without testing jointly bidirectional states.

</details>

<details>
<summary>Possible directions</summary>

- Keep the m5.1 causal source and symmetric two-pass result as fixed information controls.
- Within the same 1,000-step budget, test whether a different attention parameterization or same-position MLM can close the one-pass nucleotide gap.
- Run representation, supervised sequence-to-function, and VEP evaluations only after the nucleotide gate passes.

</details>
