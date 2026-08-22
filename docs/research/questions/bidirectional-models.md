# Can causal gLM checkpoints be cheaply adapted into bidirectional representation models?

> [!NOTE]
> **TL;DR:** One thousand BICO LoRA steps did not yield a useful one-pass MarinDNA model: nucleotide prediction remained below the causal source and Mendelian VEP was unchanged within uncertainty.

## Question

Can the bulk of genomic pretraining remain causal/autoregressive in Marin, followed by a cheap sidecar adaptation that produces a useful bidirectional representation model?

The target is a separate encoder fork for position- and sequence-level embeddings and, secondarily, variant-effect prediction.
The original causal checkpoint remains the generation model.
Training an MLM from scratch, preserving generation in the adapted fork, and building an arbitrary-order generation model are outside scope.

## Current answer

The [m5.1 MNTP adaptation experiment](../experiments/479-mntp-adaptation.md) tested the current one-pass hypothesis.
A 1,000-step rank-16 BICO LoRA run improved four-way nucleotide cross-entropy from 1.387 to 1.274 and accuracy from 33.0% to 41.1%.
The causal source reached 1.051 and 50.6% on the same targets, while a symmetric two-causal-pass control reached 0.913 and 62.5%.
The one-pass candidate therefore failed the nucleotide-information gate despite access to both directions.

Mendelian macro AUPRC moved from 0.1048 at step 0 to 0.1080 at step 1,000.
All paired checkpoint intervals included zero, so the run produced no resolved VEP improvement or degradation.

Cheap causal-to-bidirectional conversion remains unproven for MarinDNA.
The tested recipe should not receive a longer budget, and downstream VEP evaluation should wait until a one-pass candidate at least matches the causal source on paired nucleotide prediction.

<details>
<summary>Related work</summary>

| Route | Evidence | Remaining gap |
|---|---|---|
| Masked next-token adaptation | [LLM2Vec](https://arxiv.org/abs/2404.05961) removed causal attention and used 1,000 LoRA MNTP steps before contrastive training. | The evidence is from text embeddings and does not isolate MNTP for DNA. |
| Protein CLM-to-MLM transfer | [Training Compute-Optimal Protein Language Models](https://arxiv.org/abs/2411.02142) continued causal protein models with bidirectional MLM and improved two of three downstream results. | Masked training consumed most of the total tokens, so the result does not demonstrate cheap post-training of a mature checkpoint. |
| Decoder-to-embedding adaptation | [NV-Embed](https://arxiv.org/abs/2405.17428) removed causal attention and added learned latent-attention pooling. | Pooling and objective changes are confounded, and DNA transfer remains untested. |

</details>

<details>
<summary>Related experiments</summary>

- The [m5.1 MNTP adaptation experiment](../experiments/479-mntp-adaptation.md) found that a 1,000-step BICO LoRA run improved within-run nucleotide prediction but remained below the causal source; Mendelian VEP was statistically flat.
- [#3](https://github.com/Open-Athena/marin-dna/issues/3) compared causal, masked, and diffusion objectives during early promoter training and proposed a causal-to-masked curriculum, but did not convert a mature checkpoint under a small budget.
- [#314](https://github.com/Open-Athena/marin-dna/issues/314) provides causal embedding and VEP baselines without testing jointly bidirectional states.

</details>

<details>
<summary>Possible directions</summary>

- Keep the m5.1 causal source and symmetric two-pass result as fixed information controls.
- Within the same 1,000-step budget, test whether a different attention parameterization or same-position MLM can close the one-pass nucleotide gap.
- Run representation, supervised sequence-to-function, and VEP evaluations only after the nucleotide gate passes.

</details>
