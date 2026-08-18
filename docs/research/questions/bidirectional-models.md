# Can causal gLM checkpoints be cheaply adapted into bidirectional representation models?

> [!NOTE]
> **TL;DR:** Sequential causal-to-masked-objective transfer is feasible in protein models, while full-attention masked next-token prediction is the leading untested hypothesis for cheap MarinDNA adaptation; confidence is moderate that conversion is feasible and low that a small adaptation budget will improve representations or variant-effect prediction.

## Question

Can the bulk of genomic pretraining remain causal/autoregressive in Marin, where it can be run at scale, followed by a relatively cheap sidecar adaptation that produces a useful bidirectional representation model?

The target is a separate encoder fork for position- and sequence-level embeddings and, secondarily, variant-effect prediction.
The original causal checkpoint remains the generation model.
Training the best possible MLM from scratch, preserving autoregressive generation in the adapted fork, and building a unified arbitrary-order generation model are outside the scope of this question.

## Current answer

No MarinDNA checkpoint-conversion experiment has been run.
[Training Compute-Optimal Protein Language Models](https://arxiv.org/abs/2411.02142) provides the closest biological evidence that the proposed sequence can work.
The study first trained a causal protein Transformer, then initialized the same architecture from that checkpoint, reset the optimizer, removed the causal attention restriction, and continued with bidirectional 15%-mask MLM; the transfer phase used a fresh learning-rate schedule with 5% warmup.
Its 470M transfer run used 21B causal tokens followed by 85B masked tokens.
At the same reported pretraining FLOPs, a comparison model used ordinary MLM for all 106B tokens.
After LoRA fine-tuning, the transfer model improved contact-prediction P@L/5 from 0.78 to 0.80 and fold-classification accuracy from 0.65 to 0.66, while fluorescence performance remained 0.67.
Simultaneously mixing causal and masked training increased target validation loss, although the comparison may be confounded by a lower effective batch size per objective.

This result supports sequential transfer and shows that causal pretraining need not prevent later bidirectional learning.
It does **not** directly answer the MarinDNA question: most of the transfer run's compute was masked training, the objective switch happened early, and the study did not test DNA or a mature causal checkpoint whose pretraining compute was already sunk.
The scratch-MLM arm is evidence about the protein paper's transfer result, not the baseline MarinDNA needs to optimize against.

There is also a meaningful choice between ordinary MLM and masked next-token prediction.
With ordinary MLM, the representation at masked position $i$ predicts the original token $x_i$.
With [LLM2Vec](https://arxiv.org/abs/2404.05961)'s MNTP, position $i$ is masked but $x_i$ is predicted from the output at position $i-1$, while attention is bidirectional.
That one-token shift preserves the next-token alignment learned by a causal decoder and is therefore the stronger first hypothesis for a cheap adaptation.
LLM2Vec obtained useful text embeddings with a 1,000-step LoRA MNTP phase before contrastive training, showing that such conversion can be lightweight in another domain.
It did not establish that MNTP is better than matched ordinary MLM for genomic models, so same-position MLM remains a useful small-budget objective ablation rather than a from-scratch baseline.

The current hypothesis is therefore:

> A predominantly causal MarinDNA checkpoint can be forked and cheaply adapted with full attention plus MNTP into a better bidirectional representation model, while the original checkpoint remains unchanged for autoregressive use.

<details>
<summary>Related work</summary>

| Route | Setup and finding | Implication and remaining gap |
|---|---|---|
| Masked next-token adaptation | [LLM2Vec](https://arxiv.org/abs/2404.05961) removes the causal attention mask and uses MNTP, supervising a masked token from the preceding output position. Its unsupervised conversion used 1,000 LoRA steps before contrastive training. | The shifted objective matches a causal decoder's pretrained output alignment and can be a small sidecar phase. The evidence is from text models and does not isolate MNTP from ordinary MLM for DNA. |
| Protein CLM-to-MLM transfer | [Training Compute-Optimal Protein Language Models](https://arxiv.org/abs/2411.02142) sequentially continued causal protein Transformers with bidirectional 15%-mask MLM. A 470M run used 21B CLM plus 85B MLM tokens and modestly improved two of three downstream results relative to MLM from scratch at the same reported compute. | This is the closest biological precedent and favors sequential rather than simultaneous training. Its adaptation phase was most of the total compute, so it does not demonstrate cheap post-training of a mature causal checkpoint. |
| Decoder-to-embedding adaptation | [NV-Embed](https://arxiv.org/abs/2405.17428) removes the causal mask during embedding training and adds learned latent-attention pooling. | Bidirectional attention and the adaptation objective are not the only choices; pooling can materially affect sequence embeddings. DNA transfer remains untested. |
| Internal representation reports | [#246](https://github.com/Open-Athena/marin-dna/issues/246) maps functional-region embeddings from causal models, [#314](https://github.com/Open-Athena/marin-dna/issues/314) evaluates causal embeddings for VEP, and [#11](https://github.com/Open-Athena/marin-dna/issues/11) tracks MLM support. | These define the evaluation and infrastructure context. None is a causal-to-bidirectional conversion experiment. |

</details>

<details>
<summary>Related experiments</summary>

- [#3](https://github.com/Open-Athena/marin-dna/issues/3) compared causal language modeling, masked language modeling, and masked diffusion during early promoter training.
  Causal modeling led at the earliest steps and the issue proposed a causal-to-masked curriculum; it did not convert a mature checkpoint or constrain the masked phase to a small compute budget.
- [#314](https://github.com/Open-Athena/marin-dna/issues/314) evaluated frozen causal gLM embeddings across VEP datasets and representation choices.
  It provides the main baseline for a converted model, including the limitations of two-pass forward/reverse-complement representations, but does not test jointly bidirectional states.

</details>

<details>
<summary>Possible directions</summary>

### How much sidecar adaptation is enough?

- Start from a completed MarinDNA causal checkpoint rather than switching objectives during the main training run.
- Measure adaptation cost as a fraction of the sunk causal pretraining FLOPs and tokens.
  The central unknown is whether useful bidirectionality appears after a genuinely small phase, not whether long MLM training can eventually win.
- Use sequential MNTP as the first hypothesis because it retains the causal decoder's one-token output alignment.
  Compare ordinary same-position MLM only at the same small adaptation budget.
- Compare parameter-efficient and full-parameter adaptation.
  LLM2Vec supports LoRA as a cheap starting point, while the protein transfer result used continued pretraining of the model rather than a parameter-efficient conversion.
- Keep the sidecar operationally independent of Marin unless the result later justifies first-class training support.

### Do the representations actually improve?

- At the same starting checkpoint, data, and extra compute, compare:
  - the original causal checkpoint;
  - the checkpoint with full attention enabled but no adaptation;
  - matched short causal continuation;
  - MNTP adaptation;
  - optionally, ordinary MLM adaptation as an objective ablation.
- Evaluate functional-region separation in [#246](https://github.com/Open-Athena/marin-dna/issues/246), frozen-embedding VEP probes from [#314](https://github.com/Open-Athena/marin-dna/issues/314), and token-level tasks where right context should matter, such as splice-site or exon annotation.
- Test layer and pooling choices separately from the adaptation objective.
  Mean pooling is the baseline, while NV-Embed suggests learned pooling may add an independent gain.
- Attribute any improvement to full attention, masked adaptation, extra optimization, or pooling rather than treating conversion as one indivisible change.

### Does variant-effect prediction improve?

SNV scoring is fixed per orientation: mask the variant position, read the reference- and alternate-allele probabilities from the same output distribution, and compute their log-likelihood ratio in one forward pass.
For a fair comparison with the existing MarinDNA protocol, score both forward and reverse-complement orientations and average them identically across models; this is a matched evaluation control, not a consequence or test of bidirectionality.

- Does a converted model's strand-averaged masked-site SNV score beat the equivalently strand-averaged causal score and frozen-embedding probe at matched context and parameter count?
- How should indels and other multi-base variants be scored without giving one objective extra context or easier normalization?

### Candidate experiment ladder

1. **Sidecar conversion smoke test.**
   Fork one small or medium completed MarinDNA checkpoint, leave the causal source checkpoint untouched, and run a tightly capped MNTP adaptation outside Marin.
   Measure masked-token validation loss and a small set of representation probes over the adaptation trajectory.
2. **Matched controls.**
   Compare the original checkpoint, full attention without training, and causal continuation at the same extra compute.
   Add ordinary MLM only as a matched adaptation-objective ablation; do not train an MLM from scratch.
3. **Budget and update ablation.**
   Sweep a small number of adaptation budgets and compare LoRA with full-parameter updates.
   Report the incremental FLOPs and tokens relative to the original CLM run.
4. **Representation gate.**
   Run the broader [#246](https://github.com/Open-Athena/marin-dna/issues/246) and [#314](https://github.com/Open-Athena/marin-dna/issues/314) evaluations only if the smoke test shows a useful gain at an acceptably small budget.

</details>
