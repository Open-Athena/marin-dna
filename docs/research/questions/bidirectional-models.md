# Can causal gLM checkpoints be cheaply adapted into bidirectional representation models?

> [!NOTE]
> **TL;DR:** A one-seed 1B MarinDNA pilot shows that 1,000 full-parameter MNTP steps can cheaply create bilateral context use and a small validation-loss transfer advantage, but the converted checkpoint regressed sharply on all tested VEP endpoints, so conversion is technically feasible while downstream usefulness and any case for longer adaptation remain unestablished.

## Question

Can the bulk of genomic pretraining remain causal/autoregressive in Marin, where it can be run at scale, followed by a relatively cheap sidecar adaptation that produces a useful bidirectional representation model?

The target is a separate encoder fork for position- and sequence-level embeddings and, secondarily, variant-effect prediction.
The original causal checkpoint remains the generation model.
Training the best possible MLM from scratch, preserving autoregressive generation in the adapted fork, and building a unified arbitrary-order generation model are outside the scope of this question.

## Current answer

The [full-attention MNTP adaptation experiment](../experiments/479-mntp-adaptation.md) ran the first MarinDNA checkpoint conversion.
The standalone pilot continued the released 1B m5.1 checkpoint for 1,000 full-parameter masked-next-token-prediction steps on one Lambda GH200, with matched scratch MNTP and causal-continuation arms plus source and full-attention/no-adaptation controls.
It was technically valid. The completed pilot plus checkpoint, stability, alignment, and dependency audits cost an estimated $24.73 at list price, including failed, recovery, and cancelled diagnostic attempts.

Transferred MNTP reached slightly lower pooled validation loss than scratch (0.39727 versus 0.39954) and lower single-mask loss (0.31008 versus 0.31315).
It acquired dependence on both flanks, while the causal controls remained right-blind.
The strict control criterion was only partially met: transferred MNTP exceeded full attention without adaptation on the right-flank probe but not the left.

These behavioral gains did not transfer to variant-effect prediction.
Transferred single-orientation MNTP scored 0.1151 Mendelian macro AUPRC, 0.1003 complex-trait global AUPRC, and 0.1427 SGE accession/consequence macro AUPRC, compared with 0.3951, 0.1342, and 0.3577 for source CLM with forward/reverse-complement averaging.
Single-orientation transferred scores stayed within one AUPRC point of their own FWD+RC scores, but no task passed the required source-improvement gate.
Complete-flank ablations confirmed that both flanks affected individual scores, and ±64-base window shifts were stable, yet neither diagnostic rescued downstream performance.

A targeted integrity audit found no checkpoint serialization, deterministic replay, coordinate, tokenizer, shifted-readout, shared-loss-path, or optimization-instability bug.
Source save/reload and replayed CLM step 400 were bit-exact across 51,623 odd/X variants and both strands, and all three 400-step training replays matched their original per-step losses exactly.
Continued-CLM degradation was progressive rather than immediate: fixed-plan validation loss stayed at 0.23138 through step 1, was 0.23131 at step 10, then rose to 0.27310 at 100, 0.33297 at 400, and 0.35965 at 800 before partially recovering to 0.35010 after cooldown.
Its gradient norms were mild, with no post-warmup spikes.
This supports destructive optimization from the fresh optimizer and high registered peak learning rates rather than a load/save or inference mismatch.

The audit did find one bug in the original dependency diagnostic: it compared a batch-one wild-type baseline with batch-1,020 substitutions, so BF16 batch-shape numerics contaminated the maps.
Corrected same-call maps at the three step-1,000 checkpoints supersede that analysis.
Transferred MNTP had past/future mean dependency 0.05314/0.05334, scratch MNTP 0.03056/0.02917, and continued CLM 0.12510/0 exactly.
Both MNTP arms use context on both sides; the continued-CLM map's entire forbidden future triangle is exactly zero.

The result is evidence that cheap behavioral conversion works, not that the resulting checkpoint is a useful representation model.
It argues against automatically extending this exact one-seed MNTP recipe to 10,000 steps.
It does not answer whether ordinary MLM, a different update budget or parameterization, layer/pooling choices, or supervised sequence-to-function training can exploit the bilateral states.
See the [audited compact result snapshot](https://github.com/Open-Athena/marin-dna/tree/issue-479-mntp-pilot-audited-result/.agents/artifacts/479-mntp-adaptation), [checkpoint audit](https://wandb.ai/gonzalobenegas/marin/runs/gavkgtmf), [stability audit](https://wandb.ai/gonzalobenegas/marin/runs/q67hbkp4), and [final dependency](https://wandb.ai/gonzalobenegas/marin/runs/yl5sgffn) runs.

Before this direct evidence,
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

The current hypothesis is therefore narrower:

> A predominantly causal MarinDNA checkpoint can be cheaply adapted into a behaviorally bidirectional model, but usefulness must be demonstrated on representation or supervised sequence-to-function tasks before spending on longer adaptation; 1,000-step MNTP VEP performance is negative evidence, not a continuation signal.

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

- The [full-attention MNTP adaptation experiment](../experiments/479-mntp-adaptation.md) completed the first mature-checkpoint MNTP conversion pilot.
  It established technical feasibility, a small transferred-versus-scratch validation advantage, corrected bilateral final-checkpoint context use, and negative source-relative VEP at 1,000 steps. Its integrity audit found no training/inference bug and superseded the original batch-shape-contaminated dependency maps.
- [#3](https://github.com/Open-Athena/marin-dna/issues/3) compared causal language modeling, masked language modeling, and masked diffusion during early promoter training.
  Causal modeling led at the earliest steps and the issue proposed a causal-to-masked curriculum; it did not convert a mature checkpoint or constrain the masked phase to a small compute budget.
- [#314](https://github.com/Open-Athena/marin-dna/issues/314) evaluated frozen causal gLM embeddings across VEP datasets and representation choices.
  It provides the main baseline for a converted model, including the limitations of two-pass forward/reverse-complement representations, but does not test jointly bidirectional states.

</details>

<details>
<summary>Possible directions</summary>

### How much sidecar adaptation is enough?

- Use the released 1B m5.1 source, controls, and compact metrics from #479 as the fixed baseline for any follow-up rather than repeating the completed smoke test.
- Measure adaptation cost as a fraction of the sunk causal pretraining FLOPs and tokens.
  The central unknown is whether useful bidirectionality appears after a genuinely small phase, not whether long MLM training can eventually win.
- Treat #479's sequential MNTP result as the fixed objective baseline rather than the default continuation path.
  Compare ordinary same-position MLM at the same small adaptation budget before considering more MNTP steps.
- Compare parameter-efficient and full-parameter adaptation.
  LLM2Vec supports LoRA as a cheap starting point, while the protein transfer result used continued pretraining of the model rather than a parameter-efficient conversion.
- Keep the sidecar operationally independent of Marin unless the result later justifies first-class training support.
- If causal continuation is used as a control again, preserve or retune optimizer state and peak learning rates; #479's fresh high-learning-rate continuation progressively damaged validation loss and VEP despite stable gradients.

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

#479 answers the first 1,000-step MNTP comparison negatively for direct masked-site SNV scoring: the converted checkpoint did not beat source CLM on Mendelian, complex-trait, or SGE endpoints.

- Does ordinary same-position MLM, a materially different adaptation budget, or a representation-level probe avoid the VEP regression?
- How should indels and other multi-base variants be scored without giving one objective extra context or easier normalization?

### Candidate experiment ladder

1. **Objective attribution.**
   At the same small sidecar budget and source checkpoint, compare transferred MNTP with transferred ordinary same-position MLM.
   Keep the existing source, no-adaptation, and causal-continuation controls; a second from-scratch masked model is lower priority than isolating the objective.
2. **Representation gate.**
   Test layer and pooling choices on frozen functional-region or token-level probes where right context has a clear mechanism.
   Direct masked-site VEP should remain a diagnostic rather than the sole gate.
3. **Supervised sequence-to-function test.**
   Compare the source and converted checkpoints in a matched accessibility or other per-base prediction setup without treating the negative VEP result as proof that representations cannot help.
4. **Budget and update ablation.**
   Consider longer adaptation, LoRA, or additional seeds only after an objective or representation test shows a source-relative gain.
   Report incremental tokens, FLOPs, and cloud cost against the #479 baseline.

</details>
