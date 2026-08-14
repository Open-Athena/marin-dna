# Can causal gLMs become bidirectional representation and arbitrary-order generation models?

## TL;DR

Causal checkpoints can plausibly gain bidirectional representations through mixed objectives or lightweight adaptation, but no MarinDNA experiment has shown that this improves representation quality or VEP while preserving autoregressive generation.
Confidence is low; the main gap is a matched conversion ablation with explicit generation-retention tests.

## Question

Can we adapt existing MarinDNA causal/autoregressive checkpoints into models that use both left and right sequence context, without retraining from scratch or sacrificing their useful autoregressive capabilities?
The primary goal is better position- and sequence-level embeddings for downstream genomic tasks; secondary goals are stronger variant-effect prediction and flexible sequence generation, including fill-in-the-middle (FIM), inpainting/infilling, and arbitrary-order generation.

## Current answer

No MarinDNA checkpoint-conversion experiment has been run.
The shortest test of the representation hypothesis is to remove the causal mask, continue training with a masked or masked-next-token objective, and compare against causal continued pretraining at matched data and compute.

Three end states should be kept separate.
A converted bidirectional encoder may improve token and pooled representations while losing ordinary generation.
A mixed-mode decoder could preserve left-to-right generation while adding bidirectional representation and infilling, but objective interference is a risk.
Masked diffusion offers bidirectional states and arbitrary-order generation through one interface, with different likelihood semantics and iterative sampling cost.
Fill-in-the-middle is a useful generation-preserving control but does not create jointly bidirectional per-position states.

The current hypothesis is feasible but low confidence.
Language and biological-sequence precedents show each component separately; none demonstrates that a causal MarinDNA checkpoint can gain better VEP representations while retaining its strongest autoregressive behavior.
The first decision gate should measure representation quality, zero-shot and probed VEP, ordinary generation retention, and compute cost before attempting a unified arbitrary-order model.

<details>
<summary>Related work</summary>

| Route | Setup and finding | Implication and remaining gap |
|---|---|---|
| Causal decoder to bidirectional encoder | [LLM2Vec](https://arxiv.org/abs/2404.05961) enables full attention and masked next-token/contrastive adaptation; [NV-Embed](https://arxiv.org/abs/2405.17428) removes the causal mask during embedding training and adds learned pooling. | A trained decoder can be converted with limited continued training. DNA transfer and retention of autoregressive generation remain untested. |
| Mixed causal and bidirectional decoder | [MAGNET](https://aclanthology.org/2025.acl-long.1325/) combines representation learning, infilling, and retained generation; [UniMAE](https://aclanthology.org/2025.acl-short.57/) is a lighter representation-focused variant. | One checkpoint may support several modes. Objective cooperation and genomic scaling remain open. |
| Causal infilling | [FIM](https://arxiv.org/abs/2207.14255), [GLM blank infilling](https://arxiv.org/abs/2103.10360), and [ProtFIM](https://arxiv.org/abs/2303.16452) autoregressively reconstruct missing spans. | This is a cheap control that preserves generation. It does not yield jointly bidirectional hidden states or arbitrary-order denoising. |
| Masked diffusion | [Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524), DNA model [D3LM](https://arxiv.org/abs/2603.01780), track-conditioned [Nona](https://doi.org/10.1101/2025.11.06.687036), and protein model [DPLM](https://proceedings.mlr.press/v235/wang24ct.html) use iterative masked denoising. | The interface supports bidirectional representation, inpainting, and generation. Sampling cost and VEP likelihood semantics differ from a causal LM. |
| Block diffusion | [Fast-dLLM v2](https://arxiv.org/abs/2509.26328), [Efficient-DLM](https://arxiv.org/abs/2512.14067), and [Block Diffusion](https://arxiv.org/abs/2503.09573) preserve causality across blocks and denoise within a block. | Conversion may be smoother and generation faster, but only the active block is bidirectional. |
| Internal representation reports | [#246](https://github.com/Open-Athena/marin-dna/issues/246) maps functional-region embeddings from causal models and [#11](https://github.com/Open-Athena/marin-dna/issues/11) tracks masked-language-model support. | They define evaluation and infrastructure context. Neither is a conversion experiment or evidence that bidirectionality improves MarinDNA. |

</details>

<details>
<summary>Related experiments</summary>

- [#3](https://github.com/Open-Athena/marin-dna/issues/3) compared causal language modeling, masked language modeling, and masked diffusion during early promoter training.
  Causal modeling led at the earliest steps and the issue proposed a causal-to-masked curriculum; it did not convert a mature checkpoint or test representation retention.
- [#110](https://github.com/Open-Athena/marin-dna/issues/110) explored mixed next-token, fill-in-the-middle, and autoregressive blank-infilling objectives.
  It motivates generation-preserving controls but contains no completed conversion result.
- [#314](https://github.com/Open-Athena/marin-dna/issues/314) evaluated frozen causal gLM embeddings across VEP datasets and representation choices.
  It provides a baseline for any converted model, including the limits of two-pass FWD/RC representations, but does not test jointly bidirectional states.

</details>

## Possible directions

### What exactly should be converted?

- Is the first target a separate bidirectional encoder fork of a MarinDNA checkpoint, or one checkpoint that can switch between causal, bidirectional, FIM, and/or diffusion modes?
- How much adaptation is needed after removing the causal mask: no training, parameter-efficient training, or full continued pretraining?
  Which layers change most?
- Should the first recipe use fixed-rate MLM, LLM2Vec-style masked next-token prediction, a variable-mask diffusion objective, or a mixture with the original next-token loss?
- Does the answer differ for attention-based Transformers versus causal convolution/SSM architectures?
  Removing an attention mask is straightforward for the former; models such as Hyena/Mamba need an explicit forward/backward construction rather than an attention-mask change.

### Do the representations actually improve?

- At equal checkpoint, data, and adaptation compute, do bidirectional embeddings improve:
  - functional-region separation in [#246](https://github.com/Open-Athena/marin-dna/issues/246);
  - frozen-embedding VEP probes from [#314](https://github.com/Open-Athena/marin-dna/issues/314);
  - token-level tasks where downstream context should matter, such as splice-site or exon annotation?
- Which layer and pooling rule work best?
  Mean pooling is a baseline, but NV-Embed suggests learned pooling may matter.
- Do improvements come from bidirectionality, the masked objective, extra training, or contrastive learning?
  Each needs a matched ablation.

### Does variant-effect prediction improve?

SNV scoring is fixed per orientation: mask the variant position, read the reference- and alternate-allele probabilities from the same output distribution, and compute their log-likelihood ratio in one forward pass.
This is the standard zero-shot scoring protocol for masked protein and genomic language models.
For a fair comparison with the existing MarinDNA protocol, score both the forward and reverse-complement orientations and average them identically across models; this is a matched evaluation control, not a consequence or test of bidirectionality.

- Does a converted model's strand-averaged masked-site SNV LLR beat the equivalently strand-averaged causal score and frozen-embedding probe at matched context and parameter count?
- How do we define a comparable score for indels and other multi-base variants without giving one model extra context or an easier normalization?

### Which generation capability do we actually need?

- Is FIM/blank infilling sufficient for near-term applications, or do we need true arbitrary-order iterative generation?
- For masked diffusion, should sampling unmask single bases, spans, or blocks?
  Can known sequence outside a target interval remain exactly fixed during inpainting?
- Can one model retain high-quality left-to-right generation while gaining infilling and refinement, or is a dedicated bidirectional/diffusion fork cleaner?
- What biological generation tests should gate progress: held-out infill recovery, motif preservation, k-mer/GC distributions, sequence novelty, predicted regulatory activity, or task-specific design constraints?

### Candidate experiment ladder

1. **Checkpoint-conversion smoke test.**
   Start from one small/medium MarinDNA Qwen3 checkpoint.
   Compare matched-budget causal continued pretraining against bidirectional masked adaptation.
   Measure held-out causal loss, masked-token accuracy, [#246](https://github.com/Open-Athena/marin-dna/issues/246) embeddings, and [#314](https://github.com/Open-Athena/marin-dna/issues/314) probes.
2. **Objective ablation.**
   Compare fixed-rate MLM or masked next-token prediction, mixed NTP + masked training, and a lightweight contrastive term.
   Keep data, steps, and parameter updates matched.
3. **Infilling control.**
   Train a causal FIM/blank-infilling arm at the same budget.
   This tests whether practical infilling can be obtained without changing the model into an encoder or diffusion model.
4. **Masked-diffusion arm.**
   Only after the cheaper conversions are understood, adapt the same checkpoint with a variable mask-rate objective and test full-sequence inpainting/arbitrary-order generation.
