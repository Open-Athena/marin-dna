# What context sizes do genomic language models need for different biological tasks, and how should models acquire and use that context?

> [!NOTE]
> **TL;DR:** m5.1 Mendelian VEP benefits from inference context up to its 255 bp training length, has a protocol-specific result at 511 bp, and fails sharply at 1023 bp; matched training-context and genuinely long-range task comparisons remain necessary because useful context depends on the task, objective, and acquisition method.

## Question

What context sizes do genomic language models need across biological tasks, model scales, pretraining, inference, and downstream adaptation?
When a task needs more context than a short-window checkpoint saw during training, how should the model acquire and use it while retaining nucleotide-level resolution?
At fixed compute and data, how do second-stage long-context language modeling, direct long-context downstream fine-tuning, and a hierarchical local-to-global architecture compare?
Which strategy best preserves what the short-context model learned while adding useful dependencies across tens to hundreds of kilobases?

## Current answer

Context needs are task- and protocol-dependent, and inference length should not be treated as interchangeable with training length.
For the fixed 255 bp-trained m5.1 checkpoint on Mendelian development variants, cropping inference from 255 to 31 bp reduced macro AUPRC from 0.3945 to 0.1941 zero-shot and from 0.4779 to 0.3174 with newly trained frozen probes.
Inference extension to 511 bp produced a small zero-shot gain that was statistically significant in the paired analysis but no significant probe change.
Extension to 1023 bp sharply reduced both protocols relative to 511 bp.
These results show that executable rotary positions do not guarantee useful extrapolation beyond the training context.
They do not establish the best pretraining context because checkpoint weights were fixed.

A 256-versus-512 bp pretraining comparison found no clear difference on the Promoter VEP subset; other consequence subsets were not tested.
No MarinDNA experiment directly compares ways to add genuinely long-range context to a short-window model or measures a task that requires dependencies across tens to hundreds of kilobases.
The 1023 bp m5.1 result remains far below that regime but shows that direct extension baselines should align training and inference context deliberately.

Three strategies remain viable.
Continued long-context language modeling could produce a reusable base model but adds the most compute and risks diluting functional signal with abundant background sequence.
Direct long-context downstream fine-tuning is the minimum-complexity baseline and aligns context learning with labels, but sparse labels may not teach reusable long-range structure.
A local-to-global model reuses the short-context encoder and delegates cross-window interactions to a second model; full-resolution tiling preserves base-level outputs, while pooling is probably required at much longer contexts.

A short local encoder has improved a 2,114 bp supervised accessibility model in published work, which supports feasibility of local-to-global transfer.
That setup freezes the local encoder, never lets it attend across chunk boundaries, and remains far below chromosome-scale context, so it does not establish long-context language understanding.

Confidence remains low on context lengths beyond local VEP windows and on a universal acquisition strategy.
Direct downstream extension should be the baseline, local-to-global modeling is the leading scalable option for nucleotide-resolution tasks, and continued language modeling is justified only if gains transfer across several long-range tasks.
Any comparison must match parameters, training tokens, and compute, align training and inference context deliberately, and verify dependence on distant sequence.

<details>
<summary>Related work</summary>

- [ARSENAL](https://www.biorxiv.org/content/10.64898/2026.02.05.703637v3) pretrains a 350 bp masked DNA model on ENCODE cCREs, tiles frozen per-base embeddings across a 2,114 bp input, and feeds them to a ChromBPNet-style dilated CNN.
  Across five cell lines it improves accessibility count prediction and caQTL/dsQTL scoring over a matched one-hot model.
  The [implementation](https://github.com/amanpatel101/arsenal-chrombpnet/blob/dcfa42b1786713e131bb113f4c6d20acc046185d/chrombpnet/chrombpnet.py#L188-L223) freezes the encoder by default, and its [center-out chunking](https://github.com/amanpatel101/arsenal-chrombpnet/blob/dcfa42b1786713e131bb113f4c6d20acc046185d/chrombpnet/chrombpnet.py#L391-L456) retains one embedding per base.
  This is direct evidence for full-resolution local-to-global transfer.
  It does not test long-context pretraining, cross-chunk attention in the gLM, pooling, or contexts beyond 2,114 bp.
- [AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0) predicts thousands of functional tracks from 1 Mb sequence using a multiscale supervised architecture.
  It demonstrates that long context and nucleotide-scale outputs can coexist through hierarchical computation.
  It does not test initialization from a short-context self-supervised gLM, so it is an architecture precedent rather than evidence for transfer.

</details>

<details>
<summary>Related experiments</summary>

- [Inference-context sensitivity of m5.1 Mendelian VEP](../experiments/485-m5-1-inference-context.md) holds one 255 bp-trained checkpoint fixed across 31–1023 bp inference windows.
  Cropping below 255 bp lowers both macro metrics, 511 bp has a zero-shot-only gain, and 1023 bp fails sharply, but the experiment does not compare training contexts or a genuinely long-range task.
- [#37](https://github.com/Open-Athena/marin-dna/issues/37) compared 256 bp and 512 bp pretraining contexts on the Promoter VEP subset and found no clear difference.
  It leaves both short windows viable for that subset and proposed downstream extension or hierarchy, but the single-subset comparison is too narrow and too short-range to distinguish the three long-context strategies.

</details>

<details>
<summary>Possible directions</summary>

- Compare checkpoints trained at 255, 511, and 1023 bp with crop and extension ladders, and transfer one fixed probe across contexts.
- Choose a task that requires distant sequence by construction and verify its dependence with distance-stratified cropping, masking, shuffling, or element perturbation.
- At matched parameters, tokens, and compute, compare direct downstream extension with frozen per-base tiling, pooled local-to-global modeling, and joint local-to-global training.
- Measure how window sampling, loss weighting, pooling, and tiling affect local grammar, nucleotide-resolution outputs, and likelihood interpretation.
- Update locus and sequence-similarity leakage controls before using longer overlapping windows.

</details>
