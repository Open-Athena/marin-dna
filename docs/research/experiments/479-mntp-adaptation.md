# One-pass MNTP adaptation of MarinDNA m5.1

> [!NOTE]
> **TL;DR:** Several full-attention and causal-preserving adaptations failed to match the causal source in a single forward pass, and the evaluated MNTP models remained near-random on Mendelian VEP.

## Findings

None of the tested conversion routes produced a useful single-forward-pass bidirectional model.
A zero-initialized gated causal/full LoRA route came closest, but its cross-entropy remained worse than the causal source.
A symmetric two-causal-pass control outperformed the source causal readout, showing that complementary right-context information exists outside the single-pass constraint.

Evaluated MNTP models reached Mendelian macro AUPRC 0.1048–0.1151, close to the 0.10 random-ranking baseline and far below the source CLM at 0.3951.
The final run did not establish an improvement over optimization steps.
Gonzalo Benegas interprets this near-random VEP result as evidence that the adapted representations are poor for the intended use.

## Evidence

The fixed language-modeling validation panel selected one deterministic eligible nucleotide from each of 128 sequences in the pinned `validation` split of CDS, downstream, enhancer, ncRNA, and upstream data.
All readouts reused these 640 targets.
Lower four-way cross-entropy and higher accuracy are better.

The trained conversion candidates covered full-parameter transferred MNTP, damage-calibrated attention annealing with LoRA, a gated causal/full LoRA route, and reflected-RoPE [BICO](https://aclanthology.org/2024.emnlp-main.754/) LoRA at two learning rates.
Scratch MNTP was a control, while three mask-token choices and predictor-row-only future attention were frozen zero-update diagnostics.
Corrected causal continuation, a symmetric two-causal-pass readout, and dual-mode VEP routing were additional controls rather than single-pass conversion candidates.
The canonical [experiment issue](https://github.com/Open-Athena/marin-dna/issues/479) contains the complete run matrix.

| Readout | Four-way cross-entropy | Accuracy |
|---|---:|---:|
| Causal source | 1.0508 | 50.63% |
| Best one-pass candidate: gated causal/full LoRA | 1.058 | 51.25% |
| Symmetric two-causal-pass control | 0.9134 | 62.50% |

The gated candidate's cross-entropy was confidence-supported worse than the source despite its slightly higher point accuracy.
Every other one-pass route reached at most 44.84% accuracy.
Frozen-base runs reproduced the source causal readout exactly when their adapters were disabled.

Mendelian VEP scored 16,140 odd-autosome/X development variants in match groups containing one pathogenic positive and nine matched negatives.
The reported macro trajectory and paired intervals used 16,100 rows in the eight consequence subsets with at least 30 complete match groups.
The original transferred and scratch models reached AUPRC 0.1151 and 0.1112, while the final reflected-RoPE LoRA trajectory stayed between 0.1048 and 0.1113.
Every paired checkpoint interval versus step 0 included zero.

## Limitations

The routes were tested sequentially rather than as a matched benchmark and differed in trainable parameters, optimizer settings, masking, and inference cost.
The earlier transferred and scratch VEP models used a superseded loss normalization.
All trained candidates used one seed, one source checkpoint, and at most 1,000 steps.
The direct masked-site MNTP and source CLM VEP scores use different readouts, so their AUPRC values do not isolate the effect of bidirectional attention.
Same-position MLM, the full variable-time DiffuLLaMA objective, full-parameter BICO, causal infilling, and distillation were not tested.

## Related questions

- [Can causal gLM checkpoints be cheaply adapted into bidirectional representation models?](../questions/bidirectional-models.md)

## Research record

- [Experiment issue #479](https://github.com/Open-Athena/marin-dna/issues/479)
