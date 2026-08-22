# One-pass MNTP adaptation of MarinDNA m5.1

> [!NOTE]
> **TL;DR:** Several full-attention and causal-preserving adaptations failed to match the causal source in a single forward pass, and the evaluated MNTP models remained near-random on Mendelian VEP.

## Findings

No tested candidate produced a useful single-forward-pass bidirectional model.
Ordinary full-attention MNTP, attention annealing, predictor-row-only future attention, and reflected-RoPE LoRA all remained worse than the causal source on paired nucleotide prediction.
Changing among `[MASK]`, `[UNK]`, and `[PAD]` did not explain the full-attention deficit.
A zero-initialized gated causal/full LoRA route came closest, but it required two computation paths and still failed cross-entropy non-inferiority.
The symmetric two-causal-pass control was better than the source causal readout, confirming that complementary right-context information exists outside the single-pass constraint.

The original transferred and scratch MNTP models reached Mendelian macro AUPRC 0.1151 and 0.1112, compared with 0.3951 for the source CLM.
The final standard-rate reflected-RoPE LoRA trajectory stayed between 0.1048 and 0.1113, close to the 0.10 random-ranking baseline defined by one positive and nine matched negatives per group.
The final run did not establish better-than-random VEP or a change over optimization steps; every paired checkpoint interval versus step 0 included zero.
Gonzalo Benegas interprets this near-random VEP result as evidence that the adapted representations are poor for the intended use.

The frozen-base runs reproduced the source causal readout exactly when their adapters were disabled.
The evidence does not support selecting a VEP checkpoint or extending any tested recipe beyond 1,000 steps.

## Evidence

The fixed validation panel used 640 identical masked nucleotide targets: 128 sequences from each of CDS, downstream, enhancer, ncRNA, and upstream data.
The approaches were tested sequentially and differ in trainable parameter count and inference cost.

| Tested route | Setup | Outcome |
|---|---|---|
| Full-parameter transferred and scratch MNTP | Ordinary full attention with shifted masked-token prediction for 1,000 steps. | The corrected transferred model reached CE/accuracy 1.260/41.56%, below the causal source at 1.051/50.78%; the earlier transferred and scratch models had Mendelian AUPRC 0.115/0.111. |
| Mask-token controls | No training; compare a new `[MASK]` row with existing `[UNK]` and `[PAD]`. | Full-attention accuracy was 27.2–27.7% for all three, ruling out the mask-token choice as the main cause. |
| Damage-calibrated attention annealing with LoRA | Open future edges gradually through step 800, then train 200 steps at full attention. | Final CE/accuracy was 1.366/31.88%, versus source 1.051/50.63%. |
| Predictor-row-only future attention | No training; only the shifted predictor row can attend to future keys. | CE/accuracy was 1.207/44.84%, versus causal 1.052/51.09%. |
| Zero-initialized gated causal/full LoRA | Mix a frozen causal path with a trained full-attention path. | Final CE/accuracy was 1.058/51.25%; CE was confidence-supported worse than source, and the right-context gate failed. |
| [BIdirectional Causal language model Optimization (BICO)](https://aclanthology.org/2024.emnlp-main.754/) reflected-RoPE LoRA | Map future-key RoPE offsets into the negative-offset range seen during causal pretraining; test `1e-5` and `5e-5`. | The two runs ended at 1.283/40.94% and 1.274/41.09%; both were below source 1.051/50.63%. |
| Symmetric two-causal-pass control | Combine native causal forward and reverse-complement distributions without updating the model. | CE/accuracy reached 0.913/62.50%; it passed the nucleotide gate but violates the one-pass goal. |
| Dual-mode VEP routing control | Combine the two-pass central conditional with the source full-sequence score residual. | Mendelian/complex/SGE AUPRC was 0.394/0.134/0.356 versus source 0.396/0.134/0.358; the route is not a converted single-pass model. |

Corrected causal continuation served only as an optimizer control.
Its five-component validation CE increased from 0.7690 to 0.7737 over 1,000 AdamW steps at `1e-5`.

The final standard-rate BICO run used full reflected-RoPE attention, excluded masked keys, and trained a rank-16 LoRA adapter while keeping the source parameters frozen.

| Readout | Four-way cross-entropy | Accuracy |
|---|---:|---:|
| Causal source | 1.050770 | 50.63% |
| BICO LoRA step 0 | 1.387224 | 32.97% |
| BICO LoRA step 1,000 | 1.273889 | 41.09% |
| Symmetric two-pass control | 0.913447 | 62.50% |

Training used 94,000 sequences, 24.1 million model tokens, a physical batch of 94 without accumulation, and a `5e-5` peak learning rate.
The schedule warmed up for 100 steps, stayed constant through step 800, and decayed through step 1,000.

The Mendelian point estimates used 16,140 odd-autosome/X development variants.
Every complete match group contained one pathogenic positive and nine matched negatives, making 0.10 the expected AUPRC for random ranking.
Paired uncertainty used 2,000 seed-0 match-group bootstrap replicates over 16,100 rows in the eight consequence subsets with at least 30 groups.

| Step | Mendelian macro AUPRC | Change from step 0 | Paired 95% interval |
|---:|---:|---:|---:|
| 0 | 0.104816 | 0 | reference |
| 100 | 0.111331 | +0.006515 | [-0.000809, +0.013839] |
| 400 | 0.108776 | +0.003960 | [-0.002557, +0.010476] |
| 1,000 | 0.107966 | +0.003150 | [-0.003377, +0.009677] |

## Limitations

The approaches were sequential rather than a matched benchmark and differed in trainable parameters, optimizer settings, masking, and inference cost.
The earlier transferred and scratch VEP models trained with a superseded loss normalization, so their scores describe those checkpoints without isolating the intended corrected objective.
All trained candidates used one seed, one source checkpoint, and at most 1,000 steps.
Same-position MLM, the full variable-time DiffuLLaMA objective, full-parameter BICO, causal infilling, and distillation were not tested.
The nucleotide gates intentionally give each bidirectional candidate both sequence directions at the target, so the deficits are failed engineering gates rather than matched-objective likelihood comparisons.
The Mendelian checkpoint intervals measure change within the BICO run; they do not compare direct masked-site scoring with the source CLM's different full-sequence VEP score.

## Related questions

- [Can causal gLM checkpoints be cheaply adapted into bidirectional representation models?](../questions/bidirectional-models.md)

## Research record

- [Experiment issue #479](https://github.com/Open-Athena/marin-dna/issues/479)
