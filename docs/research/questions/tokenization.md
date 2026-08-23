# Tokenization

> [!NOTE]
> **TL;DR:** Single-nucleotide tokenization remains the MarinDNA default because it preserves base-level prediction and the only internal comparison worsened Mendelian VEP as fixed k-mer size increased, although repeat weighting was confounded; confidence is moderate pending a matched causal-next-token comparison with fixed-block and learned semantic tokenizers.

## Question

Which tokenizer, if any, should MarinDNA use for causal next-token pretraining?

Single-nucleotide tokens with a beginning-of-sequence token are the baseline.
They make the prediction target and genomic coordinate explicit, preserve single-base sensitivity, and avoid segmentation assumptions.
Moving away from them requires strong matched evidence.

Two motivations for alternative tokenization should be kept separate:

1. **Efficiency and context:** represent more base pairs with fewer model tokens, reducing attention and decoding cost or increasing the genomic span visible at a fixed token context.
2. **Biological abstraction:** learn discrete units that align with reusable sequence patterns and compress weakly constrained or unpredictable background sequence, so less model capacity is spent predicting base-level noise.

This question is restricted to tokenizers that produce a finite sequence of discrete IDs that the current decoder-only Marin model can train on with standard causal next-token cross-entropy.
A tokenizer may be trained separately, including with a VQ-VAE-style reconstruction objective, and may emit fixed- or variable-span codes.
The Marin language-model objective must remain ordinary next-token prediction.
Factorized nucleotide supervision, masked or diffusion language modeling, auxiliary per-base losses, and tokenization schemes that require a different backbone are out of scope.

## Current answer

Keep single-nucleotide tokenization as the default.
The current evidence does not justify changing it.

MarinDNA's current preprocessing contract is explicitly character-level and target-aligned.
This gives every nucleotide its own autoregressive prediction step and keeps likelihood-based variant scoring straightforward.
The cost is a long token sequence and substantial compute spent on bases that may be weakly constrained or intrinsically hard to predict.

The only direct internal experiment, [#64](https://github.com/Open-Athena/marin-dna/issues/64), compared character, non-overlapping 4-mer, and non-overlapping 8-mer promoter models.
TraitGym Mendelian validation performance on odd chromosomes degraded as k increased.
The character baseline used repeat downweighting while the k-mer arms did not, so the character-versus-k-mer difference is not fully isolated; the 4-mer-versus-8-mer ordering is cleaner.
This is evidence against adopting coarse fixed k-mers by default, not a definitive tokenizer sweep.

External evidence is objective- and setup-dependent.
Carbon reports that fixed 6-mers substantially outperform learned BPE under its autoregressive ablation and provide a six-fold sequence-length reduction, but its final recipe later changes the language-model objective with Factorized Nucleotide Supervision.
VQDNA shows that a VQ-VAE can learn discrete, pattern-aware genomic codes, but it evaluates those codes with masked-code modeling and supervised downstream tasks rather than causal next-token pretraining.
Neither result establishes that an alternative tokenizer improves MarinDNA under the constraint here.

The highest-value direction is therefore a semantic learned tokenizer, especially one derived from representations of an existing MarinDNA model.
A frozen character-level model could supply span representations; a separately trained quantizer and decoder could turn them into discrete codes; and a new Marin model could then train on those codes with unchanged next-token cross-entropy.
This is an untested hypothesis.
It must beat simple compression controls and show that its codes capture useful biology rather than GC content, repeat identity, phase, or tokenizer-boundary artifacts.

Confidence is moderate that single-nucleotide tokenization is the right default today and low that it is globally optimal.
We have no internal learned-tokenizer result and no evidence that a semantic codebook can safely compress weakly constrained sequence while retaining base-sensitive VEP and generation.

<details>
<summary>Related work</summary>

- [Carbon: Decoding the Language of Life](https://www.biorxiv.org/content/10.64898/2026.05.22.727119v1) surveys single-nucleotide, fixed k-mer, BPE, and vector-quantized approaches.
  In its matched 3B, 50B-token ablation, fixed 6-mers reached 43.25% sequence recovery versus 21.68% and 23.48% for 32k- and 50k-vocabulary BPE, and BRCA2 AUROC of 75.04% versus 63.40% and 66.51%.
  Its interpretation is that variable BPE boundaries are poorly aligned with causal DNA generation, while fixed 6-mers are neutral compression units.
  The result supports testing fixed blocks before BPE in an autoregressive model.
  It does not compare 6-mers against MarinDNA's single-base baseline under matched data, model, and compute, and Carbon's later FNS loss and nucleotide-marginal inference are outside this question.
- [DNABERT-2](https://openreview.net/forum?id=oMLQB4EZE1) uses learned BPE with masked language modeling and reports strong multispecies downstream results.
  It shows that BPE can be useful for genomic representation learning, but it does not resolve Carbon's causal-boundary objection or provide evidence for BPE under MarinDNA next-token training.
- [VQDNA](https://proceedings.mlr.press/v235/li24bm.html) learns a vector-quantized genome vocabulary with a VQ-VAE reconstruction stage, then freezes the vocabulary for masked-code pretraining and downstream fine-tuning.
  It reports parameter-efficient results across 32 datasets and biologically associated code patterns.
  This is the closest precedent for model-learned semantic tokens.
  Its masked encoder setup, hierarchical residual quantization, and downstream evaluation do not show that the same codes are effective causal prediction targets.
  It also does not test initializing the tokenizer from representations of an already trained causal genomic model.

</details>

<details>
<summary>Related experiments</summary>

- [#64](https://github.com/Open-Athena/marin-dna/issues/64) compared character, non-overlapping 4-mer, and non-overlapping 8-mer tokenization on the mammalian promoter training setup.
  Odd-chromosome TraitGym Mendelian validation performance worsened with larger k.
  The character arm's repeat downweighting was not applied to the k-mer arms, so a clean replication should hold the data and loss weighting fixed.

</details>

<details>
<summary>Possible directions</summary>

- Replicate the character, fixed 4-mer, 6-mer, and 8-mer comparison with uniform weighting, matched compute, and matched base exposure.
- Train a small causal model over fixed-span VQ codes derived from a frozen MarinDNA checkpoint, with one-hot VQ and fixed-k-mer controls.
- Audit compression ratio, bits per base, reconstruction, code stability, reverse-complement and boundary behavior, and associations with annotations after controlling for GC and repeats.
- Advance only if a learned tokenizer improves a preregistered quality or efficiency target without a material loss in substitution- and indel-sensitive scoring.

</details>
