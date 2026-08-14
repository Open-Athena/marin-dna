# Tokenization

## Metadata

| Field | Value |
|---|---|
| Question ID | `RQ-0456` |
| Status | `active` |
| Overall confidence | `medium` |
| Evidence considered through | `2026-08-13` |
| Predecessor issues | [#456](https://github.com/Open-Athena/marin-dna/issues/456) |

## Question and scope

Which tokenizer, if any, should MarinDNA use for causal next-token pretraining?

Single-nucleotide tokens with a beginning-of-sequence token are the baseline. They make the prediction target and genomic coordinate explicit, preserve single-base sensitivity, and avoid segmentation assumptions. Moving away from them requires strong matched evidence.

Two motivations for alternative tokenization should be kept separate:

1. **Efficiency and context:** represent more base pairs with fewer model tokens, reducing attention and decoding cost or increasing the genomic span visible at a fixed token context.
2. **Biological abstraction:** learn discrete units that align with reusable sequence patterns and compress weakly constrained or unpredictable background sequence, so less model capacity is spent predicting base-level noise.

This question is restricted to tokenizers that produce a finite sequence of discrete IDs that the current decoder-only Marin model can train on with standard causal next-token cross-entropy. A tokenizer may be trained separately, including with a VQ-VAE-style reconstruction objective, and may emit fixed- or variable-span codes. The Marin language-model objective must remain ordinary next-token prediction. Factorized nucleotide supervision, masked or diffusion language modeling, auxiliary per-base losses, and tokenization schemes that require a different backbone are out of scope.

## Current answer

Single-nucleotide tokenization remains the MarinDNA default and current recommendation. It preserves base-level prediction and evaluation, and our only internal comparison found worse Mendelian VEP as fixed k-mer size increased, although that experiment had a repeat-weighting confound. Confidence is moderate: we have enough evidence to require a strong case for moving away from single bases, but not enough to claim they are optimal. The main gap is a matched comparison of single-base, fixed-block, and learned semantic tokenizers under ordinary causal next-token prediction. The most interesting alternative is a learned discrete tokenizer that spends fewer model steps on weakly constrained or noisy sequence while preserving useful biological signal.

Keep single-nucleotide tokenization as the default. The current evidence does not justify changing it.

MarinDNA's current preprocessing contract is explicitly character-level and target-aligned. This gives every nucleotide its own autoregressive prediction step and keeps likelihood-based variant scoring straightforward. The cost is a long token sequence and substantial compute spent on bases that may be weakly constrained or intrinsically hard to predict.

The only direct internal experiment, [#64](https://github.com/Open-Athena/marin-dna/issues/64), compared character, non-overlapping 4-mer, and non-overlapping 8-mer promoter models. TraitGym Mendelian validation performance on odd chromosomes degraded as k increased. The character baseline used repeat downweighting while the k-mer arms did not, so the character-versus-k-mer difference is not fully isolated; the 4-mer-versus-8-mer ordering is cleaner. This is evidence against adopting coarse fixed k-mers by default, not a definitive tokenizer sweep.

External evidence is objective- and setup-dependent. Carbon reports that fixed 6-mers substantially outperform learned BPE under its autoregressive ablation and provide a six-fold sequence-length reduction, but its final recipe later changes the language-model objective with Factorized Nucleotide Supervision. VQDNA shows that a VQ-VAE can learn discrete, pattern-aware genomic codes, but it evaluates those codes with masked-code modeling and supervised downstream tasks rather than causal next-token pretraining. Neither result establishes that an alternative tokenizer improves MarinDNA under the constraint here.

The highest-value direction is therefore a semantic learned tokenizer, especially one derived from representations of an existing MarinDNA model. A frozen character-level model could supply span representations; a separately trained quantizer and decoder could turn them into discrete codes; and a new Marin model could then train on those codes with unchanged next-token cross-entropy. This is an untested hypothesis. It must beat simple compression controls and show that its codes capture useful biology rather than GC content, repeat identity, phase, or tokenizer-boundary artifacts.

Confidence is moderate that single-nucleotide tokenization is the right default today and low that it is globally optimal. We have no internal learned-tokenizer result and no evidence that a semantic codebook can safely compress weakly constrained sequence while retaining base-sensitive VEP and generation.

## Confidence and limitations

Single-nucleotide tokenization remains the MarinDNA default and current recommendation. It preserves base-level prediction and evaluation, and our only internal comparison found worse Mendelian VEP as fixed k-mer size increased, although that experiment had a repeat-weighting confound. Confidence is moderate: we have enough evidence to require a strong case for moving away from single bases, but not enough to claim they are optimal. The main gap is a matched comparison of single-base, fixed-block, and learned semantic tokenizers under ordinary causal next-token prediction. The most interesting alternative is a learned discrete tokenizer that spends fewer model steps on weakly constrained or noisy sequence while preserving useful biological signal.

The highest-value direction is therefore a semantic learned tokenizer, especially one derived from representations of an existing MarinDNA model. A frozen character-level model could supply span representations; a separately trained quantizer and decoder could turn them into discrete codes; and a new Marin model could then train on those codes with unchanged next-token cross-entropy. This is an untested hypothesis. It must beat simple compression controls and show that its codes capture useful biology rather than GC content, repeat identity, phase, or tokenizer-boundary artifacts.

Confidence is moderate that single-nucleotide tokenization is the right default today and low that it is globally optimal. We have no internal learned-tokenizer result and no evidence that a semantic codebook can safely compress weakly constrained sequence while retaining base-sensitive VEP and generation.

## Operational consequence

Keep single-nucleotide tokenization as the default. The current evidence does not justify changing it.

## Supporting evidence

- The current [MarinDNA batch tokenizer](https://github.com/Open-Athena/marin-dna/blob/c3021ec6926b9f8f329c06679fec1d10539d0389/snakemake/analysis/evals_v2/src/marin_dna_evals/levanter/batch_tokenizer.py#L19-L92) requires a one-to-one character/token mapping and aligns case-based weights to next-token targets. This defines the operational baseline and exposes an implementation gap: compressed or variable-span tokenizers need a different preprocessing path even if the model loss remains unchanged.
- [Carbon: Decoding the Language of Life](https://www.biorxiv.org/content/10.64898/2026.05.22.727119v1) surveys single-nucleotide, fixed k-mer, BPE, and vector-quantized approaches. In its matched 3B, 50B-token ablation, fixed 6-mers reached 43.25% sequence recovery versus 21.68% and 23.48% for 32k- and 50k-vocabulary BPE, and BRCA2 AUROC of 75.04% versus 63.40% and 66.51%. Its interpretation is that variable BPE boundaries are poorly aligned with causal DNA generation, while fixed 6-mers are neutral compression units. The result supports testing fixed blocks before BPE in an autoregressive model. It does not compare 6-mers against MarinDNA's single-base baseline under matched data, model, and compute, and Carbon's later FNS loss and nucleotide-marginal inference are outside this question.
- [DNABERT-2](https://openreview.net/forum?id=oMLQB4EZE1) uses learned BPE with masked language modeling and reports strong multispecies downstream results. It shows that BPE can be useful for genomic representation learning, but it does not resolve Carbon's causal-boundary objection or provide evidence for BPE under MarinDNA next-token training.
- [VQDNA](https://proceedings.mlr.press/v235/li24bm.html) learns a vector-quantized genome vocabulary with a VQ-VAE reconstruction stage, then freezes the vocabulary for masked-code pretraining and downstream fine-tuning. It reports parameter-efficient results across 32 datasets and biologically associated code patterns. This is the closest precedent for model-learned semantic tokens. Its masked encoder setup, hierarchical residual quantization, and downstream evaluation do not show that the same codes are effective causal prediction targets. It also does not test initializing the tokenizer from representations of an already trained causal genomic model.

## Contradictory evidence

The predecessor issue did not maintain a separate contradictory-evidence section. Its caveats and negative results are preserved in Current answer and Supporting evidence.

## Related experiments

- [#64](https://github.com/Open-Athena/marin-dna/issues/64) compared character, non-overlapping 4-mer, and non-overlapping 8-mer tokenization on the mammalian promoter training setup. Odd-chromosome TraitGym Mendelian validation performance worsened with larger k. The character arm's repeat downweighting was not applied to the k-mer arms, so a clean replication should hold the data and loss weighting fixed.

## Open questions

### Evidence required to leave the baseline

- What improvement would justify moving away from single nucleotides: downstream quality at matched training FLOPs, throughput at matched quality, longer useful context, or some combination?
- How should comparisons account for both token budget and nucleotide exposure? Report model tokens, raw bases, realized epochs, training FLOPs, peak memory, and wall-clock throughput. Use bits per base rather than raw token cross-entropy when vocabulary size and bases per token differ.
- Which base-sensitive capabilities are non-negotiable? At minimum, compare likelihood-based VEP on permitted development chromosomes, generation/reconstruction fidelity, reverse-complement behavior, and sensitivity to substitutions and indels near token boundaries.

### Semantic and noise-aware tokens

- Can a tokenizer allocate short or specific codes to constrained sequence and coarser codes to weakly constrained background without using downstream labels at inference time?
- How should "semantic" be measured? Candidate observables include association with motifs and genomic annotations, stability across species and reverse complements, code reuse across sequence variants, and improved downstream performance after controlling for GC, repeats, and span length.
- Does compressing unpredictable sequence remove genuine but unannotated biology? Reconstruction error, code entropy, and downstream failures should be stratified by conservation, repeat class, and genomic region.
- If a token such as a generic background code is lossy, what sequence does it decode to? Lossy semantic codes still permit next-token training, but they change the modeled object and may make exact generation or nucleotide likelihoods unavailable. This tradeoff must be explicit.

### Using MarinDNA to build a tokenizer

- Which existing checkpoint, layer, pooling rule, and span sizes produce useful inputs to a quantizer?
- Should the first learned tokenizer quantize fixed-span hidden states, learn variable boundaries, or use a small encoder over local MarinDNA representations?
- Can a lightweight VQ encoder-decoder reconstruct sequence accurately while collapsing biologically equivalent or weakly constrained spans into shared codes?
- Are learned codes stable across random seeds, species, and checkpoints, or do code identities and boundaries drift too much to support a reusable pretokenized corpus?
- Does initializing from MarinDNA representations add value over VQ trained directly on one-hot bases, random projections, fixed k-mers, or ordinary sequence clustering?

### Candidate experiment ladder

1. **Clean fixed-token replication.** Compare character, 4-mer, 6-mer, and 8-mer tokenization on one fixed corpus with the same examples, uniform loss weighting, model family, optimizer, training FLOPs, and development evaluations. Report both matched-compute and matched-base-exposure views. This settles whether [#64](https://github.com/Open-Athena/marin-dna/issues/64) reproduces without the repeat-weighting confound.
2. **Frozen-model VQ pilot.** Use span representations from one released MarinDNA checkpoint to train a small VQ encoder-decoder, materialize discrete code sequences, and train a small causal model over those codes with ordinary next-token cross-entropy. Include one-hot VQ and fixed-k-mer controls.
3. **Semantic audit.** Measure compression ratio, bits per base, reconstruction, code usage, annotation and motif enrichment, conservation, repeat/GC confounding, reverse-complement consistency, and boundary sensitivity before scaling the language model.
4. **Matched downstream gate.** Compare the best learned tokenizer with the single-base baseline at matched FLOPs on odd autosomes plus chromosome X for development. Do not use even-autosome or chromosome-Y VEP labels during tokenizer selection. Advance only if the tokenizer improves a preregistered downstream or efficiency target without a material loss in base-sensitive scoring.

The learned-tokenizer direction should stop if codes mainly reproduce simple composition or repeat labels, if reconstruction failures concentrate in constrained bases, if codebooks are unstable, or if gains disappear against fixed-block controls at matched compute.

## History

- 2026-08-14 — Migrated from the predecessor research-question issue [#456](https://github.com/Open-Athena/marin-dna/issues/456). The issue remains the historical source for its original body and comments.
