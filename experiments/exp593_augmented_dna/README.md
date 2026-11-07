# Augmented d768 DNA preparation

Prepare deterministic training presentations for [experiment #593](https://github.com/Open-Athena/marin-dna/issues/593).
This project runs on CPU and has no training or dispatch entry point.
Training remains subject to explicit approval.

## Generate and inspect examples

```bash
uv sync --locked
uv run --locked pytest
uv run --locked exp593-prep \
  --prep /home/exedev/tmp/mdna-exp593-data-prep \
  --output artifacts/prep-v1 --examples 100000 --samples 24
```

The prep directory contains the earlier metadata audit: pinned raw training record-batch samples, and a screened validation candidate pool.
The audit reuses the sampled training windows over independently keyed absolute occurrences; it does not replay the complete training shuffle.
The JSON report gives realized example rates, conditional RC rates, input-token rates, scored-target rates, length quantiles and padding.
Example JSONL includes the rendered text, typed text/DNA spans and transform provenance.
The stored audit generated 100,000 examples from 2,640 raw training windows covering all 65 species.
Observed FIM frequency was 50.066%, RC frequency 50.060%, and RC frequency within bare DNA 50.018%.
Padding occupied 5.917% of allocated tokens, and text occupied 0.884% of actual input tokens.
The median input length was 5,651 tokens, with a range of 2,048–8,192.
The 576 semantic components produced 59,983 distinct rendered prompt signatures, excluding DNA content from the signature.

## Presentation policy

| Objective | Probability | Observed in audit | Scored DNA |
| --- | ---: | ---: | --- |
| Bare causal DNA | 40% | 39.992% | Every available next base |
| Annotated causal DNA | 10% | 9.942% | DNA following the prompt |
| Short FIM | 37.5% | 37.556% | Missing 1, 2 or 3 bases, equally likely |
| Longer FIM | 12.5% | 12.510% | Missing 16, 32, 64, 128, 256 or 512 bases, equally likely |

Both FIM flanks precede the answer, with at least 128 bases in each flank.
Twenty percent of FIM gaps are explicitly centered; the rest have uniformly sampled valid positions.
The right flank is presented first in 25% of FIM examples, with labels retaining its identity.
Instruction position, flank labels, separators and syntax vary independently.

Every training objective uses the historical 50% Bernoulli RC draw, including bare DNA.
RC preserves case and complements all IUPAC bases before cropping and gap selection.
The draw is keyed by seed 472 and absolute occurrence, reproducing exp586 independently of presentation choices.
Separate random streams make replay independent of call order or process reslicing.

Half of examples preserve native PlantCAD softmask case: lowercase bases mark repeats.
The other half remove case-based mask information, equally split between uniform lowercase and uppercase DNA.
No example inverts repeat-mask case.
Bare DNA has no instructions or metadata; it may use any of these three case modes.
Formatted examples explain mask availability in 85% of cases.
Uniform-case explanations say that case does not encode repeat masks; uppercase never means repeat-masked.

For formatted examples, species names appear with probability 70%, taxonomy paths 35%, chromosome/contig 25%, assembly code 15%, recorded source strand 15%, and presentation orientation 20%.
These independent probabilities are conditional on a formatted example.
Species-only names and partial/full lineage strings coexist, with source-paper and accepted NCBI species names where applicable.
Source strand is explicitly described as recorded metadata, never as the orientation of the cropped sequence.
Exact genomic window coordinates are omitted because their offsets and coordinate convention are not established.
The complete registry retains the source paper identities, taxonomy provenance, common names and published-tree coverage; ambiguous common names are not sampled as species aliases.

The finite phrase catalog contains curated semantic components, with eight metadata syntaxes and three FIM layouts.
Field subsets, ordering, labels, structural renderers, lineages, length, crop, gap and RC combine these components into many presentations.
No runtime LLM paraphrasing is used.

## Tokenization, lengths and objective weights

Language text uses the pinned Marin BPE tokenizer.
DNA spans always use one existing token per character, preserving case and disabling cross-base BPE merges.
The full vocabulary is retained and no new special tokens are added.

Global batches choose 2,048, 4,096, 6,144 or 8,192 allocated tokens with probabilities 15%, 25%, 25% and 35%.
Each example independently fills 87.5–100% of its batch bucket, with a minimum total input length of 2,048.
The 2,048 bucket is therefore always fully occupied.
Text tokens are budgeted before selecting a DNA crop; there is no truncation of an assembled example.
One raw window contributes one example, and each row has its own attention segment with padding marked invalid.
Documents are never packed together.

Loss is applied only to scored DNA positions; prompt text, FIM flanks and padding have zero loss weight.
Weights sum to one per example, so short FIM contributes its intended example-mixture weight despite having few output targets.
The later trainer must aggregate these weights globally, including the denominator, before reducing the loss.
It must not normalize each device's loss separately when local example counts differ.

The planned training budget remains 216,158,699,520 actual input tokens, including text and excluding padding.
With variable lengths, 412,290 updates no longer correspond to that budget; the trainer must bind the warmup and decay to the actual input-token clock.
The optimized exp586 perf5 production runtime at `de5f428fc7b2bac8b6e40697f905c8b99ca3580d` is the intended training baseline.
Its compact lowercase embedding lookup needs adaptation for the larger input vocabulary.
This preparation has not measured augmented training throughput or integrated the training loop.

## Validation and norms

The original 512 validation indices and lowercased one-base tokenization remain fixed, with the metric `eval/loss`.
Three additional loss views reuse the same predictions and original case: softmasked targets, unmasked targets, and a 10:1 unmasked:softmasked weighted token loss.
Uppercase `N` stays in the primary loss and unmasked partition; its target count is separately reported.
The weighted loss is `(10 * unmasked_loss_sum + repeat_loss_sum) / (10 * unmasked_count + repeat_count)`.

An additional fixed 512-window diagnostic, `eval/loss_disjoint`, is sampled from the 11,581 previously screened validation candidates.
Its recorded source intervals do not intersect any training source interval, considering both strands.
Selection is deterministic, covers all 65 species and also prevents overlaps between the selected source intervals themselves.
This removes recorded genomic overlap; it does not guarantee absence of homology or repeats.
The main validation sample and this diagnostic have separate immutable manifests.
The later trainer should evaluate initially, at the end of warmup, approximately every 10,000 original-length-update equivalents of input tokens, and at the final token budget.

`diagnostics.model_norms` returns W&B-ready scalar metrics for the optimized Grug parameter/gradient trees.
The policy collects norms every 1,000 updates: global parameter/gradient L2 norms, plus attention Q, router and expert-down projection norms in layers 0, 4 and 7.
This is twenty scalars total, with each expert bank aggregated and no per-expert histograms.
Pass FP32 master weights and globally reduced, unscaled gradients before clipping or optimizer transforms.
Reduce logical global arrays on device; never sum physical copies of replicated parameters.
The collected scalars should join the ordinary log payload in `eric-czech/marin` on the CoreWeave W&B endpoint when training is authorized.
There is no W&B run created by this preparation.

## Provenance

Raw dataset revision: `4a444fff5520b992aa978d92a5af509a81977098`.
Language tokenizer revision: `a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2`.
The packaged species registry matches all 65 raw organism codes to the paper's Table S2 and current NCBI lineages.
Its complete taxonomy tree has no branch lengths; it is not a dated phylogeny.
The published phylogeny has 59 exact species matches and six explicitly unresolved species.
Source URLs are packaged alongside the registry.
