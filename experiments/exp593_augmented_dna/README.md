# Augmented d768 Grug MoE DNA adaptation

Train one model for [experiment #593](https://github.com/Open-Athena/marin-dna/issues/593), starting from the same text-pretrained checkpoint as #586.
All implementation and artifacts belong to the permanent, unmerged [companion PR #594](https://github.com/Open-Athena/marin-dna/pull/594).
The user has authorized preparation, single-node H100 validation, and full training when ready.
Data preparation and the single-node H100 smoke are complete.
The full fixed-panel accelerator evaluator passed on eight H100s in about 65 seconds, including all 36 greedy completions.
The four-node production loop passed sixteen bounded updates and native checkpoint discovery.
The single production run has been submitted on one eight-H100 node as `dna-exp593-augmented-d768-pretrained-seed0-v1` in personal `eric-czech/marin`; worker initialization and numerical progress are being monitored.
The accelerator evaluator reads the immutable preparation bundle, rechecks historical membership and native-case FIM reconstruction, and scores raw token NLL with binary masks.
Its fixed generation panel contains two minimal/formatted pairs for each of the nine gap lengths (36 presentations); each completion is greedy, uses the full original vocabulary, and stops at the caller-specified length.
The `exp593_dna.smoke --evaluate-panels` option exercises all causal and FIM panels after the bounded training/checkpoint smoke.
An evaluation-only retry can restore that validation run's own full-state checkpoint with `--evaluate-checkpoint`.

## Run training

```bash
# Bounded validation uses a separate run and checkpoint namespace.
python -m exp593_dna.train --cluster cw-us-east-02a --nodes 1 \
  --run-id dna-exp593-loop-smoke-v1 --validation-updates 16
# Submit production only after this exact profile passes validation.
python -m exp593_dna.train --cluster cw-us-east-02a --nodes 1
```

These entry points dispatch eight processes and eight H100s per node through Iris at batch priority.
Use the explicit Iris user and target configuration in [operations.md](operations.md), and forward current CoreWeave and personal W&B credentials without recording their values.
The single production identity is `dna-exp593-augmented-d768-pretrained-seed0-v1` in `eric-czech/marin` at the CoreWeave endpoint.
Scientific bindings exclude placement and are read back before checkpoint discovery; resumed execution restores the latest complete optimizer/data state and preserves that run identity.
CPU rendering prefetches one batch while device collectives stay ordered on the main thread.
Periodic checkpoints occur approximately every 30 minutes, with two temporary recovery saves retained; warmup, quarter-budget and final checkpoints are permanent.
The run stops after the first complete 64-example update reaching the nominal 216,158,699,520 input-token budget.
It records the actual input count and overshoot, which is less than 524,288 tokens (2.43 parts per million), preserving the minimum example length and equal per-example objective at the last update.
Padding never advances the schedule.

## Reproduce preparation

```bash
uv sync --locked --extra cpu
uv run --locked pytest
uv run --locked exp593-prep \
  --prep /home/exedev/tmp/mdna-exp593-data-prep \
  --annotations /home/exedev/tmp/mdna-exp593-data-prep/frozen-annotations/train.parquet \
  --output artifacts/prep-v2 --examples 100000 --samples 32
uv run --locked python -m exp593_dna.fim_validation \
  --manifest artifacts/prep-v2/validation_disjoint_512.json \
  --partitions /home/exedev/tmp/mdna-exp593-data-prep/by-species \
  --annotations /home/exedev/tmp/mdna-exp593-data-prep/frozen-annotations/validation.parquet \
  --output artifacts/prep-v2
```

`prepare_inputs` downloads and verifies the original sequence-only exp586 Parquet objects.
`partition_inputs` joins their original flat row indices to pinned Hugging Face metadata and checks frozen native-case sequence samples.
`reference_sources` obtains versioned public reference FASTA, GFF and assembly reports with official checksum verification.
`build_annotations` projects those features onto exact existing DNA windows.
`freeze_annotations` builds compact split-specific Parquet annotation banks with row/hash bindings.
`publish_inputs` uploads a content-addressed input bundle and verifies every object's complete readback.
The remote trainer must consume the original train objects plus the frozen metadata and annotation bundle; reference FASTA is never training input.

## Source restriction and annotations

Training is restricted to the original 2,638,656 PlantCAD train windows of 8,192 bases, at dataset revision `4a444fff5520b992aa978d92a5af509a81977098`.
No new sequences, reconstructed parent regions, validation rows or evaluation data enter training.
The native corpus contains 21,615,869,952 DNA bases.
The stored metadata field `assembly` identifies organism codes, rather than providing assembly accessions or release versions.
The packaged registry resolves all 65 codes to source-paper identities and NCBI lineages.
It preserves species-only names and partial/full taxonomy paths; ambiguous common names are excluded from sampled species aliases.
The published dated phylogeny has 59 exact species matches and six unresolved species; the complete packaged taxonomy tree has no branch lengths.

Reference features cover translation starts, translation stops, splice donors and splice acceptors across mapped training windows.
They are not selected from evaluation coordinates or labels.
Each accepted annotation requires one exact full-window DNA match, at a documented original tiling offset, in the recorded parent interval and strand.
Case is ignored only for mapping identity; the original source DNA and its case are retained as targets.
Contig aliases are explicit and DNA-verified; ambiguous mappings fall back to random placement.
Unsupported trans-spliced or conflicting transcripts are skipped.
Start/stop bounds are inferred from complete CDS when explicit codon features are absent; annotation biology can therefore be imperfect.

The first frozen bank annotates 909,773 train windows across 23 species, 34.479% of the original corpus.
The train bank is approximately 109 MB compressed and uses an Arrow-backed row index rather than materializing millions of Python feature objects per worker.
The corresponding validation bank is separate and cannot pass the training augmentation entry point.
All other genomes and rows remain in training, with random FIM placement when no usable annotation exists.

## Objective and joint augmentation

| Exclusive objective lane | Configured examples | Observed examples | Scored DNA |
| --- | ---: | ---: | --- |
| Bare causal DNA | 40% | 39.992% | Every available next base |
| Annotated causal DNA | 10% | 9.942% | DNA following the prompt |
| Short FIM | 37.5% | 37.556% | Missing 1, 2 or 3 bases, equally likely |
| Longer FIM | 12.5% | 12.510% | Missing 16, 32, 64, 128, 256 or 512 bases, equally likely |

Loss weights sum to one per example, so FIM receives approximately 50% of the objective despite its few target tokens.
The model's fused cross-entropy must reduce the weighted numerator and denominator globally.
Only DNA targets are scored: causal DNA or the missing FIM span.
Prompt text, flanks and padding have zero loss weight, and no EOS/end-fill token is supervised.
Completion length remains caller-controlled, including prompts that omit a length hint.
There is no training loss weighting by repeat status, landmark class or species.

All objectives independently receive 50% Bernoulli reverse complement augmentation, including bare causal DNA and minimal FIM.
RC preserves case and supports the full IUPAC alphabet.
The seed-472 draw is keyed by absolute occurrence, retaining the exp586 draw independently of presentation choices and process placement.
Half of examples preserve the native PlantCAD convention: lowercase nucleotides indicate masked repeats.
One quarter remove masks by lowercasing all DNA; one quarter uppercase all DNA.
No presentation inverts the repeat-mask convention.
Bare DNA therefore supports uniformly uppercase or uniformly lowercase input without instructions.
Formatted uniform-case examples describe masks as unavailable.

Within FIM, 25% of examples use minimal structural labels without optional metadata or prose.
The remaining FIM examples share metadata augmentation with the annotated-causal lane.
Conditional on formatted eligibility, species appears with probability 70%, taxonomy 35%, chromosome/contig 25%, assembly code 15%, source strand 15%, presentation orientation 20%, mask availability 85%, and descriptive prose 80%.
Source strand describes the recorded source metadata, not an inferred orientation of the cropped presentation.
Unverified genomic window coordinates are never rendered.
Eight metadata syntaxes, three FIM layouts and 584 curated semantic components combine field subsets, order, labels and phrasings without runtime LLM generation.

Both FIM flanks precede the answer, retaining at least 128 bases per flank.
The right flank is presented first in 25% of FIM examples, with labels preserving its meaning.
Eighty percent include an explicit gap length; minimal examples use a compact form such as `[gap:3]`.
Random control gaps are centered with probability 20% and otherwise uniformly positioned in the valid interior.
Short FIM uses landmark guidance with probability 75% among examples with eligible sites, retaining random controls and all unmapped examples.
Guidance samples an available landmark class uniformly, then a unique site uniformly.
Half of guided gaps maximize motif overlap and half lie fully outside the motif within a 32-base neighborhood.
RC and crop transforms are applied to feature coordinates before choosing gaps.
Long FIM retains random positions.

The updated audit exercised 100,000 occurrences over 2,640 frozen train-window samples covering all 65 species.
It measures realized transforms on those reused samples, rather than replaying the full shuffled training corpus.
FIM accounts for 50.066% of examples and normalized objective mass; RC is active in 50.060% of all examples.
Minimal FIM accounts for 12.422% of all examples.
Annotation guidance is active in 9.310% of all examples and 24.790% of short FIM examples.
The JSON report separates overlapping marginal, conditional and joint counts; those counts do not sum to 100%.
The earlier prep-v1 artifacts remain historical and predate minimal FIM, length hints and landmark guidance.

## Tokenization, lengths and scheduling

Language text uses the pinned Marin BPE tokenizer at revision `a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2`.
DNA uses one original token ID per character, preserving case and disabling cross-base BPE merges.
The full 128,256-token vocabulary remains available and no new tokens are introduced.
Global batches contain 64 independent examples in one shared length bucket: 2,048, 4,096, 6,144 or 8,192 tokens with probabilities 15%, 25%, 25% and 35%.
Examples fill 87.5–100% of their bucket, with a minimum total length of 2,048 tokens.
Text is budgeted before selecting the DNA crop; documents are never packed.
Observed padding is 5.917% of allocated tokens, and the median actual length is 5,651 tokens.

The budget is 216,158,699,520 actual input tokens, including text and flanks and excluding padding.
Variable-length training uses the actual input-token clock, with 2,161,115,136 warmup tokens and immediate linear decay to 5% of peak.
The original 524,288-token reference batch and horizon anchor the 1x Hero optimizer heuristic; weight decay remains zero.
Track input DNA, input text, scored targets, consumed examples, FIM examples and RC examples separately.
Update count and row epochs are not substitutes for the input-token budget.

## Validation and diagnostics

The primary sample preserves the historical 512 row indices, lowercase character tokenization and metric name `eval/loss`.
Its 4,193,792 targets include 396,290 originally lowercase targets; uppercase `N` remains in the primary and unmasked partitions and is also counted separately.
Three diagnostic views reuse the same predictions: `eval/loss_softmasked`, `eval/loss_nonsoftmasked` and `eval/loss_weighted_10to1`.
The last metric is `(10 * unmasked_loss_sum + masked_loss_sum) / (10 * unmasked_count + masked_count)` and does not alter training loss.

An additional fixed 512-window panel, `eval/loss_disjoint`, covers all 65 species and has no recorded source-interval overlap with any train row, considering both strands and excluding touching endpoints.
This controls source-region overlap, not sequence homology.
The FIM panel reuses those same 512 held-out windows as 1,024 matched minimal/formatted presentations.
Each pair shares the exact DNA crop, gap, case, orientation and length hint; formatted members always include species and mask-availability text.
NLL is averaged per example; teacher-forced base accuracy is measured per target.
Autoregressive exact match uses actual caller-length generation, with a separate case-folded DNA exact match.
Metrics are stratified by presentation, short/long gaps, gap length and guided/random selection, with explicit example and target counts.
The fixed manifest marks up to two generation candidates per gap length; accelerator profiling will set the bounded generation budget, and exact-match denominators count only generated completions.

Evaluate initially, after warmup, every 5,242,880,000 actual input tokens, and at the final budget.
Parameter and gradient norms are collected every 1,000 updates: overall plus attention Q, router and expert-down projection norms in layers 0, 4 and 7.
The twenty scalars use logical global arrays, FP32 master weights and globally reduced, unscaled gradients before optimizer transforms.
No per-expert histograms are collected.

## Optimized runtime and ownership

The package depends on the optimized exp586 project at immutable commit `de5f428fc7b2bac8b6e40697f905c8b99ca3580d`.
The coherent Marin runtime is `0.2.224.dev40027069267`, with JAX `0.11.1` and the original optimized model/optimizer source.
The old Iris client missed the controller's build-date floor by one day.
This release satisfies that floor while retaining the Haliax API used by the optimized model; later releases remove that API.
The source model is `s3://marin-us-east-02a/marin/grug/rav-ladder-d768-v2/2026.08.18/checkpoints/step-11420`.
It is a fresh DNA adaptation from text-pretrained weights.
The optimized lowercase-only compact embedding shortcut must accommodate text and uppercase IDs before production use.
The H100 smoke must validate all four buckets, loss aggregation, diagnostics and checkpoint replay before full training.
Controller freshness and live capacity must be rechecked at dispatch.

`python -m exp593_dna.smoke --run-id dna-exp593-smoke-20261107-v1 --cluster cw-rno2a --nodes 1` is the bounded validation entry point, invoked through an explicitly placed Iris coordinator.
It performs four updates covering every context bucket, then an exact full-state checkpoint roundtrip and one resumed update.
Its input-token schedule is shortened for validation; the optimizer peak still comes from the full scientific horizon.
It uses complete BF16 compute replicas and the perf5 partitioned FP32 expert masters/momentum, with the generic embedding gather supporting metadata and both DNA cases.
Norms are exercised on every smoke update; production retains the approved 1,000-update cadence.
This entry point cannot run the full production horizon.

The November 7 one-node smoke passed all four buckets at global batch 64, recorded twenty finite norm scalars, and restored the complete checkpoint exactly.
One subsequent GPU update had FP32 master relative L2 variation of 0.0003148; update replay is therefore not bitwise deterministic.
The successful smoke used 1,369,620 actual input tokens over five updates.
See `artifacts/smoke-v1/result.json` for the checkpoint binding and [personal W&B run](https://forge.coreweave.com/api/wandb/eric-czech/marin/runs/dna-exp593-smoke-20261107-v1).

W&B belongs to the personal `eric-czech/marin` project on `https://forge.coreweave.com/api/wandb`.
Artifacts belong under `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/`.
See `operations.md` for the approved single-model placement and recovery policy.
