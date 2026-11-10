# Augmented d768 Grug MoE DNA adaptation

Train augmented d768 models for [experiment #593](https://github.com/Open-Athena/marin-dna/issues/593), starting from either the same text-pretrained checkpoint as #586 or its final 216B DNA checkpoint.
All implementation and artifacts belong to the permanent, unmerged [companion PR #594](https://github.com/Open-Athena/marin-dna/pull/594).

Experiment #593 also includes three full-state annealing branches from the old 1T run's 181.818B, 363.637B, and 545.455B checkpoints.
Their additional input horizons are fixed at 216B, 168B, and 120B, with annotation-only FIM ramping from 0% to 30%, uniformly lowercase DNA, no biological metadata, and preserved native optimizer history.
Use `python -m exp593_dna.anneal_train --trial <181b|364b|545b> --cluster <peer> --nodes <count>` through the normal Iris coordinator; bounded validation additionally requires a distinct smoke ID and `--validation-updates 64`.
Production remains explicitly gated while downsampled benchmark monitoring is prepared and validated.
See [annealing-operations.md](annealing-operations.md) for the new identities, ledger, qualification requirements, and placement policy.

The user has authorized preparation, single-node H100 validation, and full training when ready.
Data preparation and the single-node H100 smoke are complete.
The [text-initialized final model](artifacts/production-final-v1/result.json) completed 216,158,719,114 actual input tokens at update 626,011.
Its final fixed panels, 640 functional records, all 128 stored arrays and all 626 expected sparse norm checkpoints are verified; W&B finished and every GPU task plus both Iris jobs succeeded.
The [DNA216B-initialized final model](artifacts/dna216b-final-v1/result.json) also completed 216,158,719,114 additional actual inputs at update 626,011 after its [zero-replay full-state handoff](artifacts/dna216b-production-reslice-rno-n8-v3/result.json) to 64 Reno H100s.
Its final validation, all 128 stored arrays, 626 sparse norm points, eight successful GPU process exits and terminal Iris jobs passed the same acceptance gates.
The [paired results](artifacts/paired-final-v1/report.md) include both zero-update baselines and all eight quarter/final checkpoints, which match all eight input/example counters between conditions.
For DNA initialization, FIM NLL fell from 17.804113 to 1.086264 while primary causal NLL rose from 0.947501 to 1.112691; bare conservation AUROC fell from 0.645721 to 0.586044.
These frozen development diagnostics show a readout trade-off and do not establish downstream benchmark gains or isolate augmentation from initialization, prior DNA exposure and learning-rate differences.
Both training conditions are complete, their GPUs are released, and the final checkpoint is available to the separate evaluation task in [PlantCAD2 PR23](https://github.com/plantcad/plantcad2/pull/23).
The full fixed-panel accelerator evaluator passed on eight H100s in about 65 seconds, including all 36 greedy completions.
The four-node production loop passed sixteen bounded updates and native checkpoint discovery.
The four-node Reno validation passed all sixteen updates, all four buckets and the causal/FIM/functional panels, with zero routing drops and verified native checkpoint discovery.
The original production identity is `dna-exp593-augmented-d768-pretrained-seed0-v1` in personal `eric-czech/marin`; its 32→64 H100 handoff evidence is in artifacts/production-reslice-rno-n8-v1/.
Its later recovery on the ordinary 64-H100 entry point, including twenty fresh norm metrics after a scheduler interruption, is recorded in [the recovery evidence](artifacts/production-recovery-rno-n8-v1/result.json).
Its first 1,000 updates processed 339,790,830 input tokens with finite losses, zero routing drops and twenty finite parameter/gradient norm metrics; see artifacts/production-startup-v1/result.json.
The exact native scientific bindings and resume update zero confirm fresh initialization from the original text checkpoint.
The accelerator evaluator reads the immutable preparation bundle, rechecks historical membership and native-case FIM reconstruction, and scores raw token NLL with binary masks.
Its fixed generation panel contains two minimal/formatted pairs for each of the nine gap lengths (36 presentations); each completion is greedy, uses the full original vocabulary, and stops at the caller-specified length.
The `exp593_dna.smoke --evaluate-panels` option exercises all causal and FIM panels after the bounded training/checkpoint smoke.
An evaluation-only retry can restore that validation run's own full-state checkpoint with `--evaluate-checkpoint`.

## Run training

```bash
# Bounded validation uses a separate run and checkpoint namespace.
python -m exp593_dna.train --cluster cw-us-east-02a --nodes 1 \
  --run-id dna-exp593-loop-smoke-v1 --validation-updates 16
# Resume the existing text-initialized identity on its qualified full-state target.
python -m exp593_dna.train --cluster cw-rno2a --nodes 8 \
  --checkpoint-interval-seconds 600 --keep-temporary-checkpoints 6
# Independent DNA-initialized bounded validation, followed by its own production run.
python -m exp593_dna.train --cluster cw-us-east-02a --nodes 1 --initialization dna216b \
  --run-id dna-exp593-dna216b-loop-smoke-v1 --validation-updates 16
# Resume this existing identity from its native checkpoint on the qualified full-state target.
python -m exp593_dna.train --cluster cw-rno2a --nodes 8 --initialization dna216b \
  --checkpoint-interval-seconds 600 --keep-temporary-checkpoints 6
```

These entry points dispatch eight processes and eight H100s per node through Iris at batch priority.
Use the explicit Iris user and target configuration in [operations.md](operations.md), and forward current CoreWeave and personal W&B credentials without recording their values.
The two production identities are `dna-exp593-augmented-d768-pretrained-seed0-v1` and `dna-exp593-augmented-d768-dna216b-lrm0p5-seed0-v1` in `eric-czech/marin` at the CoreWeave endpoint.
The latter imports only effective learned weights from exp586's permanent `step-412290`, then starts with a fresh optimizer and step/input/example counters at zero.
Effective router bias is retained as part of the learned model; source optimizer moments, data cursors and learning-rate counters are never imported.
Both runs use the same augmentation policy, global batch, fixed panels, 216,158,699,520-input budget and 2,161,115,136-input warmup.
The DNA-initialized run halves both peak rates to Muon 0.0016599690210542552 and Adam 0.0003830697740894435.
The original text-initialized scientific bindings remain unchanged.
The DNA-initialized condition passed sixteen bounded updates and all fixed/functional panels on one East node and four Reno nodes, with zero routing drops and verified native checkpoints; see artifacts/dna216b-loop-smoke-v1/result.json.
These checks preserve the fresh optimizer and zero-based source import.
The DNA-initialized production run's first 1,000 updates have finite losses, zero routing drops and twenty finite norm scalars; see artifacts/dna216b-production-startup-v1/result.json.
Its first 339,790,830 input tokens exactly match the original text-initialized run's deterministic 1,000-update input prefix.
The DNA-initialized run reached eight Reno nodes / 64 H100s after full-state resident validation and a physically fenced handoff; see artifacts/dna216b-production-reslice-rno-n8-v1/result.json.
Its first 1,000 resumed updates preserve the same input cursor and have finite losses, zero routing drops and twenty finite norm scalars.
A separate 64→8 H100 recovery check restored the DNA-initialized production checkpoint and optimizer on one East node, then passed sixteen disposable updates and all fixed/functional panels; see [the recovery evidence](artifacts/dna216b-recovery-east-n1-v1/result.json).
The validation used a no-op tracker and was physically released without promotion or production writes.
Following repeated Reno interruptions, a separate prepared eight-H100 East allocation was promoted after the old writer's physical termination was verified; [the production handoff](artifacts/dna216b-production-reslice-east-n1-v1/result.json) preserves the same W&B identity and full state.
This temporary fallback kept training advancing while larger allocations were prepared.
The subsequent [graceful eight-to-64-H100 handoff](artifacts/dna216b-production-reslice-rno-n8-cp120-v1/result.json) preserved the exact step-137600 state without replay, passed the first 1,000 resumed updates and twenty fresh norms, and independently verified all 640 resume diagnostic records.
The later [checkpoint-policy handoff](artifacts/dna216b-checkpoint-policy-rno-n8-v1/result.json) retained 64 H100s and the exact step-142700 state without replay, verified another 1,000 updates / twenty fresh norms / 640 diagnostic records, and committed its first periodic save after 601 seconds of training with a 76-second native write.
Scientific bindings exclude placement and are read back before checkpoint discovery; resumed execution restores the latest complete optimizer/data state and preserves that run identity.
The [November 9 DNA216B preemption recovery](artifacts/dna216b-recovery-20261109-v1/result.json) preserved its original Reno dispatch and run identity; the unused East eight-node preparation was released without qualifying or being promoted.
A later [64→32 H100 East recovery qualification](artifacts/dna216b-recovery-east-n4-v1/result.json) passed seventeen disposable updates, all buckets and norm branches, and every fixed/functional panel.
That unused preparation was physically released after the original Reno writer recovered again; it created no W&B run and wrote no production state.
The [second November 9 DNA216B recovery](artifacts/dna216b-recovery-20261109b-v1/result.json) preserves the same dispatch and W&B identity after another priority eviction, including its restored cursor, first new save and independently verified resume diagnostics.
Its captured logged progress exceeds the prior peak, while the first new save is durable replay progress below that peak.
The [text run's four-node East recovery](artifacts/production-recovery-east-n4-v1/result.json) preserved its W&B identity and complete optimizer/data state after a physically fenced handoff, with fresh norms and independently checked resume diagnostics.
That record also excludes a later priority-interrupted DNA216B three-quarter save from benchmark handoff; its missing metadata prevents treating it as a completed milestone.
The [DNA run's four-node East handoff](artifacts/dna216b-production-reslice-east-n4-v1/result.json) subsequently restored its latest complete optimizer/data state and resumed updates under the same W&B identity after further Reno priority evictions.
Its independently checked resume panels cover all 124 fixed metrics and 640 functional records; the interrupted three-quarter checkpoint remains unavailable.
The November 10 [32-to-64-H100 Reno handoff](artifacts/dna216b-production-reslice-rno-n8-v2/result.json) queued and qualified the new gang while East continued training, then restored the exact step-486400 optimizer/data state after all old GPU tasks and both old jobs succeeded.
It preserves the same personal W&B identity with zero replay; the first 1,000 resumed updates, twenty fresh norms, all fixed/functional resume panels and a new native recovery checkpoint are verified.
The measured optimizer-update gap was 242.628 seconds, including save, exit verification, promotion, restore and resume evaluation.
The later [November 10 East recovery](artifacts/dna216b-production-recovery-east-n4-v2/result.json) restored step-536640 / 185,261,041,122 inputs after repeated Reno priority evictions produced no new updates for an hour.
The old root and child were terminal and a post-cancellation backend observation showed no old pods before the qualified 32-H100 allocation was promoted under the same W&B identity.
The first 1,000 resumed updates, twenty fresh norms, all 124 fixed metrics and 640 functional records passed verification.
Its first new native save at 185,913,633,685 inputs verifies durable replay progress below the preemption high-water mark of 186,122,874,586; 861,833,464 previously logged inputs require replay from the restored cursor.
The next [32-to-64-H100 handoff](artifacts/dna216b-production-reslice-rno-n8-v3/result.json) kept East productive while Reno acquired and qualified eight nodes, then restored the exact step-548800 state at 189,465,674,168 inputs after all four old GPU tasks and both old jobs succeeded.
It retained the same personal W&B identity with zero replay and a measured 274.012-second optimizer-update gap.
The first 1,000 resumed updates, twenty fresh norms, 124 fixed metrics, 640 functional records and a new native checkpoint at 190,358,758,813 inputs passed verification.
CPU rendering prefetches one batch while device collectives stay ordered on the main thread.
For resizing, add `--standby-id`, `--standby-checkpoint` and `--standby-from-job` to acquire a separate allocation while production continues.
The standby uses a no-op tracker, restores the existing production checkpoint, validates all buckets and diagnostics, and holds its prepared allocation behind an explicit promotion gate.
It writes only under its unique `placements/<standby-id>/` prefix until an operator supplies the readiness digest, exact checkpoint/clock and proof that both old Iris writer jobs are terminal.
Promotion discards all probe updates, restores the selected production cursor and resumes the same W&B identity without requesting a new gang.
Promoted allocations also accept an explicit checkpoint-and-exit request for subsequent handoffs; see [operations.md](operations.md).
On re-entry after promotion, immutable control evidence selects ordinary strict recovery from the newest production checkpoint, even after the original standby source and handoff checkpoint have been pruned.
Periodic checkpoints default to 30 minutes with two temporary recovery saves retained; warmup, quarter-budget and final checkpoints are permanent.
The interval and temporary retention are operational settings exposed by `--checkpoint-interval-seconds` and `--keep-temporary-checkpoints`.
The initial two-minute policy committed two native saves that each took 81 seconds, making checkpoint cost about 40% of a repeated training/save cycle.
The examples above therefore request ten-minute saves with six temporary checkpoints for subsequent ordinary placements, giving a newly written temporary checkpoint at least an hour of lifetime while reducing estimated checkpoint cost to about 12% at that measured write time.
The first ten-minute-policy save was read back, hashed and discovered natively; its 76-second write implies about 11% of a repeated training/save cycle, excluding pointer verification and pruning overhead.
Two-minute saves with thirty temporary checkpoints remain an emergency option when shorter preemption windows prevent less frequent saves from becoming durable.
These settings are excluded from scientific bindings and take effect when launching a new allocation; existing serialized jobs retain their configured policy.
A four-device CPU regression also restores the complete state across replica-group counts 1→2→1, checking exact values and the changed FP32-master/replicated-BF16 layout; actual GPU reslicing remains a separate runtime check.
The run stops after the first complete 64-example update reaching the nominal 216,158,699,520 input-token budget.
It records the actual input count and overshoot, which is less than 524,288 tokens (2.43 parts per million), preserving the minimum example length and equal per-example objective at the last update.
Padding never advances the schedule.

## Development diagnostics

The conservation development panel is frozen before any prediction in `artifacts/development-v1`.
Its 512 sites contain 256 published conserved and 256 neutral labels, matched by Sorghum chromosome and target base, with nonoverlapping 512-base contexts.
Selection uses odd Sorghum chromosomes from the upstream `Evolutionary_constraint/train.tsv`; these records are evaluation-only and never enter the language-model corpus or annotation sampler.
Every selected context matches reference GCF_000003195.3 exactly and avoids the conservatively expanded source regions from every original Sorghum training row.
The verifier checks all 44 original training metadata shards and all 2,638,656 row records against the immutable input manifest.
This establishes coordinate separation, not absence of homologous sequence in other training genomes.

The upstream file named `valid.tsv` is excluded: all 38,060 PlantCAD2 Andropogoneae benchmark windows contain its central 512-base contexts.
That audit read only the benchmark sequence column.
Published coordinates are one-based, with the labeled nucleotide at source sequence index 255; exact reference matching verifies the conversion to zero-based, half-open intervals.
The scoring protocol is frozen in the manifest: average bare left/right-RC true-base log-probability for conservation, with a separate minimal-FIM view, AUROC and average precision.
The junction panel contains 32 donor sites, 32 acceptor sites and 64 matched random controls from 64 existing held-out windows across 22 species.
The two-base source motifs are canonical, and fixed reverse complements produce both sequence orientations; the rendered minimal prompts reveal no feature class.
Random controls share the same DNA crop and orientation and stay more than 32 bases outside every annotated landmark neighborhood.
The source-row metadata, native-case DNA hashes and annotation bank are checked against the immutable preparation bundle before selection.
The native trainer scores these panels at initialization/resume, each quarter-budget milestone and completion, using the same allocation as training.
The custom inference loop reuses the validated Grug kernels because generic prediction loops do not accept this sharded model state or character-forced DNA encoding.
It uses BF16 model computation and an FP32 full-vocabulary projection/logsumexp; it never renormalizes probabilities over DNA tokens alone.
The conservation contexts are padded to a fixed 1024-token inference bucket; junction recovery uses the existing 8192-token decoder and generates both missing bases without gold prefixes.
Per-site predictions, metrics, input manifest and scorer implementation hashes are published as immutable content-addressed native objects before their W&B metrics are reported.
The primary `eval/loss` protocol and scientific training bindings remain unchanged.
The readout passes a compiled FP32 reference comparison and a separate BF16 tolerance check on a small synthetic model, including a partial final batch.
Both functional panels were evaluated throughout production, and the original run's first-quarter checkpoint and frozen prediction artifacts are independently verified in artifacts/production-quarter-1-v1/result.json.
The [text-initialized half-budget checkpoint](artifacts/production-quarter-2-v1/result.json) and [DNA-initialized first-quarter checkpoint](artifacts/dna216b-quarter-1-v1/result.json) have verified permanent metadata, complete fixed panels and independently checked 640-record functional artifacts.
The two first-quarter checkpoints have identical input, DNA, text, target, example, FIM and RC clocks at update 156,559 / 54,039,954,011 additional input tokens.
The [DNA-initialized half-budget checkpoint](artifacts/dna216b-quarter-2-v1/result.json) also has verified permanent metadata, fixed panels and all 640 functional prediction records.
Its complete clock matches the text half-budget checkpoint at update 313,074 / 108,079,388,453 additional inputs, including DNA, text, targets, examples, FIM and RC counts.
The [text-initialized three-quarter checkpoint](artifacts/production-quarter-3-v1/result.json) is verified at 162,119,176,962 additional inputs, including its exact historical fixed-panel row and all 640 functional prediction records.
The [DNA-initialized three-quarter replacement](artifacts/dna216b-quarter-3-v1/result.json) matches that complete clock and has verified fixed panels, 640 functional records and a complete 128-array storage readback.
It uses a fresh retry directory; the earlier interrupted base destination and its unsaved diagnostics remain excluded.
A separate storage readback decoded all 128 saved arrays (16,716,379,160 bytes), found finite values and update/optimizer counters at 469,566; it did not perform a new JAX restore.
These fixed development panels provide diagnostics; benchmark evaluation remains separate.
The large junction payload is native-only; the repository retains its manifest and four paired examples.

```bash
uv run --locked python -m exp593_dna.development \
  --source-root /home/exedev/tmp/mdna-exp593-development \
  --prep-root /home/exedev/tmp/mdna-exp593-data-prep \
  --inputs-manifest artifacts/prep-v2/inputs_manifest.json \
  --output artifacts/development-v1
```

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
The initial fresh-state eight-node Reno checks stalled; the autotuning-sharding workaround did not resolve the overall runtime problem and was removed.
The later resident standby validated full-state resume on eight Reno nodes; see artifacts/standby-rno-n8-v1/result.json.
The four-node Reno validation passed with the original optimized runtime settings and scientific bindings; see artifacts/loop-smoke-rno-n4-v1/result.json.
The old Iris client missed the controller's build-date floor by one day.
This release satisfies that floor while retaining the Haliax API used by the optimized model; later releases remove that API.
The original source model is `s3://marin-us-east-02a/marin/grug/rav-ladder-d768-v2/2026.08.18/checkpoints/step-11420`.
The original condition is a fresh DNA adaptation from text-pretrained weights; the second condition restarts adaptation from the final 216B DNA weights with a fresh optimizer.
The optimized lowercase-only compact embedding shortcut must accommodate text and uppercase IDs before production use.
The H100 smoke must validate all four buckets, loss aggregation, diagnostics and checkpoint replay before full training.
Controller freshness and live capacity must be rechecked at dispatch.
Use W&B and committed execution receipts to establish progress; an Iris pending response can lag actual execution, as documented in [the monitoring policy](operations.md#placement-and-monitoring).

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
See `operations.md` for the approved two-model placement and recovery policy.

## Full-state annealing continuations

The additional three runs continue the old 1T exp586 checkpoints using the immutable recipes in `src/exp593_dna/anneal.py` and entry point `python -m exp593_dna.anneal_train`.
These restore the entire native optimizer and model state, retain its lifetime counters, and track additional inputs with a separate stage clock.
All DNA inputs are lowercase, with no biological metadata; every FIM example is drawn from annotated original training windows.
The source corpus, annotation provenance, reverse-complement augmentation, variable-length examples, and per-example objective normalization remain explicit in the scientific checkpoint bindings.
See [annealing-operations.md](annealing-operations.md) for placement and recovery policy.

The compact primary panel preserves the historical 512 examples and `eval/loss`, with repeat/nonrepeat/10:1 partitions only for that primary loss.
FIM monitoring includes aggregate base accuracy, short-gap exact match, and suffix benefit.
Real-task monitoring freezes 2,048 examples per task, uniformly sampled across every source shard with natural class prevalence, for twenty benchmark tasks and maize allele frequency.
These are test-derived diagnostic panels, not independent development sets, and cannot change the fixed training schedules or training sampler.
All task input bytes and memberships are bound by `assets/anneal_task_panel.json`; they are never opened by the training sampler.

Both raw causal and minimal FIM views are scored on lowercase inputs.
Standard conservation/core/motif tasks fold uppercase and lowercase output mass before conditioning on ACGT, align RC positions and complementary base axes, and combine strands by normalized geometric mean.
Multi-base task answers use the established teacher-forcing convention; their motif accuracy is distinct from the separately generated short-gap FIM accuracy.
Minimal FIM reserves only its own structural-label overhead and trims remote flank ends, so its contexts differ slightly from the matched minimal/metadata contexts in the full eval suite.
SV uses the historical forward-only junction likelihood ratio and leave-one-base-out FIM; its lowercase raw view differs from the latest uppercase SV comparison plot.
AF uses its own source center at index 4096, becoming index 1024 in the centered 2048-base window and index 1023 on RC.
Its raw view retains the historical full-vocabulary suffix ALT-minus-REF likelihood ratio, while its FIM view uses case-folded site-only ALT/REF log odds.
One scalar per task/view is logged: conservation AUROC, core/SV average precision, motif accuracy, and AF Spearman correlation.
Production task panels run initially, at curriculum end, every approximately 20.97B additional inputs, and finally; compact validation runs approximately every 5.24B inputs.
Disposable validation uses a separately named 32-example-per-task pilot, with metrics under `validation/tasks`, never the production task curves.
Use `python -m exp593_dna.anneal_validation --trial <recipe> --cluster <cluster> --nodes <nodes> --run-id <new-smoke-id>` for new bounded 64-update checks.
This diagnostic wrapper suppresses only the optional in-process traceback timer after two native faults coincided with its first dump; use external Iris task profiles instead.
It rejects production configurations and reuses the native dispatcher, all validation checks, optimizer updates, and checkpoint roundtrip without changing scientifically bound training files.

The 545.455B legacy native checkpoint has 18,468,864 saved BF16 output-projection values that differ from the BF16 cast of its FP32 master, with maximum absolute difference 0.00146484375.
A complete CPU audit found that all other 35 parameter/master pairs match exactly and all model/master values are finite.
The user approved preserving both arrays verbatim for the initial evaluation and first normal optimizer update, retaining every optimizer leaf.
The exception is restricted to this exact source at stage zero, its audited mismatch count/magnitude, and agreement of every other pair; nonfinite state is always rejected.
The first normal update must produce a finite model matching its masters, and the source discrepancy is recorded in W&B import provenance.
