# exp582: d1536 MoE DNA adaptation

Compare language-pretrained and randomly initialized d1536 models under the plan in [MarinDNA #582](https://github.com/Open-Athena/marin-dna/issues/582).
Pretrained DNA character encoding, its 600-update training/replay validation, and GPU evaluation with synthetic CPU scoring/probes passed.
The bounded real-data pilot also passed for both saved checkpoints: 128 examples on each of the 20 original tasks and 2,112 maize AF variants, with both frozen probes and no training updates.
Validation results are recorded in [the logbook](../../.agents/logbooks/exp582-validation.md).
Both current conditions completed 600 finite H100 updates with pooled EP receiver capacity 32 and sender transport capacity 8; their last 200 recorded training updates logged zero expert drops in every layer.
Both checkpoint-replay diagnostics restore exactly, while subsequent live/live and live/restored updates show comparable numerical divergence.
The user accepted non-bitwise update reproducibility; exact checkpoint restoration remains required.
Both current conditions passed GPU likelihood/repeatability/batch-context checks and the synthetic scoring/frozen-probe pipeline with native Sonic inference.
These are bounded diagnostics, not evidence that production-scale training or exact update reproducibility has been validated.

## Runtime

Use Python 3.12 and uv 0.10.3 or newer, below 0.12.
The local lock was created with uv 0.11.31; CoreWeave's task image currently uses uv 0.10.3.
Python 3.13 required a failing source build of Marin's transitive `fasttext-wheel` dependency; Python 3.12 has the published binary wheel.
The initial dependency selection is the current coherent Marin release `0.2.141.dev37311598536`, inspected on 2026-10-05.
The CPU/GPU extras, PyTorch indexes, CUDA 13 dependencies, and aarch64 PJRT override follow current upstream Marin at `187a34fa46cfe8feedc9d4573a2886e293d1e643`.
FlashAttention is pinned to that upstream lockfile's `4.0.0b28`: b33 inserts a tensor argument before the stream argument and breaks Marin's positional backward-preprocessing call.
Marin's published wheel does not include the Grug experiment modules, so a source-pinned subset is packaged under `src/experiments/` with file hashes in `upstream-manifest.json`.
That subset uses Apache-2.0, whose complete text is in the repository's [LICENSE](../../LICENSE); original source notices are retained.
The model is the existing Grug implementation; experiment-owned adapters belong under `src/exp582_moe/`.

```bash
uv sync --locked --extra cpu
uv run --locked --extra cpu pytest
```

Use `--extra gpu` on authorized GPU workers.
Every CoreWeave submission uses `--priority batch --user eczech`; W&B uses the user's credentials with entity `eric-czech` and project `marin`.
Diagnostic retries query the last stored W&B history step before logging, preserving monotonic history even when the local SDK counter resets.

## DNA character encoding

Both conditions use one token per DNA character, lowercase normalization, and no automatically inserted BOS/EOS.
`pretrained_dna_tokenizer` splits every character before BPE lookup, preserving all 128,256 original vocabulary IDs, embeddings and output rows; no merge can cross a character boundary.
`language_tokenizer` still returns the original tokenizer for future ordinary-text evaluation.
Pretrained ambiguity letters keep their original vocabulary IDs; scratch keeps historical ambiguity-to-UNK behavior and vocabulary-only EOS.
Every training window yields exactly 8,192 tokens, and one corpus pass contains 21,615,869,952 input tokens.
New training checkpoints record the complete tokenizer SHA-256; replay and evaluation reject an encoding mismatch.
The earlier BPE canaries remain historical evidence and do not validate the new pretrained input distribution.
After relocation from a preempted east attempt to Reno, the new pretrained character canary completed 600 finite updates and exact tokenizer-bound checkpoint restoration.
Only its first two updates dropped assignments (13.2264% and 0.5830%); updates 3–600 logged zero drops.
The saved clock is exactly 39,321,600 input tokens; its six replay-control updates are separate.
The new pretrained character GPU evaluation and synthetic CPU scoring/probe checks passed against that checkpoint and tokenizer digest.

## Routing diagnostics

`python -m exp582_moe.routing_probe` diagnoses the current pooled expert-parallel path on H100s; Marin's existing dropless grouped-GEMM/FSDP path is available as a fallback.
It retains the d1536 architecture, one 8,192-base window per example, and each condition's tokenizer.
Its bounded warmup-to-peak schedule is explicitly diagnostic; production still uses the agreed token schedule.
Each run logs total and per-layer drops, expert-load concentration, real token counts, loss, LR, and step time to W&B and saves its final state under a separate TTL smoke identity.
The optional `--router-fp32` control preserves FP32 router weights and balancing biases through parameter and compute casts; all other parameters retain the native BF16 policy.
This numerical ablation uses the same native training step and EP kernels and is recorded explicitly in W&B configuration.
`--qb-update-rate` optionally damps the native global-histogram balancing thresholds; its default of one preserves the native update exactly.
`--peak-lr-multiplier 0` provides a zero-LR control while routing thresholds continue to adapt; MuonH/AdamH renormalization can still introduce tiny weight changes, so it is not an exact weight freeze.
`--transport-capacity-factor` can size sender buffers separately from receiver capacity when investigating highly concentrated routing.
The successful 600-update controls use `--capacity-factor 32 --transport-capacity-factor 8` with native BF16 and `--qb-update-rate 1`; the default 1.15 capacities remain available for reproducing the failed baseline.
These capacities accommodate concentration and require another fit/drop check when the production batch is selected.
The optional `--replay-control` compares two independent live copies with a checkpoint-restored copy over two additional updates each, after saving the primary endpoint.
It checks exact initial values, clocks, dtypes, and shardings, compares the live copies after each update, then releases one copy before restoring and comparing the two-update endpoint.
This retains at most two full states; the six control updates are recorded separately from the checkpoint's training clock.
After a diagnostic failure following a successful save, `--resume-replay-checkpoint` and `--resume-metadata-digest` recover only the replay stage from the matching endpoint, preserving its optimizer and data cursor without primary training updates or checkpoint writes.
Recovery also requires `--resume-reference-uri` and `--resume-reference-sha256` identifying an immutable copy of the original W&B configuration; mismatched routing, precision, resolved model/optimizer settings, reference token counts, and endpoint LR scalars are rejected.
This recovery starts from restored state and cannot recreate a comparison with the original live state lost in the failed process.

## Evaluation

The [evaluation branch](https://github.com/eric-czech/plantcad2/pull/2) owns the archived dense comparisons and the original PR20 scoring/probe implementations.

The final comparison also evaluates the untouched d1536 language checkpoint as a negative control. It uses the same character-bound language tokenizer as the DNA-trained pretrained model, with zero additional DNA training. Because this older checkpoint predates tokenizer fingerprints, evaluation accepts the missing fingerprint only for its exact checkpoint URI and pinned raw and canonical metadata digests.
This project pins that package in the optional `eval` extra and supplies the native model adapter in `exp582_moe.eval_runtime`.
The private evaluation repository requires authenticated package access; training environments omit that extra.
The adapter keeps one window per example, rejects truncation and expert dropping, and returns real-token log probabilities and hidden states.
Both current character-encoded conditions passed CPU/GPU evaluation and synthetic probe checks.
The first-batch repeated-forward and native-loss checks have zero and at most 8.35e-7 error, respectively, across both conditions; the first-request batch-context differences are zero.
Character encoding makes the four-candidate causal-prefix scorer exactly the A/C/G/T-logit softmax in both conditions.
Variant likelihoods retain full-vocabulary normalization, and each variant feature now covers exactly one base; both normalizations are exported for allele-frequency comparisons.

The evaluation smoke currently supports one eight-H100 node.
`python -m exp582_moe.eval_smoke` restores a digest-pinned smoke checkpoint on Marin's dropless inference mesh and evaluates separate synthetic windows.
It checks per-token likelihoods against native fused cross entropy, compares dense/blocked projection on identical hidden states, requires repeated-forward equality, measures batch-context parity, and exports token likelihoods plus base-weighted/variant-token features for the pinned CPU scoring package.
Its default inference backend is the existing H100-compatible `sonic` implementation; the scatter control exhibited non-repeatable BF16 forward outputs and is not approved for evaluation.
The adapter requests FP32 projection output to match native loss accumulation when weights are BF16.
Synthetic probe diagnostics do not replace the real benchmark's genomic-block cross-validation or constitute scientific results.

`python -m exp582_moe.real_eval` evaluates the pinned real-data pilot: 128 examples per original task and 2,112 maize variants, preserving archived genomic train/test memberships.
The evaluation project prepares original-format Parquet rows and reuses PR20 row-range loading, task formulas, aggregators, grouped ridge fitting, bootstrap intervals and saved probes.
The native adapter uses batches of eight and emits the original worker schemas in 64-row task chunks or 32-row AF chunks, bound to checkpoint, tokenizer, preparation and implementation hashes.
A hash-pinned evaluation wheel is read from CoreWeave storage; no private GitHub credentials are forwarded.
The AF adapter calls the original FP32 suffix reduction, and nucleotide IDs are resolved by actual character encoding.
Restarting the same identity validates and reuses committed chunks, while incomplete chunks are recomputed.
The legacy scratch checkpoint requires an explicit opt-in restricted to its exact URI and verified metadata digest because it predates tokenizer metadata.
The completed pilot passed likelihood/probability agreement, repeatability, batch independence, checksum-bound artifact collection, original task/AF reduction and saved-probe prediction replay.
A deliberate four-chunk pause/resume verified reuse without changing the input or implementation binding.
Both probes retain the original five-fold grouped CV and 1,000 block-bootstrap draws, with 1,408 train and 704 test variants from archived memberships.
The small sample and 600-update weights do not establish final model quality; interruption/resume timing also must not be extrapolated as steady-state full-benchmark throughput.
## Full training

`python -m exp582_moe.train --condition pretrained --lr-multiplier 1 --cluster cw-rno2a --nodes 8` selects one actual trial and dispatches its H100 gang.
Submit its CPU driver through Iris with the same target cluster, `--priority batch`, and `--user eczech`.
The six seed-zero trials cross pretrained/random initialization with LR multipliers 0.5, 1, and 2.
Both 32- and 64-H100 placements use a global batch of 64 separate windows and one native Hero optimizer/QB update, with no gradient accumulation.
The 128-H100 implementation preserves that batch using the existing Hero context axis of size two and node-local EP8; it does not change the 8,192-token window or combine examples.
Admission still requires the target qualification recorded in training Operations.
`python -m exp582_moe.placement_smoke` qualifies either selected condition on 64 or 128 H100s from its digest-pinned permanent peak checkpoint, using a separate W&B smoke identity and seven-day temporary output prefix.
It evaluates the 512 fixed windows, takes 25 native updates with the actual training schedule and data cursor, saves and restores the full state, checks exact local-shard SHA-256 hashes on every rank, and takes one post-restore update.
The restore uses an abstract exemplar after releasing the original state, avoiding two resident full states; the shard hashes never gather a complete global model array.
These diagnostic updates do not advance the six actual trials.
The shortened stage ends after 21,440 updates: 11,240,734,720 input tokens, approximately 0.52 corpus passes.
Warmup retains its original duration of approximately 5.62B tokens; update 10,720 uses peak LR and produces a permanent full-state checkpoint before linear cooldown.
Cooldown reaches 5% of peak LR on the final update.
The original 562.022B-token horizon remains the reference for Hero's peak-LR and epsilon heuristics; shortening the schedule does not recompute those values.
The baseline MuonH and Adam LRs are 0.00187806248 and 0.000433399034, with the agreed token schedule applied to both.
Historical dense results remain comparisons at different training exposures, not token-matched controls for this shortened stage.

The six resume sources and their metadata/configuration hashes are pinned in `src/exp582_moe/schedule_transition.json`.
Five trials resume before the nominal peak; pretrained 0.5× restores the earliest surviving DNA checkpoint at update 10,771, takes one peak-LR update, and permanently saves update 10,772.
Its cooldown is 52 updates shorter so all six retain the same final token budget.
This small, explicitly approved timing exception does not recreate the missing update-10,720 state.
The prior weights, optimizer moments, pending router state, and data/token clocks are restored without resetting.

The raw Parquet reader uses the original block-shuffle/mixture machinery and absolute occurrence augmentation, reads only each process's assigned windows, and validates object identity and row-group boundaries against `corpus.json`.
Committed checkpoints bind the complete scientific configuration, tokenizer, corpus, optimizer, pending QB state and exact data/token clocks; placement may change on resume.
Revised states and their immutable binding are written under `<original-checkpoint-root>/short-linear-v2/`; original checkpoints and bindings remain intact.
Recovery resumes from the latest committed revised state, or the pinned original source before the first revised save, and retains two temporary recovery saves plus permanent peak/final states.
Before saving, the primary process removes any incomplete checkpoint at that exact destination and synchronizes the gang; committed checkpoints are never overwritten.
This permits a resized gang to retry an interrupted save whose partial arrays used different chunk shapes.
Set `EXP582_CHECKPOINT_DEBUG=1` on the Iris driver to forward native crash tracebacks and checkpoint stage logging to all workers, without adding forced garbage collection or allocation tracing.
Each training row records `train/schedule_revision=2` under the existing W&B run ID.
The `run_progress` denominator changes to the shorter stage; that percentage jump is a reporting change, not newly trained tokens.
The first recovery save is after 25 updates on each process start, including after a restore, followed by saves approximately every fifteen minutes.
Use `--checkpoint-interval-seconds 300` for a five-minute interval when repeated interruptions warrant the extra checkpoint I/O; this operational setting does not change the scientific configuration or run identity.
Held-out loss uses 512 fixed, evenly spaced validation windows after update 25, every 2,000 updates, and at the two permanent boundaries; its drop fraction is logged alongside it.
See [training Operations](exp582_training_operations.md) for launch, monitoring and recovery policy.
