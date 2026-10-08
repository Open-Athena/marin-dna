---
topic: exp582-validation
issue: https://github.com/Open-Athena/marin-dna/issues/582
description: Implementation and runtime validation before the d1536 DNA experiment
author: eric-czech
---

# exp582 validation logbook

## Scope and stop condition

Implement the agreed d1536 pretrained-versus-random experiment and validate the runtime, data, initialization, optimizer, checkpoint/resume, and evaluation paths.
The user reviewed validation and authorized six actual seed-zero trials, then shortened their stage to 11.24B tokens on October 6.
Complete that stage with thirty-minute W&B-led monitoring; additional seeds and longer stages require later review.
The detailed working plan is `scratch/exp582_implementation_plan.md` in the experiment clone.
Use W&B `eric-czech/marin` with the user's credentials.
Every CoreWeave submission uses `--priority batch --user eczech` and one checkpoint writer per trial.
Validation results belong here and in chat; the evaluation PR OP remains brief and reserved for final experiment results.

## Validation index

| ID | Validation | State |
| --- | --- | --- |
| V1 | Iris v3 capacity helper, current uv dependencies, storage and W&B access | Passed locally and on H100 after documented dependency fixes |
| V2 | Data/tokenizer/augmentation contracts and token accounting | Full raw-window audit and character-token contract establish exact corpus counts; 512 augmented real-window encodings match direct original IDs |
| V3 | Model restore, positional behavior, padding and router parity | Full-model character-mode restore, 600-update H100 training and synthetic GPU evaluation passed in both conditions |
| V4 | Optimizer groups/heuristic and token-driven schedule | Heuristic, LR injection, and CPU schedule fork passed; production batch unresolved |
| V5 | Full-model batch-priority canary on an approved whole-node target | Earlier capacity-32/8 controls passed; fresh pretrained character-mode 600-update control passed on Reno after relocation |
| V6 | Save/resume equivalence and pre-cooldown continuation | Exact round trips passed; user accepts non-bitwise updates; new character-mode exact restoration and replay control passed |
| V7 | Variant scoring, strand handling, frozen probes, archived dense comparisons | Historical archive and both character-mode GPU/CPU synthetic checks passed; real benchmark pipeline remains pending |

## Decisions

- Use the freshest compatible Marin release set via uv; preserve experiment semantics explicitly rather than inheriting new recipe defaults.
- The v3 capacity helper must preserve freshness, reachability, and free-plus-held accounting checks and reject unsupported versions.
- The user selected H100s in response to the proposed validation scope; retain one gang at a time, at most 64 H100s and one hour per job, starting with one eight-GPU node where feasible.

## Entry log

### 2026-10-05 — Validation work authorized

- Starting MarinDNA commit: `45a125eed9cd25dd11ce8449649a202a552af1ad`; branch `codex/exp582`, independent of the exp472 experiment tip.
- Previous read-only checks: direct CW S3 reads of d1536 step 15128, the raw-data manifest, and the dense 0.22T checkpoint succeeded; the original 12 helper tests passed.
- Live fleet reads showed all three production peers on availability metric v3 while the helper still required v2.
- Upstream commit `595c5a0f112043d70d9bf563d3eff3990552a14c` defines v3: explicit priority ordering adds SYSTEM without changing the interpretation of reported free GPU counts.
- Current PyPI inspection reports the coherent Marin release `0.2.141.dev37311598536` for core, Iris, Levanter, Fray, Rigging, and Zephyr.
- Latest upstream Marin main inspected: `187a34fa46cfe8feedc9d4573a2886e293d1e643`.
- Next: validate the helper change against fixtures and the live response, resolve the experiment's dependencies, and establish the evaluation branch.

### 2026-10-05 — V1 helper tests and current-runtime findings

- Command: `uv run --locked pytest -q` at the MarinDNA root.
- Result: 151 passed, 1 skipped; the CW helper subset has 15 passing cases, including SYSTEM holds, unsupported versions, stale data, and accounting failures.
- The old exp472 Iris client decodes the newly added SYSTEM enum as an integer, so it cannot supply the v3 helper's string-band contract; use the current client rather than broadening the helper to reinterpret unknown enums.
- The current Iris release resolves with `--prerelease=allow`; the first import failed because this also selected an incompatible httpx 1.0 prerelease, while aiobotocore still imports `httpx.TimeoutException`.
- Add a documented `httpx>=0.28.1,<1` compatibility constraint while retaining the latest coherent Marin release.
- The current H100 ladder explicitly supports d512 and d768 only, using EP8 and pooled-wave communication.
- d1536 on H100 remains unvalidated; the proposed initial GPU validation target is GB200, matching the source run's hardware family.

### 2026-10-05 14:02 UTC — V1 live capacity validation passed

- Command: `uv run --no-project --prerelease=allow --with marin-iris==0.2.141.dev37311598536 --with 'httpx>=0.28.1,<1' python .agents/skills/run-training-sweep-cw/scripts/utilization.py --cluster marin --timeout-seconds 30`.
- Result: the v3 helper passed every strict check and reported 160/256 free H100s on `cw-us-east-02a`, 155/512 on `cw-rno2a`, and 64/832 GB200s on `cw-us-east-08a`.
- User steering: prefer H100s; use the current utilization helper.
- Decision: validate H100 first, initially one eight-GPU node, retaining the proposed one-gang/64-GPU ceiling and one-hour-per-job limit.
- The current Iris client understands SYSTEM-priority band names; no numeric-enum workaround is necessary.
- The latest Marin project resolves through uv, but Python 3.13 triggers a transitive `fasttext-wheel==0.9.2` source-build failure (`uint64_t` missing `<cstdint>` under the local compiler).
- Use supported Python 3.12 with published wheels while retaining all current Marin library versions.

### 2026-10-05 — Local runtime gates before the H100 canary

- Current project: `experiments/exp582_plantcad2_moe`; `uv run --locked --extra cpu pytest -q` passed all 11 tests.
- Current Marin core/Iris/Levanter/Fray/Rigging/Zephyr are coherently pinned to `0.2.141.dev37311598536`; JAX and JAXlib are `0.11.1`.
- W&B authenticates as `eric-czech` and can access `eric-czech/marin`.
- The old DNA tokenizer is serialized as BPE with an empty merge list; the new WordLevel character tokenizer matches its IDs for canonical/ambiguous and mixed-case inputs, retains lowercase normalization, and adds vocabulary-only EOS.
- The original language tokenizer has a BOS postprocessor; encoding explicitly disables added specials.
- Direct S3 reads of source checkpoint metadata, its 31,097-byte manifest, and the first training parquet succeeded.
- The current pretrained architecture matches every one of the checkpoint's 36 FP32 master-array paths and shapes, totaling 11,534,398,464 parameters.
- The 4K and 8K configurations have identical parameter shapes; no positional parameter resize is needed.
- One thousand real training windows in both orientations yielded BPE lengths 3,562–4,055, mean 3,822.688; this is a first-shard sample, not a corpus-wide audit.
- The copied augmentation policy matches the actual exp472 functions on 1,003 occurrence indices, including the 32-bit boundary.
- Reduced native-model execution with SConv, hybrid attention, latent experts, and histogram QB preserves valid hidden states and router statistics under right padding; capacity was deliberately high to remove dropping as a confound.
- A singleton CPU batch exposes JAX 0.11.1 sharding-annotation loss in hybrid KV broadcasting and segmented-attention metadata slicing.
- The local model patch pins the hybrid KV branch output; singleton segmented attention remains an upstream limitation, while the two-example test passes.
- The Hero source LR/beta2/epsilon values reproduce exactly; token-driven LR injection survives Optax updates and retains AdamH's norm-preserving behavior.
- The token clock uses exact host integers in committed checkpoint metadata, avoiding JAX int32 overflow at 216B tokens.
- A real TensorStore checkpoint round trip preserves all reduced-model/optimizer arrays and a 216B-scale token clock exactly.
- Transfer initialization must carry effective router bias into the first pending-QB state, because the native step always overwrites bias from pending QB; a fresh zero pending state would erase the source routing state.
- The upstream `restore_template_from` deletes source arrays; the experiment's replay-comparison restore constructs a non-destructive template instead.
- Next: a three-update, full-d1536 H100 canary with real windows, then checkpoint replay; keep all production trials unlaunched.

### 2026-10-05 14:38 UTC — First submission exposed an unnecessary uv constraint

- Snapshot `a78fb97`; published draft PR: https://github.com/Open-Athena/marin-dna/pull/583.
- `/eczech/exp582-random-canary-a1` was accepted at batch priority, then failed during the CPU driver's dependency setup before GPU allocation or W&B registration.
- Error: task image uv `0.10.3` does not satisfy project `required-version ==0.11.31`.
- Changed the tooling constraint to `>=0.10.3,<0.12`; all Marin/JAX dependency pins remain unchanged.
- Validation: `uv tool run --from uv==0.10.3 uv lock --check` and `uv tool run --from uv==0.10.3 uv sync --locked --extra cpu --no-dev` both passed.
- This is a diagnosed startup failure, not a preemption response; the next dispatch retains the same trial, W&B, and checkpoint identities.

### 2026-10-05 — H100 compilation failure and review follow-up

- `/eczech/exp582-random-canary-a2` initialized all eight H100s and constructed the full model, then failed compiling the first update.
- The first error is in the FA4 backward preprocessing call: `BlockArgument` has no `element_type`; subsequent coordination errors are fallout.
- W&B ended with no `run_progress` or completed update, so this is a failed validation despite its `finished` state.
- The dispatch was recorded as failed and the SQLite backup was uploaded and verified; no replacement is submitted until the compiler failure has a concrete diagnosis.
- Independent review found that child priority needed to be explicit, the uv tool constraint rejected the task image, routing counts needed assertions, and continuation needed a runtime branch test.
- Explicit batch priority and the uv constraint are fixed; the canary now verifies native routing-accounting invariants and logs drop counts.
- The continuation branch test remains outstanding.
- CI run `37326287505` failed on formatting and a missing final newline, both corrected locally.
- Installed the repository pre-commit hook; the complete `uv run --locked pre-commit run --all-files --show-diff-on-failure` passes, and the experiment's 11 tests pass.
- GitHub pre-commit run `37328107909` passed on published commit `0c1ad82`.
- Compiler diagnosis: FlashAttention `4.0.0b33` inserted optional tensor `mOlo` before `stream`; Marin's positional call therefore passes a CUDA stream where the new tensor is expected.
- Pin FlashAttention to `4.0.0b28`, the version in current upstream Marin's lockfile, whose signature matches the existing call.
- All Marin library pins remain at the current coherent release; this dependency exception has a concrete API-compatibility reason.

### 2026-10-05 — Full-model H100 updates and remaining validation failures

- With FlashAttention b28, the random canary completed three full-d1536 updates on one eight-H100 node; post-compilation updates took 0.880 and 0.840 seconds at eight windows per update.
- The checkpoint saved and restored, but comparison failed because the native memory-transfer helper was invoked eagerly with an abstract mesh.
- Move that transfer inside the compiled state comparison; the reduced checkpoint round-trip test exercises the corrected helper.
- Expert-assignment drop fractions were 88.9%, 27.3%, and 73.3%; finite loss is insufficient to approve this small-batch routing configuration.
- Source Hero discussion explicitly warns that fewer sequences and longer context increase dropping; its reported ladder dynamics were much lower than this canary.
- Production batch size and routing behavior remain unresolved; no production run is authorized by these updates.
- A CPU test using the actual MuonH, AdamH, and Adam groups now saves nonzero optimizer state, restores it, and verifies that cooldown and long-horizon branches agree at the fork and diverge at the expected LRs without rewarming.
- All 12 experiment tests pass; the full reduced-model CPU optimizer path also exposed an upstream singleton-mesh sharding limitation, so the continuation test uses small matrices for all three native groups.

### 2026-10-05 — Exact checkpoint restoration; replay gate failed

- Attempt a4 reached GPU replay comparison after the memory-transfer fix and found a maximum array difference of `0.00570900738`.
- Disable the background W&B tracker wrapper so the primary process reads the real resumed history offset and broadcasts it to all ranks.
- Attempt a5, source `c8ca1a1`, compares the saved and restored state immediately, before either path donates or updates arrays.
- Immediate restoration has maximum absolute error **0**, with matching token clock, array dtypes, and shardings.
- Repeating two updates from that identical state and the same batches produces the differences below.
- These measurements locate the divergence after restoration; they do not establish its numerical cause or acceptable tolerance.

| State group | Maximum absolute difference | Relative L2 difference |
| --- | ---: | ---: |
| BF16 parameters | 0.00105953217 | 0.00085249677 |
| FP32 master parameters | 0.00108105037 | 0.00044205820 |
| Optimizer | 0.00152832083 | 0.00121229293 |
| Pending QB router state | 0.00604896247 | 0.00197760900 |

- Completed-update count and real-token clock still agree exactly.
- Flush rejection evidence with `run.finish(exit_code=1)` before the rank barrier, preventing peer termination from killing the primary before W&B uploads the failure.
- W&B now correctly reports `failed`, phase `failed`, and `run_progress=0.6`: https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-random-smoke-v1.
- The saved checkpoint is step 1 under the canary's TTL prefix; there is no completed step-3 checkpoint.
- The canary uses eight examples and an accelerated smoke schedule; its drop rates and timings are not production-batch estimates.
- Initially suspected that fixed capacity over an 8K padded shape could grant BPE more capacity per real token; subsequent source inspection below resolves this concern.
- The earlier reduced padding test intentionally used high capacity and does not settle this production-capacity question.
- Independent review found no comparison or failure-reporting defect that explains away the replay differences.
- **Stopped GPU submissions at this unresolved numerical/routing gate, as requested; no production trials or pretrained GPU canary were launched.**
- All five dispatches are terminal; no active experiment job or checkpoint writer remains.
- Verified durable state backup: `s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/operations/validation/20261005T152936Z-5356c81515e7405fa5434368dc0ed669.sqlite`, 81,920 bytes, SHA-256 `ea030723730e89056f180ca0353f74c0b43796d9844ebfd636864ca5093bd4cd`.

### 2026-10-05 — Evaluation preparation and local verification

- Published draft evaluation PR: https://github.com/eric-czech/plantcad2/pull/2, branch `codex/exp582`; its brief OP contains no validation results.
- Archived the previously computed dense 20-task, allele-frequency, and frozen-probe results with immutable source revisions and file hashes.
- The loader verifies task coverage, row counts, revisions, and the historical probe split hash; it does not run dense inference or refit dense probes.
- Five evaluation-package tests pass, including exact character-scoring equivalence, BPE boundary retokenization, no added boundary tokens or truncation, strand handling, and base-weighted embedding pooling.
- The proposed BPE next-base scorer retokenizes each causal prefix plus candidate base, scores the divergent token suffix, and normalizes A/C/G/T scores.
- This scores canonical tokenizations, not a marginal over all tokenizations; it remains provisional pending protocol review, and no held-out MoE metrics were computed.
- Allele-frequency likelihood uses independently tokenized REF/ALT windows and averages both strands; already-computed full-vocabulary dense results are available as a comparator.
- The native CPU evaluation adapter matches direct model log probabilities and hidden states while keeping one genomic window per example.
- Independent review identified that non-addressable global arrays require `process_allgather(..., tiled=True)` in JAX 0.11.1; the adapter now requests that explicitly and bypasses gathering for locally addressable arrays.
- GPU evaluation, full pretrained restoration, and end-to-end frozen-probe integration remain pending at the stop boundary.
- All 13 experiment tests pass after the distributed gather correction; the full repository pre-commit checks also pass.
- Independent review verified the final gather fix and found no additional concrete defects; distributed GPU behavior remains untested.
- Mainline tracking for the reusable v3 helper fix: https://github.com/Open-Athena/marin-dna/issues/584.
- GitHub build, pre-commit, and selected test checks passed on canary commit `c8ca1a1`; the final documentation/evaluation-adapter snapshot receives a separate CI run.

### 2026-10-05 — Routing investigation reopened

- User authorized experiments to fix high dropping, with the existing EP backend prioritized for both conditions for three hours from 15:57:46 UTC; fallback investigation may begin after 18:57:46 UTC if EP remains unsuitable.
- Existing one-gang H100 scope, batch priority, `eczech`, W&B `eric-czech/marin`, and one-hour job limit remain in force.
- Direct W&B history verifies `moe/drop_fraction` values 0.8886730671, 0.2733597755, and 0.7334260941 for the random canary; these are real expert assignments, with zero padding assignments.
- No pretrained GPU measurement existed at that point; the screenshot provided by the user shows the random canary's three points.
- Direct S3 access to the source checkpoint still passes after refreshing the configured CW keys.
- Independent source review found no exp582-specific initialization, QB-sign, wave-capacity, or denominator defect.
- Initial routing concentration is a hypothesis: four base embeddings selecting roughly 32 of 384 experts could yield approximately 90% dropping with balanced capacity 1.15.
- The inherited router matmul emits BF16 logits before its FP32 cast, and the old three-step canary jumps almost immediately to peak LR; both are possible contributors to investigate explicitly.
- Historical references: [histogram QB ablations #8033](https://github.com/marin-community/marin/issues/8033), [global histogram and MoonEP #7891](https://github.com/marin-community/marin/issues/7891), [EP/FSDP design #8062](https://github.com/marin-community/marin/issues/8062), [ragged capacity campaign #8317](https://github.com/marin-community/marin/issues/8317), and [later Hero context measurements #9615](https://github.com/marin-community/marin/issues/9615).
- The later Hero's ragged launcher requires a patched aarch64 GPU runtime; the existing portable dropless FSDP implementation remains the H100 fallback, without altering the model's parameter structure.
- Added a bounded routing diagnostic with distinct raw windows, unchanged RC and tokenizers, per-layer drop/concentration metrics, and a labeled 100-update warmup followed by peak LR.
- This accelerated diagnostic schedule is for exercising routing and stability; the agreed production token schedule is unchanged.
- Initial acceptance target: after warmup, overall assignment dropping below 1% on average, no layer persistently above 5%, finite loss with stable routing, and successful final checkpoint writing.
- Strong concentration, oscillation, or numerical failures require additional diagnosis even if the aggregate threshold passes.

### 2026-10-05 — Pretrained routing baseline

- The first diagnostic driver failed before GPU allocation because an unused private evaluation Git dependency required authentication; move it to an optional `eval` extra instead of forwarding GitHub credentials into training.
- Source `fa7289a` restored the complete d1536 pretrained weights and effective QB state on eight H100s, then completed 200 updates with a 100-update diagnostic warmup and capacity factor 1.15.
- [W&B pretrained baseline](https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-pretrained-routing-smoke-v1) reports initial dropping of 87.02%, falling to a final-20 mean of 2.0457%, with worst-layer final-20 mean 4.6171%.
- Initial per-layer top-32 assignment shares exceeded 98%; all 384 experts were active by update 99, supporting initial routing concentration as a contributor.
- The final loss is 2.73358, all model/optimizer state is finite, and direct CW reads verify the step-200 checkpoint metadata and manifest with 6,116,544 real input tokens and 1,600 examples.
- Completion of this diagnostic does not meet the routing acceptance target: steady dropping remains above 1%.
- Current Levanter `ep_fixed_pooled_wave_all_to_all.py` sizes logical sender capacity from local valid assignments and logical receiver capacity from globally summed valid assignments; padded shapes only set physical buffer maxima.
- Thus the suspected extra BPE capacity from padding is not present in this implementation; existing runtime assignment-count checks also exclude padding.
- Next controls: the same scratch diagnostic, followed by a modest capacity increase if receiver overflow remains dominant.

### 2026-10-05 — Scratch routing baseline and numerical control

- The matching [scratch baseline](https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-random-routing-smoke-v1) completed 200 updates with finite state but final-20 average dropping of 68.8773%; the worst layer averaged 83.3159%.
- Thus the longer diagnostic warmup alone is insufficient for scratch, unlike the strong improvement in pretrained routing.
- Prepare independent controls for capacity and precision before changing the accepted training configuration.
- The optional `RouterFloat32Policy` preserves router weights and balancing biases through both native parameter and compute casts, producing FP32-typed router logits under the existing XLA matmul policy.
- This jointly tests router parameter, logit, and bias precision; it does not isolate those components or force IEEE-FP32 multiplication globally.
- Independent review approves this probe-only policy approach; the architecture, native optimizer step, and EP kernels remain unchanged.
- Replay investigation: installed Levanter selects `deterministic=False` for H100 FlashAttention backward; this motivates a live-copy versus restored-copy control but does not establish the source of the observed replay divergence.

### 2026-10-05 — Capacity-only control

- [Scratch capacity 2.0](https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-random-routing-cf2-smoke-v1), source `218b7cf`, kept the baseline BF16 policy, seed, data order, LR heuristic, and 200-update/100-warmup schedule.
- Both sender and receiver capacity factors increased from 1.15 to 2.0; final-20 mean dropping improved from 68.8773% to 53.7897%, still unacceptable, with worst-layer mean 72.4891%.
- All state remained finite, and direct CW reads verified the final step-200 checkpoint with 13,107,200 input tokens and 1,600 examples.
- The modest capacity increase is insufficient; the next independent control changes router precision at the original capacity 1.15.
- Added optional three-way replay after endpoint checkpointing: two independently buffered live states and one restored state take identical updates, with differences recorded after each update.
- Its local donated-buffer/checkpoint control passes; no numerical tolerance has been relaxed, and GPU replay-control execution remains pending.

### 2026-10-05 — FP32 router control did not solve dropping

- [FP32 router control](https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-random-routing-fp32-smoke-v1), source `9cdfef5`, completed 200 finite updates and a directly verified checkpoint at the original capacity 1.15.
- Its final-20 mean dropping was 69.4472%, with worst-layer mean 84.3124%; preserving router precision does not resolve the scratch failure in this test.
- Do not adopt FP32 as the proposed fix on this evidence; the next control returns to native BF16 and interpolates pending QB thresholds with rate 0.1.
- Damping changes the balancing update, preserving the architecture and native optimizer: 90% previous threshold plus 10% new global-histogram estimate, then the usual centering/application on the next step.
- An explicit zero-LR control is available, but independent review found that native MuonH/AdamH renormalization still produces tiny FP32 changes at zero LR (locally measured maximum about `3.4e-9`); this is not an exact frozen-weight control, and its documentation now says so.
- No zero-LR run has been launched or used as frozen-weight evidence.
- All 19 local tests and repository pre-commit checks pass before the damping control.

### 2026-10-05 — Damping rejected; expanded receiver capacity

- [QB rate 0.1](https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-random-routing-qb01-smoke-v1), source `23950d0`, completed 200 finite updates but worsened final-20 mean dropping to 91.0348%, with worst-layer mean 95.3448%.
- The step-200 checkpoint is directly readable; this candidate is rejected for training suitability.
- Revert to the native BF16 policy and undamped QB for the next control, increasing receiver capacity to 32 and sender transport capacity to 8.
- Prior histories show individual expert loads reaching roughly 33–47 times the balanced mean; modest receiver factors cannot accept those loads.
- This larger-capacity diagnostic tests whether the existing backend can accommodate concentration within H100 memory; its speed and remaining drops must be measured before adopting any recipe.
- The EP investigation still ends at 18:57:46 UTC if no suitable implementation has been established for both conditions.

### 2026-10-05 — Expanded capacity passes the short scratch check

- [Receiver capacity 32 / transport capacity 8](https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-random-routing-cf32-smoke-v1), source `d9156d2`, completed 200 finite updates with final-20 mean dropping of 0.006293% and worst-layer mean 0.082331%.
- The final loss is 1.31610; direct CW reads verify step 200 and 13,107,200 real input tokens.
- Runtime is approximately 2.02 seconds per update versus roughly 1.19 at the original capacity; accepting concentrated routing has a measurable cost.
- Independent review confirms unchanged routing/top-k/QB rules and combine weights; larger buffers accept more assignments and can change floating-point rounding, so bitwise parity is not implied.
- Receiver factor 32 is not a zero-drop guarantee; expert concentration within striped destination pools can still overflow it.
- Waves stripe valid assignments by destination rank, rather than contiguous sequence chunks; BPE padding alone does not establish the cause of the first-update spike.
- Proceed to 600-update checks for both conditions with native BF16, undamped QB, and the same capacity settings, followed by three-way numerical replay.
- Preserve routing implementation `8328c5d` for both longer controls while evaluation development proceeds separately.

### 2026-10-05 — Evaluation numerical and integration preparation

- PlantCAD2 `8c8152c` adds synthetic separate-window requests and saved-output checks for next-base probabilities, allele reversal, strand averaging, and fixed-alpha toy frozen probes.
- All six evaluation-package tests pass; independent review found no concrete defects and confirmed that these checks do not independently prove GPU likelihood alignment or represent benchmark results.
- A BF16 runtime comparison exposed a maximum log-probability discrepancy of 0.00015068 because the adapter rounded its projection to BF16 before casting to FP32.
- Requesting FP32 projection output makes the reduced model match native per-token cross entropy within the predeclared test tolerance of 1e-6; both FP32 and BF16 adapter tests now pass.
- Add a bounded full-GPU synthetic validation entry point using native dropless inference, a fused per-token loss oracle, measured batch-context parity, and digest-bound request/output artifacts.
- This changes evaluation only; training controls retain their existing source snapshot and EP backend.
- All 22 local experiment tests pass before publication; GPU evaluation remains pending.

- Independent review caught and fixed missing required W&B `step` arguments, nonfinite native-oracle/repeated-output acceptance, and an unsupported 64-GPU oracle batch.
- Evaluation is explicitly limited to one eight-H100 node, uses retry-aware synchronized W&B steps, and flushes shared numerical failures before allowing peers to exit.
- Nonfinite-oracle regression tests pass; the full local suite has 23 passing tests.
- A direct metadata audit of all 44 raw CW parquet shards found 2,638,656 rows; at 8,192 bases per row, the agreed stage target is exactly ten full character-token corpus passes.
- This metadata audit does not replace a corpus-wide content or BPE-length scan.

### 2026-10-05 — Pretrained 600-update training succeeded; replay exceeded memory

- The [pretrained capacity-32 long control](https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-pretrained-routing-cf32-long-smoke-v1) completed 600 updates with finite model and optimizer state.
- Drops were 11.4572% and 0.7468% on the first two updates, then zero through the subsequently observed history.
- Direct CW reads verify step 600, 18,351,925 real input tokens, 18,347,125 loss targets, and 4,800 examples.
- The following three-state replay diagnostic was OOMKilled (exit 137) in its 600-GB pod; W&B remained stale in the replay phase, so the full validation trial is not marked complete.
- Refactor replay to keep two full states: compare independent live copies after each update, release one, then restore and compare the two-update endpoint.
- This changes diagnostic execution only; the scratch long control retains the training implementation and hyperparameters from `8328c5d`.
- A separate restoration diagnostic can reuse the saved pretrained endpoint, but cannot recover the lost original live state for a direct comparison.
- Independent review approved the two-state lifetime ordering; it reduces memory pressure without guaranteeing the pod's peak memory.
- The replay-only entry point validates an immutable, digest-bound copy of the original W&B configuration against current routing, precision, model, optimizer, LR reference batch, endpoint LR scalars, and data cursor.
- W&B serializes enums by name and rounds the last float64 digits; configuration comparison accounts for that representation at relative tolerance `1e-14`, with zero absolute tolerance, without relaxing any training-state replay criterion.
- The original pretrained configuration matches the recovered recipe in local validation; capacity or optimizer changes are rejected by regression tests.

### 2026-10-05 — Full raw-data and tokenizer-history audits

- Direct CW reads scanned all 44 training shards and all 2,638,656 windows: every sequence is exactly 8,192 ASCII letters, with no nulls or invalid lengths.
- 188,313 windows contain non-ACGT letters; existing ambiguity handling is retained.
- This establishes the existing training mirror's content contracts and the exact ten-pass scratch budget; it does not independently establish byte identity with the Hugging Face source.
- All four published `marin-community/marin-tokenizer` revisions contain byte-identical `tokenizer.json` files, from November 2025 through March 2026, preceding the Hero run.
- Raw SHA-256 is `881c9c36c359e1617afef6f7583403567931b7b4f43f6552d2b2155a131650a2`; canonical tokenizer SHA-256 is `a43c95c64d7293d06f0c6c4713f95e0036738248f881315910c703307bdd2a2b`.
- This resolves repository revision drift; the original language run itself did not record a tokenizer artifact digest.

### 2026-10-05 — Correct reverse-complement probe fixtures before GPU evaluation

- Comparing the synthetic fixtures against archived Result 11 exposed an incorrect reverse-complement variant index: use 4,095 for a forward index of 4,096 in an 8,192-base window.
- PlantCAD2 `08d2e35` fixes the coordinate mapping and adds the actual FP32 strand-averaged `[REF, ALT - REF]` feature contract to the synthetic validation.
- The four variant-pair fixtures now exercise train-only preprocessing and a bounded toy probe as well as the 32-window readout check; neither produces benchmark results.
- Seven evaluation-package tests and 25 training-project tests pass, and independent review reports no remaining findings.
- Regenerated 52-request fixtures have SHA-256 `27fd0a2ee35a1ee9ac6ffa721686962fdd06471e320f29f2f370c509e6771b59` (scratch) and `71c68b2fffddd2cb7bcf76a9cae2301c9a40b3f45ddc77b822d5c098349cc0e0` (pretrained), with CW upload/readback verification.
- The earlier planned pretrained evaluation fixture was never dispatched and is superseded by the corrected v2 evaluation identity.

### 2026-10-05 — Scheduler preemption and resumed diagnostic logging

- The scratch 600-update control was preempted twice for higher-priority work; Iris restarted the existing dispatch, with no replacement or priority change.
- Its endpoint-only diagnostic checkpoint policy requires replaying the interrupted attempt from initialization; production still requires periodic recovery checkpoints.
- The resumed local W&B step was zero while the server's next step was 272, so early repeated updates were discarded; W&B progress resumed once new steps exceeded that history boundary.
- The retained early history and later history therefore span different physical attempts; final-window statistics and the final checkpoint clock are the relevant endpoint evidence.
- Add a server-history offset lookup to all three diagnostic entry points, with a shared failure signal across ranks instead of silently defaulting to zero.
- The live W&B API lookup succeeds, and regression tests cover a reset local counter, nonprimary ranks, and lookup failure; all 28 local tests pass.

### 2026-10-05 — EP works for both 600-update controls

- The [scratch long control](https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-random-routing-cf32-long-smoke-v1), training source `e3678a3`, completed 600 finite updates and the two-state replay diagnostic.
- The last 20 recorded training updates have exactly zero drops as logged, both overall and in every layer: scratch updates 581–600, pretrained updates 580–599.
- Pretrained update 600 completed with finite state and a verified checkpoint, but its routing row was not flushed before the replay failure; zero dropping at that update is not independently established.
- Scratch final-20 mean loss is 1.31072; pretrained's last 20 recorded training updates average 2.76521.
- These losses use different tokenizers and are not a scientific comparison between conditions.
- Direct CW reads verify scratch step 600 with 39,321,600 input tokens, 39,316,800 loss targets, and 4,800 examples; metadata SHA-256 is `c003ecade50776f2932184a7cef40ac42ecba3ece1e1967562673949f38a27bd`.
- The different input-token clocks reflect separate bounded 600-update diagnostics, not the production experiment's matched 216,158,699,520-token budget.
- Scratch's first layer remains concentrated: its final-20 top-32 assignment share averages 57.96%, with 362.55 active experts on average; layers 4–15 use all 384 experts.
- Larger buffers accommodate this concentration; they do not prove balanced routing or guarantee zero drops later.
- The existing pooled EP implementation is retained with receiver capacity 32, transport capacity 8, native BF16, and native QB; no training-backend fallback was needed within the three-hour investigation window.
- Scratch restoration matches all saved state exactly, including layout; subsequent updates are not bitwise reproducible even between two independent live copies.
- After two updates, live/live versus live/restored parameter relative L2 differences are 0.00089707 versus 0.00089513; master-weight differences are 0.00028279 versus 0.00028176.
- This is evidence of execution variability independent of checkpoint restoration, not proof of its precise cause or acceptance of a production reproducibility tolerance.
- Pretrained replay recovery and full-GPU synthetic evaluation remain to be completed before stopping for review.

### 2026-10-05 — Replay recovery buffer-lifetime fix

- Pretrained recovery restored the saved state exactly and verified matching layout, configuration, endpoint LR, and data cursor.
- Its first donated update failed because JAX reported an external reference to a buffer; the endpoint LR check retained a NumPy view of a pinned-host scalar.
- Copy those diagnostic scalar values and flush JAX runtime replay failures to W&B; neither change alters training mathematics or checkpoint contents.
- The local CPU donation check does not reproduce the pinned-host GPU failure, so GPU retry is required; all 28 existing project tests and pre-commit pass.
- The retry from `baa0ccf` successfully executed both live-copy updates, confirming that the scalar-copy fix removes this startup failure; restored-branch comparison is in progress.
- At the end of the three-hour EP investigation, both 600-update training controls have succeeded; the remaining work validates restoration and evaluation rather than continuing backend selection.

### 2026-10-05 — Pretrained replay recovery completed

- [Pretrained replay recovery](https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-pretrained-routing-cf32-replay-smoke-v1) completed with source `baa0ccf`; the NumPy-copy repair was confirmed on H100.
- The original step-600 checkpoint is unchanged and directly readable; no new primary training updates or checkpoint writes occurred.
- Restored values, clock, dtype, sharding, and layout match exactly.
- After two updates, live/live versus live/restored parameter relative L2 differences are 0.00025793 versus 0.00025931; master-weight differences are 0.00005261 versus 0.00005315.
- These similar differences support execution variability independent of restoration, but do not establish its cause or a production acceptance tolerance.
- Recovery begins with a restored endpoint, so it cannot compare against the original live process lost after the earlier replay OOM.
- Both replay controls are now complete as measurement procedures; exact numerical update reproducibility is not claimed.

### 2026-10-05 — GPU evaluation caught a native-loss mismatch

- The pretrained v2 evaluation completed all 52 finite, dropless adapter requests, then rejected native per-token loss parity: maximum absolute error 0.542883396 versus the unchanged 0.0002 bound.
- No evaluation output was accepted or used scientifically.
- The small BF16 test used the full-vocabulary reference CE branch; add a vocabulary above the 4,096-token block threshold to exercise streaming CE.
- That CPU check passes; independent review also checked frozen hidden states at the true 128,256-token vocabulary and found error at most 9.54e-7.
- The GPU diagnostic now compares dense and native blocked loss on the same saved hidden values, explicitly replicating the output head as native CE does, and repeats the identical adapter forward.
- This distinguishes projection/partitioning from forward-execution differences without changing the adapter, checkpoint, or acceptance tolerance.

### 2026-10-05 — Scatter forward variability localized; native Sonic evaluation control

- Source `ed04478` localized the pretrained mismatch: the same-hidden dense projection matches cached adapter logs exactly, and native blocked CE differs by only 7.15e-7.
- Repeating the identical adapter forward changes log probabilities by as much as 0.33118 and hidden values by 14.8125; the full native-loss discrepancy is 0.30565 on this attempt.
- This establishes forward nondeterminism on the scatter inference path, rather than a loss-indexing or projection discrepancy.
- Scatter combines with BF16 scatter-add; its accumulation order is a plausible cause, not yet isolated from the other GPU kernels.
- Try the existing H100-compatible `sonic` backend, which retains the grouped expert GEMMs and uses an FP32 gather-and-sum combination.
- This is an inference-only backend control; training keeps the validated pooled EP configuration.
- Retain all likelihood tolerances and additionally reject any difference on the repeated-identical-forward check.
- Use v3 evaluation run identities with the same corrected synthetic fixtures and source checkpoints, preserving failed v2 evidence.

### 2026-10-05 — Pretrained Sonic evaluation and synthetic probes passed

- Source `74aa17b`; run: https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-pretrained-eval-smoke-v3.
- The same saved step-600 pretrained checkpoint passed all 52 synthetic requests with the native Sonic inference backend.
- On the first batch of eight requests, repeated-forward differences are exactly zero for both hidden states and token log probabilities; the first-request changed-batch-context differences are also zero.
- On that first batch, native fused per-token loss and adapter likelihoods agree within 4.76837158203125e-7; same-hidden dense projection matches the adapter exactly.
- This resolves the observed evaluation variability with the existing Sonic implementation; it does not isolate the exact kernel cause in scatter.
- The pinned CPU evaluation package `08d2e35467e236324eca9d1810262397b2e534ed` passed nucleotide-candidate normalization/right-flank independence, LLR sign and RC symmetry, strand-aware variant pooling, and frozen whole-window/variant probe plumbing.
- Toy probes use synthetic labels and fixed-alpha fitting; these are implementation checks, not benchmark results or validation of the real blocked-CV pipeline.
- Direct CW S3 reads verified the source checkpoint metadata digest, request/output hashes, and evaluation metadata hash; the immutable CPU receipt also records the evaluator commit, backend, numerical checks, and binding digests.
- Output SHA-256: `bdbadee11d03c0a9dc8a8689d1d05b6fb0f90b26c47b4f335391e25d151088a0`.
- CPU receipt: `s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp582_plantcad2_moe/exp582-plantcad2-d1536-pretrained-eval-smoke-v3/evaluation/cpu-validation-19429e7add1c2e1829b039b8f60df2c99003227d855dd42710eef8d636a7ae26.json`.
- W&B reports finished and progress 1; the exact Iris dispatch is terminal and succeeded.

### 2026-10-05 — Scratch evaluation passed; validation stopped for review

- Source `74aa17b`; run: https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-random-eval-smoke-v3.
- All 52 synthetic requests passed with separate character-token inputs within the 8K ceiling and native Sonic inference: 44 full 8,192-token windows plus eight 4,097-token causal prefixes.
- Native-loss, same-hidden projection, and repeated-forward maximum differences on the first batch are all exactly zero, as are changed-batch-context differences for the first request.
- The same pinned CPU package passed the nucleotide scoring, strand/sign, feature pooling, and synthetic frozen-probe contracts.
- Output SHA-256: `f9bf3fbeddb7c01b0622e35f90277cda31d66b9106e3369a22914f1f7f414af4`.
- CPU receipt: `s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp582_plantcad2_moe/exp582-plantcad2-d1536-random-eval-smoke-v3/evaluation/cpu-validation-08c443765a8a7ed2efd624a8a063470a5b85bb6814e8a5f9bbac17f67970a6f3.json`.
- Both GPU runs finished with progress 1, both immutable CPU receipts were uploaded/read back with SHA verification, and both exact Iris dispatches succeeded.
- No further GPU validation or production training is being submitted; the requested evaluation stop condition is reached.
- Retain pooled EP for training with receiver capacity 32 / transport capacity 8, native BF16 and QB; use native Sonic for inference after rejecting the scatter control.
- These larger EP buffers accommodate routing concentration rather than demonstrating balanced expert usage; their memory and drop behavior must be checked at the eventual production batch.
- Exact checkpoint restoration passed, but live/live and live/restored updates diverge similarly; no exact-update reproducibility claim or relaxed acceptance tolerance has been approved.
- Remaining production work: full BPE stream audit, batch/accumulation and whole-window token-budget boundaries, production data/resume/checkpoint orchestration, and scientific BPE scoring/real blocked-CV evaluation review.
- Local verification: 29 experiment tests pass, all repository pre-commit hooks pass, and published Sonic source/receipt helpers passed independent review.
- Historical dense results were only copied/read; the synthetic evaluation artifacts are not final experiment results.

- Final SQLite integrity check passed: zero active dispatches, zero submitted GPUs, and zero unreconciled dispatch intents across 20 dispatches.
- Five failed or superseded logical trials remain unfinished as historical evidence; the stop event explicitly prohibits automatically resubmitting them.
- Final immutable backup: `s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/operations/validation/20261005T193541Z-f200d0e6af9a40699325749ad16c0d24.sqlite`, 151,552 bytes, SHA-256 `b7ba33d337f5cf91a0df994d1c9090f8cc73a9d49c2921952b995a7d127c6f8b`; upload and direct readback verified.

### 2026-10-05 — Character encoding authorized; validation reopened

- User accepted non-bitwise-identical updates; exact checkpoint restoration remains a hard requirement.
- User approved character-level DNA encoding for the pretrained condition using the existing 128,256-token language vocabulary and instructed continued validation.
- `pretrained_dna_tokenizer` preserves every vocabulary ID and the decoder, adds lowercase normalization, splits every character before lookup, and removes automatic BOS insertion.
- All ASCII letter IDs and offsets match direct original-vocabulary lookup; `a/c/g/t` remain `64/66/70/83`, and EOS remains available at `128001` without insertion into data.
- The original `language_tokenizer` remains separately available for future text evaluation.
- Training rejects any window that is not exactly 8,192 tokens; full windows remain separate, with no packing.
- Together with the completed full raw-corpus content audit, this establishes 21,615,869,952 input tokens per pass and 216,158,699,520 for ten passes in both conditions; a variable-length BPE stream audit is no longer applicable.
- Existing ambiguity handling remains explicit: pretrained uses original letter IDs while scratch maps noncanonical letters to UNK.
- New checkpoints and W&B configuration record the tokenizer digest; replay and evaluation reject mismatched tokenizer artifacts.
- Local validation: 31 tests passed, including vocabulary/offset preservation, serialization, both RC choices, no inserted specials, accidental-BPE rejection, and checkpoint tokenizer mismatch rejection.
- Reopen one eight-H100 gang at batch priority under eczech, with new identities and a 21:30 UTC cutoff; repeat fresh pretrained training, replay and evaluation, then stop for review.

### 2026-10-05 — Character-mode data/scoring preflight passed

- Direct CW S3 reads reverified the original language checkpoint metadata digest `7198833f20fd9d1f442e6d0d9ef9738e634d674ff225ed276e7b56d71cf60227` and manifest before the fresh canary.
- On 512 augmented real windows, encoded IDs exactly match original-vocabulary character lookup, with 8,192 input tokens and 8,191 loss targets per window.
- The tokenizer has 128,256 entries and SHA-256 `d1cf7a63f3e0e0778a7ef137020e6f39d04be1e6da14fe22d6283eb36cddae5e`; its serialized artifact and the new 52-request fixtures were uploaded to CW and directly read back.
- Preflight receipt: `s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp582_plantcad2_moe/evaluation-fixtures/pretrained-char-preflight-6e02ec31989151e3ec7b43e4edae97cdee7146d5c3963e051e8332645055bda1.json`.
- Character tokenizer artifact: `s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp582_plantcad2_moe/evaluation-fixtures/pretrained-char-tokenizer-d1cf7a63f3e0e0778a7ef137020e6f39d04be1e6da14fe22d6283eb36cddae5e.json`.
- Pin evaluation package `8a30e9d8fa73d2418f88d8573af91bbcf47f9e67`; all eight package tests pass, including the original pretrained base IDs in a 128,256-way output vocabulary with differing non-DNA probability mass.
- That test verifies both the conditional four-logit base softmax and full-vocabulary ALT-minus-REF likelihood; normalization rules remain distinct and explicit.
- Independent reviews cleared the tokenizer/checkpoint changes and evaluation package; the CPU receipt helper was updated to select the character tokenizer and preserve separate run-qualified receipts.
- Updated issue #582's Implementation section to reflect character encoding, equal ten-pass exposure, and the resulting direct nucleotide scoring.

### 2026-10-05 — Character-mode optimizer parity and published review passed

- Compared the live pretrained-character W&B configuration with the completed scratch control and recomputed both through the pinned Hero heuristic at 65,536 real input tokens per update.
- Both resolve to MuonH LR `0.0006639953578875058`, Adam LR `0.00015322969797403982`, beta2 `0.999499874937461`, epsilon `2.8335610496006187e-14`, zero weight decay and no clipping.
- This verifies the bounded diagnostic's settings; it does not choose the production batch or replace the agreed production token schedule.
- Independent review cleared published DNA commits `665a6c7`/`7e2a1de`, evaluator `8a30e9d`, and the run-qualified CPU receipt helpers.
- The new GPU run uses source `665a6c7`; `7e2a1de` pins the reviewed evaluator without changing training.
- Local checks passed: 31 experiment tests, eight evaluation-package tests, and repository pre-commit hooks.
- GitHub's first quality run for `7e2a1de` never obtained a hosted runner and executed no check steps; one retry was requested.

### 2026-10-05 — Partial pretrained-character GPU evidence

- Source `665a6c7`; run: https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-pretrained-char-routing-cf32-long-smoke-v1.
- The fresh original-language initialization produced 83 logged finite-loss updates, consuming 5,439,488 character tokens before a batch preemption.
- Expert-assignment drop fractions were `0.13226401805877686` at update 1, `0.00583040714263916` at update 2, and zero at every logged update 3–83.
- Initial loss was `11.36540699005127`; the last 20 logged updates averaged `1.3189799904823303` loss and `2.151351797254756` seconds per update.
- This is partial training evidence only: the 600-update endpoint, final full-state finiteness, tokenizer-bound checkpoint/replay and new pretrained GPU evaluation have not passed.
- No replacement was submitted for the preemption; the existing Iris dispatch retains recovery ownership and the recorded validation cutoff.
- Both GitHub quality attempts for `7e2a1de` failed to obtain a hosted runner before executing any steps; this is separate from the passing local pre-commit checks.

### 2026-10-05 — Reno relocation after capacity review

- The user directed attention to free Reno H100s after prolonged absence of W&B progress in east.
- Verified the current eight-H100 resource profile and CoreWeave shared storage contract; direct source metadata/manifest/raw-shard access passed and no new step-600 checkpoint existed.
- Published operations revision `c91de91` permits capacity-backed early relocation and requires five-minute availability checks during active validation; the renewed pass cutoff is 22:30 UTC with one-hour jobs unchanged.
- The original root and GPU child were both verified terminal before replacement; the replacement preserves the trial's W&B/checkpoint identity.
- Because this bounded diagnostic writes only its endpoint, relocation restarts from original language weights rather than claiming a resume from update 83.
- Independent review requested making the already-performed child termination check explicit in the policy; that wording is now required.

### 2026-10-05 — Pretrained character training and checkpoint/replay passed

- The relocated canary completed on eight Reno H100s at batch priority under eczech; both its driver and GPU child succeeded.
- Executed snapshot `c91de91` retains training implementation `665a6c7`; run: https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-pretrained-char-routing-cf32-long-smoke-v1.
- The completed attempt contains 600 distinct updates at W&B history steps 84–683; old-attempt duplicate history rows are excluded from these statistics.
- All 600 losses and the final complete model/optimizer state are finite.
- Only updates 1 and 2 drop expert assignments (`0.13226401805877686` and `0.00583040714263916`); updates 3–600 have zero overall drops, and the final 200 updates have zero drops in every layer.
- Last-20 mean loss is `1.2880720376968384`; mean step time is `2.1397647559875623` seconds at the diagnostic batch of eight.
- The step-600 checkpoint records 4,800 windows, 39,321,600 input tokens and 39,316,800 loss targets, with tokenizer digest `d1cf7a63f3e0e0778a7ef137020e6f39d04be1e6da14fe22d6283eb36cddae5e`.
- Direct CW reads verified metadata SHA-256 `773bd56b9bc6d3ecebda6fd83026ecaed9b773ea091aa4407135d6b11b6cf8ee` and the 128-array manifest at `s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp582_plantcad2_moe/exp582-plantcad2-d1536-pretrained-char-routing-cf32-long-smoke-v1/checkpoints/step-600`.
- Initial restored/live-copy maximum differences are exactly zero, with matching layouts and clocks; all six additional replay-control updates passed finite-state checks.
- After two updates, master-weight relative L2 differences are `8.93593969522044e-5` for live/live and `9.01806852198206e-5` for live/restored; execution variation remains comparable under the user's accepted non-bitwise policy.
- The six replay-control updates do not advance the saved primary token clock.
- Proceed only to synthetic GPU evaluation/scoring/probes, then stop for review; production batch and real benchmark integration remain separate gates.
- GitHub checks for `82662cc` are green, including pre-commit, build and the selected test workflow.

### 2026-10-05 — Pretrained character evaluation passed; validation stopped

- Executed source `82662cc` on eight Reno H100s at batch priority under eczech; run: https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-pretrained-char-eval-smoke-v1.
- All 52 synthetic requests passed: 44 full 8,192-token windows and eight 4,097-token causal prefixes, each kept as a separate example.
- Native-loss and same-hidden native/projection maximum errors on the first eight requests are `8.344650268554688e-7`; repeated-forward log-probability and hidden-state differences are exactly zero.
- Changed-batch-context log-probability and hidden-state differences for the first request are exactly zero.
- The pinned evaluation package `8a30e9d8fa73d2418f88d8573af91bbcf47f9e67` passed CPU base scoring, full-vocabulary likelihood, strand/sign, feature pooling and synthetic frozen-probe contracts using the original pretrained nucleotide IDs.
- Output SHA-256: `710bcff5ca9eb075bb1f21e49df69fc68e894105018d437b4464f522f2265dc9`.
- CPU receipt: `s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp582_plantcad2_moe/exp582-plantcad2-d1536-pretrained-char-eval-smoke-v1/evaluation/cpu-validation-3a5a9cd955d436d8aa8c53a07318ff4e9df5a818484a99e7898a5bd3f884c2bc.json`.
- Direct CW upload/readback verified the receipt, checkpoint digest, tokenizer digest and exported artifacts; W&B reports progress 1 and `validation/cpu_contracts_passed=true`.
- The exact Iris driver and GPU child both succeeded, with no failure or preemption in the evaluation dispatch.
- No new scratch run was necessary because its character tokenizer and training/evaluation paths were unchanged; its earlier 600-update and native Sonic evaluation controls remain the matching validation evidence.
- Retain pooled EP receiver/transport capacities 32/8 for training in both conditions and native Sonic for inference; production batch sizing must still check buffer fit and drops.
- The synthetic fixtures do not validate the real 20-task/allele-frequency/probe pipeline or genomic-block cross-validation and are not final experiment results.
- Remaining work before production: batch/accumulation selection with exact heuristic recomputation, whole-window stage boundaries, production data/resume/checkpoint orchestration and real benchmark integration.
- Validation is stopped for user review; no production training was launched, and historical dense results were not recomputed.
- SQLite integrity passed with zero active dispatches, zero submitted GPUs and zero unreconciled intents across 23 dispatches; 12 logical trials are completed and five failed/superseded trials remain historical, with explicit no-resubmission instructions.
- Final immutable backup: `s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/operations/validation/20261005T220420Z-74631fbab1914286b1935cf2755cc80a.sqlite`, 159,744 bytes, SHA-256 `756eb8a7b18c60281701dd64aecbbedcb457ed025480bfd860cee0d901105b5b`; upload and direct readback verified.

### 2026-10-05 — Real benchmark pilot authorized and prepared

- User authorized evaluating both existing step-600 checkpoints on 128 examples per original task and approximately 2,000 maize AF variants, including frozen probes; no training updates.
- Pinned task data revision `d340debe0c8402c84f0696cd2002f87c2f7ba6db` yields 2,560 examples across all 20 tasks, sampled from four separated row groups per task with seed 582.
- The 2,112 AF variants contain 1,408 training and 704 test rows, preserving archived exp472 split membership and verifying reference alleles and train/test window separation.
- Archived dense inputs/split definitions are read only; no dense predictions or fitted models are recomputed.
- Request manifest SHA-256 is `58d5f6f0e644b4188f39421037fe07413f9eee856d11eb9d6195db67bc2b87a4`, with 13,568 independent 8,192-character inference windows.
- Pinned historical metric fixtures revealed the exact five-base SV boundary convention; parity tests cover that convention, ambiguous bases, motif accuracy and arithmetic-mean core scores.
- Evaluation package source is `f60b48ccaf6699fa939dadeb16a3d1d446d80580`; its 12 CPU tests pass.
- The native pilot runner verifies separate CPU/GPU probability calculations and exports immutable chunks bound to checkpoint, tokenizer, manifest and implementation hashes for an intentional resume test.
- Runtime inference, interruption/resume and real-data probe results remain pending at this entry.
- First real-data dispatch `random-real-eval-a1` failed before inference because W&B initialization timed out on its API request; both the driver and GPU child were verified terminal.
- W&B had created a run record but no progress or validation phase, so its apparent running state did not establish completed work.
- The runner now preserves the original startup exception when a tracker is not yet initialized; a targeted regression test covers this secondary error-masking bug.
- No evaluation chunks were produced; retry retains the logical run identity and four-chunk pause test.

### 2026-10-06 — Reuse PR20 evaluation and isolate telemetry startup

- The user challenged the whole-benchmark JSON expansion estimate and required reuse of the old plantcad2 evaluation process.
- Replaced the custom real-data metric/probe implementation and sequence-request format with copied PR20 source at `475fba58f81b79efa234d0018eb0776d1a960bfd`, original Parquet rows, row-range plans and original worker schemas.
- Only inference dispatch, native timing/memory metadata, module imports, shared suffix reduction and an optional archived pilot split require changes in the copied source; the evaluation branch records file hashes and reasons.
- Task formulas, SV boundaries, strand selection, full-task metrics, grouped ridge fitting, 1,000 genomic-block bootstrap draws and estimator serialization use the old implementations.
- Exact input membership/sequence/SV-coordinate parity passed for all 2,560 task rows and 2,112 AF variants; archived split and source hashes are unchanged.
- The original task/input/partition tests plus evaluation-package tests pass (23 tests); the training project passes 35 tests, including eight-device CPU sharding with full four-column SV probabilities.
- New artifact chunks contain only task predictions or AF predictions/embeddings, not placeholder hidden features for ordinary tasks.
- All three earlier real-evaluation GPU attempts failed before model inference during W&B initialization, on both Reno and east; all roots/children are terminal and no evaluation chunks or real metrics were produced.
- CW CPU authenticated GraphQL and SDK initialization succeeded; local Levanter tracker initialization also succeeded.
- After GPU termination, the empty random v1 W&B record rejected `resume=must` but initialized successfully with `resume=allow`; this CPU diagnostic logged no inference progress or scores and does not establish the GPU startup cause.
- The GPU environment now explicitly forwards the user's W&B key instead of relying on parent environment inheritance; no secret values are logged.
- Validation remains limited to the two existing step-600 checkpoints, batch priority/eczech and one eight-H100 gang; no training.

### 2026-10-06 — Original-protocol GPU canary reaches model restoration

- Source `0b1b0e6` and evaluation package `ee4eb4f1618bd70c0140f9de24a07431f055722e` passed independent published-diff review; 35 training-project tests, 23 evaluation tests and all CI checks passed.
- The exact CPU telemetry preflight authenticated as eric-czech and initialized both v2 run identities with progress zero and no inference.
- Reno GPU attempt `random-real-eval-v2-a1` then initialized W&B successfully and restored the step-600 checkpoint.
- It failed in the new native inference leaf when JAX could not infer the sharding of a four-column gather from the FSDP-sharded output head; W&B retained the original error.
- Both root and child are terminal; no evaluation chunks were committed.
- Strengthening the eight-device CPU test with the checkpoint's parameter sharding reproduced the exact error before the fix.
- The adapter now explicitly replicates only the four selected DNA output columns for the inference leaf and CPU oracle; no task formula or checkpoint weight changes.
- Earlier W&B timeouts are not assigned a proven root cause: explicit key forwarding and CPU initialization preceded the successful GPU connection.

### 2026-10-06 — Original-protocol canary and archived split checks pass

- Source `306056e` completed four original-format chunks (256 real task rows) on eight Reno H100s at batch priority under eczech.
- Repeated and changed-batch-context predictions are exactly equal; direct DNA projection agrees with the independent CPU projection within `3.8743019104003906e-7`.
- Direct CoreWeave reads verified all four receipts, prediction checksums, row counts and checkpoint/input/source bindings.
- The deliberate four-chunk pause ended both root and child successfully; resume keeps the same run identity and implementation binding.
- Original five-fold inner CV on the actual pilot split yields train/validation sizes 1126/282, 1127/281, 1126/282, 1127/281 and 1126/282, with disjoint genomic groups and no held-out test rows.
- A direct chromosome/position check also found zero overlapping 8,192-base train/validation windows in each of these five pilot folds.
- GitHub pre-commit, build and selected tests for `306056e` passed.

### 2026-10-06 — Scratch real-data evaluation passes end to end

- Source `306056e` and evaluation package `ee4eb4f1618bd70c0140f9de24a07431f055722e` evaluated the existing scratch step-600 checkpoint without training updates.
- All 106 original-format chunks completed; four task chunks were reused exactly after the deliberate pause.
- Both root and child succeeded, and all artifact receipts, input/checkpoint/source bindings and array checksums passed direct CoreWeave readback.
- The original reducers computed all 20 tasks (128 rows each) and AF correlations for 2,112 variants; both original grouped-CV probes fitted 1,408 training rows and evaluated 704 held-out rows.
- Reloaded serialized probe predictions exactly match saved predictions; independent saved-output review checked all task metrics, AF alignment, finite features, train-only scaling/CV and historical memberships without refitting.
- Pilot task group scores: conservation `0.470703125`, motif `0.0205078125`, core/non-core `0.5434496042840113`, SV `0.5156881267896678`, composite `0.3875871671434198`.
- All-variant A/C/G/T AF Pearson/Spearman are `0.053722671784755975`/`0.04651814414909852`; full-vocabulary values are `0.05369793822003601`/`0.04649856468639337`.
- On the 704 test rows, zero-shot Spearman is `0.0587130211215258`, whole-window probe `0.21585322960917655`, and variant-token probe `0.1695879636090382`.
- Whole-window/variant-token ridge alphas are `1e5`/`1e8`; the latter is the upper grid edge but its improvement over `1e7` is only `0.0002438237`, below the original truncation-risk threshold `0.002`.
- Task timing spans include the deliberate pause/restart; the inherited full-benchmark extrapolation must not be treated as steady-state throughput or a cost forecast.
- Small balanced pilot samples and 600-update weights do not establish model quality rankings against archived full-dataset dense results; no dense computations were rerun.
- Twelve CPU result/prediction/estimator artifacts and their receipt were uploaded and directly verified: `s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp582_plantcad2_moe/exp582-plantcad2-d1536-random-real-eval-smoke-v2/real-evaluation/cpu/receipt-c69fae2c720a9597bb76235aa4239e47050a91791c4883f8aaed195a2791ab6e.json`.

### 2026-10-06 — Pretrained real-data evaluation passes; validation stopped

- The same source `306056e` and evaluation package `ee4eb4f1618bd70c0140f9de24a07431f055722e` completed all 106 chunks for the pretrained character-mode step-600 checkpoint, without training updates.
- Both real-data pilots ran on one eight-H100 Reno gang at a time, with batch priority and user eczech; both roots and children are terminal.
- W&B run IDs are `exp582-plantcad2-d1536-random-real-eval-smoke-v2` and `exp582-plantcad2-d1536-pretrained-real-eval-smoke-v2` under `eric-czech/marin`; both record progress one and `validation/cpu_contracts_passed=true`.
- Pretrained task/AF probability oracle maximum errors are `3.129243850708008e-7`/`1.4901161193847656e-7`; repeated-forward and batch-context differences are exactly zero.
- Original reducers, five-fold grouped ridge CV, 1,000 block-bootstrap draws and saved-estimator prediction replay pass for both conditions.
- Independent review rechecked all chunks, hashes, task metrics, AF/feature alignment, finite outputs, archived memberships, training-only scaling, CV and saved probe predictions; no correctness blocker remains.
- Both variant-token probes choose the upper alpha `1e8`, with no truncation flag under the unchanged original heuristic; retain this caveat in later full evaluations.
- Per-condition samples are 128 per task (2,560 total), 2,112 AF variants, 1,408 probe training rows and 704 held-out probe rows.
- The following results are validation diagnostics from 600-update checkpoints and consequence-balanced pilot samples, not final comparisons with archived dense models.

| Condition | Conservation | Motif | Core/non-core | SV | Composite | AF Spearman (2,112) | Test zero-shot Spearman (704) | Test whole-window probe | Test variant-token probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 0.470703125 | 0.020507812 | 0.543449604 | 0.515688127 | 0.387587167 | 0.046518144 | 0.058713021 | 0.215853230 | 0.169587964 |
| pretrained | 0.474392361 | 0.043945312 | 0.424155567 | 0.502666733 | 0.361289993 | 0.030217746 | 0.029543687 | 0.165036411 | 0.167105092 |

- Pretrained CPU receipt (12 artifacts, upload/readback verified): `s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp582_plantcad2_moe/exp582-plantcad2-d1536-pretrained-real-eval-smoke-v2/real-evaluation/cpu/receipt-87e5ae98248cb14d84744db3bfffef7ece8f79dbee47434b4eb1c5a7f4c5d875.json`.
- The evaluation branch's documentation-only follow-up `2a1a655` clarifies full-window causal masking; the GPU wheel and training lock retain the tested `ee4eb4f` implementation.
- Both projects' tests pass (35 training, 23 evaluation), and the GPU canary's FSDP output-column gather failure has a reproducing CPU sharding test and verified runtime fix.
- SQLite integrity is `ok`, with zero active dispatches across 30 attempts; no production training or historical dense recomputation was launched.
- Final verified immutable backup: `s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/operations/validation/20261006T014046Z-a07a04cc7f634200bb75fd01b2c3cf14.sqlite`, 200,704 bytes, SHA-256 `683d87dc393c156a2459bb2f3b037aa351938de385a139d5c9ca0e038712b32b`.
- Validation has reached the user-requested stop; remaining production gates are recorded in the implementation plan.

### 2026-10-06 — Actual six-trial training authorized

The user explicitly authorized the six actual trials and capacity-based allocation across H100 clusters, always batch priority and user eczech.
This supersedes the earlier validation-only stop and single-gang validation scope.
Training Operations now lives in `experiments/exp582_plantcad2_moe/exp582_training_operations.md`; its separate SQLite database and immutable backup prefix preserve the completed validation history.

The full-run driver keeps global batch 64 fixed across 32/64-H100 placements, with a single native optimizer/QB update per batch.
The Hero baseline rates are MuonH 0.00187806248 and Adam 0.000433399034; multipliers are 0.5, 1 and 2 for each initialization condition, seed zero.
The exact stage and pre-cooldown boundaries are updates 412,290 and 371,061.
The indexed raw reader reuses exp472's actual Levanter shuffle/mixture classes, binds all 50 Parquet objects, and resumes by absolute occurrence rather than replaying earlier batches.
A direct CoreWeave S3 check verified the language source checkpoint metadata and encoded occurrences spanning start, epoch, cooldown and final-stage boundaries.
The independent review added real-framework shuffle/resume parity and row-group validation tests.
The training driver adds immutable science binding, two retained rolling recovery checkpoints, permanent pre-cooldown/final states, and fixed 512-window held-out loss with drop accounting.

### 2026-10-06 — Independent full-run review and CPU replica check

Independent review confirmed published branch head `59335f278ccc993774d9cb4be04d5234a1793f2e` directly against the Git remote.
The reviewed corrections defer progress one until the final checkpoint and validation succeed, cast validation's pending QB bias to the native BF16 compute policy, preserve and report failures through W&B, select the native inline-watch runtime defaults, and synchronize state, metrics and watch outputs before measuring step time.
The 14 corpus tests passed, including direct parity with Levanter's `LmDataConfig.train_set`, absolute-index resume across epoch and block boundaries, and rejection of changed object or row-group identities.
AST comparison also confirmed that the installed exp472 and current Levanter shuffle, mixture and key-derivation implementations match; the wrapper tests alone do not establish historical raw-cache row order.

A separate bounded local CPU check used eight explicit devices with four replicas and two expert partitions, a reduced native model with reference attention and fixed all-to-all, and one native optimizer update with gradient/update norm watch enabled.
Each of its two layers counted all 256 expert assignments across the replicas, both watch scalars were finite, and a complete checkpoint save/restore preserved state exactly (`max_abs=0`) and retained its token clock.
This checks replica reduction and checkpoint mechanics; it does not validate H100 kernels, multi-host transport, pinned-host offload, or memory fit at the production shape.
Those remaining runtime properties are monitored in the actual authorized trials, with the first 32-H100 trial establishing two-windows-per-device fit before additional trials use that shape.

### 2026-10-06 — Recovery checkpoint cadence

Published runtime snapshot `998a47c1a00279cc1c3e56ca166ce296539d0b7b` makes recovery-save cadence an operational CLI setting and saves after 25 additional updates following each restore.
This addresses repeated loss of unsaved work when interruptions recur sooner than the previous fixed save interval; the detailed preemption diagnosis and checkpoint-cost measurements live in training SQLite.
The new setting remains outside the immutable scientific binding and does not change optimizer updates, token clocks, data order, run IDs or checkpoint ownership.
Permanent pre-cooldown and final saves remain mandatory, and W&B now records the complete checkpoint pause duration.
`uv run --locked --extra cpu pytest` passed all 54 tests, repository pre-commit passed, and the existing six checkpoint bindings remained valid in direct CW S3 checks.
Independent review cleared the published diff against `970a434e275060c168cbdaa4a52776fd218556c1` with no actionable finding.
Training Operations governs deployment after a committed save or interruption; each dispatch's exact CLI interval is recorded in SQLite.

### 2026-10-06 — Shorter schedule authorized and migration validated

The user replaced the 216.159B-token stage with 11,240,734,720 input tokens (21,440 updates; approximately 0.52 corpus passes), keeping the original warmup duration and then cooling linearly to 5% of peak LR.
The original 562.022B-token heuristic reference remains unchanged, preserving peak MuonH/Adam rates, epsilon, beta2 and all other optimizer settings.
The permanent peak checkpoint normally follows update 10,720, which explicitly uses peak LR; the actual last update uses the 5% floor.
Historical dense results now have different training exposures and must be presented accordingly.

The user explicitly authorized stopping all six old submissions before the migration.
Every old root and child was verified terminal; observations and stop outcomes were backed up through the sweep persistence helper.
Direct CW S3 reads pinned full-state resume updates 10,771 / 8,427 / 8,816 for pretrained and 6,477 / 9,625 / 8,285 for scratch, in ascending LR order.
Pretrained 0.5× had passed the nominal peak and retained only updates 10,771 and 11,119.
The user approved restoring the earlier retained state, taking one peak-LR update and permanently saving update 10,772, with a 52-update shorter cooldown to the common final step.
This explicitly accepted exception cannot recreate the missing update-10,720 weights.

The source paths, clocks, metadata digests and original scientific-binding hashes are maintained in `src/exp582_moe/schedule_transition.json`.
Revised checkpoints and bindings live in `short-linear-v2/` beneath each original checkpoint root; original state is neither rewritten nor pruned.
Each resume validates the old source configuration and the new revision's checkpoint lineage, and refuses post-peak recovery without the permanent peak checkpoint.
Peak-boundary recovery repeats its held-out evaluation if needed.
W&B IDs remain unchanged, with an explicit revision marker on each training row; percentage rebasing is excluded from interpretations of newly trained work and placement rates spanning revisions.

Commands: `uv run --locked --extra cpu pytest -q` in the experiment and `uv run --locked pre-commit run --all-files --show-diff-on-failure` at the repository root.
Results: all 55 project tests and repository pre-commit passed; the built wheel includes all six exact transition entries; direct runtime validation of all six original source bindings passed.
Independent review found and corrected the final-update cooldown boundary and identified the need to distinguish progress normalization from optimizer-step advancement.
These local checks precede full-model runtime verification of the revised peak save and cooldown.

### 2026-10-06 — Revised training resumed and permanent peak verified

Published runtime: `3cb5b5725a385251bcd60f16c58368b5c18d00e5`; independent review verified the published diff and found no remaining actionable issue.
All six revised submissions use 64 H100s each, split equally across Reno and East, with batch priority and user eczech.
By 12:43 UTC all six had fresh revision-2 training rows with finite loss/gradient/update diagnostics and zero reported expert drops.
The user explicitly requested thirty-minute monitoring once the trials were back up.

Pretrained 0.5× restored the pinned update-10,771 state and logged update 10,772 at MuonH LR `0.0009390312404772878` and Adam LR `0.0002166995170332203`, exactly its peak rates.
It saved the permanent full-state checkpoint at `s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/checkpoints/exp582-plantcad2-d1536-pretrained-lrm0p5-seed0-v1/2026.10.06/checkpoints/short-linear-v2/step-10772`.
Direct CW readback verified the revised scientific binding, `is_temporary=false`, update 10,772, 5,647,630,336 input tokens and metadata SHA-256 `c11b634fe90349accb0453d86da3ab906d86aae98b5d4cf0bf7c9b13849136dd`.
W&B history steps 11,445–11,449 verify the peak update and immediate linear decay on updates 10,773–10,776; the first 24 revised updates had zero drops.
Its peak held-out loss was `1.1552812457084656` on the fixed 512 validation windows, with zero held-out expert drops.
The subsequent temporary recovery checkpoint at update 10,796 retained the permanent peak state.
These runtime checks establish the requested migration and peak-save behavior, not final model quality.

### 2026-10-06 — Interrupted checkpoint retry after gang resizing

Runtime `bd9e674305e1d14e378447a364ce404e85bf56b6` exposed a checkpoint retry defect in scratch 1×.
Its 32-H100 attempt reached update 11,189 but timed out waiting for all checkpoint writers; a subsequent 64-H100 attempt reached the same save destination and failed with `Expected chunk_shape [2,48,768,768] but received [4,48,768,768]`.
The incomplete destination lacked `metadata.json`, yet its array metadata survived and Levanter opened those arrays without replacing their layout.
Both attempts completed 25 finite training updates with zero expert drops before their checkpoint failures.

The training driver now coordinates primary-only cleanup of the exact uncommitted destination before saving, and refuses to overwrite any destination containing a commit marker.
This relies on the sweep's existing one-active-writer rule; completed checkpoints and other step directories are excluded.
A regression test reproduces the native TensorStore chunk-shape mismatch, removes the incomplete save, writes with the new chunk shape, restores the exact array, and verifies committed-state protection.
Commands: `uv run --locked --extra cpu pytest -q` in the experiment and `uv run --locked pre-commit run --all-files --show-diff-on-failure` at the repository root.
Results: all 56 project tests and repository checks passed.

After verifying both failed roots and children were terminal, direct CW cleanup removed only `short-linear-v2/step-11189` for scratch 1×.
Before/after SHA-256 checks preserved `latest.json`, `training.json`, committed update 11,164 (`c514bab4b324bd27d55417d53a7c27559bf16d07c09de813096f493fb6ea7286`) and permanent peak update 10,720 (`ba7f1b6e41eef86cd4ca6410e7af370acb7662d2f0475d0d38d0602edfb98f5a`).
The durable cleanup receipt is in training SQLite; the next runtime check is a successful full-model save beyond the failed boundary.

### 2026-10-06 — Native crash during the checkpoint retry

The published cleanup fix `3b8851e16922b2dfdbbd5c9fb5d884b41ad0a7e2` passed independent review and all GitHub checks.
Scratch 1× resumed update 11,164 on 64 Reno H100s and produced finite, zero-drop training rows before requesting the update-11,189 checkpoint.
All nodes reported native segmentation faults two seconds after checkpoint planning began, before staging completed; the previous chunk-layout exception did not recur.
The cause remains unresolved, and source review found no supported model-buffer lifetime issue in the added broadcast.
The other five trials continued training and saving checkpoints.

`EXP582_CHECKPOINT_DEBUG=1` now forwards Python fault handling and checkpoint stage logs to GPU workers without enabling forced GC or allocation tracing.
The next bounded diagnostic is one retry from the same verified state with these diagnostics enabled.
All 56 project tests, both native state tests with diagnostics enabled, and repository pre-commit passed.
The incomplete update-11,189 destination has no commit marker; direct reads confirmed that update 11,164 and permanent peak 10,720 still match their recorded metadata hashes.
Reusable incomplete-checkpoint handling is tracked separately in #585.

### 2026-10-06 — Full-model checkpoint retry verified

The diagnostic runtime `dc5fc82da0caf27b44cced4648ccefc95a4a1786` passed independent review and all GitHub checks.
Scratch 1× successfully committed update 11,189 at 17:18 UTC in 85.04 seconds, then resumed finite, zero-drop training.
Direct CW reads verified metadata SHA-256 `dddbf1058d6b1ade1124d5b1355614e9b150a8c4c9d50effcbb2cc1fb1cf2310`, all four token/data clocks, its unchanged scientific binding, and the permanent peak checkpoint's original hash.
This verifies recovery past the repeatedly failing save destination; the one native crash did not recur, and its underlying cause remains unresolved.
Diagnostics remain enabled on this dispatch to capture a recurrence.

### 2026-10-06 — All six permanent peak checkpoints verified

Direct CW readback now verifies every trial's permanent peak checkpoint against its revised scientific binding, tokenizer, four training clocks and `is_temporary=false`.
Five peaks are at update 10,720; pretrained 0.5× uses the previously approved update-10,772 exception.
Scratch 0.5× committed the last remaining peak at 17:55:45 UTC, with metadata SHA-256 `2dde4a8af9a6c3342431fc93160acea21117986d916c52bfa2f95b83a48abc01`.
Its W&B history verifies the exact peak rates followed by the revised linear cooldown, and the recovery checkpoint at update 11,009 retains that permanent peak.
The fixed 512-window peak held-out losses are 1.155281 / 1.135247 / 1.138671 for pretrained and 1.201102 / 1.168985 / 1.157385 for scratch, in ascending LR order; all six peak evaluations reported zero expert drops.
Vocabulary differences prevent direct cross-condition loss comparisons, and these peak measurements are not final downstream evaluation results.
The verification receipt and individual checkpoint digests are in the immutable SQLite backup `s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/operations/training/20261006T181638Z-e275ac0fabd845ae80ffda62a7f7b117.sqlite` (SHA-256 `29f062a72cddfef1cacca82a3da60cfc7b494c1ffe5700b9c7be465347f8adf8`).

### 2026-10-06 — Prioritize the current held-out leaders

The user selected and confirmed pretrained 1× and scratch 2× using the latest `eval/loss` within each initialization, with values 1.095174 at update 14,000 and 1.142412 at update 12,000 respectively.
These measurements use the fixed 512 validation windows but are not a comparison at equal exposure across all LRs.
Finish both selected trials before resuming the other four, retaining all six identities and full-state checkpoints.
The user also requested trying 128 H100s per selected trial; the batch of 64 rules out simply doubling data parallelism.

The existing Hero backend implements context parallelism, which can split each sequence across two devices without changing the batch or context length.
`XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run --locked --extra cpu pytest -q tests/test_context_parallel.py` passed in the experiment project.
The reduced native model uses reference attention and fixed all-to-all; after one update, it restores the full state from CP1 into CP2, takes another native optimizer update, and checks parameters, moments, pending router bias, loss, routing counts, drop counts and gradient/update norms against the CP1 continuation.
All comparisons passed at absolute tolerance 1e-5 and relative tolerance 1e-4, with zero drops.
This establishes a local numerical oracle, not H100 kernel compatibility, multi-host performance, or production memory fit; 128-H100 targets remain unvalidated.

### 2026-10-06 — Prepare full-model 128-H100 qualification

The placement extension after `885fcdf0878f28bf34fbee9603856d50b4a17b3a` retains global batch 64 with CP2, EP8 and eight replicas; Trainer initialization excludes the context axis from batch sharding.
`placement_smoke.py` reads each selected trial's immutable permanent peak checkpoint, verifies its metadata hash and scientific binding, evaluates the existing 512-window sample, and takes 26 native updates under the actual schedule and data cursor.
After update 25 it saves to a separate seven-day smoke prefix, releases the original state, restores through an abstract exemplar, and verifies the exact addressable-shard SHA-256 on every process before the last update.
It never writes the actual trial's checkpoint root or W&B identity.
The extended eight-CPU oracle passed in 71.41 seconds, including abstract-exemplar restoration and exact shard hashes; the ordinary project suite passed 59 tests with the eight-device test skipped.
These checks prepare the H100 test but do not qualify H100 memory, kernels or throughput; both 128-H100 targets remain unvalidated pending the bounded remote runs.
Independent review found that the generic smoke launcher selected the runtime defaults for disabled gradient/update watching, unlike actual training.
The placement smoke now explicitly enables the same inline-watch runtime defaults as training, avoiding the Hero backend's documented incompatible collective-overlap setting.

### 2026-10-06 — Suspend Reno qualification during inventory incident

The user requested cancelling our Reno work after the dashboard repeatedly reported zero total H100s.
The pending 128-H100 qualification was cancelled before its GPU child or W&B run existed, so full-model CP2 validation remains outstanding.
Operations now suspends Reno admissions while the two selected trials continue on their existing East placements.

### 2026-10-07 — Both full-model 128-H100 placements qualified

Runtime commit `05c0f7b52332647facd0b891c3e54216d45ed91e` passed both native placement checks on 16 Reno H100 nodes with CP2, node-local EP8, batch 64 and full 8,192-token examples.
Each run used `python -m exp582_moe.placement_smoke --condition <condition> --run-id <run-id> --cluster cw-rno2a --nodes 16 --source-metadata-sha256 <pinned permanent-peak digest>`; the exact batch/eczech submission commands and immutable receipts are in validation SQLite.
Both completed 26 finite native updates with zero aggregate and per-layer expert drops, and verified exact full-state addressable-shard hashes after saving update 10,745 and restoring it before the final update.
Direct CW readback verified both saved metadata, original permanent-peak hashes, scientific bindings and token/data clocks.

| Condition | Original 512-window peak loss | CP2 restored peak loss | Median CP2 update, excluding compilation | Recent East CP1 update |
| --- | ---: | ---: | ---: | ---: |
| Pretrained 1× | 1.135246992 | 1.135248378 | 1.662 seconds | 3.502 seconds |
| Scratch 2× | 1.157385364 | 1.157391131 | 1.479 seconds | 2.835 seconds |

The tiny loss differences are within the accepted non-bitwise numerical tolerance.
These are short placement measurements, not final downstream evaluation results or guarantees of end-to-end speedup.
The first launcher waited for Reno admission, although Iris labeled its gated pod as building; the later scratch check reached native training approximately five minutes after submission.
Operations reopens qualified Reno placements, but each actual move still requires fresh free capacity, a verified recent checkpoint and a worthwhile saving after startup and rollback costs.

W&B: [pretrained placement](https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-pretrained-lrm1-cp2-n16-smoke-v1), [scratch placement](https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-random-lrm2-cp2-n16-smoke-v1).

### 2026-10-07 — Pretrained 1× completed the shortened first stage

The [pretrained 1× trial](https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-pretrained-lrm1-seed0-v1) completed at 03:04 UTC with 21,440 updates, 11,240,734,720 input tokens, 11,239,362,560 loss targets and 1,372,160 examples.
Its final fixed-sample language-model evaluation used 512 windows and produced loss 1.0352993607521055 with zero expert drops, compared with 1.135246992111206 at the permanent peak checkpoint.
The final dispatch ran runtime `8ae5a9cd0ecadfeb03c05789326d317dff6e5982` on 128 Reno H100s, resuming full state from update 20,770; its last 670 CP2 updates were finite, had zero expert drops, and passed exact token-clock and LR-schedule checks.
The exact batch/eczech command and placement lineage are retained in training SQLite under dispatch `pretrained-lrm1-a4`.

Direct CW verification confirmed the permanent final checkpoint at `s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/checkpoints/exp582-plantcad2-d1536-pretrained-lrm1-seed0-v1/2026.10.06/checkpoints/short-linear-v2/step-21440`, its scientific binding and the original permanent peak's unchanged hash.
Final metadata SHA-256 is `3f175a36575f0fc55ecf0dc5fc0bd9ce4b22923f54f31f5bfa26c60322c44b88`.
W&B progress reached one and both the exact Iris root and GPU child succeeded before the trial was marked complete.
The immutable verification receipt is in `s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/operations/training/20261007T031142Z-797e121269834046a74632761136a0dd.sqlite`, SHA-256 `5a48c13a8684980d83aa7cb1005cbcd01c4b817d6a33239ef241544a5726e162`.

### 2026-10-07 — Scratch 2× completed and final-checkpoint evaluation pilots passed

The [scratch 2× trial](https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-random-lrm2-seed0-v1) completed at 04:11 UTC at the same 21,440 updates and token/data clocks as pretrained 1×.
Its final fixed 512-window evaluation loss was `1.041194662451744` with zero expert drops.
Direct CW verification confirmed the permanent final checkpoint at `s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/checkpoints/exp582-plantcad2-d1536-random-lrm2-seed0-v1/2026.10.06/checkpoints/short-linear-v2/step-21440`, with raw metadata SHA-256 `8e4249e1b37a4d37ab272c513ff5aa7831cbb990865fb7384236bd40d5a8e6f7` and canonical metadata digest `4a1ddbd00f0e204a662c20b431ec68dc72d398ce77022318755758b1df98c347`.

A stale continuation briefly submitted roots for three paused trials after scratch completed.
All three were cancelled before any GPU child was created, pretrained 0.5× was never submitted, and training SQLite now has zero active dispatches.
The user's newer instruction keeps all four remaining training trials paused until explicitly resumed.

The final-checkpoint evaluation pilots ran on one eight-H100 Reno node per condition at batch priority under `eczech`.
The initial attempts failed before inference because the launcher passed the raw metadata byte hash where `restore_weights` requires the canonical sorted-JSON digest; the corrected identities preserve both values explicitly.
[Pretrained 1×](https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-pretrained-lrm1-seed0-final-eval-smoke-v2) and [scratch 2×](https://wandb.ai/eric-czech/marin/runs/exp582-plantcad2-d1536-random-lrm2-seed0-final-eval-smoke-v2) then completed all 106 pilot chunks with no errors or reused outputs.
Direct CW reads verified every manifest, receipt, array checksum and checkpoint/input/source binding.
Repeated inference and batch invariance were exact; independent ACGT projection errors were at most `6.86e-7`.
The original PR #20 reducers reproduced 20 tasks with 128 examples each, 2,112 allele-frequency rows, and the archived 1,408/704 probe split; serialized probe estimators exactly reproduced their saved predictions.
These values are correctness pilots and are not publishable final results.
The immutable evaluation state is under `s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/operations/evaluation/`; backup `20261007T044753Z-bc6ce8eca07e4091b25c9f66ffc6ad6d.sqlite` has SHA-256 `b6e71c9f36d5bc396de9ebc45a6a0875543f1d6ffa4e2844708e1645e945bcf3`.
