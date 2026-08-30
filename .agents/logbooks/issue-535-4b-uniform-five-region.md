# Issue 535: 4B uniform five-region continuation

Tracking issue: https://github.com/Open-Athena/marin-dna/issues/535

## Task state

The user authorized one preemptible `v5p-8` slice in `us-east5-a`, first for a 20-step operational smoke and then for the 160,000-step production continuation if the smoke verifies restore and artifact writing.
The run stays at batch priority, retains GCS, has no online validation suite, and does not fall back to another accelerator family.
The replacement smoke is queued for its TPU host after the coordinator successfully acquired the new artifact lock.

## Prior-work brief

The direct parent is W&B run `eric-czech/marin/dna-bolinas-scaling-v0.5-h2944-p4B-fa02c3` and permanent native checkpoint `step-215573`.
Its W&B config fixes the 4.02B Llama geometry, global batch 1536, per-device parallelism 192, AdamH hyperparameters, tokenizer, and three-region cache identities.
The parent checkpoint metadata was read from GCS and reports step 215573 with `is_temporary=false`.
Its reported theoretical FLOP rate is consistent with four v5p chips, so `v5p-8` is the parent topology and preserves the original no-accumulation batch layout.

The five-region recipe is W&B run `eric-czech/marin/dna-bolinas-mix-v0.9-p1B-i24-exp135-zoonomia-m5.1-bef41e` and historical Marin experiment `exp135_bolinas_mix_sweep.py`.
Its config records equal 0.20 weights, the exact five hashed cache paths, lowercase loss weight 0.01, restart-on-exhaustion behavior, data seed 19, and full-state continuation through `trainer.initialize_from`.

Current Marin issue #358 and Marin commit `53b5b33041f742c7f4991223b0085e41ece4c458` establish the current lazy `ArtifactStep` consumer and Iris API.
The current generic `marin.experiment.train.train_lm(init_from=...)` resolves to `initialize_from_checkpoint_path`, whose Levanter implementation restores a state and then resets its step to zero.
That helper is therefore a negative lead for this experiment.
The implementation instead sets `TrainerConfig.initialize_from`, which restores model, optimizer, RNG, and global step and resumes from a new output namespace after preemption.

Levanter's stock HF cadence fires on absolute multiples of an interval.
Because the parent step is not a multiple of 20,000, the stock callback would miss the requested added-step offsets.
The experiment installs an exact-step export hook for global steps 235573 through 375573.
The native retention policy uses an exact one-shot interval at global step 343573; Levanter's forced final hook writes step 375573.

All five GCS cache train manifests were checked directly and are non-empty.
Their recorded token counts are 62,037,687,296 CDS, 17,481,258,496 upstream, 5,248,475,136 downstream, 3,910,928,384 ncRNA exon, and 24,739,788,800 non-promoter cCRE tokens.
The historical cache records do not pin Hugging Face revisions, so the experiment pins the realized hashed GCS cache paths rather than re-reading mutable Hub datasets.
The vendored character tokenizer is pinned to source revision `a73e9d9ee636f722b4c378703c9e2997857809b2` and checked by per-file SHA-256.

## Decisions

- Use experiment ID prefix `DNA-535`, branch `codex/issue-535-4b-uniform-five-region`, and W&B group `dna-exp535`.
- Pin Marin and Iris to coherent commit `53b5b33041f742c7f4991223b0085e41ece4c458`.
- Use global batch 1536 and per-device parallelism 192 on the four-device `v5p-8`, matching the parent topology and giving no gradient accumulation.
- Use data seed 535 while preserving the restored training RNG and optimizer state.
- Keep the peak AdamH and Adam learning rates and all inherited optimizer coefficients.
- Use cycle lengths `[215573, 160000]`, 16,000 rewarmup steps, 112,000 plateau steps, and 32,000 cooldown steps.
- Keep rolling native recovery states, permanent global step 343573, forced final global step 375573, and eight exact HF exports.
- Do not serialize `WANDB_API_KEY` into the artifact graph; inherit it from the Iris job environment.

## Actions

- 2026-08-27: Verified the five cache roots and parent checkpoint metadata in GCS.
- 2026-08-27: Queried the parent and m5.1 W&B configs for exact model, optimizer, data, and continuation settings.
- 2026-08-27: Created the permanent experiment branch and began the locked self-contained project.
- 2026-08-27: Committed and pushed initial snapshot `5fe4205e`.
- 2026-08-27: Remote locked tests found that lazy artifacts require plain calendar versions and that a CPU test with the TPU extra must explicitly select the CPU JAX backend.
- 2026-08-27: Corrected both preflight-only failures; neither job reached training or requested an accelerator.
- 2026-08-27: Remote locked test job `/gonzalo/exp535-config-tests-v2` passed all five configuration tests on commit `4e15974d`.
- 2026-08-27: Canceled smoke coordinators v1 and v2 before TPU allocation after finding coordinator-placement and missing-CLI-version errors.
- 2026-08-27: Smoke v3 created the exact `v5p-16` child, but Iris classified its inherited placement as on-demand and reported `tier_blocked` against the preemptible pool; no TPU worker was allocated.
- 2026-08-27: Added an experiment-local compatibility shim that makes `preemptible=True` a hard Iris capacity constraint for the TPU child while retaining an on-demand CPU coordinator.
- 2026-08-27: Remote locked test job `/gonzalo/exp535-config-tests-v3` passed all six configuration tests, including the persisted hard-preemptible constraint check.
- 2026-08-27: Committed and pushed corrected launch snapshot `66de67d3`.
- 2026-08-27: Dispatched smoke `/gonzalo/exp535-4b-five-region-smoke-v4`; its child is correctly routed as two co-scheduled preemptible `v5p-16` tasks in `us-east5-a`.
- 2026-08-27: Google returned repeated `us-east5-a` TPU stockouts while Iris tried to satisfy the lower v5p quota tier, so tier monotonicity is holding the exact `v5p-16` request pending.
- 2026-08-27: Left the approved smoke queued without changing region or accelerator; no TPU worker had been allocated at the last observation.
- 2026-08-30: The `v5p-16` smoke eventually received intermittent allocations and accumulated 32 preemptions before failing prior to restore or compilation.
- 2026-08-30: Diagnosed the terminal failure as current Levanter constructing validation sets for train-only cache components whose validation sources and caches do not exist.
- 2026-08-30: Confirmed from the parent W&B hardware metrics that its four v5p chips correspond to one `v5p-8` slice, and the user approved retrying on that parent topology.
- 2026-08-30: Added an explicit train-only data configuration, fresh smoke identities, and the `v5p-8` placement at snapshot `6420170c`.
- 2026-08-30: Passed all seven locked project tests, including cloudpickle round-trip coverage of the train-only data override, and passed the non-launching artifact-graph preflight.
- 2026-08-30: Dispatched replacement smoke `/gonzalo/exp535-4b-five-region-smoke-v5p8-v1` at 19:42 UTC.
- 2026-08-30: The coordinator acquired the artifact lock and submitted hard-preemptible child `_run_on_tpu-b2081357`; the child is pending because the scheduler has 56 of the required 104 TPU-host CPU cores available.

## Launch record

Current smoke dashboard: https://iris.oa.dev/#/job/%2Fgonzalo%2Fexp535-4b-five-region-smoke-v5p8-v1

Failed `v5p-16` smoke dashboard: https://iris.oa.dev/#/job/%2Fgonzalo%2Fexp535-4b-five-region-smoke-v4

The exact `v5p-8` smoke request is queued for TPU capacity with zero failures and zero preemptions.
Production remains undispatched until the operational smoke verifies restoration, progress, native checkpointing, and the final HF export.

## Entry log

### 2026-08-30 19:42 UTC - Parent-topology replacement smoke

Hypothesis: The original four-chip `v5p-8` topology should preserve the parent's batch behavior while requiring less contested capacity than the prior `v5p-16` request.

Configuration: Continue native full state from global step 215573 for 20 smoke steps with five train components weighted 0.20 each, global batch 1536, per-device parallelism 192, WSD learning-rate settings unchanged, no online validation datasets, and one hard-preemptible `v5p-8` slice in `us-east5-a`.

Code snapshot: `6420170c365b9373610abdb4ee334494c8fff12e`.

Validation: `uv run --locked pytest -q` passed seven tests, the package import succeeded, the train-only data subclass survived serialization through the TPU pod configuration, and the non-launching graph resolved the expected parent checkpoint, target step, exact 0.20 weights, and fresh artifact identity.

Launch: `/gonzalo/exp535-4b-five-region-smoke-v5p8-v1` created child `/gonzalo/exp535-4b-five-region-smoke-v5p8-v1/_run_on_tpu-b2081357`.

Observed result: The CPU coordinator is running and holds the artifact lock.
The TPU child is pending with zero failures and zero preemptions because 56 of the requested 104 TPU-host CPU cores are currently available.

Interpretation: Configuration and credential setup have cleared the coordinator path, while real `v5p-8` capacity is the current constraint.

Next action: Keep the smoke queued and verify native restore, absence of validation construction, compilation, 20 optimizer steps, the final native checkpoint, and the HF export before dispatching production.
