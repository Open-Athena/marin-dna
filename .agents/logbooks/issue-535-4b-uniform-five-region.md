# Issue 535: 4B uniform five-region continuation

Tracking issue: https://github.com/Open-Athena/marin-dna/issues/535

## Task state

The user authorized one preemptible `v5p-16` slice in `us-east5-a`, first for a 20-step operational smoke and then for the 160,000-step production continuation if the smoke verifies restore and artifact writing.
The run stays at batch priority, retains GCS, has no online validation suite, and does not fall back to another accelerator family.

## Prior-work brief

The direct parent is W&B run `eric-czech/marin/dna-bolinas-scaling-v0.5-h2944-p4B-fa02c3` and permanent native checkpoint `step-215573`.
Its W&B config fixes the 4.02B Llama geometry, global batch 1536, per-device parallelism 192, AdamH hyperparameters, tokenizer, and three-region cache identities.
The parent checkpoint metadata was read from GCS and reports step 215573 with `is_temporary=false`.

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
- Use global batch 1536 and per-device parallelism 192 on the eight-device `v5p-16`, giving no gradient accumulation.
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

## Launch record

Pending committed constraint-fix snapshot and corrected smoke dispatch.
