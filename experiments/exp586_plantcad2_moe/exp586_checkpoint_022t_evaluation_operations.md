# exp586 shorter-run checkpoint evaluations

Read this document before every execution pass.
Use `run-training-sweep-cw` for capacity, dispatch, W&B progress and transactional state.

## Scope and invariants

Evaluate permanent checkpoints from `exp586-plantcad2-d768-pretrained-lrm1-seed0-v1` matched by actual W&B `run_progress` to the previously evaluated language-only, 10B, 182B and 545B checkpoints of the independent 1T run, plus the shorter run's final checkpoint.
The immutable `scratch/checkpoint-evals-022t/catalog.json` owns the selection and exact identities.
Reuse complete verified language-only and final short-run results when checkpoint, rows, metrics and split contracts match.
Use the unchanged `63da70cf56bdad5943ef6cad23ae5c69f8ae6778752caca3d6918b4d9ffa818b` scientific runtime and the previous trajectory's analysis wheels, memberships, grids and readouts.
For the 2.16B peak checkpoint only, use the exact-identity admission compatibility in `codex-exp586-checkpoint-022t-compat`, pinned by `scratch/checkpoint-evals-022t/runtime-binding-legacy.json`.
Only the metadata validator differs; all numerical scoring and model files remain byte-identical.
The checkpoint predates the backend metadata field; retain that field as unknown rather than fabricate a training backend.
Both numerical validation gates remain mandatory.
Run all zero-shot scores, both contexts, AF zero-shot and all AF regression probes, and the twelve classification probe tasks.
Exclude motif probes.
Refit each new checkpoint's frozen heads independently, using development data only.
Keep target spans and labels outside classification inputs; preserve established SV allele and AF variant contracts.
Report tables and comparisons only in chat, with no GitHub updates.

## Operator choices

Use H100s only, whole eight-GPU nodes, on Reno and East, at most 544 GPUs concurrently, conditional on refreshed live free capacity.
The user authorizes aggressive use of idle GPUs; this ceiling retains the previous study's agent-selected limit.
Every root and descendant must resolve to batch priority under `eczech`.
Use W&B `eric-czech/marin`, tag `dna-exp586`.
Follow W&B for progress; use Iris for exact-root liveness, startup failures or concrete placement questions.
Retain one writer per immutable chunk and one active dispatch per logical trial.
Use 30-minute heartbeats, one-hour reslice and three-hour restart thresholds.
The hard sweep deadline is 2026-11-18T23:59:59Z; recovery timing is independent of this overall deadline.
CPU-only reduction and fitting may overlap GPU extraction, using batch roots with 32 CPUs and 128 GB RAM each.

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| cw-rno2a | H100 | 1 | 8 | eligible | Established identical evaluation runtime |
| cw-us-east-02a | H100 | 1 | 8 | eligible | Established identical evaluation runtime |

## Durable ownership and recovery

Use `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/evaluations/checkpoint-trajectory-022t-v1/` for inputs, gates, zero-shot and AF extraction, analysis, and reports.
Classification features and heads remain under `evaluations/frozen-classification-v1/checkpoint-trajectory-022t-v1/`.
Use only `scratch/checkpoint-evals-022t/evaluation.sqlite` and the skill's transactional persistence helper.
After every state-changing transaction and before another external action, publish a consistent immutable size/SHA256/readback-verified SQLite backup under `operations/checkpoint-evaluation-022t/`.
Immutable keys contain UTC timestamp, action label and UUID.
Recover the newest integrity-checked immutable backup if the local ledger is lost.
Use `scratch/checkpoint-evals-022t/storage.py` for fresh Secret Manager CoreWeave credentials and verified upload/download.
Stop if direct CoreWeave S3 reads fail.
Record CPU job identities/results as ledger events and append research milestones to `.agents/logbooks/exp586.md`.

## Validation and dispatch

Before production extraction, require direct permanent checkpoint metadata/manifest reads and hash bindings, original validation-loss replay, finite zero-drop output, repeat/batch invariance and independent ordered FP32 reference parity.
Require the existing six-case classification input-exclusion and hidden-state gate.
Allow bounded gates concurrently for independent checkpoint identities on the established runtime.
After both checkpoint gates pass and its first full worker proves W&B progress, admit homogeneous workers concurrently subject to live capacity.
Use sixteen full workers and eight feature workers per new checkpoint; only partition assignments differ from the previous eight-worker study.
All source rows, chunk boundaries and scoring are unchanged.
Only explicit agent decisions may start, stop or replace dispatches; helpers execute persisted selections.
Verify complete native receipts, independent held-out metrics and raw-feature/head round trips before reporting results.

## Compatibility change

2026-11-04T11:59Z: The first permanent checkpoint lacks backend metadata and fails strict admission before inference.
A separate pinned runtime accepts only its exact URI, canonical metadata digest and raw metadata digest, preserving the original metadata and all scientific computation.
Existing later-checkpoint evaluations retain the original frozen runtime.
The retry is permitted only after exact failed roots and descendants are terminal and both original dispatches have ended in SQLite.
