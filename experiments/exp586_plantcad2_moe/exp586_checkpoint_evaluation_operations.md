# exp586 checkpoint evaluation operations

Read this document before every execution pass.
Use `run-training-sweep-cw` for capacity, dispatch, W&B progress, and durable state.

## Scope

The user authorized the five-checkpoint study on 2026-11-04.
The immutable catalog is `scratch/checkpoint-evals/catalog.json` and its S3 receipt.
Evaluate the original d768 language source, then the permanent 10B, 182B, 545B, and final 1T DNA checkpoints.
Reuse the verified final 1T evaluation and classification-probe results.
Run all existing zero-shot tasks, AF zero-shot and frozen regression probes, and twelve frozen classification tasks for the four new checkpoints.
Exclude motif probes.
Keep all existing input memberships, context lengths, readouts, labels, and development-only probe selection rules unchanged.
Refit each checkpoint's probe heads independently.
Return results in chat; no GitHub issue, PR, comment, or draft updates are authorized.

## Validation

Require direct S3 metadata and checkpoint manifest access, pinned metadata hashes, and exact source-token clocks.
Require finite, zero-drop readouts, exact repeat/batch invariance, and independent ordered FP32 scatter parity for probabilities, suffix logs, and AF features at 2K and 8K before full scoring.
Compare native atomic scatter diagnostically and replay the original 512-window loss, against the saved training evaluation where available.
Require the established flank/SV hidden-state gate before full classification extraction.
The language-only control uses the original full vocabulary and the same character-forced DNA encoding as the DNA-trained models.
Its unadapted routing is evaluated dropless, including stress and ambiguous-base cases.

## Execution

Use H100s only, whole eight-GPU nodes, on Reno and East, at most 544 GPUs concurrently.
Every root uses `--priority batch --user eczech`; children explicitly use batch priority.
Use W&B `eric-czech/marin`, experiment tag `dna-exp586`.
Read the utilization helper before dispatch or capacity changes.
Follow W&B progress and use Iris for exact-job liveness or concrete failures only.
Allow concurrent identical workers after the checkpoint's gate passes and a first full worker demonstrates W&B progress.
Maintain one writer per immutable chunk and one active dispatch per logical trial.
Execution partitions may be rebalanced without changing rows, chunk keys, scoring, or probe memberships.
Use 30-minute heartbeats, reslice after one hour without progress, and restart after three hours without progress.
The hard deadline is 2026-11-17T23:59:59Z.
CPU-only reduction and fitting may overlap GPU extraction and use batch roots with up to 32 CPUs and 128 GB RAM each.

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| cw-rno2a | H100 | 1 | 8 | eligible | Established zero-shot, AF, and classification inference runtime |
| cw-us-east-02a | H100 | 1 | 8 | eligible | Established zero-shot, AF, and classification inference runtime |

## Durable ownership

Use `s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/evaluations/checkpoint-trajectory-v1/` for the study catalog, gates, full evaluations, reductions, and summary.
Classification extraction retains its existing native owner, with distinct checkpoint labels under `evaluations/frozen-classification-v1/checkpoint-trajectory-v1/`.
Use `scratch/checkpoint-evals/evaluation.sqlite` with the skill's transactional mutations.
Publish consistent immutable size/SHA256-verified backups under `operations/checkpoint-evaluation/` before dispatch and after each state-changing transaction.
Recover the newest integrity-checked immutable backup if the local ledger is lost.
Persist CPU job identities and results as ledger events.
Keep research chronology in `.agents/logbooks/exp586.md`.

## Operating exceptions

Admit up to four bounded validation dispatches per target concurrently because they share an established runtime and have disjoint checkpoint/output identities.
For full extraction, admit homogeneous workers concurrently only after a checkpoint gate passes and its first worker proves W&B progress.
Reduction reuses existing full-score chunks to derive matched classification baselines before fitting, avoiding another inference pass.

## Capacity expansion

2026-11-04: Following the current user instruction to use more idle H100s aggressively, expand the initial 256-GPU plan to 448 GPUs after both gates pass for every checkpoint.
Overlap six classification workers per checkpoint with the eight zero-shot/AF workers per checkpoint, reserving all submitted but not-yet-running nodes against reported free capacity.
Admission remains conditional on live capacity, and all jobs remain batch priority under eczech.

2026-11-04 06:05 UTC: The current user instruction explicitly asks for aggressive use of additional available GPUs.
After the first full worker finished, live capacity showed another 102 free H100s across Reno/East.
Expand the working fleet ceiling from 448 to 544 to permit splitting up to twelve remaining longer workers, subject to refreshed capacity before each selected group.
The full candidate plan would use approximately 536 GPUs before other workers finish; the ledger records the smaller groups actually admitted as shared capacity changes.
This revises an agent-selected allocation limit; the user has not imposed a smaller GPU or cost ceiling.
Keep per-dispatch live-capacity checks and whole-node admission, and release every GPU after its scoring partition finishes.

## CPU analysis

Publish the original zero-shot reduction and matched classification baselines first, without waiting for AF fits.
Then use the PlantCAD2 `checkpoint_af.py` runner for separate 2K/8K CPU fits from verified cached AF embeddings and predictions.
It calls the original four-readout fitter unchanged, with the same split, grids, bootstrap, and estimator checks.
Use `scratch/checkpoint-evals/af-wheel-receipt.json` for its exact code identity and per-job content-pinned configs; no GPU inference is repeated.
The unchanged classification fitter consumes the existing memberships and the checkpoint's matched zero-shot vectors.
Keep CPU identities/results in ledger events and verify their native receipts before reporting completion.

## Execution partitions

The optional executor in `codex-exp586-trajectory-scale` preserves every frozen scoring file and verifies their aggregate identity before running.
It selects `original_jobs[part::2]`, reuses the existing checksum-bound chunks, and writes separate part manifests.
Remote canaries on East and Reno reproduced all six canonical task, AF-score, and AF-embedding objects bitwise at both context lengths.
Before replacing a worker, cancel its exact root and verify every descendant is terminal; the control ledger records the selected parent and replacement parts.
Keep the stopped parent's logical trial as an aggregate and never recover it independently while its two parts own the chunks.
`scratch/checkpoint-evals/merge_selected_worker.py` validates complete disjoint coverage and provenance, publishes the canonical report, and records aggregate completion in W&B and SQLite.
