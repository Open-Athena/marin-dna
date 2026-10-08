# exp582 CoreWeave Validation Operations

> This document governs validation managed with the `run-training-sweep-cw` skill.
> Read it in full before each observation or dispatch pass.

## Invariants

Validation only; stop for user review after evaluation validation and before production training.
Every root is submitted to the `marin` controller with exact target, `--priority batch --user eczech`.
Child jobs explicitly request batch priority and have a one-hour execution timeout.
This also prevents direct invocation from falling back to interactive priority outside an Iris parent.
Keep one active dispatch and checkpoint writer per trial and one gang total.
Use whole eight-H100 nodes; never reclaim higher-priority capacity.
Use W&B `eric-czech/marin` with the user's key as the source of progress.
Read Iris logs only to diagnose a concrete startup, runtime, or placement failure.
Stop if direct CoreWeave S3 access fails.

## Validation Definition

Entry point: `python -m exp582_moe.gpu_smoke`.
The initial bounded catalog comprises its `random` and `pretrained` conditions, followed by evaluation validation.
Training smoke identities and TTL checkpoints are defined in `src/exp582_moe/gpu_smoke.py`.
This entry point performs only three updates and checkpoint replay; it cannot start a production trial.
Full training batch sizing, corpus auditing, and evaluation validation remain separate gates in the task logbook.
The user authorized renewed routing diagnostics on 2026-10-05 after reviewing the failed canary.
`python -m exp582_moe.routing_probe` defines the bounded routing catalog: random/pretrained initialization with the existing pooled backend; a dropless grouped-GEMM/FSDP fallback is available after the EP investigation.
Its explicit warmup and update limit apply only to diagnostics; they do not replace the agreed production token schedule.

## Operator Choices

H100 preferred; use `cw-us-east-02a` or `cw-rno2a` with no cross-family moves.
One gang at a time, at most 64 GPUs, starting with eight GPUs.
Iris user: `eczech`; priority: batch.
Each job is limited to one hour; the newly authorized real-data evaluation pass stops by 2026-10-06 04:00 UTC (midnight America/New_York) if unfinished.
This pass uses the existing two step-600 checkpoints for 128 rows per original task and approximately 2,000 AF/probe variants; no training updates are authorized.
The user steered the stalled pass toward available Reno H100s at 21:23 UTC; this renews the bounded window for that migration and evaluation.
The earlier pass ended before its 20:00 UTC deadline; the user subsequently authorized the character-tokenization change and renewed validation.
The pass deadline stops compute; it does not shorten recovery clocks or authorize subsequent production work.
User steering at 2026-10-05 15:57:46 UTC: prioritize the existing EP implementation for both conditions for up to three hours, ending 18:57:46 UTC (2:57:46 PM America/New_York).
If EP is still unsuitable then, continue stability validation with another backend within the remaining validation scope.
Correctness and stability take precedence over throughput.

The local SQLite working copy is `scratch/exp582-validation/exp582_validation.sqlite` at the repository root.
Durable owner: exp582's native Marin storage.
Immutable backups live under `s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/operations/validation/` as `<UTC>-<uuid>.sqlite`.
Use the skill's transactional helper and `backup` command, then upload with S3FS using the CoreWeave endpoint and current Secret Manager credentials.
Verify size and SHA-256 by reading the uploaded object before any next external action.
Recover by listing backups newest first, downloading a complete object, checking SQLite integrity through the helper, and using the newest valid backup.
Never overwrite a backup key.

## Operating Policy

Use heartbeat 30 minutes, ordinary reslice after one hour, restart after three hours, and pending target limit one.
During active bounded validation, refresh fleet availability at least every five minutes and before placement decisions.
If W&B shows no new progress for ten minutes, the current target has no free whole-node capacity and another validated H100 target has capacity, permit an earlier move based on that combined evidence.
Cancel the current root and verify both the root and its GPU child/gang are terminal before replacement; preemption alone remains insufficient.
These short validation jobs have no unattended sweep loop; observe startup, perform useful local work, and query W&B during this active validation pass.
Preemption alone is not a reason to replace a job.
Investigate concrete startup failures before retrying; stop if the safe response is unclear.
Completion requires W&B `run_progress >= 1` and direct reads of the expected checkpoint metadata and manifest.

Eligibility below permits bounded smoke dispatch only; memory fit and training suitability require runtime evidence.

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| cw-us-east-02a | H100 | 1 | 8 | eligible | User authorized bounded routing diagnostics; new backends require canaries |
| cw-us-east-02a | H100 | 2 | 16 | unvalidated | — |
| cw-us-east-02a | H100 | 4 | 32 | unvalidated | — |
| cw-us-east-02a | H100 | 8 | 64 | unvalidated | — |
| cw-rno2a | H100 | 1 | 8 | eligible | Same validated whole-node H100 profile; shared CoreWeave storage and explicit gang environment verified from framework/KB; bounded canary only |
| cw-rno2a | H100 | 2 | 16 | unvalidated | — |
| cw-rno2a | H100 | 4 | 32 | unvalidated | — |
| cw-rno2a | H100 | 8 | 64 | unvalidated | — |

## Change Record

2026-10-05: stopped further submissions after exact checkpoint restoration passed but subsequent update replay diverged, with high expert-assignment drop rates also unresolved.
All five dispatches are terminal and no checkpoint writer remains active.
The final observation and stop reason are saved in the verified immutable SQLite backup recorded in the validation logbook.
Pretrained restoration and GPU evaluation are still pending; no placement is approved for production training.
2026-10-05: user reviewed the excessive dropping and authorized experiments to fix it, reopening bounded validation within the existing H100 scope and deadline.
Production training remains gated on validation and review.

2026-10-05: existing pooled EP passed both 600-update controls within the three-hour investigation window.
Both native Sonic evaluation controls and their synthetic CPU scoring/probe checks subsequently passed, reaching the requested stop condition.
Stop for user review; do not resume failed/superseded diagnostic trials or start production from this operations document.
The validation logbook records remaining scientific and production gates, numerical limitations, and the final durable backup.

2026-10-05: user accepted non-bitwise-identical updates while retaining exact checkpoint restoration.
User then approved pretrained DNA character encoding with original language vocabulary IDs and instructed continued validation.
Reopen only new character-mode training/evaluation identities on the same one-gang H100/batch/eczech scope; old trials remain immutable evidence.
Validate a fresh 600-update pretrained pooled-EP control from the original language checkpoint, its tokenizer-bound save/replay, then native Sonic evaluation and synthetic scoring/probes; stop afterward for review.

2026-10-05 21:25 UTC: user directed attention to available Reno capacity after prolonged absence of W&B progress.
Permit capacity-backed early reslicing during bounded validation and mark the identical eight-H100 Reno profile eligible for a canary; retain all numerical criteria, identities, one-writer constraints and batch/eczech requirements.
Renew the pass cutoff to 22:30 UTC for this relocation and evaluation, retaining one hour per job and no production authorization.

2026-10-05: the relocated character-mode canary completed all 600 updates, exact checkpoint restoration and replay controls on Reno.
Its native Sonic GPU evaluation and synthetic CPU scoring/probe checks passed, reaching the renewed stop condition.
Validation is stopped for user review; do not start production or automatically resubmit historical failed/superseded trials.
The logbook records the final verified state backup and remaining production and real-benchmark integration work.

2026-10-05 23:43 UTC: user authorized the proposed real-data evaluation pilot after reviewing the synthetic checks.
Reopen bounded evaluation only, retaining one eight-H100 gang, batch/eczech, one-hour jobs, five-minute capacity checks and the existing recovery rules.
Use new real-evaluation identities, resumable immutable output chunks, and existing step-600 checkpoints; stop after both conditions and CPU metrics/probes are verified or by the renewed cutoff.

2026-10-06: user required closer reuse of the PR20 evaluation code and challenged whole-corpus memory assumptions.
The revised pilot uses original Parquet row ranges and task/AF/probe reducers with a native JAX inference adapter; the source-pinned evaluation wheel and all inputs are checksum-bound.
Earlier real-evaluation attempts produced no inference progress and are terminal; do not blindly replace recurring W&B initialization failures.
Use separate v2 identities for the revised protocol; first verify their telemetry setup, then run a bounded inference/resume check before completing both pilots.
The existing cutoff, one-gang limit and batch/eczech requirement remain unchanged.

2026-10-06: both real-data evaluation pilots and original CPU metrics/probes passed, including the deliberate scratch pause/resume and independent saved-output review.
The authorized validation scope is complete and stopped for user review; do not submit further jobs or automatically resume any historical failed/superseded trial.
Production training remains outside this authorization.
The logbook records receipts and the final verified durable state backup.
