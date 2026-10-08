# exp582 CoreWeave Training Sweep Operations

> This document governs actual training under the `run-training-sweep-cw` skill.
> Read it in full at every heartbeat.
> The user's October 6 authorization to start six trials supersedes the earlier validation-only stop.

## Invariants

Use H100s, whole-node gangs, `--priority batch`, and `--user eczech` for every root; children explicitly request batch priority too.
Keep exactly one active writer per logical trial, with stable W&B and checkpoint identities across placement changes.
Use the user's credentials for `eric-czech/marin`.
W&B establishes training progress; Iris establishes dispatch liveness and placement, and diagnoses concrete failures.
Do not replace work merely because Iris reports a preemption.
Stop if direct CoreWeave S3 access fails.

## Sweep Definition

[`train.py`](src/exp582_moe/train.py) defines the six authorized condition/LR combinations and permanent checkpoint identities.
[`corpus.py`](src/exp582_moe/corpus.py) preserves the exp472 shuffle and absolute-occurrence augmentation semantics; the adjacent inventory binds all input objects.
[`schedule.py`](src/exp582_moe/schedule.py) owns the Hero heuristics and agreed token schedule.
No additional seeds, conditions, or stages are authorized by this launch.
The October 6 prioritization selects pretrained 1× and random 2× by their latest logged held-out losses within each initialization.
Finish both selected trials before resuming pretrained 0.5×, pretrained 2×, random 0.5× or random 1×.
Keep the other four trials paused even when inventory reports them as unplaced; they retain their original identities and full-state checkpoints.
Both selected trials completed on October 7.
The user then superseded the earlier resume-four policy: keep the other four trials paused while evaluation focuses exclusively on the two completed leaders' final update-21,440 checkpoints.
Do not admit another training dispatch until the user explicitly returns the sweep to training.
The user requests 128-H100 placements for the selected pair where feasible.
Qualify that shape before admission, preserving the scientific configuration, global batch and token clocks; until then keep the selected trials on their validated targets.
GPU placement must not change global batch, optimizer or QB semantics, token schedule, initialization, or data order.
The user authorized the shorter schedule on October 6; `schedule_transition.json` pins its full-state resume sources and the approved late peak update for pretrained 0.5×.
Original checkpoints remain immutable; the active schedule writes beneath the same trial root in `short-linear-v2/`.

## Operator Choices

The user delegates capacity-based allocation for all six trials on the two H100 clusters, excluding GB200 and the CI cluster.
At 22:26 UTC on October 6, the user requested cancellation of our Reno jobs after its dashboard again reported zero total H100s.
Actual training admissions in Reno were suspended pending reassessment of its inventory and scheduling health.
At 00:28 UTC on October 7, the user requested reassessment after Reno appeared to recover.
The controller again reported 512 total H100s and 133 free, with a recent peer contact but a capacity observation about six minutes old.
The bounded qualification exception below tested recovery while both actual trials continued in East.
Full-model H100 qualification has now passed for both initializations, including 26 native CP2 updates, the existing held-out evaluation, and exact full-state checkpoint restoration.
Reopen the qualified Reno targets under the normal 90-second capacity limit.
Before moving a progressing trial, compare measured throughput gain against its remaining work, checkpoint rollback, and observed startup costs; a successful qualification alone does not justify a move.
The overall ceiling remains 384 GPUs; current policy prioritizes the two selected trials, with up to 128 H100s each after shape qualification.
Actual admissions must fit current reported free capacity.
This is an allocation ceiling chosen within that delegation, not a reservation.
The user authorized completion of the six declared first-stage trials and supplied no sweep-wide calendar cutoff.
Do not apply the skill's suggested two-week deadline as an unapproved early stop.
Individual Iris dispatches have a fourteen-day timeout; if a timeout occurs before the stage finishes, recover from the verified checkpoint under the same trial identity and placement policy.
Use this file as the authoritative training Operations document; the earlier validation Operations remains historical.

The working SQLite database is `scratch/exp582-training/exp582_training.sqlite`.
Its durable owner is `s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/operations/training/`.
Use immutable `<UTC timestamp>-<UUID>.sqlite` keys, upload with s3fs through `https://cwobject.com`, and verify length and SHA-256 by direct download.
The ignored `persist.py` wrapper publishes a consistent SQLite backup after each helper mutation and before the next external action.
If local state is lost, list that exact prefix and restore the newest backup passing length, checksum when recorded, and SQLite integrity checks.
Never overwrite backup objects or fall back to the validation database.

## Operating Policy

Use `heartbeat_every=30m`, `reslice_after=1h`, and `restart_after=3h`.
Set `pending_target_limit=3` to permit the explicitly requested six-trial initial launch across shared target shapes; never spend the same capacity twice.
Begin the thirty-minute cadence only after W&B proves that at least one actual trial has advanced.
The current chat lacks a persistent automation tool; use active-session, time-based agent passes and do not claim a background automation exists.
Each pass reads this document and SQLite, observes the entire W&B fleet, checks exact dispatch liveness, refreshes utilization, and makes one coordinated placement/recovery decision.
Helpers may collect facts or persist exact chosen actions; they must never choose retries, stops, placements, or convergence.
On replacement dispatches, credit only training history whose `_timestamp` is later than submission; W&B's existing summary and `running` state can still describe the previous worker during startup.
The observation helper checks this timestamp before recording progress against a new dispatch.
For revised dispatches also require `train/schedule_revision=2` on a fresh training metric.
The shorter target rebases W&B `run_progress`; never interpret that percentage jump as new training.
Do not use SQLite placement rates spanning schedule revisions to rank placements.
Use post-migration optimizer-step advancement and measured update throughput, preserving the original observations and recording the migration explicitly.

Both clusters passed prior eight-H100 model/checkpoint validation.
The authorized actual trials established two-windows-per-device memory fit, native updates, checkpoint writes and held-out evaluation on the 32-GPU profile in both conditions.
If that shape cannot fit, retain the same identity and scientific configuration on 64 GPUs after its old root and child are terminal.
Multi-node topology is checked in the actual worker; each EP8 group must stay within one node.

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| cw-rno2a | H100 | 4 | 32 | eligible | Previously qualified shape; Reno scheduling recovery verified October 7. |
| cw-rno2a | H100 | 8 | 64 | eligible | Previously qualified shape; Reno scheduling recovery verified October 7. |
| cw-us-east-02a | H100 | 4 | 32 | eligible | Actual pretrained trial qualified the two-window profile. |
| cw-us-east-02a | H100 | 8 | 64 | eligible | Validated one-window-per-device model; explicit replicated EP8 mesh. |
| cw-rno2a | H100 | 16 | 128 | eligible | Both initializations passed full-model CP2 updates, evaluation and exact checkpoint restoration. |
| cw-us-east-02a | H100 | 16 | 128 | unvalidated | CP2 launcher preserves batch 64; reduced CPU oracle passed, full H100 placement smoke remains. |

Use [`placement_smoke.py`](src/exp582_moe/placement_smoke.py) for bounded full-model qualification while the selected trials continue.
Track these distinct smoke identities in the existing validation database and publish its immutable backups under `operations/validation/` using `scratch/exp582-validation/persist.py`.
Include their reserved GPUs when applying this document's overall allocation ceiling and current fleet plan.
They read permanent peak checkpoints and write only separate temporary validation prefixes.
For this bounded qualification only, allow one pending 128-H100 smoke when the peer is reachable, its last controller contact is at most 60 seconds old, and its reported free capacity fits the gang with an observation age at most ten minutes (`utilization.py --max-age-seconds 600`).
This explicit exception accommodates Reno's delayed backend refresh without stopping progressing training; batch scheduling remains the final admission gate.
Report the capacity observation's age and do not apply this exception to stopping or expanding an actual trial.
On October 6 the operator also supplied a Reno dashboard observation of 179 free H100s at 21:23 UTC, while later federation inventory incorrectly advertised zero total H100s alongside thousands of running tasks.
For that degraded-feed incident, permit one 128-H100 batch qualification to queue against the operator-reported headroom, with Iris enforcing actual admission; do not treat the zero-total response as a healthy capacity snapshot or stop either selected training trial.

Monitor token progress, loss and gradient/update norms, per-layer and aggregate expert drops/load, actual LRs, data cursor, throughput/loading time, memory, held-out loss and its drop fraction, and committed checkpoint age.
Transient early QB redistribution is expected from the validation runs; sustained or returning drops require investigation.
Use held-out loss to compare LRs within a condition; vocabulary differences limit direct cross-condition loss comparisons.
When repeated checkpoint rollbacks erase forward progress, adjust checkpoint cadence based on observed save cost and uninterrupted training time.
Deploy that operational change after a fresh committed checkpoint or a subsequent interruption, preserving the scientific binding and avoiding unnecessary loss of live work.
An isolated startup failure can be corrected and retried; correlated failures require a shared-cause diagnosis before replacements.
Expand healthy trials only when the entire destination gang is already reported free, without counting GPUs that would be released by stopping the current gang.
Require measured throughput benefit and a verified recent checkpoint before such an expansion; dynamic capacity can still change before admission.
Before stopping a dispatch, persist its final available W&B progress observation so work preceding a move remains attributed to the original allocation.
For every replacement, first verify the old exact root and child are terminal and reconcile any uncertain intent.
Completion requires `run_progress >= 1` and direct verification of the committed final checkpoint; W&B state alone is insufficient.
Verify the revised final checkpoint and permanent peak checkpoint against the pinned revised scientific binding.

## Change Record

2026-10-06 02:57 UTC — Free capacity changed during a planned 32-to-64 expansion, forcing recovery on the original shape.
Healthy expansions now require the full destination gang to be free before cancellation, reducing exposure to races over released capacity.

2026-10-06 03:00 UTC — A resumed W&B identity retained its previous worker's summary while the replacement was loading.
The observation helper now excludes pre-submission metric timestamps from the replacement's progress accounting.

2026-10-06 04:10 UTC — Removed the inferred two-week sweep cutoff, which the user had not requested.
The authorized stopping condition remains completion of the declared first stage in all six trials, subject to later user instructions.
