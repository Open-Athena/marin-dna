# Persistence

One local SQLite working copy records dynamic sweep state.
Its immutable backups under the durable owner recorded in Operations make that state recoverable across worktree, VM, and session loss.
The model covers heartbeat reconciliation, placement ranking, recovery, completion, and reporting.

## Concepts

| Concept | Essential facts |
| --- | --- |
| Sweep | Schema version and creation time; the file path identifies the sweep |
| Trial | Opaque ID and parameters, W&B ID, checkpoint root, status, and `run_progress` high water |
| Dispatch | Immutable intent and attempt, trial, exact Iris ID, cluster, GPU, nodes, total GPUs, priority, redacted command, submission reconciliation, clocks, and end state |
| Observation | UTC time, trial and dispatch, W&B state, `run_progress`, and binary Iris running state |
| Event | UTC time, kind, associated identities, and concise evidence |

## Relationships And Invariants

- A trial owns one W&B and checkpoint identity and at most one active dispatch.
- An active dispatch is either an unsubmitted/unreconciled intent or a confirmed Iris submission.
- A trial's dispatches have persisted attempt numbers; never parse names for identity.
- Persist times as canonical UTC ISO-8601 strings and progress as finite, nonnegative values.
- Dispatches retain stopped, failed, and superseded placement history.
- Commands and experiment parameters contain no secrets.
- A completed trial has `run_progress >= 1`, a verified checkpoint, and no active dispatch.
- Completion is terminal; it is idempotent only for the already completed trial and never permits a later dispatch.

## Supported Mutations

Use only the helper's transactional mutations for maintained operation; do not edit SQLite with ad hoc SQL.

- `trial-add` registers a logical trial and its stable W&B/checkpoint identity.
- `dispatch-intent` atomically reserves the next attempt and exact Iris job name before submission.
- `dispatch-confirm` marks that exact intent as accepted by Iris.
- `dispatch-end` records a verified terminal or definitely-not-submitted result.
- `observe` records one attributed W&B/Iris observation and advances the progress high-water mark atomically.
- `trial-complete` requires progress at least `1`, a verified checkpoint time, and no active dispatch.
- `event` records exceptional evidence not already captured by a state mutation.
- `backup` creates a consistent, integrity-checked immutable copy and prints its size and SHA-256 checksum.

Every mutation commits its associated event in the same transaction.

## Intent Before Submit

For every submission:

1. Run `dispatch-intent` with an exact unique Iris job name and the redacted command.
2. Run `backup`, upload the resulting file to a new immutable key under the Operations prefix, and verify the recorded size and SHA-256.
3. Submit only the exact persisted Iris job name.
4. On an unambiguous acceptance, run `dispatch-confirm` and publish another backup.
5. On an unambiguous pre-submission rejection, run `dispatch-end` with a not-submitted outcome and publish another backup.
6. On timeout, interruption, or an unknown result, leave the intent active.
   Query only that exact Iris name when service returns; never create a replacement until the intent is confirmed or ended.

At heartbeat entry, restore the newest valid immutable backup if needed and reconcile every active intent before any other action.

Operations owns target eligibility and policy.
SQLite owns observed facts and action history; it does not own transient fleet utilization, rankings, or the Operations `Change Record`.

## Supported Decisions

- **Placement:** historical `target_rate`, productive and pending counts per exact `(cluster, GPU, nodes, GPUs)` target.
- **Recovery:** dispatch clocks begin at submission or latest progress; the trial clock survives restarts and reslices and resets only on progress.
- **Liveness:** W&B establishes training state and progress; Iris answers only whether the exact dispatch is running.
- **Accounting:** active placement separates submitted GPUs from GPUs on W&B-running work.
- **Failures and completion:** attempt history and observations distinguish isolated failures, repeated failures, abandonment, and verified completion.

## Control Snapshot

```bash
uv run .agents/skills/run-training-sweep-cw/scripts/persistence.py \
  snapshot scratch/<sweep>/expXXX_sweep.sqlite --reslice-after-hours <hours>
```

The read-only JSON snapshot includes every unfinished trial, active dispatches, progress clocks, latest observations, target rates and pending counts, fleet accounting, actionable conditions, event counts, and a bounded event window.
Snapshot generation fails when identities, intervals, progress high-water marks, or completion state are inconsistent.

## Event History

Expected event kinds include:

- `dispatch_intent`, `dispatch_submitted`, `dispatch_not_submitted`, `dispatch_terminal`
- `target_unschedulable`, `wandb_registered`, `progress`, `stall_started`
- `failure_retryable`, `systemic_failure`, `restart`, `reslice`
- `trial_completed`, `checkpoint_verified`, `client_floor_failed`

Action events carry concise evidence.
Heartbeat prose and no-change explanations remain in session chat; research decisions and milestones belong in the task logbook.

## Helper

```bash
uv run .agents/skills/run-training-sweep-cw/scripts/persistence.py --help
```

Use `backup <database> <new-local-snapshot>` to create a consistent local file before uploading it to the recorded durable prefix.
Never overwrite an existing snapshot key.
