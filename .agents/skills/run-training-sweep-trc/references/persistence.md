# Persistence

One local SQLite working copy records dynamic sweep state.
Its immutable backups under the durable owner recorded in Operations make that state recoverable across worktree, VM, and session loss.
The model covers heartbeat reconciliation, placement ranking, recovery, regional racing, completion, and reporting.

## Concepts

The bundled SQLite schema represents these concepts:

| Concept | Essential facts |
| --- | --- |
| Sweep | Identity, schema version, creation and start times |
| Trial | Opaque ID, experiment parameters, current status |
| Run | Trial, region, W&B ID, checkpoint root, status, winner flag, `run_progress` high-water state, regional progress clock |
| Dispatch | Immutable intent and attempt, run, exact Iris ID, actual TPU slice and chips, priority, redacted command, submission reconciliation, progress clock, and end state |
| Observation | UTC time, run and dispatch, W&B state, `run_progress`, and binary Iris running state |
| Event | UTC time, kind, associated identities, concise evidence |

Current status and high-water fields may be materialized for simple heartbeat
queries; timestamped observations, immutable dispatches, and events preserve the
evidence behind them.

## Relationships And Invariants

- A trial has at most one run per region and at most one winning run.
- A run has a sequence of uniquely numbered dispatches and at most one active
  dispatch.
- An active dispatch is either an unsubmitted/unreconciled intent or a confirmed Iris submission.
- All persisted times are UTC.
- W&B and Iris IDs remain opaque and unique. Attempt numbers come from persisted
  state, never parsed names.
- A dispatch's run supplies its region; the dispatch supplies the slice
  and chip count actually submitted. Historical placement therefore survives
  changes to the Operations target grid.
- Dispatches, observations, and events retain stopped, failed, superseded, and
  losing history. Stop-and-replace, winner-and-cancel, and completion transitions
  are atomic.
- Commands and experiment parameters contain no API keys, tokens, or secret values.
- Completion is terminal; it is idempotent only for the recorded winner and never permits another regional run, dispatch, or replacement winner.

## Supported Mutations

Use only the helper's transactional mutations for maintained operation; do not edit SQLite with ad hoc SQL.

- `trial-add` registers a logical trial.
- `run-add` registers a regional run and its stable W&B/checkpoint identity.
- `dispatch-intent` atomically reserves the next attempt and exact Iris job name before submission.
- `dispatch-confirm` marks that exact intent as accepted by Iris.
- `dispatch-end` records a verified terminal or definitely-not-submitted result.
- `observe` records one attributed W&B/Iris observation and advances the regional progress high-water mark atomically.
- `trial-complete` atomically selects the verified winner and closes losing regional runs after all dispatches are terminal.
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

Operations owns target eligibility and policy. SQLite owns observed facts and
action history; it does not own transient rankings or duplicate the Operations
`Change Record`.

## Supported Decisions

The model supports these recurring queries:

- **Placement:** each target combines historical `target_rate` with current productive
  and pending dispatch counts. Rates include zero-progress and pre-W&B wall time;
  rankings remain ephemeral.
- **Recovery:** a dispatch clock begins at submission or its latest progress. The
  regional clock begins at first submission or the regional run's latest progress;
  same-region replacements preserve it.
- **Liveness:** the latest observation attributed to the active dispatch establishes
  its W&B state and progress; its Iris flag answers only whether it is running.
- **Accounting:** active dispatch placement yields submitted chips; that dispatch's
  W&B state separates submitted chips from chips currently training.
- **Failures:** attempt history, end state, failure events, and placement make
  isolated retries and correlated failures distinguishable.
- **Racing and completion:** run high-water progress, winner state, and
  checkpoint verification identify race winners and completed trials.
- **Reporting:** observation history supports progress since the prior heartbeat,
  time since progress, time to first progress, and action history.

## Control Snapshot

Rebuild a compact snapshot at the start of a heartbeat to inventory work, then
again after recording fresh observations to support decisions:

```bash
uv run .agents/skills/run-training-sweep-trc/scripts/persistence.py \
  snapshot scratch/<sweep>/expXXX_sweep.sqlite --reslice-after-hours <hours>
```

The JSON includes every unfinished trial and its runs, current dispatches, regional
and dispatch progress clocks, latest observations, target rates and current
productive/pending counts, fleet accounting, actionable conditions, event counts,
and a bounded recent-event window. Coverage counts expose omitted stable history.

The snapshot is read-only and ephemeral. It summarizes history inside SQLite; it
does not become another status file or durable report. Query raw rows only to
investigate a specific decision. Snapshot generation fails when core identities,
dispatch intervals, progress high-water marks, or completion state are inconsistent.

## Event History

Expected event kinds include:

- `dispatch_intent`, `dispatch_submitted`, `dispatch_not_submitted`, `dispatch_terminal`
- `target_unschedulable`
- `wandb_registered`, `progress`, `stall_started`
- `failure_retryable`, `systemic_failure`
- `restart`, `reslice`, `relocate`
- `race_won`, `race_lost`
- `trial_completed`, `checkpoint_verified`
- `client_floor_failed`

Action events carry concise evidence.
Heartbeat prose and no-change explanations remain in session chat; research decisions and milestones belong in the task logbook.

## Helper

The bundled helper initializes the model, records events, emits snapshots, and
checks integrity:

```bash
uv run .agents/skills/run-training-sweep-trc/scripts/persistence.py --help
```

Use `backup <database> <new-local-snapshot>` to create a consistent local file before uploading it to the recorded durable prefix.
Never overwrite an existing snapshot key.
