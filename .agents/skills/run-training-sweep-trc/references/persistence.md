# Persistence

One local SQLite database is the durable record of dynamic sweep state. Its model
covers heartbeat reconciliation, placement ranking, recovery, regional racing,
completion, and reporting.

## Concepts

The bundled SQLite schema represents these concepts:

| Concept | Essential facts |
| --- | --- |
| Sweep | Identity, schema version, creation and start times |
| Trial | Opaque ID, experiment parameters, current status |
| Run | Trial, region, W&B ID, checkpoint root, status, winner flag, `run_progress` high-water state, regional progress clock |
| Dispatch | Immutable attempt, run, Iris ID, actual TPU slice and chips, priority, redacted command, submission, progress clock, and end state |
| Observation | UTC time, run and dispatch, W&B state, `run_progress`, and binary Iris running state |
| Event | UTC time, kind, associated identities, concise evidence |

Current status and high-water fields may be materialized for simple heartbeat
queries; timestamped observations, immutable dispatches, and events preserve the
evidence behind them.

## Relationships And Invariants

- A trial has at most one run per region and at most one winning run.
- A run has a sequence of uniquely numbered dispatches and at most one active
  dispatch.
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

- `dispatch_submitted`, `dispatch_stopped`, `dispatch_terminal`
- `target_unschedulable`
- `wandb_registered`, `progress`, `stall_started`
- `failure_retryable`, `systemic_failure`
- `restart`, `reslice`, `relocate`
- `race_won`, `race_lost`
- `trial_completed`, `checkpoint_verified`
- `client_floor_failed`

Action events carry concise evidence. Heartbeat prose and no-change explanations
remain in session chat.

## Helper

The bundled helper initializes the model, records events, emits snapshots, and
checks integrity:

```bash
uv run .agents/skills/run-training-sweep-trc/scripts/persistence.py --help
```
