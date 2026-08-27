# Persistence

One local SQLite database is the durable record of dynamic sweep state. Its model
covers heartbeat reconciliation, placement ranking, recovery, completion, and reporting.

## Concepts

| Concept | Essential facts |
| --- | --- |
| Sweep | Schema version and creation time; the file path identifies the sweep |
| Trial | Opaque ID and parameters, W&B ID, checkpoint root, status, and `run_progress` high water |
| Dispatch | Immutable attempt, trial, Iris ID, cluster, GPU, nodes, total GPUs, priority, redacted command, clocks, and end state |
| Observation | UTC time, trial and dispatch, W&B state, `run_progress`, and binary Iris running state |
| Event | UTC time, kind, associated identities, and concise evidence |

## Relationships And Invariants

- A trial owns one W&B and checkpoint identity and at most one active dispatch.
- A trial's dispatches have persisted attempt numbers; never parse names for identity.
- All persisted times are UTC.
- Dispatches retain stopped, failed, and superseded placement history.
- Commands and experiment parameters contain no secrets.
- A completed trial has `run_progress >= 1`, a verified checkpoint, and no active dispatch.

Operations owns target eligibility and policy. SQLite owns observed facts and action
history; it does not own transient fleet utilization, rankings, or the Operations
`Change Record`.

## Supported Decisions

- **Placement:** historical `target_rate`, productive and pending counts per exact
  `(cluster, GPU, nodes, GPUs)` target.
- **Recovery:** dispatch clocks begin at submission or latest progress; the trial
  clock survives restarts and reslices and resets only on progress.
- **Liveness:** W&B establishes training state and progress; Iris answers only
  whether the exact dispatch is running.
- **Accounting:** active placement separates submitted GPUs from GPUs on W&B-running work.
- **Failures and completion:** attempt history and observations distinguish isolated
  failures, repeated failures, abandonment, and verified completion.

## Control Snapshot

```bash
uv run .agents/skills/run-training-sweep-cw/scripts/persistence.py \
  snapshot scratch/<sweep>/expXXX_sweep.sqlite --reslice-after-hours <hours>
```

The read-only JSON snapshot includes every unfinished trial, active dispatches,
progress clocks, latest observations, target rates and pending counts, fleet
accounting, actionable conditions, event counts, and a bounded event window.
Snapshot generation fails when identities, intervals, progress high-water marks,
or completion state are inconsistent.

## Event History

Expected event kinds include:

- `dispatch_submitted`, `dispatch_stopped`, `dispatch_terminal`
- `target_unschedulable`, `wandb_registered`, `progress`, `stall_started`
- `failure_retryable`, `systemic_failure`, `restart`, `reslice`
- `trial_completed`, `checkpoint_verified`, `client_floor_failed`

Action events carry concise evidence. Heartbeat prose and no-change explanations
remain in session chat.

## Helper

```bash
uv run .agents/skills/run-training-sweep-cw/scripts/persistence.py --help
```
