# Throughput

## Placement Rates

A target's `target_rate` measures how quickly it advances training after charging all wall-clock time spent there:

```text
target_rate = total increase in run_progress high-water / total dispatch hours
```

Each increase counts once, on the dispatch that first establishes it.
Dispatch time includes queueing, startup, compilation, evaluation, checkpoints, preemption, stalls, failures, and zero-progress attempts.

The persistence helper calculates one rate per `(cluster, GPU, nodes, GPUs)`.
Use the emitted value rather than routine SQL.
Prefer higher measured rates when other evidence is comparable, but combine them with current progress, pending headroom, and fleet capacity.
Do not normalize the rate by GPU count; rough compute-equivalent gangs are only cold-start evidence.

## Determine Liveness

- `training now`: W&B state is `running`.
- `recent progress`: the `run_progress` high water increased within `reslice_after`.
- `pending`: the current dispatch has not made progress within `reslice_after`.
- Iris running: the dispatch may execute; never proof of training.

Completion requires `run_progress >= 1` and a reachable expected checkpoint.

## Report Every Heartbeat

Report in session chat with enough detail to audit what is running, what changed, and what happens next.
Include:

- UTC observation time.
- Submitted and W&B-running dispatches and GPUs, plus current free/total capacity on relevant clusters.
- Complete, recently progressing, pending, and not-yet-running trials.
- Each active trial's target, W&B state, progress and change, and time since increase.
- Every submit, stop, restart, or reslice and its operational reason.
- Failed, stalled, and unregistered runs; isolated/systemic classification and action due.
- Relevant target rates, productive/pending counts, fleet headroom, and blockers.
- A specific scheduled next-check time and the actions expected if nothing improves.

Include checkpoint/resume evidence, unschedulable targets, priority violations, client-floor failures, or decisions needing operator approval when relevant.
Do not hide a problem inside aggregate counts.

## Keep The Heartbeat In Chat

- Never copy observations, status, progress, actions, or the next check into Operations.
- Never store the heartbeat report in SQLite.
- Never create another log, runbook, or recurring status file.
