# Throughput

## Placement Rates

A target's `target_rate` measures how quickly it advances training after charging
all wall-clock time spent on that target. Assume all trials in the sweep are
comparable for placement.

```text
target_rate = total increase in run_progress high-water / total dispatch hours
```

Each high-water increase counts once, on the target whose dispatch first establishes
it. Dispatch hours run from submission through the current heartbeat for active work,
or through the recorded end and final observation for ended work. Queueing, startup,
compilation, evaluation, checkpoints, preemption, stalls, failures, and zero-progress
attempts all count.

For example, `0.12` progress across eight dispatch-hours gives a `target_rate` of
`0.015` progress per hour, including any zero-progress time in those eight hours.

The persistence helper calculates one rate per `(region, slice, chips)` across the
sweep. Use the emitted value; do not reconstruct it with routine SQL. Recompute every
heartbeat and prefer a higher rate when other evidence is comparable, but combine it
with current target progress, pending headroom, and agent judgment. Never normalize
by chip count.

For fleet progress under regional racing, use the maximum regional `run_progress`
for each logical trial. Do not sum replicas.

## Determine Liveness

- `training now`: W&B state is `running`.
- `recent progress`: the `run_progress` high-water mark increased within
  `reslice_after`.
- `pending`: the current dispatch has not made progress within `reslice_after`.
- Iris running: the dispatch may execute; never proof of training.

Completion requires `run_progress >= 1` and a reachable expected checkpoint.

## Report Every Heartbeat

Report in the session chat. Choose the clearest format for the current fleet; do not
follow a fixed template. Include enough detail that the operator can audit what is
running, what changed, and what will happen next.

Always include:

- **Control context:** UTC observation time.
- **Fleet accounting:** submitted dispatches and chips, W&B-running regional runs
  and chips, and the gap between submitted and training capacity.
- **Trial accounting:** counts of complete, recently progressing, pending, and
  not-yet-running trials. Identify affected trials, not only aggregate counts.
- **Active progress:** for each live or recently active regional run, give trial,
  region, slice, W&B state, current and high-water `run_progress`, change since the
  prior heartbeat, and time since the last increase.
- **Actions:** every submit, stop, restart, reslice, relocate, or race cancellation
  since the prior heartbeat, with its operational reason. If nothing changed, say
  why waiting is still the best action.
- **Recovery:** every failed, stalled, or unregistered run; whether failures appear
  isolated or systemic; its time since progress; the next restart/reslice/relocate
  threshold; and the action due.
- **Placement evidence:** each relevant target's productive and pending dispatches,
  `target_rate`, time to first progress when known, and any eligibility or regional
  input blocker affecting the next placement.
- **Next check:** a specific UTC time backed by a scheduled trigger, plus the actions
  expected if progress remains unchanged.

Include when relevant:

- Replica progress and the condition for ending a regional race.
- Checkpoint or resume status when recovering, reslicing, relocating, or completing.
- Iris `unschedulable` targets, client-floor failures, priority constraints, or
  decisions needing operator approval.
- An apparent leader when useful. Let experiment context determine what “leader”
  means; this skill does not define or require an optimization metric.

Do not hide a problem inside aggregate counts. Omit facts only when genuinely
inapplicable, not because they are unchanged.

## Keep The Heartbeat In Chat

- Never copy observations, status, progress, actions, or the next check into Operations.
- Never store the heartbeat report in SQLite.
- Never create another log, runbook, or recurring status file.
