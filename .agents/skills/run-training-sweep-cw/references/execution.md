# Execution

## Use Four Identities

- **Logical trial:** one opaque experiment configuration.
- **Run:** the trial's single W&B and shared checkpoint identity.
- **Dispatch:** one immutable Iris submission attempt.
- **Target:** one allowed cluster, GPU variant, and gang size in Operations.

Never parse W&B or Iris names.

## Keep Iris Operations Simple

Submit one exact unique root through `--cluster marin`, with an exact `--target-cluster`, `--priority batch`, and `--user`.
Stop or inspect only an exact root, read [fleet utilization](utilization.md), and recognize exact `unschedulable` results.

W&B is the truth for training progress.
Iris reports execution and capacity, not progress.
Inspect logs, parent/child details, retry history, or pending reasons only for a concrete failure or placement question.

An Iris timeout or service failure blocks actions.
Schedule one later pass that checks only Iris, then reconcile affected exact dispatch intents and submissions before resuming.
Do not infer that a timed-out submission failed and do not create a replacement.

## Use Visible Capacity

Read one valid fleet snapshot after W&B observation and before building the fleet plan.
At batch priority, use reported free GPUs; higher-priority holds explain the gap but are not available to reclaim.
Never combine capacity from separate backends to fit one gang.

Respect the operator's GPU cap and whole-node targets.
Reserve capacity for every planned dispatch so later choices in the same pass cannot spend it again.
Refresh utilization and replan after material stops or submissions.
The scheduler remains the final admission authority, so treat submitted work as pending until W&B proves progress.

Use credible free capacity to reduce wall-clock time.
This may start unplaced trials or enlarge a healthy run when the expected benefit exceeds restart overhead.
Let measured throughput and current progress override rough peak-FLOP equivalence.

## Handle Unschedulable Targets

On an unproven target, verify the requested cluster, GPU, nodes, and total GPUs.
Record `target_unschedulable`, mark only that exact target `ineligible`, and reslice to another eligible target.
If the target worked previously, investigate before changing eligibility; simultaneous results may indicate a systemic problem.

## Make Every Dispatch Unique

- Use `dispatch-intent` to assign the attempt and persist one exact unique Iris root name before submission.
- Publish the intent-bearing SQLite backup to the durable owner before calling Iris.
- Submit only the exact persisted name.
- Use `dispatch-confirm` only after Iris unambiguously accepts it.
- If the result is unknown, keep the intent active and reconcile that exact name before any replacement.
- Give every root one immutable dispatch row and use `dispatch-end` for its terminal result.
- Stop and verify the current root before replacing it.
- Allow at most one active dispatch per trial.

Resume comes from the trial checkpoint, not the Iris name.

## Classify Failures Before Retry

Observe the whole W&B fleet before acting on a failed run.
Retry an isolated failure from the same checkpoint after stopping its exact root.
If failures recur or correlate across trials or targets, pause blind replacement and investigate a shared cause.
Resume with a concrete basis, contain affected work, or ask the operator when the safe response is unclear.

## Rebalance Every Heartbeat

Use `heartbeat_every=30m`, `reslice_after=1h`, `restart_after=3h`, and `pending_target_limit=1` unless evidence warrants a change.
Protect a dispatch with W&B progress inside `reslice_after`; consider every other dispatch for reslicing.
Restart when neither moving nor progressing for `restart_after`.
A replacement starts a new dispatch clock; the trial clock resets only on progress.

A target's pending count is its active dispatches without recent progress.
Treat a planned dispatch as pending until it proves progress, and keep the count at or below `pending_target_limit` when admitting it.
This is an admission rule, not a reason to evict progressing work later.

For each reslice candidate:

1. Rank different eligible targets with valid free capacity and pending headroom.
2. Combine `target_rate`, current progress, prior runs, gang fit, and useful exploration.
3. Reslice when a credible target is better; otherwise restart the current target when due or leave it unchanged with a concrete reason.

- **Restart:** same cluster, GPU, nodes, and checkpoint; new dispatch.
- **Reslice:** different cluster, GPU, or nodes; same run and checkpoint.

Stop before replacing and replan after every material action.
Never change priority to solve scarcity, and never replace a dispatch for preemption alone.

## Handle Client Revision Rejections

Apply this only when a submission says the Iris client is older than the required revision floor.
Record `client_floor_failed`, then stop new submissions and recovery-driven stops.
Follow `marin-experiment` to resolve a coherent current Marin and Iris dependency set, regenerate the lockfile, validate the launch import, and prove a canary submission is accepted.
Do not change `BUILD_DATE` alone or restart the shared Iris cluster.
If the dependency update is outside the approved task scope, stop and ask the operator with the reported and required dates.
