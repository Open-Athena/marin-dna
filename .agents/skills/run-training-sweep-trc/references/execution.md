# Execution

## Use Four Identities

- **Logical trial:** one opaque experiment configuration.
- **Regional run:** one logical trial in one region; owns W&B and checkpoint identity.
- **Dispatch:** one immutable Iris submission attempt.
- **Target:** one allowed region, TPU slice, and chip count; lives in the operations document.

Never parse W&B or Iris names.

## Keep Iris Operations Simple

Allowed routine job actions:

- Submit an exact unique job.
- Stop an exact job.
- Ask whether an exact dispatch is running.
- Recognize exact `unschedulable` results.
- Treat every other successfully returned job state as not-running for liveness purposes.

Iris never proves training progress. Do not routinely inspect logs, summaries, task counts,
parent-child structure, retry history, pending reasons, or cluster capacity. W&B is the truth.

An Iris timeout or service failure blocks all sweep work. Do not infer whether the request
succeeded, inspect W&B, or take another action. Schedule one later pass that checks only Iris;
keep waiting until it responds, then reconcile the affected exact dispatch intents and submissions before resuming.
Do not create a replacement for a timed-out submission.

Inspect deeper only for a recorded reason, such as the same first-attempt failure reproducing
across regions or a dispatch command failing before W&B registration. Do not derive placement
policy from unfamiliar Iris internals.

## Handle Unschedulable Targets

On an unproven target, `unschedulable` normally exposes an invalid region and slice
combination admitted to the grid; it does not mean temporary scarcity.

1. Verify the dispatch requested the intended region, slice, and chip count.
2. Record `target_unschedulable`; do not retry the exact target.
3. Mark only that target `ineligible` in Operations and add a terse `Change Record`
   entry naming the grid change and its cause.
4. Reslice or relocate immediately to another eligible target.

Do not generalize one result to a region or TPU family. If the exact target worked
previously, pause it and investigate before changing eligibility. Treat simultaneous
results across previously valid targets as a systemic problem.

## Make Every Dispatch Unique

- Use `dispatch-intent` to assign the attempt and persist one exact unique Iris job name before submission.
- Publish the intent-bearing SQLite backup to the durable owner before calling Iris.
- Submit only the exact persisted name.
- Use `dispatch-confirm` only after Iris unambiguously accepts it.
- If the result is unknown, keep the intent active and reconcile that exact name before any replacement.
- Use a unique Iris job name, such as `<opaque-wandb-id>-<slice>-<unique-dispatch-id>`.
- Never recover attempt numbers or metadata by parsing that name.
- Give every Iris job one immutable dispatch row and use `dispatch-end` for its terminal result.
- Stop the current dispatch before replacing it.
- Allow at most one active dispatch per regional run.

Resume comes from the regional checkpoint, not the Iris name.

## Classify Failures Before Retry

Observe the whole W&B fleet before acting on a `failed` run.

- If the failure is isolated, stop its dispatch if still running and immediately submit a
  unique replacement on the same region and slice from the regional checkpoint.
- If failures recur after replacement or cluster across otherwise independent trials,
  regions, or targets, pause replacements and investigate a shared cause. Inspect deeper
  Iris details only with this reason recorded.
- Resume with a concrete basis, contain or stop affected work, or wait for operator
  direction when the safe response is unclear. Do not blindly retry.

## Rebalance Every Heartbeat

No command reveals available TRC capacity. Submission is the measurement. Begin an
unknown two-week sweep with:

```text
heartbeat_every = 30 minutes
reslice_after = 1 hour
restart_after = 3 hours
relocate_after = 3 days
pending_target_limit = 1
```

Confirm the four timing settings during the interview; record them and
`pending_target_limit` in Operating Policy. Retain the defaults unless a concrete
condition warrants changing them. Each heartbeat preserves dispatches with progress
within `reslice_after` and considers every other dispatch for reslicing. A dispatch
becomes restart-eligible after neither moving nor progressing for `restart_after`;
its regional run becomes relocation-eligible after no progress for `relocate_after`.
Only a new W&B `run_progress` high-water mark proves progress.

A replacement starts a new dispatch clock. Restarting or reslicing never resets the
regional clock; only progress does.

### Admit Destinations

A target's pending count is its active dispatches without progress during
`reslice_after`; productive dispatches do not count. Treat a planned dispatch as
pending until it proves progress. Its placement must leave the count at or below
`pending_target_limit`.

Build one fleet plan, reserving pending headroom after each proposed placement so
later decisions see it. This limit controls admission only: never stop progressing
work or evict dispatches merely because the count later rises.

### Choose An Action

For each reslice candidate:

1. Search for a different eligible target in the same region with pending headroom.
2. Rank credible targets with `target_rate` and judgment informed by current
   progress, prior experiments, optional fleet utilization, and useful exploration.
3. If no same-region move is justified and relocation is due, repeat the search in a
   different validated region.
4. If no move is justified and restart is due, restart the current target.
5. Otherwise leave the dispatch unchanged and state the reason.

Reslice by default when a credible alternative exists; a no-op requires a concrete
reason. Apply the same target accounting to new trials and requested replicas. Stop
before replacing, and replan when an action changes material facts.

- **Restart:** same region, slice, and regional checkpoint.
- **Reslice:** different target in the same region, same checkpoint.
- **Relocate:** different validated region, separate regional run from zero, no
  transferred data.

Recompute evidence every heartbeat and never copy rankings into Operations. Probe
with real trials; never change the priority band to solve scarcity.

## Race Regions Optionally

Default to one live regional run per logical trial.

With two replicas:

- Start distinct validated regions.
- Maintain two live regional runs while the trial is incomplete and resources permit.
- Let both run until one fully completes.
- Stop every nonterminal sibling only after `run_progress >= 1` and the expected
  checkpoint is reachable.
- Record siblings as race losses.

Discourage more than two replicas without a specific operator reason.

With one replica, relocation remains allowed after its timeout. It replaces the active region
and starts from zero.

## Handle Client Revision Rejections

Apply this only when a new dispatch fails with an error like
`marin-iris client is too old (build <date>; minimum <date>)`; exact wording may
vary. Do not check the revision on every loop.

On rejection:

1. Record `client_floor_failed`.
2. Stop new submissions and recovery-driven stops.
3. Follow `marin-experiment` to resolve a coherent current Marin and Iris dependency
   set, regenerate the lockfile, and validate the launch import.
4. Verify the runtime-reported date and prove a canary submission is accepted.
5. Resume on acceptance; otherwise stop and give the operator the dates, failure,
   and options.

Do not change `BUILD_DATE` alone or restart or reconfigure the shared Iris cluster.
If the dependency update is outside the approved task scope, stop and ask the
operator with the reported and required dates.
