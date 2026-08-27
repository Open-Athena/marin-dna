---
name: run-training-sweep-cw
description: Maximize sweep throughput and minimize wall-clock time to completion across CoreWeave H100 and GB200 GPUs. Use when a training sweep defines multiple configurations as trials and an agent must validate GPU gang profiles, use live fleet capacity, dispatch and recover Iris jobs, monitor W&B progress, persist execution state, and report until every trial finishes.
---

# Run CoreWeave GPU Training Sweep

Finish the declared sweep as fast as possible without violating its operations document.

Use approved CoreWeave GPU capacity at batch priority. Iris handles preemptions;
do not replace a dispatch for a preemption alone—act from W&B progress and the
recovery policy.

## Contract

- Treat each experiment-defined configuration as an opaque logical trial.
- Let the training code own configuration and training semantics. Assume standard
  W&B training fields; inspect code for launch, data, checkpoint, and gang behavior.
- Keep one W&B and checkpoint identity per trial. Cluster, GPU type, and gang size
  are placement, so reslicing preserves that identity.
- Own only validation, placement, dispatch, monitoring, recovery, accounting, and reporting.
- Never generate configurations or decide experimental convergence.
- Compose with `marin-experiment` for project setup, coherent Marin and Iris
  versions, snapshots, and launch authorization. This skill does not authorize
  remote compute by itself.
- Follow `wandb-reporting` for MarinDNA run, group, and project naming.
- Do not move a trial between accelerator families unless the operator explicitly
  allows it. Persist and report the resulting hardware lineage as a reproducibility
  caveat.

## State

Keep each durable fact in one authoritative form:

- **Operations (text):** policy, choices, constraints, exceptions, and operating
  conclusions not reliably recovered from code or data.
- **Experiment and helper code:** executable behavior.
- **SQLite (data):** structured observations, identities, attempts, action results,
  clocks, and history.

Session chat is the interface, not state. Put heartbeat reports and decisions needing
operator input there. Anything needed by a later heartbeat belongs in text, code, or
data.

Never restate code or configuration in Operations; prefer a source reference. Create
no other experiment log, runbook, or recurring status file. Correct the authoritative
form instead of adding a competing note.

## Process

Read every reference in this table before setup. It assigns ownership; later
sections define the phase contracts.

| Phase | Purpose | Detailed references |
| --- | --- | --- |
| Initialize | Gather choices and establish durable state | [operations.md](references/operations.md), [targets.md](references/targets.md), [persistence.md](references/persistence.md) |
| Validate | Establish safe targets and launch behavior | [validation.md](references/validation.md) |
| Operate | Observe, place, recover, report, and schedule | [execution.md](references/execution.md), [throughput.md](references/throughput.md), [persistence.md](references/persistence.md), [utilization.md](references/utilization.md) |
| Finish | Verify completion, checkpoints, and integrity | [throughput.md](references/throughput.md), [persistence.md](references/persistence.md) |

Use a valid [utilization snapshot](references/utilization.md) before dispatching or
changing capacity. It informs placement; W&B remains the source of training progress.

## Initialize

### Interview Briefly

Ask only for missing information. Offer the recommended answer first.

1. **Training entry point and trial catalog.** Require both.
2. **Time limit.** Recommend two weeks. Explain shorter means faster recovery;
   longer is useful only when healthy trials need it.
3. **Compute and GPU scope.** Require an explicitly approved remote-compute scope
   and overall GPU limit before any smoke or production dispatch. Recommend the
   limit from the training code and prior runs. Confirm whether to use both H100
   and GB200, both H100 clusters (`cw-us-east-02a` and `cw-rno2a`), and GB200 on
   `cw-us-east-08a`; recommend all three production clusters by default. Recommend
   keeping cross-family reslicing disabled unless throughput outweighs the
   reproducibility cost.
4. **Sweep timing.** Recommend `heartbeat_every=30m`, `reslice_after=1h`, and
   `restart_after=3h`. Ask whether to use the defaults.

Create the document from [operations.md](references/operations.md). Default to
`scratch/<sweep>/expXXX_operations.md`; offer a tracked experiment-side file only
if requested. Build its candidate grid from [targets.md](references/targets.md), then
initialize SQLite:

```bash
uv run .agents/skills/run-training-sweep-cw/scripts/persistence.py \
  init scratch/<sweep>/expXXX_sweep.sqlite
```

## Validate

Follow [validation.md](references/validation.md) before the first dispatch. Validate
the experiment entry point, shared inputs and checkpoints, target compatibility,
W&B observation, fleet visibility, and Iris submission. Submit only `eligible`
targets and show the first assembled dispatch command to the operator.

## Operate

Run each heartbeat as a complete agent decision pass. Helpers may calculate or
render facts; they may not choose or perform actions. Never delegate decisions to a
monitor, dispatcher, scheduled loop, or recovery script.

At every heartbeat:

1. **Refresh context and inventory:** enter from authoritative state, not memory.
   Reread Operations and relevant code, then build the inventory snapshot.
2. **Observe, reconcile, decide, and act:** query W&B first, then exact Iris
   liveness and fleet utilization. Persist observations, rebuild the decision
   snapshot, and coordinate one fleet plan with [execution.md](references/execution.md).
   Persist each result and replan after a material action.
3. **Finish or continue:** if finished or out of time, stop, verify, and summarize.
   Otherwise report in chat and schedule exactly one time-based next pass for
   `heartbeat_every`. Never rely on event-based monitoring.

Build inventory and decision snapshots with:

```bash
uv run .agents/skills/run-training-sweep-cw/scripts/persistence.py \
  snapshot scratch/<sweep>/expXXX_sweep.sqlite --reslice-after-hours <hours>
```

Rebuild after material actions when needed. A snapshot is an ephemeral,
decision-complete projection, not another status record.

## Finish

A trial finishes when `run_progress >= 1` and the expected checkpoint is reachable.
W&B `finished` alone is insufficient.

When every trial finishes or the time limit expires, stop remaining dispatches,
verify completion and checkpoints, check SQLite integrity, and report the outcome as
defined by [throughput.md](references/throughput.md). Include the accelerator and
placement lineage for every trial that moved across hardware families. Do not
schedule another pass.
