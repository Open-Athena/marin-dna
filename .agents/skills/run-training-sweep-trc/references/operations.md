# Operations

Create one living, tracked `expXXX_operations.md` beside the experiment project for each sweep.
The agent must read it completely at the start of every heartbeat.

Use terse English for rules, choices, constraints, exceptions, and operating conclusions not reliably recovered from code or data.
Never restate code or configuration; reference its source.
Never write heartbeat observations, job status, progress, rankings, action queues, reports, or next-check times here.

## Required Structure

```markdown
# expXXX Sweep Operations

> This document governs a training sweep managed with the `run-training-sweep-trc` skill.
> Read it in full at the start of every heartbeat before inspecting code, SQLite, W&B, or Iris.

## Invariants

State the constraints that must hold throughout the sweep.

## Sweep Definition

Point to the training entry point and configuration catalog.
State only regional data, initialization, and checkpoint constraints not clear from code or SQLite.

## Operator Choices

Record the absolute sweep deadline, regional replica count, approved chip limit, approved regions and TPU families, exclusions, whether cross-family reslicing is allowed, priority band, and document location.
Record the durable SQLite backup owner and prefix, immutable key format, upload/download method, checksum verification, and newest-valid-backup recovery rule.
State clearly that the sweep deadline stops the whole sweep and that recovery timing is controlled separately by `reslice_after`, `restart_after`, and `relocate_after`.

## Operating Policy

State `heartbeat_every`, `reslice_after`, `restart_after`, `relocate_after`, and `pending_target_limit`; isolated/systemic failure handling; regional racing; and completion behavior.

Maintain the authoritative target grid here:

| Region | Bucket | Slice | Chips | State | Reason |
| --- | --- | --- | ---: | --- | --- |
| `<canonical region>` | `<configured bucket>` | `<exact Iris slice>` | `<count>` | `unvalidated` | — |

Create the candidate rows from `targets.md`, then validate them.
State is `unvalidated` while validation is pending, `eligible` when dispatch is permitted, or `ineligible` when a verified restriction prohibits dispatch.
`Reason` is required only for `ineligible` targets and names that restriction.

Keep every considered region and slice visible.
Only `eligible` targets may be dispatched.
A trial-specific failure does not make a target globally ineligible.
A verified invalid-target result, including Iris `unschedulable`, makes only its exact target ineligible.
Treat `unschedulable` on a previously working target as an anomaly to investigate rather than routine grid pruning.

## Change Record

Only while the autonomous loop is running, record important, unexpected changes made in response to unusual conditions.
Every entry must correspond to an actual change to this document, the SQLite data model, or execution code.
Give UTC time, cause, exact change, and operational effect.
Do not duplicate SQLite event details or session reports.

None.
```

## Keep A Selective Change Record

Use `Change Record` only during the autonomous loop, never during setup or validation, or ordinary development.
Require an unexpected or unusual condition to cause an actual change to Operations, the SQLite model, or execution code.
Examples include a validated token cache disappearing, changing a recovery threshold, or updating the coherent Marin/Iris dependency set after a client-floor rejection.

Update the relevant Operations section in place so it states the current rule.
Then add one terse entry with the cause, change, and effect.
Ask before changing Operator Choices or experiment semantics.

Never record routine dispatches, retries, progress, utilization, or placement rankings here.
In particular, never turn a region/slice throughput advantage into Operations policy or target eligibility; recompute rankings from SQLite every loop.
Consolidate related changes rather than preserving a blow-by-blow sequence.
