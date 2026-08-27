# Operations

Create one living, tracked `expXXX_operations.md` beside the experiment project for each sweep.
The agent must read it completely at the start of every heartbeat.

Use terse English for rules, choices, constraints, exceptions, and operating
conclusions not reliably recovered from code or data. Never restate code or
configuration; reference its source. Never write heartbeat observations, job
status, progress, rankings, action queues, reports, or next-check times here.

## Required Structure

```markdown
# expXXX CoreWeave Sweep Operations

> This document governs a training sweep managed with the
> `run-training-sweep-cw` skill. Read it in full at the start of every heartbeat
> before inspecting code, SQLite, W&B, or Iris.

## Invariants

State the constraints that must hold throughout the sweep, including one writer
per trial, shared checkpoint identity, whole-node gangs, and batch priority.

## Sweep Definition

Point to the training entry point and configuration catalog. State only input,
initialization, checkpoint, or placement constraints not clear from code or SQLite.

## Operator Choices

Record the absolute sweep deadline, approved GPU limit, clusters, GPU families, exclusions, whether cross-family reslicing is allowed, Iris user, batch priority, and document location.
Record the durable SQLite backup owner and prefix, immutable key format, upload/download method, checksum verification, and newest-valid-backup recovery rule.
State clearly that the sweep deadline stops the whole sweep and that recovery timing is controlled separately by `reslice_after` and `restart_after`.

## Operating Policy

State `heartbeat_every`, `reslice_after`, `restart_after`, and
`pending_target_limit`; isolated/systemic failure handling; reslicing; and
completion behavior.

Maintain the authoritative target grid here:

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| `<exact Iris peer>` | `<variant>` | `<gang replicas>` | `<total>` | `unvalidated` | — |

Create the candidate rows from `targets.md`, then validate them. State is
`unvalidated` while validation is pending, `eligible` when dispatch is permitted,
or `ineligible` when a verified restriction prohibits dispatch. `Reason` is
required only for `ineligible` targets and names that restriction.

Keep every considered cluster and gang shape visible. Only `eligible` targets may
be dispatched. A trial-specific failure does not make a target globally ineligible.
A verified invalid-target result makes only its exact target ineligible; investigate
the same result on a previously working target before changing eligibility.

## Change Record

Only while the autonomous loop is running, record important, unexpected changes
made in response to unusual conditions. Every entry must correspond to an actual
change to this document, the SQLite data model, or execution code. Give UTC time,
cause, exact change, and operational effect. Do not duplicate SQLite event details
or session reports.

None.
```

## Keep A Selective Change Record

Use `Change Record` only when an unusual condition changes Operations, the SQLite
model, or execution code. Update the relevant section in place, then add one terse
entry. Ask before changing Operator Choices or experiment semantics.

Never record routine dispatches, retries, progress, utilization, rankings, or next
checks here. Recompute placement evidence from current utilization and SQLite.
