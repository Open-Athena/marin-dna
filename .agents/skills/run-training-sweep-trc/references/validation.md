# Validation

Establish which experiment-defined configurations and regional targets can be launched, observed, and checkpointed safely.
Derive the needed facts from the training entry point, its configuration, and existing framework behavior before the first sweep dispatch.

## Minimum From The Script

The script needs only to launch one selected, experiment-defined configuration from explicit parameters.
Environment variables are preferred.
Region and TPU placement may instead be inputs to the existing launcher or training framework.

If the agent cannot select one configuration unambiguously, propose the smallest script change and ask before making it.
Do not add fields merely to simplify orchestration.

## Inspect Before Launch

Read the script and the helpers it calls.
Verify:

- Configuration inputs and the exact one-trial command.
- How data, token caches, initialization inputs, and checkpoints resolve by region.
- How a same-region restart resumes and how a cross-region restart begins separately.
- Which TPU slices the script and training framework can actually run.

Do not transcribe code or configuration into Operations.
Prefer source references; record only resulting constraints, exceptions, and operating conclusions.

## Validate Regional Inputs First

Before treating a region as eligible:

1. Resolve its data and token-cache locations from code or configuration.
2. Confirm the needed objects exist and are usable using cheap existing metadata, manifests, counts, or framework validation.
3. Confirm initialization inputs, such as checkpoints used when not starting from scratch, are available in that region.
4. Update the target state in Operations; give a reason only when it is ineligible.

Never copy or move data between regions.
A missing cache makes the region ineligible; it does not make the region disappear.
Keep the resulting restriction visible in Operations.

Use configured bucket mappings.
In particular, `europe-west4` uses `marin-eu-west4`; never derive its bucket name.

## Validate Checkpoint Locality

Determine the resolved checkpoint location from the script, framework configuration, preview, or a cheap smoke run.

- A same-region reslice must resume from the same regional checkpoint storage.
- A cross-region relocation must use separate regional storage and start from the beginning unless the required state already exists there.
- Never allow two live jobs to write the same checkpoint location.

Do not impose a checkpoint naming scheme.
Stop before launch if the location or writer ownership is ambiguous.

## Validate Placement

Start from the complete candidate grid defined by [targets.md](targets.md).
Determine eligibility from:

- Known platform restrictions.
- The script or training framework's placement check.
- A smoke run when static checks are insufficient.

Do not guess memory feasibility.
Mark combinations `eligible`, `ineligible`, or `unvalidated` in the operations-document target grid; only submit eligible combinations.

## Assess Parallelism Fit

Compare the target grid's chip counts and per-chip HBM with how the script chooses data, tensor, context, or equivalent parallelism.
Warn, but do not block, when a broad grid appears unsupported.
For example, if global batch size is smaller than the chip count of a target and no tensor or context parallelism makes that target usable, ask the operator to verify it.
Apply judgment to narrow grids; this advisory alone does not change target eligibility.

## Validate W&B Observation

Assume training runs provide W&B `state` and `run_progress`.
Use `state` for current liveness and the `run_progress` high-water mark for progress, completion, stall timing, and placement throughput.

Use the sweep's normal W&B routing and group.
Follow `wandb-reporting` and require MarinDNA run names to map back to `dna-exp<N>`.
Require region as a structured W&B field or tag so regional runs can be distinguished.
Treat run IDs as opaque; never extract metadata from an ID or display name.

## Validate Submission

Before the first launch:

- Confirm the Iris client meets the required revision floor.
- Confirm the exact command and secrets that must be forwarded; keep secrets out of Operations and SQLite.
- Record the existing priority band.
  Never change it without explicit operator approval.
- Show the first submission command to the operator for review.
