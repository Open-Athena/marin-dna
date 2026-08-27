# Validation

Establish which experiment-defined configurations and CoreWeave targets can be
launched, observed, and checkpointed safely. Derive the needed facts from the
training entry point, its helpers, and existing framework behavior before the first
sweep dispatch.

## Minimum From The Script

The script needs only to launch one selected, experiment-defined configuration on
one selected cluster and gang size from explicit parameters. Environment variables
are preferred.

If the agent cannot select one configuration and placement unambiguously, propose
the smallest script change and ask before making it. Do not add a manifest or fields
merely to simplify orchestration.

## Inspect Before Launch

Read the script and the helpers it calls. Verify:

- Configuration inputs and the exact one-trial command.
- Data, token-cache, initialization, W&B, and checkpoint resolution.
- Production run and checkpoint identity do not depend on cluster, GPU, or nodes.
- Supported cluster profiles, gang sizes, batch fit, restart, and reslice behavior.

Do not transcribe code or configuration into Operations. Prefer source references;
record only resulting constraints, exceptions, and operating conclusions.

## Assess Hardware Profiles

For every candidate target, inspect how the code selects GPU variant, GPUs per
node, node count, CPU/RAM/disk, and data/tensor/context/pipeline parallelism. Confirm
the effective global batch and optimizer semantics remain intended across profiles.
Do not guess memory feasibility; use existing evidence or a smoke run.

The completed exp199 sweep provides compact examples:

- [cluster profiles and measured batch fitting](https://github.com/Open-Athena/MarinFold/blob/a3782d1c04b842731aaf692397643fa83354e1e3/experiments/exp199_optimize_contacts_v1_afdb_esm/gpu/exp199_sweep_cw.py#L491-L605)
- [smoke and production identity separation](https://github.com/Open-Athena/MarinFold/blob/a3782d1c04b842731aaf692397643fa83354e1e3/experiments/exp199_optimize_contacts_v1_afdb_esm/gpu/exp199_sweep_cw.py#L672-L738)
- [placement applied outside artifact identity](https://github.com/Open-Athena/MarinFold/blob/a3782d1c04b842731aaf692397643fa83354e1e3/experiments/exp199_optimize_contacts_v1_afdb_esm/gpu/exp199_sweep_cw.py#L741-L827)
- [whole-node gang request and temporary smoke output](https://github.com/Open-Athena/MarinFold/blob/a3782d1c04b842731aaf692397643fa83354e1e3/experiments/exp199_optimize_contacts_v1_afdb_esm/gpu/exp199_sweep_cw.py#L830-L923)

Use these as patterns, not an interface the new script must copy.

## Validate Inputs And Checkpoints

Confirm every selected cluster can use the resolved CoreWeave S3 inputs,
initialization state, and checkpoint root. Ensure the child gang receives the
required storage environment explicitly. A restart or reslice must resume the same
trial identity, and two live jobs must never write it concurrently.

## Validate Placement

Start from the complete candidate grid in [targets.md](targets.md). Determine exact
target eligibility from known restrictions, code/framework checks, and smoke runs
where static evidence is insufficient. Mark each target `eligible`, `ineligible`,
or `unvalidated` in Operations; submit only eligible targets.

## Validate W&B Observation

Assume training runs provide W&B `state` and `run_progress`. Use `state` for current
liveness and the `run_progress` high-water mark for progress, completion, stall
timing, and placement throughput.

Use the sweep's normal W&B entity, project, and group. Follow `wandb-reporting` and
require MarinDNA run names to map back to `dna-exp<N>`. Require cluster, GPU, and
nodes as structured config or tags. Treat run IDs as opaque; never extract metadata
from an ID or display name.

## Validate Submission

Before the first launch:

- Confirm the Iris client meets the required revision floor.
- Confirm the exact command and forwarded secrets; keep secrets out of Operations
  and SQLite.
- Use the `marin` controller, exact `--target-cluster`, `--priority batch`, and an
  explicit `--user`. The target cluster must match the script's placement input.
- Confirm the root driver remains alive for its child gang and the gang receives
  the intended batch priority.
- Obtain a valid fleet-utilization snapshot and show the first command to the operator.
