---
name: marin-experiment
description: Set up, launch, monitor, and debug a Marin-launched DNA training or evaluation experiment on the shared iris cluster. Use when creating a new numbered experiment, packaging it as an independent locked project, choosing a current Marin dependency set, launching remote training, or diagnosing coordinator, worker, accelerator, or resume failures. Use run-training-sweep-cw or run-training-sweep-trc for capacity-aware operation after a multi-trial catalog exists. Do not use for Snakemake pipelines or one-off analyses.
---

# Marin Experiment

Keep each experiment reproducible on its permanent branch. Use a self-contained directory with its own `pyproject.toml`, `uv.lock`, and launch module, and cite the resulting commit from the experiment issue.

For a multi-trial sweep, use this skill to build and validate the experiment project.
Once the trial catalog and one-trial launch command exist, route ongoing placement, recovery, and completion to `run-training-sweep-cw` or `run-training-sweep-trc`.

## Build A Self-Contained Project

- Put Marin packages required by dispatched workers in base `dependencies`. Iris and Zephyr workers may install the shipped project without optional extras; placing Marin only in an extra can produce `ModuleNotFoundError: No module named 'zephyr'`.
- Depend on MarinDNA through a commit-pinned Git URL so remote workers resolve the same code. Verify that every imported symbol exists at that revision.
- Define every dependency group referenced by the launch configuration, including an empty development group if the remote sync expects one.
- Keep accelerator-only dependencies in an explicit group or extra supported by the current Marin release. Copy package indexes, platform markers, and source routing from current upstream consumers only when that release still requires them.
- Commit the complete manifest and lockfile before launch.

## Resolve Current Versions

Marin and iris compatibility changes too quickly for this skill to carry version pins.

1. Read the experiment's tracking issue and the current compatibility discussion. Use [issue #356](https://github.com/Open-Athena/marin-dna/issues/356) while it remains the active MarinDNA launch tracker.
2. Inspect current consumer projects such as [`marin-community/marin-experiments`](https://github.com/marin-community/marin-experiments) and [`Open-Athena/MarinFold`](https://github.com/Open-Athena/MarinFold).
3. Select one coherent Marin release set that satisfies the controller freshness requirement and the launch API used by the experiment. Migrate the launch module when its API is obsolete instead of indefinitely pinning an old release.
4. Record the source and date used to choose the versions in the experiment issue or logbook.
5. Lock and validate locally:

```bash
uv lock
uv sync --locked
uv run python -c "import launch"
```

Inspect the resolved dependency tree when accelerator extras, custom indexes, Transformers, Torch, or evaluation packages constrain the solution. Do not copy an exact version from an old skill, issue comment, or experiment without revalidating it.

## Prepare The Launch

- Read the current `iris --help` and the launch API used by the selected Marin release before constructing the command.
- Obtain explicit user approval before launching paid remote resources.
- Pin or snapshot the experiment branch before submitting the job.
- Propagate required dependency groups and environment variables to every remote step according to the current API. Parent coordinator settings may not propagate to workers.
- Follow `wandb-reporting` for run and group naming. MarinDNA experiment runs must map back to `dna-exp<N>`.
- Match compute placement to the data location and current accelerator availability. Verify current supported regions and device names instead of copying an old pool list.

## Monitor A New Combination

Watch the first minutes of a new script, dependency set, or compute configuration.

1. Confirm the coordinator imports the launch module and constructs the graph.
2. Confirm dispatched workers import their runtime dependencies and begin the intended step.
3. Confirm the expected accelerator count and device type. Treat CPU fallback as a failure when an accelerator was requested.
4. Confirm W&B reports an advancing step and sane throughput, loss, and memory use.
5. Check mounts, credentials, and output paths before leaving the job unattended.

Use event-driven status tools or coarse polling that respects shared-node safety. Do not rely on submission success as evidence that the coordinator or workers started.

## Emit The Experiment Record

On success, write one experiment-level provenance record: `<prefix>/experiments/dna-exp<N>/.experiment.json`.
It is the experiment-grain counterpart of Marin's per-step [`ArtifactRecord`](https://github.com/marin-community/marin/blob/0554ce869c6dee2a3a7c00bc9caa9e2994ac435d/lib/marin/src/marin/execution/artifact.py#L149) (`.artifact.json`): it ties the experiment's identity (name and tracking issue), pinned inputs, arms and W&B run ids, and checkpoint output paths together, so the experiment is catalogable and reproducible from storage alone.
Write it beside, not inside, the run directories so it never collides with Marin's own `.artifact.json` at step output paths.

The writer is `marin_dna.experiment_record` (`ExperimentRecord`, `ExperimentDep`, `ExperimentRun`, `write_experiment_record`).
It validates the conventions at construction and again on read: `name` must be `dna-exp<N>` with `<N>` equal to the tracking-issue number, every dep revision must be a full 40-hex sha (branch names are not pins), and every `run_id` must carry `dna-exp<N>` so W&B runs filter by experiment.

In the launch module, after the runner returns (it raises on failure, so this only runs on success):

```python
import os

from rigging.filesystem import marin_prefix, prefix_join
from marin_dna.experiment_record import (
    ExperimentDep, ExperimentRecord, ExperimentRun,
    utc_now_iso, write_experiment_record,
)

def main() -> None:
    steps = {arm: build_arm(arm) for arm in selected_arms()}
    run_steps(steps.values())  # whatever runner the selected Marin release uses
    record = ExperimentRecord(
        name=NAME,  # "dna-exp<N>"
        issue=f"https://github.com/Open-Athena/marin-dna/issues/{N}",
        created_at=utc_now_iso(),
        config=CONFIG,  # the launch's pinned constants: model geometry, optimizer, batch, steps, seed
        deps=[
            # arm-specific pins carry arm=; deps every run consumed leave it None
            *(
                ExperimentDep("hf-dataset", DATASET_REPOS[arm], DATASET_REVISIONS[arm], arm=arm)
                for arm in steps
            ),
            ExperimentDep("git", "https://github.com/Open-Athena/marin-dna", MARIN_DNA_REVISION),
        ],
        runs=[
            ExperimentRun(run_id=RUN_IDS[arm], output_path=step.path(), arm=arm)
            for arm, step in steps.items()
        ],
        provenance={"marin_pin": MARIN_PIN, "launched_by": os.environ.get("USER", "")},
    )
    # prefix_join takes exactly (prefix, relative) — pass the subpath as ONE argument.
    write_experiment_record(record, prefix_join(marin_prefix(), f"experiments/{NAME}"))
```

- `step.path()` resolves the output path a step wrote under the current lazy-artifact API.
  Verify the resolved paths against where the runs actually wrote the first time you use this with a new Marin release.
- Remote-store writes need the matching fsspec backend (for example `gcsfs` for `gs://`) importable where the launch module runs; the Marin experiment environment provides it.

## Diagnose By Symptom

| Symptom | Check and action |
|---|---|
| A worker cannot import `zephyr` or another Marin package | Confirm the package is in base dependencies and present in the worker's actual sync command. |
| The client is below the controller freshness floor | Resolve a current coherent Marin release set and regenerate the lockfile. Do not upgrade one Marin package in isolation. |
| The coordinator cannot import a launch API symbol | Compare the launch module with the selected release. Migrate to the current API or choose a compatible set documented by a current consumer. |
| Torch, accelerator, or evaluation extras do not resolve | Inspect direct dependencies, explicit indexes, platform markers, and override constraints in the current upstream project. |
| A timeout or credential exists on the coordinator but fails on a worker | Re-declare the variable or secret on each remote step using the current launch API. |
| The job reports no accelerator or remains pending | Verify the requested device and placement against current iris capacity before changing regions or device families. |
| Compilation runs out of accelerator memory | Reduce per-device microbatch or adjust the supported parallelism setting while preserving the intended effective batch size. |
| A preemptible step fails | Inspect the latest checkpoint and relaunch only after confirming the current configuration can resume it. |
| The job exits during teardown | Verify the final checkpoint, evaluation output, and required sidecars before deciding that the experiment must rerun. |

## Record The Outcome

- Link the job, W&B runs, output artifacts, and commit-pinned experiment code from the tracking issue.
- Record deviations from the intended configuration and whether the final artifacts passed validation.
- Update this skill only for a durable workflow lesson. Keep release pins, capacity observations, and individual run results in their time-stamped source artifacts.

Use public repositories, issues, W&B, and the public Marin Discord archive as evidence. Do not cite internal channels.
