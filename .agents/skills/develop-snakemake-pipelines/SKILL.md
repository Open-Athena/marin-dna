---
name: develop-snakemake-pipelines
description: Develop, modify, test, dry-run, and execute MarinDNA Snakemake projects. Use whenever working under snakemake/, changing a Snakefile, rule, profile, pipeline-local Python package, Conda environment, or lockfile, invoking Snakemake locally or through remote compute, creating a new pipeline, or investigating an unexpected rerun.
---

# Develop Snakemake Pipelines

Treat each runnable pipeline as an independent Python project and execution boundary.

## Work In The Owning Project

1. Find the nearest pipeline root containing `pyproject.toml`, `uv.lock`, `README.md`, `src/`, `tests/`, and `workflow/Snakefile`.
2. Read its README, manifest, and `workflow/profiles/default/config.yaml` before changing or running it.
3. Run commands from that project root through its committed environment:

```bash
uv sync --locked --group dev
uv run --locked pytest
uv run --locked snakemake -n <targets>
```

Use the targets intended for the task. Put pipeline-wide defaults such as cores, Conda use, and storage providers in the checked-in profile. Keep external bioinformatics tools in rule-specific Conda environments.

## Keep Python Testable

- Put maintained pipeline behavior in the project's `src/` package.
- Keep inline Snakemake `run:` blocks as thin glue around package functions.
- Expose maintained command-line tools through package entry points.
- Add tests for Python behavior and assert output contracts where silent corruption is possible.

## Inspect Before Executing

1. Run the owning project's tests.
2. Dry-run before every real Snakemake invocation.
3. Inspect the planned jobs, inputs, outputs, resources, and rerun reasons.
4. Stop and ask before executing if the plan includes an unintended upstream, unrelated, expensive, or destructive job. Timestamp changes and default rerun triggers do not establish intent.
5. Follow inherited compute-safety rules before local heavy work. Obtain explicit user approval before launching paid remote compute, including SkyPilot resources.
6. Monitor a new script, configuration, or compute combination during its first minutes. Check progress rate, expected devices, mounts, authentication, and early failures.

## Create A Pipeline

Compare current maintained pipelines and select the closest structural example. Copy only the applicable project conventions: manifest, lockfile, source package, tests, workflow, profile, and README. Verify dependencies, rules, storage configuration, and remote-execution assumptions independently.

Do not keep a runnable template project on `main`. If pipeline creation becomes frequent and error-prone, add and test a deterministic generator in this skill.

## Document Changes

Update the owning README when behavior, configuration, dependencies, execution, outputs, or recovery procedures change. Keep run results and research findings in the tracking issue or research record.
