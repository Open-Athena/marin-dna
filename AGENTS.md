# Project Guidelines

MarinDNA develops genomic language models. Prioritize reproducibility and correctness, especially where genomic data can be silently corrupted.

## Genomic Invariants

- Use 0-based, half-open genomic coordinates internally. Convert 1-based or closed formats such as GTF, VCF, SAM, `pyfaidx.get_seq()`, and samtools-style region strings at the tool boundary, and state any deviation.

## Project Boundaries

- Each runnable workflow is its own dependency, source, test, and execution boundary. Put maintained Python in the owning project's `src/` package.
- Keep the root `src/marin_dna/` package limited to lightweight genomic primitives shared by independent projects. It must not depend on training frameworks, Snakemake, W&B, plotting libraries, or pipeline-specific clients.
- Prefer straightforward project-local code. Duplicate unstable logic across projects until a genuinely shared abstraction has emerged.

## Verification And Documentation

- Add tests for non-trivial maintained behavior in the owning project. For data pipelines, assert contracts such as schemas, row counts, value ranges, sequence lengths, coordinate bounds, and build or strand consistency.
- Run `uv run --locked pytest` from every changed Python project's root. Use `uv` for Python dependencies.
- Type-annotate parameters and return values in every project's `src/` package using current built-in generic and union syntax.
- Treat the root README as a human-facing landing page. Keep it focused on the project's purpose, research questions, resources, community, and citation. Put setup and internal workflow guidance in this file or scoped documentation.
- Update a workflow README when its user-visible behavior, configuration, outputs, or operating procedure changes.
  Put package API contracts and implementation details in scoped reference documentation or docstrings.
  Keep chronological experiment records in tracking issues and branches, and keep accepted interpretations in `docs/research/`.
- In Markdown prose, put each sentence on its own source line and do not hard-wrap at a fixed column.

## Skills

Skills are task playbooks in `.agents/skills/<name>/SKILL.md`, also reachable as `.claude/skills/`.
Before non-trivial work, check for a matching skill and follow it.

- Keep repository-wide rules only in this file, prose rules only in `writing-style`, and commit, pull-request, review, and monitoring steps only in Repository Lifecycle below.
  A skill points at those sources instead of restating them, and names the skills it composes with rather than copying their content.
- Treat the frontmatter `description` as the routing contract: one line stating what the skill does and when to select it.
  Direct-task skills say when to use them; skills a task must not pull in on its own say `only when explicitly requested`; scheduled scrubs say `only from its scheduler or an explicit request`; skills that other skills invoke say `delegated by another selected workflow`.
- Declare `schedule_cron` (five-field cron) and `schedule_tz` (IANA zone) together on a skill whose cadence this repository defines.
  The fields document intent; the scheduler itself runs outside the repository.
- Keep a skill's entry point compact.
  Put detail it needs only sometimes in `references/` files it loads on demand, and put runnable helpers in `scripts/` gated by the root pytest suite.
- `uv run --locked python infra/check_skill_metadata.py` validates every skill: YAML frontmatter, `name` equal to the directory and unique, a single-line description, paired schedule fields, a string `allowed-tools`, every repository path or well-known root file (`AGENTS.md`, `README.md`, `pyproject.toml`, …) in code spans, fenced blocks, or relative links, and drift traps for retired paths and skill names.
  Pre-commit runs it on every commit and the Quality workflow on every pull request, so a moved path fails lint until every skill linking it is updated.
  Bare skill-name mentions are not path-checked: when renaming or retiring a skill, add its old name to `DRIFT_TRAPS` in the checker and grep for remaining mentions.
- Do not edit vendored skills directly; follow `maintain-vendored-skills`.
  When a local path an adapted vendored skill links to moves, update that link and record the deviation in the manifest in the same change.
- `scrub-reflection-self-improvement` and `scrub-docs-code-parity` hunt for stale guidance on their declared schedules, and `update-docs` covers skill docs whenever work changes behavior.

## Development Setup

Use `uv` for Python dependencies. Set up and test the lightweight root project with:

```bash
uv sync --locked --group dev
uv run --locked pytest
```

Install and run repository-level quality checks with:

```bash
uv run --locked pre-commit install
uv run --locked pre-commit run --all-files --show-diff-on-failure
```

Each runnable Snakemake workflow is an independent project. From every changed workflow directory, run:

```bash
uv sync --locked --group dev
uv run --locked pytest
uv run --locked snakemake -n
```

External bioinformatics programs remain in each rule's Conda environment. A root change runs the core project and dependent pipelines in CI; scheduled CI validates every project and lockfile.

## Repository Lifecycle

- Merge only reusable, maintained framework code and documentation to `main`. Experiments, one-off analyses, competitor baselines, and dead ends remain on permanent branches and are cited with commit-pinned links.
- If work on an experiment or another branch that will not merge to `main` uncovers a bug, missing capability, or generally reusable improvement, use `file-issue` to create a separate GitHub issue for the mainline change.
  Do not rely on the incidental fix being recovered from the experiment branch later.
- Use a stable lowercase branch name: `<agent>/issue-<number>-<summary>` when an issue exists, otherwise `<agent>/<summary>`.
- Add `agent-generated` to every issue or pull request created by an agent.
- Close an issue only after its completion criteria are met and its body and final comment record the outcome.
  For research issues, follow the disposition and interpretation-merge gate in `run-research` and `maintain-knowledge-base`.
- At the start of a task, identify every foreseeable approval, permission, credential, budget, paid-resource request, and scope decision needed to finish the work.
  Ask for any missing authority or decision together in the first message.
- Once those gates are cleared, complete the authorized work and its validation without requesting intermediate human feedback.
  Return earlier only when an unforeseen blocker, new authority requirement, or material scope decision requires the user.
- For repository changes intended for review, commit and push the branch, open or update a draft pull request, run an independent review over the published diff, and address its findings without asking for separate permission for these delivery steps.
- Mark the pull request ready and ask for human feedback only after implementation, validation, publication, and independent review are complete.
- Never push directly to `main` or merge or close a pull request without explicit user approval.
