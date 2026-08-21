# Project Guidelines

MarinDNA develops genomic language models. Prioritize reproducibility and correctness, especially where genomic data can be silently corrupted.

## Genomic Invariants

- Use 0-based, half-open genomic coordinates internally. Convert 1-based or closed formats such as GTF, VCF, SAM, `pyfaidx.get_seq()`, and samtools-style region strings at the tool boundary, and state any deviation.
- Treat the canonical human reference as the Ensembl release 115 GRCh38 soft-masked primary assembly with Ensembl sequence names (`1` through `MT`). Other `hg38` variants are not interchangeable. Use `access-reference-genomes` when choosing or retrieving a mirror.
- For labeled variant-effect prediction data, use odd-numbered autosomes and chromosome X for development, training, validation, model selection, probing, and tuning. Reserve even-numbered autosomes and chromosome Y for final test evaluation. Accessing held-out labels, predictions, effect measurements, or aggregate metrics requires explicit user permission. This restriction does not apply to unlabeled reference sequence or functional-genomics data unless that dataset defines a stricter split.

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

## Inference Workloads

- Treat one-off and experimental inference loops as production inference workloads.
  Prefer an established automatic evaluation loop, such as Hugging Face `Trainer.predict`, when it can express the required outputs.
  If a custom loop is necessary, record the capability the established loop could not provide.
- Batch accelerator inference and use bfloat16 (`bf16`), model compilation, multiple data-loader workers, pinned memory, and prefetching when the model, hardware, and framework support them.
  Record why any applicable optimization is disabled.
- Before a long run, define output-specific comparison fields and tolerances and compare a small sample against an uncompiled reference path.
- Measure steady-state throughput after warmup, state its unit, and record whether data loading and preprocessing are included.

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
- For completed feature-branch work, commit and push the branch, open or update a draft pull request, run an independent review over the published diff, address its findings, and mark the pull request ready when human feedback is wanted.
- Never push directly to `main` or merge or close a pull request without explicit user approval.
