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
- Treat the root README as a human-facing landing page. Keep it focused on the project's purpose, research outputs, runnable entry points, installation, and navigation. Do not put internal APIs, schemas, backend choices, or compatibility notes there.
- Update a workflow README when its user-visible behavior, configuration, outputs, or operating procedure changes. Put package API contracts and implementation details in scoped reference documentation or docstrings. Keep experimental findings in the tracking issue or research record.

## Repository Lifecycle

- Merge only reusable, maintained framework code and documentation to `main`. Experiments, one-off analyses, competitor baselines, and dead ends remain on permanent branches and are cited with commit-pinned links.
- Never push directly to `main`, merge or close a pull request, or close a research-question issue without explicit user approval.
