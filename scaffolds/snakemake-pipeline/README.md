# Snakemake pipeline project scaffold

Copy this directory to the pipeline's permanent location, then rename the distribution and import package in `pyproject.toml`, `src/`, and `tests/`. Keep pipeline-specific Python and tests inside the copied project.

Refresh the template lockfile after renaming:

```bash
uv lock
uv sync --locked --group dev
uv run --locked pytest
uv run --locked snakemake -n
uv run --locked snakemake
```

Commit the generated `uv.lock`. Add external command-line tools through rule-specific Conda environments under `workflow/envs/`.
