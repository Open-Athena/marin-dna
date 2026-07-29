<h1 align="center">MarinDNA</h1>

<p align="center">Open development of genomic language models — data, modeling, and evaluation.</p>

<p align="center"><sub>Inspired by <a href="https://github.com/marin-community/marin">Marin</a>.</sub></p>

## Research

Tracked as GitHub issues under a two-axis label taxonomy — **Type** (the kind of work) × **Area** (the part of the project). See [AGENTS.md](AGENTS.md#issue-labels) for what each label means.

**By type** — [research questions](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Aresearch-question) (durable north-stars) ·
[experiments](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Aexperiment) (preregistered runs) ·
[exploratory analyses](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Aeda)

**By area** — [evals](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Aevals) ·
[data](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Adata) ·
[modeling](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Amodeling) ·
[baselines](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Abaselines) ·
[hyperparameter-optimization](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Ahyperparameter-optimization) ·
[interpretation](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Ainterpretation)

## Leaderboard

Variant effect prediction leaderboards (under construction): [openathena.ai/marin-dna](https://openathena.ai/marin-dna/).

## Examples

- **[Inference and VEP notebook](examples/model_inference_and_vep.py):**
  code-visible Marimo tutorial for loading a pinned MarinDNA model, inspecting
  sequence outputs, and running zero-shot BRCA1 variant-effect prediction;
  [open the GPU-verified revision on Molab](https://molab.marimo.io/github/Open-Athena/marin-dna/blob/b4e61079612b0a970b63940f7b4782bbbd2f849c/examples/model_inference_and_vep.py).
- **[Sequence explorer](https://molab.marimo.io/notebooks/nb_MrPpr5xYcN3HGt5tLY86bk/app):**
  task-focused interactive application for sequence-logo and
  nucleotide-dependency visualizations.

See [examples/README.md](examples/README.md) for the distinction, hardware
requirements, source links, and launch instructions.

## News

- **2026-05-26** — *Poster* — [Data curation strategies for genomic language models](docs/posters/cshl26/poster.pdf) at the [CSHL 90th Symposium "AI in Biology"](https://meetings.cshl.edu/meetings.aspx?meet=SYMP&year=26).

## Installation

```bash
uv sync
```

<details>
<summary>Optional installs (all opt-in)</summary>

| Selector | Purpose |
|---|---|
| `--group dev` | Pre-commit, ruff, pytest, snakefmt. |
| `--group aws-cli` | `awscli` for snakemake rules that shell out to `aws s3 cp` (e.g. `evals/ldscore_download`). |
| `--group genome-s3` | Modern `s3fs` for `Genome(s3://…)` FASTA reads (e.g. evals_v2). Mutually exclusive with `aws-cli` (which pins older fsspec/s3fs). |

**Marin-launched experiments** (`marin`/`levanter`/`iris`/jax) are *not* installed
from here — each experiment is a self-contained directory with its own
`pyproject.toml` (marin in base deps). See the
[`marin-experiment` skill](.agents/skills/marin-experiment/SKILL.md).

</details>

## Development

```bash
# Install dev dependencies and pre-commit hooks
uv sync --group dev
uv run pre-commit install

# Run quality checks
uv run pre-commit run

# Run tests
uv run pytest
```

## Project Structure

See [AGENTS.md](AGENTS.md#code-structure).

## Community

Join the [Marin Discord](https://discord.gg/J9CTk7pqcM); MarinDNA discussion happens in the `#dna` channel.

## Citation

If you find datasets, models, or experiments from this repo useful, please cite:

> MarinDNA: open development of genomic language models. Open Athena, 2026.
> https://github.com/Open-Athena/marin-dna

BibTeX:

```bibtex
@misc{marin-dna,
  title  = {MarinDNA: open development of genomic language models},
  author = {{Open Athena}},
  year   = {2026},
  url    = {https://github.com/Open-Athena/marin-dna},
}
```
