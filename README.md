<h1 align="center">MarinDNA</h1>

<p align="center">Open development of genomic language models — data, modeling, and evaluation.</p>

<p align="center"><sub>Inspired by <a href="https://github.com/marin-community/marin">Marin</a>.</sub></p>

## Experiments

Tracked as GitHub issues. See the
[experiment-labeled issues](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Aexperiment).

## Leaderboard

Variant effect prediction leaderboards (under construction): [openathena.ai/marin-dna](https://openathena.ai/marin-dna/).

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
