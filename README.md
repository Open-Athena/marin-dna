<h1 align="center">MarinDNA</h1>

<p align="center">Open development of genomic language models — data, modeling, and evaluation.</p>

<p align="center"><sub>Inspired by <a href="https://github.com/marin-community/marin">Marin</a>.</sub></p>

## News

- **2026-05-26** — *Poster* — [Data curation strategies for genomic language models](docs/posters/cshl26/poster.pdf) at the [CSHL 90th Symposium "AI in Biology"](https://meetings.cshl.edu/meetings.aspx?meet=SYMP&year=26).

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

## Resources

- [🤗 Models and datasets on Hugging Face](https://huggingface.co/marin-dna)
- [🏆 Variant effect prediction leaderboard](https://openathena.ai/marin-dna/)
- [🧬 Interactive sequence explorer](https://molab.marimo.io/notebooks/nb_MrPpr5xYcN3HGt5tLY86bk/app)
- [💻 Model inference and BRCA1 variant effect prediction notebook](https://molab.marimo.io/github/Open-Athena/marin-dna/blob/20f48809013a34c4cf4bee94582e9ba94b56b5b2/examples/model_inference_and_vep.py)

## Example

This example tokenizes 64 enhancer sequences and trains a tiny
Qwen3 model from scratch for 10 steps on CPU.
See the standalone [`examples/train_tiny_dna/`](examples/train_tiny_dna/)
directory for the runnable script, locked environment, and run instructions.

```python
from fray.types import ResourceConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner
from marin.experiment.data import tokenized
from marin.experiment.train import train_lm

ARTIFACT_VERSION = "2026.07.29"

# 1. Tokenize a small sample of enhancer sequences.
enhancers_tokenized = tokenized(
    name="tokenized/zoonomia-ccre-non-promoter-tutorial",
    source="marin-dna/zoonomia-v1-v3_ccre_non_promoter-tutorial",
    tokenizer="marin-dna/tokenizer-char-bos",
    text_key="sequence",
    sample_count=64,
    version=ARTIFACT_VERSION,
)

# 2. Define a tiny Qwen3 decoder.
tiny_qwen3 = Qwen3Config(
    max_seq_len=256,  # 255 DNA bases plus the BOS token
    hidden_dim=32,
    intermediate_dim=128,
    num_heads=4,
    num_kv_heads=2,
    num_layers=2,
)

# 3. Train for a few steps on CPU.
tiny_enhancer_model = train_lm(
    name="checkpoints/tiny-dna-qwen3-cpu",
    version=ARTIFACT_VERSION,
    model=tiny_qwen3,
    optimizer=AdamConfig(learning_rate=6e-4, weight_decay=0.1),
    datasets={enhancers_tokenized: 1.0},
    batch_size=4,
    seq_len=tiny_qwen3.max_seq_len,
    num_train_steps=10,
    z_loss_weight=None,
    evals=None,
    resources=ResourceConfig.with_cpu(),
)

if __name__ == "__main__":
    StepRunner().run([lower(tiny_enhancer_model)])
```

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
