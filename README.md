<h1 align="center">MarinDNA</h1>

<p align="center">Open development of genomic language models — data, modeling, and evaluation.</p>

<p align="center"><sub>Inspired by <a href="https://github.com/marin-community/marin">Marin</a>.</sub></p>

## News

- **2026-08-03** — *Blog post* — [A 1B standard Transformer rivals Evo 2 40B on variant effect prediction](https://openathena.ai/blog/marin-dna/).
- **2026-05-26** — *Poster* — [Data curation strategies for genomic language models](docs/posters/cshl26/poster.pdf) at the [CSHL 90th Symposium "AI in Biology"](https://meetings.cshl.edu/meetings.aspx?meet=SYMP&year=26).

## Research

Tracked as GitHub issues under a two-axis label taxonomy: **Kind** selects the issue structure and lifecycle; **Topic** describes the affected system or research area. See [AGENTS.md](AGENTS.md#issue-taxonomy) for the definitions.

**By kind** — [research questions](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Aresearch-question) (durable, human-declared syntheses) · [experiments](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Aexperiment) (bounded research with a hypothesis or goal) · [tasks](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Atask) · [bugs](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Abug)

**By topic** — [evals](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Aevals) · [data](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Adata) · [modeling](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Amodeling) · [baselines](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Abaselines) · [hyperparameter optimization](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Ahyperparameter-optimization) · [interpretation](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Ainterpretation) · [infrastructure](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Ainfrastructure) · [communication](https://github.com/Open-Athena/marin-dna/issues?q=is%3Aissue+label%3Acommunication)

## Resources

- [🤗 Models and datasets on Hugging Face](https://huggingface.co/marin-dna)
- [🏆 Variant effect prediction leaderboard](https://openathena.ai/marin-dna/)
- [🧬 Interactive sequence explorer](https://molab.marimo.io/notebooks/nb_MrPpr5xYcN3HGt5tLY86bk/app)
- [💻 Model inference and BRCA1 variant effect prediction notebook](https://molab.marimo.io/github/Open-Athena/marin-dna/blob/5d1925fe0d6569c0ee0c29db06b8f287c2347065/examples/model_inference_and_vep.py)

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

The repository root is the lightweight `marin_dna` core project:

```bash
uv sync --locked --group dev
uv run --locked pytest
```

Each runnable Snakemake pipeline is an independent Python project with its own manifest, lockfile, source package, tests, and `.venv`:

| Project | Package |
|---|---|
| `snakemake/analysis/evals_v2/` | `marin_dna_evals` |
| `snakemake/zoonomia_projection_dataset/` | `marin_dna_zoonomia_projection` |
| `snakemake/training_dataset/genome_selection/` | `marin_dna_genome_selection` |
| `snakemake/training_dataset/dataset_creation/` | `marin_dna_training_dataset` |

Run a pipeline from its own directory:

```bash
cd snakemake/analysis/evals_v2
uv sync --locked --group dev
uv run --locked pytest
uv run --locked snakemake -n
```

External bioinformatics programs remain in each rule's Conda environment. Marin-launched experiments and the tutorial under `examples/train_tiny_dna/` remain self-contained projects with their own lockfiles.

New pipelines should start from [`scaffolds/snakemake-pipeline/`](scaffolds/snakemake-pipeline/), which includes the same manifest, lockfile, package, test, workflow, and profile layout.

## Development

Install repository-level quality tooling from the core project:

```bash
uv sync --locked --group dev
uv run --locked pre-commit install
uv run --locked pre-commit run --all-files --show-diff-on-failure
uv run --locked pytest
```

A root change runs core plus every dependent pipeline in CI. A pipeline-only change runs that project's locked tests and dry-run. The scheduled CI check validates every project and lockfile.

## Project Structure

The repository is the coordination boundary; each independently runnable workflow is its own dependency and execution boundary. Reusable genomic primitives live in `src/marin_dna/`; pipeline-specific Python lives beside its workflow under that pipeline's `src/`; and experiments remain isolated by branch/worktree. See [AGENTS.md](AGENTS.md#code-structure).

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
