<div align="center">

# GLM Experiments

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

</div>

## Description

GLM experiments project

## Installation

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies and project
uv sync

# activate the virtual environment
source .venv/bin/activate
```

## Development Setup

To set up the development environment with linting, formatting, and testing tools:

```bash
# install dependencies including dev group
uv sync --group dev

# install pre-commit hooks
uv run pre-commit install
```

## Testing

Run tests using pytest:

```bash
# run all tests
uv run pytest

# run only fast tests (exclude slow tests)
uv run pytest -k "not slow"

# run tests from a specific file
uv run pytest tests/test_train.py

# run with verbose output
uv run pytest -v
```

You can also use the Makefile shortcuts:

```bash
# run fast tests
make test

# run all tests
make test-full
```

## Download dataset locally

```bash
uv run hf download songlab/gpn-animal-promoter-dataset --repo-type dataset --local-dir data/gpn-animal-promoter-dataset
uv run hf download gonzalobenegas/Angiosperm_16_genomes_sharded --repo-type dataset --local-dir data/gonzalobenegas/Angiosperm_16_genomes_sharded
uv run hf download gonzalobenegas/genomes-v2-genome_set-animals-intervals-v1_512_256 --repo-type dataset --local-dir data/gonzalobenegas/genomes-v2-genome_set-animals-intervals-v1_512_256
```

## How to run

Train model with default configuration

```bash
# train on CPU
uv run python glm_experiments/train.py trainer=cpu

# train on GPU
uv run python glm_experiments/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
uv run python glm_experiments/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
uv run python glm_experiments/train.py trainer.max_epochs=20 data.batch_size=64
```

### Loading a Checkpoint

```python
# replace with CLMLitModule if necessary
from glm_experiments.models.lm_lit_module import MLMLitModule

# Load plain pytorch model from checkpoint
model = MLMLitModule.load_from_checkpoint("logs/train/runs/<timestamp>/checkpoints/{step}.ckpt").net
```

## Upload checkpoints to HF

```bash
uv run hf auth login
uv run hf upload-large-folder --repo-type dataset gonzalobenegas/dev-glm-experiments-logs logs
```
