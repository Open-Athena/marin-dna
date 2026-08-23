# Eval experiments

## Setup

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

Run the baseline model evaluation:

```bash
source .venv/bin/activate
snakemake --cores all --dry-run
```
