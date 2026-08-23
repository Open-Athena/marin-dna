# CLAUDE.md - Complete Development Guide

This document provides comprehensive coding standards, development practices, technical details, and project roadmap for AI assistants working on the GLM Experiments project.

---

## Table of Contents

- [Project Roadmap](#project-roadmap)
- [Experiments Directory](#experiments-directory)
- [Core Principles](#core-principles)
- [Code Quality Standards](#code-quality-standards)
- [Technology Stack](#technology-stack)
- [Installation and Setup](#installation-and-setup)
- [Project Architecture](#project-architecture)
- [Server Environment](#server-environment)
- [Agentic Git Flow](#agentic-git-flow)
- [Running Code](#running-code)
- [Testing](#testing)
- [Development Tools](#development-tools)
- [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
- [Best Practices](#best-practices)
- [Common Issues](#common-issues)

---

## Project Roadmap

This section tracks the development roadmap and feature planning for the gLM experiments project.

High-level goals:
- Perform experiments training genomic language models (gLMs)

Requirements for training framework:
- Reproducible (as long as you fix the seed)
- bf16
- torch compile
- DDP should work fine for having 8 gpus and 1 node:
   - fast
   - not bottlenecked by CPU (make sure to have the appropriate number of workers)
   - deterministic when setting seed
   - no dataloader bugs (e.g. make sure that data is not repeated or missed)
- If using transformers, use flash-attention (v2/v3) and rope

Testing approach:
- Test things that are easy and don't require GPU training (e.g., data loading, tokenization, config instantiation)
- Skip tests that require expensive compute (e.g., full training runs)

Next steps:
- Reproduce GPN-Animal-Promoter
Paper: https://www.biorxiv.org/content/10.1101/2025.02.11.637758v2
Code: https://github.com/songlab-cal/gpn/tree/main/analysis/gpn_animal_promoter
Model: https://huggingface.co/songlab/gpn-animal-promoter
Training dataset: https://huggingface.co/datasets/songlab/gpn-animal-promoter-dataset
Eval dataset: https://huggingface.co/datasets/songlab/TraitGym (subset to promoter variants)

---

## Experiments Directory

The `experiments/` directory is for exploratory analysis with less stringent code standards than the main package. Each experiment has its own isolated environment.

### Directory Structure

```
experiments/
├── evals/                       # Model evaluation experiments
│   ├── .venv/                  # Isolated Python environment (Python 3.13)
│   ├── workflow/               # Snakemake workflow
│   │   ├── Snakefile          # Main workflow definition
│   │   └── rules/             # Modular workflow rules
│   │       ├── common.smk     # Shared imports and genome download
│   │       ├── model.smk      # Model loading and LLR computation
│   │       ├── traitgym.smk   # TraitGym dataset processing
│   │       ├── sat_mut_mpra.smk  # Saturation mutagenesis MPRA data
│   │       ├── metrics.smk    # AUPRC and Spearman metrics
│   │       └── gnomad.smk     # gnomAD dataset (placeholder)
│   ├── config/                # Configuration files
│   │   └── config.yaml        # Dataset, model, and scoring config
│   ├── results/               # Generated outputs (gitignored)
│   │   ├── genome.fa.gz       # Reference genome
│   │   ├── model/             # Downloaded model checkpoints
│   │   ├── dataset/           # Processed datasets (parquet)
│   │   ├── features/          # LLR and absLLR scores
│   │   ├── prediction/        # Signed predictions
│   │   └── metrics/           # Final evaluation metrics (TSV)
│   ├── requirements.txt       # Python dependencies
│   └── README.md              # Setup and usage instructions
```

### experiments/evals: Model Evaluation Pipeline

**Purpose**: Evaluate genomic language models (specifically GPN-Animal-Promoter checkpoints) on variant effect prediction tasks using Snakemake workflows.

**Key Technologies**:
- **Snakemake**: Workflow management for reproducible analysis
- **biofoundation**: Library for genomic data and model inference (GPN adapter)
- **PyTorch**: Model loading and inference (bf16, torch compile configurable)
- **HuggingFace**: Model and dataset loading
- **Pandas**: Data processing with parquet files

### Workflow Overview

The evaluation pipeline consists of these stages:

1. **Genome Download** (`common.smk`):
   - Downloads human reference genome (GRCh38) from Ensembl

2. **Dataset Preparation** (`traitgym.smk`, `sat_mut_mpra.smk`):
   - **TraitGym Mendelian**: Mendelian trait variants (promoter region, non-exonic)
   - **TraitGym Complex**: Complex trait variants (promoter region, non-exonic)
   - **Saturation Mutagenesis MPRA**: 9 promoters (F9, GP1BA, HBB, HBG1, HNF4A, LDLR, MSMB, PKLR, TERT)

3. **Model Inference** (`model.smk`):
   - Downloads GPN-Animal-Promoter checkpoints (8 timepoints: 10k-500k steps)
   - Computes **Log Likelihood Ratios (LLR)** using masked language modeling
   - Computes **absolute LLR (absLLR)** for variant effect magnitude
   - Parallelized inference using `dataloader_num_workers=workflow.cores`

4. **Prediction Generation** (`model.smk`):
   - Converts features to signed predictions:
     - `LLR.minus.score`: -LLR (for pathogenicity)
     - `absLLR.plus.score`: absLLR (for effect magnitude)

5. **Metrics Calculation** (`metrics.smk`):
   - **AUPRC (Average Precision)**: For binary classification tasks (TraitGym)
   - **Spearman correlation**: For regression tasks (saturation mutagenesis)

### Configuration

**Key parameters in `config/config.yaml`**:
- `context_size: 512` - Sequence context window for model inference
- `per_device_batch_size: 128` - Batch size for inference
- `torch_compile: False` - Torch compilation (overhead not worth it for small datasets)
- `models`: Dictionary mapping checkpoint names to paths
- `sat_mut_mpra_promoter`: List of promoters to evaluate

### Running the Pipeline

```bash
# Navigate to experiment directory
cd experiments/evals

# Activate isolated environment
source .venv/bin/activate

# Dry run to see planned jobs
snakemake --cores all --dry-run

# Execute workflow
snakemake --cores all

# Execute specific target
snakemake --cores all results/metrics/traitgym_mendelian_promoter/AUPRC/10000_LLR.minus.score.tsv
```

### Results Structure

Final metrics are organized by:
- **Dataset**: `traitgym_mendelian_promoter`, `traitgym_complex_promoter`, `sat_mut_mpra_promoter_{PROMOTER}`
- **Metric**: `AUPRC`, `Spearman`
- **Model + Scoring**: `{checkpoint}_{scoring}.tsv` (e.g., `10000_LLR.minus.score.tsv`)

Example output:
```
results/metrics/traitgym_mendelian_promoter/AUPRC/10000_LLR.minus.score.tsv
```

### Development Guidelines

**For experiments directory**:
- ✅ Use separate `uv` environments per experiment (`.venv` in each subdir)
- ✅ Less stringent code standards than main package
- ✅ Focus on exploratory analysis and rapid iteration
- ✅ Use Snakemake for reproducible workflows
- ✅ Parquet format for intermediate data (efficient columnar storage)
- ❌ No need for pre-commit hooks, Black formatting, or comprehensive tests
- ❌ Don't integrate with main package testing/CI
- ⚠️ Keep experiments isolated and self-contained

**Key differences from main package**:
- Main package (`glm_experiments/`): Production training code, PyTorch Lightning, Hydra, strict standards
- Experiments (`experiments/`): Ad-hoc analysis, Snakemake workflows, exploratory

---

## Core Principles

### Simplify Relentlessly

**Remove complexity aggressively** - The simplest design that works is usually best. When in doubt, choose the simpler solution.

- Prefer straightforward implementations over clever abstractions
- Avoid premature optimization
- Remove unused code, dead code, and unnecessary abstractions
- If code becomes bloated or overwrought, simplify it

### DRY (Don't Repeat Yourself)

- Factor out common functionality into reusable components
- Use existing code and utilities rather than duplicating logic
- Check `glm_experiments/utils/` for existing functionality before creating new utilities
- When refactoring, identify and extract common patterns

### Test-Driven Development

- Write tests before or alongside implementation
- Tests should validate actual functionality, not just exercise code paths
- Prefer integration tests that test real behavior over unit tests with heavy mocking
- Use existing test fixtures from `tests/conftest.py` rather than creating new mocks

---

## Code Quality Standards

### Formatting and Style

- Follow existing project conventions:
  - **Black** formatting with 99 character line length
  - **isort** for import sorting (black profile)
  - **flake8** for linting (with project-specific ignores)
  - **pyupgrade** to use Python 3.13+ syntax
- Run `make format` or `uv run pre-commit run -a` before committing
- All code must pass pre-commit hooks

### Documentation

- All non-trivial functions must have comprehensive docstrings
- Use Google-style docstrings with type hints
- Document complex logic and design decisions
- Keep docstrings up-to-date with code changes

### Type Hints

- Use type hints for function parameters and return values
- Leverage Python 3.13+ type features
- Use `DictConfig` from `omegaconf` for Hydra configs

---

## Technology Stack

### Core Frameworks

- **PyTorch Lightning 2.x**: Framework for organizing PyTorch code, handles training loops, distributed training
- **Hydra 1.3+**: Configuration management framework enabling dynamic hierarchical configuration composition
- **Python 3.13**: Required Python version
- **uv**: Modern Python package manager (replaces conda/pip)

### Additional Tools

- **torchmetrics**: Proper metric calculation for multi-GPU setups
- **pytest**: Testing framework
- **pre-commit**: Code formatting and linting automation
- **GitHub CLI (`gh`)**: Command-line interface for GitHub operations
- **Black, isort, flake8**: Code formatting and linting tools

### Package Structure

- **Package name**: `glm-experiments` (hyphenated for PyPI)
- **Package directory**: `glm_experiments/` (underscore for Python imports)
- **Entry points**: `train_command` and `eval_command` via `pyproject.toml`

---

## Installation and Setup

### Prerequisites

- Python 3.13+
- Git
- GitHub CLI (`gh`) - Installation: https://cli.github.com/
- Access to project repository

### Installation Steps

```bash
# Clone repository
git clone https://github.com/your-org/glm-experiments.git
cd glm-experiments

# Install dependencies using uv
uv sync                  # Install main dependencies
uv sync --group dev      # Install development dependencies

# Install pre-commit hooks
uv run pre-commit install

# Verify installation
uv run python --version         # Should be 3.13+
uv run pytest --version        # Verify pytest is available
gh --version            # Verify GitHub CLI is installed
```

### GitHub CLI Setup

```bash
# Authenticate with GitHub
gh auth login

# Follow prompts to authenticate via web browser
# Verify authentication
gh auth status
```

### Environment Setup

Create a `.env` file in project root for private configuration (gitignored):

```bash
# .env file
WANDB_API_KEY=your_key_here
PROJECT_ROOT=/path/to/project
```

---

## Project Architecture

### Directory Structure

```
glm-experiments/
├── glm_experiments/              # Main package
│   ├── data/                    # Data modules and dataloaders
│   ├── models/                  # Lightning modules and components
│   ├── utils/                   # Utility functions
│   ├── train.py                 # Training entry point
│   └── eval.py                  # Evaluation entry point
│
├── configs/                     # Hydra configuration files
│   ├── callbacks/               # Callback configs
│   ├── data/                    # Data/datamodule configs
│   ├── debug/                   # Debugging configs
│   ├── experiment/              # Experiment configs (version controlled)
│   ├── logger/                  # Logger configs
│   ├── model/                   # Model configs
│   ├── trainer/                 # Trainer configs (CPU, GPU, DDP)
│   ├── eval.yaml                # Main evaluation config
│   └── train.yaml               # Main training config
│
├── tests/                       # Test suite
│   ├── conftest.py              # Shared fixtures
│   └── test_*.py                # Test files
│
├── CLAUDE.md                    # This file
└── pyproject.toml               # Dependencies and configuration
```

### PyTorch Lightning Patterns

#### Lightning Module Structure

All models inherit from `LightningModule`:

```python
from lightning import LightningModule

class MyModel(LightningModule):
    def __init__(self, net, optimizer, scheduler):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train/loss", loss)  # Use / separator
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val/loss", loss)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]
```

#### Logging Metrics

Use `/` separator for hierarchical organization:

```python
self.log("train/loss", loss)
self.log("train/acc", accuracy)
self.log("val/loss", val_loss)
```

#### Using torchmetrics

For proper metric calculation in multi-GPU:

```python
from torchmetrics import Accuracy

class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.train_acc(y_hat, y)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
```

### Hydra Configuration

#### Configuration Instantiation

Use `hydra.utils.instantiate()` for all dynamic objects:

```yaml
# configs/model/my_model.yaml
_target_: glm_experiments.models.my_module.MyLitModule
net:
  _target_: glm_experiments.models.components.my_net.MyNet
  input_size: 784
  hidden_size: 256
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
```

```python
# Instantiate from config
model = hydra.utils.instantiate(config.model)
```

#### Config Composition

Main configs use `defaults` for composition:

```yaml
# configs/train.yaml
defaults:
  - _self_
  - data: gpn_animal_promoter
  - model: bert_bytenet_small
  - trainer: gpu

# Override from command line
# uv run python glm_experiments/train.py trainer.max_epochs=20
```

#### Experiment Configs

Located in `configs/experiment/`:

```yaml
# configs/experiment/my_experiment.yaml
# @package _global_

defaults:
  - override /data: gpn_animal_promoter
  - override /model: bert_bytenet_small

tags: ["experiment", "baseline"]
seed: 42

data:
  batch_size: 128

model:
  optimizer:
    lr: 0.001
```

### Important Patterns

#### Task Wrapper

`@task_wrapper` decorator handles:
- Exception logging
- Wandb cleanup
- Output directory logging

All entry points use this decorator.

#### Root Utils

`rootutils.setup_root()` called at start to:
- Add project root to PYTHONPATH
- Set PROJECT_ROOT environment variable
- Load .env file

#### Tags

Tag experiments for tracking:

```bash
uv run python glm_experiments/train.py tags=["experiment", "baseline"]
```

---

## Server Environment

### AWS EFS Filesystem

This project runs on AWS EC2 with AWS EFS mounted storage:

- **Project location**: `/home/ubuntu/efs/projects/glm-experiments`
- **Filesystem type**: NFS4 (AWS EFS)
- **Persistent storage**: `~/efs` directory (survives instance restarts)
- **Non-persistent storage**: Home directory (`~`) is ephemeral
- **Shell configuration**: `~/efs/env` (sourced to set up environment)

### Installation Guidelines

- ⚠️ **Install all packages and tools to `~/efs`** - home directory is not persistent
- Use `uv` for Python package management
- Virtual environments should be in project directory or under `~/efs`
- Global tools must be installed to `~/efs/bin`
- The `~/efs/env` file configures PATH and environment variables - source it to access installed tools

### Finding Persistent Storage

On any system:

```bash
# List all filesystems
df -Th                    # Show all filesystems with types
findmnt                   # Show mount tree
lsblk -f                  # Show block devices

# Check specific path
df -Th /path/to/check
findmnt -T /path/to/check
```

**Patterns**:
- Network shares: Look for `nfs`, `nfs4`, `cifs` filesystem types
- Attached volumes: Look for `ext4`, `xfs`, `btrfs` on non-root partitions
- Cloud mounts: Check `/mnt`, `/data`, `/persistent`

**Rule of thumb**: Never assume home directories are persistent on cloud instances.

---

## Agentic Git Flow

This project uses a structured, issue-driven workflow with five phases.

### Workflow Phases

1. **Formalize goals** - Document planning, standards, validation
2. **Develop issues** - Create detailed issue specifications with agents
3. **Agent execution** - Let agent implement based on issue
4. **Code review cycles** - Human review with iterative refinement
5. **PR review and merge** - Independent review then squash-and-merge

### Phase 1: Formalize Goals

Ensure these documents exist:
- **CLAUDE.md** (this file) - Coding standards, development guide, and project roadmap
- **Design docs** - Feature specifications (if needed)

### Phase 2: Develop Issues

#### Step 1: Agent-Assisted Issue Development

1. Agent reads planning documents and codebase
2. Agent generates Markdown issue description with:
   - Clear problem statement
   - Proposed solution with code snippets
   - Test requirements
   - Acceptance criteria
3. Save as Markdown file (e.g., `issue-draft.md`)

#### Step 2: Human Review

4. Human reviews and edits Markdown directly
5. Optional: Code reviewer subagent reviews

#### Step 3: Create Issue

```bash
gh issue create --body-file issue-draft.md --title "Feature: descriptive title"
```

**Feature Strategy**: For large features, break into:
1. Refactoring issue (prepare codebase)
2. Implementation issue (add functionality)

### Phase 3: Agent Execution

**Recommended Agent Prompt:**
```
Read issue #[N] using `gh issue view [N]`.

Think hard to brainstorm clarifying questions.
If anything is unclear, STOP and ask before proceeding.

Pull latest main and create feature branch: [N]-descriptive-name

Copy code from issue description directly as starting point.
Refine as needed, but STOP if deviating significantly.

Continue until:
- All issue requirements 100% complete
- All tests pass (make test-full)
- Code formatted (make format)
```

### Phase 4: Code Review Cycles

#### Checkpoint 1: Issue Review (Before Creating)

- Review issue draft carefully
- Edit for clarity and completeness
- Use code reviewer subagent if helpful

#### Checkpoint 2: Mid-Implementation (During Development)

**Tools**:
- VSCode Source Control view
- Vim Fugitive / Emacs Magit
- `git diff` or `git diff --staged`

**Best Practices**:
- Commit/stage progress to preserve good work
- Use conversation rewinding (Esc twice) if agent goes off track
- Catch problems early

#### Checkpoint 3: Pre-Merge PR Review

After implementation complete, create PR and conduct independent review.

### Phase 5: PR Review and Merge

#### PR Review Checklist

Use separate Claude instance or human reviewer:

1. **Issue Compliance**
   ```bash
   gh issue view [N]
   gh pr view [PR] --web
   ```
   - Verify 100% completion of requirements

2. **Code Formatting**
   ```bash
   make format
   ```
   - Verify pre-commit hooks pass

3. **Documentation**
   - Verify docstrings comprehensive
   - Check Google-style format with type hints

4. **Clean Code Review**
   - Check for code smells
   - Verify project patterns followed

5. **Antipattern Scan**
   - Check for over-engineering
   - Verify DRY principles

6. **Design Compliance**
   - Verify Hydra `_target_` instantiation
   - Check PyTorch Lightning patterns

7. **Test Implementation Audit** ⚠️ **CRITICAL**
   - Verify NO fake implementations
   - Verify NO mock objects with fake data
   - Verify NO `pytest.mark.skip` without reason
   - **Flag fake tests as design problems**

8. **Test Suite**
   ```bash
   make test-full
   uv run pytest --cov=glm_experiments --cov-report=term-missing
   ```

9. **Quality Checks**
   ```bash
   make format
   ```

10. **Final Review**
    - No unnecessary complexity
    - Code is simple as possible
    - Implementation is maintainable

#### Creating PRs

```bash
# Push branch
git push -u origin [N]-descriptive-name

# Create PR
gh pr create --title "Closes #[N]: title" \
             --body "Implements requirements from issue #[N]" \
             --assignee @me
```

**PR Checklist**:
- [ ] All tests pass (`make test-full`)
- [ ] Code formatted (`make format`)
- [ ] All requirements 100% complete
- [ ] Real tests (no fakes!)
- [ ] Comprehensive docstrings
- [ ] Documentation updated
- [ ] No unnecessary complexity
- [ ] Branch named `[N]-description`

#### Merge Strategy

```bash
gh pr merge [PR] --squash --delete-branch
```

**Rationale**: Clean commit history where each commit represents validated code.

### GitHub CLI Commands

#### Issue Management

```bash
gh issue create --body-file file.md --title "Title"
gh issue view [N]
gh issue list
gh issue edit [N] --add-assignee @me
```

#### PR Management

```bash
gh pr create --title "Title" --body "Description"
gh pr view [PR]
gh pr view [PR] --web
gh pr list
gh pr status
gh pr merge [PR] --squash --delete-branch
```

#### Branch Management

```bash
git checkout -b [N]-descriptive-name
git push -u origin [N]-descriptive-name
```

### Small Steps Principle

**Work from working code to working code**:
- Each commit maintains working codebase
- Tests pass at each checkpoint
- Avoid large architectural changes in single PR
- Break complex features into refactoring + implementation

### Bloat Prevention

Before finalizing PR:
- Delete unused functions, classes, imports
- Remove unnecessary abstractions
- Simplify complex code
- Remove debug code

**Tools**:
```bash
flake8 --select=F401  # Find unused imports
grep -r "TODO\|FIXME" glm_experiments/
```

---

## Running Code

### Training

```bash
# CPU training
uv run python glm_experiments/train.py trainer=cpu

# GPU training
uv run python glm_experiments/train.py trainer=gpu

# Multi-GPU DDP
uv run python glm_experiments/train.py trainer=ddp trainer.devices=4

# With experiment config
uv run python glm_experiments/train.py experiment=example

# Override parameters
uv run python glm_experiments/train.py trainer.max_epochs=20 data.batch_size=64
```

### Evaluation

```bash
uv run python glm_experiments/eval.py ckpt_path="/path/to/checkpoint.ckpt"
```

### Debugging

```bash
# Fast dev run (1 epoch)
uv run python glm_experiments/train.py debug=default

# Fast dev run with 1 batch
uv run python glm_experiments/train.py debug=fdr

# Profiling
uv run python glm_experiments/train.py debug=profiler

# Overfit to 1 batch
uv run python glm_experiments/train.py debug=overfit
```

### Hyperparameter Search

```bash
# Grid search
uv run python glm_experiments/train.py -m data.batch_size=32,64,128 model.optimizer.lr=0.001,0.0005

# With experiment config
uv run python glm_experiments/train.py -m experiment=example data.batch_size=32,64
```

### Multiple Seeds

```bash
uv run python glm_experiments/train.py -m seed=1,2,3,4,5 trainer.deterministic=True logger=csv tags=["benchmark"]
```

---

## Testing

### ⚠️ CRITICAL ANTI-FAKE TEST BARRIER ⚠️

If you think "I'll just create a simple mock that returns..." or "I'll make a fake implementation..." then **STOP IMMEDIATELY**. This violates the real-testing principle.

Instead:
1. **Use existing fixtures**: Check `tests/conftest.py`
2. **Use actual models**: Load real models, don't mock
3. **Test real behavior**: Validate actual functionality
4. **Ask for help**: Hard testing = design smell

**The temptation to fake is a design smell** - address the underlying issue.

### Test Requirements

- All new features must have tests
- Tests in `tests/` directory, mirror source structure
- Use pytest fixtures from `conftest.py`
- Mark slow tests with `@pytest.mark.slow`
- Tests must pass with `pytest`

### Running Tests

```bash
# All tests
uv run pytest

# Exclude slow tests
uv run pytest -k "not slow"

# Specific file
uv run pytest tests/test_train.py

# With coverage
uv run pytest --cov=glm_experiments --cov-report=term-missing

# Using Makefile
make test        # Fast tests
make test-full   # All tests
```

### Writing Tests

```python
# tests/test_my_feature.py
import pytest

def test_my_feature(cfg_train):
    """Test with real config."""
    # Use actual implementation
    result = my_function(cfg_train)

    # Verify real behavior
    assert result.is_valid()
    assert result.metric > 0.0
```

---

## Development Tools

### Pre-commit Hooks

```bash
# Install (one-time)
uv run pre-commit install

# Run manually
uv run pre-commit run -a

# Or use Makefile
make format
```

**Hooks**:
- Trailing whitespace removal
- End of file fixes
- YAML validation
- Black formatting
- isort import sorting
- pyupgrade syntax updates
- flake8 linting
- bandit security checks
- prettier YAML formatting
- shellcheck for shell scripts

### Makefile Commands

```bash
make format       # Run pre-commit hooks
make test         # Fast tests
make test-full    # All tests
make clean        # Clean autogenerated files
```

### Logging and Experiment Tracking

**Log Structure**:
```
logs/
├── train/
│   ├── runs/YYYY-MM-DD_HH-MM-SS/
│   │   ├── .hydra/
│   │   ├── checkpoints/
│   │   └── wandb/
│   └── multiruns/
└── eval/
```

---

## Anti-Patterns to Avoid

### Fake Tests
- ❌ Mock objects with fake data
- ❌ Placeholder implementations
- ❌ `pytest.mark.skip` without reason
- ❌ Tests not validating real behavior

### Over-Engineering
- ❌ Premature abstractions
- ❌ Complex inheritance when composition works
- ❌ Overly generic code
- ❌ Unnecessary design patterns

### Code Duplication
- ❌ Copy-pasting instead of extracting
- ❌ Reimplementing existing utilities
- ❌ Not checking for existing solutions

### Configuration Complexity
- ❌ Hardcoded values
- ❌ Complex nested configs when flat works
- ❌ Not using Hydra composition

---

## Best Practices

### Agentic Git Flow
1. **Issue-driven development**: All work starts with GitHub issue
2. **Use `gh` CLI extensively**: Read issues, create PRs
3. **Feature branch naming**: `[N]-description` format
4. **Small steps**: Working code to working code
5. **Squash-and-merge**: Clean commit history

### Code Quality
6. **Simplify relentlessly**: Simplest design that works
7. **Write real tests**: No fakes or mocks
8. **DRY principle**: Factor out common functionality
9. **Bloat prevention**: Remove dead code
10. **Type hints**: Use Python 3.13+ features

### Project-Specific
11. **AWS EFS awareness**: Install to `~/efs`
12. **Use torchmetrics**: For multi-GPU metrics
13. **Name metrics with `/`**: e.g., `train/loss`
14. **Hydra instantiation**: Use `hydra.utils.instantiate()`
15. **Python 3.13**: Required version

---

## Common Issues

### Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**: `rootutils.setup_root()` called in entry points automatically

### Config Not Found

**Problem**: Hydra can't find config

**Solution**: Check `_target_` paths match actual module structure:
```yaml
_target_: glm_experiments.models.my_model.MyModel  # Must match import
```

### DDP Issues

**Problem**: Distributed training fails

**Solution**:
- Check `trainer=ddp` config
- Verify GPUs with `nvidia-smi`
- See PyTorch Lightning docs

### Path Issues

**Problem**: Files not found

**Solution**: Use Hydra path interpolation:
```yaml
data_dir: ${paths.root_dir}/data
# or
data_dir: ${oc.env:PROJECT_ROOT}/data
```

### Tests Failing in CI

**Problem**: Tests pass locally, fail in CI

**Solution**:
- Ensure deps in `pyproject.toml`
- No hardcoded paths - use fixtures
- Tests don't depend on missing data

### Pre-commit Failing

**Problem**: Hooks fail on commit

**Solution**:
```bash
make format
# Or specific hook
uv run pre-commit run black --all-files
```

---

## Quick Reference

### Essential Commands

```bash
# Development
make format              # Format code
make test               # Fast tests
make test-full          # All tests
uv sync                 # Install deps
uv sync --group dev     # Install dev deps

# Git workflow
gh issue view [N]                                    # Read issue
git checkout -b [N]-description                      # Create branch
gh pr create --title "Title" --body "Description"   # Create PR
gh pr merge [PR] --squash --delete-branch           # Merge PR

# Training
uv run python glm_experiments/train.py trainer=gpu          # Train on GPU
uv run python glm_experiments/train.py experiment=example   # Run experiment
uv run python glm_experiments/train.py debug=fdr            # Fast debug
```

### Key Files

```
CLAUDE.md                        # This file
glm_experiments/                # Main package
configs/                        # Hydra configs
tests/                          # Test suite
pyproject.toml                  # Dependencies
```

---

## References

- **[PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)** - Official docs
- **[Hydra Docs](https://hydra.cc/)** - Official docs
- **[Agentic Git Flow Blog](https://matsen.group/general/2025/11/01/agentic-git-flow.html)** - Workflow inspiration
- **`.pre-commit-config.yaml`** - Formatting configuration
- **`pyproject.toml`** - Dependencies and pytest config
- **`tests/conftest.py`** - Test fixtures
- remove the statement about not testing. let's do test things that are easy and not require e.g. gpu training.
- You should follow my instructions in @CLAUDE.md (draft an issue in markdown, let me approve, then you create the issue).
- the experiments directory is for more exploratory analysis (less stringent code standards), managed with separate uv environments.
