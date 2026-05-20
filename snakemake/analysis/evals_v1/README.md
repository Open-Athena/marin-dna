# Evaluation Pipeline v1

This pipeline evaluates genomic language models by computing variant effect predictions and comparing them against labeled datasets.

## Overview

The pipeline:
1. Downloads evaluation datasets from HuggingFace (created by `snakemake/evals/`)
2. Downloads genome reference (GRCh38)
3. Runs inference to compute LLR scores and embedding distances for each variant using model checkpoints
4. Computes evaluation metrics (AUPRC, AUROC, Spearman) globally and on annotation subsets
5. Generates comparison plots

## Setup

### Python Dependencies

The pipeline uses the main project's Python environment. If you haven't already installed dependencies:

```bash
cd /path/to/marin-dna
uv sync
```

### Storage

Pipeline results are stored in S3 (`s3://oa-bolinas/snakemake/analysis/evals_v1/`). A default Snakemake profile at `workflow/profiles/default/config.yaml` configures this automatically, so no extra flags are needed.

You need AWS credentials with S3 access:
- **On EC2**: Attach an IAM role with `AmazonS3FullAccess` to the instance
- **On your laptop**: Run `aws configure` with an IAM user's access key

### Configuration

Edit `config/config.yaml` to specify:

1. **Models to evaluate**: Specify training runs with **either** a local
   `base_path` **or** a `gcs_path` (parent of `step-{N}/`), plus context
   size and the steps to score:
   ```yaml
   models:
     # Local checkpoint (legacy convention — typically EFS):
     - name: gpn_promoter
       base_path: /path/to/training/run
       context_size: 512
       steps: [10000, 20000, 50000, 100000]

     # GCS checkpoint (new) — pulled by `download_model_step` from
     # rules/download.smk. `context_size: 255` if the tokenizer prepends BOS.
     - name: exp136-proj_v30
       gcs_path: gs://marin-us-central1/checkpoints/.../hf
       context_size: 255
       steps: [9999]
   ```

2. **Datasets**: Evaluation datasets from HuggingFace
   ```yaml
   datasets:
     - name: traitgym_mendelian
       hf_path: gonzalobenegas/bolinas_evals-traitgym_mendelian
       split: test
       metrics: [AUPRC, AUROC]

     # Full TraitGym Mendelian benchmark (3380 vars). `train+test` of our
     # derivative covers the same coordinates as `songlab/omim_traitgym`'s
     # canonical test split and keeps the per-subset annotations.
     - name: traitgym_mendelian_full
       hf_path: gonzalobenegas/bolinas_evals-traitgym_mendelian
       split: train+test
       metrics: [AUPRC]
   ```

3. **Inference settings**: Performance tuning (doesn't affect output, won't trigger recomputation)
   ```yaml
   inference:
     batch_size: 128
     num_workers: 4
     data_transform_on_the_fly: true
     torch_compile: false  # Enable for faster inference (requires PyTorch 2.0+)
   ```

## Usage

Run from the pipeline directory:

```bash
cd snakemake/analysis/evals_v1
```

Run the complete pipeline:

```bash
uv run snakemake
```

For GPU-bound inference on `gcs_path`-backed checkpoints, use the SkyPilot
launcher (mirrors `evals_v2/sky/run.yaml`):

```bash
sky launch snakemake/analysis/evals_v1/sky/run.yaml -c evals-v1
# subsequent runs (cluster reuse):
sky exec evals-v1 snakemake/analysis/evals_v1/sky/run.yaml
```

Run specific targets:

```bash
# Just compute scores for one dataset/model
uv run snakemake results/scores/traitgym_mendelian/gpn_promoter/10000.parquet

# Just compute metrics for one dataset/model/step
uv run snakemake results/metrics/traitgym_mendelian/gpn_promoter/10000.parquet

# Just generate plot for one model
uv run snakemake results/plots/metrics_vs_step/gpn_promoter.svg
```

Dry run to see what will be executed:

```bash
uv run snakemake -n
```

## Output

### Directory Structure

All results are stored in S3 under `s3://oa-bolinas/snakemake/analysis/evals_v1/results/`. Snakemake stages files locally as needed and uploads outputs back to S3 automatically.

```
results/
├── genome.fa.gz                        # GRCh38 reference genome
├── scores/
│   └── {dataset}/
│       └── {model}/
│           └── {step}.parquet         # Variant scores (LLR + embedding distances)
├── metrics/
│   └── {dataset}/
│       └── {model}/
│           └── {step}.parquet         # Metrics (global + per subset)
└── plots/
    └── metrics_vs_step/
        └── {model}.svg                # Metric progression across training for each model
```

### Scores Files

Parquet files with columns (aligned by row index with source dataset):
- `llr`: Raw log-likelihood ratio
- `minus_llr`: Negated LLR (higher = more deleterious)
- `abs_llr`: Absolute LLR (higher = more impactful)
- `embed_last_l2`: L2 distance between reference and alternate embeddings (last layer)
- `embed_middle_l2`: L2 distance between reference and alternate embeddings (middle layer)

### Metrics Files

Parquet files with columns:
- `metric`: Metric name (AUPRC, AUROC, Spearman)
- `score_type`: Scoring method (minus_llr, abs_llr, embed_last_l2, embed_middle_l2)
- `subset`: Annotation subset or "global"
- `value`: Metric value

Note: `step` and `dataset` are encoded in the file path, not as columns.

### Plots

- **metrics_vs_step/{model}.svg**: Per-model plots showing metric progression across training steps. Each subplot shows a (dataset, subset) combination with lines for each scoring method (minus_llr, abs_llr, embed_last_l2, embed_middle_l2).

## Annotation Subsets

Datasets created by `snakemake/evals/` include a `subset` column with annotation categories:
- `noncoding_transcript_exon`: Non-coding transcript exon variants
- `three_prime_UTR`: 3' UTR variants
- `five_prime_UTR`: 5' UTR variants
- `proximal_nonexonic`: Proximal non-exonic variants (near genes)
- `distal_nonexonic`: Distal non-exonic variants (far from genes)

Metrics are computed both globally (all variants) and separately for each subset.

## Implementation Details

### Code Organization

The pipeline uses a clean separation between Snakemake rules and Python logic:

- **`src/marin_dna/evals/`**: Core Python module with type hints and tests
  - `inference.py`: LLR and embedding distance computation using biofoundation
  - `metrics.py`: Metric computation functions
  - `plotting.py`: Plotting utilities

- **`workflow/rules/`**: Thin Snakemake wrappers
  - `inference.smk`: Download datasets and run inference
  - `metrics.smk`: Compute and aggregate metrics
  - `plots.smk`: Generate plots

### Dependencies

- **biofoundation**: LLR and embedding distance inference utilities
- **transformers**: HuggingFace model loading
- **datasets**: HuggingFace dataset loading
- **pandas**: Data manipulation
- **scikit-learn**: Metrics (AUPRC, AUROC)
- **scipy**: Statistics (Spearman correlation)
- **matplotlib**, **seaborn**: Plotting

## Troubleshooting

### Out of Memory

Reduce batch size in config:
```yaml
inference:
  batch_size: 64  # Default is 128
```

### Missing Checkpoints

Verify checkpoint paths in config exist:
```bash
ls /path/to/training/run/step-10000
```

### HuggingFace Authentication

Some datasets may require authentication:
```bash
huggingface-cli login
```

## Extending the Pipeline

### Adding New Datasets

1. Create dataset using `snakemake/evals/` pipeline
2. Add to config:
   ```yaml
   datasets:
     - name: my_dataset
       hf_path: username/bolinas_evals-my_dataset
       split: test
       metrics: [AUPRC]
   ```

### Adding New Metrics

1. Add metric function to `src/marin_dna/evals/metrics.py`:
   ```python
   METRIC_FUNCTIONS["MyMetric"] = lambda label, score: my_metric_fn(label, score)
   ```

2. Reference in config:
   ```yaml
   datasets:
     - name: my_dataset
       metrics: [AUPRC, MyMetric]
   ```

### Adding New Plots

1. Implement plotting function in `src/marin_dna/evals/plotting.py`
2. Add rule in `workflow/rules/plots.smk`
3. Add output to `rule all` in `Snakefile`
