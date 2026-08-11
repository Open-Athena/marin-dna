# Training Dataset Pipeline

This pipeline creates genomic training datasets from NCBI RefSeq genomes. It downloads genomes, filters them based on taxonomic and quality criteria, extracts specific genomic regions (promoters, exons, CDS), and creates training/validation datasets that are uploaded to HuggingFace.

## Pipeline Stages

### 1. Genome Selection (`genome_selection/`)

Downloads and filters genomes from NCBI RefSeq based on:
- Taxonomic group (e.g., metazoa, vertebrates)
- Assembly quality level
- Genome size
- Taxonomic deduplication

**Output:** A curated list of genomes (`filtered.parquet`) for downstream dataset creation.

See [genome_selection/README.md](genome_selection/README.md) for detailed configuration and usage.

### 2. Dataset Creation (`dataset_creation/`)

Takes the filtered genome list and:
- Downloads genome assemblies and annotations
- Extracts genomic intervals (promoters, exons, CDS)
- Creates sliding windows over regions
- Generates train/validation splits
- Shards and compresses data
- Uploads to HuggingFace

**Output:** Training datasets on HuggingFace Hub, organized by taxonomic groups and interval types.

See [dataset_creation/README.md](dataset_creation/README.md) for detailed configuration and usage.

## Prerequisites

- Python dependencies: Each runnable pipeline owns a project-local `pyproject.toml` and committed `uv.lock`
- Conda: Required for bioinformatics CLI tools (bedtools, twoBitToFa, etc.)

## Typical Workflow

1. Configure and run genome selection:
   ```bash
   cd genome_selection
   # Edit config/config.yaml
   uv sync --locked --group dev
   uv run --locked snakemake -n
   uv run --locked snakemake
   ```

2. Use the output (`genome_selection/results/genomes/filtered.parquet`) as input for dataset creation:
   ```bash
   cd dataset_creation
   # Copy or symlink genome list to config/genomes.parquet
   # Edit config/config.yaml
   uv sync --locked --group dev
   uv run --locked snakemake -n
   uv run --locked snakemake
   ```
