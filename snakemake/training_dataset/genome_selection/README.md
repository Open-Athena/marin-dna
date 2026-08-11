# Genome Selection Pipeline

This pipeline downloads genome metadata from NCBI RefSeq for a specified taxonomic group and filters them based on quality criteria and taxonomic deduplication.

## What it does

1. **Download genome metadata** - Queries NCBI RefSeq for all genomes within a specified taxon (e.g., "metazoa")
2. **Download taxonomy** - Retrieves taxonomic classification information for each genome
3. **Filter genomes** - Applies filters based on:
   - Assembly quality level (e.g., Complete Genome, Chromosome, Scaffold, Contig)
   - Genome size (to ensure compatibility with downstream tools)
   - Priority preferences (favors specific reference assemblies)
4. **Deduplicate** - Keeps only one genome per taxonomic rank (e.g., one genome per family)

## Setup

Python dependencies are managed by the main project (see `../../../README.md` for installation).

Conda must be available for bioinformatics CLI tools.

### Storage

Pipeline results are stored in S3 (`s3://oa-bolinas/snakemake/training_dataset/genome_selection/`). A default Snakemake profile at `workflow/profiles/default/config.yaml` configures S3 storage, conda, and cores automatically.

You need AWS credentials with S3 access:
- **On EC2**: Attach an IAM role with `AmazonS3FullAccess` to the instance
- **On your laptop**: Run `aws configure` with an IAM user's access key

## Configuration

Edit `config/config.yaml` to customize the pipeline:

### Required Parameters

- **`taxon`** - NCBI taxonomic group to query (e.g., "metazoa", "vertebrata", "mammalia")
  - Browse available taxa at https://www.ncbi.nlm.nih.gov/datasets/genome/

- **`max_genome_size`** - Maximum genome size in bytes (default: 4,000,000,000 = 4 GB)
  - Ensures compatibility with faToTwoBit tool

- **`min_assembly_level`** - Minimum assembly completeness (default: "Contig")
  - Options: "Complete Genome", "Chromosome", "Scaffold", "Contig"
  - Higher levels are more complete but fewer genomes available

- **`deduplicate_taxonomic_rank`** - Taxonomic rank for deduplication (default: "family")
  - Keeps only one genome per rank (e.g., one genome per family)
  - Common options: "species", "genus", "family", "order", "class", "phylum"

### Optional Parameters

- **`priority_genomes`** - List of preferred genome accessions (e.g., GCF_000001405.40 for human GRCh38)
  - When multiple genomes in same rank, prioritizes these

- **`exclude_genomes`** - List of genome accessions to exclude
  - Useful for removing problematic assemblies

## Usage

```bash
cd snakemake/training_dataset/genome_selection
uv sync --locked --group dev
uv run --locked pytest
uv run --locked snakemake -n
uv run --locked snakemake
```

## Output

- **`results/genomes/filtered.parquet`** - Curated list of genomes with metadata
  - Contains: accession, species name, assembly level, genome size, taxonomic classification
  - Ready to use as input for the dataset creation pipeline
