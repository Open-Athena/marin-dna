# Experiment 428: replicate codon-context SAE features

This permanent experiment branch implements the preregistered replication in [issue #428](https://github.com/Open-Athena/marin-dna/issues/428), informing the durable research question in [issue #288](https://github.com/Open-Athena/marin-dna/issues/288). It is intentionally self-contained and is not intended to merge into `main`.

## Scientific design

The experiment freezes the three block-19 SAE features found in #426 before constructing or scoring the new panel. The primary endpoint is held-out, phase-and-transcript-substitution-conditional AUROC for feature 11064 from the 5M-activation SAE using the maximum-absolute forward/reverse-complement reducer and direction +1. Secondary endpoints use feature 12658 from the 5M SAE and feature 13637 from the 25M SAE with signed-mean aggregation.

The panel is drawn from chromosome 21 missense and synonymous variants in `songlab/hg38-variant-consequences`. One-megabase genomic blocks are assigned once to discovery, validation, or test (17/6/6 blocks), preventing positional leakage. Ensembl release-109 CDS annotations reconstruct transcript strand, codon phase, codons, and coding consequences using 0-based, half-open internal coordinates. The sampler retains only variants with consensus strand and codon position among transcripts reproducing the source consequence, then balances the two labels exactly within every retained codon-position × transcript-oriented substitution stratum in each split.

Discovery data fit a positional-31-bp-plus-alt logistic baseline. Validation chooses only its regularization strength. The untouched test blocks compare that baseline with the three frozen SAE scores. Confidence intervals use 1,000 label-by-stratum genomic-block bootstrap replicates. FWD and RC are always retained separately before applying the frozen aggregate.

## Local checks

From this directory:

```bash
uv sync --frozen
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Build the panel only from the commit that will be cited from the issue:

```bash
EXPERIMENT_COMMIT="$(git rev-parse HEAD)" uv run python panel.py \
  --source ../../scratch/issue422/source/21.parquet \
  --gtf ../../scratch/issue426/Homo_sapiens.GRCh38.109.gtf.gz \
  --fasta ../../scratch/issue418/reference/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz \
  --output-dir ../../scratch/issue428/panel
```

`panel.py` applies native-library thread caps before importing NumPy or Polars, requires at least 6 GiB of available memory, and holds the shared nonblocking `/tmp/marin-dna-local-heavy.lock` for the complete local build. Every other task on the shared node must use the same lock for potentially heavy local work. If the lock or memory gate fails, move the work to Sky rather than bypassing the guard.

The GPU extraction and registered analysis commands are added after the panel passes its preflight invariants. The intended GPU is one AWS A10G; no second paid resource may be launched while another paid Sky resource is active.

## Output contract

`panel/manifest.json` pins input hashes, coordinate conversions, block assignment, candidate sampling, transcript annotation coverage, exact stratum balance, and the panel hash. Subsequent extraction and analysis manifests will pin the two reused SAE exports, selected feature IDs, ref/alt × FWD/RC activations, registered score construction, test metrics, bootstrap intervals, and figure hashes.
