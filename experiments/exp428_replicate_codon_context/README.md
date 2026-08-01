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

## Launch and retrieval

The CPU panel and analysis stages use one AWS `m6i.2xlarge`; the fixed-feature extraction uses one AWS `g5.2xlarge` A10G. The CPU and GPU may overlap, but no second CPU or GPU worker is needed. The CPU cluster autodowns after 90 idle minutes and the GPU after 30. At current Sky estimates they cost about $0.38/hour and $1.21/hour respectively.

Commit and push the complete protocol before any registered work. `launch.py` prints commands by default and mutates cloud state only with `--execute`:

```bash
COMMIT="$(git rev-parse HEAD)"
uv run python launch.py panel --commit "$COMMIT" --execute

mkdir -p ../../scratch/issue428/retrieval
sky rsync-down exp428-cpu \
  '~/exp428-artifacts/panel' \
  ../../scratch/issue428/retrieval/

uv run python launch.py extract --commit "$COMMIT" --execute
sky rsync-down exp428-gpu \
  '~/exp428-artifacts/extraction' \
  ../../scratch/issue428/retrieval/

uv run python launch.py analyze --commit "$COMMIT" --execute
sky rsync-down exp428-cpu \
  '~/exp428-artifacts/analysis' \
  ../../scratch/issue428/retrieval/
```

Each Sky task clones and checks out the exact `EXPERIMENT_COMMIT`; the scripts independently assert that it is the current checkout. The CPU task records `/usr/bin/time -v` peak RSS for the real panel and analysis. The GPU task records CUDA peak allocation/reservation and deliberately uses eager `run_with_cache`: the pinned Qwen/SAELens hook path is known-correct in eager mode, while the prior compile attempt did not preserve the dynamic hook cache.

## Output contract

`panel/manifest.json` pins input hashes, coordinate conversions, block assignment, candidate sampling, transcript annotation coverage, exact stratum balance, and the panel hash. `extraction/manifest.json` verifies the exact #426 source commit and hashes of both reused SAE exports, then pins the three selected feature IDs and dense ref/alt × FWD/RC outputs. `analysis/manifest.json` pins the registered score construction, discovery/validation-only sequence baseline, untouched test metrics, 1,000 block-bootstrap intervals, orientation diagnostics, and hashes of both SVG/PNG figures.
