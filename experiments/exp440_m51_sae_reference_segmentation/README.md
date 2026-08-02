# exp440: m5.1 SAE reference segmentation

This permanent, unmerged experiment branch implements the preregistered first
stage of [issue #440](https://github.com/Open-Athena/marin-dna/issues/440): a
balanced high-purity human-reference panel for testing whether layer-resolved
m5.1 SAE features identify local gene and regulatory annotation states.

The first stage is deliberately focal and local. It is not a genome-wide dense
segmentation model and it makes no claim about long-range regulatory
interactions.

## Frozen panel

The panel contains 2,048 deterministic 255 bp windows from each of seven
disjoint classes: CDS, 3′ UTR, TSS-region/5′ UTR, ncRNA exon, non-promoter cCRE,
pure intron, and pure intergenic sequence. Candidate windows must have a
class-specific fraction of exactly 1.0. Sampling takes the smallest BLAKE2b-128
hashes of `class|name`, so row selection does not depend on sequence, model
activation, or biological association strength.

Inputs are the published v4 human annotation windows, their phyloP summaries,
and the matching Ensembl release 115 GRCh38 two-bit reference. The builder
checks exact byte counts and SHA-256 digests before reading any input and emits
a manifest with the selected class counts and output digest.

## Build the panel

From this directory:

```bash
uv sync --group dev

EXPERIMENT_COMMIT=<40-character-commit> \
RUN_ID=dna-exp440-reference-state-panel-r1 \
uv run python build_panel.py \
  --labels /path/to/min0.20.parquet \
  --phylop /path/to/phyloP_447m_windows.parquet \
  --genome /path/to/genome.2bit \
  --output-dir /path/to/dna-exp440-reference-state-panel-r1
```

The command refuses to overwrite an existing output directory. It writes
`panel.parquet` first and `manifest.json` last.

Run the focused tests with:

```bash
uv run pytest
```

## Downstream analysis

The next scripts on this branch will extract center-position activations from
the frozen block 1, 10, and 19 SAEs for FWD and RC separately and run the
preregistered one-vs-rest association scan. Experimental results and plots
belong in issue #440 rather than this runbook.
