# exp422: m5.1 SAE variant-consequence features

This unmerged experiment tests whether reference-to-alternate changes in the
issue #418 m5.1 block-10 SAE distinguish broad VEP and CRE-aware consequence
classes. The analysis and results are tracked in GitHub issue #422.

The first panel is intentionally limited to chromosome 21. Dataset positions
remain 1-based in the frozen panel and are converted to 0-based coordinates only
at the FASTA boundary during sequence extraction.

## Frozen panel

Download the pinned chr21 shard once, then build the deterministic balanced
panel:

```bash
uv run hf download songlab/hg38-variant-consequences 21.parquet \
  --repo-type dataset \
  --revision eb3022cc6797b9369cca16af72ff3c4197df343a \
  --local-dir ../../scratch/issue422/source

uv run python sample_panel.py \
  --input ../../scratch/issue422/source/21.parquet \
  --output ../../scratch/issue422/input/panel.parquet
```

The sampler uses deterministic 1 Mb genomic-block splits and selects 256
discovery, 128 validation, and 128 held-out variants from each of 35
`consequence_cre` classes. Reuse the written panel and its manifest unchanged
when comparing future SAE layers, training budgets, seeds, or dictionaries.

## Tests

```bash
uv lock
uv sync --frozen
uv run pytest
uv run ruff check .
```

## GPU extraction

`extract.py` preserves forward and reverse-complement results separately. For
each orientation it writes sparse reference/alternate SAE activations and a
dense raw-residual delta array. `manifest.json` pins the model revision, block,
SAE weights and configuration, panel/source hashes, protocol, and every output
hash so the same panel can be compared across later SAE versions.

The checked-in Sky task mounts the frozen panel, issue #418 SAE, and GRCh38
reference. Dry-run it before requesting an EC2 A10G:

```bash
sky launch -y -d --dryrun sky.yaml
```

After explicit approval for paid compute, launch with the commit containing the
extractor:

```bash
sky launch -y -d -c dna-exp422-consequence \
  --env EXPERIMENT_COMMIT=<40-character-commit> sky.yaml
```

The task uses one EC2 A10G, validates CUDA during setup, auto-stops after 30
idle minutes, and writes results under
`artifacts/dna-exp422-variant-consequences-seed288/`. Download that directory
before the cluster is torn down.
