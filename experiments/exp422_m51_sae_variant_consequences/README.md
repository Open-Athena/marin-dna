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
  --output ../../scratch/issue422/input/panel.parquet \
  --fasta ../../scratch/issue418/reference/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz
```

The sampler uses deterministic 1 Mb genomic-block splits and selects 256
discovery, 128 validation, and 128 held-out variants from each of 35
`consequence_cre` classes. Before exact selection, every oversampled candidate
must have an A/C/G/T-only 255 bp GRCh38 window and an exact center-REF match;
invalid candidates are recorded in the manifest and deterministically replaced
within the same class/split. Reuse the written panel and its manifest unchanged
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

## Held-out analysis

After retrieving and hash-validating the extraction directory, run the sparse
individual-feature, multiclass-probe, sequence-control, context, and plot
analysis locally:

```bash
export ANALYSIS_COMMIT="$(git rev-parse HEAD)"
uv run python analyze.py \
  --extraction-dir ../../scratch/issue422/run \
  --panel ../../scratch/issue422/input/panel.parquet \
  --fasta ../../scratch/issue418/reference/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz \
  --output-dir ../../scratch/issue422/analysis
```

Feature and transform selection use discovery/validation blocks only. Test AUPRC
and macro-F1 are read once, with AUPRC intervals bootstrapped over held-out 1 Mb
blocks. The script reports FWD and RC separately before the fixed arithmetic-mean
view and includes raw-residual, substitution, and 31 bp k-mer-delta controls.
