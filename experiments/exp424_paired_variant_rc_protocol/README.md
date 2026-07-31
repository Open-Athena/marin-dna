# exp424: paired-variant and reverse-complement feature protocol

This permanent unmerged experiment analyzes the hash-verified issue #422 sparse
SAE activations for issue #424. It does not rerun the genomic language model.

The analysis asks three questions without looking at held-out test results during
selection:

1. Does a feature respond to the locus in both alleles, switch on/off across the
   variant, or change magnitude while remaining active?
2. Does the same feature ID behave similarly on the forward and
   reverse-complement sequence, or is a different SAE feature the better match?
3. Which fixed same-ID strand reducer (signed mean, absolute mean, RMS, or max)
   transfers best from validation to the held-out test split?

Run from this directory after the issue #422 extraction and analysis artifacts
are available:

```bash
uv sync --dev
uv run pytest
ANALYSIS_COMMIT=<40-character-commit> uv run python analyze.py \
  --extraction-dir ../../scratch/issue422/run \
  --panel ../../scratch/issue422/input/panel.parquet \
  --prior-selection ../../scratch/issue422/analysis/selected_individual_features.parquet \
  --output-dir ../../scratch/issue424/analysis
```

All inputs are hash-checked against the issue #422 extraction manifest. The
output manifest records those hashes, the analysis commit, protocol constants,
and hashes of every result artifact.
