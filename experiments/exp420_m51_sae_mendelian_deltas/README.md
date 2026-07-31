# exp420: Mendelian variant SAE deltas

This unmerged experiment asks whether ref-to-alt changes in the issue #418
m5.1 block-10 JumpReLU SAE expose interpretable mechanisms associated with
pathogenic labels and functional subsets in the official Mendelian-traits
evaluation dataset.

The analysis is tracked in GitHub issue #420. Its chromosome split and feature
selection protocol were frozen in the issue before variant activations were
inspected. Dataset positions are converted from VCF-style 1-based coordinates
to 0-based coordinates only at the FASTA boundary; all internal intervals are
0-based half-open.

## Local preparation and tests

```bash
uv lock
uv sync --frozen
uv run pytest
uv run python analysis.py prepare --output ../../scratch/issue420/input/panel.parquet
```

`prepare` downloads the pinned public Hugging Face dataset, joins the existing
official m5.1 score/probe outputs from S3, validates every matched group, and
writes a compact input parquet plus manifest. The GPU job receives that parquet,
the already verified SAE, and the canonical GRCh38 FASTA as file mounts; it does
not receive an AWS credential.

## GPU run

From this directory, first inspect the planned SkyPilot action:

```bash
sky launch -d --dryrun sky.yaml
```

Then launch with the committed experiment SHA:

```bash
sky launch -d -c dna-exp420-mendelian-deltas \
  --env EXPERIMENT_COMMIT=<40-character-commit> sky.yaml
```

Retrieve `artifacts/$RUN_ID`, validate hashes and tables locally, and bring the
cluster down immediately. The run writes compact selected-feature results,
per-row selected scores, and contexts rather than persisting the full dense
feature-delta matrices.

## Forward/RC score aggregate

After inspecting the two orientations separately, combine their independently
selected scores without averaging feature IDs:

```bash
uv run python aggregate.py \
  --selected-scores <run>/selected_scores.parquet \
  --results-json <run>/results.json \
  --output-dir <run>/aggregate
```

The aggregator centers scores within each unlabeled match group, scales each
orientation by its discovery+validation standard deviation, and takes a fixed
equal-weight mean. No test labels or test-tuned weights enter the aggregate.
