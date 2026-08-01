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

## Direct consequence-subset analysis

The remaining issue scope uses a separate all-feature extraction and analysis
task, leaving the reproduced pathogenic-label workflow unchanged:

```bash
sky launch -d --dryrun sky.subsets.yaml
sky launch -d -c dna-exp420-mendelian-subsets \
  --env EXPERIMENT_COMMIT=<40-character-commit> sky.subsets.yaml
```

`extract_all_features.py` runs FWD and RC together in one bf16 prediction loop,
uses four CPU data-loader workers, disables KV caching, and enables
`torch.compile`. It stores the union of nonzero reference/alternate SAE
activations as two compressed Parquet files, preserving `ref_activation`,
`alt_activation`, and signed `delta`, plus a 41-bp context table. Dense feature
matrices are temporary analysis state and are never transferred.

`subset_analysis.py` ranks signed and absolute candidate features on discovery
chromosomes, selects only on validation chromosomes, and evaluates chr11/X
once. Missense-versus-synonymous is reported separately for `label=0` and
`label=1`, first for FWD/RC and then for their equal-weight pretest-standardized
score mean. It also reports substitution and focal-excluded GC controls and a
four-class analysis restricted to subsets with at least 20 match groups in
every split. All output artifacts are hash-complete.

## Primary prediction targets

The follow-up analysis reorganizes the results around the three intended
endpoints: `label` pooled across all subsets without stratification, `label`
within each subset, and `subset` pooled across both labels. The primary metric
is held-out row AUPRC. Label prevalence is exactly 10% in every match group and
therefore both within each subset and overall.

```bash
uv run python prediction_targets.py \
  --panel <panel.parquet> \
  --extraction-dir <all-feature-extraction> \
  --within-subset-summary <aggregate_summary.parquet> \
  --within-subset-manifest <aggregate_manifest.json> \
  --output-dir <prediction-target-output> \
  --extraction-commit <all-feature-extraction-commit> \
  --analysis-commit <prediction-analysis-commit>
```

The script is CPU-only and uses `prediction_primitives.py` so it does not load
the model, SAE Lens, or CUDA dependencies. It verifies every input hash before
reconstructing one orientation at a time, then writes hash-complete tables,
plots, and a Markdown summary.
