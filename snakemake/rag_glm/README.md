# `rag_glm`

Issue #402's isolated prototype pipeline for fixed-layout retrieval-augmented
genomic-language-model documents. It consumes the completed Zoonomia projection
as an immutable upstream artifact; it never runs `halLiftover` or overwrites
outputs from `snakemake/zoonomia_projection_dataset/`.

All genomic coordinates are 0-based, half-open.

## Inputs

- `s3://oa-bolinas/snakemake/zoonomia_projection_dataset/results/projection/min0.20/all_species_with_sequence.parquet`
- The committed 108-species family-deduplicated cohort in
  `../zoonomia_projection_dataset/config/species_zoonomia_447_family_dedup.tsv`

The conservation/projection snapshot is:

- human reference: GRCh38, Ensembl release 115
- conservation score: `phyloP_447m`
- base threshold: `2.2162`
- 255-base window filter: at least `0.20` of positions above the threshold
- projection: Zoonomia 447-mammalian 2022 v1 Cactus HAL

## Phase-A species audit

`rule species_audit` deterministically samples 8,192 human anchors and reports:

- projection success for every species in the 108-species source cohort;
- ambiguous-window and ambiguous-base rates;
- a same-position sequence-identity proxy for redundancy among the provisional
  seven non-human slots.

Outputs:

```text
results/audit/species_statistics.tsv
results/audit/panel_pairwise_identity.tsv
results/audit/sample_anchor_ids.txt
results/audit/summary.json
```

Run from this directory. Always dry-run first:

```bash
uv run snakemake --profile workflow/profiles/default -n \
  results/audit/summary.json

uv run snakemake --profile workflow/profiles/default \
  results/audit/summary.json
```

The intended execution environment is SkyPilot:

```bash
sky launch -c dna-exp402-audit sky/audit.yaml
sky logs dna-exp402-audit
sky down dna-exp402-audit
```

The Sky job performs the dry-run gate before the real invocation. Its outputs
are stored under `s3://oa-bolinas/snakemake/rag_glm/`.

## Training dataset

`rule build_training_dataset` pivots all frozen source anchors into eight fixed
slots, fills missing non-human slots with 255 Ns, excludes chromosome 18 from
training, chooses exactly 2,048 chromosome 18 validation anchors, and applies
one whole-document reverse-complement augmentation to training only. It emits
32 train Parquets, one validation Parquet, `manifest.json`, and the reviewed HF
dataset card under `results/dataset/zoonomia-rag-v1-v1/`.

The upload target is deliberately restricted to the two additive RAG rules:

```bash
COMMIT_SHA=$(git rev-parse HEAD)
sky launch -c dna-exp402-data sky/dataset.yaml \
  --env COMMIT_SHA="$COMMIT_SHA"
sky logs dna-exp402-data
sky down dna-exp402-data
```

The launch performs a dry-run first. It must show only
`build_training_dataset` and `upload_training_dataset`; any upstream projection
rule is an error and must stop the run.

## Mendelian RAG harness

The evaluation build preserves the pinned 255-base Mendelian harness splits
and derives non-human windows without new lift-over. Mendelian `pos` is
1-based at the source boundary; all coordinates introduced by this pipeline
are 0-based, half-open.

Variant-centered windows are not direct keys in the conserved projection. The
build chooses the containing 255-base conserved human anchor whose center is
closest to the SNV (ties by source start and anchor ID), propagates the
variant's anchor offset through each existing strand-aware projected interval,
and extracts a centered window from the archived species 2bit genome. A
variant/species without a valid containing projection receives `N × 255`.

The output has one row per variant per strand with all seven non-human
sequences, fixed-slot audit metadata, `context`, `ref_completion`, and
`alt_completion` already materialized. Scoring therefore requires only the
pinned HF dataset and a model checkpoint.

Build and upload on SkyPilot (the task dry-runs before execution and permits
only the six additive harness rules):

```bash
COMMIT_SHA=$(git rev-parse HEAD)
sky launch -c dna-exp402-mendelian sky/mendelian.yaml \
  --env COMMIT_SHA="$COMMIT_SHA"
sky logs dna-exp402-mendelian
sky down dna-exp402-mendelian
```

Outputs are stored under `results/mendelian/` in the existing RAG S3
namespace and uploaded to
`bolinas-dna/evals_mendelian_traits_rag_harness_255_v1`.
