---
tags:
- biology
- genomics
- dna
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/*.jsonl.zst
  - split: validation
    path: data/validation/*.jsonl.zst
---

# `marin-dna/vertebrate-<pipeline_version>-<region>`

Review status: **draft; do not upload until generated values are checked**.

Human-anchored 255 bp vertebrate sequences from the Zoonomia 447-mammal Cactus
alignment and UCSC hg38 MultiZ 100-way alignment for the `<region>` cohort.
Source FASTA/2bit letter case is preserved.

Non-human rows project only the central human nucleotide and extract the 255 bp target window centered on its unique mapped locus.

Anchor eligibility uses the pipeline's pinned phyloP conservation filter.
Sequence case is independent of that filter: lowercase bases preserve source
repeat masking, uppercase bases preserve source non-repeat-masked sequence, and
conservation scores never rewrite emitted characters or case.

Produced by the
[`vertebrate_projection_dataset` pipeline](https://github.com/Open-Athena/marin-dna/blob/<COMMIT_SHA>/snakemake/vertebrate_projection_dataset/README.md).
Replace `<COMMIT_SHA>` with the exact producing revision; never use a branch URL.

## Provenance

- Human reference: hg38, one row per retained human anchor.
- Mammals: `<ZOONOMIA_SPECIES_COUNT>` family-deduplicated targets from the
  Zoonomia 447-mammal Cactus HAL.
- Non-mammals: `<MULTIZ_SPECIES_COUNT>` family-deduplicated targets from the
  UCSC hg38 MultiZ 100-way MAFs.
- Species manifest revision: `<COMMIT_SHA>`.
- Dataset revision: `<HF_REVISION_AFTER_UPLOAD>`.

## Splits

- `train`: `<TRAIN_ROWS>` rows from non-chromosome-18 human source anchors;
  configured reverse-complement augmentation may be present.
- `validation`: `<VALIDATION_ROWS>` original-orientation rows sampled
  deterministically from chromosome-18 human source anchors, capped at 16,384.
- Validation tokens including one BOS per row: `<VALIDATION_TOKENS>`.
- Validation seed: `<VALIDATION_SEED>`.

Unsampled chromosome-18 rows and all chromosome-18 reverse complements are
discarded. Split membership depends only on the human source interval.

## Species counts

Replace this section with the generated backend/clade table and attach the
generated per-species validation counts.

## Schema

Replace this section with the generated schema. It includes stable row/anchor
identity, 0-based half-open human and target coordinates, region, taxonomy,
alignment backend, mapping provenance, sequence orientation, augmentation, and
the 255 bp source-case-preserving sequence.

## Pre-upload checklist

- [ ] Commit-pinned pipeline URL resolves to the producing code.
- [ ] Row, token, backend/clade, and per-species counts match generated files.
- [ ] Coordinate/split/case assertions and focused tests passed.
- [ ] QC breadth and rejection distributions were reviewed.
- [ ] ZRS recovered multiple non-mammal clades.
- [ ] Manual UCSC/raw-MAF and HAL spot checks were recorded.
