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

Human-anchored 255 bp sequences projected through checksum-pinned chain files for the `<region>` cohort.
Source FASTA/2bit letter case is preserved.
Describe the recorded chain and genome origins from `metadata/assets.json`; historical cohort labels do not establish the provenance of supplied bytes.
If any input is synthetic, explicitly label the card as a fabricated software fixture, not a biological dataset.

Non-human rows project only the central human nucleotide and extract the 255 bp target window centered on its unique mapped locus.

State whether anchors were generated with the configured phyloP filter or read from a pinned external/smoke catalog without applying an additional eligibility filter.
Sequence case is independent of anchor selection and is copied verbatim from each archive.
For repeat-masked biological inputs, lowercase bases preserve source repeat masking; conservation scores never rewrite emitted characters or case.

Produced by the [`vertebrate_projection_dataset` pipeline](https://github.com/Open-Athena/marin-dna/blob/<COMMIT_SHA>/snakemake/vertebrate_projection_dataset/README.md).
Replace `<COMMIT_SHA>` with the exact producing revision; never use a branch URL.

## Provenance

- Human reference: hg38, one row per retained human anchor.
- Mammals: `<MAMMAL_SPECIES_COUNT>` targets; recorded chain and genome origins: `<MAMMAL_ORIGINS>`.
- Non-mammals: `<NON_MAMMAL_SPECIES_COUNT>` targets; recorded chain and genome origins: `<NON_MAMMAL_ORIGINS>`.
- Projector: UCSC liftOver for all non-human targets; chain and genome digests are recorded in `metadata/assets.json`.
- Species manifest revision: `<COMMIT_SHA>`.
- Dataset revision: `<HF_REVISION_AFTER_UPLOAD>`.

## Splits

- `train`: `<TRAIN_ROWS>` rows after removing the validation sample and applying the configured reverse-complement augmentation.
- `validation`: `<VALIDATION_ROWS>` original-orientation rows sampled uniformly without replacement with seed `<VALIDATION_SEED>` before augmentation.
- Validation tokens including one BOS per row: `<VALIDATION_TOKENS>`.
- Validation seed: `<VALIDATION_SEED>`.

The split is row-level and does not stratify by chromosome, species, or human anchor.
Different species projections from one human anchor may occur on opposite sides of the split.
The reverse complement of a selected validation row is excluded from training.

## Species counts

Replace this section with the generated backend/clade table.

## Schema

Replace this section with the generated schema.
It includes stable row/anchor identity, 0-based half-open human and target coordinates, region, taxonomy, alignment backend, mapping provenance, sequence orientation, augmentation, and the 255 bp source-case-preserving sequence.

## Pre-upload checklist

- [ ] Commit-pinned pipeline URL resolves to the producing code.
- [ ] Row, token, backend/clade, selection, and validation-composition counts match generated files.
- [ ] Coordinate/split/case assertions and focused tests passed.
- [ ] QC breadth and rejection distributions were reviewed.
- [ ] ZRS recovered multiple non-mammal clades.
- [ ] Sampled chain mappings and assembly-matched sequence windows were reviewed.
