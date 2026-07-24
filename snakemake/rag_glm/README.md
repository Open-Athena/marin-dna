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
