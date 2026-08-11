# `rag_glm`

Issue #402's isolated prototype pipeline for fixed-layout retrieval-augmented
genomic-language-model documents. It consumes the completed Zoonomia projection
as an immutable upstream artifact; it never runs `halLiftover` or overwrites
outputs from `snakemake/zoonomia_projection_dataset/`.

All genomic coordinates are 0-based, half-open. This directory is an independent locked Python project: pipeline code lives in `src/marin_dna_rag_glm/`, tests live in `tests/`, and `uv.lock` fixes its runtime separately from the repository root.

Set up the project before running any target:

```bash
uv sync --locked --group dev
uv run --locked pytest
uv run --locked snakemake -n
```

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
uv run --locked snakemake --profile workflow/profiles/default -n \
  results/audit/summary.json

uv run --locked snakemake --profile workflow/profiles/default \
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
and directly projects each exact unique variant-centered human window through
the pinned Zoonomia 447-mammalian 2022 v1 HAL. Mendelian `pos` is 1-based at
the source boundary; the BED and every other coordinate introduced here are
0-based, half-open.

The isolated projection uses `halLiftover --noDupes` and the canonical v1
quality contract: each query must resolve to one target chromosome and strand,
the merged pre-resize span must be in `[128, 512]`, and the target interval is
midpoint-resized and bounds-checked to exactly 255 bases. Sequences come from
the already-archived per-species 2bit genomes and are reverse-complemented for
negative target strands. A variant/species that does not survive these checks
receives `N × 255`. The training corpus and canonical Zoonomia projection
outputs remain immutable; all new outputs live under `results/mendelian/`.

The output has one row per variant per strand with all seven non-human
sequences, fixed-slot audit metadata, `context`, `ref_completion`, and
`alt_completion` already materialized. Scoring therefore requires only the
pinned HF dataset and a model checkpoint.

Build and upload on SkyPilot. The task stages the 1.18-TiB HAL onto instance
NVMe, dry-runs before execution, and permits only the five additive harness
rule types:

```bash
COMMIT_SHA=$(git rev-parse HEAD)
sky launch -c dna-exp402-mendelian-hal sky/mendelian.yaml \
  --env COMMIT_SHA="$COMMIT_SHA"
sky logs dna-exp402-mendelian-hal
sky down dna-exp402-mendelian-hal
```

The expected DAG is 17 jobs: one exact-window build, seven direct projections,
seven 2bit extractions, one harness materialization, and one Hugging Face
upload. Any training-data or canonical Zoonomia projection rule is an error.

Outputs are stored under `results/mendelian/` in the existing RAG S3
namespace and uploaded to
`marin-dna/evals_mendelian_traits_rag_harness_255_v1`.

## Complex-traits and SGE RAG harnesses

The same staged HAL can build retrieval-conditioned versions of the other two
variant leaderboards without repeating the 1.18-TiB transfer. The source
revisions are pinned independently:

| Benchmark | Source revision | Source variants (train/test) | HF output |
| --- | --- | ---: | --- |
| Complex traits | `22f86a89c65cb8f3007ac3cc2739f40efefa4340` | 11,630 / 10,000 | `marin-dna/evals_complex_traits_rag_harness_255_v1` |
| SGE | `225d3d1ea32a4af547891b13c33b5e92a5aae849` | 23,853 / 14,888 | `marin-dna/evals_sge_rag_harness_255_v1` |

Both builds preserve every original source column and add the same frozen RAG
fields as the Mendelian harness. Source `pos` is converted from 1-based to a
0-based centered interval only at the build boundary. The human REF allele is
asserted against the archived `Homo_sapiens` 2bit before any projection, and
the two source splits must be variant-disjoint. Whole-document reverse
complements, exact 2,048-token geometry, one row per variant per strand, and
full-slot missing values are asserted during assembly.

Dry-run the two cards (32 additive jobs) before execution:

```bash
uv run --locked snakemake --profile workflow/profiles/default \
  --config commit_sha="$COMMIT_SHA" hal_path="$HAL_PATH" \
  --allowed-rules build_benchmark_variant_windows \
    project_benchmark_variant_windows extract_benchmark_orthologs \
    build_benchmark_rag_harness \
  -n \
  results/complex_traits/dataset/evals_complex_traits_rag_harness_255_v1/README.md \
  results/sge/dataset/evals_sge_rag_harness_255_v1/README.md
```

Review each generated `README.md` and `manifest.json` before invoking its
`upload_benchmark_rag_harness` target. Intermediate projections and final
datasets live under `results/{complex_traits,sge}/` in the RAG S3 namespace;
the canonical leaderboard sources and Zoonomia projection artifacts remain
read-only.
