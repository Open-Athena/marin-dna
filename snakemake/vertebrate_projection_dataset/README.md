# Vertebrate projection dataset

This independent Snakemake pipeline builds 255 bp, hg38-human-anchored training
examples from three sources:

- the human hg38 reference sequence, once per anchor;
- one family-deduplicated mammal projection target from each of 107 families in
  the Zoonomia 447-mammal Cactus HAL; and
- one family-deduplicated non-mammal target from each of 28 vertebrate families
  represented in the UCSC hg38 MultiZ 100-way alignment.

It does not read from or write to
`snakemake/zoonomia_projection_dataset/results`. Its versioned output namespace
is `snakemake/vertebrate_projection_dataset/results/<pipeline_version>/<tier>/`.

All coordinates inside the library, manifests, intermediate Parquets, and
published datasets are **0-based, half-open**. MAF reverse-strand coordinates
are converted at the parser boundary. Sequence case is preserved from the
source genome; phyloP is used only to select human anchors and never changes
emitted sequence characters or case.

## Pinned inputs and cohorts

The full anchor tier independently reproduces the current anchor recipe from
the values in `config/config.yaml`: 255 bp windows on hg38 primary chromosomes,
128 bp step, phyloP-447m threshold 2.2162, and at least 20% conserved bases.
Anchors retain stable IDs and explicit human source coordinates. Region labels
use the pinned v4 labeling parameters in the same config.

`config/species_candidates.tsv` records all 107 non-human Zoonomia targets and
38 non-mammal MultiZ candidates. `config/species_selected.tsv` contains the 107
Zoonomia targets and the 28 deterministic one-per-family MultiZ selections.
Human is deliberately absent from both projection-target sets and is added as
`human_reference` exactly once per anchor. Both manifests record alignment and
scientific names, assembly, taxonomy ID, family, clade, phylogenetic rank,
backend, pinned assembly ranking fields, selection status, and reason. Loading
either manifest recomputes the selection decision and asserts selected-family,
taxonomy-ID, and assembly uniqueness.

The source manifests can be regenerated with:

```bash
uv run python scripts/issue417_build_species_manifest.py
uv run python scripts/issue417_build_multiz_mirror_manifest.py
```

The species script queries pinned NCBI taxonomy/assembly metadata, so review its
diff before accepting a regeneration. The committed files are the pipeline
inputs; runtime species selection never relies on name matching.

## MultiZ mirror and local staging

`config/multiz_mirror.tsv` pins the 24 primary-chromosome compressed MAFs, both
species-tree representations, source README/checksum metadata, byte sizes, MD5
checksums, source URLs, and the immutable S3 prefix
`s3://oa-bolinas/staging/multiz100way/hg38/ucsc-2015-05-12`.

Mirroring is an explicit, mutating bootstrap operation and is not in the normal
DAG. Run it only after reviewing the manifest and S3 destination:

```bash
uv run snakemake \
  --snakefile snakemake/vertebrate_projection_dataset/workflow/Snakefile \
  mirror_multiz_bootstrap
```

Normal projection rules stage only configured chromosomes from S3 to local
NVMe and verify size and MD5 before use. They fail on a missing/mismatched S3
object and never fall back to UCSC. The Zoonomia HAL follows the same S3-to-NVMe
staging pattern. Target-genome sequence extraction uses UCSC's `gbdb` 2bit
endpoint, which is available for all 28 selected current and legacy assemblies;
this does not substitute for or bypass the mirrored MAF projection input. UCSC
downloads are capped at four concurrent transfers and retry refused/transient
connections so a full-worker startup cannot overload the shared endpoint.

## Projection contract

The HAL and MAF adapters emit the same fragment schema. One shared library
implementation then:

1. groups fragments by `(query_name, species)`;
2. rejects inconsistent metadata, duplicated/overlapping mappings,
   multi-chromosome mappings, multi-strand mappings, invalid bounds, and spans
   outside the configured 128–512 bp pre-resize range;
3. midpoint-resizes the retained target span to exactly 255 bp within target
   chromosome bounds; and
4. extracts sequence from the target assembly, reverse-complementing
   case-preserving IUPAC DNA only when the target strand is negative.

Every rejection has one explicit machine-countable reason. Every accepted row
has at least the anchor ID and source interval, region label, species/assembly/
clade/backend provenance, target interval/strand/source size, fragment count,
aligned-base count, and 255 bp sequence.

The full tier is deliberately organized around bounded-memory intermediates.
Each chromosome MAF is parsed into species-clustered Parquet row groups, the
shared contract runs independently for each chromosome/species pair, and the
accepted/rejected outputs are then streamed into per-species Parquets. Sequence
combination, non-chromosome-18 training writes, QC aggregation, and inspection
candidate selection also use lazy streaming scans. Only one species' contract
rows, the human anchor catalog, the capped chromosome-18 validation candidates,
or the small deterministic inspection sample is materialized at a time.
The conservation-filtered anchor BED is compressed through a temporary plain
file, fully decompressed to verify its row count, and atomically installed only
after the gzip stream passes its CRC check.

## Runbook

The checked-in profile caps local work at four cores and carries pipeline-wide
defaults. Always dry-run before real execution:

```bash
uv run snakemake \
  --snakefile snakemake/vertebrate_projection_dataset/workflow/Snakefile \
  --dry-run
```

The default `smoke` tier uses two mammals, five non-mammals spanning birds,
reptiles, amphibians, ray-finned fish, and jawless vertebrates, chromosomes 7
and 18, two ZRS positive-control anchors, and small CDS/cCRE/background anchors.
After the dry-run shows only intended work, execute the smoke tier with the same
command without `--dry-run`.

Inspect the full graph with:

```bash
uv run snakemake \
  --snakefile snakemake/vertebrate_projection_dataset/workflow/Snakefile \
  --config tier=full \
  --dry-run
```

Do not launch the full projection or any paid/cloud job without explicit user
approval. If the dry-run plans to recompute an upstream or unrelated artifact,
stop before running it.

### SkyPilot execution

The 1.26 TB HAL cannot run on a normal root volume. `sky/project.yaml` owns the
EC2 launch setup: `c6id.12xlarge` in `us-east-2`, both 1,425 GB instance-store
NVMes combined as RAID0, explicit free-space checks, Cactus binaries, and
symlinks that keep Snakemake state and all generated results off the 100 GB root
volume.

The 74.7 GB MultiZ source mirror is bootstrapped separately and resumably. Each
uploaded object records its pinned MD5 as S3 user metadata; reruns skip objects
only when both byte size and MD5 metadata match.

```bash
sky launch -c vertebrate-multiz-mirror \
  snakemake/vertebrate_projection_dataset/sky/mirror.yaml

sky launch -c vertebrate-project \
  snakemake/vertebrate_projection_dataset/sky/project.yaml \
  --env TIER=smoke \
  --env PIPELINE_COMMIT_SHA="$(git rev-parse HEAD)"

sky exec vertebrate-project \
  snakemake/vertebrate_projection_dataset/sky/project.yaml \
  --env TIER=full --env TARGET=all --env DRY_RUN=1 \
  --env PIPELINE_COMMIT_SHA="$(git rev-parse HEAD)"
```

Actively inspect first-run setup, HAL transfer rate, mounted capacity, rule
progress, and ZRS/QC outputs. Reuse the same node for a later approved full run
with the same `sky exec` command and `--env DRY_RUN=0`; terminate it with
`sky down vertebrate-project` when inspection is complete.

HAL staging downloads to a temporary filename and atomically renames it only
after the S3 object size matches and `halStats --genomes` succeeds. The staged
HAL and generated results live on instance-store NVMe: they survive `sky exec`
jobs but not instance termination. Keep the cluster running until required
results have been copied to durable storage or uploaded through the reviewed
dataset targets.

Every HAL projection/extraction path also depends on the tier-specific
`metadata/hal_stage_validated.txt` record. This rechecks exact S3/local byte
size and HAL readability even when a complete staged HAL predates the current
workdir sync, so an existing NVMe copy cannot bypass validation.

## Splits and output datasets

Each configured region cohort gets one directory containing both
`train.parquet` and `validation.parquet`:

- training contains only non-chromosome-18 human source anchors and may include
  configured reverse-complement augmentation;
- validation candidates are original-orientation rows from chromosome-18 human
  source anchors;
- the deterministic SHA-256 sampler first represents every eligible species,
  then water-fills quotas as evenly as availability permits, up to 16,384 rows;
  and
- unsampled chromosome-18 rows and all chromosome-18 reverse complements are
  discarded, never returned to training.

Each cohort also writes `validation_selection.tsv`,
`validation_species_counts.tsv`, and `split_summary.json`, including the seed,
stable row IDs, per-species eligible/selected counts, and realized token count.
At 16,384 rows, validation has exactly 4,194,304 tokens including BOS.

The full tier additionally writes `datasets/cds_mammals_only/`, which applies
the CDS label after the complete shared projection/acceptance pass and then
retains only `human_reference` and `zoonomia_cactus` rows. The regular
`datasets/cds/` cohort retains those identical rows plus
`ucsc_multiz100way`. These are the two preregistered matched model arms; no
coordinate, acceptance, split, augmentation, or sampling rule differs between
them.

Generated dataset cards contain the exact committed pipeline SHA, split row
counts, schema, and selected species counts. Review them before the explicit
upload target:

```bash
uv run snakemake \
  --snakefile snakemake/vertebrate_projection_dataset/workflow/Snakefile \
  all_hf
```

`all_hf` writes external Hugging Face state and is intentionally not a default
target. `config/HF_DATASET_CARD_TEMPLATE.md` is the pre-run review draft.

## QC and manual review

The default DAG writes:

- `qc/per_anchor.parquet`: mammal/non-mammal/total recovery, requested fraction,
  recovered clades, deepest clade, and no-mapping count per anchor;
- `qc/per_anchor_scope.parquet`: recovery by anchor, backend, and clade;
- `qc/rejection_counts.parquet`: explicit reasons including `no_mapping`;
- `qc/aggregates.parquet`: region/split/backend/clade counts, median, q10/q25/
  q75/q90, mean fraction, and fraction of anchors reaching the clade; and
- `qc/manual_inspection.md` plus accepted/rejected TSV samples.

The inspection sample deterministically includes several CDS and cCRE rows,
accepted fragmented mappings when present, explicit rejection examples, and
human plus backend/clade representatives for both ZRS anchors. Automated checks
fail unless each ZRS anchor recovers at least two non-mammal clades. The report
remains marked **pending human review** until a reviewer completes its UCSC/raw
MAF and HAL coordinate spot checks and records any exclusions.

Compare aggregate CDS breadth with cCRE/enhancer breadth after the real build;
broader CDS recovery is a biological expectation, not a per-anchor invariant.

## Tests

Focused tests cover MAF gaps, fragmented blocks, reverse strands, coordinate
conversion, duplicate and ambiguous mappings, bounds, case-preserving sequence
orientation, manifest decisions, split allocation, mirroring checks, QC, and
inspection sampling:

```bash
uv run pytest tests/pipelines/vertebrate_projection_dataset
```

Run the full repository suite before committing:

```bash
uv run pytest
```

The matched CDS mammals-only versus combined-vertebrate model comparison is
preregistered in `reports/cds_model_sanity_experiment.md`. It must remain
unlaunched until the projection passes QC and the user explicitly approves paid
training.
