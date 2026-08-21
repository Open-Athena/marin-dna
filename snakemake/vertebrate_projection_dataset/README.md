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
is `results/<pipeline_version>/<producer_commit>/<config_sha256>/<tier>/` relative to this project. The producer-keyed namespace prevents a clean worker from reusing outputs created by different code or resolved configuration.

Run commands from `snakemake/vertebrate_projection_dataset` so this project uses its own pinned environment and lockfile. Install it with:

```bash
uv sync --locked --group dev
```

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
uv run --locked marin-dna-build-vertebrate-species-manifest
uv run --locked marin-dna-build-multiz-mirror-manifest --output config/multiz_mirror.tsv
uv run --locked marin-dna-build-twobit-manifest --output config/twobit_manifest.tsv
```

The species script queries pinned NCBI taxonomy/assembly metadata, and the 2bit script queries UCSC's published checksum indexes plus its public S3 mirror metadata, so review either diff before accepting a regeneration. The committed files are the pipeline inputs; runtime species selection never relies on name matching. `config/twobit_manifest.tsv` pins the exact v1 human and 28 MultiZ-target archives by byte size and either UCSC-published MD5 or a locally reproducible S3 multipart ETag.

## MultiZ mirror and local staging

`config/multiz_mirror.tsv` pins the 24 primary-chromosome compressed MAFs, both
species-tree representations, source README/checksum metadata, byte sizes, MD5
checksums, source URLs, and the immutable S3 prefix
`s3://oa-bolinas/staging/multiz100way/hg38/ucsc-2015-05-12`.

Mirroring is an explicit, mutating bootstrap operation and is not in the normal
DAG. Run it only after reviewing the manifest and S3 destination:

```bash
uv run --locked snakemake \
  --profile workflow/profiles/default \
  mirror_multiz_bootstrap
```

Normal projection rules stage only configured chromosomes from S3 to local NVMe and verify size and MD5 before use. They fail on a missing/mismatched S3 object and never fall back to UCSC. The Zoonomia HAL follows the same S3-to-NVMe staging pattern. All HAL and MAF staging writes to a sibling partial file, validates it, and atomically installs it.

Target-genome sequence extraction retains the v1 UCSC human and `gbdb` 2bit sources. Each download must match `config/twobit_manifest.tsv`; before MultiZ extraction, every chromosome size used by an accepted MAF mapping must exactly match `twoBitInfo` for that archive. UCSC downloads are capped at four concurrent transfers and retry refused/transient connections so a full-worker startup cannot overload the shared endpoint.

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
accepted/rejected outputs are then streamed into per-species Parquets. Each HAL
species contract reads its fragment Parquet once, sorts fragments into target
coordinate order, and derives consistency, bounds, duplicate, and overlap
flags with vectorized group summaries. Rejection priority and resizing are also
vectorized, and an assertion requires exactly one accepted or rejected row per
input group. HAL contract rules reserve 10 GB, limiting full-worker concurrency to preserve memory headroom.

Human, HAL, and MultiZ sequence extraction all use one compiled
`twoBitToFa -bed` call per genome. BED6 strand is honored by `twoBitToFa`, and
soft-masked case from the 2bit source is preserved. Sequence rules reserve 4 GB. Sequence
combination, non-chromosome-18 training writes, QC aggregation, and inspection
candidate selection use lazy streaming scans.
Publication sharding retains the established deterministic eager shuffle for
inputs of at most 100 million post-augmentation rows. Larger inputs use a
deterministic hash sort and balanced partitioned NDJSON sink in Polars'
streaming engine, allowing the sort to spill rather than materializing the
entire publication cohort in RAM.
The conservation-filtered anchor BED is compressed through a temporary plain
file, fully decompressed to verify its row count, and atomically installed only
after the gzip stream passes its CRC check.

## Runbook

The checked-in profile caps local work at two cores and makes `s3://oa-bolinas/snakemake/vertebrate_projection_dataset/` the default storage prefix. Every output not explicitly marked `local()` is uploaded by Snakemake and can be restored automatically on another worker; `local()` is reserved for large or regenerable NVMe-only staging files and verification receipts. Durable paths include the producing commit and a SHA-256 of the fully resolved Snakemake config, and every namespace stores a matching `metadata/producer.json`. Always commit the complete recipe before execution and dry-run before a real run:

```bash
uv run --locked snakemake -n \
  --profile workflow/profiles/default
```

That dry-run consults the durable S3 state. For a credential-free graph check only, use `--default-storage-provider none`; CI uses that override, but real pipeline executions must retain the profile default.

The historical issue #417 staging snapshot remains a provenance artifact only; it is not an implicit pipeline input or fallback. A fresh v1 execution populates a producer-keyed namespace under the canonical storage prefix.

### Issue #473 center-seeded projection experiment

Issue #473 is implemented as the separate additive rule module
`workflow/rules/issue_473_fixed.smk`, loaded only through the standalone
`workflow/Issue473.smk` entrypoint; the shared `workflow/Snakefile`,
established projection rules, and established outputs remain unchanged. The committed
`config/issue_473_immutable_sources.tsv` pins every direct #417 and exp351
input by S3 URI, byte size, and full-object CRC64NVME checksum. Per-file #417
anchor and rejection restores must also appear in the pinned #417 artifact
inventory and receive a local checksum receipt before use.

The fixed catalog contains CDS, 3-prime UTR, noncoding-RNA exon, 5-prime
UTR/TSS, and the exact 116,162 exon-free exp351 enhancer-centered anchors. The
landmark pilot evaluates full-window and center-seeded widths 1, 17, 33, 65,
and 129 on the same deterministic anchor sample:

```bash
uv run --locked snakemake -n issue_473_fixed_landmark_pilot \
  --snakefile workflow/Issue473.smk \
  --profile workflow/profiles/default \
  --config tier=full
```

After pilot review, the complete approved projection target is:

```bash
uv run --locked snakemake -n issue_473_fixed_projection_experiment \
  --snakefile workflow/Issue473.smk \
  --profile workflow/profiles/default \
  --config tier=full
```

The complete target restores the immutable #417 standard-region full-window
sequences and QC instead of recomputing them. It adds only the missing
enhancer-centered full-window projection and the fixed-catalog center-width-1
projection, then produces paired diagnostics, anchor-level uncertainty,
manual-inspection samples, matched chromosome-18 validation views, and the
three newly required dataset artifacts. The existing immutable full-window CDS
dataset remains the fourth matched training input.
The default `smoke` tier uses two mammals, five non-mammals spanning birds,
reptiles, amphibians, ray-finned fish, and jawless vertebrates, chromosomes 7
and 18, two ZRS positive-control anchors, and small CDS/cCRE/background anchors.
After the dry-run shows only intended work, launch the smoke tier with `sky/project.yaml` as described below. Do not execute the projection on the shared development node.

Inspect the full graph with:

```bash
uv run --locked snakemake -n \
  --profile workflow/profiles/default \
  --config tier=full
```

Do not launch the full projection or any paid/cloud job without explicit user
approval. If the dry-run plans to recompute an upstream or unrelated artifact,
stop before running it.

### Issue #473 random-validation control

The CDS full-window random-validation control is a separate additive workflow.
It reads the immutable original-orientation accepted-sequence table from #417
and does not include or modify an established rule. It uniformly samples
16,384 CDS rows without replacement using seed 42, removes those rows from
training, and only then adds reverse complements to the remaining training
rows. Validation contains the sampled original-orientation rows only.

Inspect the complete public-dataset graph with:

```bash
uv run --locked snakemake -n \
  issue_473_random_validation_all_hf_files \
  --snakefile workflow/Issue473RandomValidation.smk \
  --profile workflow/profiles/default
```

The public repository is
[`marin-dna/vertebrate-v1-issue473-fullwindow-cds-random-val`](https://huggingface.co/datasets/marin-dna/vertebrate-v1-issue473-fullwindow-cds-random-val).
The dedicated Sky launcher builds and validates files before upload. Set
`TARGET=issue_473_random_validation_all_hf_files`, `DRY_RUN=0`, and
`ALLOW_HF_UPLOAD=0` for the non-uploading build. After reviewing its generated
card and manifest, set `TARGET=issue_473_random_validation_all_hf`,
`DRY_RUN=0`, and `ALLOW_HF_UPLOAD=1` to publish. The uploader explicitly sets
and verifies public repository visibility.

The workflow writes no projection QC or split-composition report. Its persisted
split summary records only the fixed seed and row counts needed to reproduce
the dataset.

### Shared local-node safety

Run data-scale validation, tracing, global sorts/group-bys, and working sets
larger than 500 MB on the SkyPilot worker, not on the shared development node.
Any potentially heavy command that must run locally must first acquire the
nonblocking `/tmp/marin-dna-local-heavy.lock`, require at least 6 GiB of
`MemAvailable` and a one-minute load below 2, and run with `nice -n 10` and
`ionice -c2 -n7`. Set `POLARS_MAX_THREADS=2` and `RAYON_NUM_THREADS=2`, and set
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`NUMEXPR_NUM_THREADS` to 1 before importing their runtimes. Abort a newly
started command if `MemAvailable` falls below 4 GiB or the one-minute load
exceeds 3 during its first minute.

### SkyPilot execution

The 1.26 TB HAL cannot run on a normal root volume. `sky/project.yaml` owns the
EC2 launch setup: `c6id.12xlarge` in `us-east-2`, both 1,425 GB instance-store
NVMes combined as RAID0, explicit free-space checks, Cactus binaries, and
symlinks that keep Snakemake state and the local working copies of generated results off the 100 GB root volume. Snakemake uploads every non-`local()` result to the profile's canonical S3 prefix as rules complete.

`halLiftover` is single-threaded. Its rule declares one thread and 2 GB of
memory so the 48-core worker can project species concurrently without reserving
four idle cores per process; the memory limit leaves headroom for full-tier execution.

The combined projection table is much larger than an individual species file.
QC, manual inspection, and each cohort writer therefore reserve the shared
`final_large_scan` resource. Its capacity is one in both the default profile
and the SkyPilot command, so these consumers run serially instead of
materializing several copies of the table and exhausting worker memory.

The 74.7 GB MultiZ source mirror is bootstrapped separately and resumably. Each
uploaded object records its pinned MD5 as S3 user metadata; reruns skip objects
only when both byte size and MD5 metadata match.

```bash
sky launch -c vertebrate-multiz-mirror \
  snakemake/vertebrate_projection_dataset/sky/mirror.yaml \
  --env PIPELINE_COMMIT_SHA="$(git rev-parse HEAD)"

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

HAL staging downloads to a temporary filename and atomically renames it only after the S3 object size matches and `halStats --genomes` succeeds. The staged HAL and other explicitly `local()` intermediates live only on instance-store NVMe and do not survive termination. Normal results use NVMe as their local working copy and are uploaded automatically to `s3://oa-bolinas/snakemake/vertebrate_projection_dataset/`; a later worker restores them through Snakemake rather than an ad hoc copy step. Do not terminate a worker while rules are still running or before any needed `local()` artifact has been consumed.

Every HAL projection/extraction path also depends on the tier-specific local `metadata/hal_stage_validated.txt` record. Because both the HAL and its receipt are `local()`, a clean worker cannot reuse a durable receipt independently of the NVMe file it certifies; a newly staged or changed HAL is revalidated before use.

## Splits and output datasets

Each configured region cohort first gets internal `train.parquet` and
`validation.parquet` files. Parquet is the efficient projection, split, and
card-count intermediate; it is not the published training format.

- training contains only non-chromosome-18 human source anchors and includes
  the configured reverse-complement augmentation;
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
At 16,384 rows, validation has exactly 4,194,304 tokens including BOS. These QC
sidecars remain in the pipeline results and are never copied into the Hugging
Face artifact directory.

The full tier additionally writes `datasets/cds_mammals_only/`, which applies
the CDS label after the complete shared projection/acceptance pass and then
retains only `human_reference` and `zoonomia_cactus` rows. The regular
`datasets/cds/` cohort retains those identical training rows plus `ucsc_multiz100way`. Projection, acceptance, augmentation, and sampling algorithms are identical, and the mammals-only training rows are an exact subset of the combined arm. Validation is sampled independently within each cohort because their eligible species sets differ, so the realized validation rows, and therefore validation-loss levels, must not be compared between arms.

For Hugging Face, `all_hf_files` follows the established Zoonomia publication path: deterministically shuffle each split, write 64 full-tier train shards (four in smoke) and one validation shard as JSONL, then zstd-compress them. On a dedicated HF worker, an explicitly supplied `PIPELINE_COMMIT_SHA` selects the matching producer/config namespace; the worker restores its producer manifest, source split Parquets, and active-species manifest through default Snakemake S3 storage. Local JSONL.zst artifacts are rebuilt on that worker and are never recovered through a separate issue-specific snapshot path.
Each isolated `hf/<cohort>/` directory contains only:

```text
README.md
data/train/shard_NNNN.jsonl.zst
data/validation/shard_0000.jsonl.zst
```

Generated cards contain the exact committed pipeline SHA, split row counts, schema, selected species counts, and explicit `data/<split>/*.jsonl.zst` loader paths. The validation manifest also records the resolved-config SHA-256 and refuses a producer manifest that does not match the restored namespace. `all_hf_files` then rejects any missing or unexpected file, validates zstd integrity and each shard’s boundary-record schema and split invariants, reconciles every shard row count to its source Parquet, and writes a content-hash manifest outside the upload tree. Build these review artifacts without external writes:

```bash
uv run --locked snakemake \
  --profile workflow/profiles/default \
  all_hf_files
```

After explicit human approval, `all_hf` serially uploads only those isolated artifact directories with the Xet client. Before each upload it rejects unexpected existing Hub paths; after upload it requires the exact remote tree, LFS sizes and SHA-256 hashes, and a byte-identical card at the resulting revision. It writes external Hugging Face state and is intentionally neither a default target nor part of `all_hf_files`. Upload completion markers are temporary and local, so a clean invocation always rechecks mutable Hub state instead of trusting a durable receipt. The large-folder client uses one worker per repository to avoid overwhelming the Hub LFS batch endpoint; interrupted uploads are resumable.
`config/HF_DATASET_CARD_TEMPLATE.md` is the pre-run review draft.

### Intentional differences from the Zoonomia-only publisher

- each repository has both `train` and `validation`, as required by #417;
- validation is the fixed chromosome-18, original-orientation sample rather
  than a second repository;
- rows carry the expanded human-source, taxonomy, backend, and mapping
  provenance schema; and
- reverse complements are materialized in the auditable internal train split
  before the established shuffle/shard step.

The published encoding and layout remain JSONL.zst; internal Parquet and every
QC TSV/JSON/Parquet stay off Hugging Face.

## QC and manual review

The default DAG writes:

- `qc/per_anchor.parquet`: mammal/non-mammal/total recovery, requested fraction,
  recovered clades, deepest clade, and no-mapping count per anchor;
- `qc/per_anchor_scope.parquet`: recovery by anchor, backend, and clade;
- `qc/rejection_counts.parquet`: explicit reasons including `no_mapping`;
- `qc/aggregates.parquet`: region/split/backend/clade counts, median, q10/q25/
  q75/q90, mean fraction, and fraction of anchors reaching the clade; and
- `qc/manual_inspection.md` plus accepted/rejected TSV samples.

The full-dataset inspection sample deterministically includes several CDS and
cCRE rows, accepted fragmented mappings when present, and explicit rejection
examples. ZRS is QC only: the two named loci are projected in a separate
sidecar check and are intentionally not appended to the conservation-filtered
training grid. The smoke/sidecar check fails unless each ZRS anchor recovers at
least two non-mammal clades. Both reports remain marked **pending human review**
until a reviewer completes the UCSC/raw MAF and HAL coordinate spot checks and
records any exclusions.

Compare aggregate CDS breadth with cCRE/enhancer breadth after the real build;
broader CDS recovery is a biological expectation, not a per-anchor invariant.

## Tests

Focused tests cover MAF gaps, fragmented blocks, reverse strands, coordinate
conversion, duplicate and ambiguous mappings, bounds, case-preserving sequence
orientation, manifest decisions, split allocation, mirroring checks, QC, and
inspection sampling:

```bash
uv run --locked --group dev pytest
```

The root repository tests are separate from this independently locked project. Experimental results and model comparisons are tracked in [issue 417](https://github.com/Open-Athena/marin-dna/issues/417) rather than this runbook.
