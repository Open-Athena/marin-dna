# Functional-specialist anchor workflow

Issue [#517](https://github.com/Open-Athena/marin-dna/issues/517) is implemented as an additive workflow that reuses the production center-nucleotide projection contract without changing the existing v2 uniform-anchor workflow.

The authoritative entry point is `workflow/functional.Snakefile`, and its pinned configuration is `config/functional_anchors.yaml`.

## Annotation and coordinates

The annotation source is the complete Ensembl GRCh38 release 115 GTF at `Homo_sapiens.GRCh38.115.gtf.gz`.

All qualifying transcripts participate, including alternative transcripts.

The recipe does not use RefSeq and does not restrict the annotation to transcripts tagged `Ensembl_canonical`.

The only separate regulatory annotation is ENCODE SCREEN Registry V4, restricted to dELS and pELS cCREs for the enhancer arm.

GTF coordinates are converted from 1-based inclusive to 0-based half-open at the `load_annotation` boundary.

All later intervals, Parquets, audits, projection requests, and datasets remain 0-based and half-open.

## Anchor contract

The five arms are `cds`, `utr3`, `tss_region`, `ncrna`, and `enhancer`.

Protein-coding CDS is the highest-priority base class.

The 3′ UTR arm is derived transcript-by-transcript from protein-coding exons and CDS bounds, then excludes every base covered by any protein-coding CDS.

The TSS arm is the union of protein-coding TSS ±256 bp bands and protein-coding 5′ UTR sequence.

The ncRNA arm includes Ensembl biotypes `lncRNA`, `miRNA`, `snoRNA`, `snRNA`, `ribozyme`, `scaRNA`, and `vault_RNA`, with pseudo, pseudogenic, and partial annotations excluded.

The enhancer arm centers one 255 bp window on each dELS or pELS cCRE, retains overlapping cCRE windows, and rejects any window that overlaps an annotated exon.

Priority ownership is `cds > utr3 > tss_region > ncrna > enhancer`.

Every candidate records raw and priority-owned coverage for all five arms, its winning arm, functional-union coverage, exon coverage, and source-feature contributor count.

Exact duplicate windows are collapsed while their source contributors remain in the provenance table.

The pinned phyloP-447m base threshold is 2.2162.

Anchors with at least 10% conserved bases enter the projection catalog, anchors with at least 20% enter the training catalog, and anchors in `[10%, 20%)` remain in the deferred catalog for possible relabeling after projection.

## Safe execution sequence

Install the independently locked project environment from this directory:

```bash
uv sync --locked --group dev
```

Run the unit tests and credential-free default DAG check:

```bash
uv run --locked pytest
uv run --locked snakemake -n \
  --snakefile workflow/functional.Snakefile \
  --default-storage-provider none \
  --cores 2
```

The default `all` target stops after candidate construction, conservation gating, and the pending preprojection review report.

It does not stage the HAL or MAF inputs and does not run cross-species projection.

Before approving projection, review `anchors/audit/preprojection_sample.tsv`, `preprojection_review.md`, `feature_summary.tsv`, `raw_overlap.tsv`, `human_sequence_summary.tsv`, `chromosome_summary.tsv`, `construction_drop_summary.tsv`, `ownership_drop_summary.tsv`, `development_overlap.tsv`, `construction_drops.parquet`, and `window_ownership.parquet` in the producer-keyed result namespace.

The development-overlap audit pins `marin-dna/evals_mendelian_traits` revision `4aed58e50c5dea0b878a665007af2ef9e5108e9f` and its `train` split.
It rejects every chromosome outside odd autosomes and X, removes complete mature-miRNA match groups, and converts the benchmark's 1-based variant positions to 0-based half-open points at the audit boundary.
It never reads the held-out split.

Materialize the complete full-tier audit on the isolated issue #517 worker:

```bash
sky launch -c issue-517-functional-project \
  sky/functional_project.yaml \
  --env TIER=full \
  --env TARGET=all \
  --env PIPELINE_COMMIT_SHA="$(git rev-parse HEAD)"
```

After the audit passes, reuse the same staged worker for the real cross-backend smoke:

```bash
sky exec issue-517-functional-project \
  sky/functional_project.yaml \
  --env TIER=smoke \
  --env TARGET=all_projection \
  --env PIPELINE_COMMIT_SHA="$(git rev-parse HEAD)"
```

The smoke stages and validates the HAL on local NVMe.
Keep the worker only through the approved full projection so the 1.26 TB HAL is not downloaded twice.

Validate the paid projection graph without executing it:

```bash
uv run --locked snakemake -n \
  --snakefile workflow/functional.Snakefile \
  --default-storage-provider none \
  --cores 2 \
  all_projection
```

Do not execute `all_projection` locally or on remote compute until a human explicitly approves the spend and records the decision in issue #517.

After an approved projection, `all_projection` writes the shared recovery QC, five functional training splits, human-sequence GC/repeat/ambiguity audits, a pending mapping-inspection report, and draft cards for `marin-dna/functional-{arm}`.

Run the approved full projection on the retained worker:

```bash
sky exec issue-517-functional-project \
  sky/functional_project.yaml \
  --env TIER=full \
  --env TARGET=all_projection \
  --env PIPELINE_COMMIT_SHA="$(git rev-parse HEAD)"
```

Terminate the worker with `sky down issue-517-functional-project` only after the required durable S3 artifacts have been restored and verified.

Build publication artifacts on the retained remote worker only after the projection review is accepted:

```bash
sky exec issue-517-functional-project \
  sky/functional_project.yaml \
  --env TIER=full \
  --env TARGET=all_functional_hf_files \
  --env PIPELINE_COMMIT_SHA="<producer-commit>"
```

`all_functional_hf_files` restores the producer-pinned splits from S3, builds the exact JSONL.zst release trees on remote NVMe, and writes the durable content-hash manifest without writing to Hugging Face.

The `all_functional_hf` target uploads five repositories and is an external write that requires separate explicit publication approval.

After that approval is recorded, make an authorized `marin-dna` credential available on the worker and run:

```bash
sky exec issue-517-functional-project \
  sky/functional_project.yaml \
  --env TIER=full \
  --env TARGET=all_functional_hf \
  --env PIPELINE_COMMIT_SHA="<producer-commit>"
```

The upload target revalidates every local file and the mutable Hub state, requires public ungated repositories, verifies the resulting revision without credentials, and writes a temporary per-arm receipt containing the immutable revision.

Record those revisions in issue #517 and pin them in every training consumer.

Training must consume these public immutable Hugging Face revisions rather than the internal S3 producer paths.

Training and held-out VEP evaluation remain separate approval gates and are not targets of this workflow.
