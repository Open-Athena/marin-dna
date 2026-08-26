# Linclust conservation analysis

This independent Snakemake project tests whether symmetric clustering of fixed mammalian genome windows recovers human phyloP conservation.
The coordinating design and evaluation contract are in [issue #521](https://github.com/Open-Athena/marin-dna/issues/521).

The project is transitioning from Phase 0 into bounded biological sensitivity checks.
Its default target runs the synthetic MMseqs2 release gate under three deterministic input orderings.
The explicit `smoke` target stages and samples the resolved panel on EC2.
The explicit `exhaustive` target can cluster every retained human, mouse, and opossum tile, but its first run was deliberately stopped after the Linclust candidate pass showed severe under-clustering.
The `tune_homology` target instead tests a bounded three-species positive-control fixture before any further whole-genome run; whole-panel clustering, scoring, and sealed even-autosome evaluation are not yet default targets.

All maintained behavior, tests, dependency state, workflow rules, profiles, and SkyPilot configuration are owned by this directory.
The workflow writes only under `s3://oa-bolinas/snakemake/analysis/linclust_conservation/`.
It does not modify or write through any existing training-dataset S3 path.
Result paths include the pipeline version, producing Git commit, and resolved-configuration SHA-256 so a later code or configuration change cannot silently reuse an earlier result.
Staged genome keys include the same run identity, so a canary, full smoke, or later code revision cannot overwrite an earlier run's input objects.

## Contracts

Internal genomic coordinates are 0-based and half-open.
The fixed window length is 255 bp and the stride is 128 bp, matching `vertebrate_projection_dataset` and producing 256 model tokens after BOS.
A window is rejected if any character is outside case-insensitive `A`, `C`, `G`, or `T`.
A window is rejected if more than 50% of its characters are lowercase.
Ambiguous bases are never deleted because deletion would join unrelated sequence across assembly gaps.
Every retained record ID encodes the versioned assembly accession, source sequence name, start, end, and strand.
Only NCBI sequence-report rows in a principal nuclear assembly unit with role `assembled-molecule`, `unlocalized-scaffold`, or `unplaced-scaffold` are eligible.
NCBI may label that unit `Primary Assembly` or with the reference strain name, such as `C57BL/6J`; the role filter plus explicit non-nuclear and mitochondrial exclusions define the invariant.
Alternate loci, fix patches, novel patches, and the separate non-nuclear mitochondrial unit are excluded from clustering and counted in per-assembly receipts.

The current assembly selector consumes NCBI Datasets genome and taxonomy JSONL reports.
It accepts only current, annotated RefSeq assemblies at `Complete Genome` or `Chromosome` level.
It keeps one assembly per NCBI order, forces the current human and mouse RefSeq assemblies for Primates and Rodentia, then ranks other candidates by assembly level, contig N50, total length, species, and accession.
Orders without an eligible assembly remain unselected and are explicitly marked as requiring a fallback decision.
The selected manifest must record the NCBI Datasets version, UTC retrieval time, source URI, source checksum, and eventual sequence SHA-256.

The query frozen on 2026-08-25 selected 20 orders from 268 current annotated RefSeq reference assemblies.
Macroscelidea, Scandentia, Sirenia, and Tubulidentata have only scaffold-level reference candidates and remain recorded as fallback decisions.
Seventeen selected accession versions existed in the prior 2bit mirror at panel resolution; `GCF_027887165.2`, `GCF_041296235.1`, and `GCF_054371585.1` initially required fresh NCBI downloads.
The completed three-genome canary's immutable opossum 2bit is now an ETag-guarded reuse source, so the exhaustive canary copies all three inputs into a new run-derived namespace.

Human tuning uses odd-numbered canonical autosomes.
The final held-out evaluation uses even-numbered canonical autosomes once, after the Linclust configuration, feature set, model, and footprint aggregation rule freeze.
X, Y, and mitochondrial sequence are excluded from the primary split.
The explicit RefSeq GRCh38.p14 assembled-molecule to UCSC hg38 chromosome dictionary must pass chromosome-length and sampled-sequence equality checks before any phyloP value is read.

The target is `hg38.phyloP447way.bw` at threshold `2.2162`.
Missing or unaligned values contribute zero to the numerator and the denominator remains 255.
Track access and assembly-mapping failures are errors.

Deployed score features are allowlisted Linclust membership and representative-member alignment statistics.
phyloP, repeat fraction, GC content, coordinates, annotations, and species-tree values cannot enter the score.

## Candidate MMseqs2 release gate

MMseqs2 `18.8cc5c` is a candidate release until the committed synthetic target passes.
The Conda environment pins the exact Bioconda build `18.8cc5c=hd6d6fdc_0`.

The fixture contains exact duplicates, an exact reverse complement, controlled substitutions, controlled indels, low-complexity sequence, 25% soft-masked sequence, and multiple equal candidates.
It runs Linclust under three deterministic hash orderings.
The gate requires exact forward and reverse-complement records to share one cluster and requires the complete cluster partition to remain unchanged across orderings.
Representative identity may change when the partition does not.

The workflow uses the MMseqs2 modules directly:

1. `createdb` builds the nucleotide database.
2. `linclust` produces cluster assignments under the recorded configuration.
3. `createtsv` exports representative-member membership.
4. `align --alignment-mode 3` realigns the forward cluster edges and self matches.
5. `createsubdb` materializes the cluster representatives.
6. `search --search-type 3 --strand 0` recovers reverse-complement representative-member alignments.
7. `convertalis` exports exact identity, alignment length, coverage, coordinates, E-value, and bit score, after which the workflow chooses the best orientation and validates exactly one alignment for every cluster edge.

The default candidate configuration is in `config/config.yaml`.
Passing the synthetic gate does not select the final biological configuration.

## Setup and validation

Run commands from this directory:

```bash
uv sync --locked --group dev
uv run --locked pytest
uv run --locked snakemake -n \
  --profile workflow/profiles/default \
  --default-storage-provider none
```

The `--default-storage-provider none` override is for a credential-free graph check.
Real executions retain the checked-in S3 profile.

The tiny synthetic fixture is permitted on the shared development node after the dry-run:

```bash
uv run --locked snakemake \
  --profile workflow/profiles/default \
  --default-storage-provider none \
  --cores 2
```

Do not run real genome windowing, whole-panel clustering, global sorting or grouping, or another data-scale target on the shared development node.

## Live manifest resolution

NCBI Datasets CLI `18.36.0` is pinned in `workflow/envs/ncbi_datasets.yaml`.
The committed manifest, not a live NCBI query, becomes the input to sequence processing.

The resolver expects:

- genome summary JSONL from an annotated, non-atypical, current RefSeq Mammalia query;
- taxonomy summary JSONL for every candidate taxonomic ID; and
- a source-inventory TSV with exact accession, URI, checksum type, and checksum.

`linclust-conservation-source-inventory` checks selected accessions against the existing training-dataset 2bit mirror with S3 `HeadObject` calls.
Only exact versioned-accession matches are accepted.
Missing accessions must be fetched from NCBI and added to the source inventory before the final manifest can be pinned.
The audit writes both the matching inventory and an explicit missing-accession report so available sources remain visible when the panel is only partially mirrored.

Resolve the live query and mirror audit without changing the default synthetic target:

```bash
uv run --locked snakemake \
  --profile workflow/profiles/default \
  --default-storage-provider none \
  results/manifest/missing_sources.tsv
```

The matched 2bit objects are copied into this workflow's new S3 namespace only after their ETags and sizes are recorded and rechecked.
The original objects remain unchanged.
Fresh NCBI archive downloads retry up to four times with bounded exponential backoff, remove any partial archive before retrying, and record the successful attempt number in the staging receipt.
Sequence reports must still classify each extracted sequence as an assembled molecule, unlocalized scaffold, or unplaced scaffold; alternate loci and patch sequences are excluded.
Every staged 2bit receives a full SHA-256 during smoke extraction, and freshly downloaded source FASTA files receive a separate SHA-256 in their staging receipts.

## SkyPilot

The approved contract worker is an `m6i.large` in `us-east-2` with an 80 GB root disk.
Launch it from a committed snapshot:

```bash
sky launch -c linclust-cons-contracts \
  snakemake/analysis/linclust_conservation/sky/contracts.yaml \
  --env PIPELINE_COMMIT_SHA="$(git rev-parse HEAD)"
```

Inspect the first run's environment setup, MMseqs2 version, rule plan, release-gate receipt, resource report, and S3 uploads.
Terminate the worker after the receipt is durable:

```bash
sky down linclust-cons-contracts
```

The approved smoke does not authorize an unbounded parameter sweep or full-panel production run.

Run the first end-to-end canary on human, mouse, and the freshly downloaded `GCF_027887165.2` assembly:

```bash
sky launch -c linclust-cons-canary3 \
  snakemake/analysis/linclust_conservation/sky/canary3.yaml \
  --env PIPELINE_COMMIT_SHA="$(git rev-parse HEAD)"
```

The canary exercises two ETag-guarded server-side copies, one current NCBI download and 2bit conversion, 6,000 tiled candidates, and the same strand-aware MMseqs2 receipt contracts as the panel smoke.
Its workflow outputs live under the separate `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/canary3/` prefix.

Launch the bounded real-data smoke from the committed branch snapshot:

```bash
sky launch -c linclust-cons-smoke \
  snakemake/analysis/linclust_conservation/sky/real_smoke.yaml \
  --env PIPELINE_COMMIT_SHA="$(git rev-parse HEAD)"
```

The target samples 2,000 tiled candidates per selected assembly, applies the production 255 bp sequence filters, clusters no more than 40,000 retained windows, and writes receipts under the new workflow-owned S3 prefix.
It copies the 17 exact mirror hits server-side and downloads and converts only the three missing current assemblies.
Its workflow outputs live under the separate `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/panel20-smoke/` prefix.
Terminate the worker after the receipt and finalized staged manifest are durable.

Run the exhaustive three-genome sensitivity canary on an `r7i.4xlarge` with 128 GB RAM and a 500 GB root disk:

```bash
sky launch -c linclust-cons-exhaustive3 \
  snakemake/analysis/linclust_conservation/sky/exhaustive_canary3.yaml \
  --env PIPELINE_COMMIT_SHA="$(git rev-parse HEAD)"
```

The stopped run enumerated all 73,545,023 candidate 255 bp tiles from human, mouse, and opossum and retained 47,767,099 after the ambiguity and majority-repeat filters.
Its first Linclust candidate pass produced 45,500,465 preliminary clusters, only a 4.75% reduction, before entering Linclust's mandatory internal alignment.
That result is sufficient to reject the current whole-genome recipe, so the run was cancelled and its worker terminated before a complete assignment table was produced.
The extracted FASTAs and filtering receipts remain durable under the separate `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/canary3-exhaustive/` prefix.

The next target uses real projected orthologs from the preserved issue #417 artifacts:

```bash
uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --configfile config/homology_tuning.yaml \
  --profile workflow/profiles/default \
  tune_homology
```

The fixture takes 512 human anchor IDs that have clean 255 bp rows in human, mouse, and armadillo, for 1,536 sequences total.
Its bounded diagnostic grid varies nucleotide k-mer scale, spaced k-mers, masking, coverage down to 0.70, and minimum identity down to 0.40.
Linclust's mandatory alignment and a direct alignment of only the retained cluster edges run on this small fixture; no genome-scale representative-to-all search is launched.
The receipt reports cluster count relative to the 512-anchor ideal, exact three-species anchor recovery, true-pair recall, and false cross-anchor merges.

The follow-up target holds that truth fixture fixed and replaces Linclust candidate generation with the more sensitive MMseqs2 clustering workflow:

```bash
uv run --locked snakemake \
  --snakefile workflow/Snakefile \
  --configfile config/homology_search_clustering.yaml \
  --profile workflow/profiles/default \
  search_cluster_homology
```

The four bounded variants compare cascaded and single-step clustering, greedy set cover and connected components, and one relaxed single-step threshold.
Every variant uses sensitivity 7.5, permits all 1,536 prefilter results per query, and aligns only the resulting representative-member edges for diagnostics.
The committed SkyPilot task uses a two-core, 16 GiB `r7i.large` because the nucleotide search prefilter exceeds the shared development node's local memory budget even for the small fixture.
Its outputs use the separate `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/homology-search-clustering/` prefix.

A 511 bp window with stride 128 remains a useful later geometry test because its worst grid-phase overlap is 447/511 (87.5%), above the current 80% coverage threshold.
The current projection artifacts contain only 255 bp sequences, so the 511 bp whole-genome run is deferred until the clustering recipe shows adequate recovery on the bounded truth set.

## Current outputs

The default target writes:

- `results/<pipeline-version>/<commit>/<config-sha256>/contracts/order_<seed>/controls.fasta`;
- `results/<pipeline-version>/<commit>/<config-sha256>/contracts/order_<seed>/clusters.tsv`;
- `results/<pipeline-version>/<commit>/<config-sha256>/contracts/order_<seed>/alignments.tsv`;
- `results/<pipeline-version>/<commit>/<config-sha256>/contracts/order_<seed>/mmseqs_version.txt`;
- `results/<pipeline-version>/<commit>/<config-sha256>/contracts/order_<seed>/resources.txt` with `/usr/bin/time -v` output;
- `results/<pipeline-version>/<commit>/<config-sha256>/contracts/order_<seed>/release_gate.json`; and
- `results/<pipeline-version>/<commit>/<config-sha256>/contracts/mmseqs2_release_gate.json`, the cross-ordering gate receipt.

The explicit `smoke` target additionally writes a fully pinned staged assembly manifest, per-assembly filtering and checksum receipts, Linclust membership and alignment tables, complete stage resource reports including peak temporary bytes, and a versioned `smoke/receipt.json`.

The explicit `exhaustive` target writes per-assembly exhaustive-filter receipts, a complete compressed Linclust assignment table, stage resource reports, and an `exhaustive/receipt.json` with singleton, cluster-size, and distinct-genome support summaries.

The research chronology and exact milestone commands belong in `.agents/logbooks/linclust-conservation.md` and issue #521.
