---
topic: linclust-conservation
issue: https://github.com/Open-Athena/marin-dna/issues/521
description: Recover mammalian conservation from symmetric Linclust statistics.
author: OpenAI Codex
---

# Linclust Conservation: Research Logbook

## Scope

- Goal: Test whether symmetric clustering of 255 bp windows across order-deduplicated mammalian RefSeq assemblies recovers held-out human phyloP conservation.
- Primary metrics: Spearman correlation with `phyloP_fraction` and conserved-base footprint curves on sealed even autosomes.
- Constraints: Use 0-based, half-open coordinates internally; exclude windows with ambiguous bases or more than 50% soft masking; use only Linclust-derived inference features; do not inspect even-autosome results before the method freezes.
- Coordinating issue: [#521](https://github.com/Open-Athena/marin-dna/issues/521)
- Experiment prefix: `LINC-CONS`
- Shared tags: `LINC-CONS`, `issue-521`, `linclust`

## Current TL;DR

The project is in Phase 0.
MMseqs2 18.8cc5c passes exact reverse-complement and input-order partition gates at 255 bp.
The frozen metadata query selects 20 eligible mammalian orders; 17 exact accession versions are reusable from the existing mirror and three require fresh NCBI downloads.
No real-data result exists.

## Baseline

- Date: 2026-08-25
- Code refs: [`38792946`](https://github.com/Open-Athena/marin-dna/commit/38792946).
- Baseline numbers: no Linclust conservation baseline exists yet.

## Hypothesis Queue

### Active

- `LINC-CONS-H1`: Distinct-genome support ranks human windows monotonically with `phyloP_fraction` after repeat filtering.
  Evidence: [issue design and prior work](https://github.com/Open-Athena/marin-dna/issues/521).
  Next test: pass the synthetic MMseqs2 release gate, resolve the current manifest, and run a chromosome-21 or balanced-sample smoke.
- `LINC-CONS-H3`: Existing training-dataset 2bit objects cover most or all of the newly selected accessions and can be checksum-verified before copying into the new workflow namespace.
  Evidence: the exact-version audit found 17 mirror hits among 20 selected accessions; see `LINC-CONS-002`.
  Next test: ETag-guarded server-side copies for the 17 hits and fresh NCBI staging for the other three.

### Blocked

None.

### Falsified / Dead End

None.

### Promoted

- `LINC-CONS-H2`: MMseqs2 18.8cc5c recovers exact reverse complements and produces an input-order-stable canonical partition under the Phase 0 nucleotide configuration.
  Decision: pin release 18.8cc5c for the bounded real-data smoke; see `LINC-CONS-002`.

## Decision Log

- 2026-08-25: Use a new project at `snakemake/analysis/linclust_conservation/` and a new S3 prefix at `s3://oa-bolinas/snakemake/analysis/linclust_conservation/`.
- 2026-08-25: Copy matching, checksum-verified existing genome objects into the new workflow namespace; leave every existing S3-backed rule and object untouched.
- 2026-08-25: Treat MMseqs2 18.8cc5c as a candidate release until it passes the synthetic strand and input-order gate.
- 2026-08-25: Paid SkyPilot EC2 execution is approved for the bounded smoke.
- 2026-08-25: Use 255 bp windows with a 128 bp stride to match `vertebrate_projection_dataset`; 255 sequence bases plus BOS produce 256 model tokens.

## Negative Results Index

None.

## Background Research Brief

- Effort: low follow-up to the medium forage already recorded in #521.
- Stop rule: stop when local code and current upstream release information no longer change the Phase 0 experiment order.
- Date: 2026-08-25

### Question

Which existing MarinDNA contracts and upstream MMseqs2 evidence should constrain the first implementation?

### Current Marin Context

The `genome_selection` workflow supplies the RefSeq, annotated, non-atypical query convention.
The `vertebrate_projection_dataset` workflow supplies deterministic contig-N50 ranking, explicit assembly validation, producer-keyed S3 layout, and SkyPilot runbook patterns.
Issue #120 reports that sensitive MMseqs2 nucleotide search recovered 70.6% of a conserved human-to-mouse cCRE population at 96.6% precision, but it does not test symmetric whole-panel clustering.
Issue #255 shows that order deduplication has region-dependent downstream effects, so order-level sampling is a cost and diversity choice rather than a guaranteed substitute for denser panels.

### External Prior Art

MMseqs2 release 18.8cc5c is the latest published upstream release as of this entry.
Upstream issue soedinglab/MMseqs2#858 remains open and contains exact reverse-complement failures under 15.6f452.
The upstream workaround discussion is not conclusive for every fixture, so reverse-complement recovery remains a release gate.

### Negative / Failed Leads

No existing MarinDNA pipeline clusters all retained windows from whole order-deduplicated mammalian assemblies and evaluates Linclust-only features against a sealed phyloP chromosome split.

### Recommended Next Experiments

#### 1. Synthetic release gate

- Minimum experiment: exact duplicates, exact reverse complements, controlled substitutions and indels, low complexity, soft masking, and three deterministic input orderings.
- Baseline/control: exact forward duplicates must cluster.
- Expected signal: exact reverse complements share the forward cluster and the partition is stable across orderings.
- Falsifier: strand separation or partition changes across orderings.
- Cost/risk: minutes on a two-vCPU worker; no biological claim.

#### 2. Current-manifest and source-reuse audit

- Minimum experiment: resolve one current annotated RefSeq assembly per mammalian order, force human and mouse, then match exact accession versions against the existing 2bit mirror.
- Baseline/control: current human and mouse RefSeq accessions and explicit S3 ETags.
- Expected signal: most selected accessions already exist and can be copied without redownloading.
- Falsifier: missing or stale accessions require fresh NCBI downloads.
- Cost/risk: metadata queries plus S3 HEAD requests.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
| --- | --- | --- | --- | --- | --- |
| Issue #521 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/521 | Experiment contract and prior-work synthesis | High | Coordinating record |
| Issue #120 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/120 | MMseqs2 nucleotide-search sensitivity baseline | High | Human-anchored pairwise task |
| Issue #255 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/255 | Order-deduplication motivation and caveat | High | Training result, not clustering |
| MMseqs2 release 18-8cc5c | External code | https://github.com/soedinglab/MMseqs2/releases/tag/18-8cc5c | Candidate release | High | Must pass local gate |
| MMseqs2 issue #858 | External issue | https://github.com/soedinglab/MMseqs2/issues/858 | Reverse-complement failure mode | High | Open upstream |

### Handoff

- Suggested issue update: post after the first reproducible contract snapshot, not for scaffolding alone.
- Open questions: exact current panel membership, mirror hit rate, strand-gate outcome, and first smoke resource estimate.
- Stop reason: the sources agree on the Phase 0 ordering and identify no additional prerequisite.

## Entry Log

### 2026-08-25 15:30 UTC - LINC-CONS-001 project prologue

- Hypothesis: A standalone workflow can enforce the data and MMseqs2 contracts before any mammalian-scale processing.
- Commit Hash: pending first snapshot.
- Command: implementation in progress; exact validation commands will be recorded after the first snapshot.
- Config: 255 bp windows, 128 bp stride, repeat fraction at most 0.5, phyloP threshold 2.2162, candidate MMseqs2 18.8cc5c, and three input orderings.
- Result: branch `codex/issue-521-linclust-conservation` and the independent project boundary were created.
- Interpretation: the first runnable target should be the synthetic release gate; live manifest resolution follows after the package and workflow dry-run pass.
- Next action: lock dependencies, run tests, dry-run the DAG, and execute the tiny synthetic fixture locally.

### 2026-08-25 20:16 UTC - LINC-CONS-002 Phase 0 gate and manifest audit

- Hypothesis: MMseqs2 18.8cc5c should recover exact reverse complements without changing the canonical partition across deterministic input orderings, and most current selected assemblies should have exact-version mirror matches.
- Commit Hash: [`38792946`](https://github.com/Open-Athena/marin-dna/commit/38792946)
- Commands: `uv run --locked pytest`; `uv run --locked snakemake -n --default-storage-provider none`; `uv run --locked snakemake --default-storage-provider none --forceall`; `uv run --locked snakemake --default-storage-provider none results/manifest/provisional_selected.tsv`; `uv run --locked snakemake --default-storage-provider none results/manifest/missing_sources.tsv`; and `uv run --locked snakemake --default-storage-provider none --conda-create-envs-only smoke`.
- Config: 255 bp windows, 128 bp stride, 2,000 smoke candidates per assembly, repeat fraction at most 0.5, MMseqs2 18.8cc5c at minimum identity 0.5 and coverage 0.8, and NCBI Datasets 18.36.0.
- Result: 25 unit tests passed.
  The 11-record synthetic fixture produced four clusters under each of three orderings.
  Exact forward and reverse-complement controls remained in one cluster; the representative changed by ordering but the canonical partition did not.
  The complete local target took 8.99 seconds and peaked at 136,596 KiB RSS, while the largest individual MMseqs stage peaked at 21,732 KiB.
  The query retrieved 268 current annotated RefSeq reference assemblies and selected 20 eligible orders.
  Macroscelidea, Scandentia, Sirenia, and Tubulidentata require a scaffold-level fallback decision.
  Exact 2bit accession matches exist for 17 selected assemblies, totaling 14.01 GiB; `GCF_027887165.2`, `GCF_041296235.1`, and `GCF_054371585.1` require fresh downloads.
- Interpretation: 18.8cc5c is accepted for the first real-data smoke.
  The source-reuse hypothesis is partially supported and reduces fresh downloads from 20 assemblies to three.
  This is a pipeline contract result, not evidence of biological conservation sensitivity.
- Next action: publish the snapshot and issue update, then launch the approved 40,000-candidate EC2 smoke through SkyPilot using only the new S3 namespace.
