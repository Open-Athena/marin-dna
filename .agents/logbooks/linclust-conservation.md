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

Phase 0 and the three-genome real-data canary are complete.
MMseqs2 18.8cc5c passes exact reverse-complement and input-order partition gates at 255 bp.
The frozen metadata query selects 20 eligible mammalian orders; 17 exact accession versions are reusable from the existing mirror and three require fresh NCBI downloads.
The human, mouse, and opossum canary retained 3,900 of 6,000 sampled windows and produced 3,900 singleton clusters, so it validates the workflow but does not yet recover a cross-genome conservation signal.

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

- `LINC-CONS-006`: The first three-genome canary produced no non-singleton clusters among 3,900 retained windows at the frozen threshold.
- `LINC-CONS-006`: Sky jobs 1 through 11 exposed checkout, installer, version-parsing, transient-download, and NCBI assembly-unit-schema failures before the successful immutable run.

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

### 2026-08-25 20:53 UTC - LINC-CONS-003 independent review remediation

- Hypothesis: The bounded real-data smoke can preserve the Phase 0 contracts under Snakemake remote storage while retaining one exact, strand-aware alignment for every Linclust assignment.
- Commit Hash: [`38526851`](https://github.com/Open-Athena/marin-dna/commit/38526851)
- Commands: `uv run --locked pytest`; `uv run --locked snakemake -n --profile workflow/profiles/default --default-storage-provider none --forceall`; a 47-job smoke dry-run with the filesystem storage provider; `uv run --locked snakemake --profile workflow/profiles/default --default-storage-provider none --cores 2 --forceall`; and `uvx --from uv==0.11.31 uv run --locked pre-commit run --all-files --show-diff-on-failure`.
- Config: 255 bp windows, 128 bp stride, 2,000 candidates for each of 20 assemblies, MMseqs2 18.8cc5c, and an 11-mer strand-aware search limited to 4 GiB of index memory.
- Result: An independent review found seven correctness or scaling problems involving remote-path handling, reverse-complement alignment export, quadratic validation and footprint code, staged-object revalidation, and S3 authorization failures.
  All seven findings were addressed.
  Thirty-one unit tests, repository-wide pre-commit, the credential-free forced dry-run, and the 47-job remote-storage dry-run passed.
  The forced synthetic execution again produced four stable clusters under all three orderings, with exactly 11 alignment rows for 11 assignment rows and an explicitly verified reverse-complement alignment in every run.
  The execution took 9.49 seconds and peaked at 141,160 KiB RSS.
  MMseqs2's default 15-mer nucleotide index required roughly 6 GiB even for the tiny fixture, while the pinned 11-mer search used roughly 32 MiB and passed the strand gate.
- Interpretation: The workflow is ready for the bounded EC2 smoke.
  This remains a software and data-contract result, not evidence for or against biological conservation sensitivity.
- Next action: push the reviewed snapshot, run the approved EC2 smoke, validate its durable S3 receipts, and terminate the worker.

### 2026-08-25 21:20 UTC - LINC-CONS-004 immutable canary preparation

- Hypothesis: A three-genome canary can exercise both verified S3 reuse and fresh NCBI staging while binding every result and release-gate receipt to an exact code and configuration identity.
- Commit Hash: [`072e3f56`](https://github.com/Open-Athena/marin-dna/commit/072e3f56)
- Commands: `uv run --locked pytest`; `uv run --locked snakemake -n smoke --profile workflow/profiles/default --default-storage-provider none --config selection_path=config/assembly_canary3.tsv --forceall`; `uvx --from uv==0.11.31 uv run --locked pre-commit run --all-files --show-diff-on-failure`; and a disposable exact-commit fetch, mixed reset, diff, and untracked-file check matching the Sky setup.
- Config: human `GCF_000001405.40`, mouse `GCF_000001635.27`, and fresh-download opossum `GCF_027887165.2`; 255 bp windows; 2,000 candidates per assembly; MMseqs2 18.8cc5c; and `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/canary3/` as the run prefix.
- Result: A second independent review found five provenance and gate-completeness problems.
  Result paths now contain the pipeline version, full producing commit, and resolved-configuration SHA-256.
  The smoke receipt validates the complete three-ordering release gate, exact MMseqs2 version and configuration, all 11 synthetic controls, reverse-strand recovery, and exact FASTA-to-assignment membership.
  Sky now rejects a worktree that differs from the requested commit.
  All 36 tests and repository-wide pre-commit passed, and the forced canary dry-run produced the expected 23-job DAG with two ETag-guarded copies and one NCBI download.
- Interpretation: The canary is ready to publish and launch on the approved EC2 worker.
  This remains a software and data-contract test and does not yet measure biological conservation sensitivity.
- Next action: push the commit and logbook entry, run a clean independent review over the published diff, then launch and validate the three-genome EC2 canary.

### 2026-08-25 21:36 UTC - LINC-CONS-005 final review integrity fixes

- Hypothesis: The canary can preserve immutable input identity and exact alignment-table cardinality across independently versioned runs.
- Commit Hash: [`d1125519`](https://github.com/Open-Athena/marin-dna/commit/d1125519)
- Commands: `uv run --locked pytest`; `uv run --locked snakemake -n smoke --profile workflow/profiles/default --default-storage-provider none --config selection_path=config/assembly_canary3.tsv --forceall`; and `uvx --from uv==0.11.31 uv run --locked pre-commit run --all-files --show-diff-on-failure`.
- Config: the `LINC-CONS-004` three-genome canary configuration, with staged 2bit keys additionally namespaced by pipeline version, producing commit, and resolved-configuration SHA-256.
- Result: The final independent review found two remaining integrity gaps: staging keys shared across run identities and validation that accepted unique alignment pairs absent from the assignment table.
  Both findings were addressed.
  The exact alignment pair set and row count are now required, all 37 tests and repository-wide pre-commit pass, and the forced canary dry-run remains a 23-job DAG with three run-isolated staged genome keys.
- Interpretation: The reviewed canary is ready for the approved EC2 execution after the remediation snapshot is published and independently rechecked.
  This remains a software and data-contract result and does not yet measure biological conservation sensitivity.
- Next action: publish and re-review the remediation snapshot, launch the three-genome canary, validate the durable S3 receipts, and terminate the worker.

### 2026-08-25 22:09 UTC - LINC-CONS-006 three-genome EC2 canary

- Hypothesis: A bounded human, mouse, and opossum canary can exercise two verified S3 copies and one fresh NCBI download while producing a complete immutable Linclust receipt.
- Commit Hash: [`633235e4`](https://github.com/Open-Athena/marin-dna/commit/633235e43c64c1ad10a7507b44bf5403296627d1)
- Commands: SkyPilot job 12 ran `uv run --locked pytest` and the 23-job `smoke` target on an AWS `m6i.large` worker in `us-east-2`; local post-run checks used `jq` for receipt and staging invariants plus an `awk` exact-set comparison of cluster assignments and alignments; `sky down -y linclust-cons-canary3` terminated the worker.
- Config: human `GCF_000001405.40`, mouse `GCF_000001635.27`, opossum `GCF_027887165.2`, 2,000 candidate 255 bp windows per assembly, 128 bp stride, repeat fraction at most 0.5, and MMseqs2 18.8cc5c.
- Result: All 39 tests and all 23 Snakemake jobs passed at the exact producing commit and configuration SHA-256 `e148100086879399eefbaf9f9c1a066911b9e2763feb49be2b2903bf36538f19`.
  The workflow copied the exact human and mouse 2bit objects into run-isolated keys and freshly downloaded the opossum assembly from NCBI.
  It retained 1,308 human, 1,357 mouse, and 1,235 opossum windows from 6,000 candidates, for 3,900 windows and 994,500 retained bases in total.
  Linclust produced 3,900 singleton clusters and 3,900 alignment edges.
  The independently downloaded tables contained exactly the same 3,900 unique representative/member pairs with no missing, extra, or duplicate edge.
  MMseqs2 used 3.32 CPU seconds, 1.81 wall seconds, 51,612 KiB peak RSS, and 12,713,223 peak temporary bytes for the retained-window input.
  The immutable receipt is `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/canary3/results/v1/633235e43c64c1ad10a7507b44bf5403296627d1/e148100086879399eefbaf9f9c1a066911b9e2763feb49be2b2903bf36538f19/smoke/receipt.json`.
  SkyPilot confirmed that `linclust-cons-canary3` no longer exists after teardown.
- Negative results: Jobs 1 through 9 failed before workflow execution because the rsynced worktree Git pointer was not portable, manually expanded commit identifiers were wrong, a tracked symlink was absent from the mixed checkout, the moving Miniforge installer returned HTTP 500, or `uv --version` included a platform suffix.
  Job 10 reached real data but the fresh NCBI download ended on a transient HTTP/2 error, which led to bounded retries.
  Job 11 staged all genomes but rejected mouse because NCBI labels its principal assembly unit `C57BL/6J`, which led to accepting strain-labeled principal units while retaining role and mitochondrial exclusions.
- Interpretation: The canary validates immutable staging, fresh-download retries, NCBI sequence selection, 255 bp sampling, Linclust execution, exact strand-aware alignment export, receipt production, and cleanup on real genomes.
  It provides no cross-genome conservation evidence because all retained windows were singletons at the frozen threshold in this sparse random sample.
  That negative signal is preliminary and does not by itself falsify `LINC-CONS-H1`.
- Next action: publish and independently review the exact canary snapshot, then decide whether the next bounded experiment should increase panel density, sample density, or sensitivity before the full 20-order evaluation.

### 2026-08-25 22:43 UTC - LINC-CONS-007 20-genome calibration smoke

- Hypothesis: Sampling 2,000 candidates from every selected assembly can validate the complete 20-order staging and execution graph and provide a small resource calibration.
- Commit Hash: [`40651240`](https://github.com/Open-Athena/marin-dna/commit/40651240f32a31c146daa2430276445a4d3dab92)
- Commands: SkyPilot job 1 ran the 57-job `smoke` target on an AWS `m6i.large` worker in `us-east-2`; local post-run checks validated receipt provenance, 20 unique accessions and orders, staging source counts, exact assignment/alignment pairs, and cross-genome membership; `sky down -y linclust-cons-smoke` terminated the worker.
- Config: 20 assemblies selected by the fixed one-best-eligible-assembly-per-order rule, rather than by a target count or random draw; 2,000 candidate 255 bp windows per assembly; 128 bp stride; repeat fraction at most 0.5; and MMseqs2 18.8cc5c.
- Result: All 57 jobs passed at configuration SHA-256 `7814a4787d9436bbdd4ad147642a4f611c04ee79eb3ce71c801be62653ef4f29`.
  The workflow copied 17 exact mirror objects and downloaded three assemblies from NCBI.
  It retained 25,401 of 40,000 candidates, or 63.50%, and produced 25,401 singleton clusters with zero cross-genome clusters.
  MMseqs2 used 17.49 wall seconds, 114,652 KiB peak RSS, and 88,354,410 peak temporary bytes.
  The immutable receipt is `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/panel20-smoke/results/v1/40651240f32a31c146daa2430276445a4d3dab92/7814a4787d9436bbdd4ad147642a4f611c04ee79eb3ce71c801be62653ef4f29/smoke/receipt.json`.
  SkyPilot confirmed that `linclust-cons-smoke` no longer exists after teardown.
- Interpretation: The run validates the full-panel contracts and provides a filter-rate and small-input resource calibration.
  It is not a biological sensitivity test because independently choosing 2,000 of roughly 20 million to 28 million possible tiles per assembly almost never selects homologous loci in two genomes.
  The user therefore redirected the next experiment to exhaustive tiling of three genomes.
- Next action: run all retained human, mouse, and opossum tiles through one Linclust database before expanding the panel.

### 2026-08-25 22:49 UTC - LINC-CONS-008 exhaustive three-genome preparation

- Hypothesis: Exhaustive human, mouse, and opossum tiling will place homologous loci in the same database and can reveal whether the frozen Linclust configuration has any cross-genome sensitivity.
- Commit Hash: [`8468995a`](https://github.com/Open-Athena/marin-dna/commit/8468995ad63e46b7ebd1d4ba51d3be4dc66a0f4f)
- Commands: `uv run --locked pytest`; credential-free default and 21-job exhaustive Snakemake dry-runs; an exact Conda solve for MMseqs2 and Zstandard; `uv run --locked snakemake --profile workflow/profiles/default --default-storage-provider none --cores 2 --forceall`; and changed-file pre-commit checks.
- Config: human `GCF_000001405.40`, mouse `GCF_000001635.27`, and opossum `GCF_027887165.2`; every 255 bp tile at stride 128; 250,000-candidate bounded extraction batches; MMseqs2 18.8cc5c; an AWS `r7i.4xlarge` with 128 GB RAM; a 500 GB root disk; and `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/canary3-exhaustive/` as the new run prefix.
- Result: The sequence report contains exactly 24,214,098 human, 21,314,017 mouse, and 28,016,908 opossum candidates, totaling 73,545,023 windows and 18,753,980,865 input bases before filtering.
  Applying the panel smoke's observed retention rate predicts roughly 46.7 million retained windows, but the run will record exact per-assembly counts.
  The extractor now streams the exhaustive interval grid without materializing it and limits each temporary BED and raw FASTA batch.
  The clustering target avoids the representative-to-all reverse-strand search, compresses the complete `createtsv` assignment table with Zstandard, and streams the receipt calculation with bounded memory.
  The opossum 2bit from `LINC-CONS-006` is now an ETag-guarded reuse source, so all three inputs will be copied into new run-derived staging keys.
  All 41 tests, pre-commit hooks, both dry-runs, the exact environment solve, and the 11-job synthetic execution passed.
- Interpretation: Three exhaustive genomes are a substantially better bounded sensitivity test than sparse samples from 20 genomes.
  The clustering-only receipt is intentional: if cross-genome clusters exist, exact representative-member alignments can be designed around the selected edges without launching an all-representative-to-all-window search dominated by singleton representatives.
- Next action: publish this preparation snapshot, launch the approved exhaustive EC2 run, monitor extraction and Linclust resource use, validate the durable receipt and compressed assignments, and terminate the worker.

## Background Research Brief - Exhaustive sensitivity pivot

- Effort: low targeted follow-up.
- Stop rule: stop when the exact three-genome tile count, observed retention rate, upstream Linclust scaling guidance, and a bounded execution design are sufficient to choose one EC2 configuration.
- Date: 2026-08-25

### Question

Should the next sensitivity experiment use sparse windows from 20 genomes or every window from three genomes?

### Current Marin Context

The three-genome and 20-genome sparse canaries produced only singleton clusters.
With tens of millions of possible tiles per assembly, independent samples of 2,000 windows have negligible coordinate overlap and cannot reliably expose homologous loci.
Issue #120 demonstrates that sensitive human-to-mouse MMseqs2 nucleotide search can recover conserved sequence, but that human-anchored result does not establish Linclust sensitivity.

### External Prior Art

The [Linclust paper](https://www.nature.com/articles/s41467-018-04964-5) motivates approximately linear scaling in sequence count through a bounded number of selected k-mers per sequence.
The [MMseqs2 repository](https://github.com/soedinglab/MMseqs2) describes Linclust as the fast, less-sensitive clustering workflow and documents a nucleotide k-mer scaling option that is more sensitive than the current fixed 20-k-mer baseline.
The [MMseqs2 user guide](https://github.com/soedinglab/MMseqs2/wiki) gives a minimum clustering memory heuristic near one byte per residue before workflow overhead.

### Negative / Failed Leads

Adding more independently sparse genomes does not fix the missing-locus-pair problem.
Extrapolating the 25,401-window peak RSS directly would understate the large fixed and per-residue costs of sorting Linclust seeds.
Running the existing representative-to-all strand search after a mostly singleton Linclust result would approach an unintended all-vs-all workload and is not a suitable first exhaustive step.

### Ranked Recommended Experiments

#### 1. Exhaustive three-genome clustering sensitivity

- Minimum experiment: cluster every retained human, mouse, and opossum tile once under the frozen baseline.
- Baseline/control: the existing synthetic strand gate and raw singleton rate.
- Expected signal: nonzero clusters supported by two or three distinct source genomes.
- Falsifier: no or negligible cross-genome support despite exhaustive locus availability.
- Cost/risk: tens of millions of records and substantial scratch, bounded to one 128 GB EC2 worker and one configuration.

#### 2. Bounded Linclust sensitivity adjustment

- Minimum experiment: if the exhaustive baseline remains insensitive, repeat a deliberately small real-data fixture with increased k-mer sampling and spaced k-mers before another exhaustive run.
- Baseline/control: the frozen 20-k-mer, contiguous-k-mer configuration.
- Expected signal: improved cross-genome support without an unacceptable resource multiple.
- Falsifier: support remains absent or resource growth is incompatible with panel scale.
- Cost/risk: one small fixture first; no blind full-scale parameter sweep.

### Hypothesis Update

`LINC-CONS-H1` remains open.
The sparse singleton results update confidence in the sampling design, not in the biological hypothesis, because corresponding loci were usually absent from the same input database.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
| --- | --- | --- | --- | --- | --- |
| Issue #521 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/521 | Experiment contract and bounded sensitivity decision | High | Coordinating record |
| Issue #120 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/120 | Sensitive nucleotide-search precedent | High | Human-anchored, not symmetric Linclust |
| Linclust paper | External paper | https://www.nature.com/articles/s41467-018-04964-5 | Linear-scaling mechanism | High | Protein-heavy benchmark |
| MMseqs2 repository | External code | https://github.com/soedinglab/MMseqs2 | Current nucleotide Linclust capabilities | High | Primary upstream source |
| MMseqs2 user guide | External documentation | https://github.com/soedinglab/MMseqs2/wiki | Memory heuristic and workflow guidance | Medium | Versioned behavior still measured locally |
| Panel smoke receipt | S3 artifact | `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/panel20-smoke/results/v1/40651240f32a31c146daa2430276445a4d3dab92/7814a4787d9436bbdd4ad147642a4f611c04ee79eb3ce71c801be62653ef4f29/smoke/receipt.json` | Retention and small-input resource calibration | High | Immutable producing identity |

### Handoff

- Suggested issue update: record the panel smoke as a calibration result and the exhaustive three-genome run as the next sensitivity experiment.
- Open questions: exact retained-window count, Linclust peak RSS and scratch, cross-genome cluster count, and whether the frozen 20-k-mer baseline has sufficient sensitivity.
- Stop reason: the available evidence changes the experiment ordering decisively and supports one bounded exhaustive launch.

### 2026-08-25 23:45 UTC - LINC-CONS-009 stopped exhaustive three-genome baseline

- Hypothesis: Exhaustive 255 bp tiling of human, mouse, and opossum will put enough homologous loci in the same database for the frozen Linclust recipe to reduce the input substantially before its internal alignment stage.
- Commit Hash: [`a64725bf`](https://github.com/Open-Athena/marin-dna/commit/a64725bf2ce623c1097e4b56c1b8ad1cfbec9dfe)
- Commands: SkyPilot job 2 ran all 41 tests, the 21-job dry-run, staging, exhaustive extraction, `mmseqs createdb`, and the first Linclust k-mer/set-cover pass on AWS `r7i.4xlarge` in `us-east-2`; after the preliminary cluster count, `sky cancel -y linclust-cons-exhaustive3 2` stopped the job and `sky down -y linclust-cons-exhaustive3` terminated the worker.
- Config: configuration SHA-256 `27136f290026a996f45ab3e18363d8e07e4af4293559aa3b69284691ae29c128`; 255 bp windows at stride 128; human `GCF_000001405.40`, mouse `GCF_000001635.27`, and opossum `GCF_027887165.2`; MMseqs2 18.8cc5c at identity 0.5, coverage 0.8, contiguous k-mers, and the default nucleotide k-mer scale 0.2; 16 threads, 128 GiB RAM, and 500 GB disk.
- Result: The extractor retained 15,718,463 of 24,214,098 human candidates, 14,551,154 of 21,314,017 mouse candidates, and 17,497,482 of 28,016,908 opossum candidates.
  In total it retained 47,767,099 of 73,545,023 candidates, or 64.9495%, and wrote 14,653,924,748 FASTA bytes.
  `createdb` completed in 2 minutes 7 seconds.
  The first Linclust k-mer pass took 5 minutes 18 seconds, found 3,857,889 candidate connections, and produced 45,500,465 preliminary clusters from 47,767,099 sequences.
  Thus 95.25% as many clusters as input sequences remained before Linclust's mandatory internal candidate alignment.
  The alignment stage then used all 16 cores and roughly 20 to 29 GiB RSS for more than 25 minutes, with ample memory and disk, before the user-directed stop.
  No final assignment table or distinct-genome histogram exists.
  The three extracted FASTAs, extraction receipts, manifests, and staging receipts remain durable under `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/canary3-exhaustive/results/v1/a64725bf2ce623c1097e4b56c1b8ad1cfbec9dfe/27136f290026a996f45ab3e18363d8e07e4af4293559aa3b69284691ae29c128/`.
- Negative results: SkyPilot job 1 stopped before workflow execution because the setup command used an incorrectly expanded commit hash; job 2 used the exact requested hash.
- Interpretation: The baseline was deliberately stopped, not failed.
  A 4.75% preliminary reduction is already too weak to justify paying for its long internal alignment or a later representative-to-all search.
  This rejects the frozen whole-genome recipe but does not reject clustering after substantial sensitivity or representation changes.
- Next action: measure cluster recovery and purity on a small fixture where the true homologous groups are known before considering any other whole-genome run.

### 2026-08-26 00:10 UTC - LINC-CONS-010 projected-homology tuning fixture

- Hypothesis: If the main failure is sparse or phase-mismatched genome sampling, Linclust should compress a clean three-species projected-ortholog fixture close to one cluster per human anchor; if it does not, the clustering recipe itself lacks sensitivity.
- Commit Hash: [`0c456c03`](https://github.com/Open-Athena/marin-dna/commit/0c456c03)
- Commands: downloaded the exact preserved issue #417 human, mouse, and armadillo projection Parquets from S3; built the fixture with the pipeline-local bounded selector; ran eight MMseqs2 18.8cc5c Linclust configurations and direct representative-member alignment only on the retained fixture edges; ran `uv run --locked pytest`, default, smoke, and 19-job tuning dry-runs, and scoped pre-commit checks.
- Config: 512 shared projected human anchors with one clean 255 bp row each for human `hg38`, mouse `GCF_000001635.26`, and armadillo `GCF_000208655.1`; 1,536 sequences and 1,536 known within-anchor species pairs; candidate k-mer scales 0.2, 0.3, 0.5, and 1.0; contiguous or spaced k-mers; coverage 0.80, 0.75, or 0.70; minimum identity 0.50 or 0.40; and masked or unmasked prefilter variants.
- Fixture result: The three immutable projection sources share 931,775 anchor IDs.
  The deterministic 4,096-anchor candidate prefix yielded 512 clean complete groups after rejecting four candidates for ambiguity and 24 for majority lowercase.
  The resulting FASTA SHA-256 is `6af225840dfe5269b7b95e7a5f97b2ac594b4f989f814c604c956fb425010d92`; the truth TSV SHA-256 is `959585e53f61166678c533902175d92782d8b6ce89bb353960a42263609d6c3e`.
  Local construction took 1.68 seconds and peaked at 370,428 KiB RSS.

| Variant | Clusters / ideal 512 | Exact 3-species anchors | True pairs recovered / 1,536 | Pair precision | Impure clusters |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1,014 / 1.980× | 180 / 35.2% | 699 / 45.5% | 99.43% | 3 |
| scale 0.3 | 993 / 1.939× | 193 / 37.7% | 733 / 47.7% | 99.46% | 3 |
| spaced, scale 0.3 | 955 / 1.865× | 210 / 41.0% | 788 / 51.3% | 99.49% | 3 |
| spaced, scale 0.3, coverage 0.75 | 945 / 1.846× | 214 / 41.8% | 802 / 52.2% | 99.50% | 3 |
| preceding variant plus identity 0.40 | 945 / 1.846× | 214 / 41.8% | 802 / 52.2% | 99.50% | 3 |
| spaced, scale 1.0 | 898 / 1.754× | 243 / 47.5% | 878 / 57.2% | 99.55% | 3 |
| unmasked spaced, scale 0.3 | 914 / 1.785× | 228 / 44.5% | 849 / 55.3% | 99.07% | 2 |
| spaced, scale 0.5, coverage 0.70, identity 0.40 | 885 / 1.729× | 250 / 48.8% | 897 / 58.4% | 99.34% | 4 |

- Alignment result: The accepted true representative-member edges have mean identity 0.846 and mean query/target coverage about 0.904 under the baseline.
  The best-recall variant lowers those means only to 0.837 identity and 0.892 coverage, so accepted edges remain strong while 41.6% of known true pairs are still missed.
  Baseline pair recall is 51.2% for armadillo-human, 41.4% for armadillo-mouse, and 43.9% for human-mouse.
  The best-recall variant reaches 64.1%, 52.5%, and 58.6%, respectively.
- Negative results: A direct sensitive MMseqs2 nucleotide search was not run on the shared node after its fixed prefilter allocation could not fit within the node's safe memory limit, even with a split-memory request.
  No remote worker or further paid compute was launched for it.
- Interpretation: The clean known-homology fixture falsifies the idea that the full-genome result is explained only by missing corresponding loci.
  The tested Linclust recipes remain severe under-clusterers: even the most relaxed bounded variant produces 1.73 times the ideal cluster count and recovers fewer than half of complete three-species groups.
  High precision shows that the dominant failure is missed homologs rather than cross-anchor over-merging.
  Lowering identity from 0.50 to 0.40 did nothing at the otherwise matched setting, while stronger k-mer sampling produced the largest improvement; the heuristic candidate stage is therefore the clearest current bottleneck.
- Next action: do not launch the prepared 511 bp whole-genome run.
  Decide whether to test a fundamentally more sensitive search/graph construction on the same truth fixture or to stop the Linclust approach for this conservation representation.

### 2026-08-26 00:16 UTC - LINC-CONS-011 durable homology-tuning run

- Hypothesis: The committed 19-job Snakemake target should reproduce the exploratory projected-homology results and publish every fixture, cluster assignment, retained-edge alignment, receipt, and summary under a commit- and configuration-addressed S3 prefix.
- Commit Hash: [`0a44c10f`](https://github.com/Open-Athena/marin-dna/commit/0a44c10f9d03833a64db0d355e6d0e7568f1193e)
- Commands: ran `uv run --locked snakemake --snakefile workflow/Snakefile --configfile config/homology_tuning.yaml --profile workflow/profiles/default tune_homology` locally under the shared-node heavy-work guard.
- Result: all 19 jobs completed in 59.57 seconds with exit status 0 and peak RSS 403,864 KiB.
  The durable summary exactly reproduced the eight exploratory configurations, including 1,014 clusters, 35.2% complete-anchor recovery, and 45.5% true-pair recall at baseline versus 885 clusters, 48.8% complete-anchor recovery, and 58.4% true-pair recall for the best-recall setting.
  Pair precision remained 99.34% or higher for those two endpoints.
- Artifacts: `s3://oa-bolinas/snakemake/analysis/linclust_conservation/results/v1/0a44c10f9d03833a64db0d355e6d0e7568f1193e/8e82a80f8811aada7e0ec514ae14182676c0fc386d087811cda24bd11b28c30c/homology_tuning/` contains the fixture, per-variant assignments and retained-edge alignments, receipts, resource records, `summary.tsv`, and `summary.json`.
- Interpretation: The result is durable and reproducible enough to reject further whole-genome work with this recipe.
  Alignment was deliberately confined to retained edges in the 1,536-sequence positive control; no whole-genome alignment, 511 bp run, remote worker, or EC2 instance was launched.
- Next action: stop this experiment here unless a subsequent decision explicitly selects a different, more sensitive candidate-generation or graph-construction method for the same bounded fixture.

### 2026-08-26 00:46 UTC - LINC-CONS-012 sensitive-search clustering design

- Effort: low targeted background-research follow-up after the Linclust recipe failed on the projected-homology truth fixture.
- Stop rule: stop foraging once primary MMseqs2 guidance and Marin's prior nucleotide-search benchmark identify one bounded clustering workflow that directly tests the candidate-sensitivity hypothesis.
- Question: Can the sensitive MMseqs2 search-and-cluster workflow recover substantially more known projected homologs than Linclust while preserving cluster purity?
- Current Marin context: LINC-CONS-010 recovered 45.5% of true pairs at baseline and 58.4% in the best Linclust variant, with pair precision above 99%.
  Issue #120 found that MMseqs2 nucleotide search at sensitivity 7.5 recovered 70.6% of conserved human-to-mouse cCRE partners at 96.6% precision, establishing a higher-recall pairwise-search precedent without establishing symmetric clustering behavior.
- External prior art: the official MMseqs2 guide describes `cluster` as a sensitive prefilter, alignment, and graph-clustering workflow and describes Linclust as faster and less sensitive.
  `--single-step-clustering` avoids the initial Linclust cascade and exposes the sensitive all-vs-all prefilter directly.
  Greedy set cover forms representative-centered clusters, while connected components uses transitive graph reachability and can therefore improve recall at greater false-merge risk.
- Negative leads: another whole-genome Linclust threshold sweep would repeat the measured candidate bottleneck.
  Minimap2 was dominated by MMseqs2 in issue #120, and LASTZ offered higher recall at roughly 11 times the CPU cost; neither is the smallest next discriminator.
  The pinned MMseqs2 nucleotide prefilter exceeds the shared node's 500 MiB local-work ceiling even on the 1,536-sequence fixture, so the actual search belongs on a small memory-optimized EC2 worker.
- Recommended experiment: hold the 512-anchor, 1,536-sequence truth fixture fixed and compare four MMseqs2 18.8cc5c variants at sensitivity 7.5 and `--max-seqs 1536`: cascaded set cover with reassignment; single-step set cover; single-step connected components; and single-step set cover at identity 0.40 and coverage 0.70.
  Primary metrics are cluster count versus the 512 ideal, complete-anchor recovery, true-pair recall, pair precision, and impure-cluster count.
  The experiment is falsified as a useful replacement if recall remains near Linclust or if improved recall comes from broad cross-anchor merging.
- Commit Hash: [`29d03a86`](https://github.com/Open-Athena/marin-dna/commit/29d03a86177148322b87648fa5a823f257489247)
- Validation: 44 project tests passed; the default 11-job, Linclust-tuning 19-job, and new search-clustering 11-job dry-runs passed; all scoped pre-commit hooks passed.
- Source ledger:

| Source | Type | Location | Claim used for | Confidence | Notes |
| --- | --- | --- | --- | --- | --- |
| Issue #521 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/521#issuecomment-5418934791 | Linclust baseline and stopped whole-genome recipe | High | Coordinating record |
| Issue #120 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/120 | MMseqs2, minimap2, and LASTZ recall/compute comparison | High | Different human-to-mouse top-hit task |
| MMseqs2 guide | Official docs | https://github.com/soedinglab/MMseqs2/wiki#clustering-databases-using-mmseqs-cluster-or-mmseqs-linclust | Search, simple clustering, cascade, and cluster-mode semantics | High | Current guide; execution remains pinned to 18.8cc5c |
| Marin interval alignment | Marin code | https://github.com/Open-Athena/marin-dna/blob/29d03a86177148322b87648fa5a823f257489247/snakemake/training_dataset/dataset_creation/workflow/rules/interval_alignment.smk#L125-L181 | Maintained nucleotide-search flags and sensitivity 7.5 precedent | High | Pairwise projection workflow |
| Search-clustering config | Marin code | https://github.com/Open-Athena/marin-dna/blob/29d03a86177148322b87648fa5a823f257489247/snakemake/analysis/linclust_conservation/config/homology_search_clustering.yaml | Exact bounded experiment matrix | High | Permanent experiment branch |

- Hypothesis queue update: add `LINC-CONS-H2`, that sensitive single-step search will improve true-pair recall over the 58.4% Linclust best while retaining at least 95% pair precision.
- Next action: launch the exact committed target on one `r7i.large`, inspect the first variant's memory and progress, publish the result to the separate workflow-owned S3 prefix, and terminate the worker.

### 2026-08-26 00:51 UTC - LINC-CONS-013 sensitive MMseqs2 clustering result

- Hypothesis: MMseqs2 clustering at sensitivity 7.5 will improve true-pair recall over the 58.4% best Linclust variant while retaining at least 95% pair precision.
- Commit Hash: [`2008742a`](https://github.com/Open-Athena/marin-dna/commit/2008742a5ad678efd591cbcd7c563d84c23adaea)
- Commands: launched SkyPilot job 1 on `linclust-cons-search-cluster`, which ran 44 tests, the 11-job dry-run, and `search_cluster_homology` on AWS `r7i.large` in `us-east-2`; after all artifacts uploaded, `sky down -y linclust-cons-search-cluster` terminated the worker.
- Config: configuration SHA-256 `c4445d3d61d5011c6a4d58ff44ec25197cebc9f440ae850b9394e11056ae6428`; MMseqs2 18.8cc5c; sensitivity 7.5; spaced k-mers; lowercase and low-complexity masking; up to 1,536 prefilter results per query; and the unchanged 512-anchor, 1,536-sequence truth fixture.

| Variant | Clusters / ideal 512 | Complete anchors | True-pair recall | Pair precision | Impure clusters |
| --- | ---: | ---: | ---: | ---: | ---: |
| cascaded set cover, 0.50 identity / 0.80 coverage | 956 / 1.867× | 207 / 40.4% | 786 / 51.2% | 98.99% | 2 |
| single-step set cover, 0.50 / 0.80 | 956 / 1.867× | 207 / 40.4% | 786 / 51.2% | 98.99% | 2 |
| single-step connected components, 0.50 / 0.80 | 955 / 1.865× | 208 / 40.6% | 788 / 51.3% | 98.99% | 2 |
| single-step set cover, 0.40 / 0.70 | 923 / 1.803× | 223 / 43.6% | 835 / 54.4% | 99.05% | 2 |

- Resources: one single-step set-cover variant took 10.02 seconds for `mmseqs cluster` and peaked at 8,398,384 KiB RSS.
  The sensitive prefilter dominated memory; direct representative-member alignment took 0.04 seconds and 22,016 KiB RSS.
- Artifacts: `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/homology-search-clustering/results/v1/2008742a5ad678efd591cbcd7c563d84c23adaea/c4445d3d61d5011c6a4d58ff44ec25197cebc9f440ae850b9394e11056ae6428/` contains the fixture, per-variant assignments and alignments, receipts, resource records, and summaries.
- Interpretation: `LINC-CONS-H2` is falsified for these settings.
  The best sensitive-clustering variant recovered 54.4% of true pairs, below the 58.4% best Linclust variant, and remained at 1.80 times the ideal cluster count.
  Identical cascaded and single-step set-cover results show that the cascade itself is not the limiting step on this fixture.
  Connected components added only two recovered pairs, so graph traversal does not repair the missing-edge problem.
- Next action: run exhaustive `--prefilter-mode 2` alignment on a deterministic 128-anchor subset and feed the complete accepted-edge graph into set-cover and connected-component clustering.
  This is the smallest direct test of whether MMseqs2 alignment thresholds can recover the projected homologs once k-mer candidate generation is removed.

### 2026-08-26 00:58 UTC - LINC-CONS-014 exhaustive accepted-edge graph result

- Hypothesis: Removing MMseqs2's k-mer prefilter will recover substantially more projected homologs, and connected components may join additional true three-species groups beyond greedy set cover.
- Commit Hash: [`389b3d8e`](https://github.com/Open-Athena/marin-dna/commit/389b3d8e421d9052f4c82dea3a3954a99c52db79)
- Commands: launched SkyPilot job 1 on `linclust-cons-exhaustive-graph`, which ran 44 tests, the 11-job dry-run, and `exhaustive_graph_homology` on AWS `r7i.large` in `us-east-2`; after all artifacts uploaded, `sky down -y linclust-cons-exhaustive-graph` terminated the worker.
- Config: configuration SHA-256 `95b5b340dcb6c027a35e18af2819ba960fc9c512da9a87e7cd5dbc7ccca3f49e`; deterministic first 128 clean anchors from the same human, mouse, and armadillo projection sources; 384 sequences and 384 known within-anchor species pairs; MMseqs2 18.8cc5c; set cover or connected components; and a sensitivity-7.5 k-mer control versus `--prefilter-mode 2` exhaustive pair alignment.

| Variant | Clusters / ideal 128 | Complete anchors | True-pair recall | Pair precision | Impure clusters |
| --- | ---: | ---: | ---: | ---: | ---: |
| k-mer search, 0.50 identity / 0.80 coverage, set cover | 281 / 2.195x | 33 / 25.8% | 136 / 35.4% | 100% | 0 |
| no prefilter, 0.50 / 0.80, set cover | 254 / 1.984x | 49 / 38.3% | 179 / 46.6% | 100% | 0 |
| no prefilter, 0.50 / 0.80, connected components | 254 / 1.984x | 49 / 38.3% | 179 / 46.6% | 100% | 0 |
| no prefilter, 0.40 / 0.70, set cover | 241 / 1.883x | 55 / 43.0% | 198 / 51.6% | 100% | 0 |

- Resources: the sensitivity-7.5 k-mer search took 8.60 seconds and peaked at 8,397,440 KiB RSS.
  Exhaustive no-prefilter search took 0.65 to 0.69 seconds and peaked at 22,272 KiB RSS on this 384-sequence fixture.
  Graph clustering and direct representative-member alignment each took at most 0.02 seconds.
- Artifacts: `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/homology-exhaustive-graph/results/v1/389b3d8e421d9052f4c82dea3a3954a99c52db79/95b5b340dcb6c027a35e18af2819ba960fc9c512da9a87e7cd5dbc7ccca3f49e/` contains the fixture, accepted-edge assignments and alignments, receipts, resource records, and summaries.
- Interpretation: removing the k-mer prefilter improves true-pair recall by 11.2 percentage points at fixed thresholds, confirming a material candidate-generation loss.
  Identical set-cover and connected-component partitions show that graph aggregation is not limiting this subset.
  The relaxed exhaustive result still misses 48.4% of known true pairs despite perfect observed cluster precision, so the accepted-alignment threshold frontier remains unresolved.
- Next action: keep the same 128-anchor exhaustive fixture and trace a bounded identity/coverage frontier down to an E-value-only control.
  Stop lowering thresholds when recall saturates or false cross-anchor merges materially reduce precision; do not launch another whole-genome run yet.

### 2026-08-26 01:02 UTC - LINC-CONS-015 exhaustive threshold-frontier design

- Hypothesis: The remaining missed projected homologs are predominantly below the current 70% bidirectional coverage threshold rather than below 40% identity, so lowering coverage should improve recall before cross-anchor merges damage purity.
- Commit Hash: [`c6489dce`](https://github.com/Open-Athena/marin-dna/commit/c6489dced976f11c59cba275c250006aca981379)
- Minimum experiment: on the deterministic 128-anchor, 384-sequence fixture, run no-prefilter set-cover clustering at 0.40 identity with coverage 0.70, 0.60, and 0.50; then lower identity to 0.35 and 0.30 at 0.50 coverage, test 0.30 identity / 0.30 coverage, and finish with the existing E-value threshold as the only acceptance filter.
- Baseline/control: reproduce the 0.40 identity / 0.70 coverage result from LINC-CONS-014 in the new commit-addressed namespace.
- Expected signal: a monotonic recall gain with a visible precision knee identifies the least permissive useful threshold.
- Falsifier: recall remains far below one even for the E-value-only graph, or improved recall occurs only through broad impure merges.
- Cost/risk: at most 147,456 directed alignments per variant; the preceding exhaustive search took less than one second and 23 MiB RSS per variant on `r7i.large`.
- Validation: 44 project tests passed; the new 17-job dry-run and all scoped pre-commit hooks passed.
- Next action: add the exact Snakemake configuration and a separate SkyPilot/S3 run namespace, validate and snapshot it, then launch one bounded worker.

### 2026-08-26 01:08 UTC - LINC-CONS-016 exhaustive threshold-frontier result

- Hypothesis: Lowering bidirectional coverage will improve recall before lower identity becomes relevant or impure cross-anchor merges appear.
- Commit Hash: [`c6489dce`](https://github.com/Open-Athena/marin-dna/commit/c6489dced976f11c59cba275c250006aca981379)
- Commands: launched SkyPilot job 1 on `linclust-cons-threshold-frontier`, which ran 44 tests, the 17-job dry-run, and `exhaustive_graph_homology` on AWS `r7i.large` in `us-east-2`; after all artifacts uploaded, `sky down -y linclust-cons-threshold-frontier` terminated the worker.
- Config: configuration SHA-256 `6f9c7edff869d3cfb93509ce9091f165eefd4aac71cd691d3fba8b6c885610a1`; the same 128-anchor, 384-sequence fixture; MMseqs2 18.8cc5c; no k-mer prefilter; set-cover clustering; E-value 0.001; and identity/coverage thresholds from 0.40/0.70 through an E-value-only endpoint.

| Threshold | Clusters / ideal 128 | Complete anchors | True-pair recall | Pair precision | Impure clusters |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity 0.40 / coverage 0.70 | 241 / 1.883x | 55 / 43.0% | 198 / 51.6% | 100% | 0 |
| 0.40 / 0.60 | 223 / 1.742x | 65 / 50.8% | 226 / 58.9% | 100% | 0 |
| 0.40 / 0.50 | 211 / 1.648x | 72 / 56.2% | 245 / 63.8% | 100% | 0 |
| 0.35 / 0.50 | 211 / 1.648x | 72 / 56.2% | 245 / 63.8% | 100% | 0 |
| 0.30 / 0.50 | 211 / 1.648x | 72 / 56.2% | 245 / 63.8% | 100% | 0 |
| 0.30 / 0.30 | 194 / 1.516x | 83 / 64.8% | 273 / 71.1% | 100% | 0 |
| E-value 0.001 only | 192 / 1.500x | 84 / 65.6% | 276 / 71.9% | 100% | 0 |

- Artifacts: `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/homology-exhaustive-frontier/results/v1/c6489dced976f11c59cba275c250006aca981379/6f9c7edff869d3cfb93509ce9091f165eefd4aac71cd691d3fba8b6c885610a1/` contains the fixture, assignments, retained-edge alignments, receipts, resource records, and summaries.
- Interpretation: the hypothesis is supported for coverage and falsified for identity in the measured range.
  Lowering coverage from 0.70 to 0.50 gains 12.2 percentage points of recall, while lowering identity from 0.40 to 0.30 at 0.50 coverage changes nothing.
  The E-value-only graph still misses 28.1% of true pairs and produces 1.5 times the ideal cluster count despite perfect observed purity.
  The 255 bp representation therefore has an alignability ceiling in addition to candidate-generation loss.
- Communication: posted the sensitive-search, exhaustive-graph, and threshold-frontier metrics, S3 paths, commits, run details, conclusion, and next action to issue #521 at https://github.com/Open-Athena/marin-dna/issues/521#issuecomment-5419212244 and re-fetched the stored comment.
- Next action: extract 511 bp windows around the exact same projected centers from the three compatible cached 2bit genomes, then compare the previous best Linclust recipe with fixed-threshold and E-value-only exhaustive controls.

### 2026-08-26 01:16 UTC - LINC-CONS-017 bounded 511 bp geometry design

- Hypothesis: Extending each projected-center window from 255 to 511 bp will improve both k-mer candidate discovery and the exhaustive alignment ceiling without introducing cross-anchor merges.
- Data lineage: reuse the exact human, mouse, and armadillo projection Parquets and the companion `hg38`, `Mus_musculus`, and `Dasypus_novemcinctus` 2bit genomes from the immutable issue #417 staging namespace.
  The projection contract maps a one- or two-base orthologous center and pads adjacent target-genome sequence, so the 511 bp fixture can be produced by re-centering each accepted target interval without rerunning HAL.
- Coordinate contract: preserve 0-based half-open target coordinates; use each row's pre-resize projected midpoint; require each 511 bp interval to remain within its recorded target sequence size; pass BED6 to pinned UCSC `twoBitToFa` 482 so negative-strand sequences are reverse complemented into human-anchor orientation.
- Fixture: select from the same deterministic 1,024-anchor prefix under the original 255 bp cleanliness gate, extract all eligible 511 bp candidates, filter ambiguous or majority-lowercase expanded sequences, retain 128 complete three-species groups, and report how many match the original clean 128-anchor prefix.
- Candidate comparison: run the frozen Linclust baseline, the previous best spaced-k-mer/scale-0.5 recipe, and a spaced-k-mer/scale-1.0 arm.
- Alignment ceiling: compare sensitivity-7.5 k-mer search, no-prefilter 0.50/0.80 and 0.40/0.70 graphs, and an E-value-only no-prefilter graph.
- Expected signal: lower clusters-per-ideal and higher exact-anchor and true-pair recovery than the matched 255 bp controls while pair precision remains at least 95%.
- Falsifier: 511 bp fails to improve candidate or exhaustive recall, or gains arise only through impure cross-anchor merges.
- Cost/risk: one bounded 384-sequence run; source download is approximately 2.85 GB of immutable in-region inputs, outputs use a new namespace, and no whole-genome clustering is authorized by this design.
- Next action: implement and unit-test the expanded-center fixture builder, validate the combined Snakemake DAG, snapshot it, and run it on one `r7i.large`.

### 2026-08-26 01:23 UTC - LINC-CONS-018 bounded 511 bp geometry result

- Hypothesis: Extending projected-center windows from 255 to 511 bp will improve Linclust candidate discovery and the exhaustive MMseqs2 alignment ceiling without introducing cross-anchor merges.
- Commit Hash: [`dc704fa5`](https://github.com/Open-Athena/marin-dna/commit/dc704fa5d1e0636177653f7ff7d33b7c0dcd5954)
- Commands: launched SkyPilot job 1 on `linclust-cons-window511`, which ran 45 tests, the 19-job dry-run, `tune_homology`, and `exhaustive_graph_homology` on AWS `r7i.large` in `us-east-2`; after all artifacts uploaded, `sky down -y linclust-cons-window511` terminated the worker.
- Config: configuration SHA-256 `97f523755b1a079221549f29813e097ae71deec6f0acc8378bc7736d59cad213`; 128 complete human, mouse, and armadillo groups; 384 sequences; 511 bp strand-oriented windows around projected centers; MMseqs2 18.8cc5c; three Linclust candidate recipes; and four exhaustive-graph controls.
- Fixture audit: 123 of the 128 selected anchors match the original deterministic 255 bp prefix.
  Five replacements were required because expansion introduced an ambiguous base or majority-lowercase sequence in at least one species.

| 511 bp variant | Clusters / ideal 128 | Complete anchors | True-pair recall | Pair precision | Impure clusters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linclust baseline | 320 / 2.500x | 15 / 11.7% | 17.7% | 83.95% | 11 |
| Linclust previous 255 bp recipe | 184 / 1.438x | 3 / 2.3% | 25.0% | 22.59% | 105 |
| Linclust dense spaced k-mers | 181 / 1.414x | 3 / 2.3% | 25.8% | 23.29% | 105 |
| sensitivity-7.5 k-mer graph, 0.50 / 0.80 | 308 / 2.406x | 21 / 16.4% | 22.4% | 87.76% | 11 |
| no prefilter, 0.50 / 0.80 | 316 / 2.469x | 24 / 18.8% | 24.0% | 100% | 0 |
| no prefilter, 0.40 / 0.70 | 313 / 2.445x | 25 / 19.5% | 25.0% | 100% | 0 |
| no prefilter, E-value only | 280 / 2.188x | 39 / 30.5% | 37.0% | 98.61% | 1 |

- Matched-anchor control: restricting both E-value-only partitions to the exact same 123 projected anchors gives 71.3% true-pair recall and 65.0% exact-anchor recovery at 255 bp versus 37.7% and 31.7% at 511 bp.
  Both induced partitions have 100% pair precision.
- Resources: the 511 bp E-value-only search took 0.82 seconds and peaked at 22,016 KiB RSS; dense Linclust took 0.11 seconds and peaked at 22,016 KiB RSS on the 384-sequence fixture.
- Artifacts: `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/homology-window511/results/v1/dc704fa5d1e0636177653f7ff7d33b7c0dcd5954/97f523755b1a079221549f29813e097ae71deec6f0acc8378bc7736d59cad213/` contains the expanded fixture, assignments, alignments, receipts, resource records, and summaries.
- Interpretation: the hypothesis is falsified.
  Expanding around a projected one- or two-base center adds target-genome flanks that are not guaranteed to be homologous, dilutes the conserved block, and substantially lowers recovery for every species pair.
  The severe impurity of the dense 511 bp Linclust variants also makes them unusable despite their superficially lower cluster counts.
- Next action: keep 255 bp, investigate the scalable candidate stage directly, and test explicit shorter nucleotide k-mers before implementing a new candidate algorithm.

### 2026-08-26 01:38 UTC - LINC-CONS-019 short-seed Linclust exploratory result

- Hypothesis: Linclust's automatically selected nucleotide k-mer is too long for divergent 255 bp mammalian windows, and an explicit shorter k-mer can approach the exhaustive-alignment ceiling without sacrificing purity.
- Scope: zero-cloud-cost local exploration on the existing 128-anchor, 384-sequence 255 bp fixture; all commands completed in less than two seconds and stayed below the shared-node heavy-work threshold.
- Fixed settings: MMseqs2 18.8cc5c; 0.40 minimum identity; 0.70 bidirectional coverage; E-value 0.001; 20 base k-mers plus scale 0.5; spaced k-mers; set-cover clustering; and lowercase plus low-complexity masking unless named otherwise.

| Candidate setting | Clusters / ideal 128 | Complete anchors | True-pair recall | Pair precision |
| --- | ---: | ---: | ---: | ---: |
| automatic 17-mer | 238 / 1.859x | 43.0% | 52.3% | 100% |
| explicit 15-mer | 222 / 1.734x | 50.0% | 58.9% | 100% |
| explicit 12-mer, unmasked | 205 / 1.602x | 56.2% | 65.4% | 100% |
| explicit 11-mer, masked | 212 / 1.656x | 51.6% | 62.0% | 100% |
| explicit 9-mer, masked, default hash shift | 203 / 1.586x | 57.0% | 66.1% | 100% |
| explicit 9-mer, masked, hash shift 1 | 200 / 1.562x | 58.6% | 67.4% | 100% |
| explicit 7-mer, masked | 257 / 2.008x | 35.2% | 44.8% | 100% |

- Threshold warning: lowering the 11-mer Linclust acceptance threshold to 0.30 identity / 0.50 coverage reduced pair precision to 52.1%; the E-value-only arm recovered 80.7% of true pairs but precision fell to 41.0% with 52 impure clusters.
  Unlike the exhaustive graph, greedy Linclust's intermediate representatives can therefore amplify false short-seed connections when acceptance thresholds are loose.
- Stability warning: 9-mer recall ranged from 65.6% to 67.4% across tested hash shifts, so seed selection contributes about two percentage points of variation even on this small fixture.
- Interpretation: automatic 17-mer selection was a material, previously untested source of candidate loss.
  A masked 9-mer at the existing 0.40/0.70 acceptance gate comes within 4.4 percentage points of the 71.9% exhaustive E-value-only ceiling while retaining perfect observed purity.
  However, a 9-mer has only 262,144 possible keys, so clean-fixture accuracy alone cannot establish scalability across 47.8 million real tiles.
- Next action: inject the 384 truth sequences into increasing real-background samples from the already extracted three-genome FASTAs, then measure recovery, impurity, runtime, and peak memory for short-seed Linclust variants.

### 2026-08-26 01:47 UTC - LINC-CONS-020 real-background seed-scaling design

- Hypothesis: explicit 9- to 13-mer Linclust will retain most of its bounded projected-homology recovery as unrelated real genomic tiles are added, while the fixed 0.40 identity / 0.70 coverage gate prevents decoy contamination.
- Data lineage: reuse the immutable 15.7 million human, 14.6 million mouse, and 17.5 million opossum retained-tile FASTAs from LINC-CONS-009 by ETag- and size-guarded streaming reads.
  Do not re-extract or copy the source genomes and do not alter the existing all-tiles namespace.
- Fixtures: balanced deterministic FASTA prefixes containing 100,000, 1,000,000, or 5,000,000 background records, followed by the unchanged 128-anchor, 384-sequence human/mouse/armadillo truth fixture.
- Variants: automatic nucleotide k-mer length and explicit 13-, 11-, and 9-mers; all use MMseqs2 18.8cc5c, hash shift 1, spaced k-mers, lowercase and low-complexity masking, 20 base k-mers plus scale 0.5, 0.40 identity, 0.70 bidirectional coverage, E-value 0.001, and greedy set cover.
- Evaluation: require every combined sequence to occur exactly once in the assignment table; report truth-pair recall and precision, exact three-species recovery, truth-to-truth false merges, truth clusters contaminated by decoys, singleton fraction, total clusters, wall/CPU time, and peak RSS.
- Expected signal: the shorter seed retains at least 60% truth-pair recall at one million background tiles with no truth-to-truth false pair and less than 5% truth-record contamination.
- Falsifier: recovery collapses toward the automatic-k baseline, truth clusters acquire material decoy contamination, or time/memory grows too rapidly to extrapolate to the 47.8-million-tile input.
- Cost/risk: twelve Linclust runs on one `r7i.4xlarge`, each capped at 45 minutes, using an estimated 2 GB of newly materialized input FASTAs and a separate `homology-background-scaling` S3 prefix.
  The user authorized up to $20 of EC2 spend for the overnight scalable-clustering investigation.
- Next action: run all project tests and the credential-free DAG, commit and push the exact target, launch the EC2 worker, inspect the 100,000-record results before trusting the larger arms, and terminate the worker when the complete summary is durable.

### 2026-08-26 03:05 UTC - LINC-CONS-020 real-background seed-scaling result

- Hypothesis: explicit 9- to 13-mer Linclust will retain most of its bounded projected-homology recovery as unrelated real genomic tiles are added.
- Commit Hash: [`2732c884`](https://github.com/Open-Athena/marin-dna/commit/2732c8847a666b27a9a4dec47a5027bb257259fb)
- Config: configuration SHA-256 `dba172739b490b90a1d032bfd65d9002a926ad2dcb2b311a15d772d5708054bb`; 100,000, 1,000,000, and 5,000,000 balanced real background tiles plus the unchanged 384 truth sequences; automatic k-mer selection or fixed 13-, 11-, or 9-mers; and the fixed 0.40 identity / 0.70 coverage acceptance gate.

| Background | Seed | Recall | Strict precision | Exact anchors | Decoy contamination | Clusters | MMseqs wall time |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | automatic k17 | 54.9% | 100.0% | 46.1% | 0.0% | 97,181 | 2.44 s |
| 100,000 | k13 | 59.4% | 98.7% | 49.2% | 0.8% | 96,736 | 5.21 s |
| 100,000 | k11 | 43.2% | 98.2% | 31.2% | 0.8% | 96,944 | 16.35 s |
| 100,000 | k9 | 3.9% | 100.0% | 3.1% | 0.0% | 99,320 | 17.90 s |
| 1,000,000 | automatic k17 | 54.9% | 100.0% | 46.1% | 0.0% | 982,569 | 17.77 s |
| 1,000,000 | k13 | 46.1% | 98.3% | 35.9% | 0.8% | 979,726 | 155.37 s |
| 1,000,000 | k11 | 7.3% | 93.3% | 4.7% | 0.5% | 993,612 | 199.22 s |
| 1,000,000 | k9 | 0.8% | 100.0% | 0.8% | 0.0% | 999,401 | 176.92 s |
| 5,000,000 | automatic k17 | 54.4% | 99.1% | 45.3% | 0.5% | 4,841,525 | 141.34 s |
| 5,000,000 | k13 | 18.8% | 85.7% | 12.5% | 0.8% | 4,887,287 | 1,008.32 s |
| 5,000,000 | k11 | 1.6% | 100.0% | 1.6% | 0.0% | 4,981,122 | 1,047.15 s |
| 5,000,000 | k9 | 0.0% | 100.0% | 0.0% | 0.0% | 4,997,933 | 896.27 s |

- Resources: at five million backgrounds, automatic k17 produced about 46.8 million candidate alignments, while each shorter fixed seed produced 609 to 668 million alignments.
  Automatic k17 was about 7.1 times faster than k13 and required 12.5 GiB peak RSS.
- Artifacts: `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/homology-background-scaling/results/v1/2732c8847a666b27a9a4dec47a5027bb257259fb/dba172739b490b90a1d032bfd65d9002a926ad2dcb2b311a15d772d5708054bb/` contains all fixtures, assignments, resource records, and evaluator receipts.
  A summary recovery at commit [`5873ef6f`](https://github.com/Open-Athena/marin-dna/commit/5873ef6fe1fd9047153d703c2319ca19dddf81a2) reads those immutable receipts and writes the recovered summary under configuration SHA-256 `4b0573d845754433eeda21866d4d26ef5884e877c399a22634f7095f1bda5d16`.
- Interpretation: the hypothesis is falsified.
  Fixed short seeds saturate their finite key spaces as the database grows, so candidate groups become dominated by unrelated sequences and the center-selection heuristic stops exposing the truth edges.
  Automatic k17 adapts to the database and preserves recall, time, and observed purity, but clusters only about 3.2% of five million real tiles.
  The result rejects the expectation that this recipe will naturally collapse three-species tiles to approximately `n_sequences / n_species` clusters.
- Next action: keep automatic k17 and test whether a fixed number of independent hash selections recovers complementary verified edges without changing the asymptotic `O(N)` form.

### 2026-08-26 03:20 UTC - LINC-CONS-021 automatic-k hash ensemble result

- Hypothesis: four automatic-k17 Linclust passes with independent hash shifts will recover complementary verified homology edges while preserving real-background purity.
- Commit Hash: [`5873ef6f`](https://github.com/Open-Athena/marin-dna/commit/5873ef6fe1fd9047153d703c2319ca19dddf81a2)
- Commands: SkyPilot job 4 ran four hash shifts at one million and five million backgrounds on AWS `r7i.4xlarge` in `us-east-2`, merged their verified cluster edges with a streaming union-find, evaluated the combined partitions, uploaded all 25 workflow outputs, and terminated the worker.
- Config: configuration SHA-256 `24e867dbc6a333a03b50f33bb8f0ae2bd0dac387cb4ece6e677b1a6b1af4fda9`; automatic k17; hash shifts 1, 67, 123, and 251; 20 base k-mers plus length scale 0.5; and unchanged acceptance thresholds.

| Background | Result | Recall | Strict precision | Exact anchors | Decoy contamination | Clusters | Aggregate MMseqs wall time |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000,000 | best single hash | 54.9% | 100.0% | 46.1% | 0.0% | 982,569 | 19.95 s |
| 1,000,000 | four-hash union | 56.2% | 100.0% | 47.7% | 0.0% | 980,620 | 80.37 s |
| 5,000,000 | best single hash | 54.4% | 99.1% | 45.3% | 0.5% | 4,841,525 | 161.38 s |
| 5,000,000 | four-hash union | 56.2% | 97.7% | 47.7% | 0.5% | 4,815,329 | 641.90 s |

- Merge audit: the four five-million-sequence inputs each had 158,859 to 159,013 non-representative assignment edges, while their union had 185,055 edges.
  The different hashes therefore found different real-background edges, but very few additional truth edges.
- Artifacts: `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/homology-background-ensemble/results/v1/5873ef6fe1fd9047153d703c2319ca19dddf81a2/24e867dbc6a333a03b50f33bb8f0ae2bd0dac387cb4ece6e677b1a6b1af4fda9/` contains all per-hash assignments and receipts, merged partitions, merge audits, and summaries.
- Interpretation: the hypothesis is weakly supported for complementarity but rejected as a useful scalable recipe.
  Four hashes gain only 1.3 percentage points of recall over the best single pass at five million backgrounds, multiply MMseqs time by four, and reduce strict precision by 1.4 percentage points.
  Transitive union did not cause broad contamination in this fixture, but the marginal verified-edge yield is too small.
- Next action: sweep the number of selected k-mers within one automatic-k17 pass, which is the sensitivity control documented by Linclust and avoids paying repeated database construction and clustering costs.

### 2026-08-26 03:30 UTC - LINC-CONS-022 automatic-k sampling-density design

- Hypothesis: spending the candidate budget on denser automatic-k17 sampling within one pass will recover more truth edges per unit of time than four independent hash passes.
- Commit Hash: [`6c86097e`](https://github.com/Open-Athena/marin-dna/commit/6c86097e44676f07ae398afa7831a01e17894981)
- Config: configuration SHA-256 `225aac22428329931549bf90c6805c404c84ce2bdfe2aaae757c5bd973eb387a`; exactly 20, 80, 148, or 239 selected k-mers per 255 bp sequence; automatic k-mer length; hash shift 1; zero length-based sampling scale; and the unchanged 0.40 identity / 0.70 coverage gate.
- Scale: the same 100,000, 1,000,000, and 5,000,000 balanced real-background fixtures, yielding twelve Linclust runs and twelve evaluator jobs.
- Baseline relation: 148 selected k-mers approximates the previous `20 + 0.5 × 255` setting, while 239 is the maximum number of start positions for a contiguous 17-mer in a 255 bp window.
- Expected signal: denser sampling improves recall monotonically and the best single pass reaches or exceeds the 56.2% four-hash recall in less than 641.9 seconds at five million backgrounds.
- Falsifier: recall saturates near the previous single-pass result, strict precision falls materially, or candidate cost grows faster than the marginal truth-edge yield.
- Artifacts: `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/homology-background-density/results/v1/6c86097e44676f07ae398afa7831a01e17894981/225aac22428329931549bf90c6805c404c84ce2bdfe2aaae757c5bd973eb387a/` is the separate durable output namespace.
- Execution note: fresh AWS Sky images exposed a hard-coded Miniforge and stale Conda assumption before scientific work began.
  The pinned task now detects Miniforge or Miniconda, upgrades Conda to 25.7.0, runs all 50 tests and the 30-job dry-run, and only then starts the experiment.
- Next action: monitor SkyPilot job 4 on `linclust-cons-background-density`, stop the worker after the durable summary lands, and compare recall, strict precision, clusters, candidate cost, wall time, and peak RSS.

### 2026-08-26 03:43 UTC - LINC-CONS-022 automatic-k sampling-density result

- Hypothesis: spending the candidate budget on denser automatic-k17 sampling within one pass will recover more truth edges per unit of time than four independent hash passes.
- Commit Hash: [`6c86097e`](https://github.com/Open-Athena/marin-dna/commit/6c86097e44676f07ae398afa7831a01e17894981)
- Commands: SkyPilot job 4 completed all twelve Linclust and twelve evaluation arms on AWS `r7i.4xlarge` in `us-east-2`; the worker remained warm only for the immediately following bounded algorithm comparison.
- Config: configuration SHA-256 `225aac22428329931549bf90c6805c404c84ce2bdfe2aaae757c5bd973eb387a`; automatic k17; hash shift 1; zero length-based sampling scale; and 20, 80, 148, or 239 selected k-mers per 255 bp sequence.

| Background | Selected k-mers | Recall | Strict precision | Exact anchors | Decoy contamination | Clusters | Wall time | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 20 | 31.0% | 100.0% | 22.7% | 0.0% | 97,453 | 1.33 s | 97 MiB |
| 100,000 | 80 | 51.3% | 100.0% | 42.2% | 0.0% | 97,212 | 1.57 s | 278 MiB |
| 100,000 | 148 | 54.9% | 100.0% | 46.1% | 0.0% | 97,188 | 1.96 s | 468 MiB |
| 100,000 | 239 | 52.9% | 100.0% | 43.8% | 0.0% | 97,235 | 2.03 s | 655 MiB |
| 1,000,000 | 20 | 31.0% | 100.0% | 22.7% | 0.0% | 986,764 | 11.98 s | 838 MiB |
| 1,000,000 | 80 | 51.3% | 100.0% | 42.2% | 0.0% | 983,582 | 16.36 s | 1.71 GiB |
| 1,000,000 | 148 | 54.9% | 100.0% | 46.1% | 0.0% | 982,559 | 19.93 s | 2.73 GiB |
| 1,000,000 | 239 | 52.9% | 100.0% | 43.8% | 0.0% | 982,530 | 22.79 s | 4.08 GiB |
| 5,000,000 | 20 | 31.0% | 99.2% | 22.7% | 0.3% | 4,896,355 | 68.67 s | 3.06 GiB |
| 5,000,000 | 80 | 51.0% | 99.0% | 42.2% | 0.5% | 4,855,873 | 113.54 s | 7.53 GiB |
| 5,000,000 | 148 | 54.9% | 99.1% | 46.1% | 0.5% | 4,841,534 | 157.78 s | 12.6 GiB |
| 5,000,000 | 239 | 53.4% | 97.6% | 44.5% | 0.5% | 4,839,481 | 184.24 s | 19.4 GiB |

- Artifacts: `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/homology-background-density/results/v1/6c86097e44676f07ae398afa7831a01e17894981/225aac22428329931549bf90c6805c404c84ce2bdfe2aaae757c5bd973eb387a/` contains all fixtures, assignments, resource records, evaluator receipts, and summaries.
- Interpretation: the hypothesis is falsified.
  Recall saturates at the existing 148-k-mer setting and does not improve monotonically at maximum density.
  At five million backgrounds, 239 k-mers lose 1.6 recall percentage points relative to 148, take 17% longer, use 54% more peak memory, and lower strict precision by 1.4 percentage points.
  One-pass density tuning therefore cannot close the candidate-recovery gap, although the 20-to-148 progression confirms that sampling density is material below the saturation point.
- Next action: benchmark DECIPHER Clusterize, whose rare-k-mer and relatedness-sorting phases are specifically designed to retain distant relationships without relying on one representative-centered common-k-mer bucket.

### 2026-08-26 03:44 UTC - LINC-CONS-023 relatedness-sorting clustering design

- Hypothesis: DECIPHER Clusterize will preserve more projected truth pairs than Linclust's 54.9% single-pass plateau as real background grows because it prioritizes rare k-mers and then sorts sequences by relatedness before bounded alignment phases.
- Commit Hash: [`14f8b65`](https://github.com/Open-Athena/marin-dna/commit/14f8b65dcf9da4e3f035f4d1da05f03e8680bec1)
- Data: reuse the exact 100,000- and 1,000,000-tile balanced real-background construction and unchanged 128-anchor truth fixture from LINC-CONS-020.
- Variants: DECIPHER 3.6.0 Clusterize with center linkage, 0.70 minimum coverage, distance cutoffs 0.5 or 0.6, lowercase and low-complexity masking, deterministic seed 521, and fixed phase and alignment bounds.
- Expected signal: materially higher recall than Linclust at one million backgrounds without material truth-to-truth false merges or decoy contamination, at a runtime and memory footprint that can plausibly extend to five million tiles.
- Falsifier: recall does not improve, precision becomes unusable, or the one-million-tile arm is already too slow or memory-intensive to scale.
- Cost/risk: four bounded runs on the already warm approved `r7i.4xlarge`; a new pinned R/DECIPHER Conda environment; and a separate `homology-background-clusterize` S3 namespace.
- Validation: 50 project tests passed; the 13-job dry-run and all scoped pre-commit hooks passed before launch.
- Next action: monitor SkyPilot job 5, evaluate the completed 100,000- and 1,000,000-tile partitions against the same truth/decoy contracts, and decide from measured scaling whether a five-million-tile arm is justified.

### 2026-08-26 04:36 UTC - LINC-CONS-023 relatedness-sorting clustering result

- Hypothesis: DECIPHER Clusterize will preserve more projected truth pairs than Linclust's 54.9% single-pass plateau as real background grows because it prioritizes rare k-mers and relatedness sorting.
- Commit Hash: [`14f8b65`](https://github.com/Open-Athena/marin-dna/commit/14f8b65dcf9da4e3f035f4d1da05f03e8680bec1)
- Config: configuration SHA-256 `4973dddae6448c85b9581edc1255fcf30067eb248703a6140689095e8fe85451`; DECIPHER 3.6.0; center linkage; 0.70 minimum coverage; fixed phase bounds 20,000/2,000/2,000; at most 200 alignments; 50 rare k-mers; and deterministic seed 521.

| Background | Distance cutoff | Clusters | Singleton fraction | Truth recall | Truth-pair precision | Strict cluster-pair precision | Truth-decoy contamination | Wall time | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 0.6 | 189 | 0.0% | 25.5% | 5.2% | 0.00027% | 100% | 207.54 s | 1.20 GiB |
| 1,000,000 | 0.6 | 310 | 0.0% | 28.6% | 4.5% | 0.000005% | 100% | 2,208.40 s | 8.22 GiB |
| 100,000 | 0.5 | 1,392 | 0.04% | 59.6% | 58.1% | 0.0175% | 100% | 452.47 s | 1.22 GiB |

- Phase audit: the one-million distance-0.6 arm spent 269 seconds partitioning into 60,812 detectable-homology groups, 395 seconds sorting by relatedness, and 1,533 seconds in final 9-mer clustering.
  Relatedness sorting contributed to every clustered sequence and rare 11-mers contributed to 94.9%, but those links overwhelmingly joined unrelated genomic tiles.
- Stop decision: cancelled the redundant one-million distance-0.5 arm 33 seconds after it started.
  Evaluator-only Snakemake targets then produced durable receipts for the two completed distance-0.6 assignments under their original commit/config provenance.
- Artifacts: `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/homology-background-clusterize/results/v1/14f8b65dcf9da4e3f035f4d1da05f03e8680bec1/4973dddae6448c85b9581edc1255fcf30067eb248703a6140689095e8fe85451/` contains the completed assignments, resource reports, versions, and evaluator receipts.
- Interpretation: the hypothesis is falsified for this representation and threshold regime.
  Clusterize's linear memory scaling is credible, but low-similarity local matches make almost all real genomic windows non-singletons and collapse them into giant mixed clusters.
  A stricter cutoff improves truth recall and pair precision but remains orders of magnitude below a plausible cluster count and contaminates every truth record.
  Ideal linear extrapolation of the default one-million runtime is also about one day per 47.8-million-tile parameter arm on this instance.
- Next action: test a candidate-only graph that removes high-frequency seeds, emits only cross-genome edges, and prevents any component from acquiring two records from the same source genome.

### 2026-08-26 04:42 UTC - LINC-CONS-024 repeat-capped species-aware seed-graph design

- Hypothesis: the short-seed signal that succeeded on the bounded truth fixture can survive real background if high-frequency buckets are discarded and candidate competition is restricted across source genomes, avoiding both Linclust's representative dilution and Clusterize's giant mixed components.
- Data: the same 100,000 balanced human, mouse, and opossum background records plus the unchanged 384-record projected human/mouse/armadillo truth fixture.
- Candidate construction: canonicalize every contiguous DNA k-mer across strands; choose a deterministic bottom hash sample per sequence; discard any seed observed above a fixed record-frequency cap; emit only record pairs from different source genomes; and require one or two shared retained seeds.
- Partition constraint: process candidate edges by descending shared-seed support and merge components only when their source-genome sets are disjoint.
  Background human and mouse accessions are explicitly aliased to the corresponding truth source labels, so the invariant cannot be bypassed by identifier format.
- Variants: 9-mer/128 seeds/frequency 8/minimum support 2; 13-mer/128 seeds/frequency 8 or 32/minimum support 2; 15-mer/128 seeds/frequency 8/minimum support 1; and 17-mer/148 seeds/frequency 8/minimum support 1.
- Evaluation: report truth-pair recall before graph aggregation, unique and qualifying candidate-pair counts, accepted edges, cluster and singleton counts, post-partition truth recall and precision, decoy contamination, runtime, and peak RSS.
- Expected signal: at least one variant preserves 50% candidate truth recall, keeps every component source-unique, avoids broad truth-decoy contamination, and emits a candidate set small enough for later bounded alignment.
- Falsifier: frequency caps erase the truth signal, qualifying candidates explode despite the caps, or greedy source-unique matching still replaces most truth partners with decoys.
- Cost/risk: five candidate-only 100,000-background arms, each hard-capped at 20 minutes, on one approved `r7i.4xlarge`; no sequence alignment and a separate `homology-background-seed-graph` S3 namespace.
- Validation: 53 project tests passed and the 14-job credential-free dry-run succeeded.
- Next action: snapshot and push the exact target, launch one EC2 worker, inspect every completed arm before considering any larger scale, and terminate the worker after the durable summary lands.

### 2026-08-26 05:14 UTC - LINC-CONS-024 repeat-capped species-aware seed-graph result

- Hypothesis: repeat caps and source-unique components will retain short-seed sensitivity while avoiding both representative dilution and giant mixed components.
- Commits: [`8ee311ed`](https://github.com/Open-Athena/marin-dna/commit/8ee311ede108f9f93b6b43063fb3826923f5a11c) implemented the graph; [`aa5f89e0`](https://github.com/Open-Athena/marin-dna/commit/aa5f89e0f570ffe76bc37183f4234337e9c2bd4b) made the fresh-worker Conda setup reproducible; [`9f9aee5a`](https://github.com/Open-Athena/marin-dna/commit/9f9aee5a2aae5b0a07461b994b5d8b100ec321f4) refined the repeat caps and seed lengths; and [`e860986c`](https://github.com/Open-Athena/marin-dna/commit/e860986cd86fe43026ad8db57eb97cefe8f77cd1) scaled the best 100,000-background arm to one million.
- Configs: initial SHA-256 `eb748c96deca704580cb58a119aa44240e3a2d092130892d61a5e12cc9ba0ed0`; refinement SHA-256 `1de6a571dc11a2324b6eb6a0fed4bf91a60a2bde8d0d60cb01efdf9447caaff1`; scaling SHA-256 `d3bc2ddff5f94e4753062541cd7daf00d9f2ccafb9406ebdb8f3758de6746d62`.
- Execution: three successful SkyPilot jobs ran on AWS `r7i.4xlarge` in `us-east-2`; every component contained at most one record from each of the four effective source labels, no alignment was performed, and each worker was terminated after its durable summary landed.

| Background | Variant | Candidate recall | Graph recall | Strict precision | Exact anchors | Decoy contamination | Wall time | Peak RSS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | k13, 128 seeds, cap 32, support 2 | 87.8% | 74.7% | 37.9% | 18.0% | 75.8% | 32.04 s | 1.63 GiB |
| 100,000 | k17, 148 seeds, cap 8, support 1 | 76.8% | 73.7% | 56.0% | 41.4% | 38.5% | 31.54 s | 1.64 GiB |
| 100,000 | k17, 239 seeds, cap 2, support 1 | 31.0% | 35.7% | 62.0% | 19.5% | 13.8% | 35.76 s | 2.67 GiB |
| 100,000 | k17, 239 seeds, cap 8, support 2 | 75.0% | 72.1% | 66.4% | 49.2% | 25.5% | 35.77 s | 2.71 GiB |
| 100,000 | k19, 237 seeds, cap 8, support 1 | 71.4% | 69.8% | 76.4% | 53.9% | 15.1% | 35.43 s | 2.71 GiB |
| 1,000,000 | k19, 237 seeds, cap 8, support 1 | 71.4% | 70.1% | 50.9% | 33.6% | 45.6% | 362.39 s | 25.6 GiB |

- Negative controls: the 100,000-background k9/support-2 arm recovered no true pairs, while the k13 and k15 arms contaminated at least 75.3% of truth records.
  Increasing k17 density from 148 to 239 seeds raised candidate recall but also raised truth-decoy competition unless support or repeat caps became strict enough to erase much of the truth signal.
- Artifacts: the initial sweep is at `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/homology-background-seed-graph/results/v1/aa5f89e0f570ffe76bc37183f4234337e9c2bd4b/eb748c96deca704580cb58a119aa44240e3a2d092130892d61a5e12cc9ba0ed0/`.
  The refinement is at `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/homology-background-seed-graph-refinement/results/v1/9f9aee5a2aae5b0a07461b994b5d8b100ec321f4/1de6a571dc11a2324b6eb6a0fed4bf91a60a2bde8d0d60cb01efdf9447caaff1/`.
  The one-million scaling arm is at `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/homology-background-seed-graph-scaling/results/v1/e860986cd86fe43026ad8db57eb97cefe8f77cd1/d3bc2ddff5f94e4753062541cd7daf00d9f2ccafb9406ebdb8f3758de6746d62/`.
- Interpretation: the graph implementation has approximately linear runtime and memory, and its source-set invariant prevents giant components.
  The scientific hypothesis is nevertheless falsified because weak exact-seed links increasingly replace true partners with genomic decoys as the background grows.
  The promising 100,000-background exact-anchor result does not survive one-million-background competition.
- Next action: align only the graph pairs in truth-containing one-million-background components to test whether verification can remove decoys without erasing the candidate-recall gain.

### 2026-08-26 05:25 UTC - LINC-CONS-025 bounded seed-graph alignment diagnostic

- Hypothesis: exhaustive nucleotide alignment of the small set of graph pairs in truth-containing components will remove seed-only decoys while retaining enough true pairs to beat the 54.9% Linclust recall baseline.
- Commit: [`1167295a`](https://github.com/Open-Athena/marin-dna/commit/1167295ae01ae842572aae64314b9f45fed1fe42).
- Config: configuration SHA-256 `aee07d35701d33142b728fbb54ab3fa8a81f9338558eee53a4f188f0abe79f7d`; exact one-million-background k19 graph inputs; MMseqs2 18.8cc5c no-filter nucleotide search; and E-value-only, 0.40 identity/0.70 coverage, or 0.50 identity/0.80 coverage gates.
- Scope: streaming the one-million assignment table selected 496 records in 185 truth-containing components.
  Those components contained 269 true pairs, 13 false truth-to-truth pairs, 214 truth-decoy pairs, and 33 background-background pairs.
  Only this 496-record subset was aligned.

| Alignment gate | Global truth recall | Graph true-pair retention | Truth-decoy retention | Retained-pair precision |
| --- | ---: | ---: | ---: | ---: |
| E-value <= 0.001 | 52.3% | 74.7% | 0.0% | 100.0% |
| identity >= 0.40, coverage >= 0.70, E-value <= 0.001 | 41.1% | 58.7% | 0.0% | 100.0% |
| identity >= 0.50, coverage >= 0.80, E-value <= 0.001 | 37.2% | 53.2% | 0.0% | 100.0% |

- Resources: the 496-record MMseqs2 diagnostic took 0.44 seconds wall time, 1.42 CPU seconds, and 22,016 KiB peak RSS on an AWS `r7i.2xlarge`; the worker was terminated immediately after success.
- Artifacts: `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/homology-seed-alignment-diagnostic/results/v1/1167295ae01ae842572aae64314b9f45fed1fe42/aee07d35701d33142b728fbb54ab3fa8a81f9338558eee53a4f188f0abe79f7d/` contains the selected FASTA, labeled graph pairs, alignments, resource record, version, and evaluation receipt.
- Interpretation: alignment verification perfectly removes false and decoy graph pairs in this diagnostic, but even the E-value-only gate lowers global recall to 52.3%, below the 54.9% Linclust baseline.
  The hypothesis is falsified, and adding the intended coverage gate makes the deficit larger.
  The scalable architecture remains attractive—sparse candidate generation followed by bounded verification—but this exact-seed candidate generator does not provide a useful recall/precision frontier.
- Next action: stop tuning this recipe.
  Further work should start from a different source of candidate evidence, such as positional or syntenic context, rather than more short-seed hashes, denser sampling, or broader alignment.

### 2026-08-26 11:50 UTC - LINC-CONS-026 full 20-genome Linclust design

- Hypothesis: the measured best scalable Linclust recipe can complete on every retained 255 bp tile from the exact 20-order panel within the approved $20 budget and produce a useful empirical distribution of singleton, within-genome, and cross-genome clusters.
- Commit: [`21c3528f`](https://github.com/Open-Athena/marin-dna/commit/21c3528f014a15aa40e4a1858f3e22dfd26b7a87).
- Data: exactly the 20 selected assemblies in `config/assembly_selection.tsv`, one per mammalian order, totaling 60.112 Gb and approximately 469.6 million candidate windows before ambiguity and majority-repeat filtering.
- Reuse: 18 exact versioned 2bit inputs have pinned S3 sources; giraffe `GCF_054371585.1` and donkey `GCF_041296235.1` require fresh NCBI staging into this run's commit- and configuration-derived namespace.
- Recipe: 255 bp windows at stride 128; automatic k-mer length; exactly 148 selected spaced k-mers per sequence; hash shift 1; 0.40 minimum identity; 0.70 bidirectional coverage; lowercase and low-complexity masking; and MMseqs2 `18.8cc5c`.
- Compute: one spot AWS `x2iedn.16xlarge` in `us-east-2` with 64 vCPUs, 2 TiB RAM, a 1.5 TB root disk, a 1.4 TB MMseqs split-memory limit, and a five-hour Linclust timeout.
- Cost bound: SkyPilot reported $1.91 per compute hour on 2026-08-26, making the five-hour Linclust phase at most $9.55 in compute before setup and receipt time; the task uses `--down` and the total approved ceiling remains $20.
- Scale estimate: applying the measured three-genome 65.0% retention rate predicts about 305 million retained windows.
  Linear extrapolation from the five-million-window 148-k-mer run predicts approximately 770 GiB peak RSS and 2.7 hours of Linclust wall time at 16 vCPUs, before the 64-vCPU machine's additional parallelism.
- Outputs: the complete compressed assignment table, per-assembly filter receipts, staged manifest, resource reports, and bounded-memory cluster summary will use the separate `s3://oa-bolinas/snakemake/analysis/linclust_conservation/runs/panel20-linclust/` prefix.
- Alignment scope: the target performs Linclust's mandatory internal verification only and does not launch a genome-scale representative-to-all alignment.
- Validation: 56 project tests, the exact 55-job credential-free dry-run, scoped pre-commit checks, and the SkyPilot resource dry-run passed.
- Falsifier: the run exceeds the five-hour Linclust cutoff, exceeds the memory or disk bounds, cannot acquire the configured spot instance, or projects above $20 before completion.
- Command: `sky launch -y --down -c linclust-cons-panel20 snakemake/analysis/linclust_conservation/sky/panel20_linclust.yaml --env PIPELINE_COMMIT_SHA="$(git rev-parse HEAD)"`.
- Next action: push the snapshot, launch the bounded task, inspect the first completed staging and tiling jobs, then monitor the full clustering receipt and terminate any residual cloud state.
