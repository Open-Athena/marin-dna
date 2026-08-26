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
