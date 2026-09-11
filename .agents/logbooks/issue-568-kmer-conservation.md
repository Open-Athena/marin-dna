---
topic: issue-568-kmer-conservation
issue: https://github.com/Open-Athena/marin-dna/issues/568
description: Window geometry, complete k-mer signal, sketching, and indexing for local genomic retrieval.
author: gbenegas
---

# Local k-mer conservation retrieval: task logbook

## Scope

- Experiment series: `KMER-CONS`.
- Goal: establish the exact local k-mer signal before measuring additional sketch and index losses.
- Primary metric: known locus-pair recall at unique candidate budgets 1, 10, and 100.
- Constraints: user-authorized $30 EC2 ceiling; fixed genomic coverage; component-level development/held-out split; no genome-scale production or model-training claim.
- Permanent branch: `codex/issue-568-kmer-conservation`.
- Issue: [#568](https://github.com/Open-Athena/marin-dna/issues/568).

## Hypothesis queue

1. `KMER-CONS-001`: complete local canonical k-mer sets retain projected homology better with explicit window geometry.
   The development screen and synthetic planted tracts test this first.
2. `KMER-CONS-002`: independent-permutation MinHash preserves this signal with lower search/storage cost.
   Test only surviving representations; compare scans separately from compatible banded LSH.
3. `KMER-CONS-003`: indexed retrieval improves the held-out recall–cost frontier against matched Linclust and exact inverted-index baselines.
4. `KMER-CONS-004`: a surviving local score enriches independent conservation annotations after composition/repeat and species-redundancy controls.
   Conditional on the retrieval gate; projected-center recovery alone cannot establish this claim.

## Background research brief

- Effort: low, targeted follow-up to the issue's existing prior-work synthesis.
- Date: 2026-09-11.
- Stop rule: stop when the pinned fixture and representation/index choices are clear.

The internal control is #521's pinned workflow at `e02d1637dc41f886ffdc5dd071228314f2a58631`.
It selects common projected query names deterministically, rejects non-255 bp, ambiguous, and majority-lowercase source sequences, and writes exactly 128 three-species anchors for its bounded controls.
The source projection tables and underlying three 2bit genomes still exist with matching ETags; the deleted objects were the experiment's later outputs.
The original tables use `hg38`, `GCF_000001635.26`, and `GCF_000208655.1`; no reference-build substitution is appropriate for regenerating this fixture.

The strongest historical Linclust result was 54.9% pair recall in five million windows, but it is not a matched baseline for changed contexts or backgrounds.
The 71.9% no-prefilter control and the 255-to-511 bp recall loss motivate measuring full sets before choosing a sketch size.
MinCNE is direct code prior art for local windows, per-feature hash minima, buckets, and verification; it does not establish whole-genome sensitivity for this fixture.
MashMap3/minmers concerns unbiased local Jaccard estimation in a long-sequence regime; short windows need their own sensitivity measurement.
The datasketch 2.0 documentation explicitly distinguishes independent permutation signatures, Jaccard estimation, and LSH compatibility, including a change from legacy hashing.
The experiment locks the actual installed version and must not apply ordinary banded LSH to a bottom-k sketch.

| Source | Type | Claim used for |
| --- | --- | --- |
| [#521 interpretation](https://github.com/Open-Athena/marin-dna/blob/922e41149c8f9bdb3130f780f66163a0241ff045/docs/research/experiments/521-linclust-conservation.md) | Marin research | Historical recovery limits and separate scaling failure |
| [Pinned fixture](https://github.com/Open-Athena/marin-dna/blob/e02d1637dc41f886ffdc5dd071228314f2a58631/snakemake/analysis/linclust_conservation/src/marin_dna_linclust_conservation/homology_fixture.py) | Marin code | Exact control selection and center expansion |
| [MinCNE](https://github.com/srbehera/MinCNE/blob/c376342a7ffa429801e9afe617428b55c6060117/MinCNE.cpp) | External code | Local sketch construction and bucketing |
| [Minmers](https://pubmed.ncbi.nlm.nih.gov/37603771/) | Paper | Local sketch estimation; transfer to short elements is unproven |
| [datasketch MinHash](https://ekzhu.com/datasketch/minhash.html) | Official docs | Estimator and permutation compatibility |
| [datasketch LSH](https://ekzhu.com/datasketch/lsh.html) | Official docs | Compatible banded candidate retrieval |

## Entry log

### 2026-09-11 20:57 UTC — KMER-CONS-001 protocol and worker

- Baseline commit: `922e41149c8f9bdb3130f780f66163a0241ff045`.
- User authority: work on #568 with up to $30 in EC2 costs.
- Protocol: `experiments/kmer_conservation/README.md` and `config/screen.json`, written before any retrieval metrics were generated.
- Worker: `i-060389732f44b3395`, `c7i.4xlarge`, 16 vCPUs/32 GiB, `us-east-2a`, launched `2026-09-11T20:57:10Z`.
- Cost estimate: $0.714/hour, approximately $7.14 at ten hours; 60 GiB gp3 plus public IPv4 add less than $0.20 over that period.
  The AWS pricing API was denied; the estimate uses the published instance price and will be checked against the AWS public price list on the worker.
- Cost enforcement: instance-initiated shutdown behavior `terminate`; `shutdown -h +600` scheduled during user-data bootstrap; encrypted root EBS `DeleteOnTermination=true`.
  The timer and termination behavior were queried and confirmed.
- Resources: estimated analysis peak below 16 GiB on EC2; no full-data work on the shared local VM.
- Storage owner: `s3://oa-bolinas/issues/568/`, versioned per producing snapshot.
- Validation: initial 19 genomic/metric contract tests passed remotely.
- Initial command: `/usr/bin/time -v uv run --locked kmer-fixture --root /data/issue568 --prior /data/issue568/prior` from the independent experiment project.
- Gate fixed before inspecting outcomes: at least 80% development recall at C=10 and at most 10% injected-decoy fraction among returned top-ten candidates.
  This is an exploratory operational criterion, not a universal detection threshold.
- Next action: validate the regenerated controls, measure exact full-set scores, and audit the protocol before the held-out read.

### 2026-09-11 21:28 UTC — KMER-CONS-001 exact-set screen and audited fixture

- Producing snapshot: `65f6f36b` for the fixture, full-set screen, and synthetic controls.
- The first extraction failed on redundant manual soft-mask handling; py2bit 1.0.1 already preserves lowercase with `storeMasked=True`.
  The corrected extraction completed in 73 seconds at about 2.5 GiB peak RSS.
- The first split audit exposed shared hash domains between selecting additional anchors and assigning components to splits: 140/152 components fell in held-out data.
  No retrieval metrics had been generated.
  A separate `split:` hash domain fixed this; the final fixture has 87 development and 65 held-out components.
  A regression test covers sampling/split independence.
- Every one of the 768 extracted center-255 sequences equals its source projection sequence or reverse complement.
  The original 128-anchor FASTA SHA-256 is `30f823f5b2943c520d3b72d7bf754151a30f8ce4d6fb429647ba288eeebce996`.
  The original truth TSV SHA-256 is `868fd4a9bd67e61363c4f4897063558b5c1a30d992d6eafd88e46a720ffc4676`.
- Fixed inputs: 768 anchor contexts, 3,000 random genomic contexts, 768 composition/repeat/complexity-matched genomic contexts, 192 repeat-rich challenges, and 768 injected shuffled decoys.
  All contexts are 4,096 bp.
  Merging homologs and overlapping contexts turns 256 anchor identities into 152 locus components.
- Command: `/usr/bin/time -v uv run --locked kmer-screen --root /data/issue568 --split dev`, from the independent experiment project, with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=2`.
- Result: all 20 width/k settings completed in 6m21s, with per-setting peaks below 1.2 GiB.
  Exact W255/k9/half-stride recovered 256/261 development locus pairs at C=10 (98.08%); W1024/k13 recovered 248/261 (95.02%).
  Their query-scoring times were 13.25 and 1.38 seconds, respectively.
- Independent review fixed per-arm RSS accounting (one process per setting) and the missing pre-truncation unique-candidate-locus count before the screen.
- Decision: advance these two representations into separate 32/128/512-permutation MinHash scans and LSH with 1/2/4 rows per band.
  Record cold construction plus query cost, hot query cost, storage, and raw candidate work separately.
  A footprint reduction alone must not be described as a demonstrated runtime improvement.
- Follow-up command: `bash run_followups.sh` runs the 72-planted-locus screen and development-only full/half/quarter strides, masking, and whole-context scoring.
- Baseline/diagnostic implementation: `7913f09b`; 25 tests pass.
  `bash run_methods.sh` executes matched Linclust 18-8cc5c and the separate sketch/index matrix.
  `python -m kmer_conservation.diagnostics --root /data/issue568 --split dev --mode union` combines the two representations with reciprocal rank fusion and one final candidate budget.
  `--mode verify --width 255 --k 9` and `--width 1024 --k 13` apply a separate full-window edit-distance gate with no post-verification backfill.
- Scientific limit: the target background is independently sampled and generally does not contain known orthologs of the query background.
  It can measure known-locus retrieval and collisions, but would confound a selector-enrichment test if injected homology availability were treated as an unbiased genome-wide universe.
  No selector or evolutionary-constraint claim follows from these recall values.
- AWS's streaming public regional price list confirms `$0.7140000000/Hrs`, effective 2026-09-01, for Linux/shared/on-demand `c7i.4xlarge` in Ohio.

### 2026-09-11 21:41 UTC — physical-locus audit supersedes preliminary group metrics

- A further audit found six homology-linked split groups with disjoint genomic intervals in at least one species (13 group/species combinations).
  Collapsing them for candidate budgets can combine separate genomic loci.
  The initial figures above and in the issue comments are superseded for the issue's physical-locus metric.
- Stopped the running MinHash comparison before any held-out query was scored.
- Correction: retain homology-linked components for split isolation; construct query and candidate loci from within-species, same-chromosome interval overlap only.
  Evaluate all known source-locus/target-locus pairs using original anchor identities, and count query work once even when it has several known targets.
  Candidate retrieval itself remains unrestricted across the full target-species universe.
- Version two preserves every sequence, coordinate interval, record ID, input order, and development/held-out assignment; an assertion checks this during relabeling.
  Its compressed contexts SHA-256 is `ce25b2812841bc0c09cfb7ebec47a2d08168c120422a45e9827b4f292121e90b`.
  It contains 157 human, 156 mouse, and 160 armadillo physical anchor loci while retaining 152 split groups (87 development / 65 held-out).
  All 768 center-sequence checks still pass; every physical anchor locus is now a connected genomic interval in one species.
- Tests now cover a source locus with two disjoint homologous targets: recall at C=1 must be 1/2, not 1, and recall at C=2 must be 1.
  The 25-test suite passes.
- Independent review also corrected union cost accounting: carry both constituent retrieval costs, feature/index preparation, and raw postings separately from list-fusion work.
  Verifier resource totals explicitly cover C=100, with no claim that these are C=1 or C=10 costs.
- Next: rerun all affected development comparisons on version two, choose from that corrected matrix, then freeze and score held-out data.
