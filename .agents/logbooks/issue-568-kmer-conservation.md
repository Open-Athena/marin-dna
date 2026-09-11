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
