---
topic: issue-489-likelihood-dynamics
issue: https://github.com/Open-Athena/marin-dna/issues/489
description: Track likelihood-derived token rankings through the m1 to m1.3 training lineage.
author: Codex
---

# Issue 489 likelihood dynamics: Task Logbook

## Scope

- Goal: Measure how per-token forward NLL, four-nucleotide entropy, conservation ranking, and easy-token membership change through five checkpoints on the m1 to m1.3 lineage.
- Primary metrics: Conservation AUPRC, low/high trajectory classes, lowest-loss 10% Jaccard, and subsequent loss reduction by current-loss decile.
- Constraints: Use the five complete validation probes released with the MarinDNA blog, score one forward orientation, retain unfiltered per-token atoms, and keep the primary nonrepeat central span at `[32, 223)`.
- Coordinating issue: https://github.com/Open-Athena/marin-dna/issues/489
- Experiment prefix: `LD489`
- Shared tags: `LD489`, `issue-489`, `m1-lineage`

## Current TL;DR

- The MarinDNA blog Hugging Face collection identifies the five intended training-validation probes.
- The manifest is the three issue #478 probes plus the Zoonomia enhancer and ncRNA probes.
- The first three probes use RefSeq `GCF_000001405.40` coordinates.
- The two Zoonomia probes use bare Ensembl release 115 GRCh38 sequence names and 0-based half-open coordinates.
- The scoring workflow must keep these reference joins separate and assert sequence identity before deriving repeat labels.
- The versioned metadata, forward atom cache, and earliest-versus-terminal pilot are implemented.
- Eight focused tests pass.
- The corrected Snakemake dry-run is pending on remote compute because the local parser measured about 1.0 GiB RSS.
- No scoring run has launched.

## Baseline

- Date: 2026-08-21
- Code refs: issue #478 permanent branch `79db49cedcb9c8bb9c45bd35d7d6f8fd078e91b2`; blog lineage scaffold commit `40fcee94`.
- Baseline numbers: issue #478 scored 16,384 windows in each of CDS, upstream, and downstream over the central span `[32, 223)`.

## Decision Log

- 2026-08-21: Use the five validation probes listed in the blog collection.
- 2026-08-21: Pin every Hugging Face dataset to its immutable current revision.
- 2026-08-21: Treat the blog collection as the experiment manifest even though `val_ncrna` and `val_enhancer` are clean validation recipes rather than exact partitions of the v3 training shards.
- 2026-08-21: Preserve separate RefSeq and Ensembl reference assets and fail on any sequence mismatch.

## Hypothesis Queue

### Active

- `LD489-H1`: Conservation AUPRC increases along the m1 to m1.3 path, with the largest gains in the ncRNA and enhancer probes after the five-region mixture is learned.
- `LD489-H2`: A stable low-loss core emerges during training, so lowest-loss 10% Jaccard is higher for adjacent late checkpoints than early checkpoints and remains nontrivial from the first to terminal checkpoint.
- `LD489-H3`: Tokens in high current-loss deciles receive larger loss reductions at the next and terminal checkpoints than tokens already in low-loss deciles.
- `LD489-H4`: The checkpoint trend remains after stratifying by region, repeat status, GC, target position, and chromosome-held-out 7-mer NLL.

### Blocked

- None.

### Falsified / Dead End

- The five validation datasets are not the five `zoonomia-v1-val_*` recipes considered initially.
- The blog collection uses three `genomes-v5-validation-*` probes and only two `zoonomia-v1-val_*` probes.

### Promoted

- None.

## Background Research Brief

- Effort: Low.
- Stop rule: Stop after the blog collection, dataset cards, immutable Hub revisions, issue #478 inputs, and blog lineage configuration agree on the input manifest.
- Date: 2026-08-21.

### Question

Which five complete validation datasets should issue #489 score so the probes match the training data represented in the MarinDNA blog?

### Current Marin Context

Issue #489 specifies five regions and asks to reuse complete existing validation sets.
Issue #478 established the CDS, upstream, and downstream probes and the `[32, 223)` primary span.
The original m1.3 training definition names five training sources but only three validation sources, so the training script alone does not identify the requested five-probe evaluation panel.

### Internal Prior Work

- Issue #478 used the three `genomes-v5-validation-*` probes pinned below.
- The blog lineage configuration records the available on-path m1, m1.1, m1.2, and m1.3 checkpoints and their GCS sources.
- All four stages share 2,097,152 training tokens per optimizer step.

### External Prior Art

- The MarinDNA Hugging Face collection describes five training datasets and five training-validation probes.
- The collection explicitly pairs CDS, upstream, downstream, enhancer, and ncRNA labels with the five repositories pinned below.
- The Zoonomia dataset cards define 255 bp Ensembl release 115 human windows, conservation prefiltering, case-encoded phyloP labels, and no reverse-complement augmentation.

### Negative / Failed Leads

- The full seven-recipe `zoonomia-v1-val_*` family does not match the collection manifest.
- The v3 ncRNA and cCRE training repositories expose a training split and do not themselves provide the complete human validation probes requested by issue #489.

### Evidence Map

#### Claim: The blog collection is the five-panel manifest for issue #489

- Support: https://huggingface.co/collections/marin-dna/a-1b-standard-transformer-rivals-evo-2-40b-on-vep lists the five datasets below as training-validation probes.
- Contradictions: The original m1.3 training script configures only three validation datasets.
- Directness to Marin: Exact release collection for the blog and model lineage under study.
- Confidence: High.
- Action: Pin these five revisions in the #489 workflow.

#### Claim: The two validation families require different reference assets

- Support: The three genomes-v5 cards use RefSeq `GCF_000001405.40` identifiers, while the two Zoonomia cards use bare Ensembl release 115 sequence names.
- Contradictions: None found.
- Directness to Marin: Exact input cards and schemas.
- Confidence: High.
- Action: Join repeat masking separately and assert uppercase sequence equality for every window.

### Recommended Next Experiments

#### 1. Earliest-versus-terminal pilot

- Minimum experiment: Score a fixed small prefix of every validation probe at the earliest and terminal selected checkpoints.
- Baseline/control: Reconstruct aggregate forward loss from per-token NLL and compare finite counts, IDs, shapes, and case totals across checkpoints.
- Expected signal: Identical row and token identities with finite NLL and entropy for every scorable A/C/G/T target.
- Falsifier: Any row-order drift, coordinate mismatch, missing token, non-finite atom, or disagreement with the aggregate loss kernel beyond the runtime tolerance.
- Cost/risk: Two 1B checkpoint loads across five small panels on one paid GPU.
- Sources: Issue #489 and issue #478.

### Hypothesis Queue Update

- Add: `LD489-H1` through `LD489-H4`.
- Revise: None.
- Falsify / stop: Do not use a five-member Zoonomia validation family as the manifest.
- Promote: The blog collection manifest into workflow configuration.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
| --- | --- | --- | --- | --- | --- |
| Issue #489 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/489 | Scope, metrics, checkpoint count, output namespace | High | Current coordinating issue |
| Blog collection | Official release collection | https://huggingface.co/collections/marin-dna/a-1b-standard-transformer-rivals-evo-2-40b-on-vep | Five validation probes | High | Collection labels each probe |
| Issue #478 branch | Marin code | `79db49cedcb9c8bb9c45bd35d7d6f8fd078e91b2` | Three probe pins, per-token scorer, central span | High | Permanent research branch |
| Blog lineage scaffold | Marin code | `40fcee94` | Checkpoint paths and available steps | High | Pinned historical config |
| Zoonomia validation cards | Official dataset docs | Hugging Face dataset pages | Coordinate assembly, recipes, conservation encoding | High | Exact datasets in collection |

### Handoff

- Suggested issue `Prior work` block: The blog Hugging Face collection resolves the five-panel manifest as the three issue #478 probes plus `zoonomia-v1-val_enhancer` and `zoonomia-v1-val_ncrna`.
- Suggested logbook entry: This entry.
- Open questions: Whether the primary analysis should include chromosome Y from these unlabeled validation probes; the project held-out-label restriction does not apply to unlabeled reference and functional-genomics inputs.
- Stop reason: The exact input manifest and reference split are resolved.

## Input Manifest

| Region | Hugging Face repository | Revision | Rows | Coordinate/reference family |
| --- | --- | --- | ---: | --- |
| CDS | `marin-dna/genomes-v5-validation-intervals-v5_255_255` | `daff592f213aaa1cab1711d477a79ff6b1bc4ef4` | 16,384 | RefSeq `GCF_000001405.40` |
| upstream | `marin-dna/genomes-v5-validation-intervals-v1_255_255` | `a761bc0b663a9827303f3112e4667d53d5326fac` | 16,384 | RefSeq `GCF_000001405.40` |
| downstream | `marin-dna/genomes-v5-validation-intervals-v15_255_255` | `d7b27eecd68453934ebb3e7e6e78d5401789faa5` | 16,384 | RefSeq `GCF_000001405.40` |
| enhancer | `marin-dna/zoonomia-v1-val_enhancer` | `d40d1e067b2a56ac812af122de029eb79cab1106` | 16,384 | Ensembl release 115 GRCh38 |
| ncRNA | `marin-dna/zoonomia-v1-val_ncrna` | `76a18c1bbf07ac9bd064722431bbdab894b9e6c6` | 16,384 | Ensembl release 115 GRCh38 |

## Entry Log

### 2026-08-21 14:47 UTC - `LD489-001` resolve the validation manifest

- Hypothesis: The blog release collection identifies the five complete validation probes intended by issue #489.
- Commit Hash: Pending.
- Command: Queried the public Hugging Face collection and dataset APIs, inspected issue #478 at `79db49cedcb9c8bb9c45bd35d7d6f8fd078e91b2`, and inspected the historical lineage config at `40fcee94`.
- Config: Five 16,384-row, 255 bp validation probes at the revisions in the input manifest.
- Result: The collection lists the three issue #478 probes plus the enhancer and ncRNA Zoonomia probes.
- Interpretation: The collection is the authoritative experiment manifest.
- Next action: Add an isolated, versioned #489 scoring path and an earliest-versus-terminal pilot target.

### 2026-08-21 15:08 UTC - `LD489-002` scaffold the versioned atom cache and pilot

- Hypothesis: The issue #478 per-token kernel can be isolated for forward-only #489 scoring while preserving aggregate LL behavior.
- Commit Hash: Pending.
- Command: `uv run --locked pytest tests/test_likelihood_dynamics_489.py`.
- Config: Eight focused unit tests; shared-node thread limits; nonblocking `/tmp/marin-dna-local-heavy.lock`; `nice -n 10`; `ionice -c 2 -n 7`.
- Result: Eight tests passed in 11.89 seconds.
- Result: Peak RSS was 1,018,944 KiB, dominated by importing the locked Torch stack.
- Interpretation: The reference mismatch guard, independent conservation/repeat case, token alignment, four-nucleotide entropy, stable identity, provenance, and pilot comparison contracts pass.
- Interpretation: Even a focused evals_v2 test process exceeds the 500 MiB shared-node working-set ceiling.
- Next action: Do not repeat local pytest for this task.
- Next action: Run subsequent parser and dry-run checks on remote compute after approval.

### 2026-08-21 15:08 UTC - `LD489-003` first pilot dry-run

- Hypothesis: The additive pilot target resolves to ten scoring cells and five reusable metadata cells without changing `rule all`.
- Commit Hash: Pending.
- Command: `uv run --locked snakemake -n --profile workflow/profiles/default likelihood_dynamics_489_pilot`.
- Config: Shared-node thread limits; nonblocking heavy-work lock; no jobs executed because this was a dry-run.
- Result: Snakemake stopped during parsing because `checkpoint` is a reserved Snakefile keyword in a top-level loop.
- Result: Peak RSS was 1,014,996 KiB even though no DAG jobs ran.
- Interpretation: Rename the loop variable to `checkpoint_cfg`.
- Interpretation: The dry-run did not reach DAG construction, so remote input and job counts remain unverified.
- Next action: The reserved keyword is renamed.
- Next action: Do not repeat the memory-heavy parser locally.
- Next action: Run the corrected dry-run on the remote execution node before launching the pilot.
