---
topic: issue-478-conservation-repeat-predictability
issue: https://github.com/Open-Athena/marin-dna/issues/478
description: Conservation by repeat predictability across the 46M-4B scaling ladder
author: gonzalobenegas
---

# Conservation by Repeat Predictability: Research Logbook

## Current TL;DR

- Status: implementation and local gates complete; no new inference has run.
- Issue #296 already validated a per-token loss kernel and cached FWD loss for the eight scaling checkpoints on a CDS-centric Ensembl dataset. Issue #478 will port that kernel into an additive `evals_v2` workflow and add the validation-matched RefSeq repeat mask, reverse-complement averaging, three region families, controls, block uncertainty, and versioned artifacts.
- Cloud and data access are available for the pinned Hugging Face datasets, GCS checkpoints, cached issue #274 scores, and `GCF_000001405.40.2bit` on S3.
- Paid GPU runs are capped at $20. The 46M checkpoint is the first execution gate.
- All eight checkpoints use the same mixture: CDS 0.7319, upstream 0.2062, downstream 0.0619; uppercase weight 1.0 and lowercase weight 0.01.
- Codon position and canonical two-base splice donor/acceptor status are preregistered secondary diagnostics for the CDS dataset only.

## Scope

- Goal: decide whether absolute loss, 46M predictive entropy, or loss reduction with model scale should advance to a fixed-compute training-weight experiment.
- Primary metrics: per-base NLL; 46M predictive entropy; 46M-to-4B and adjacent-rung loss reduction; fraction of positive loss reductions; block-bootstrap uncertainty for the issue #478 contrasts.
- Primary strata: region x validation-case conservation x RefSeq repeat status.
- Controls: GC content, target position, and a 7-mer predictability baseline. Homology or corpus-exposure density will be included only if pinned compatible metadata is available.
- Constraints: 0-based half-open coordinates internally; exact validation-matched RefSeq assembly and sequence names; additive S3 namespace; no local workload with an estimated working set above 500 MiB; $20 paid-GPU limit; smallest model first.
- Coordinating issue: https://github.com/Open-Athena/marin-dna/issues/478
- Branch: `codex/issue-478-conservation-repeat-predictability`
- Experiment prefix: `CRP`
- Shared tags: `CRP`, `issue-478`, `evals-v2`

## Baseline

- Date: 2026-08-19
- Code ref: `330bd32a8812a81fb561d3f3d1d85aba8d3e4a5b`
- Issue #274 cached scores: `s3://oa-bolinas/snakemake/analysis/evals_v2/results/ll_gap/scores/`
- Issue #296 durable artifacts: `s3://oa-bolinas/analysis/issue296/`
- RefSeq soft-mask source: `s3://oa-bolinas/snakemake/training_dataset/dataset_creation/results/genome/GCF_000001405.40.2bit`
- Baseline numbers: no issue #478 result yet. The forward-strand regression target is the issue #274 per-window sums and counts for each model x region cell.

## Hypothesis Queue

### Active

- `CRP-001`: repeat bases have lower 46M NLL and smaller improvement with scale than nonrepeat bases. Next test: 46M joined-artifact and scoring pilot, followed by the endpoint comparison if all gates pass.
- `CRP-002`: among nonrepeat bases, validation-case conserved bases have larger 46M-to-4B loss reduction after the prespecified controls. Next test: full-ladder scoring after the 46M pilot.
- `CRP-003`: CDS first and second codon positions are more predictable than third positions when an unambiguous reading frame is available. Next test: reuse the issue #296 phase-aware annotation logic on the joined CDS artifact.
- `CRP-004`: canonical 2-bp CDS-flank donor/acceptor sites show strand/context-specific predictability. Next test: CDS-only pilot annotation counts, then endpoint comparison.

### Blocked

- None.

### Falsified / Dead End

- None.

### Promoted

- None.

## Decision Log

- 2026-08-19: use the three pinned validation datasets and revisions from issue #478 and `evals_v2/config/config.yaml`.
- 2026-08-19: use the matching RefSeq `GCF_000001405.40` soft-mask asset. Do not substitute the canonical Ensembl release 115 reference used by other evals.
- 2026-08-19: extend `evals_v2` additively under an issue-specific target and S3 namespace. Preserve the existing issue #274 `ll_gap` rules and artifacts.
- 2026-08-19: port the validated issue #296 per-token kernel into the owning `evals_v2` package instead of designing a second kernel.
- 2026-08-19: run the 46M checkpoint first and stop on schema, coordinate, strand-mapping, or #274 regression failure.
- 2026-08-19: keep total paid GPU spend at or below $20.
- 2026-08-19: keep codon and splice feature strata secondary and CDS-only; do not apply them to upstream or downstream windows.
- 2026-08-19: use full-vocabulary true-base NLL for #274 parity and nucleotide-renormalized entropy for the 46M predictability diagnostic.

## Background Research Brief

- Effort: medium. Searched Marin issues, branches, artifacts, run configs, and primary literature on reducible loss, conservation weighting, repeats, and homology leakage.
- Prior Marin evidence: #296 validated the token kernel and found codon-position ordering plus splice-site strand asymmetry on a separate Ensembl CDS-centric set. #177 pinned training repeat treatment to the source soft mask with lowercase loss weight 0.01. #274 supplies the exact FWD regression cache.
- External evidence: Rho-1 defines reducible loss against a reference model trained on a curated target distribution, so same-corpus 46M-to-4B loss reduction is a scale-sensitivity signal, not Rho-1 reducible loss. GPN/GPN-MSA support evaluating conservation/repeat weighting and warn that repeat behavior is heterogeneous. Homology-leakage work motivates an explicit exposure-metadata limitation.
- Negative result: the pinned HF validation datasets expose only `id` and mixed-case `seq`; no compatible per-window exposure or homology-density field was found.
- Source ledger: #274, #296, #177; Rho-1 (arXiv:2404.07965); GPN-MSA (PMCID: PMC10592768); GPN (PMCID: PMC10622914); homology-leakage preprint (bioRxiv 2025.01.22.634321); Dark Regulome preprint (arXiv:2606.06834, exploratory only).

## Negative Results Index

- None.

## Entry Log

### 2026-08-19 22:25 UTC - CRP-001 prologue and access audit

- Hypothesis: the existing issue #296 per-token scorer and issue #274 pipeline can support the issue #478 experiment through an additive extension.
- Commit Hash: `330bd32a8812a81fb561d3f3d1d85aba8d3e4a5b` (starting point)
- Command: GitHub issue and prior-work reads; targeted `rg`; read-only AWS, GCS, Hugging Face, SkyPilot, and GitHub access checks.
- Config: issue #478 pinned datasets and eight scaling checkpoints; RefSeq `GCF_000001405.40`; 46M-first; $20 paid-GPU cap.
- Result: all required stores and launch credentials are reachable. Issue #296 contains a validated per-token loss kernel and full-ladder FWD caches, but it excludes reverse-complement scoring, repeats, two region families, and `evals_v2` integration.
- Interpretation: implementation can reuse prior tested logic. The first new remote run should validate the joined repeat artifact and FWD/RC per-base mapping on 46M before launching larger checkpoints.
- Next action: finish the medium prior-work brief, inspect and port the issue #296 kernel, and define the additive `evals_v2` artifact contracts.

### 2026-08-19 23:04 UTC - CRP-001 implementation and local gates

- Hypothesis: an additive `evals_v2` path can preserve exact per-base genomic alignment while reproducing issue #274.
- Commit Hash: working tree on `codex/issue-478-conservation-repeat-predictability`.
- Command: focused `uv run --locked pytest`; `snakemake -n predictability_478_pilot`; `snakemake -n predictability_478`.
- Config: version `v1`; RefSeq `GCF_000001405.40`; central `[32,223)`; 10-Mb genomic blocks; 1,000 bootstrap replicates; CDS-only codon and canonical 2-bp splice diagnostics.
- Result: 21 focused tests pass. Both the 46M pilot and full-ladder DAGs resolve through the S3 storage provider. No model inference has run.
- Interpretation: local schema, BOS, RC reversal, annotation ambiguity, exact-sequence, 7-mer, regression-comparison, score-direction, bootstrap, and control-model gates are ready for the remote pilot.
- Next action: run the full evals_v2 test gate and launch only `predictability_478_pilot`.
