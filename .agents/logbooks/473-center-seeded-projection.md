---
topic: center-seeded-vertebrate-projection
issue: https://github.com/Open-Athena/marin-dna/issues/473
description: Compare center-seeded and full-window vertebrate projection policies.
author: gonzalobenegas
---

# Center-seeded vertebrate projection: Research Logbook

## Current TL;DR

The additive, pilot-free producer completed one bundled HAL call per species
and one MAF scan per chromosome for `full_window` versus `center_1`. Exact
fixed-grid accounting shows that center-1 recovers more pairs in CDS, ncRNA
exon, and UTR3, but fewer in enhancer-centered cCRE and TSS/UTR5. The
universal recovery hypothesis is therefore falsified. Sampled exact named-PSL
coverage intervals include zero in all five regions, while chromosome and
strand agreement are 1.0 and median emitted-center displacement is 3 bp.

The three new datasets are public, ungated, and revision-pinned on Hugging
Face. The exact #417 CDS full-window trajectory is reused; CDS center-1,
enhancer full-window, and enhancer center-1 are training on Iris at matched
tokens. The isolated development-only evaluator passed its pinned-runtime,
one-file loader, score, and official-metric gates and is incrementally scoring
durable checkpoints. An earlier builder-based attempt materialized the held-out
file before failing prior to inference; CSP-053 records that incident. The
retired attempt produced no prediction or metric, and the active evaluator
loads only the permitted development file.

## Scope

- Goal: compare `full_window` and `center_1` on fixed anchors, then use matched
  CDS and enhancer-centered cCRE training to make a region-specific decision.
  Wider landmark widths are deferred to a separate follow-up.
- Primary projection metrics: accepted recovery, aligned coverage, unaligned
  flank, ambiguity and rejection reasons, and agreement between policy loci.
- Primary training metric: development-split VEP AUPRC at matched tokens.
- Constraints: internal coordinates are 0-based and half-open; active
  evaluation loads only permitted development labels and produces no held-out
  predictions or aggregates; existing S3-backed Snakemake rules and output
  paths remain unchanged.
- Coordinating issue: https://github.com/Open-Athena/marin-dna/issues/473
- Branches: draft implementation PR #477 and permanent experiment branch
  `codex/exp473-center-seeded-projection-training`.
- Experiment IDs: `CSP-001`, `CSP-002`, ...
- Shared tags: `CSP`, `issue-473`, `projection-policy`

## Current Baseline

- Producing issue: https://github.com/Open-Athena/marin-dna/issues/417
- Maintained implementation: commit
  [`1c360e19`](https://github.com/Open-Athena/marin-dna/commit/1c360e19490e360326de0e25325818d5900a0bb3)
- Source anchors: 1,136,854 conservation-filtered 255 bp human windows.
- Projection targets: 107 Zoonomia mammals and 28 MultiZ non-mammals.
- Full-window accounting: 121,843,701 accepted (79.39%), 13,474,015
  explicitly rejected (8.78%), and 18,157,574 with no mapping (11.83%) over
  153,475,290 requested anchor-target pairs.
- Contract: project the full 255 bp source interval, require one compatible
  target chromosome and strand without overlap, accept a 128--512 bp outer
  target span, and resize its midpoint to 255 bp.

## Hypothesis Queue

### Active

- `CSP-H4`: the downstream policy effect differs between CDS and
  enhancer-centered cCRE specialists. Current test: the matched-token training
  trajectories and paired development/intersection evaluations now running.

### Blocked

- None. The remaining critical path is authorized remote training and
  development-only evaluation compute.

### Falsified / Dead End

- `CSP-H1`: center-1 is not a universal recovery improvement and does not have
  its largest gain in enhancer-centered cCREs. It wins CDS, ncRNA exon, and
  UTR3, but loses enhancer-centered cCRE and TSS/UTR5.

### Inconclusive / Deferred

- `CSP-H2`: sampled exact named-PSL coverage does not establish a full-window
  advantage; paired 95% anchor-bootstrap intervals include zero in all five
  regions.
- `CSP-H3`: the wider-width frontier was explicitly deferred. No 17, 33, 65,
  or 129 bp policy is produced, evaluated, or selected in this issue.

### Promoted

- The additive bundled full-scale producer, fixed-grid QC, sampled trace, and
  three public revision-pinned datasets passed their preregistered gates and
  became the inputs to the matched training comparison.

## Decision Log

- 2026-08-19: Keep source-anchor coordinates separate from backend request
  coordinates. Anchor identity, region label, and chromosome split continue to
  use the original 255 bp interval.
- 2026-08-19: Preserve the `full_window` 128--512 bp gate exactly. Centered
  landmarks use integer gates `ceil(width / 2)` through `2 * width`.
- 2026-08-19: Do not launch alignment, publication, or training compute before
  the policy DAG and smoke tests are reviewed.
- 2026-08-19: Treat the existing S3-backed pipeline as immutable. Implement
  #473 with new rule, module, test, documentation, target, and output paths;
  copy code when policy assumptions cannot be isolated by composition.

## Negative Results Index

- The baseline projection contract coupled the pre-resize target-span gate to
  the 255 bp emitted sequence length. This rejected the valid `center_1` gate
  before classification. `CSP-001` separates those invariants; the emitted
  target remains 255 bp while a mapped landmark may span 1--2 bp.

## Background Research Brief

- Effort: medium
- Stop rule: stop when internal and external sources no longer change the
  ranked first implementation and experiment steps.
- Date: 2026-08-19

### Question

Can a short central landmark improve distant-species recovery without moving
the emitted 255 bp target window away from the homologous human locus, and does
any recovery gain improve region-specialist training?

### Current Marin Context

Issue #417 established a backend-uniform full-window contract and exact
accepted, rejected, and no-mapping accounting. Issue #473 changes the source
interval submitted to each backend while holding the human anchors, target
cohort, assemblies, and downstream training recipe fixed.

### Internal Prior Work

- [Issue #153](https://github.com/Open-Athena/marin-dna/issues/153) found that
  HAL and MAF produced nearly identical full-window projections at benchmark
  scale while HAL was about 400 times faster. It selected a backend; it did not
  compare source-landmark policies.
- [Issue #417](https://github.com/Open-Athena/marin-dna/issues/417) built the
  current vertebrate dataset and showed that non-mammalian CDS rows improved
  development-split Mendelian missense AUPRC by 0.0344 and SGE missense AUPRC
  by 0.0118 at step 4,999. It supplies the operational baseline and training
  recipe.
- [Issue #351](https://github.com/Open-Athena/marin-dna/issues/351) found a
  suggestive distal VEP advantage for enhancer-centered human anchors, 0.366
  versus 0.308 AUPRC. The centered arm saw about 9.7 epochs versus 3.7, so the
  result does not identify the projection policy effect.
- [Issue #353](https://github.com/Open-Athena/marin-dna/issues/353) showed that
  human-CDS nucleotide projection loses recovery with evolutionary distance
  and that stronger self-supervised CDS loss did not produce uniform VEP gains.
  Projection yield and language-model loss are therefore insufficient decision
  criteria.
- The canonical [genomic-anchor research question](https://github.com/Open-Athena/marin-dna/blob/d83046c85b0d17d5a8e5ea3d8a07947a8f60abd9/docs/research/questions/genomic-anchors.md)
  identifies the matched projection-policy comparison as the next decision
  gate.

### External Prior Art

- The [HAL format paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3654707/)
  defines coordinate mapping over a hierarchical alignment graph that includes
  rearrangements and paralogy relationships.
- The [HAL tool documentation](https://github.com/ComparativeGenomicsToolkit/hal)
  says `halLiftover` maps BED intervals base by base and follows paralogy
  relations. A one-base result is evidence for that mapped base, not for the
  homology of the surrounding extracted window.
- The [UCSC MAF specification](https://genome.ucsc.edu/FAQ/FAQformat) defines
  starts as 0-based and measures negative-strand starts on the reverse-
  complemented source. The existing adapter's conversion to forward 0-based,
  half-open coordinates remains required for every landmark width.
- [HALPER](https://pmc.ncbi.nlm.nih.gov/articles/PMC7520040/) uses mapped focal
  positions such as peak summits to select and extend regulatory-element
  orthologs. This supports testing a center landmark, while HALPER's neighboring
  fragment extension means it is not equivalent to extracting a fixed 255 bp
  window from one mapped base.
- A 2025 study of [positionally conserved regulatory elements](https://pmc.ncbi.nlm.nih.gov/articles/PMC12165850/)
  found that anchor-point methods can recover regulatory candidates missed by
  sequence liftover at long evolutionary distances. Those candidates require
  separate functional or positional evidence; higher yield does not establish
  surrounding-window homology.

### Negative / Failed Leads

- No prior Marin experiment directly compares `center_1` with `full_window` on
  the same anchors, species, backends, and matched-token training recipe.
- The #351 centered-enhancer result changes human anchor geometry and training
  epochs. It cannot answer the source-landmark projection question.
- The #153 HAL-versus-MAF result cannot select a projection policy because both
  arms used the same full-window acceptance and resize semantics.

### Evidence Map

#### Claim: source anchors and projection landmarks need separate coordinates

- Support: #473 fixes human anchor identity and asks each policy to submit a
  different source subinterval. HAL BED and MAF intersection operate on the
  submitted interval; split assignment and paired analysis operate on the
  original human anchor.
- Contradictions: none found.
- Directness to Marin: exact implementation boundary in the maintained #417
  pipeline.
- Confidence: high.
- Action: implemented and tested in `CSP-001`.

#### Claim: center-seeded recovery needs an alignment-evidence counterweight

- Support: HAL maps base by base, HALPER protects focal positions while adding
  neighboring fragments, and positionally conserved regulatory candidates can
  lack direct sequence alignment over the full element.
- Contradictions: a unique one-base mapping may still identify the correct
  syntenic locus, especially in conserved CDS. The surrounding 255 bp quality
  must be measured rather than assumed.
- Directness to Marin: high for coordinate semantics; medium for downstream
  sequence quality.
- Confidence: medium.
- Action: report aligned coverage, left/right unaligned flank, locus agreement,
  and downstream endpoints in addition to recovery.

### Recommended Next Experiments

#### 1. Sampled landmark frontier

- Minimum experiment: up to 10,000 anchors per region, stratified by source
  chromosome and conservation-score quantile, projected with widths 1, 17, 33,
  65, 129, and the 255 bp baseline.
- Baseline/control: immutable #417 `full_window` outputs where anchor catalogs
  match; a newly built full-window arm for enhancer-centered anchors.
- Expected signal: recovery weakly decreases and aligned evidence weakly
  increases with width.
- Falsifier: strong non-monotonicity after paired backend and species checks.
- Cost/risk: one batched HAL and MAF pass if backend interfaces permit; no model
  training.

#### 2. Full-scale `center_1` projection

- Minimum experiment: the five fixed region catalogs from #473 across the
  existing 107 + 28 target cohort, with exact requested-grid accounting.
- Baseline/control: #417 full-window outputs plus the enhancer-centered
  full-window baseline.
- Expected signal: higher distant-clade recovery with lower aligned coverage.
- Falsifier: coordinate, strand, central-index, bounds, or manual-inspection
  failure; lower recovery in a region/backend without an explained acceptance
  difference.
- Cost/risk: staged 1.26 TB HAL and 74.7 GB MultiZ mirror; explicit approval
  required.

#### 3. Matched-token CDS and enhancer training

- Minimum experiment: four 0.25B arms at 5,000 x 8,192 tokens, seed 0, evaluated
  only on permitted `evals_v2` development rows.
- Baseline/control: region-matched `full_window` specialists.
- Expected signal: region-dependent AUPRC delta; recovery alone does not select
  the policy.
- Falsifier: no home-domain gain or an offsetting regression.
- Cost/risk: paid training and evaluation; explicit approval required.

### Hypothesis Queue Update

- Add: none beyond the four preregistered #473 hypotheses.
- Revise: make aligned-window evidence an explicit counterweight to recovery in
  every policy comparison.
- Falsify / stop: do not use #351 or #153 as a direct policy result.
- Promote: implement the coordinate-separation seam before adding the policy
  DAG.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| #473 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/473 | Experiment contract | High | Current coordinating issue |
| #417 | GitHub issue and Marin code | https://github.com/Open-Athena/marin-dna/issues/417 | Baseline policy, accounting, training result | High | Exact preserved artifacts |
| #153 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/153 | Backend comparison | High | Same full-window policy in both arms |
| #351 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/351 | Enhancer-centering precedent | Medium | Epoch-confounded |
| #353 | GitHub issue | https://github.com/Open-Athena/marin-dna/issues/353 | Projection reach and endpoint disagreement | Medium | Different mmseqs2 method |
| Hickey et al. 2013 | Paper | https://pmc.ncbi.nlm.nih.gov/articles/PMC3654707/ | HAL mapping model | High | Primary HAL paper |
| HAL documentation | External code/docs | https://github.com/ComparativeGenomicsToolkit/hal | Base-wise liftover semantics | High | Upstream repository |
| UCSC format FAQ | Official docs | https://genome.ucsc.edu/FAQ/FAQformat | MAF coordinate conversion | High | Format authority |
| HALPER | Paper | https://pmc.ncbi.nlm.nih.gov/articles/PMC7520040/ | Focal-position ortholog construction | Medium | Related, not identical policy |
| Ma et al. 2025 | Paper | https://pmc.ncbi.nlm.nih.gov/articles/PMC12165850/ | Positional recovery beyond direct alignment | Medium | Regulatory context |

### Handoff

- Suggested issue `Prior work` block: the current issue already records the
  relevant internal work. Add the policy-foundation commit after it is pushed.
- Open questions: exact deterministic quantile sampler, producer-keyed policy
  paths, and whether one HAL invocation can batch all landmark widths without
  losing policy attribution.
- Stop reason: internal and external sources converged on the same first step:
  separate source anchors from backend request intervals and test all coordinate
  boundaries before staging alignment inputs.

## Entry Log

### 2026-08-19 21:18 UTC - CSP-001 policy-coordinate foundation

- Hypothesis: backend request coordinates can vary by policy without changing
  the original anchor identity or the exact `full_window` baseline behavior.
- Commit Hash: `ef8d069ec8e22069191772fe12a6ea8e7caa2399`
- Commands:
  - `uv run --locked pytest`
  - `uv run --locked snakemake -n --profile workflow/profiles/default --default-storage-provider none`
  - `uv run --locked pre-commit run --files <seven changed project files>`
- Config: smoke tier; fixed 255 bp emitted target length; `full_window` gate
  128--512; centered widths 1, 17, 33, 65, and 129 with gates
  `ceil(width / 2)`--`2 * width`.
- Result: 94 project tests passed. The credential-free dry-run resolved the
  existing 78-job smoke DAG. Synthetic MAF tests place the mapped `center_1`
  nucleotide at human-oriented index 127 on positive and negative target
  strands. No projection job ran.
- Interpretation: the maintained adapters can now express the #473 landmark
  policies without overloading source-anchor coordinates. The policy dimension
  is not yet represented in Snakemake output paths or QC tables.
- Next action: add producer-keyed policy request artifacts and the deterministic
  landmark sample, then wire policy-specific HAL/MAF smoke targets.

### 2026-08-19 21:40 UTC - CSP-001A additive correction and CSP-002 request preparation

- Hypothesis: #473 can prepare exact, reviewable policy requests and a
  representative wider pilot without changing any established #417 rule or
  shared execution path.
- Commit Hash: `db3ecae258dfef4f963a67308fa2f1bceba4608d`
- Superseded implementation: commit `ef8d069ec8e22069191772fe12a6ea8e7caa2399`
  changed shared helpers and is reverted by `dcf4af97`. Its coordinate
  conclusions remain covered by the isolated implementation, but it must not
  be used as the experiment recipe.
- Additive boundary:
  - implementation: `src/marin_dna_vertebrate_projection/issue_473/`
  - rules: `workflow/rules/issue_473.smk`
  - runbook: `experiments/issue_473/README.md`
  - outputs: producer-keyed `results/.../<tier>/experiments/473/`
  - legacy workflow change: one include line only; no established rule,
    helper, configuration key, default target, or output contract changed.
- Sampler: up to 10,000 anchors per canonical functional region; five
  equal-count region-level conservation quantiles; coverage-first water
  filling across every observed source-chromosome by quantile stratum; SHA-256
  ordering with seed 473 within strata.
- Commands:
  - `uv run --locked pytest tests/test_issue_473_policy.py tests/test_issue_473_projection.py tests/test_issue_473_pilot.py -q`
  - `uv run --locked pytest`
  - `uv run --locked pre-commit run --files <issue #473 files>`
  - `uv run --locked snakemake -n --profile workflow/profiles/default --default-storage-provider none issue_473_request_artifacts`
  - `uv run --locked snakemake -n --profile workflow/profiles/default --default-storage-provider none issue_473_request_artifacts --config tier=full`
- Result: 17 focused tests and all 100 project tests passed; file-scoped
  pre-commit hooks passed. The smoke graph contains 14 new
  request-preparation jobs only. The credential-free full graph contains 75
  jobs: the existing anchor-production chain plus the new scored catalog,
  sampled manifest, six policy request tables, and six HAL BEDs. No job ran.
- Skill follow-up: the reusable additive-S3 invariant is isolated in draft PR
  [#474](https://github.com/Open-Athena/marin-dna/pull/474), not in the #473
  branch diff.
- Interpretation: the request and sampling contracts are ready for review.
  Actual HAL and MAF projection, QC comparison, and training remain unlaunched.
- Next action: add policy-specific HAL and MAF projection rules under the same
  experiment namespace, then dry-run and review their smoke DAG before seeking
  compute approval.

### 2026-08-19 21:55 UTC - CSP-003 additive projection smoke DAG

- Hypothesis: all six projection policies can share immutable alignment inputs
  through new experiment-scoped rules while preserving established rules,
  outputs, and full-window behavior.
- Commit Hash: `7eb8109e6830bd6bd19b8e252f42c682dd751560`
- Additive boundary: new rules consume established staging and reference
  artifacts as inputs, but every policy-derived artifact is written beneath
  producer-keyed `experiments/473/projection/` paths.
- Commands:
  - `uv run --locked pre-commit run --files <issue #473 projection files>`
  - `uv run --locked pytest`
  - `uv run --locked snakemake -n --profile workflow/profiles/default --default-storage-provider none issue_473_projection_smoke`
- Result: all 102 project tests and file-scoped pre-commit hooks passed. The
  credential-free smoke DAG resolved 292 jobs across six policies, two HAL
  species, five MultiZ species, and two smoke chromosomes, followed by six
  combined sequence tables, six standard QC aggregations, and one cross-policy
  recovery and mapping-evidence comparison. No workflow job ran.
- Execution boundary: running the target would stage the shared 1.26 TB HAL,
  staged smoke-cohort MAF inputs, and reference genomes. Local execution and
  remote compute remain unapproved.
- Interpretation: the additive projection implementation is ready for code
  review and an explicitly approved remote smoke run. This is not a scientific
  result and does not select a policy.
- Next action: obtain explicit remote-compute and data-staging permission, run
  the smoke target, inspect coordinate, accounting, and recovery outputs, and
  publish the reproducible result to issue #473.

### 2026-08-19 22:20 UTC - CSP-004 authorized remote smoke launch

- Authorization: the user explicitly approved the #473 data work and model
  training on Iris, while the repository's development-only chromosome split
  and no-merge boundary remain in force.
- Commit Hash: `59fa33caa9a410563099c72715ff69b50ad50887`
- Worker: Sky cluster `vertebrate-project`, AWS `c6id.12xlarge`, Iris job 4;
  launched at 2026-08-19 22:19:59 UTC.
- Commands:
  - `sky launch -c vertebrate-project sky/issue_473.yaml --env TIER=smoke --env TARGET=issue_473_projection_smoke --env DRY_RUN=1 --env PIPELINE_COMMIT_SHA=59fa33caa9a410563099c72715ff69b50ad50887`
  - `sky exec vertebrate-project sky/issue_473.yaml --env TIER=smoke --env TARGET=issue_473_projection_smoke --env DRY_RUN=0 --env PIPELINE_COMMIT_SHA=59fa33caa9a410563099c72715ff69b50ad50887`
- Setup validation: uv 0.11.31, Cactus 3.1.4, a 2.6 TB RAID0
  workspace with 2.5 TB available, and the exact 1,262,706,573,453-byte HAL
  S3 object were verified. The S3-aware remote dry-run resolved 292 jobs with
  config SHA `4fe838e3eaf48321393f4efd7c658abfa5d4424f5f1d0c2ef5475ecfa29de324`.
- Launcher correction: the established launcher rejected uv's architecture
  suffix. A new issue-scoped launcher parses the version token without editing
  that shared launcher or any established rule.
- Progress at 22:22 UTC: job 4 was `RUNNING`; at least 30 of 292 steps had
  completed, including reference and MultiZ staging artifacts. The 1.26 TB HAL
  was actively staging to local NVMe. No scientific result is claimed yet.
- Next action: monitor at coarse intervals, validate exact durable artifacts
  and paired QC after completion, then terminate the staging worker.

### 2026-08-19 23:12 UTC - CSP-005 smoke receipt and full launch

- Smoke result: Iris job 4 completed all 292 jobs successfully. Atomic HAL
  staging finished at approximately 23:09 UTC, about 49 minutes after the
  transfer began.
- Durable smoke evidence:
  - `policy_summary.parquet`: 3,914 bytes, full-object CRC64NVME
    `YNFn/FuDWf4=`.
  - `full_window_pairwise.parquet`: 2,913 bytes, full-object CRC64NVME
    `X9AnswgT3FM=`.
  - Both objects were restored from S3 and parsed successfully.
- Accounting: every policy closed the 35-pair request grid exactly. Full
  window and centered widths 1, 17, 33, and 65 each accepted 12 pairs;
  `center_129` accepted 11 and explicitly rejected one. `center_1` and
  `full_window` had identical accepted sets in this small smoke.
- Fixed implementation commit:
  `eb69d1d3174599bfa0a54372a3b7cf774abc04ab`. The exact catalog contains
  518,764 #417 standard-region anchors plus 116,162 exp351 enhancer-centered
  anchors. Immutable direct inputs are pinned by S3 path, byte size, and
  full-object checksum.
- Validation: all 111 project tests and changed-file hooks passed. The
  committed full target resolved 10,658 jobs in a credential-free dry-run.
- Full execution: Iris job 5 launched at 2026-08-19 23:12:53 UTC on the
  existing `vertebrate-project` worker with target
  `issue_473_fixed_projection_experiment`, `tier=full`, and producer commit
  `eb69d1d3174599bfa0a54372a3b7cf774abc04ab`.
- Published update:
  https://github.com/Open-Athena/marin-dna/issues/473#issuecomment-5349129178
- Interpretation: the smoke validates execution and accounting but is too small
  to select a policy. Pilot and full paired QC remain the decision evidence.
- Next action: monitor job 5, validate durable outputs and manual samples,
  complete sampled raw-alignment traces, then prepare reviewed dataset
  artifacts and four matched training runs.

### 2026-08-19 23:45 UTC - CSP-006 immutable restore diagnosis and trace target

- Full-run status: Iris job 5 stopped at 29 of 10,655 steps, before any
  projection. Ten direct immutable restores all failed the same fail-fast
  assertion; the HAL, NVMe worker, and completed S3 outputs remain intact.
- Root cause: the source objects return exact byte sizes but no
  `ChecksumType` or `ChecksumCRC64NVME` fields from S3 `HEAD` or
  checksum-enabled `GET`. The direct manifest's pinned CRC values are valid:
  a restored 21,683-byte species manifest recomputed to `YPwbCiNeodY=`,
  exactly matching the committed value.
- Recovery boundary: do not weaken or edit the existing restore rules. New
  additive pre-stage code restores only the ten direct objects, 24 scored
  chromosome tables, and 270 rejection-evidence objects consumed by #473;
  it verifies live size, computes CRC64NVME and SHA-256 locally, writes the
  existing receipt paths atomically, and records timing and throughput.
- Trace implementation commit:
  `69371d56264ba69d34e1ea17540e89499aa10be2`. A separate Snakemake target,
  `issue_473_fixed_hal_alignment_trace`, maps a stable accepted-row sample
  from emitted species windows back to the original human anchors with
  `halLiftover --outPSLWithName`. It reports exact emitted-window and
  emitted-to-anchor base coverage while leaving genome-wide quantities
  explicitly unavailable.
- Validation: all 115 locked project tests passed; file-scoped pre-commit
  hooks passed; workflow parsing registered the new target. Synthetic PSL
  tests cover clipping, split blocks, empty/species partitioning, and negative
  query coordinates.
- Interpretation: job 5 produced no scientific evidence and is not a partial
  result. The pre-stage is a provenance-preserving operational recovery for
  absent S3 checksum metadata, not a change to projection semantics.
- Next action: run the pre-stage on the existing worker, verify all exact
  objects, then resume the exact producer graph without weakening its checks.

### 2026-08-20 00:46 UTC - CSP-007 verified pre-stage and corrected full run

- Pre-stage result: Iris job 7 restored and independently verified all 304
  immutable objects consumed by #473: 10 direct inputs, 24 scored chromosome
  tables, and 270 rejection-evidence objects. The verified total was
  10,196,517,248 bytes. Every atomic receipt records expected and observed
  size, full-object CRC64NVME, and SHA-256; job duration was 6m44s.
- Producer correction: commit
  `f764b7f1fa34ea730842117239dd179a7e3be572` adds a new full-run launcher
  supplying the three #473-only configuration values. No established rule,
  shared config, or S3 output path was edited.
- Dry-run result: Iris job 10 completed in 9m27s and resolved exactly 10,329
  jobs from config SHA-256
  `bf8367c285f955407cfb2dba6102661b2e528261b64fe52095d52a688cd6d039`.
  The graph contained no immutable restore job and no regeneration of #417
  scored/rejection artifacts.
- Full execution: Iris job 11 launched at 2026-08-20 00:32:49 UTC on the
  retained `vertebrate-project` worker. The live graph entered execution after
  resolving the same 10,329 jobs and exact producer/config namespace. At
  00:42 UTC it had completed 58 steps, including HAL validation, fixed anchor
  catalog construction, producer/species receipts, request tables, and the
  first reference artifacts. Outputs were uploading to canonical S3 storage
  as each rule completed.
- Publication implementation: commit
  `ebf9a6d8ed64c90fea55344222334dea281d5ba5` adds a standalone publisher for
  the three new datasets. It reads only the exact producer namespace and
  defines only new card, shard, validation, and opt-in upload rules. The
  established publication workflow is unchanged.
- Validation: all 117 locked project tests passed under the shared-node lock
  in 8.26s with 292,400 KiB peak RSS. Changed-file hooks passed. The
  standalone publisher resolves only its eight issue-specific rules.
- Interpretation: pre-staging repaired an operational S3 metadata gap while
  retaining stronger local full-object verification. Job 11 has begun valid
  work but no full-scale scientific result is claimed before target success
  and durable artifact inspection.
- Next action: monitor job 11 to completion, validate full paired QC and
  manual samples, run the sampled bidirectional HAL trace, then build and
  review the exact publication artifacts before the four matched Iris runs.

### 2026-08-20 01:26 UTC - CSP-008 additive development-evaluation scaffold

- Evaluation boundary: added experiment-local config generation, paired
  metrics, result analysis, tests, and Sky launchers on the permanent training
  branch. No maintained `evals_v2` rule, config, profile, or shared output was
  edited.
- Official scorer matrix: 40 checkpoints across the four preregistered arms,
  steps 500 through 5,000 every 500 steps. The generated config hard-codes the
  `train` split and pinned Mendelian, Complex, and SGE revisions. CDS arms run
  Mendelian + SGE; enhancer arms run Mendelian + Complex.
- Paired analysis: exact evaluation-row identity is asserted before comparing
  policies. AUPRC and #459 Group SMD use one aligned match-group bootstrap;
  the same subset seed is reused across all checkpoints so trajectory and
  within-checkpoint `center_1 - full_window` uncertainty are paired. All eight
  registered Mendelian specialist subsets are mandatory.
- Additional-seed boundary: the analysis only records whether two consecutive
  checkpoint deltas share a direction or the endpoint interval excludes zero.
  It cannot launch another arm; issue #473's separate decision and compute-
  approval gate remains in force.
- Validation: the six new evaluation tests and the five existing training/
  format tests pass in bounded separate processes. Peak RSS was 492,684 KiB;
  changed-file pre-commit hooks pass.
- Dry-run finding: a local parse exposed that command-line Snakemake config is
  recursively combined with the workflow's built-in config. The generator now
  explicitly empties every unrelated optional model registry (`nuc_dep`, UMAP,
  LL-gap, and probe) to prevent base-model leakage. Creating the Python 3.13
  evaluator environment and parsing the workflow reached 1,087,856 KiB, above
  this shared node's 500 MiB limit, so the corrected graph will be dry-run on
  remote compute and no further local evaluator parse will be attempted.
- Outputs: the analysis launcher writes point tables, aligned bootstrap draws,
  paired deltas, official Complex/SGE tables, trajectory plots, summary, and
  SHA-256 manifest to an evaluation-commit-keyed S3 prefix.
- Next action: commit and push this scaffold, remotely dry-run the corrected
  overlay, continue monitoring projection job 11, then use the evaluator only
  after the four authorized training runs produce immutable checkpoint roots.

### 2026-08-20 01:33 UTC - CSP-009 remote evaluator graph validation

- Snapshot: the additive training/evaluation branch was pushed at
  `863f9367d0ad75b776618394c8c4ffeb770bf4a4`.
- Remote dry-run: Sky cluster `exp473-eval-dryrun`, job 3, succeeded on one
  on-demand `c6i.xlarge` in `aws/us-east-2`. The temporary worker was
  terminated immediately after success.
- Exact graph: 201 jobs total: 40 `download_model`, 80 `compute_scores`, 80
  `compute_metrics`, and one `all`. The matrix is four arms by ten checkpoints;
  every checkpoint has Mendelian plus its region-meaningful secondary endpoint
  (SGE for CDS, Complex for enhancer). No unrelated interpretation, probe,
  LL-gap, or base-model job entered the graph.
- Safety: Snakemake reported `split=train`; the command used `-n`, so no model,
  score, metric, or other scientific output was written.
- Operational fixes: Sky's image carried uv 0.12.5 while the evaluator locks uv
  0.11.31, and synchronized worktrees retain a local-only `.git` indirection.
  The permanent launchers now pin uv 0.11.31 and require the full pushed
  `EXP473_EXPERIMENT_COMMIT` as an explicit environment input instead of
  calling `git rev-parse` remotely. Both launcher YAML files parse.
- Next action: push the launcher fix, continue monitoring projection job 11,
  and do not start scoring until training has produced all exact checkpoint
  roots.

### 2026-08-20 01:43 UTC - CSP-010 acceptance and exposure-accounting audit

- Live execution: projection job 11 remained healthy at 769/10,329 steps.
  A read-only worker inspection found 43 concurrent `halLiftover` processes on
  48 CPUs with one-minute load 43.19, confirming that the apparent two-job
  scheduler log message was a refill batch rather than a concurrency cap.
- Acceptance audit: the issue body was checked against the committed producer,
  trace, publisher, training, and evaluation surfaces. All five functional
  regions, six sampled landmark policies, full-scale `full_window` and
  `center_1` comparisons, paired anchor uncertainty, sampled exact HAL
  coverage, three new datasets, four matched model arms, and development-only
  evaluation are represented. Scientific completion remains unclaimed until
  their runtime artifacts pass their respective gates.
- Training-exposure semantics: each published `train_rows` count includes one
  `+` and one reverse-complement `-` row per biological anchor-species pair.
  Every arm presents 40,960,000 sequences and 10,485,760,000 tokens. The final
  report will record both published-row effective epochs
  (`40,960,000 / train_rows`) and biological-pair presentations
  (`40,960,000 / (train_rows / 2)`), alongside tokens per published row
  (`10,485,760,000 / train_rows`), so augmentation cannot be mistaken for
  independent projection yield.
- Invariants: the active projection and every prepared downstream surface are
  additive; no established S3-backed rule or shared output path was edited.
  Held-out even-autosome and chromosome-Y labels, predictions, measurements,
  and aggregate metrics remain untouched.
- Next action: leave the saturated projection worker undisturbed, validate its
  full QC/manual artifacts at target success, then run the sampled HAL trace
  before any dataset upload or model launch.

### 2026-08-20 02:06 UTC - CSP-011 publication mutation preflight

- Repository state: authenticated Hugging Face inspection found all three
  reserved issue #473 dataset repositories absent. The established full-window
  CDS dataset is private, so implicit repository visibility was not accepted
  for the new uploads.
- Privacy guard: additive publisher commit
  `406b6feb06525b0631d818dac6017830eced0b38` creates an absent issue #473
  dataset repository explicitly as private, refuses to upload to a
  pre-existing public repository, and reasserts private visibility after the
  validated upload. The shared uploader and every established publication rule
  remain unchanged.
- Revision retention: the same PR retains each local
  `publication/upload.done/<dataset>` receipt instead of marking it temporary.
  Each receipt contains the repository ID and verified 40-character immutable
  Hub revision; the runbook requires capturing all three before terminating
  the worker and pinning them in the training recipe.
- Validation: 120 locked projection-project tests passed in 11.26s with
  300,348 KiB peak RSS under the shared-node guard. Changed-file hooks passed.
  Unit tests cover private repository creation and fail-closed rejection of an
  existing public repository.
- Next action: launch publication only from `406b6feb...` after full projection
  QC and sampled HAL trace pass. Build and inspect the draft artifacts before
  invoking the separately gated upload target.

### 2026-08-20 02:28 UTC - CSP-012 paired intersection-loss evaluator

- Snapshot: additive experiment commit
  `8da6343b58e6bb0c0921c4f1e1b651568f8fc2bb` was pushed to the permanent
  `codex/exp473-center-seeded-projection-training` branch.
- Scope: a separate `IntersectionLoss.smk` imports the unchanged official
  `evals_v2` causal-LM scorer and exposes only four new
  `issue_473_intersection_*` rules. It reads the producer-pinned, unlabeled
  chromosome-18 full-window/center-1 intersection views directly from their
  immutable S3 namespace; no maintained evaluation rule or established output
  path changed.
- Statistic: the scorer reconstructs the training objective from per-row
  uppercase/lowercase log-likelihood atoms using weights 1.0/0.01, asserts
  exact paired biological row identity, and reports token-weighted
  `center_1 - full_window` NLL with an aligned human-anchor bootstrap.
  Negative deltas favor center-1.
- Graph validation: a locked local Snakemake dry-run under the `evals_v2`
  S3-default profile succeeded with exactly 82 jobs: 40 local checkpoint
  downloads, 40 S3-backed score cells, one analysis, and one target. The first
  dry-run exposed a local-checkpoint storage-identity mismatch; the committed
  workflow annotates both producer and consumer paths as local.
- Tests: all 16 experiment-project tests passed in three bounded processes;
  evaluation contracts include the exact 40-cell config, producer revision,
  chromosome/length and row-pair guards, case-weight formula, delta direction,
  complete synthetic matrix, manifest, and rule isolation. Changed-file hooks
  passed. Peak RSS was 489,348 KiB.
- Safety: the evaluator records `vep_held_out_access: false` and contains no VEP
  dataset route. Even-autosome/Y VEP labels, predictions, measurements, and
  aggregate metrics remain untouched.
- Next action: continue the projection gate; use this evaluator only after the
  four authorized arms produce immutable checkpoint roots.

### 2026-08-20 02:40 UTC - CSP-013 downstream launch gate preflight

- Producer state: Sky job 11 remained healthy at 1,279/10,329 steps (12%)
  from exact producer `f764b7f1...`; durable uploads continued. The issue body
  status and evidence links were refreshed without changing its scientific
  scope or decision log.
- Trace gate: a strict no-workdir-sync Sky task now verifies the exact trace
  rule, implementation, and common-rule file hashes; sets the explicit
  producer commit because synchronized Sky worktrees have a local-only `.git`
  indirection; dry-runs the full producer target with only its aggregate rule
  allowed; and stops before staging unless every final producer leaf exists.
- Fail-closed evidence: trace preflight job 13 stopped at that aggregate check
  because the producer is still incomplete. It did not stage source tables,
  run `halLiftover`, or write trace artifacts. Producer job 11 remained
  running. Job 12 was an earlier shell-quoting-only preflight failure and also
  wrote no scientific artifact. The corrected task will be resubmitted only
  after job 11 succeeds.
- Iris state: the pinned experiment environment reached the healthy controller
  at `iris.oa.dev`; 614/614 workers were healthy. In the selected
  `tpu_v5p-preemptible_8-us-east5-a` group, two slices were ready, none were
  booting, initializing, or failed, and demand was zero. This is a capacity
  preflight, not a reservation; Iris may need to autoscale when all four arms
  are submitted.
- Launch boundary: no `exp473` job appeared among the 100 most recent Iris
  jobs. Dataset revisions remain unset, so the four fail-closed training
  graphs cannot launch before publication.
- Next action: continue job 11 to its full QC/manual target, validate those
  durable artifacts, then resubmit the strict sampled-trace task.

### 2026-08-20 02:52 UTC - CSP-014 evaluation provenance hardening

- Snapshot: additive experiment commit
  `3fde057197849273456f1118199b2fcf06e79937` gives every official evaluator
  model a full experiment-commit-keyed name. The same names now isolate the
  separate chromosome-18 intersection-loss cells. Maintained `evals_v2`
  rules and established model outputs remain unchanged.
- Trigger: the official score parquet does not itself include `split`, and
  the shared `compute_scores` output path is keyed only by model and dataset.
  Stable issue-only model names could therefore have allowed stale score
  reuse without independently visible split provenance.
- Guard: score, metric, and downloaded-checkpoint paths are now unique to the
  exact 40-character experiment commit. Paired Mendelian analysis additionally
  requires each score bundle's matching official metric parquet to exist and
  record only `split=train`; Complex and SGE metrics retain the same check.
  The analysis launcher passes the exact commit, and the output manifest
  records it together with all verified metric inputs.
- Validation: all 17 locked experiment-project tests passed in 4.96 seconds
  with 505,924 KiB peak RSS. Changed-file hooks passed. Focused tests cover
  commit isolation, malformed commit rejection, development provenance,
  held-out rejection, missing provenance rejection, and launcher propagation.
- Graph: an official dry-run with dummy immutable checkpoint roots retained
  the exact 201-job matrix: 40 downloads, 80 scores, 80 metrics, and one
  aggregate target. Every generated key contained the full commit and every
  output was missing, demonstrating namespace isolation; dry-run wrote no
  scientific artifact. This local Snakemake process unexpectedly reached
  1,058,844 KiB peak RSS (02:51:09--02:51:20 UTC), so it is now known to
  exceed the shared node's 500 MiB local-workload limit and will not be
  repeated locally; subsequent full graph checks use the prepared remote
  launcher.
- Safety: all VEP config remains hard-coded to `train`; no even-autosome or
  chromosome-Y VEP label, prediction, effect measurement, or aggregate metric
  was accessed.
- Next action: continue the projection gate, then trace, review, private
  publication, four authorized Iris arms, intersection loss, and official
  development analysis in that order.

### 2026-08-20 03:08 UTC - CSP-015 training recipe parity audit

- Snapshot: additive experiment commit
  `7cb733e22661c967b18e9e3fe0cb04f90094ce18` restores exact #417 training
  invariants before any issue #473 arm launches.
- Trigger: a field-by-field comparison against the pinned #417 recipe at
  `0c83058b` found that the new tokenization config named an unversioned Hub
  tokenizer while the model config had no tokenizer path. The latter would
  allow Hugging Face exports to inherit the Qwen reference tokenizer rather
  than the seven-token DNA character+BOS tokenizer. The README's claim that
  the tokenizer was copied locally was therefore not yet true.
- Fix: the isolated project now vendors the exact three tokenizer files from
  `marin-dna/tokenizer-char-bos@a73e9d9e...`, verifies their SHA-256 digests
  before constructing any graph, uses the local path for tokenization, and
  sets the same path on `Qwen3Config` for every Hugging Face export. Observed
  digests exactly match #417: `02b7b977...` for the special-token map,
  `d066e668...` for `tokenizer.json`, and `4e814edc...` for the tokenizer
  config.
- Recipe parity: seed 0 is now explicit on both trainer and data order;
  per-device parallelism is 1,024; optimizer-state checkpoints are retained
  every 500 steps in addition to Marin's ten-minute rolling recovery; host
  resources match #417's 16 CPU, 56 GiB RAM, and 100 GiB disk request. Model
  geometry, 8,192-sequence global batch, 5,000 steps, exact Adam parameters,
  case-aware 1.0/0.01 loss, and 500-step Hugging Face cadence are asserted.
- Validation: all 18 locked project tests passed as three bounded processes:
  3 format tests (471,752 KiB peak RSS), 12 evaluation tests (148,056 KiB),
  and 3 materialized-recipe tests (489,312 KiB). Changed-file hooks passed.
  Tests construct all four artifact plans and verify tokenizer digests and
  provenance tags, model and optimizer fields, trainer/data seeds,
  microbatching, retained checkpoints, dataset revisions, and loss format.
- Launch gate: no issue #473 model has launched. The three new immutable
  dataset revisions remain mandatory, so publication review still precedes
  all four authorized arms. Held-out VEP data remains untouched.
- Next action: continue producer job 11 through exact QC/manual artifacts,
  then sampled HAL trace, publication review, and private upload.

### 2026-08-20 03:15 UTC - CSP-016 remote training-plan preflight

- Remote proof: Iris job `/ubuntu/exp473-recipe-preflight` succeeded as one
  CPU-only task in 1 minute 23 seconds. It used a 0.1 MB workspace bundle,
  built the independently locked project from `/app`, installed exact Marin
  commit `6bb4d746...`, and lowered the `enhancer_center_1` arm without
  `--run`. It launched no tokenize task, TPU worker, model, or scientific
  artifact.
- Packaging: remote graph construction verified all three vendored tokenizer
  files before emitting the plan. The token-cache fingerprint records
  `tokenizer: tokenizer`, exact source revision `a73e9d9e...`, all three file
  digests, the immutable dataset revision input, and the DNA format with
  `text_key=sequence`, uppercase weight 1.0, and lowercase weight 0.01.
- Materialized recipe: the remote checkpoint plan records the DNA tokenizer
  on `Qwen3Config`, 0.25B geometry, exact Adam parameters, seed 0,
  8,192-sequence batch, 1,024 per-device parallelism, 5,000 steps, ten-minute
  rolling recovery, retained native checkpoints every 500 steps, and Hugging
  Face exports every 500 steps.
- Fingerprint boundary: Marin deliberately represents an unrealized cache as
  a constant ordinary-text placeholder while lowering; the pinned runtime
  path reloads tokenizer, format, and tags from the completed cache's
  `.artifact.json`. Commit `cd7657c1549c78619dd3b7218fc32cc7c7773106`
  makes that runtime boundary fail closed: a realized issue #473 cache is
  rejected unless its record exactly matches the case-aware DNA format.
- Validation: the current 19-test project suite is covered in bounded
  processes (3 format, 12 evaluation, 4 materialized recipe/cache tests).
  The new realized-record test reconstructs a successful DNA cache and rejects
  both an ordinary-text record and a 1.0/1.0 non-case-aware record. Its focused
  run passed with 489,688 KiB peak RSS; changed-file hooks passed.
- Safety: dummy hexadecimal dataset revisions and a literal test W&B key were
  used only to lower the non-executing plan. No dataset was read, no real
  credential was transmitted, and held-out VEP data remained untouched.
- Next action: keep the four paid arms gated on producer QC, sampled trace,
  and reviewed private publication; continue monitoring producer job 11.

### 2026-08-20 03:29 UTC - CSP-017 real child-worker tokenizer preflight

- Remote proof: CPU-only Iris job
  `/ubuntu/exp473-tokenizer-worker-preflight-v3` succeeded with no failures or
  preemptions in 1 minute 3.5 seconds. Its nested child task
  `verify_tokenizer_on_worker-b15c15b5/0` loaded the tokenizer from the 0.1 MB
  packaged workspace, verified all three vendored file hashes, and encoded
  `ACGTacgt` as `[2,3,4,5,6,3,4,5,6]` with vocabulary 7, BOS 2, PAD 0,
  UNK 1, and no EOS. No TPU or scientific dataset was requested or read.
- Fail-then-fix evidence: the first true child-worker run correctly found the
  packaged files but exposed that pinned `HfMarinTokenizer` does not itself
  expose `unk_token_id`; its Hugging Face adapter does. Commit
  `0eeacd69bb1b16b0c2c9a7c976028864b1f23f58` moves all special-token
  assertions to that runtime adapter and tightens the unit stub to match the
  real API. The focused five-test file passed in 4.82 seconds with 492,276 KiB
  peak RSS under the shared-node lock and resource caps.
- Namespace proof: a second successful run revealed that a shell-prefixed
  `MARIN_PREFIX` is not forwarded by `iris job run`. The real four-arm launch
  block already used Iris `-e` correctly; commit
  `69f4514d0ebd4767db0a8add7287ccf63ee1080d` corrects the preflight example.
  The final run explicitly forwarded the variable and retained an artifact
  record at
  `gs://marin-us-east5/MarinDNA/exp473_center_seeded_projection/preflights/exp473-tokenizer-worker/2026.08.20/.artifact.json`.
  That record is clean, identifies base commit `69f4514d`, fingerprint
  `c71de7c2`, the exact Git remote and branch, the exact Iris command, and all
  three tokenizer SHA-256 values.
- Producer state: Sky job 11 remains healthy and running at exact producer
  commit `f764b7f1...`; its durable S3 uploads reached 1,368/10,329 steps
  (13%) at 03:25 UTC. It was not interrupted or duplicated.
- Next action: let job 11 finish its exact QC/manual target, then stage its two
  completed source tables and launch the strict sampled HAL trace before
  reviewing or publishing any dataset.

### 2026-08-20 06:00 UTC - CSP-018 first authorized training arm launched

- Independent arm: the established full-window CDS dataset is already pinned
  to `marin-dna/vertebrate-v1-cds@bfab878078c4ee6c0f47b760f1e5e0577549dc9d`,
  so this arm does not depend on the three issue #473 publications. Exact
  experiment snapshot `e674aab050cf170d4433f724881adeb041c4f131` launched on
  Iris as `/ubuntu/exp473-cds-full-window-v2` with the preregistered seed-0,
  0.25B, 5,000-step recipe and commit-pinned Marin environment.
- Fail-closed retry: the first coordinator
  `/ubuntu/exp473-cds-full-window` failed before graph construction because a
  shell-scoped W&B variable expanded before assignment. It created no child
  task, dataset cache, TPU request, checkpoint, or training state. The v2
  launch forwards the existing credential correctly; the secret value was not
  printed or recorded here.
- Runtime proof: v2 expanded cleanly through artifact `tokenize-bcd6d6c5` and
  coordinator `zephyr-tokenize-train-f88d1ec8-coordinator-d032874d`. Its 64
  CPU tokenization workers were all running with zero failures and zero
  preemptions, writing the exact cache root
  `gs://marin-us-east5/MarinDNA/exp473_center_seeded_projection/inputs/cds_full_window-char-bos/2026.08.20/`.
  Sample worker logs record 256 tokens per document, matching the registered
  sequence length.
- Producer: immutable Sky job 11 remains healthy and was not interrupted. It
  reached 1,613/10,329 durable steps (16%) at 05:52 UTC. A read-only resource
  audit confirmed conservative rule reservations are the throughput limit;
  the stable producer remains the authoritative S3 writer.
- Safety: no new-dataset training arm was launched, and no VEP label,
  prediction, effect measurement, or aggregate metric was accessed. The
  even-autosome/Y held-out boundary remains intact.
- Next action: monitor tokenization into the v5p-8 training child while the
  producer continues, then execute trace, reviewed private publication, and
  the other three arms when their immutable revisions exist.

### 2026-08-20 07:02 UTC - CSP-019 CDS step-500 checkpoint validated

- Checkpoint milestone: the v5p-8 task
  `/ubuntu/exp473-cds-full-window-v2/run_levanter_train_lm-622836a3`
  completed both the retained native checkpoint and Hugging Face export at
  step 500. The immutable roots are
  `gs://marin-us-east5/MarinDNA/exp473_center_seeded_projection/checkpoints/dna-exp473-0p25b-cds_full_window-v1/2026.08.20/checkpoints/step-500`
  and the corresponding `hf/step-500` directory.
- Export contract: the Hugging Face directory contains `config.json`,
  `model.safetensors`, `tokenizer.json`, and `tokenizer_config.json`. The
  config records Qwen3, vocabulary 7, BOS 2, PAD 0, no EOS, 256 positions,
  hidden width 1,152, intermediate width 4,608, 12 layers, and 9 attention
  and KV heads, matching the registered recipe.
- Tokenizer proof: Hugging Face reserialization changed raw JSON bytes, but
  canonicalized `tokenizer.json` exactly matches the vendored source at SHA-256
  `962ed016c5654513548b7365c60f3cf657435717cb2cf0bf07270ff6a9ee7bd3`.
  A bounded local-only `AutoTokenizer` round trip loaded the exported files
  with vocabulary 7, BOS 2, PAD 0, UNK 1, no EOS, and encoded `ACGTacgt` as
  `[2,3,4,5,6,3,4,5,6]`. The check ran under the shared-node lock and caps in
  1.19 seconds with 75,432 KiB peak RSS.
- Rotary metadata: Transformers warns because the inherited exact #417
  `Llama3RotaryEmbeddingsConfig` retains an 8,192-token original scale while
  this model's maximum is 256. The export nevertheless loads successfully,
  and direct comparison confirms the issue #473 recipe is identical to the
  pinned #417 model construction on this field; this is not a policy-arm
  difference.
- Training health: W&B run
  `dna-exp473-0p25b-cds_full_window-v1` remained running at step 544 with loss
  1.31469, 632,902 tokens/s, 3.31 seconds/step, and 54.18% instantaneous MFU.
  There were zero Iris failures and zero preemptions.
- Producer: Sky job 11 remained healthy and reached 1,723/10,329 durable
  steps (17%). Its exact QC/inspection target and all three new dataset
  publications remain incomplete, so trace and the other arms stay gated.
- Safety: checkpoint validation read only the unlabeled model/tokenizer
  artifacts. No held-out VEP label, prediction, effect measurement, or
  aggregate metric was accessed.
- Next action: continue the CDS arm through step 5,000 and the producer through
  its exact aggregate target; launch the trace immediately after producer
  success.

### 2026-08-20 07:43 UTC - CSP-020 checkpoint-race recovery hardened

- Failure diagnosis: the first training child
  `/ubuntu/exp473-cds-full-window-v2/run_levanter_train_lm-622836a3`
  reached step 842, then failed while writing a rolling recovery checkpoint
  with `RuntimeError: Set changed size during iteration` in Python 3.12's
  `asyncio.runners._cancel_all_tasks -> tasks.all_tasks`. This is the
  cross-thread weak-set iteration race tracked by CPython issue 80788, not a
  model, data, tokenizer, or accelerator failure. The retained step-500 native
  checkpoint and Hugging Face export remained complete and validated.
- Recovery: a valid temporary step-724 checkpoint was present under the same
  immutable checkpoint root. Retry coordinator
  `/ubuntu/exp473-cds-full-window-v3` reused the realized cache and W&B run,
  found that recovery state, and resumed training at step 725 rather than
  replaying completed work. This already-running retry carries the earlier
  source bundle; it was not interrupted merely to install the hardening.
- Hardening: commit `de03d0e854864870f3d648999ab3c46f893c5173`
  adds a project-local Python-3.12-only compatibility guard that snapshots the
  weak-reference registry atomically under the GIL, preserves asyncio's exact
  loop and completion filtering, patches both public aliases idempotently, and
  skips unknown interpreter or registry layouts. Future retries and the other
  three arms inherit the guard.
- Verification: all 23 project tests passed in bounded processes (3 asyncio
  compatibility, 3 DNA-format, 12 evaluation, and 5 experiment tests); the
  largest peak RSS was 487,964 KiB. Changed-file pre-commit hooks passed. A
  real Python 3.12 Iris preflight
  `/ubuntu/exp473-asyncio-guard-preflight-v2` then succeeded in 12.72 seconds
  with exit 0, zero failures, and zero preemptions, printing
  `exp473 asyncio guard active` after verifying the installed aliases and an
  `asyncio.run` shutdown.
- Safety: recovery reused only the authorized unlabeled CDS training cache and
  checkpoint namespace. No held-out VEP label, prediction, effect measurement,
  or aggregate metric was accessed.
- Next action: allow v3 to continue from step 725. If the same race recurs in
  its pre-hardening bundle, launch the next retry from this guarded commit and
  the newest valid checkpoint; otherwise retain the guard for the remaining
  three arms.

### 2026-08-20 08:01 UTC - CSP-021 recovered CDS arm reaches step 1,000

- Durable milestone: retry coordinator
  `/ubuntu/exp473-cds-full-window-v3` passed the original step-842 failure
  point and completed both the retained optimizer-state checkpoint at
  `checkpoints/step-1000` and the Hugging Face export at `hf/step-1000` under
  the registered GCS root. Training resumed beyond step 1,000.
- Recovery evidence: v3 started from temporary step 724, saved a replacement
  temporary checkpoint at step 811, atomically deleted the older step-724
  recovery state, saved another temporary checkpoint at step 924, and then
  completed the step-1,000 native and Hugging Face writes. Iris still reports
  zero failures and zero preemptions for v3.
- Interpretation: this validates the checkpoint namespace and restart path
  after the v2 asyncio shutdown race. The Python 3.12 guard remains necessary
  for future retries and the remaining arms because v3 was packaged before
  commit `de03d0e8` and the underlying race is nondeterministic.
- Safety: the arm continued on the authorized unlabeled full-window CDS data.
  No held-out VEP label, prediction, effect measurement, or aggregate metric
  was accessed.

### 2026-08-20 08:07 UTC - CSP-022 sampled trace handoff armed

- Automatic handoff: Sky job 14,
  `issue-473-fixed-hal-alignment-trace-after-producer`, is running beside the
  immutable producer on cluster `vertebrate-project`. It checks the exact
  producer target every ten minutes with a no-lock, target-only dry-run and
  sleeps while any required artifact is absent. Its first gate check failed
  closed as expected and entered the wait state at 08:05:46 UTC.
- Execution boundary: only after all inputs of
  `issue_473_fixed_projection_experiment` exist does the handoff stage the
  exact full-window and `center_1` source tables, record byte sizes and
  SHA-256 values, and invoke the six allow-listed sampled-HAL trace rules.
  It cannot publish datasets or launch model training.
- Reproducibility: the handoff task SHA-256 is
  `fca9d0354ac737c168be65de9535e60d50e9fd30ef83c1b3a84827a9a50bca77`.
  It asserts producer commit `f764b7f1...` plus exact hashes for the trace
  Snakefile, implementation module, and common rules before waiting.
- Producer state: job 11 remained healthy at 1,824/10,329 durable steps when
  the handoff was armed. It remains the sole writer to the producer namespace.
- Next action: monitor only terminal producer, trace-gate, checkpoint, and error
  events; review the complete QC/manual and trace artifacts before publication.

### 2026-08-20 09:02 UTC - CSP-023 disjoint additive fast producer launched

- Scheduling decision: producer job 11's exact rule-completion audit showed
  conservative memory and thread reservations leaving most of its 48 CPUs
  idle during the 5,376 MultiZ contract and 3,424 HAL fan-out jobs. The stable
  writer was not interrupted, cancelled, or modified. Instead, a second
  producer was launched in a distinct commit-keyed result namespace from
  exact snapshot `d43f059c7b0cb8efd5e2396a2bd9e085623a1731`.
- Scientific identity: the projection implementation is unchanged from
  `f764b7f1`; the only files added between the snapshots under the producer
  project are the standalone publication workflow and its tests. The new task
  retains the same full config and expected config SHA-256 `bf8367c...`.
  Runtime overrides change scheduling reservations only: MultiZ candidate
  threads are one, candidate/contract memory is 4 GB, HAL fragment/contract
  memory is 6 GB, and HAL sequence memory is 4 GB. All existing rules and
  outputs remain untouched.
- Remote execution: Sky cluster `issue-473-fast-producer`, job 1, uses one AWS
  `c6id.12xlarge` in `us-east-2` at the displayed $2.42/hour rate. It mounted
  2.6 TB RAID-0 NVMe, verified exact source/module/lockfile hashes, and restored
  all 304 immutable inputs from 08:53:14 through 09:00:33 UTC before starting
  the exact full producer target. First-minute checks showed 92 GB available
  memory, load 1.10, and 2.5 TB free.
- Isolation proof: the result path includes commit `d43f059c...`, so it shares
  neither S3 outputs nor local Snakemake state with job 11. The exact producer
  task SHA-256 is
  `0f84157ab3ba3bdd8270e37611c73fd7cab083212a92801683729cc576a9c9dc`.
- Trace handoff: job 2 on the same cluster failed closed on the incomplete
  exact target and waits at ten-minute intervals before invoking only the six
  sampled-HAL trace rules. Its task SHA-256 is
  `d00d47142b7075cf84db08b63bd8fb6493456f1b25a739f370facc8f08ac5c6e`.
- Next action: validate the 10,329-job DAG, HAL staging, actual concurrency,
  memory headroom, and early durable uploads. Whichever exact producer finishes
  first must still pass the same QC/manual and sampled-trace review gates.

### 2026-08-20 09:21 UTC - CSP-024 fast-producer DAG correction and live validation

- Append-only correction: CSP-023's expected 10,329-job DAG applies after all
  large immutable inputs are already local. This fresh worker correctly built
  a 10,354-job DAG: the same 10,329 producer jobs plus 25 local
  `stage_multiz_maf` jobs needed to restore the chromosome MAFs on the new
  node. This changes staging work only, not any scientific rule or output.
- Namespace verification: live Snakemake child commands use default S3 storage
  rooted at `s3://oa-bolinas/snakemake/vertebrate_projection_dataset/`; their
  resolved paths in the job log contain exact producer commit
  `d43f059c7b0cb8efd5e2396a2bd9e085623a1731` and config SHA-256
  `bf8367c285f955407cfb2dba6102661b2e528261b64fe52095d52a688cd6d039`.
- Live validation: at 09:21 UTC the producer had reached 174/10,354 steps with
  concurrent MultiZ candidate work active across the pilot and full runs.
  Load was 25.21 on 48 vCPUs, available memory was 76.5 GiB, and the 2.6 TiB
  NVMe array had 2.2 TiB free. The largest observed candidate processes were
  about 1.1 GiB RSS, well inside both the 4 GB reservations and node headroom.
  No error or terminal event was observed.
- Isolation and safety: the original `f764b7f1` producer and its trace handoff
  remain running and unmodified. The fast trace handoff remains fail-closed on
  the incomplete `d43f059c` target. Neither path can publish data or launch
  training before artifact review.
- Next action: continue event-driven monitoring of both exact producers, the
  matching trace gates, and durable training checkpoints. Review the first
  complete producer's QC, manual examples, and sampled alignment trace before
  private dataset publication.

### 2026-08-20 09:34 UTC - CSP-025 private publisher pinned to fast producer

- Isolated PR update: draft PR #477 now pins its standalone publication
  config to exact producer commit
  `d43f059c7b0cb8efd5e2396a2bd9e085623a1731`; the producer config SHA-256
  remains `bf8367c285f955407cfb2dba6102661b2e528261b64fe52095d52a688cd6d039`.
  No projection rule, shared output, or established publication path changed.
- Publication identity: the tested publisher commit is
  `53c765a06ca4e8489e30556145eda8083890e2fd` and its resolved config
  SHA-256 is
  `84fde31ad856677460c0c4849faa642fabff5a5e59fc410a688a0a29e633b738`.
  The resulting build namespace is disjoint from both producer namespaces.
- Verification: all 120 tests in the independently locked vertebrate
  projection project passed in 11.77 seconds. The bounded run peaked at
  300,516 KiB RSS and exited zero. PR CI was triggered by the push; the
  data-bearing build remains gated on complete producer QC and trace review.
- Privacy boundary: the reviewed uploader creates absent issue-specific
  repositories as private, refuses any pre-existing public repository,
  validates the exact remote file tree and LFS hashes, and rechecks privacy
  after upload.
- Next action: wait for the exact `d43f059c` producer and its sampled trace,
  review the QC/manual/trace artifacts, then launch the build-only target.

### 2026-08-20 09:40 UTC - CSP-026 early immutable producer receipts pass

- Identity: the restored producer receipt exactly records pipeline commit
  `d43f059c7b0cb8efd5e2396a2bd9e085623a1731`, config SHA-256
  `bf8367c285f955407cfb2dba6102661b2e528261b64fe52095d52a688cd6d039`,
  pipeline version `v1`, and tier `full`. Its downloaded file SHA-256 is
  `852d7c1826296dc1d4abb2ee71880e3d9f48be858cfa518aab09d5bef29a3377`.
- Fixed catalog: the completed summary reports 634,926 unique 255 bp anchors
  in 0-based half-open coordinates: 295,561 CDS, 67,155 3-prime UTR, 98,630
  ncRNA exon, 57,418 TSS/5-prime UTR, and the exact 116,162 exon-free
  enhancer-centered cCRE sentinels. The conservation threshold is 0.2.
- Baseline gate: the independently restored compatibility receipt proves exact
  dataframe equality to #417 for all 518,764 standard-region anchor identities
  and all 135 active target-species identities. Its SHA-256 is
  `89b2ab2909ffd4c76b9a87817e61b5eb87f863fecc6111cf2b6a27c979df04b9`.
- Pilot stratification: each of the five regions has exactly 10,000 sampled
  anchors, every anchor is 255 bp, every region covers all 24 source
  chromosomes, every row carries seed 473, and no stratum is overdrawn.
  Conservation-quantile totals are approximately balanced while preserving
  chromosome representation. The active manifest contains 107 HAL mammals
  and 28 MultiZ species across mammals, birds, reptiles, amphibians,
  ray-finned fish, lobe-finned fish, and jawless vertebrates.
- Request-boundary audit: all six request tables contain the same 50,000
  anchor identities. Their submitted widths are exactly 255, 1, 17, 33, 65,
  and 129 bp. Every interval is centered by the declared 0-based half-open
  formula; the 1 bp interval is `[s + 127, s + 128)`. The expected span gates
  are 128--512 for full-window and respectively 1--2, 9--34, 17--66,
  33--130, and 65--258 for the five centered policies. The bounded audit
  peaked at 93,272 KiB RSS and exited zero.
- Scope: these are producer identity, catalog, stratification, and request
  construction gates only. They do not establish projection recovery,
  sequence correctness, alignment coverage, or a preferred policy; those
  remain gated on the complete QC, manual sample, and sampled HAL trace.

### 2026-08-20 09:53 UTC - CSP-027 additive post-projection report prepared

- Reporting gap: the producer preserves raw accepted and rejection evidence,
  but its compact outputs do not directly report every requested
  fragment/span/sequence distribution or complete species-level
  accepted/rejected/no-mapping accounting. No producing rule or existing S3
  output was changed to close this gap.
- Additive analysis: commit
  `67f53f57e1b5c3b069f624d2dd7f0c8faa17d86e` adds a standalone streaming
  module that summarizes fragment count, landmark-aligned bases, pre-resize
  span and width ratio, target interval/strand validity, sequence length,
  ambiguity, repeat masking, and GC by policy, region, backend, species, and
  clade. It derives exact no-mapping counts from grouped accepted and explicit
  rejection counts without materializing the 85-million-cell requested grid.
- Interpretation guard: the module labels aligned fraction as applying only to
  the submitted source landmark. It never substitutes span geometry for
  emitted-window coverage; the latter remains exclusive to the sampled HAL
  trace.
- Trace-gated handoff: commit
  `3ae98937bbab737c1a2164a6e03a4b6cf45cc15b` adds a separate Sky launcher
  that waits for the exact `d43f059c` trace report, stages both accepted
  tables and all 810 new/immutable rejection files on the retained worker,
  runs the bounded-memory analysis, and uploads only to an
  analysis-commit-keyed S3 namespace. It does not alter or invoke a producer,
  publisher, training arm, or evaluator.
- Verification: both focused report tests passed; the complete changed
  vertebrate-projection project passed all 119 tests in 7.80 seconds with
  292,252 KiB peak RSS. Changed-file hooks passed for both Python files and
  the YAML launcher. The module SHA-256 pinned by the launcher is
  `e330d7acdd0c8eb1424a03c6aac7337b69e362d347737e33592c1b635b41549b`.
- Next action: snapshot and arm the report handoff behind the existing sampled
  trace gate, then continue monitoring producer and training milestones.

### 2026-08-20 09:58 UTC - CSP-028 report handoff corrected and armed

- Fail-closed attempts: Sky job 3 exited immediately because `sky exec`
  retained the producer's original workdir snapshot and the newly committed
  report module was absent. The exact module was then staged as one additive
  file and verified at its pinned SHA-256. Job 4 passed that code gate but
  exposed that the exec shell had not inherited Sky's AWS CLI path; it entered
  its sleep branch without reading or writing data.
- Correction: commit `39c005487930ff7f8fac622e3068e96dec036c47`
  exports the established Sky/miniforge/local runtime path explicitly. Only
  the broken report waiter job 4 was cancelled. Neither producer, either
  sampled-trace handoff, nor model training was interrupted or modified.
- Armed handoff: Sky job 5,
  `issue-473-post-projection-report-after-trace`, passed the exact module hash
  and environment gates, called S3 successfully, received the expected 404
  for the still-absent exact trace report, and entered the ten-minute
  fail-closed wait at 09:57:08 UTC. Its output namespace is
  `s3://oa-bolinas/snakemake/analysis/issue473/results/39c005487930ff7f8fac622e3068e96dec036c47/d43f059c7b0cb8efd5e2396a2bd9e085623a1731/bf8367c285f955407cfb2dba6102661b2e528261b64fe52095d52a688cd6d039/projection_report/`.
- Producer milestone: the fast producer reached 500/10,354 steps at 09:57
  UTC. Its atomic HAL stage had reached 1,130,853,826,560 of
  1,262,706,573,453 bytes; 75.1 GiB memory and 1.4 TiB NVMe remained
  available, with load 23.69 on 48 vCPUs. No producer error was observed.
- Next action: continue event-driven monitoring. Job 5 may stage or analyze
  data only after job 2 has produced the exact sampled-trace report.

### 2026-08-20 10:09 UTC - CSP-029 fast HAL staging complete

- Exact stage receipt: the additive producer atomically completed the local
  HAL at 10:02 UTC with the expected size of 1,262,706,573,453 bytes. The
  temporary partial path disappeared and the final
  `/mnt/nvme/vertebrate_projection/447-mammalian-2022v1.hal` path became
  available to the workflow. No persistent EBS cache was created.
- Projection transition: the scheduler immediately entered the parallel HAL
  liftover phase and reached 907/10,354 durable steps by 10:09 UTC. A live
  process audit found 23 `halLiftover` processes alongside MultiZ work, with
  78,381,740 KiB memory available and no error, traceback, kill, or OOM marker
  in the producer log.
- Independent training: the CDS full-window retry remained healthy on Iris
  with zero failures and zero preemptions. It reached step 2,442/5,000 with
  loss about 1.26 and completed a new rolling recovery checkpoint after the
  previously validated durable step-2,000 native and Hugging Face exports.
- Downstream gates: both exact sampled-trace handoffs and the post-trace
  report handoff remain fail-closed. Dataset build and upload remain gated on
  completed producer QC, manual examples, sampled trace, and report review.
- Safety: no held-out VEP label, prediction, effect measurement, or aggregate
  metric was accessed.

### 2026-08-20 11:33 UTC - CSP-030 core projection batched and #417 baseline reused

- Scope decision: the decision-relevant full-scale comparison is only
  `full_window` versus `center_1`; the 17, 33, 65, and 129 bp policies and the
  pilot are deferred. The two required request sets are now namespaced into
  one HAL input per species and one combined request table per MAF chromosome.
  Existing producer rules and their shared S3 outputs remain unchanged.
- Real-data equivalence: 100 exact center-1 requests and 100 exact enhancer
  full-window requests were projected both separately and together against
  the staged 1,262,706,573,453-byte HAL for `Mus_musculus`. Splitting the
  combined output reproduced both independent outputs byte for byte. The
  center pair shared SHA-256
  `b5fc2b76ba789a48fa9da4b54724676765802a72b09c0c6cf5db818f81c7f9af`;
  the enhancer pair shared
  `8ee5cb276a1bb368848c844ba2310c9d11f85b65f823952b1c95ba3d40ff2d94`.
- Additive implementation: draft PR #477 commit
  `d0e5380a46cd66d4c42d763b3c42da1150c92073` contains the standalone
  batched prefill workflow and hardened NVMe Sky launcher. Its remote
  preflight built the exact 634,926-anchor requests and resolved a 135-job
  prefill DAG: 107 HAL species, 24 MAF chromosomes, one combined request BED,
  and one fail-closed completion manifest.
- Producer transition: obsolete duplicate-call fast producer job 1 was
  cancelled only after synthetic tests, real HAL equivalence, request
  construction, and remote DAG preflight passed. The original producer was
  left running as backup. Batched producer job 7 is running on exact commit
  `d0e5380a`; job 6 failed in setup before data work because of an overly
  strict `uv --version` string comparison, which commit `d0e5380a` corrected.
- Training correction: the exp473 `cds_full_window` run duplicated the exact
  #417 dataset revision and matched recipe, so it was stopped around step
  2,900 and will not be used. Evaluation now pins the existing #417 root
  `gs://marin-us-east5/checkpoints/dna-exp417-cds-combined-vertebrates-p255m-b2m-5k/2026.08.01`.
  Its verified Hugging Face trajectory is steps 1,000 through 4,500 at
  500-step intervals plus terminal step 4,999. Only `cds_center_1`,
  `enhancer_full_window`, and `enhancer_center_1` remain trainable.
- Verification: all 24 independently locked experiment tests passed in 4.78
  seconds; peak RSS was 503,216 KiB. Changed-file YAML, whitespace, Ruff, and
  formatting hooks passed after their mechanical rewrite. No held-out VEP
  label, prediction, effect measurement, or aggregate metric was accessed.
- Next action: monitor batched projection through its manifest and unchanged
  downstream QC, review the sampled trace/report, pin publication to the
  resulting producer, then publish privately and launch only the three new
  training arms.

### 2026-08-20 11:54 UTC - CSP-031 corrected handoffs and issue contract

- Batched progress: Sky job 7 reached 35/135 prefill steps at 11:50 UTC with
  no observed error. Each completed HAL job uploaded the two namespaced policy
  outputs and one per-species receipt from a single combined request call.
- Trace handoff: draft PR #477 commit
  `c30c533e55f4626eb6f8e9c0d4b20ec004ca24a7` adds a durable launcher that
  pins producer commit `d0e5380a46cd66d4c42d763b3c42da1150c92073`,
  verifies the unchanged trace implementation hashes, and waits only for the
  pilot-free core-completion target. Sky job 8 is armed and currently waiting
  fail-closed for that target.
- Report handoff: experiment commit
  `f6f8c4d087b4d240282dcc61850e6783fa90c7d9` defines the corresponding
  producer-pinned report launcher. Its remote submission was not performed:
  the execution safety gate requires destination-specific authorization for
  reading the private producer outputs and uploading the derived QC report.
  No alternate route was attempted, and this does not block producer or trace
  progress.
- Research record: issue #473 now states the active contract directly: the
  wider-landmark pilot is deferred; batching is one HAL call per species and
  one MAF scan per chromosome; the four-arm comparison reuses the exact #417
  CDS full-window checkpoints and trains only three new arms at nine common
  checkpoints.
- Safety: no dataset was published, no training is running, and no held-out
  VEP label, prediction, effect measurement, or aggregate metric was accessed.
- Next action: continue the batched producer through its completion manifest,
  review full projection QC and sampled trace, and resolve the report-upload
  authorization only when the exact source and destination are ready to name.

### 2026-08-20 14:05 UTC - CSP-032 batched producer recovered and verified

- Batched execution: exact producer job 7 completed the 135-job prefill and
  2,603 of 2,610 unchanged downstream steps. It scheduled one HAL liftover per
  mammal and one MAF candidate scan per chromosome; no duplicate liftover or
  candidate-scan rules appeared in the downstream DAG.
- Failure and durable boundary: Ray killed only the final
  `issue_473_fixed_full_diagnostics` process after it reached about 85 GiB RSS
  on a 92.77 GiB worker. All projection, sequence, intersection, and earlier
  QC outputs were already durable. A target-pinned dry-run proved exactly five
  remaining jobs: two enhancer dataset writes, full diagnostics, manual
  inspection, and the core completion target.
- Dry-run correction: an initial read-only check accidentally synced the
  experiment branch to the fast producer and failed immediately with
  `MissingRule`; it ran no data job and mutated no S3 output. The PR worktree
  was restored immediately, and the corrected dry-run returned the exact
  five-job boundary above.
- Additive recovery: draft PR #477 commit
  `82918df3b7b11bf643548511e7f2ef155de59190` added a QC-only launcher whose
  allowed-rule list could not schedule projection or preprocessing. Iris job 1
  on a 256 GB worker completed the five jobs, peaked at 80,657,216 KiB RSS,
  and passed a terminal target-only dry-run reporting that all requested files
  were present and current.
- Safety: producer identity remains exact commit
  `d0e5380a46cd66d4c42d763b3c42da1150c92073` and config SHA-256
  `bf8367c285f955407cfb2dba6102661b2e528261b64fe52095d52a688cd6d039`.
  No held-out VEP label, prediction, effect measurement, or aggregate metric
  was accessed.

### 2026-08-20 14:46 UTC - CSP-033 sampled trace reviewed and publication build started

- Trace recovery: the original trace waiter was a 0.13 GB collateral victim
  of the producer worker's final-diagnostic OOM. Relaunched job 11 passed the
  completed-producer gate immediately and finished all 645 trace steps. The
  staged full-window and center-1 sequence tables were respectively
  5,888,108,590 and 6,092,534,094 bytes with SHA-256 values
  `d45a71aded8ad4aca2939aa0e62194297e85b259617e128b7f27f9d0496f32c0`
  and `2e0113bf685809d02438636b4a3ff4487b3a025be92eb2d9f2cf35b555b56e04`.
- Trace evidence: the deterministic sample contains 5,328 rows from 4,288
  anchors. Exact named-PSL measurement succeeded for 725 of 742 center-1 rows
  and 4,518 of 4,586 full-window rows; the rest are explicitly classified as
  no reverse mapping or off-expected-locus. Every source and emitted interval
  is exactly 255 bp. Anchor-clustered paired center-minus-full exact-coverage
  intervals include zero in all five regions.
- Manual review: chromosome and strand agreement are 1.0 across the
  74,524,203-row accepted union, target-locus overlap is effectively 1.0, and
  median emitted-center displacement is 3 bp. Raw named PSL rows for the
  reverse-strand and plus-strand ZRS examples reproduce the expected human
  anchor; the paired Aplodontia full/center rows are byte-identical. Direct
  `hal2fasta` spot extraction was abandoned because the 1.26 TB HAL performs
  a near-global scan for a 255 bp slice; its temporary processes were stopped
  and memory fully recovered without modifying pipeline or S3 artifacts.
- Publication gate: draft PR #477 commit
  `d7c6a93cf65571ddce48775b839b1d1a6591f8b2` changes only the publication
  config from obsolete producer `d43f059c` to reviewed producer `d0e5380a`.
  All 123 locked project tests and changed-file hooks passed. Commit
  `770b4b498b89d43b0a8337dbc4ba03f918cae1fe` then corrected only the new
  launcher's pinned `uv` version parsing after build job 1 failed safely in
  setup before reading data.
- Current build: non-upload Iris job 2 is building and validating exactly the
  three new private payloads. Its DAG contains three train shuffles, three
  validation shuffles, 195 compression jobs, three cards, one manifest, and
  one build target; Hugging Face upload rules are excluded. Only
  `cds_center_1`, `enhancer_full_window`, and `enhancer_center_1` will train;
  the #417 CDS full-window checkpoints remain the fourth evaluation arm.

### 2026-08-20 15:54 UTC - CSP-034 public publication complete and training gate cleared

- Public-only correction: “private payloads” in CSP-033 referred to the
  retained local/S3 build tree, not private Hugging Face repositories. No
  private Hugging Face upload occurred. The additive upload boundary creates
  repositories with `private=False`, refuses an existing private repository,
  and rechecks public visibility after its exact manifest verification.
- Validated build: additive publication v3 completed all 218 steps on the
  retained worker. The archived manifest is
  `s3://oa-bolinas/snakemake/vertebrate_projection_dataset/results/v1/f05f2acf085fb9f841c629d2236faa407a909ec0/921884ff6099a0fde06a2f333dc8f8967b5ca2e4d4218e8d8083e9f499000a92/issue_473_publication_v3/validation/hf_publication_manifest.archived.json`.
  The isolated publication project passed all 130 tests, and its public-only
  four-job dry-run succeeded before upload.
- Exact public revisions: `cds_center_1` is
  `marin-dna/vertebrate-v1-issue473-center1-cds@4d9a04ab6c4a6e445345fe35fbe2be41b43e7938`;
  `enhancer_full_window` is
  `marin-dna/vertebrate-v1-issue473-fullwindow-ccre-enhancer-centered@ffb9c63fae72311fb457640af9c8365b84f0edf8`;
  and `enhancer_center_1` is
  `marin-dna/vertebrate-v1-issue473-center1-ccre-enhancer-centered@23d1531f63998b5716e7895a74437e0568186bd1`.
  Unauthenticated exact-revision API reads returned `private=false` and
  `gated=false` for all three.
- Duplicate-run audit: Iris history contains only the stopped/failed
  `exp473-cds-full-window` attempts and preflights, with no center-1 or
  enhancer training job. GCS likewise contains only the obsolete
  `dna-exp473-0p25b-cds_full_window-v1` checkpoint namespace. That duplicate
  remains excluded; the fourth arm is the exact #417 checkpoint root.
- Training handoff: the experiment now source-pins the three public revisions.
  All 24 locked tests passed, and all three non-mutating Marin plans resolved
  distinct required run IDs at version `2026.08.20`. The next action is to
  snapshot this commit and launch only those three new arms on Iris.

### 2026-08-20 16:05 UTC - CSP-035 three public-data training arms launched

- Exact snapshot: permanent experiment branch
  `codex/exp473-center-seeded-projection-training` is pushed through commit
  `cb4249188dc6add8c6b2587ac0a2185f24450930`. The source hard-pins all
  three public, ungated Hugging Face dataset revisions from CSP-034; no
  dataset revision can be supplied at launch time.
- Fail-closed attempts: initial coordinators
  `/ubuntu/exp473-cds-center-1`,
  `/ubuntu/exp473-enhancer-full-window`, and
  `/ubuntu/exp473-enhancer-center-1` failed before artifact-graph creation
  because the remote shell expanded a missing `WANDB_API_KEY`. They launched
  no tokenizer, data, accelerator, checkpoint, or model-training child.
- Corrected launches: coordinators
  `/ubuntu/exp473-cds-center-1-v2`,
  `/ubuntu/exp473-enhancer-full-window-v2`, and
  `/ubuntu/exp473-enhancer-center-1-v2` are running with distinct registered
  run IDs. The key was forwarded from the existing local netrc without being
  printed or written to the experiment branch.
- Public-data proof: all three tokenizer coordinators reported unauthenticated
  Hugging Face Hub requests while resolving the exact source-pinned public
  revisions. At 16:05 UTC, each arm had 64 active training-tokenization
  workers; enhancer full-window had completed 33/64 shards, enhancer center-1
  30/64, and the larger CDS center-1 corpus had all 64 shards running. Iris
  reported zero failures and zero preemptions for these corrected launches.
- Duplicate boundary: the obsolete `cds_full_window` namespace is not running
  and will not be resumed. The exact #417 checkpoint root remains the fourth
  evaluation arm. No wider policy, pilot, or additional seed is active.
- Safety: held-out even-autosome/Y VEP labels, predictions, effect
  measurements, and aggregate metrics remain untouched. The next terminal
  gate is successful training through step 4,999 for all three new arms,
  followed by paired projection loss and official development-only evaluation.

### 2026-08-20 16:17 UTC - CSP-036 paired-loss producer pin corrected

- Pre-evaluation audit: the issue-specific paired intersection-loss config
  still named the earlier `f764b7f1` producer snapshot. Read-only S3 `HEAD`
  requests returned 404 for all four expected chromosome-18 intersection
  views in that namespace, so the workflow would have failed before scoring.
  No intersection-loss or VEP evaluation job had launched.
- Correct provenance: all four inputs exist under final batched producer
  `d0e5380a46cd66d4c42d763b3c42da1150c92073` and config SHA-256
  `bf8367c285f955407cfb2dba6102661b2e528261b64fe52095d52a688cd6d039`.
  S3 reports full-object CRC64NVME checksums and byte sizes for each CDS and
  enhancer-centered full-window/center-1 validation view.
- Additive correction: only the new experiment-local producer constant and
  its explicit identity assertion changed. No established Snakemake rule,
  shared output path, producer artifact, dataset, checkpoint, or active
  training job changed.
- Verification: all 24 locked experiment tests passed in 4.78 seconds with
  peak RSS 494,540 KiB while holding the shared heavy-work lock; changed-file
  pre-commit hooks and `git diff --check` passed.
- Safety: this check used only unlabeled chromosome-18 projection sequences.
  Held-out VEP labels, predictions, effect measurements, and aggregate
  metrics remain untouched.

### 2026-08-20 16:25 UTC - CSP-037 batched projection-report gate corrected

- Fail-closed run: authorized Sky job 12 on retained cluster
  `issue-473-fast-producer` verified the exact report-module hash, final
  producer identity, completed sampled-trace gate, and staged both accepted
  tables. It then stopped before analysis because the launcher expected 270
  new-arm rejection files but found 942. It wrote no analysis output.
- Diagnosis: read-only S3 listings show the final batched layout has 107 HAL
  rejection files, 107 HAL sequence-rejection files, 700 MultiZ rejection
  files (one per chromosome/species), and 28 MultiZ sequence-rejection files
  for each new arm, totaling 942. The reused #417 baseline has 270 files.
- Additive correction: the experiment-local launcher now requires exactly 942
  files for each new arm, 1,212 full-window rejection arguments after adding
  the 270-file baseline, and 942 center-1 arguments. No producer, shared rule,
  or S3 artifact changed.
- Verification: a new test pins the final producer and all four count gates.
  All 25 locked experiment tests passed in 4.72 seconds with 494,680 KiB peak
  RSS while holding the shared heavy-work lock; changed-file hooks passed.
- Next action: snapshot this correction and rerun only the additive report.

### 2026-08-20 16:29 UTC - CSP-038 report argument-array correction

- Fail-closed rerun: Sky job 13 passed the corrected 942-file gates for both
  new arms, then stopped before analysis because the shell arrays contain two
  elements per file: the repeated option and its path. The observed
  full-window array length was therefore 2,424, not 1,212. No output was
  written.
- Correction to CSP-037 wording: 1,212 and 942 are the full-window and
  center-1 file counts, respectively. The exact argument-array lengths are
  2,424 and 1,884. The launcher and regression test now assert those values.
- Scope: only the additive report launcher's fail-closed accounting changed.
  No data, projection, shared rule, training job, or evaluation result changed.

### 2026-08-20 16:39 UTC - CSP-039 fixed-catalog report boundary added

- Full-analysis diagnosis: Sky job 14 completed staging and projection
  accounting, then stopped while formatting Markdown because the reused #417
  full-window rejection inventory contains 4,222,019 rejected pairs from
  anchors outside issue #473's fixed catalog. Those rows intentionally have no
  #473 region label; the ten in-scope policy-by-region rows had complete
  requested, accepted, and rejected accounting.
- Additive correction: a new `fixed_catalog_report` entry point and separate
  `projection_report_batched_v2.yaml` launcher leave the existing report
  module, producer, rules, rejection artifacts, and prior output namespaces
  unchanged. The wrapper asserts that every unlabeled row is rejected-only
  `full_window` evidence from the immutable superset, removes only those rows,
  and requires complete fixed-catalog labels and requested counts before
  writing `projection_report_v2`.
- Verification: the workflow project passed all 120 locked tests, including a
  regression for the out-of-scope rejection case, in 8.12 seconds with
  291,740 KiB peak RSS. Its 78-job default Snakemake dry-run succeeded. The
  experiment project passed all 26 locked tests in 4.74 seconds with 494,532
  KiB peak RSS, and changed-file pre-commit hooks passed. All local checks held
  the shared-node heavy-work lock and respected bounded thread settings.
- Next action: snapshot and launch only the additive v2 report against final
  producer `d0e5380a46cd66d4c42d763b3c42da1150c92073`, then verify its immutable
  S3 receipts before releasing the retained report worker.

### 2026-08-20 16:43 UTC - CSP-040 retained-worker code sync made explicit

- Fail-closed launch: Sky job 15 stopped at its first module-hash check because
  the retained worker still had the earlier workdir snapshot and therefore did
  not contain `fixed_catalog_report.py`. It performed no S3 staging, analysis,
  or publication.
- Additive correction: new successor launcher
  `projection_report_batched_v3.yaml` declares `workdir: .`, matching the
  issue's evaluation and analysis launchers, so Sky transfers the exact local
  snapshot before running. The failed v2 launcher and its output namespace
  remain unchanged; v3 uses a separate local root and `projection_report_v3`
  S3 namespace.
- Verification: all 27 locked experiment tests passed in 4.75 seconds with
  494,780 KiB peak RSS under the shared-node lock, and changed-file hooks
  passed. The v3 regression pins explicit workdir sync, final producer
  identity, fixed-catalog module use, and exact rejection-array counts.

### 2026-08-20 16:53 UTC - CSP-041 fixed-catalog report complete; data workers released

- Successful report: Sky job 16 ran exact snapshot
  `ff8aaa8a8479e074751264c880d7034167a1654d`, passed both module hashes and
  the final trace gate, staged the 5.89 GB full-window and 6.09 GB center-1
  accepted tables, and completed in 2 minutes 17 seconds. The analysis process
  reached 11,654,236 KiB observed RSS while the 92 GiB worker retained 78 GiB
  available memory.
- Immutable receipt: five outputs were independently restored from
  `s3://oa-bolinas/snakemake/analysis/issue473/results/ff8aaa8a8479e074751264c880d7034167a1654d/d0e5380a46cd66d4c42d763b3c42da1150c92073/bf8367c285f955407cfb2dba6102661b2e528261b64fe52095d52a688cd6d039/projection_report_v3/`.
  Their restored SHA-256 values exactly match the manifest, which records
  0-based half-open coordinates, `fixed_catalog_scope=true`, policy widths 1
  and 255, and 942 center-1 versus 1,212 full-window rejection inputs.
- Projection result: center-1 recovered more accepted fixed-grid pairs for CDS
  (+1,068,933; 86.78% versus 84.10%), ncRNA exon (+1,359,264; 77.85% versus
  67.64%), and UTR3 (+250,768; 79.76% versus 76.99%). Full-window recovered
  more for enhancer-centered (+139,960; 80.72% versus 79.83%) and TSS/UTR5
  (+242,992; 77.80% versus 74.66%). Center-1 is therefore not a universal
  recovery improvement; it wins three of five declared regions and loses two.
- Cleanup: after S3 receipt verification, completed clusters
  `issue-473-fast-producer`, `issue-473-hf`, and `issue-473-qc-resume` were
  terminated. On `vertebrate-project`, exact jobs 11 and 14 were still
  producing or waiting on the superseded `f764b7f1` unbatched namespace and
  included excluded pilot policies. They were cancelled explicitly, then the
  cluster was terminated. No final `d0e5380a` artifact was removed.
- Training state at 16:52 UTC: CDS center-1 task
  `/ubuntu/exp473-cds-center-1-v2/run_levanter_train_lm-25c8f771` is running;
  enhancer full-window and enhancer center-1 remain pending for accelerator
  capacity. Iris reports no task error for any of the three. All inputs remain
  exact revision-pinned, public, and ungated Hugging Face datasets.

### 2026-08-20 17:03 UTC - CSP-042 matched exposure and evaluation-input audit

- Fixed training budget: every arm presents 40,960,000 sequences and
  10,485,760,000 tokens. The #473 row counts come from the restored archived
  publication manifest at SHA-256
  `0e066e4bbfb7be101d1f1e440f0880e2abe7c64d5978cae687503fdb0af59ab3`;
  the reused CDS full-window count is the public #417 receipt.

| Arm | Train rows | Biological pairs | Published-row epochs | Pair presentations | Tokens/published row |
|---|---:|---:|---:|---:|---:|
| `cds_full_window` | 66,552,602 | 33,276,301 | 0.615453 | 1.230906 | 157.555974 |
| `cds_center_1` | 68,657,166 | 34,328,583 | 0.596587 | 1.193175 | 152.726374 |
| `enhancer_full_window` | 24,889,396 | 12,444,698 | 1.645681 | 3.291362 | 421.294273 |
| `enhancer_center_1` | 24,616,580 | 12,308,290 | 1.663919 | 3.327838 | 425.963314 |

  Published rows include one forward and one reverse-complement row for each
  biological pair; the table keeps those two exposure interpretations
  separate.
- Public-only evaluation inputs: unauthenticated exact-revision API reads
  confirmed `marin-dna/evals_mendelian_traits@4aed58e5`,
  `marin-dna/evals_complex_traits@22f86a89`, and
  `marin-dna/evals_sge@225d3d1e` are each `private=false` and `gated=false`.
  The configured historical `bolinas-dna` aliases redirect to those same
  public `marin-dna` repositories and exact SHAs.
- Downstream gate audit: generated evaluation remains hard-coded to the
  official `train` split and nine steps 1,000 through 4,500 plus 4,999. Paired
  Mendelian AUPRC/Group SMD requires row identity and aligned match-group
  resamples; CDS SGE and enhancer Complex are collected as official
  development endpoints. The separate paired-loss workflow reads only
  producer-pinned unlabeled chromosome-18 intersection sequences and resamples
  human anchors.
- Runtime evidence: the first CDS center-1 TPU attempt compiled, loaded the
  exact realized cache, completed step 0, then lost its worker to an Iris
  reconciliation-threshold preemption. Iris records zero failures, one
  preemption, a pending retry with allowance 100, and no completed checkpoint.
  W&B run
  [`dna-exp473-0p25b-cds_center_1-v1`](https://wandb.ai/gonzalobenegas/marin/runs/dna-exp473-0p25b-cds_center_1-v1)
  exists with no metric summary yet. No duplicate coordinator was launched.

### 2026-08-20 17:26 UTC - CSP-043 stalled training migrated to central1

- Capacity diagnosis: after the first CDS preemption, all three exact training
  children remained pending in `us-east5`. Iris repeatedly replaced a booting
  preemptible v5p-8 slice before it reached ready state. In contrast,
  `us-central1` had a ready v5p-8 slice with no task assignment.
- No-duplicate migration: a prefix dry-run selected exactly the three v2
  coordinators and their three training children. Those six jobs were then
  terminated, and their coordinator and training-child task records were
  confirmed terminal before any replacement was submitted.
- Exact replacement coordinators are
  `/ubuntu/exp473-cds-center-1-v3`,
  `/ubuntu/exp473-enhancer-full-window-v3`, and
  `/ubuntu/exp473-enhancer-center-1-v3`, each constrained to `us-central1`.
  They preserve version `2026.08.20`, all source-pinned public dataset
  revisions, the same three arm keys, run IDs, W&B identities, checkpoint
  roots, model/optimizer/seed settings, and 5,000-step budgets.
- Scope remains exactly CDS `center_1`, enhancer `full_window`, and enhancer
  `center_1`; the exact #417 CDS full-window checkpoint remains reused. No
  wider projection policy, pilot, additional seed, private or gated Hugging
  Face repository, or held-out even-autosome/Y VEP datum was introduced.

### 2026-08-20 17:31 UTC - CSP-044 child-region inheritance corrected

- Correction to CSP-043: the three v3 *coordinators* were constrained to
  `us-central1`, but inspection of selected non-secret `job_config` fields
  showed that their training children retained the experiment's explicit
  `us-east5` resource constraint. A coordinator's region does not override an
  explicitly pinned child resource. All three children were still pending and
  had consumed no TPU time.
- A second prefix dry-run again selected exactly three v3 coordinators and
  three training children. Those six jobs were terminated before modifying or
  relaunching the experiment, preserving the no-duplicate boundary.
- The experiment-local launcher now accepts a bounded
  `EXP473_TPU_REGION` override, defaults to the established `us-east5`, and
  permits only `us-east5` or `us-central1`. The selected value controls the
  training child's resource constraint and is forwarded into its environment
  for provenance. A regression test covers the default, central1 override,
  and fail-closed invalid value; the README records that coordinator and child
  placement are distinct.
- Verification: the first test run exposed that Marin represents resources in
  the realized pod config with a fingerprint placeholder; the assertions were
  corrected to inspect the actual `train_resources` runtime argument. The
  complete locked suite then passed 28/28 tests in 5.95 seconds with peak RSS
  494,660 KiB under the shared-node lock, 9.54 GiB available memory, and 0.47
  one-minute load at start.

### 2026-08-20 17:42 UTC - CSP-045 central1 data-locality boundary

- Fail-closed evidence: all three v4 child configs correctly pinned
  `us-central1`. CDS center-1 then exited before step 0 because its realized
  cache remained in `marin-us-east5`; Marin rejects a cache and TPU in
  different regions. Enhancer full-window was stopped during setup and
  enhancer center-1 while pending, so neither trained. A prefix dry-run
  selected exactly those remaining coordinator and child jobs before stop.
- Size and location audit: the standard buckets are `marin-us-east5` in
  `US-EAST5` and `marin-us-central1` in `US-CENTRAL1`. Exact east5 caches are
  11.41 GiB for CDS center-1, 4.18 GiB for enhancer full-window, and 4.13 GiB
  for enhancer center-1. The issue-specific central1 prefix is empty.
- Artifact receipts embed their realized output and cache paths, so blindly
  copying an east5 cache tree would not make it a valid central1 artifact. The
  clean route is to rebuild the same source-pinned caches from the exact
  public, ungated Hugging Face revisions under an additive central1 prefix;
  this preserves the original east5 artifacts unchanged.
- The launcher now fails before graph creation unless `MARIN_PREFIX` uses the
  artifact bucket matching `EXP473_TPU_REGION`. This prevents another
  accelerator-side locality failure and makes the central1 rebuild/checkpoint
  namespace explicit.
- Verification: all 28 locked experiment tests passed in 6.25 seconds with
  peak RSS 494,408 KiB under the shared-node lock; changed-file hooks and
  `git diff --check` passed. The next launch will use
  `gs://marin-us-central1/MarinDNA/exp473_center_seeded_projection` for both
  rebuilt caches and new checkpoints.

### 2026-08-20 18:10 UTC - CSP-046 central1 cache rebuilds and training launch

- Three exact v5 coordinators rebuild the existing arm definitions beneath
  the additive, region-local prefix
  `gs://marin-us-central1/MarinDNA/exp473_center_seeded_projection`. Each reads
  its exact pinned public, ungated Hugging Face dataset without authentication;
  no private or gated repository and no Hugging Face token is used.
- CDS center-1 and enhancer full-window tokenization completed. CDS training
  child `/ubuntu/exp473-cds-center-1-v5/run_levanter_train_lm-dfbe430e` is
  running, while enhancer full-window child
  `/ubuntu/exp473-enhancer-full-window-v5/run_levanter_train_lm-a815366a` is
  queued. Enhancer center-1 tokenization remains in progress.
- Selected non-secret child configuration confirms both created training jobs
  request four TPU cores on `v5p-8` in `us-central1`, with 100 preemption
  retries. CDS startup reports the expected central1 cache, compilation-cache,
  and checkpoint paths and is compiling from scratch. The run retains W&B ID
  [`dna-exp473-0p25b-cds_center_1-v1`](https://wandb.ai/gonzalobenegas/marin/runs/dna-exp473-0p25b-cds_center_1-v1).
- Scope is unchanged: the three new arms are CDS `center_1`, enhancer
  `full_window`, and enhancer `center_1`; the exact #417 CDS full-window arm is
  reused rather than retrained. Evaluation remains development-only at steps
  1,000 through 4,500 and 4,999, with even-autosome/Y labeled VEP data held
  back.

### 2026-08-20 18:56 UTC - CSP-047 first durable CDS checkpoint

- CDS center-1 reached step 500 on Iris attempt 0. The trainer committed the
  native 2.85 GiB optimizer-state checkpoint under
  `gs://marin-us-central1/MarinDNA/exp473_center_seeded_projection/checkpoints/dna-exp473-0p25b-cds_center_1-v1/2026.08.20/checkpoints/step-500`
  and resumed training at step 502.
- The matching Hugging Face-format export completed under the same immutable
  root at `hf/step-500`. An independent object-name check found `config.json`,
  the 1.02 GB `model.safetensors`, `tokenizer.json`, and
  `tokenizer_config.json`; the native path contains its manifest and metadata.
- Step-500 normal validation loss was 1.320. The preceding W&B training sample
  at global step 499 reported loss 1.314, mean MFU 48.1%, and 1,048,576,000
  cumulative tokens. This ordinary per-arm validation metric is an operating
  check, not a cross-policy comparison because policy validation rows differ.
- Runtime provenance in the validation log retains region `cds`, policy
  `center_1`, public dataset revision
  `4d9a04ab6c4a6e445345fe35fbe2be41b43e7938`, and pinned tokenizer revision
  `a73e9d9ee636f722b4c378703c9e2997857809b2`. The first official development
  evaluation point remains step 1,000. Both enhancer arms remain queued for
  the same `us-central1` `v5p-8` resource with zero failures or preemptions.

### 2026-08-20 19:41 UTC - CSP-048 first official CDS checkpoint

- CDS center-1 reached and passed official evaluation step 1,000 on Iris
  attempt 0. The native optimizer-state checkpoint committed at
  `checkpoints/step-1000`, and the matching 1.02 GB Hugging Face-format export
  completed at `hf/step-1000` beneath the immutable central1 checkpoint root.
- Independent object-name checks found the native manifest and metadata plus
  the HF `config.json`, `model.safetensors`, `tokenizer.json`, and
  `tokenizer_config.json`. The trainer continued through at least W&B global
  step 1,007 after both saves.
- Step-1,000 normal validation loss was 1.3050305843. W&B at step 1,007
  reported training loss 1.2969002724, mean MFU 48.2%, and 2,113,929,216
  cumulative tokens. These per-arm losses remain operational checks only; the
  paired chromosome-18 intersection workflow is the registered cross-policy
  loss comparison.
- This is the first of nine official development-only checkpoint steps. No
  downstream evaluation is launched yet because all four model trajectories
  must expose the complete common step set. Both enhancer training children
  remain queued with zero failures or preemptions.

### 2026-08-21 00:16 UTC - CSP-049 enhancer capacity fallback entered training

- The two enhancer v5 children eventually received central1 `v5p-8` capacity
  but exited before step 0 because W&B rejected an over-64-character source
  tag. Commit `d8ec668c7877a6a58184636c118828a7dfafd5c6` bounds long tags while
  retaining a readable prefix and SHA-256 suffix. Subsequent v6 and v8
  coordinators failed before graph creation because their launch environment
  did not forward the required W&B credential; neither created a child, used a
  TPU, nor wrote a checkpoint. The v7 launch forwarded the credential but its
  east5 `v5p-8` children remained queued for capacity. Each superseded job and
  child was confirmed terminal before its replacement was submitted.
- The exact v9 children use available east5 `v6e-4` resources without changing
  model, optimizer, seed, batch size, step budget, public dataset revision,
  cache, run identity, or checkpoint identity. Enhancer full-window child
  `/ubuntu/exp473-enhancer-full-window-v9-v6e/run_levanter_train_lm-0382cac7`
  and enhancer center-1 child
  `/ubuntu/exp473-enhancer-center-1-v9-v6e/run_levanter_train_lm-57cf5f85`
  each run on four TPU chips. At 00:15 UTC they had reached steps 431 and 346,
  respectively, with recent training losses between 1.33 and 1.34.
- Both enhancer runs reuse their complete east5 caches and read the same exact
  public, ungated Hugging Face revisions without a Hugging Face token. Their
  W&B runs are
  [`enhancer_full_window`](https://wandb.ai/gonzalobenegas/marin/runs/dna-exp473-0p25b-enhancer_full_window-v1)
  and
  [`enhancer_center_1`](https://wandb.ai/gonzalobenegas/marin/runs/dna-exp473-0p25b-enhancer_center_1-v1).
- CDS center-1 remained on central1 `v5p-8`, reached step 3,853, and has
  independently verified native and Hugging Face checkpoint directories at
  steps 1,000, 1,500, 2,000, 2,500, 3,000, and 3,500. Exact normal-validation
  losses through step 3,000 are 1.3050305843, 1.2961095572, 1.2911146879,
  1.2921013832, and 1.2987722158. These remain per-arm operating checks; no
  policy comparison or held-out labeled VEP datum was read.

### 2026-08-21 00:16 UTC - CSP-050 evaluation-source snapshot

- Snapshot commit
  `84819ddbc8b9ba9bcf511e94ca2cd8e9cd94d673` is pushed to
  `codex/exp473-center-seeded-projection-training`. It records the bounded
  east5 accelerator choice (`v5p-8` or `v6e-4`, with `v5p-8` still the
  default), while central1 remains `v5p-8` only. The successful v9 accelerator
  selection is now reproducible without the launch-time resource substitution.
- Paired development-evaluation plots now share their y-axis within each
  region/metric figure, use sentence-case labels and square panels, and emit
  SVG only. Matplotlib is a base dependency because analysis workers install
  the base locked project. The first 30-test run exposed the missing dependency;
  after `uv lock`, `uv run --locked pytest` passed 30/30 tests in 8.01 seconds.
  The guarded run started with 8.68 GiB available memory and load 0.27; peak
  RSS was 543,844 KiB.
- The plotting regression rendered all four expected SVGs and no PNGs. Direct
  SVG inspection found a 607.86 × 1,131.94 point canvas, all titles, axis
  labels, subset labels, and the paired-95%-interval caption embedded, plus
  eight in-canvas 219.56 × 219.56 point clipping rectangles per figure. The
  image-view helper could not open the files because its nested Bubblewrap
  loopback setup failed before file access, so geometry and embedded-label
  inspection were used for this synthetic render gate.
- Remaining official evaluation and paired chromosome-18 outputs will use
  `84819ddbc8b9ba9bcf511e94ca2cd8e9cd94d673` as their immutable experiment
  commit. Evaluation remains blocked on all three new trajectories exposing
  the common nine-step checkpoint set through step 4,999.

### 2026-08-21 00:31 UTC - CSP-051 first durable enhancer checkpoints

- Enhancer full-window reached step 500 on v9 attempt 0. Native optimizer state
  committed beneath `checkpoints/step-500`, and the matching Hugging
  Face-format export completed beneath `hf/step-500` at the exact east5 root
  recorded in CSP-049. The trainer resumed through step 504.
- Enhancer center-1 reached the same milestone on v9 attempt 0. Its native and
  Hugging Face-format step-500 checkpoints committed at the corresponding
  east5 root, and the trainer resumed through step 502. Both jobs emitted the
  expected warning that their Hugging Face reads were unauthenticated, matching
  the public-only dataset contract.
- Independent object listings for each enhancer arm found native
  `manifest.json`, `manifest.ocdbt`, `metadata.json`, and `d/`, plus HF
  `config.json`, `model.safetensors`, `tokenizer.json`, and
  `tokenizer_config.json`. Each HF export contains four objects totaling
  1,019,426,427 bytes (972.20 MiB). Step-500 normal validation losses were
  1.331 for full-window and 1.336 for center-1; these are operating checks on
  policy-specific validation rows, not a paired policy comparison.
- CDS center-1 also reached and completed step 4,000 on its central1 v5 run.
  The same native and HF object contract was independently verified, including
  a 1,019,422,904-byte `model.safetensors` file. Step-4,000 normal validation
  loss was 1.317. The CDS trajectory now has seven of nine official checkpoint
  steps; both enhancer trajectories still need all nine official steps from
  1,000 through 4,999 before either downstream workflow launches.

### 2026-08-21 00:43 UTC - CSP-052 official development evaluation started additively

- The immutable experiment config generated from
  `84819ddbc8b9ba9bcf511e94ca2cd8e9cd94d673` was dry-run through the
  unchanged `evals_v2` Snakefile. It records `split: train`,
  `held_out_access: false`, exact evaluation-dataset revisions, the nine
  registered checkpoints, and only the four preregistered arms. The complete
  graph was exactly 36 checkpoint downloads, 72 score cells, 72 metric cells,
  and the aggregate target; no unrelated or held-out rule appeared.
- The guarded local dry-run exited 0 in 16.72 seconds with 1,010,364 KiB peak
  RSS. Because that exceeded the shared node's 500 MiB local-work guideline,
  no further Snakemake DAG construction will run on the shared node; remaining
  dry-runs use the remote evaluator.
- Rather than wait for the enhancer trajectories before beginning all serial
  GPU work, additive cluster `exp473-evaluate-cds` job 1 now targets only
  the already-complete #417 CDS full-window trajectory and CDS center-1 steps
  1,000 through 4,000. Its generated runtime config again reported 36 pinned
  checkpoints and `split=train`; the selected graph contains 16 downloads,
  32 score cells, and 32 metric cells. Outputs retain the final
  commit-keyed `results/{checkpoints,scores,metrics}/exp473-84819.../`
  identities, so subsequent complete evaluation skips them rather than
  overwriting or repeating them.
- SkyPilot selected one AWS us-east-2c `g5.xlarge` spot instance with one
  A10G (estimated 0.36 USD/hour), a 300 GB disk, automatic teardown, and
  commit-clean workdir source. Setup installed the locked `evals_v2`
  environment and authenticated GCS access; checkpoint staging began at
  00:43 UTC. Every requested metric target is either Mendelian or SGE on the
  development split. No held-out labeled VEP datum was requested or read.
- At launch time CDS center-1 had advanced to about step 4,130. Enhancer
  full-window and center-1 continued advancing at about steps 678 and 594,
  respectively, on their east5 v6e-4 children.

### 2026-08-21 01:11 UTC - CSP-053 evaluation boundary incident and isolated repair

- Correction to CSP-052: the first score process called the Hugging Face
  repository dataset builder with `split=train`, but that builder materialized
  both repository splits before selecting train. Its log reported 23,853 train
  rows and 14,888 test rows. This accessed the held-out labeled file and
  violated the development-only boundary despite the generated config. The
  process then failed on the exported `TokenizersBackend` class before model
  inference. It produced no predictions, score parquet, metric parquet, or
  aggregate metric. The cluster was terminated, the ephemeral cache was not
  inspected, and only the 16 staged checkpoint directories remain in the old
  `84819...` namespace.
- Direct runtime inspection also found that the unpinned SkyPilot image had
  NVIDIA driver 535.216.01. The locked CUDA 13 evaluator requires the validated
  issue-462 image with driver 595.71.05. No evaluation job will reuse that
  unpinned image.
- Repair commit `97a5672cfbc72aac1edbac58b05c77e416a8cecf`
  adds an isolated issue-specific Snakefile and output namespace. It downloads
  exactly each pinned public `train.parquet` through the Hugging Face file API,
  opens that one file with the parquet loader, and rejects any labeled row
  outside odd autosomes and chromosome X before calling the unchanged official
  score and metric kernels. It also loads both known tokenizer metadata forms
  through `PreTrainedTokenizerFast` and asserts the exact BOS and vocabulary
  IDs. The maintained S3-backed evals_v2 rules and output paths remain
  unchanged.
- The repaired project passed 35/35 tests on an AWS us-east-1 `m6i.xlarge`
  with Python 3.12.14 in 14.44 seconds; peak RSS was 585,724 KiB. The evaluator
  and projection-loss launchers now pin AMI `ami-0324f0ad73bdcd087` and run the
  existing GPU-runtime smoke gate before constructing a score DAG. The next
  gate is a remote dry-run on that image, followed by one monitored score cell
  whose logs must show only the direct train parquet path.

### 2026-08-21 01:14 UTC - CSP-054 requested origin/main rebase

- The clean experiment branch was rebased onto `origin/main` commit
  `ce72fbe3`. All 79 experiment commits replayed without conflicts. The
  evaluator repair recorded in CSP-053 is now commit
  `4c4848bbfb225054a6dcc05eb9273ad615696460`; the pre-rebase hash in CSP-053
  identifies the same patch before history was rewritten.
- No evaluation output uses either pre-rebase source identity. The generated
  development-evaluation and chromosome-18 configurations will use the final
  post-rebase logbook snapshot commit so every output remains tied to source
  that includes this mapping and the current main guidance.

### 2026-08-21 01:40 UTC - CSP-055 development evaluator gates passed

- Post-rebase source commit
  `ae90f6d9e4b23ebe8fb1bd2314baa66cb82b37c1` passed the complete remote
  dry-run on pinned image `ami-0324f0ad73bdcd087`: one AWS A10G, NVIDIA
  driver 595.71.05, CUDA 13.0, and locked PyTorch 2.13.0. The isolated graph
  contained exactly 36 model downloads, 72 development score cells, 72
  official metric cells, and one target, using only the four preregistered
  arms and nine requested steps.
- The first monitored score gate used the reused #417 CDS full-window step
  1,000 model and direct-downloaded only the pinned public
  `train.parquet`. The development boundary assertion accepted exactly
  16,140 odd-autosome/X rows. Forward and reverse-complement scoring both
  completed, and the unchanged official metric kernel emitted all 66 metric
  rows. No held-out file, label, prediction, measurement, or metric was read.
- An additive partial evaluation now targets every durable requested
  checkpoint: all nine reused CDS full-window steps, CDS center-1 through
  step 4,000, and both enhancer arms at step 1,000. The warm A10G graph has
  87 remaining or newly selected jobs and writes only beneath the immutable
  `results/issue473/ae90f6d.../development_eval/` namespace. It began model
  staging successfully and reached 10/87 jobs by 01:39 UTC.
- Both enhancer Iris jobs are actively training on east5 v6e-4 workers with
  zero failures and zero preemptions. At 01:35 UTC, full-window had reached
  about step 1,330 and center-1 about step 1,220, both with displayed loss
  1.32. Center-1 also committed a fresh temporary recovery checkpoint at
  step 1,217. The earlier queued state was transient accelerator capacity,
  not a persistent launch failure.

### 2026-08-21 03:07 UTC - CSP-056 enhancer export OOM and flexible recovery

- Both enhancer v9 children trained through step 2,000 and committed their
  native optimizer-state checkpoints before failing during the matching
  Hugging Face export. Full-window normal validation loss was 1.323 and
  center-1 was 1.322. Each 56 GiB container was killed while materializing
  the 1.02 GB safetensors shard, after TensorStore reported about 11 GiB of
  host memory in flight. Neither job was preempted and no training step was
  lost.
- Recovery commit `32dadc948cd3ff513bac2008f57739d80970a1c4`
  adds a bounded 96 GiB runtime option and resolves the model tokenizer to the
  absolute bundled-project path. The cache tokenizer identity remains the
  stable relative `tokenizer` value. The complete 35-test project suite passed;
  its 554,960 KiB peak RSS exceeded the shared node's 500 MiB guideline, so no
  further complete local suite runs will be made. The subsequent focused
  seven-test run stayed bounded at 478,036 KiB.
- Two first recovery submissions failed before graph creation because the
  isolated project mounts at `/app`, and that task image exposes Python at
  `/usr/local/bin/python3` rather than `/usr/bin/python3.12`. Corrected
  coordinators installed the exact 222-package lock at `/app`; their v6e-only
  children remained pending with no available slices and were terminated
  before replacement. None of these startup attempts requested a TPU or
  changed a checkpoint.
- Commit `a3d659ad20251753d6e8cf20cf4334318feb6eba` uses Fray's native flexible
  TPU request for compatible east5 `v5p-8` and `v6e-4` single-VM topologies.
  Enhancer center-1 child
  `/ubuntu/exp473-enhancer-center-1-v10r4-flex96g/run_levanter_train_lm-2d9d236a`
  immediately received a four-chip v5p-8 with 96 GiB RAM, restored temporary
  checkpoint step 1,947, and completed resumed train step 1,948. W&B correctly
  ignores the replayed steps through 2,000 because that run already recorded
  them. Full-window child
  `/ubuntu/exp473-enhancer-full-window-v10r4-flex96g/run_levanter_train_lm-cf645a99`
  is pending across both pools with zero failures and preemptions.
- CDS recovery `/ubuntu/exp473-cds-center-1-v5r6` completed its isolated sync
  and created its central1 v5p-8 child, which is pending for capacity. The
  on-demand development evaluator remains healthy and had completed 17 of 68
  resumed jobs, still reading direct `train.parquet` files only.

### 2026-08-21 03:20 UTC - CSP-057 enhancer recovery passed export boundary

- Enhancer center-1 re-committed native step 2,000, completed its 1.02 GB
  Hugging Face export at `hf/step-2000`, and continued through about step
  2,020. The same validation loss of 1.322 was reproduced. The temporary
  step-1,947 recovery checkpoint was deleted only after the durable native
  checkpoint committed.
- Enhancer full-window independently re-committed native step 2,000 and
  completed its 1.02 GB Hugging Face export at `hf/step-2000`. It returned to
  the training loop immediately afterward with displayed loss 1.31. The same
  validation loss of 1.323 was reproduced, and its temporary step-1,945
  checkpoint was deleted only after the durable native save.
- Both children run on four-chip east5 v5p-8 workers with 96 GiB container
  limits, attempt 0, zero failures, and zero preemptions. This directly
  confirms that the earlier failures were host-memory peaks during export,
  not data, model, or optimizer failures. The two enhancer trajectories now
  continue from the same run and checkpoint identities toward step 4,999.
- The on-demand development evaluator advanced to 24 of 68 jobs and continues
  to log only the pinned direct development `train.parquet` row counts.

### 2026-08-21 03:47 UTC - CSP-058 all training arms active and enhancer step-2,000 QC

- Both enhancer recoveries and the CDS center-1 recovery are now actively
  training. At 03:39 UTC enhancer center-1 had reached about step 2,300 and CDS
  center-1 about step 4,320. The CDS recovery restored the step-4,112 temporary
  checkpoint and has not repeated any durable checkpoint work.
- Enhancer full-window was preempted once after reaching step 2,180. Iris
  rescheduled attempt 1 on another 96 GiB east5 v5p-8, discovered temporary
  checkpoint step 2,093, passed the TensorStore integrity check, and explicitly
  resumed at step 2,094. No durable checkpoint was lost or rewritten.
- The isolated on-demand enhancer step-2,000 development graph completed all
  10 requested jobs: two checkpoint downloads, four score cells, and four
  official metric cells. Every cell used the direct pinned `train.parquet`
  path. The temporary A10G cluster was terminated immediately after success;
  its S3 outputs remain in the immutable `ae90f6d.../development_eval`
  namespace.
- The two Mendelian score bundles were exactly row-identical across policies:
  16,140 odd-autosome/X development rows. A preregistered 1,000-draw paired
  match-group bootstrap found that all eight step-2,000 AUPRC center-minus-full
  95% intervals include zero. The secondary splicing Group-SMD delta was
  positive (0.0382, 95% interval 0.0024 to 0.0738); the other seven Group-SMD
  intervals include zero. These are interim QC results, not the final
  trajectory interpretation.
- The four metric parquets totaled 28.9 KiB and the four score parquets 5.6
  MiB. Guarded local download and analysis used the nonblocking shared-node
  lock; the paired bootstrap peaked at 225,260 KiB RSS and completed in 1.28
  seconds. No held-out labeled file, prediction, measurement, or metric was
  requested or read.

### 2026-08-21 05:20 UTC - CSP-059 completed partial eval and cross-region recovery

- Enhancer center-1 committed and exported durable step 3,000, then continued
  through about step 3,370 with displayed loss 1.26 and zero preemptions or
  failures. Its exact four-object Hugging Face export is 1,019,426,427 bytes.
  The development-only step-3,000 evaluator is now launching on an
  auto-teardown A10G.
- The original 68-job development evaluator completed successfully. Together
  with the isolated successful runs, official metrics now exist for every
  reused CDS full-window step, CDS center-1 through step 4,500, both enhancer
  arms through step 2,000, and enhancer center-1 through step 2,500. All
  evaluator inputs were direct pinned `train.parquet` files; no held-out
  labeled file, prediction, measurement, or metric was read.
- East5 and central1 reported zero compatible free TPU workers and cloud
  capacity exhaustion; their configured non-preemptible TPU groups do not
  exist. The east5 full-window child transiently acquired capacity at 04:38,
  passed checkpoint integrity, and explicitly resumed from temporary step
  2,165 at step 2,166. It was stopped during first-batch JIT, before a new train
  step, to make a runbook-compliant region migration.
- Recovery commit `9bcf75c702c8cdb421bd3052d0cd4b7d43781c26` adds bounded
  `europe-west4` execution using `v6e-4` or canonical
  `v5litepod-16`. The fixed model, 8,192-sequence global batch, seed,
  optimizer, 5,000 steps, W&B run, and checkpoint identity are unchanged.
  Focused validation passed all seven tests at 479,248 KiB peak RSS. An initial
  test correctly rejected the noncanonical `v5e-16` spelling before it was
  replaced with Fray's supported topology name.
- The additive Europe namespace received the complete 5,288,301,383-byte
  durable step-2,000 checkpoint, the exact successfully reloaded temporary
  step-2,165 checkpoint, and the byte-complete 4,484,157,388-byte
  source-pinned tokenized cache. Training executor status was not copied. The
  first Europe coordinator exposed a tokenizer-child `cloudpickle`
  bootstrap bug before any TPU request; the completed cache copy safely pruned
  that subtree on replacement.
- Both an operator-managed Europe `v6e-4` and `v5litepod-16` slice
  were accepted by GCP but disappeared while bootstrapping, before registering
  workers or running a task. The current priority-band-2 full-window child
  `/ubuntu/exp473-enhancer-full-window-v10r11-eu-v5e16-96g-preemptible/run_levanter_train_lm-dbd65c66`
  is queued for the next complete four-worker v5litepod-16 slice. The active
  pool is not quota-blocked and has recently autoscaled, but its registered
  slices are currently occupied by priority-band-3 benchmark jobs.

### 2026-08-21 06:03 UTC - CSP-060 full-window resumed in us-east1

- Capacity inspection found that registered Europe `v5litepod-8` and `v6e-8`
  worker counts overstated immediately allocatable capacity. The v5e child
  reported zero available TPU chips, while the v6e child was blocked by only
  65.3 GB of unallocated host memory against the 96 GiB export-recovery
  request. Both children remained pending and were terminated before another
  child was launched against the same run identity.
- The bounded recovery launcher now also permits single-worker `v5litepod-8`
  and `v6e-8` execution in Europe and the established `us-east1` `v6e-4`
  pool backed by `gs://marin-us-east1`. The model, fixed 8,192-sequence global
  batch, per-device parallelism, seed, optimizer, 5,000-step schedule, W&B run,
  and checkpoint identity are unchanged. Three focused locked test runs each
  passed all seven tests; the final run peaked at 478,340 KiB RSS.
- A new additive us-east1 namespace received the source-pinned cache contract
  and data, the 5,288,301,383-byte durable step-2,000 checkpoint, and the
  3,058,278,595-byte temporary step-2,165 checkpoint. The destination cache is
  4,484,156,824 bytes, exactly the 4,484,157,388-byte Europe cache minus its
  intentionally omitted 564-byte executor-provenance file. The scientific
  cache success marker was retained; no training executor status was copied.
- Full-window child
  `/ubuntu/exp473-enhancer-full-window-v10r16-east1-v6e4-56g-preemptible/run_levanter_train_lm-cafa6940`
  received a four-chip us-east1 v6e-4 immediately. It pruned the complete cache,
  discovered temporary step 2,165, completed TensorStore error checking, and
  explicitly resumed the existing trajectory at step 2,166. The 56 GiB limit
  is the previously successful training setting; native step 2,500 will commit
  before the next Hugging Face export boundary.
- Enhancer center-1 continued independently through about step 3,620 with loss
  1.24 and no failures or preemptions. Its development-only step-3,000 graph
  completed all five jobs successfully. No held-out labeled file, prediction,
  measurement, or metric was requested or read.
- Next action: verify the first post-resume full-window train step, monitor the
  step-2,500 native checkpoint and export, and continue scheduled development
  evaluation for newly exported enhancer-center checkpoints.
