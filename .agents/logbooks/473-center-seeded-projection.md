---
topic: center-seeded-vertebrate-projection
issue: https://github.com/Open-Athena/marin-dna/issues/473
description: Compare center-seeded and full-window vertebrate projection policies.
author: gonzalobenegas
---

# Center-seeded vertebrate projection: Research Logbook

## Current TL;DR

An additive `issue_473` namespace now separates each fixed 255 bp human
source anchor from the smaller interval submitted to HAL or MAF. It defines
tested full-window and centered 1, 17, 33, 65, and 129 bp request semantics,
policy-specific target-span gates, and a deterministic pilot sampler covering
source chromosome by conservation-score quantile strata. New opt-in Snakemake
rules now resolve the end-to-end smoke projection, per-policy QC, and
cross-policy comparison without changing existing rules or shared execution
paths.

No alignment, publication, training, S3-write, or remote-compute job has run
for #473.

## Scope

- Goal: compare `full_window` and `center_1` on fixed anchors, then use a
  sampled odd-width landmark pilot to decide whether one wider policy warrants
  a later full-scale experiment.
- Primary projection metrics: accepted recovery, aligned coverage, unaligned
  flank, ambiguity and rejection reasons, and agreement between policy loci.
- Primary training metric: development-split VEP AUPRC at matched tokens.
- Constraints: internal coordinates are 0-based and half-open; held-out
  even-autosome and chromosome-Y VEP labels and aggregates are not accessed;
  paid compute requires explicit approval.
- Coordinating issue: https://github.com/Open-Athena/marin-dna/issues/473
- Branch: `codex/issue-473-center-seeded-projection`
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

- `CSP-H1`: `center_1` recovers at least as many target species per anchor as
  `full_window`, with the largest gain in non-mammalian vertebrates and
  enhancer-centered cCREs. Next test: sampled HAL and MAF policy comparison.
- `CSP-H2`: `full_window` has greater aligned coverage within emitted target
  windows. Next test: paired coverage and flank diagnostics on the union and
  intersection of accepted `(anchor, species)` rows.
- `CSP-H3`: recovery weakly decreases and alignment evidence weakly increases
  over centered widths 1, 17, 33, 65, and 129 bp. Next test: deterministic
  region, chromosome, and conservation-quantile sample.
- `CSP-H4`: the downstream policy effect differs between CDS and
  enhancer-centered cCRE specialists. Next test: matched-token training after
  projection QC passes.

### Blocked

- Full-scale projection and training are blocked on implementation review,
  smoke QC, and explicit compute approval.

### Falsified / Dead End

- None.

### Promoted

- None.

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
