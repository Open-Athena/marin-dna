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
