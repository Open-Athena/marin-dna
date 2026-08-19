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
