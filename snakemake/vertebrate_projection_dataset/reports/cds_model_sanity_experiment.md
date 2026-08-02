# CDS mammals-only vs. combined-vertebrate sanity experiment

Status: **matched training in progress** on the user-approved free Iris/TRC
allocation. The projection/QC gates and reviewed Hugging Face publication are
complete; terminal offline VEP evaluation remains pending both step-4999
exports.

## Question and directional expectation

Does adding family-deduplicated non-mammalian MultiZ projections to otherwise
identical human-plus-Zoonomia CDS training data improve coding-variant effect
prediction?

The preregistered directional expectation is that the combined-vertebrate arm
outperforms the mammals-only arm because it observes deeper evolutionary
constraint. A null or reversed result triggers a projection, duplication,
species-balance, and exposure audit; it does not authorize tuning the analysis
after seeing results.

## Frozen reporting protocol

Report all preregistered cells; do not select a dataset, consequence, strand,
or score type after seeing the result.

- Mendelian traits uses signed FWD/RC-averaged LLR (`minus_llr_avg`). Report
  the matched-pair `_global_` overall row and every real consequence subset.
- SGE uses signed FWD/RC-averaged LLR (`minus_llr_avg`). Report the
  accession-macro `both` row overall and the accession-macro
  `missense_variant` and `splicing` consequence rows.
- Per-arm AUPRC uncertainty is the frozen harness output: 1,000 bootstrap
  iterations with seed 0. Matched datasets resample `match_group` clusters;
  SGE resamples rows within each accession before macro-averaging.
- Every delta is `combined_vertebrates - mammals_only`. For matched datasets,
  resample each shared `match_group` once per iteration and recompute both
  scores on that same sample using
  `paired_metric_delta_bootstrap`. For SGE, resample shared row indices once
  within each qualifying accession, recompute both AUPRCs, then macro-average
  over the same qualifying accessions. Use 1,000 iterations and seed 0 for both
  paired paths, and report the point delta, bootstrap SE, percentile 95% CI,
  and two-sided bootstrap p-value.

## Matched arms

1. `mammals_only`: human reference plus Zoonomia mammalian projections.
2. `combined_vertebrates`: the identical rows plus selected non-mammalian
   MultiZ projections.

Both arms must pin the same:

- producing pipeline commit and species manifests;
- CDS anchor definition and chromosome-18 split policy;
- 16,384-row, original-orientation validation cap and seed;
- architecture, tokenizer, optimizer/schedule, batch construction, training
  token budget, initialization policy, and evaluation cadence; and
- coding-variant VEP harness revision and evaluation inputs.

The only intended treatment difference is the presence of non-mammalian
training sequences. Record realized rows/tokens and per-species exposure for
both arms so that accidental compute or sampling differences are visible.

## Required execution record

Recorded from the immutable launch configuration and published artifacts:

| Field | Mammals only | Combined vertebrates |
|---|---|---|
| Pipeline commit | [`d50ba5d`](https://github.com/Open-Athena/marin-dna/tree/d50ba5d6d8bd15e28ff11ad61bdd4a5aef67b733/snakemake/vertebrate_projection_dataset) | same |
| HF dataset + revision | [`marin-dna/vertebrate-v1-cds_mammals_only@d2bea76`](https://huggingface.co/datasets/marin-dna/vertebrate-v1-cds_mammals_only/tree/d2bea760f6416775772699b821b266d3ae87245e) | [`marin-dna/vertebrate-v1-cds@bfab878`](https://huggingface.co/datasets/marin-dna/vertebrate-v1-cds/tree/bfab878078c4ee6c0f47b760f1e5e0577549dc9d) |
| Species manifest commit | [`d50ba5d`](https://github.com/Open-Athena/marin-dna/blob/d50ba5d6d8bd15e28ff11ad61bdd4a5aef67b733/snakemake/vertebrate_projection_dataset/config/species_selected.tsv) | same |
| Train rows / exposure tokens | 56,549,084 / 10,485,760,000 | 66,552,602 / 10,485,760,000 |
| Validation rows / tokens | 16,384 / 4,194,304 | 16,384 / 4,194,304 |
| Model config | [Qwen3 255M matched recipe](https://github.com/Open-Athena/marin-dna/blob/260a7a77655a604ee2f9d7b0bc15776e4b7b9116/experiments/exp417_vertebrate_cds/launch.py) | same |
| Random seed(s) | 0 | 0 |
| W&B run (`dna-exp417` in name) | [`dna-exp417-cds-mammals-only-p255m-b2m-5k`](https://wandb.ai/gonzalobenegas/marin/runs/dna-exp417-cds-mammals-only-p255m-b2m-5k) | [`dna-exp417-cds-combined-vertebrates-p255m-b2m-5k`](https://wandb.ai/gonzalobenegas/marin/runs/dna-exp417-cds-combined-vertebrates-p255m-b2m-5k) |
| VEP harness commit | [`evals_v2@260a7a7`](https://github.com/Open-Athena/marin-dna/tree/260a7a77655a604ee2f9d7b0bc15776e4b7b9116/snakemake/analysis/evals_v2) + [frozen overlay](https://github.com/Open-Athena/marin-dna/blob/260a7a77655a604ee2f9d7b0bc15776e4b7b9116/experiments/exp417_vertebrate_cds/evals.yaml) + [paired report](https://github.com/Open-Athena/marin-dna/blob/260a7a77655a604ee2f9d7b0bc15776e4b7b9116/scripts/issue417_summarize_vep.py) | same |

## Execution status

Both arms use one `v6e-4` in `us-east5`, 5,000 steps, 8,192 sequences per
step, the same optimizer and tokenizer, and a 500-step validation/native-
checkpoint/Hugging Face export cadence. Only the immutable source corpus
differs.

- [Mammals-only Iris job](https://iris.oa.dev/#/job/%2Fubuntu%2Fdna-exp417-cds-mammals-only):
  passed the complete step-500 checkpoint/eval/HF-export gate and reached step
  629 by 2026-08-02 01:45 UTC with zero failures or preemptions.
- [Combined retry `r2`](https://iris.oa.dev/#/job/%2Fubuntu%2Fdna-exp417-cds-combined-vertebrates-r2):
  restored step 50 after the original run's checkpoint serialization failure,
  committed a fresh step-115 temporary checkpoint, passed the original
  step-159 failure point, and reached step 184 by 2026-08-02 01:45 UTC with
  zero failures or preemptions. The preceding `r1` submission failed before
  restore because its coordinator launch omitted the W&B credential; it did
  not alter model state or the frozen recipe.

## Required results

Report overall and consequence-level coding-variant metrics for both arms, with
uncertainty where the harness supports it. Include paired deltas, training
curves, realized compute, and failures. Do not report only the favorable metric.

| Metric / consequence | Mammals only | Combined | Delta | Uncertainty |
|---|---:|---:|---:|---:|
| Overall | pending | pending | pending | pending |
| Consequence-level rows | pending | pending | pending | pending |

## Audit if the direction is null or reversed

- confirm identical human/CDS anchors and chromosome-18 membership;
- check projection rejection rates, bounds, strand, source case, and duplicate
  `(anchor, species)` rows by backend;
- compare per-species/per-clade exposure and RC augmentation;
- inspect CDS recovery breadth and the ZRS positive control;
- verify the two runs used equal training-token budgets and evaluation inputs;
  and
- document the audit before proposing a follow-up experiment.
