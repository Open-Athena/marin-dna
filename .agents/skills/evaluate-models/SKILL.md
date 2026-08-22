---
name: evaluate-models
description: Protect held-out labeled variant-effect prediction data and design, run, interpret, and report MarinDNA genomic language-model evaluations. Use for development, training, validation, model selection, probing, tuning, or evaluation on labeled VEP data, and when selecting benchmark subsets, filtering evaluation records, ordering reported results, or deciding which results are valid for a model's training-region scope.
---

# Evaluate Models

Apply these rules before computing aggregates or choosing tables and plots.

## Protect Held-Out VEP Data

- Use odd-numbered autosomes and chromosome X for development, training, validation, model selection, probing, and tuning on labeled variant-effect prediction data.
- Reserve even-numbered autosomes and chromosome Y for final test evaluation.
- Require explicit user permission before accessing held-out labels, predictions, effect measurements, or aggregate metrics.
- Apply this restriction to labeled VEP data only.
  Unlabeled reference sequence and functional-genomics data remain available unless their dataset defines a stricter split.

## Register evals_v2 Models

- Before evaluating a model through `snakemake/analysis/evals_v2`, ensure the exact model-dataset cell is registered under `models:` in `snakemake/analysis/evals_v2/config/config.yaml`.
  Add new model entries and widen an existing model's `datasets` scope additively before running an unregistered cell.
  Do not rename, repurpose, or replace an existing model entry.
- Submit each new model registration or dataset-scope expansion in a small PR before running the evaluation.
  Record the canonical model ID in `name`, the exact checkpoint path, `window_size`, and `datasets` scope so future work can discover and reuse existing model-dataset results instead of registering aliases or recomputing them.
- Check the registry and canonical outputs before adding a model or running a model-dataset cell.
- Keep evals_v2 outputs in the workflow's fixed `s3://oa-bolinas/snakemake/analysis/evals_v2/results/` layout, including `scores/{model}/{dataset}.parquet` and `metrics/{model}/{dataset}.parquet`.
  Do not wrap these outputs in commit-keyed or issue-specific score namespaces.

## Prepare The Evaluation Frame

1. Remove mature miRNA before computing any metric, aggregate, macro average, global score, table, or plot.
   For Mendelian and complex-trait matched data, exclude the complete `match_group` when its canonical `subset` is `mature_miRNA_variant`.
   If the source has multi-valued consequence annotations, canonicalize them first and exclude the complete group when any annotation maps to `mature_miRNA_variant`.
   For an ungrouped evaluation, exclude the complete record under the same predicate.
   Assert that no excluded record or group remains in any analysis frame.
2. Determine the model's training-region scope from its data manifest or training configuration.
   Do not infer scope from the evaluation results.
3. Restrict each benchmark to biologically appropriate subsets for that scope.
   State explicitly when a requested subset is unavailable or inappropriate rather than substituting another subset.

## Select And Order Standalone Mendelian Reports

Use this mapping for standalone narrative reports of Mendelian evaluations.
For broad models, follow the blog order: macro average first, then consequence subsets in the positive-sample-count order adopted there.

| Training-region scope | Present these subsets, in this order |
| --- | --- |
| Broad or mixed genomic regions | Macro Avg, Missense, Splicing, 5′ UTR, Promoter, ncRNA, 3′ UTR, Distal, Synonymous |
| CDS | Missense, Splicing, Synonymous |
| Upstream, promoter, TSS, or 5′ UTR | 5′ UTR, Promoter |
| Downstream or 3′ UTR | 3′ UTR |
| ncRNA | ncRNA |
| Enhancer or cCRE | Distal |

Use the producing pipeline's support contract.
The current unsupervised Mendelian contract requires at least 30 positive match groups per subset.
Supervised probes also apply their pipeline-defined chromosome-support gate.

## Select And Order Standalone Complex-Trait Reports

Apply the Complex Traits support filter independently from Mendelian.
After mature-miRNA exclusion, the current train split's 30-positive gate qualifies Distal, Missense, Promoter, 3′ UTR, ncRNA, and 5′ UTR; Splicing and Synonymous do not qualify.
Use this mapping for standalone narrative reports:

| Training-region scope | Present these subsets, in this order |
| --- | --- |
| Broad or mixed genomic regions | Macro Avg, Distal, Missense, Promoter, 3′ UTR, ncRNA, 5′ UTR |
| CDS | Missense |
| Upstream, promoter, TSS, or 5′ UTR | Promoter, 5′ UTR |
| Downstream or 3′ UTR | 3′ UTR |
| ncRNA | ncRNA |
| Enhancer or cCRE | Distal |

Recompute the qualifying set from the pinned evaluation input whenever its revision changes.
Do not inherit the Mendelian subset list.

## Handle Matched-Trait Aggregates And Dashboard Views

- Preserve each benchmark's relative order above after removing subsets absent from the benchmark or invalid under its producing metric's current sample-support contract.
  Use the producing pipeline's contract rather than reimplementing the gate.
- Compute `Macro Avg` only for a broad or mixed model and only over qualifying displayed subsets.
  Do not add a macro average to a specialist-model report.
- Do not show `Global` for a specialist model because it pools regions outside that model's training scope.
- Omit `Global` by default for a broad or mixed model because it weights results by subset prevalence.
  Include it only for a stated scientific reason, label that weighting explicitly, and place it after `Macro Avg` and before the consequence subsets.
- When creating or changing a dashboard view, follow its benchmark metadata and component ordering instead of this standalone-report order.
  The current dashboard leads Mendelian with `Macro Avg`, leads complex traits with `Global`, and orders consequence subsets dynamically by positive-sample count.

## Select And Order SGE Results

Use this mapping for saturation genome-editing evaluations:

| Training-region scope | Present these subsets, in this order |
| --- | --- |
| Broad or mixed genomic regions | Macro, Missense, Splicing, Both |
| CDS | Missense, Splicing |
| Any other specialist region | Do not present SGE results unless the assay and region have a specific biological justification |

Use the producing pipeline's SGE validity contract, currently at least 30 examples in each label class per accession and base-subset cell.
Compute `Macro` only for a broad or mixed model as the equal-weight mean over qualifying base subsets.
Exclude the pooled `Both` result from `Macro`.
Do not add `Macro`, `Both`, or another aggregate to a CDS report.
SGE has no `Global` row.
Do not replace an unavailable subset with `Both` or another aggregate.

## Report The Result

- Name the benchmark, split, metric, eligible sample, training-region scope, and exclusions.
- Put the macro average first when the mapping includes it, then keep the listed subset order.
- Report omitted requested subsets and the reason for each omission.
- Keep dense numeric output in W&B or an artifact and present only the comparisons needed for the claim.
- Use `plot-research-results` for figures and its uncertainty, centering, and caption conventions.

## Compose Existing Skills

- Use `develop-snakemake-pipelines` when the evaluation is a maintained Snakemake workflow.
- Use `run-model-inference` when the evaluation requires model inference.
- Use `run-research` and `task-logbook` for one-off or multi-session evaluation investigations.
- Use `manage-research-storage` to choose durable artifact locations.
- Use `wandb-reporting` for dense metrics and run comparison.
- Use `maintain-knowledge-base` when accepted results change a research interpretation.
