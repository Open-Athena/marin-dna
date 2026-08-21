---
name: evaluate-models
description: Design, run, interpret, and report MarinDNA genomic language-model evaluations. Use when selecting benchmark subsets, filtering evaluation records, ordering reported results, or deciding which results are valid for a model's training-region scope.
---

# Evaluate Models

Apply these rules before computing aggregates or choosing tables and plots.

## Prepare The Evaluation Frame

1. Enforce the labeled variant-effect split in `AGENTS.md` before reading held-out labels, predictions, effect measurements, or aggregate metrics.
2. Remove every mature-miRNA record from every evaluation dataset before computing any metric, aggregate, macro average, global score, table, or plot.
   Match the dataset's canonical annotation for mature miRNA, including `mature_miRNA_variant` and normalized equivalents, and assert that no matched record remains.
3. Determine the model's training-region scope from its data manifest or training configuration.
   Do not infer scope from the evaluation results.
4. Restrict each benchmark to biologically appropriate subsets for that scope.
   State explicitly when a requested subset is unavailable or inappropriate rather than substituting another subset.

## Select And Order Mendelian And Complex-Trait Results

Use this mapping for both Mendelian and complex-trait evaluations:
The broad-model order matches the blog and dashboard subset order: macro average first, then subsets in descending positive-sample count.

| Training-region scope | Present these subsets, in this order |
| --- | --- |
| Broad or mixed genomic regions | Macro Avg, Missense, Splicing, 5′ UTR, Promoter, ncRNA, 3′ UTR, Distal, Synonymous |
| CDS | Missense, Splicing, Synonymous |
| Upstream, promoter, TSS, or 5′ UTR | 5′ UTR, Promoter |
| Downstream or 3′ UTR | 3′ UTR |
| ncRNA | ncRNA |
| Enhancer or cCRE | Distal |

- Preserve the relative order above after removing subsets absent from the benchmark or invalid for the evaluated sample.
- Compute `Macro Avg` only for a broad or mixed model and only over the displayed, sample-valid subsets.
  Do not add a macro average to a specialist-model report.
- Do not show `Global` for a specialist model because it pools regions outside that model's training scope.
- Omit `Global` by default for a broad or mixed model because it weights results by subset prevalence.
  Include it only for a stated scientific reason, label that weighting explicitly, and place it after `Macro Avg` and before the consequence subsets.

## Select And Order SGE Results

Use this mapping for saturation genome-editing evaluations:

| Training-region scope | Present these subsets, in this order |
| --- | --- |
| Broad or mixed genomic regions | Macro, Missense, Splicing, Both |
| CDS | Missense, Splicing |
| Any other specialist region | Do not present SGE results unless the assay and region have a specific biological justification |

Compute `Macro` only for a broad or mixed model and only over the displayed, sample-valid SGE subsets.
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
- Use `run-research` and `task-logbook` for one-off or multi-session evaluation investigations.
- Use `manage-research-storage` to choose durable artifact locations.
- Use `wandb-reporting` for dense metrics and run comparison.
- Use `maintain-knowledge-base` when accepted results change a research interpretation.
