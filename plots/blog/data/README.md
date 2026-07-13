# Vendored data

`parameter_scaling_results.csv` — the 8-size parameter-scaling ladder training
results, vendored from
[`eric-czech/marin-dna-post-202606`](https://github.com/eric-czech/marin-dna-post-202606)@`2abef91`
(`data/`). Figs 5/6 consume **`run_name` / `params`** plus the region-specific
validation losses **`eval_loss_{cds,upstream,downstream}`**. Fig 6 pairs each
variant type with the loss for its mapped training-data region; the global
`eval_loss` is intentionally not used. The `lm_eval/…/auprc` columns are the **old**
in-training eval and are **not used** — the redone figures pull AUPRC live from
the **new** eval via `marin_dna.pipelines.evals.blog_metrics`. Training results
(params, loss) are eval-independent, so this file stays valid.

Run-name → evals_v2 id: `dna-bolinas-scaling-v0.5-hH-pP` → `scaling-v0.5-hH-pP-step-215573`.
