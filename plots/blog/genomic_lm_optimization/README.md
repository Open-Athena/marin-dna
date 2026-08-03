# Genomic LM optimization blog figures

This directory is the durable source bundle for the charts in `blog/genomic-lm-optimization/`. It preserves the complete plotting project, committed input snapshots, appendix analyses, and the mapping from article figure numbers to recipes and checked-in SVG assets.

The imported project comes from [`eric-czech/marin-dna-post-202606`](https://github.com/eric-czech/marin-dna-post-202606) at commit [`2abef91b37a16fde9c9cdf1cfa0046942442b97f`](https://github.com/eric-czech/marin-dna-post-202606/commit/2abef91b37a16fde9c9cdf1cfa0046942442b97f). The Figure 9–12 typography and Figure 11 label fixes were then applied locally in this branch.

## Reproduce

Run from the MarinDNA repository root using the repository root project. Figures 13–15 import MarinDNA evaluation code and require its Parquet dependencies, so the complete historical build does not use this directory’s nested lock. The committed CSV snapshots back the transfer, loss-scaling, mixture, and historical leaderboard figures; Figures 13–15 additionally read the paired endpoint predictions used to recompute both readouts with the same chromosome-weighted AUPRC implementation:

```bash
uv run --frozen python plots/blog/genomic_lm_optimization/src/figures/__main__.py
```

Generate the current live-evals leaderboard panels separately:

```bash
uv run python -m plots.blog.figure11_leaderboard_heatmap
```

Historical-bundle outputs are written to `plots/output/blog/genomic_lm_optimization/`; the current leaderboard panels are written to `plots/output/blog/`. Both locations contain SVG, PNG, and PDF artifacts. Figure 12 also writes its Kaplan-fit report to the historical output directory, and SVG trailing whitespace is normalized.

All data plots inherit Matplotlib's default font, line, marker, and automatic
tick settings. Saving applies the single shared 1.2× whole-SVG render scale from
`src/marin_dna/blog_figure_typography.py`; individual recipes cannot override
it. Adjust plot density through subplot height and layout only. The sync command
derives the article frame width from the scaled SVG, so a browser does not
silently apply a second, figure-specific scale.

After inspecting generated SVGs and PNGs, copy only approved assets into the blog with the explicit sync command:

```bash
uv run --project plots/blog/genomic_lm_optimization python plots/blog/genomic_lm_optimization/src/sync_blog_assets.py figure1_lr_transfer figure2_beta2_epsilon_transfer
```

Refresh the committed sweep and mixture CSV snapshots from WandB with:

```bash
uv run --project plots/blog/genomic_lm_optimization python plots/blog/genomic_lm_optimization/src/data.py
```

`model_leaderboard.csv` is a dated extraction whose provenance URL is recorded in its first line.

## Article mapping

| Blog figure | Blog asset | Plot recipe or source | Inputs |
|---:|---|---|---|
| 1 | `headline_cost_performance.svg` | `src/figures/headline_cost_performance.py` | `model_leaderboard.csv`, `inference_costs.csv` |
| 2 | `data_provenance_training_datasets.svg` | Authored SVG; no plot recipe | Values documented in the article |
| 3 | `eval_datasets.svg` | Authored SVG; no plot recipe | Benchmark metadata |
| 4 | `eval_apparatus.svg` | Authored SVG; no plot recipe | Evaluation protocol |
| 5 | `promoter_cds_specialists.svg` | `plots/blog/promoter_cds_specialists.py` | Current `evals_v2` metrics |
| 6 | `upstream_cds_balance.svg` | `plots/upstream_cds_balance.py` | Current `evals_v2` metrics on S3 |
| 7 | `annotation_derived_training_pool.svg` | Authored SVG; no plot recipe | Dataset token counts |
| 8 | `parameter_transfer_methodology_v1.svg` | Authored SVG; no plot recipe | Transfer methodology |
| 9 | `figure1_lr_transfer.svg` | `src/figures/figure1_lr_transfer.py` | `transfer_validation_results.csv` |
| 10 | `figure2_beta2_epsilon_transfer.svg` | `src/figures/figure2_beta2_epsilon_transfer.py` | `transfer_validation_results.csv` |
| 11 | `figure3_region_hyper_transfer.svg` | `src/figures/figure3_region_hyper_transfer.py` | `transfer_validation_results.csv` |
| 12 | `figure4_loss_scaling.svg` | `src/figures/figure4_loss_scaling.py` | `parameter_scaling_results.csv`, `parameter_scaling_history.csv` |
| 13 | `figure5_params_vs_vep_auprc.svg` | `src/figures/figure5_params_vs_vep_auprc.py` | Scaling snapshot; paired endpoint predictions on S3 |
| 14 | `figure6_loss_vs_vep_auprc.svg` | `src/figures/figure6_loss_vs_vep_auprc.py` | Scaling snapshot; paired endpoint predictions on S3 |
| 15 | `figure6b_marin_evo2_missense.svg` | `src/figures/figure6b_marin_evo2_missense.py` | Paired MarinDNA endpoint predictions on S3; commit-pinned Evo 2 probe metrics |
| 16 | `continued_training_data_exposures.svg` | Authored SVG; no plot recipe | Figure 10 mixture-lineage definitions and token accounting |
| 17 | `figure16_offline_lineage_llr_prototype.svg` | `src/figures/figure16_offline_lineage_prototype.py` | Existing `evals_v2` Mendelian metric Parquets on S3; mixture results CSV for lineage token accounting |
| 18 | `figure16_offline_lineage_probe_prototype.svg` (collapsed) | `src/figures/figure16_offline_lineage_prototype.py` | Existing `evals_v2` probe-metric Parquets on S3; mixture results CSV for lineage token accounting |
| 19 | `figure11_leaderboard_heatmap__mendelian_llr.svg` | `../figure11_leaderboard_heatmap.py` | Current `evals_v2` Mendelian zero-shot metrics |
| 20 | `figure11_leaderboard_heatmap__mendelian_probe.svg` (collapsed) | `../figure11_leaderboard_heatmap.py` | Current `evals_v2` Mendelian probe metrics |

The upstream-reweighting analysis remains reproducible in this source bundle but is not currently included in the article.
Its displayed endpoints confound mixture proportion with continuation budget: the 40% and 50% upstream arms received about 8.3k steps, while the other five arms received about 2.5k, and no exact checkpoint step is retained across all seven runs.
See the canonical review in [issue #370](https://github.com/Open-Athena/marin-dna/issues/370).

| Retained asset | Source | Inputs |
|---|---|---|
| `mini_fig9_mixture.svg` | Authored SVG; no plot recipe | Mixture continuation design |
| `figure9_upstream_mix_auprc.svg` | `src/figures/figure9_upstream_mix_auprc.py` | `data/data_mixture_results.csv` |

Paths beginning with `src/` and CSV names in this table are relative to this directory.

## Appendix analyses

| Analysis | Recipe | Data source |
|---|---|---|
| Mixture continuation tree | `src/figures/appendix/mixture_tree.py` | Committed mixture results |
| Cooldown counterfactuals | `src/figures/appendix/cooldown_effects.py` | Committed VEP history plus WandB loss |
| Pooled vs unpooled matching | `src/figures/appendix/pooled_vs_unpooled.py` | WandB histories |
| Per-region loss scaling | `src/figures/appendix/region_loss_scaling.py` | WandB summaries |

Appendix outputs land in `plots/output/blog/genomic_lm_optimization/appendix/`. The imported `docs/outline.md` records the source projects, sweep definitions, and constants referenced by recipe comments.
