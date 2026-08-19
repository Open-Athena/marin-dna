---
name: plot-research-results
description: Create or revise MarinDNA plots and figures for experiment analysis, research issues, pull requests, reports, and documentation. Use when choosing a visual encoding, displaying uncertainty or aggregation, comparing metrics, or producing and reviewing a rendered research figure.
---

# Plot Research Results

Build the figure around the scientific comparison it must support.

## Choose The Highest-Level Suitable Interface

- Prefer the highest-level plotting interface that expresses the intended figure cleanly. Favor declarative or figure-level APIs that own faceting, labels, legends, scales, and layout.
- Use axes-level or low-level drawing when the higher-level interface cannot express the figure without distorting the comparison or requiring brittle workarounds.
- With seaborn, prefer figure-level functions such as `relplot`, `catplot`, and `displot` when they fit. Use axes-level seaborn or matplotlib for genuinely custom composition.

## Preserve The Meaning

- Label metrics, units, aggregation, sample definition, transformations, and uncertainty explicitly.
- State the uncertainty quantity, such as standard error, confidence interval, standard deviation, or range. Choose its rendering to match that meaning.
- Do not imply level comparisons between quantities that are not comparable by construction. Use facets, normalization, independent axes, or another labeled encoding appropriate to the task.
- Preserve negative results and relevant variation. Do not tune axis limits, smoothing, filtering, or color scales to exaggerate an effect.
- Use accessible colors and ensure the figure remains interpretable without relying on color alone when practical.

## Produce And Inspect The Artifact

- Emit SVG by default for a static figure. Add PNG, PDF, or another format only when a downstream consumer requires it.
- Inspect the rendered artifact before publishing it. Check labels, clipping, legend order, facet consistency, numeric scales, and whether the visual supports the stated takeaway.
- Prefer storing figures under `.agents/artifacts/<topic>/` on the permanent task or research branch. A revision-pinned gist is also acceptable, especially for an existing artifact. Use an immutable raw URL when a figure must render in a GitHub issue, pull request, or Markdown document; do not use a mutable branch-head or unpinned gist URL.
