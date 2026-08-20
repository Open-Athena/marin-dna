---
name: plot-research-results
description: Create or revise MarinDNA plots and figures for experiment analysis, research issues, pull requests, reports, and documentation. Use when choosing a visual encoding, displaying uncertainty or aggregation, comparing metrics, or producing and reviewing a rendered research figure.
---

# Plot Research Results

Build the figure around the scientific comparison it must support.

## Apply The First-Review Defaults

- Apply the defaults in this skill before first presenting a figure to a human. Explicit human direction and an established figure-family or publication style override them.
- Present one scientific comparison at a time. Start with the primary contrast, then add modifiers or interactions only when they answer the next question.
- Use one single-line overall figure title. Give each subplot or facet one single-line title that identifies only its panel distinction. Do not stack subtitles, narrative blocks, or sample counts into the figure header.
- Use the [published MarinDNA blog](https://openathena.ai/blog/marin-dna/) and its [merged figure assets](https://github.com/Open-Athena/open-athena.github.io/tree/d61d7c68f073ce2e923e1132a29dcccc7556b0e4/static/assets/images/blog/marin-dna) as the visual reference for these defaults.

## Choose The Highest-Level Suitable Interface

- Prefer the highest-level plotting interface that expresses the intended figure cleanly. Favor declarative or figure-level APIs that own faceting, labels, legends, scales, and layout.
- Use axes-level or low-level drawing when the higher-level interface cannot express the figure without distorting the comparison or requiring brittle workarounds.
- With seaborn, prefer figure-level functions such as `relplot`, `catplot`, and `displot` when they fit. Use axes-level seaborn or matplotlib for genuinely custom composition.

## Size Typography And Layout

- Keep the plotting theme's typography consistent. Improve legibility by changing the figure, facet, or intended display size rather than manually setting individual font sizes.
- Write concise, sentence-case axis labels and legend titles with an initial capital, such as `Parameters` and `Loss`. Put longer definitions in the caption or prose.
- Prefer square axes for quantitative subplots. Deviate when the data or plot form benefits materially from another aspect ratio.
- Keep spacing between subplots compact. Size the canvas so square axes fill their allotted cells without large empty gaps.
- Preserve a semantic legend title. Place the legend where it does not overlap data or create excessive whitespace.
- Map a variable to one visual channel by default. Do not vary color and line style or marker shape for the same variable unless accessibility, monochrome output, or explicit human direction requires redundant encoding.

## Preserve The Meaning

- Label metrics, units, aggregation, sample definition, transformations, and any displayed uncertainty explicitly in the caption or surrounding prose when they do not fit cleanly in the figure.
- Keep sample counts out of the plotting area by default. Include `n` only when it is needed to interpret the statistic, and state exactly what it counts.
- Do not add bootstrap intervals or another uncertainty overlay by habit. If uncertainty is negligible at the intended display scale, omit the overlay; when uncertainty is shown, name the quantity, such as standard error, confidence interval, standard deviation, or range.
- Share an axis only when the common scale supports the intended comparison without obscuring variation. Otherwise use clearly labeled independent axes, especially for panels that are not comparable by construction or have materially different useful ranges.
- Preserve negative results and relevant variation. Do not tune axis limits, smoothing, filtering, or color scales to exaggerate an effect.
- Use accessible colors. Add a redundant visual channel only when the output context requires it.

## Produce And Inspect The Artifact

- Emit SVG by default for a static figure. Add PNG, PDF, or another format only when a downstream consumer requires it.
- Inspect the rendered artifact before publishing it. Check title length, label capitalization, typography, axis aspect, subplot spacing, legend title and order, clipping, numeric scales, and whether the visual supports the stated takeaway.
- Commit every SVG referenced by a knowledge-base experiment page under `docs/research/experiments/figures/<issue>/` on `main`.
- Store other small figures under `.agents/artifacts/<topic>/` on the permanent task or research branch.
- Use a raw commit-pinned repository URL when a branch figure must render on GitHub.
