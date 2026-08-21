---
name: plot-research-results
description: Create or revise MarinDNA plots and figures for experiment analysis, research issues, pull requests, reports, and documentation. Use when choosing a visual encoding, displaying uncertainty or aggregation, comparing metrics, or producing and reviewing a rendered research figure.
---

# Plot Research Results

Apply the presentation guidance in this skill before first presenting a figure to a human.
Explicit human direction and an established figure-family or publication style may override these presentation defaults.
Scientific-integrity and repository-policy requirements remain binding.

Prefer plotting-library presentation defaults, including Seaborn's default color palette, when they express the intended comparison cleanly.
Customize those defaults only for a specific semantic, legibility, or publication constraint.
Choose statistical settings explicitly.
Do not inherit automatic aggregation or uncertainty behavior from a plotting library.

Build the figure around the scientific comparison it must support.

## Compose The Figure

- Present one scientific comparison at a time.
  Start with the primary contrast, then add modifiers or interactions only when they answer the next question.
- Keep the overall figure title, each subplot or facet title, and each axis label on one line within its allotted area.
  Shorten them or move detail to the caption before they wrap, clip, or overflow.

## Choose The Highest-Level Suitable Interface

- Prefer the highest-level plotting interface that expresses the intended figure cleanly.
  Favor declarative or figure-level APIs that own faceting, labels, legends, scales, and layout.
- Use axes-level or low-level drawing when the higher-level interface cannot express the figure without distorting the comparison or requiring brittle workarounds.
- With Seaborn, prefer figure-level functions such as `relplot`, `catplot`, and `displot` when they fit.
  Use axes-level Seaborn or Matplotlib for genuinely custom composition.

## Size, Typography, And Layout

- Keep the plotting theme's typography consistent.
  Improve legibility by changing the figure, facet, or intended display size rather than manually setting individual font sizes.
- Write concise, sentence-case axis labels and legend titles with an initial capital, such as `Parameters` and `Loss`.
  Put longer definitions in the caption or prose.
- Prefer square physical plot boxes for quantitative subplots.
  Set the box aspect, such as `aspect=1` in a Seaborn figure-level function or `set_box_aspect(1)` on a Matplotlib axes.
  Do not force equal x and y data-unit scaling.
  Deviate when the data or plot form benefits materially from another aspect ratio.
- Keep spacing between subplots compact.
  Size the canvas so square plot boxes fill their allotted cells without large empty gaps.
- Preserve a semantic legend title.
  Place the legend where it does not overlap data or create excessive whitespace.
- Map a variable to one visual channel by default.
  Do not vary color and line style or marker shape for the same variable unless monochrome output or explicit human direction requires redundant encoding.

## Preserve The Meaning

- Label metrics, units, aggregation, sample definition, and transformations explicitly in the caption or surrounding prose when they do not fit cleanly in the figure.
- Choose the estimator, aggregation, unit of independence, uncertainty method, confidence level when applicable, and resampling unit explicitly.
- Show uncertainty by default when it can be estimated.
  Name the quantity, such as standard error, confidence interval, standard deviation, or range.
  Omit it only when uncertainty is not meaningful for the statistic or explicit human direction calls for another presentation.
- Draw standard-error error bars without terminal caps.
  Set `capsize=0` or the equivalent explicitly when the plotting interface exposes that setting.
- Share an axis only when the common scale supports the intended comparison without obscuring variation.
  Otherwise use clearly labeled independent axes, especially for panels that are not comparable by construction or have materially different useful ranges.
- Preserve negative results and relevant variation.
  Do not tune axis limits, smoothing, filtering, or color scales to exaggerate an effect.

## Produce And Inspect The Artifact

- Emit SVG by default for a static figure.
  Add PNG, PDF, or another format only when a downstream consumer requires it.
- Inspect the rendered artifact before publishing it.
  Check title length, label capitalization, typography, plot-box aspect, subplot spacing, legend title and order, clipping, numeric scales, and whether the visual supports the stated takeaway.
- Commit every SVG referenced by a knowledge-base experiment page under `docs/research/experiments/figures/<issue>/` on `main`.
- Store other small figures under `.agents/artifacts/<topic>/` on the permanent task or research branch.
- Use a raw commit-pinned repository URL when a branch figure must render on GitHub.

## Embed Figures In Markdown

Center every plot embedded in Markdown with this exact wrapper:

```html
<p align="center">
  <img src="<relative-path-or-commit-pinned-URL>" alt="<accessible description>" />
</p>
```

Put the caption in Markdown prose after the closing `</p>`:

```markdown
_<Caption in Markdown prose.>_
```

Do not bake captions or figure numbers into the SVG, PNG, or other plot artifact.
