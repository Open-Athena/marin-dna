# MarinDNA paper — August 2026

This directory contains the raw-Typst manuscript converted from the published [MarinDNA article](https://openathena.ai/blog/marin-dna/). The initial conversion uses the final publication snapshot at MarinDNA commit `d8c4803cbbbffafb24890cd0c75134d78368d55c` as its content and figure source.

The Typst files are the manuscript's source of truth. Pandoc was used only for the initial mechanical conversion and is not a build dependency. There is intentionally no CI workflow for this paper.

Issue [#449](https://github.com/Open-Athena/marin-dna/issues/449) tracks the editorial conversion into a bioRxiv-ready preprint. Before changing the scientific narrative or redrawing figures, consult:

- [editorial-plan.md](editorial-plan.md) for the locked brief, frozen-snapshot ledger, headline claim–evidence map, and unresolved verification work.
- [figure-inventory.md](figure-inventory.md) for the complete frozen asset inventory and the proposed main-versus-supplement dispositions.

The issue body remains the authoritative work plan. These files are checked-in working records that make editorial and provenance decisions reviewable alongside the manuscript.

## Compile

Install [Typst 0.15.1](https://github.com/typst/typst/releases/tag/v0.15.1), the version used to verify this manuscript, and confirm it is available:

    typst --version

Then run from the repository root:

    typst compile docs/papers/marin-dna-2026-08/main.typ docs/papers/marin-dna-2026-08/paper.pdf

For live local preview:

    typst watch docs/papers/marin-dna-2026-08/main.typ docs/papers/marin-dna-2026-08/paper.pdf

## Editing conventions

- Keep one sentence per source line when editing prose.
- Use semantic labels and references instead of hard-coding figure or section numbers.
- Keep prose in `sections/`; keep document-wide presentation in `template.typ`.
- Keep figures in `figures/` and preserve their source provenance when replacing them.
- Set manuscript text and captions in Libertinus Serif.
- Use Lato for all text inside imported SVG figures, and ensure it is available when regenerating those figures.
- Preserve each SVG's normalized `data-figure-render-width`; `template.typ` scales those widths relative to the 700 px maximum so figure text remains consistent instead of stretching every asset to the full paper measure.
- Render figures directly on the page; do not add decorative container blocks, fills, borders, padding, or rounded corners.
- Place figures that were hidden behind disclosure controls in the blog under Supplementary Information and number them S1, S2, and so on.
- The author list is deliberately `TODO` until the collaborators agree on authorship and order.
