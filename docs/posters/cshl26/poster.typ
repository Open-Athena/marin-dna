// CSHL 2026 poster — Typst parallel implementation.
//
// Built on the `peace-of-posters` package (v0.6.0), which is the same
// poster framework that the `pollux` template uses. Gemini-style
// section title bars, multi-column body, custom theme.
//
// Compile with:  typst compile poster.typ
//   → poster.pdf
//
// Sources the same SVG figures as poster.html in figs/.
// Math (training + VEP formulas) uses native Typst math.

#import "@preview/peace-of-posters:0.6.0" as pop

// ─── Page setup (44 in × 44 in, matches @page in poster.html) ──────
#set page(
  paper: "a4", // overridden by width/height below
  width: 44in,
  height: 44in,
  margin: (top: 1in, bottom: 1.4in, x: 1in),
  background: rect(fill: white, width: 100%, height: 100%),
  // True page footer — sits at the bottom of every page, independent
  // of body flow. Three columns: acknowledgments | code | contact.
  // Bottom margin (1.4in) sized to hold the single-line footer with
  // a small descent below it.
  footer-descent: 0.3in,
  footer: [
    #line(length: 100%, stroke: 1pt + rgb("#1a1a1a"))
    #v(0.1in)
    #grid(
      columns: (2.8fr, 0.95fr, 0.75fr),
      align: (left, right, right),
      column-gutter: 0.4in,
      [
        #set text(size: 22pt, fill: rgb("#1a1a1a"))
        *Acknowledgments:* thanks to the
        #link("https://github.com/marin-community/marin")[Marin] team
        for their excellent framework and inspiring open development.
        Compute generously provided by the
        #link("https://sites.research.google/trc/about/")[Google TPU Research Cloud].
      ],
      [
        #set text(size: 22pt)
        *Code & experiments:*
        #link("https://github.com/Open-Athena/marin-dna")[github.com/Open-Athena/marin-dna]
      ],
      [
        #set text(size: 22pt)
        *Contact:*
        #link("mailto:gonzalo.benegas@openathena.ai")[gonzalo.benegas\@openathena.ai]
      ],
    )
  ],
)

#set text(font: "Inter", size: 24pt, fill: rgb("#1a1a1a"))
#set par(leading: 0.55em)

// ─── Custom theme (navy section bars, white bg) ───────────────────
#pop.set-theme((
  "body-box-args": (
    inset: 0.3em,
    stroke: none,
    fill: white,
  ),
  "body-text-args": (
    fill: rgb("#1a1a1a"),
  ),
  "heading-box-args": (
    inset: 0.6em,
    width: 100%,
    fill: rgb("#1d3557"),
    stroke: none,
    radius: 6pt,
  ),
  "heading-text-args": (
    fill: white,
    weight: "semibold",
    size: 30pt,
  ),
))

// ─── Header (title + authors + affiliation + OA logo) ──────────────
// 3-column grid: a transparent spacer on the left mirrors the logo
// width on the right, so the title in the centre column stays
// optically centred between them (Gemini-poster convention).
#grid(
  columns: (3.5in, 1fr, 3.5in),
  align: (left + horizon, center + horizon, right + horizon),
  column-gutter: 0.3in,
  [],
  // empty (visual balance for the logo on the right)
  [
    #set text(weight: "bold", size: 80pt)
    #par(leading: 0.35em)[Data curation strategies for genomic language models]

    #v(-0.05in)
    #set text(weight: "regular", size: 48pt)
    Gonzalo Benegas, Eric Czech

    #set text(style: "italic", size: 36pt, fill: rgb("#666666"))
    Open Athena
  ],
  image("figs/icons/oa-logo.svg", width: 3in),
)

#v(0.3in)
#line(length: 100%, stroke: 2pt + rgb("#1a1a1a"))
#v(0.2in)

// ─── Body: 3 columns ──────────────────────────────────────────────
#columns(3, gutter: 0.4in)[

  // ═════════════════ COLUMN A: Abstract + Approach ═══════════════════

  #pop.column-box(heading: "Abstract")[
    - Genomic LMs scale by training larger architectures on more diverse
      genomes — but the *composition of the training data* is itself
      a design axis.
    - We hold architecture, training objective, and compute budget
      fixed; vary the data along *functional region* and
      *evolutionary timescale*.
    - Each variant is scored zero-shot by the log-likelihood ratio
      between alt and ref alleles, evaluated on *ClinVar Mendelian*
      variants per region subset.
  ]

  // Helper: a single inline `box` of monospaced DNA — keeps the
  // surrounding text in one block so adjacent runs don't pick up
  // math-mode spacing. Embed `#text(fill: red)[X]` inside for an
  // alt-position highlight.
  #let nucs(body) = box(text(font: "Menlo")[#body])

  #pop.column-box(heading: "Training")[
    Maximize the likelihood of observed (reference) DNA.

    #align(center)[
      $ "maximize" thin log P( dots.c #nucs[CACTTGGAT] dots.c ) $
    ]
  ]

  #pop.column-box(heading: "Zero-shot variant effect prediction")[
    Score a variant by how much it changes the likelihood.

    #align(center)[
      $
        "LLR" = log frac(
          P( dots.c #nucs[CACT#text(fill: rgb("#e63946"))[C]GGAT] dots.c ),
          P( dots.c #nucs[CACTTGGAT] dots.c )
        )
      $
    ]
  ]

  #pop.column-box(heading: "Methods")[
    - *Architecture:* Qwen decoder-only Transformer; objective
      $P(x_t | x_(<t))$; FWD + RC averaging.
    - *Sizes:* 6M – 1.7B parameters; we vary _what_ we train on,
      not how large.
    - *Data:* per-experiment subsets (promoter, CDS, enhancer,
      mixtures, multi-species).
    - *Benchmark:* ClinVar Mendelian variants (pathogenic vs benign),
      per consequence subset.
    - *Score:* log-likelihood ratio (alt / ref) at the variant
      position, zero-shot.
    - *Metric:* AUPRC, with per-cluster bootstrap SE.
  ]

  #colbreak()

  // ═════════════════ COLUMN B: Functional regions (R1 + R2) ═══════════

  // Gene cartoon at the top doubles as colour-key for R1 + R2 below.
  // #v(0.15in)

  #pop.column-box(heading: [Region specialists achieve competitive performance])[
    #image("figs/region_legend.svg", width: 100%)
    - We trained specialist models, each on a single region of the genome.
    #image("figs/specialist_bars.svg", width: 100%)
    - Each specialist achieves good performance in VEP tasks on its trained region.
    - Specialists often outperform Evo 2, but not GPN-Star. There might be a ceiling to the performance of alignment-free gLMs.
  ]

  #pop.column-box(heading: [Balanced sampling rescues the under-represented region])[
    #image("figs/r2_composition.svg", width: 100%)
    #v(0.1in)
    #image("figs/r3.svg", width: 100%)

    - Proportional sampling (natural 10 / 90 ratio) under-serves the
      rare region — promoter AUPRC stays low.
    - Uniform 50 / 50 lifts promoter AUPRC into specialist range,
      with little cost on missense.
    - One mixed model can serve both regions — no per-region training needed.
  ]

  #colbreak()

  // ═════════════════ COLUMN C: Evolutionary timescales + Summary ═════

  // Phylo tree at the top doubles as colour-key for T1 + T2 below.
  #image("figs/timescale_legend.svg", width: 100%)
  #v(0.15in)

  #pop.column-box(heading: [Promoters peak at the mammals timescale])[
    #image("figs/t1.svg", width: 100%, height: 7in, fit: "contain")

    - Promoter signal peaks at mammals (~100 Mya) — broader is worse.
    - Suggests promoter regulatory grammar is largely mammal-specific.
  ]

  #pop.column-box(heading: [CDS keeps improving out to the animals timescale])[
    #image("figs/t2.svg", width: 100%, height: 7in, fit: "contain")

    - CDS signal still gaining at the animals timescale (~700 Mya).
    - Suggests protein-coding grammar generalises across deep time.
  ]

  #pop.column-box(heading: "Summary")[
    - Region specialists *match* whole-genome + multi-species
      generalists, each on its own region.
    - *Balanced sampling* (50 / 50) rescues the under-represented
      region — one mixed model can serve both.
    - Promoter signal peaks at *mammals*; CDS keeps improving to
      *animals* — region × timescale interacts.
    - Treat training-data composition as a *curation* decision,
      not just a scale-up decision.
  ]

]

// (Footer rendered via `set page(footer: ...)` above — no inline block.)
