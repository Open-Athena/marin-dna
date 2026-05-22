// CSHL 2026 poster — pollux theme prototype.
//
// Sister file to poster.typ — same content and figures, swapped onto
// the `pollux` Typst package (v0.1.0) so we can compare. Matches the
// Beamer Gemini look that the GPN-MSA poster uses:
//   - column-box headings are centred coloured text with a thin
//     underline rule (vs peace-of-posters' filled navy heading bars)
//   - filled steel-blue title band runs edge-to-edge of the page
//   - Open Sans throughout (title + body + footer)
//
// Compile with:  typst compile poster_pollux.typ
//   → poster_pollux.pdf
//
// FONT REQUIREMENT: this prototype expects the *discrete-weight*
// static TTFs of Open Sans (OpenSans-{Regular,SemiBold,Bold,...}.ttf)
// installed under ~/Library/Fonts/. The variable-font release that
// Homebrew's `font-open-sans` ships only exposes weight 400 to
// Typst, so bold (`*text*` / `#strong[…]`) won't render visibly.
// Static TTFs are at https://github.com/googlefonts/opensans/tree/main/fonts/ttf
//
// Both poster.typ and poster_pollux.typ source the same SVGs in
// figs/, so changes upstream (plots/cshl26_poster.py output) are
// reflected in both PDFs.

#import "@preview/pollux:0.1.0": *

// ─── Page setup (44 in × 44 in, matches poster.typ) ────────────────
// Zero margins on top + sides so the title band runs edge-to-edge
// (the Beamer Gemini / pollux convention). Bottom margin is kept
// non-zero so the page footer has room to render. Inner side
// padding is applied later via #pad(x: 1in) around the body and
// footer content.
#set page(
  width: 44in,
  height: 44in,
  margin: (top: 0in, bottom: 1.4in, x: 0in),
  background: rect(fill: white, width: 100%, height: 100%),
  footer-descent: 0.3in,
  footer: pad(x: 1in)[
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

#set text(font: ("Open Sans", "Inter"), size: 24pt, fill: rgb("#1a1a1a"))
#set par(leading: 0.55em)

// ─── Theme + layout ────────────────────────────────────────────────
// Steel-blue is the closest stock pollux theme to our navy. The other
// stock options are solar-orange, forest-green, crimson-accent,
// teal-mist, royal-purple. update-theme() can override individual
// colors (heading-color / fill-color / stroke-color) if we want to
// match our exact #1d3557 navy.
#set-theme(steel-blue)
#update-theme(
  // Keep pollux's stock steel-blue palette — it's the Gemini look the
  // user is after. (Steel-blue: heading + fill rgb(64,115,158),
  // stroke rgb(39,60,117).) Only override the font sizes; pollux's A0
  // layout defaults are tuned for ~33in-wide posters, ours is 44in
  // so we want a proportional bump.
  body-size:     30pt,
  heading-size:  36pt,
  title-size:    80pt,
  authors-size:  48pt,
  institutes-size: 36pt,
)

// Set the layout state so the font sizes above are applied. We start
// from layout-a0 and override; pollux reads body-size etc. off the
// theme state.
#set-poster-layout(layout-a0)

// ─── Local override: column-box with Open Sans ─────────────────────
// Pollux's stock column-box hard-codes Lato in its body's #set text;
// we shadow it here with the same structure but Open Sans, to match
// the font Beamer Gemini (and the GPN-MSA poster) actually render
// in. Heading + body both use Open Sans; Lato falls back if Open
// Sans isn't installed locally.
#let column-box(
  body,
  heading: none,
) = context {
  let pt = _state-poster-theme.get()
  let heading-color = pt.at("heading-color", default: rgb(64, 115, 158))
  let heading-size  = pt.at("heading-size", default: 42pt)
  let body-size     = pt.at("body-size", default: 40pt)
  let body-font = ("Open Sans", "Lato")

  let heading-content = if heading == none { none } else {[
    #set text(
      fill: heading-color,
      size: heading-size,
      font: body-font,
      weight: "medium",
    )
    #set align(center)
    #box(width: 100%)[#heading]
    #v(-0.75em)
    #rect(width: 100%, height: 1.5pt, fill: black)
    #v(0.5em)
  ]}

  let body-content = if body == none { none } else {[
    #set text(
      fill: black,
      size: body-size,
      font: body-font,
      weight: "regular",
    )
    #body
  ]}

  stack(dir: ttb, heading-content, box(stroke: none)[#body-content])
}

// ─── Title block ───────────────────────────────────────────────────
// Custom-rolled title band mirroring pollux's title-box style (filled
// steel-blue rectangle, white centred text) but with the OA logo
// pinned to the right — matches Beamer Gemini's convention of logos
// sitting on the title bar. Logo uses a white variant of the SVG
// (figs/icons/oa-logo-white.svg) so it reads on the steel-blue band.
#block(
  fill: rgb(64, 115, 158),       // steel-blue (pollux's fill-color)
  stroke: rgb(39, 60, 117),      // steel-blue stroke
  width: 100%,
  // x-inset = 1in so the logo and title content align with the body
  // columns' left / right edges (which use #pad(x: 1in) below).
  // Otherwise the logo would sit flush against the page edge.
  inset: (x: 1in, y: 0.5em),
  radius: 0pt,
)[
  #set text(
    fill: white,
    font: ("Open Sans", "Lato"),
    weight: "regular",
    lang: "en",
  )
  #grid(
    columns: (3.5in, 1fr, 3.5in),
    align: (left + horizon, center + horizon, right + horizon),
    column-gutter: 0.3in,
    [],  // spacer for visual balance with the logo on the right
    [
      #v(0.4in)
      #set align(center)
      #set text(size: 80pt)
      Data curation strategies for genomic language models \
      #v(0.5em, weak: true)
      #set text(size: 48pt)
      Gonzalo Benegas, Eric Czech \
      #set text(size: 36pt)
      Open Athena
      #v(0.4in)
    ],
    image("figs/icons/oa-logo-white.svg", width: 3in),
  )
]

#v(0.2in)

// ─── Body: 3 columns ───────────────────────────────────────────────
// Indent the columns from the page edges (which the title band runs
// flush to) via pad. Same 1in left/right rhythm as the footer below.
#pad(x: 1in)[
#columns(3, gutter: 0.4in)[

  // ═════════════════ COLUMN A: Abstract / Intro / Methods ═══════════

  #column-box(heading: "Abstract")[
    - *Genomic language models* (gLMs) are effective genome-wide variant effect predictors
    - *GPN-Star*, the current SOTA, requires whole-genome alignments, which are only available for select organisms
    - We hold architecture, training objective, and compute budget
      fixed; vary the data along *functional region* and
      *evolutionary timescale*.
  ]

  #v(0.4in)

  #let nucs(body) = box(text(font: "Menlo")[#body])

  #column-box(heading: "Introduction")[
    Autoregressive language models of DNA — at each position, predict
    the next nucleotide given the preceding context.

    *Training:* maximize log-likelihood of reference DNA.

    #align(center)[
      $ "maximize" thin log P( dots.c #nucs[CACTTGGAT] dots.c ) $
    ]

    *Zero-shot VEP:* score a variant by how much it changes the
    likelihood.

    #align(center)[
      $
        "LLR" = log frac(
          P( dots.c #nucs[CACT#text(fill: rgb("#e63946"))[C]GGAT] dots.c ),
          P( dots.c #nucs[CACTTGGAT] dots.c )
        )
      $
    ]
  ]

  #v(0.4in)

  #column-box(heading: "Methods")[
    - *Architecture:* Qwen decoder-only Transformer; objective
      $P(x_t | x_(<t))$; FWD + RC averaging.
    - *Sizes:* 6M – 1.7B parameters; we vary _what_ we train on,
      not how large.
    - *Data:* per-experiment subsets (promoter, CDS, enhancer,
      mixtures, multi-species).
    - *Benchmark:* ClinVar Mendelian variants (pathogenic vs benign),
      per consequence subset.
    - *Metric:* AUPRC, with per-cluster bootstrap SE.
  ]

  #colbreak()

  // ═════════════════ COLUMN B: Functional regions (R1 + R2) ═════════

  #column-box(heading: [Region specialists achieve competitive performance])[
    #image("figs/region_legend.svg", width: 100%)
    - We trained specialist models, each on a single region of the genome.
    #image("figs/specialist_bars.svg", width: 100%)
    - Each specialist achieves good performance in VEP tasks on its trained region.
    - Specialists often outperform Evo 2, but not GPN-Star. There might be a ceiling to the performance of alignment-free gLMs.
  ]

  #v(0.4in)

  #column-box(heading: [Balanced sampling rescues the under-represented region])[
    - We have good data recipes for specialist models. How do we build a generalist model? Simply train on a concatenation of the individual datasets?
    - We evaluated this approach (proportional mixing) together with balanced sampling (uniform mixing).
    #image("figs/r3.svg", width: 100%)

    - Proportional sampling (natural 10 / 90 ratio) under-serves the
      rare region — promoter AUPRC stays low.
    - Uniform 50 / 50 lifts promoter AUPRC into specialist range,
      with little cost on missense.
    - One mixed model can serve both regions — no per-region training needed.
  ]

  #colbreak()

  // ═════════════════ COLUMN C: Evolutionary timescales + Summary ════

  #column-box(heading: [Optimal timescale varies by region])[
    #image("figs/timescale_legend.svg", width: 100%)

    #image("figs/t1.svg", width: 100%, height: 5in, fit: "contain")
    - Promoter signal peaks at mammals (~100 Mya) — broader is worse.
    - Suggests promoter regulatory grammar is largely mammal-specific.

    #image("figs/t2.svg", width: 100%, height: 5in, fit: "contain")
    - CDS signal still gaining at the animals timescale (~700 Mya).
    - Suggests protein-coding grammar generalises across deep time.
  ]

  #v(0.4in)

  #column-box(heading: "Summary")[
    - Region specialists *match* whole-genome + multi-species
      generalists, each on its own region.
    - *Balanced sampling* (50 / 50) rescues the under-represented
      region — one mixed model can serve both.
    - Promoter signal peaks at *mammals*; CDS keeps improving to
      *animals* — region × timescale interacts.
    - Treat training-data composition as a *curation* decision,
      not just a scale-up decision.
  ]

  #v(0.4in)

  #column-box(heading: "References")[
    - _Placeholder — fill in._
  ]

]
]
