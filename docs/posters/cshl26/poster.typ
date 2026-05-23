// CSHL 2026 poster — Typst implementation.
//
// Built on the `pollux` Typst package (v0.1.0), matching the Beamer
// Gemini look that the GPN-MSA poster uses:
//   - column-box headings are centred coloured text with a thin
//     underline rule
//   - filled steel-blue title band runs edge-to-edge of the page
//   - Open Sans throughout (title + body + footer)
//
// Compile with:  typst compile poster.typ
//   → poster.pdf
//
// FONT REQUIREMENT: expects the *discrete-weight* static TTFs of
// Open Sans (OpenSans-{Regular,SemiBold,Bold,...}.ttf) installed
// under ~/Library/Fonts/. The variable-font release that Homebrew's
// `font-open-sans` cask ships only exposes weight 400 to Typst, so
// bold (`*text*` / `#strong[…]`) won't render visibly with it.
// Discrete static TTFs are at
//   https://github.com/googlefonts/opensans/tree/main/fonts/ttf
//
// Figures in `figs/` come from `plots/cshl26_poster.py` (matplotlib
// SVGs: timescale, r3, specialist_bars) plus a few hand-coded
// cartoons (region_legend, timescale_legend).

#import "@preview/pollux:0.1.0": *

// ─── Page setup (44 in × 44 in) ────────────────────────────────────
// Zero margins on top + sides so the title band runs edge-to-edge.
// Bottom margin holds the page footer. Body + footer add their own
// 1in side padding via #pad(x: 1in).
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

// Override the default math font (serif) so equations match the
// body. Typst warns "current font is not designed for math" — fine
// for the simple expressions in the gLM overview box.
#show math.equation: set text(font: ("Open Sans", "Inter"))

// ─── Theme + layout ────────────────────────────────────────────────
// Steel-blue pollux theme (heading + fill rgb(64,115,158), stroke
// rgb(39,60,117)). Font sizes sized for a 44in × 44in poster with a
// 2-column body (~21in per column).
#set-theme(steel-blue)
#update-theme(
  body-size: 36pt,
  heading-size: 44pt,
  title-size: 80pt,
  authors-size: 48pt,
  institutes-size: 36pt,
)
#set-poster-layout(layout-a0)

// ─── Local override: column-box with Open Sans ─────────────────────
// Pollux's stock column-box hard-codes Lato; we shadow it with the
// same structure but Open Sans for both heading and body.
#let column-box(
  body,
  heading: none,
) = context {
  let pt = _state-poster-theme.get()
  let heading-color = pt.at("heading-color", default: rgb(64, 115, 158))
  let heading-size = pt.at("heading-size", default: 42pt)
  let body-size = pt.at("body-size", default: 40pt)
  let body-font = ("Open Sans", "Lato")

  let heading-content = if heading == none { none } else {
    [
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
    ]
  }

  let body-content = if body == none { none } else {
    [
      #set text(
        fill: black,
        size: body-size,
        font: body-font,
        weight: "regular",
      )
      #body
    ]
  }

  stack(dir: ttb, heading-content, box(stroke: none)[#body-content])
}

// ─── alert-box — same as column-box but with a tinted background ───
// Beamer Gemini's `alertblock`: same heading style as column-box, but
// the whole block sits inside a light steel-blue rectangle so the
// "headline" sections (Abstract, Summary) read as visually distinct.
#let alert-box(
  body,
  heading: none,
) = context {
  let pt = _state-poster-theme.get()
  let heading-color = pt.at("heading-color", default: rgb(64, 115, 158))
  let heading-size = pt.at("heading-size", default: 42pt)
  let body-size = pt.at("body-size", default: 40pt)
  let body-font = ("Open Sans", "Lato")
  // Light steel-blue tint (~85% mix toward white).
  let alert-fill = rgb(64, 115, 158).lighten(85%)

  let heading-content = if heading == none { none } else {
    [
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
    ]
  }

  let body-content = if body == none { none } else {
    [
      #set text(
        fill: black,
        size: body-size,
        font: body-font,
        weight: "regular",
      )
      #body
    ]
  }

  block(
    fill: alert-fill,
    width: 100%,
    inset: 0.8em,
    radius: 4pt,
  )[
    #stack(dir: ttb, heading-content, box(stroke: none)[#body-content])
  ]
}

// ─── Title block ───────────────────────────────────────────────────
// Filled steel-blue band with white centred text and the OA logo
// pinned to the right. The logo is a white variant of the SVG so it
// reads on the steel-blue band.
#block(
  fill: rgb(64, 115, 158),
  stroke: rgb(39, 60, 117),
  width: 100%,
  // x-inset matches the body's #pad(x: 1in) so the logo and title
  // align with the body columns.
  inset: (x: 1in, y: 0.5in),
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
    [],
    // spacer for visual balance with the logo on the right
    [
      #set align(center)
      #set text(size: 80pt)
      Data curation strategies for genomic language models \
      #v(0.5em, weak: true)
      #set text(size: 48pt)
      Gonzalo Benegas, Eric Czech \
      #set text(size: 36pt)
      Open Athena
    ],
    image("figs/icons/oa-logo-white.svg", width: 3in),
  )
]

#v(0.2in)

// ─── Body: 2 columns ───────────────────────────────────────────────
// Two wide columns (~21in each) so figures embedded at width:100%
// render large. Inner side padding matches the footer's 1in rhythm.
#pad(x: 1in)[
  #columns(2, gutter: 0.4in)[

    // ═════════════════ COLUMN 1: Setup + R1 ═══════════════════════════

    #alert-box(heading: "Abstract")[
      - *Pretraining data composition* is widely recognized as a key driver of LLM performance, but its role in *genomic language models* (gLMs) has not been systematically studied.
      - We investigate gLM data curation along two axes: *functional regions* (CDS, promoters) and *evolutionary timescales* (humans → animals).
    ]


    #let nucs(body) = box(text(font: "Menlo")[#body])

    #column-box(heading: "gLM overview")[
      *Training:* maximize the likelihood of sequences from reference genomes (typically from healthy individuals).

      #align(center)[
        $ "maximize" thin log P( dots.c#nucs[CACTTGGAT]dots.c ) $
      ]

      *Zero-shot VEP:* score a variant by how much it changes the
      likelihood compared to the reference. Low likelihood → likely deleterious.

      #align(center)[
        $
          "LLR" = log frac(
            P( dots.c#nucs[CACT#text(fill: rgb("#e63946"))[C]GGAT]dots.c ),
            P( dots.c#nucs[CACTTGGAT]dots.c )
          )
        $
      ]
    ]

    #v(0.5in)

    #column-box(heading: "Methods")[
      - Standard architecture and training objective: *Qwen3 autoregressive Transformer*.
        - Reuse LLM infrastructure and modeling science; focus on data.
      - *Context size: 256 bp*.
        - Focus on individual functional elements (e.g. exons, enhancers), and iterate faster.
      - *Model size: $tilde$1B* (currently exploring scaling).
      - Data sources: *RefSeq annotation* for genic regions, *ENCODE SCREEN* + sequence alignment for enhancers.
      - Human VEP evaluation: classify *Mendelian pathogenic vs. gnomAD high-frequency* variants (similar to TraitGym).
    ]


    #column-box(heading: [How to do one region well])[
      - The genome is very heterogeneous; coding sequences (CDS) and regulatory regions have a different grammar.
      - We first trained specialist models, each on a single region of the genome (default: animals for CDS/promoters, mammals for enhancers).
      #image("figs/region_legend.svg", width: 100%)
      #image("figs/specialist_bars.svg", width: 100%)
      - Each specialist achieves good performance in VEP tasks on its trained region.
      - Specialists often outperform Evo 2 (a much larger model), but not GPN-Star. Alignment-free trades some VEP performance for broad applicability (no whole-genome alignment required).
    ]

    #colbreak()

    // ═════════════════ COLUMN 2: R2 + Timescales + Summary ════════════

    #column-box(heading: [How to mix regions])[
      - How can we build a generalist model? Train on the different regions concatenated?
      - We evaluated this standard approach (proportional mixing) together with balanced sampling (uniform mixing).
      #image("figs/r3.svg", width: 100%)
      - Proportional mixing shows poor performance on the minority region (promoters).
      - Uniform mixing allows good progress across both promoter and missense variants.
    ]


    #column-box(heading: [Which evolutionary timescale?])[
      - If we have a target species of interest (e.g. human), which species should we train on?
      - There is a tradeoff between dataset size, diversity, and evolutionary relevance.
      #image("figs/timescale_legend.svg", width: 100%)
      #image("figs/timescale.svg", width: 100%)
      - For regulatory regions (promoters, 3' UTR), mammals converges fast; vertebrates / animals climb more slowly but may surpass with more compute.
      - In the more conserved CDS region, the value of larger evolutionary timescales is much more evident.
    ]


    #alert-box(heading: "Summary and outlook")[
      - We explore data curation strategies along two separate axes: *functional regions* and *evolutionary timescales*.
      - Our findings provide insights for developing *efficient gLMs with robust performance across the genome*.
      - As next steps, we are exploring how data curation recipes interact with varying *model scale*. We are also interested in transfer learning applications such as *gene expression prediction*.
      - Inspired by the #link("https://github.com/marin-community/marin")[Marin] project, we go beyond public code/model/data: *the research process itself is open*, with all experiments preregistered and publicly available from day 1.
    ]

  ]
]
