#let ink = rgb("#1f1e1b")
#let accent = rgb("#2f6f63")
#let muted = rgb("#66635e")

#let max-figure-render-width = 700.0
#let figure-render-widths = (
  "figures/annotation_derived_training_pool.svg": 640.0,
  "figures/continued_training_data_exposures.svg": 700.0,
  "figures/data_provenance_training_datasets.svg": 620.0,
  "figures/eval_apparatus.svg": 640.0,
  "figures/eval_datasets.svg": 700.0,
  "figures/figure11_leaderboard_heatmap__mendelian_llr.svg": 628.41,
  "figures/figure11_leaderboard_heatmap__mendelian_probe.svg": 625.482,
  "figures/figure16_offline_lineage_llr_prototype.svg": 527.68,
  "figures/figure16_offline_lineage_probe_prototype.svg": 527.68,
  "figures/figure1_lr_transfer.svg": 370.702,
  "figures/figure2_beta2_epsilon_transfer.svg": 436.974,
  "figures/figure3_region_hyper_transfer.svg": 622.7,
  "figures/figure4_loss_scaling.svg": 661.543,
  "figures/figure5_params_vs_vep_auprc.svg": 514.969,
  "figures/figure6_loss_vs_vep_auprc.svg": 522.454,
  "figures/figure6b_marin_evo2_missense.svg": 377.506,
  "figures/headline_cost_performance.svg": 438.472,
  "figures/parameter_transfer_methodology_v1.svg": 660.0,
  "figures/promoter_cds_specialists.svg": 437.357,
  "figures/upstream_cds_balance.svg": 495.228,
)

#let paper-figure(
  path,
  caption: none,
  id: none,
  alt: none,
  width: none,
) = {
  let render-width = figure-render-widths.at(path)
  let image-width = if width == none { 100% * render-width / max-figure-render-width } else { width }
  [
    #figure(
      image(path, width: image-width, alt: alt),
      caption: caption,
    )#label(id)
  ]
}

#let supplementary-figure-ref(target) = context {
  let number = counter(figure.where(kind: image)).at(target)
  link(target)[Supplementary Figure #numbering("S1", ..number)]
}

#let paper(
  title: none,
  authors: none,
  author-metadata: none,
  date: none,
  abstract: none,
  body,
) = {
  set document(
    title: title,
    author: author-metadata,
    keywords: ("genomic language models", "DNA", "variant effect prediction", "scaling laws"),
  )

  set page(
    paper: "us-letter",
    margin: (top: 0.85in, bottom: 0.85in, x: 0.9in),
    numbering: "1",
    number-align: center,
  )
  set text(font: "Libertinus Serif", size: 10.5pt, fill: ink, lang: "en")
  set par(justify: true, leading: 0.62em, spacing: 0.72em)
  set heading(numbering: "1.")
  set figure(gap: 0.65em)
  set figure.caption(position: bottom, separator: [ ])
  set footnote.entry(gap: 0.35em)
  show link: set text(fill: accent)

  show heading.where(level: 1): it => {
    block(above: 1.2em, below: 0.65em)[
      #set text(size: 15pt, weight: "semibold", fill: ink)
      #it
    ]
  }
  show heading.where(level: 2): it => block(above: 0.9em, below: 0.45em)[
    #set text(size: 12pt, weight: "semibold", fill: ink)
    #it
  ]
  align(center)[
    #block(width: 92%)[
      #set par(justify: false)
      #text(size: 19pt, weight: "bold", fill: ink)[#title]
      #v(0.75em)
      #text(size: 11.5pt, weight: "semibold")[#authors]
      #v(0.25em)
      #text(size: 9.5pt, fill: muted)[#date]
    ]
  ]

  v(1em)
  pad(x: 7%)[
    #set text(size: 9.5pt)
    #set par(first-line-indent: 0pt)
    #align(center)[#strong[Abstract]]
    #v(0.35em)
    #abstract
  ]
  v(0.8em)

  body
}
