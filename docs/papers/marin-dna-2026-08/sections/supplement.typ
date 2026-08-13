#import "../template.typ": paper-figure

#pagebreak()
#set heading(numbering: none)
#set figure(
  numbering: "S1",
  supplement: [Supplementary Figure],
)
#counter(figure.where(kind: image)).update(0)

= Supplementary Information
<supplementary-information>

The supplementary figures report region-stratified hyperparameter-transfer results and the linear-probe trajectories corresponding to the zero-shot mixture analysis in Results.

#paper-figure(
  "figures/figure3_region_hyper_transfer.svg",
  id: "fig-region-hyperparameter-transfer",
  alt: "Hyperparameter transfer validated per genomic region",
  caption: [*Region-specific hyperparameter transfer.* Validation loss as a function of learning rate, β₂, and ε, evaluated separately for CDS, upstream, and downstream regions.],
)

#paper-figure(
  "figures/figure16_offline_lineage_probe_prototype.svg",
  id: "fig-mixture-lineage-probe",
  alt: "Nine-panel linear-probe Mendelian chromosome-weighted AUPRC (%) trajectories with error bars along each mixture lineage",
  caption: [*Linear-probe mixture-lineage trajectories.* Linear-probe Mendelian AUPRC (chromosome-averaged) versus training tokens for three model-mixture lineages. Error bars denote SE.],
)
