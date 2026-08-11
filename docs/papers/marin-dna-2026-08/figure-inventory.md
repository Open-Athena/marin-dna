# Figure inventory and disposition proposal

This inventory covers every SVG in the frozen mechanical manuscript baseline. “Proposed disposition” is an editorial recommendation for review, not an approved move. No source figure should be redrawn or combined until these dispositions are accepted. Grouping is proposed only when panels answer one question and benefit from direct comparison.

| Asset | Current placement | Claim or role | Proposed disposition | Grouping hypothesis and required work |
|---|---|---|---|---|
| headline_cost_performance.svg | Main, final-comparison Results | Final zero-shot performance versus deployed throughput | Main, move to final-comparison Results | Candidate panel with the two frozen leaderboard views only if labels remain readable. Verify frozen AUPRC, throughput, hardware, and context-length annotations. |
| data_provenance_training_datasets.svg | Main, early-mixture Results | Provenance and scale of annotation-derived and alignment-projected training data | Main, data-curation Results or Methods | Keep standalone unless it can share a consistent data overview with the annotation-derived pool without duplicating quantities. Add immutable dataset revisions. |
| eval_datasets.svg | Main, Methods | Mendelian and SGE benchmark contents and subset sizes | Main, Methods | Candidate panel A of an evaluation-design figure with eval_apparatus.svg. Add frozen dataset revisions, split labels, and sample sizes needed for interpretation. |
| eval_apparatus.svg | Main, Methods | Zero-shot LLR and frozen-embedding linear-probe readouts | Main, Methods | Candidate panel B with eval_datasets.svg; align terminology with Methods and captions. |
| promoter_cds_specialists.svg | Main, early mixture Results | Early region specialists can approach Evo 2 in matched regions but remain behind GPN-Star | Supplementary | Preserve as motivation and provenance because these models use an earlier 512-bp, no-BOS recipe and are not directly comparable to later runs. |
| upstream_cds_balance.svg | Main, early mixture Results | Equal sampling balances upstream and CDS performance better than proportional pooling | Main | Keep standalone unless a concise data-mixture figure can pair it with later lineage evidence without implying matched experimental designs. |
| annotation_derived_training_pool.svg | Main, hyperparameter-transfer Results | Approximately 85B-token proportional pool used for the scaling work | Supplementary or merge into training-data overview | If merged, clearly distinguish the scaling-study pool from the later five-region data inventory. |
| parameter_transfer_methodology_v1.svg | Main, hyperparameter-transfer Results | Reference calibration and target-scale transfer workflow | Main | Candidate panel A with direct learning-rate validation. Redraw labels only after Methods notation is fixed. |
| figure1_lr_transfer.svg | Main, hyperparameter-transfer Results | Transferred learning rate is the best observed tested value at 255M, 476M, and 1B | Main | Candidate panel B with methodology. Caption must state search points, objective, and “best observed” scope. |
| figure2_beta2_epsilon_transfer.svg | Main, hyperparameter-transfer Results | β₂ and ε transfer validation | Supplementary | Pair with region-specific transfer as extended optimizer-transfer evidence if compatible at final width. |
| figure3_region_hyper_transfer.svg | Supplementary | Transfer behavior across CDS, upstream, and downstream validation sets | Supplementary | Candidate companion panel to figure2_beta2_epsilon_transfer.svg; retain readable labels and independent scales where applicable. |
| figure4_loss_scaling.svg | Main, parameter-scaling Results | Stable loss reduction and Kaplan-style scaling fit from 46M to 4B | Main | Keep standalone. Add exact fit equation/procedure and clarify that validation loss is a weighted monitoring statistic, not a conventional held-out estimate. |
| figure5_params_vs_vep_auprc.svg | Main, downstream Results | VEP behavior across parameter scale and readout | Main | Keep standalone or pair only with figure6_loss_vs_vep_auprc.svg if both remain legible; harmonize facet order, colors, metrics, and uncertainty labels. |
| figure6_loss_vs_vep_auprc.svg | Main, downstream Results | Relationship between matched-region validation likelihood and VEP | Main | Candidate companion to parameter-versus-VEP. State fit type, Pearson correlation, model count, and SE definition. |
| figure6b_marin_evo2_missense.svg | Main, downstream Results | The zero-shot/linear-probe missense divergence recurs across MarinDNA and Evo 2 scales | Main | Keep as the explicit qualification to the scaling claim; could become a focused panel within the scaling/VEP evidentiary unit if readable. |
| continued_training_data_exposures.svg | Main, later-mixture Results | Exact region exposure histories for m5.1, m1.3, and m3.3 | Main | Candidate panel A with zero-shot lineage trajectories because the exposure history is necessary to interpret them. |
| figure16_offline_lineage_llr_prototype.svg | Main, later-mixture Results | Zero-shot lineage trajectories and endpoints | Main | Candidate panel B with exposure history only if nine facets remain readable. Replace “prototype” artifact naming when a final recipe is available. |
| figure16_offline_lineage_probe_prototype.svg | Supplementary | Linear-probe lineage trajectories and endpoints | Supplementary | Retain as the complementary readout. Match labels, scales, checkpoint annotations, and uncertainty language to the main zero-shot figure. |
| figure11_leaderboard_heatmap__mendelian_llr.svg | Main, final-comparison Results | Frozen zero-shot Mendelian comparison across MarinDNA, Evo 2, GPN-Star, and AlphaGenome | Main | Candidate panel with probe heatmap, but annotate non-equivalent scoring protocols and model revisions. Keep the evolving leaderboard out of the panel. |
| figure11_leaderboard_heatmap__mendelian_probe.svg | Main, final-comparison Results | Frozen linear-probe comparison for models with compatible embeddings | Main | Candidate companion to zero-shot heatmap. Make absence of GPN-Star and AlphaGenome explicit and preserve the aggregation caveat. |

## Proposed figure architecture

The proposal retains the evidence needed to evaluate each central claim while moving older-recipe and secondary transfer diagnostics to Supplementary Information. It does not set a target count and does not require every hypothesis below to become a multi-panel figure.

1. **Training-data provenance:** data_provenance_training_datasets.svg, possibly with a compact representation of the scaling-study pool.
2. **Evaluation design:** eval_datasets.svg plus eval_apparatus.svg as coordinated panels in Methods.
3. **Mixture balance:** upstream_cds_balance.svg; early specialist evidence moves to Supplementary Information.
4. **Hyperparameter transfer:** methodology plus direct learning-rate validation in the main text; β₂, ε, and region diagnostics in Supplementary Information.
5. **Scaling:** loss scaling as a standalone main figure; parameter-versus-VEP and loss-versus-VEP as coordinated views if readable; retain the missense readout result as an explicit evidentiary qualification.
6. **Continued-training mixtures:** exposure history and zero-shot lineage results in the main text, with linear-probe trajectories in Supplementary Information.
7. **Frozen final comparison:** deployed efficiency and both frozen leaderboard readouts as one evidentiary unit, split across figures if a combined layout would shrink labels or obscure protocol differences.

## Figure-wide verification checklist

- [ ] Record the source parquet or table, generating recipe, code commit, and model/dataset revisions for every retained panel.
- [ ] Fix final physical widths before regenerating; target readable 7–9 pt figure text at 100% PDF scale.
- [ ] Use consistent model names, order, region names, colors, markers, metric definitions, and units.
- [ ] Add non-color distinctions where needed and check grayscale and color-vision-deficiency legibility.
- [ ] State sample size, aggregation, uncertainty, and frozen evaluation snapshot where interpretation requires them.
- [ ] Use capless error bars for ±1 SE; reserve caps for intervals with defined endpoints.
- [ ] Use independent, color-coded twin y-axes when two metrics are not level-comparable, and state that the axes are independent.
- [ ] Make captions self-contained while keeping procedural detail in Methods.
- [ ] Render and inspect the PDF; do not rely on the source SVG alone.
- [ ] Emit both SVG and PNG from any regenerated plot recipe, following repository plot conventions.
