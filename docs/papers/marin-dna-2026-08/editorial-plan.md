# Editorial plan and claim–evidence ledger

This is the working editorial record for issue #449. It is not part of the rendered manuscript. The issue body remains the authoritative scope and work plan; this file records the concrete manuscript snapshot, claim boundaries, evidence mapping, and unresolved verification work needed to execute that plan.

## Locked editorial brief

- **Audience:** researchers who use machine learning in biology.
- **Central contribution:** the final 1B model and efficiency result and the data-mixture and hyperparameter-transfer recipe are co-equal. The final model establishes what was achieved; the recipe explains how it was achieved.
- **Claim language:** describe the frozen MarinDNA 1B result as statistically competitive with Evo 2 40B on the Mendelian evaluation. Do not convert a statistical tie into a superiority claim. Point-estimate differences may be reported only when their uncertainty and aggregation are stated.
- **Chronology:** retain chronology when an observation motivates a consequential methodological choice. Condense chronology that functions only as an issue-by-issue project log.
- **Structure:** Abstract; Introduction; Results; Discussion; Methods; Data, Code, Models, and Live Resources; Open Development and Provenance; end matter; References; Supplementary Information.
- **Format:** single-column Typst PDF with author-year scholarly citations and a compact, separate provenance layer.
- **Scientific scope:** editorial conversion of the frozen published result. New training, evaluation datasets, or analyses require an explicit scope decision in issue #449.

## Frozen-snapshot ledger

| Component | Frozen identifier | Status and required follow-up |
|---|---|---|
| Mechanical manuscript baseline | Commit 3b608d39b41c2330636ec647dbb25d26b0895187 | Fixed conversion baseline for editorial diffs. |
| Published article content and figures | MarinDNA commit d8c4803cbbbffafb24890cd0c75134d78368d55c | Fixed content and figure source; do not refresh from the live article or leaderboard. |
| Figure assets | The 20 SVGs committed under figures/ at the mechanical baseline | Frozen input set. Redrawn or composed figures must preserve source-data and recipe provenance. |
| Final MarinDNA model | Dashboard ID mix-v0.9-p1B-i24-exp135-m5.1-step-59158; W&B run dna-bolinas-mix-v0.9-p1B-i24-exp135-zoonomia-m5.1-bef41e; GCS checkpoint gs://marin-us-east5/checkpoints/dna-bolinas-mix-v0.9-p1B-i24-exp135-zoonomia-m5.1-bef41e/hf/step-59158 | Exact internal checkpoint is identified. Record an immutable public Hugging Face model revision before preprint release. |
| Mendelian evaluation dataset | Hugging Face revision 4aed58e50c5dea0b878a665007af2ef9e5108e9f of bolinas-dna/evals_mendelian_traits | Already cited in the converted manuscript. Verify that every frozen Mendelian panel used this exact revision. |
| SGE evaluation dataset | Hugging Face revision 225d3d1ea32a4af547891b13c33b5e92a5aae849 of bolinas-dna/evals_sge | Frozen v3 label build used by the evaluation pipeline; verify that every frozen SGE panel used this exact revision. |
| Frozen result tables | Encoded in the committed SVGs and baseline PDF, but source parquet paths and revisions are not yet recorded here | Resolve exact metric parquet paths, plot recipes, model lists, and aggregation settings for every retained quantitative panel. |
| Training and evaluation code | The converted draft contains several commit-pinned Marin links, but no single complete code snapshot is declared | Identify the commit(s) that reproduce training, dataset construction, evaluation, and throughput measurement. Use commit-pinned links in the provenance table. |
| Live resources | MarinDNA leaderboard and interactive tools | Keep discoverable in the availability section and label explicitly as evolving resources that are not the frozen manuscript result. |

## Headline claim–evidence ledger

| ID | Manuscript claim | Direct evidence in the frozen draft | Required methods or qualification | Editorial status |
|---|---|---|---|---|
| C1 | Explicit mixture balancing prevents the larger CDS dataset from dominating and produces more even coding/non-coding VEP performance. | upstream_cds_balance.svg; early specialist comparison in promoter_cds_specialists.svg; experiment provenance in issues #13, #21, and #27. | State that the early specialists used a different 512-bp, no-BOS setup and preceded the later systematic tuning recipe. Do not imply that 50/50 is globally optimal. | Supported with qualification. |
| C2 | Adding alignment-projected ncRNA and enhancer data during continued training improves corresponding ncRNA and distal readouts, and m5.1 has the best macro-average endpoint among the three frozen mixture lineages. | continued_training_data_exposures.svg; figure16_offline_lineage_llr_prototype.svg; figure16_offline_lineage_probe_prototype.svg; m5.1 dashboard entry. | Record checkpoint selection, evaluation cadence, subset definitions, whether each curve uses pooled or chromosome-weighted AUPRC, and exact uncertainty calculation. Treat exposure-order benefit as a hypothesis because the comparison is not a controlled ablation. | Endpoint claim supported; causal exposure-order claim must remain tentative. |
| C3 | A small-model hyperparameter calibration predicts the best observed learning rate at 255M, 476M, and 1B parameters while transferring across token horizon and batch size. | parameter_transfer_methodology_v1.svg; figure1_lr_transfer.svg; figure2_beta2_epsilon_transfer.svg; figure3_region_hyper_transfer.svg; commit-pinned Marin implementation links in the converted Results. | Define the reference and target search spaces, objective, number of runs, seeds, selection rule, model geometry rule, schedules, and exact transfer equations. Say “best observed setting” rather than “optimal” unless the tested grid justifies the stronger term. | Central claim supported within tested settings. |
| C4 | With transferred hyperparameters, loss decreases predictably from 46M to 4B parameters over an approximately 84B-token sweep. | figure4_loss_scaling.svg; reported Kaplan-style fit with R² = 0.999. | Record the eight exact model sizes, token counts, fit equation, fit procedure, residual diagnostics, and definition of the weighted validation statistic. Avoid treating the validation set as a conventional held-out generalization estimate. | Supported; validation-loss terminology needs tightening. |
| C5 | VEP generally improves with model scale, but zero-shot Mendelian missense regresses while linear-probe missense improves. | figure5_params_vs_vep_auprc.svg; figure6_loss_vs_vep_auprc.svg; figure6b_marin_evo2_missense.svg. | Define zero-shot LLR, embedding construction, probe split and fitting, chromosome aggregation, standard errors, and the matched-region validation score. Do not generalize the missense exception to all zero-shot VEP. | Supported with an explicit exception and scope boundary. |
| C6 | On the frozen Mendelian snapshot, MarinDNA m5.1 is statistically competitive with Evo 2 40B and closes much of the distal gap, while each model retains consequence-specific strengths. | figure11_leaderboard_heatmap__mendelian_llr.svg; figure11_leaderboard_heatmap__mendelian_probe.svg; headline_cost_performance.svg; commit 21343bc paired audit. | Across eight subsets and 1,610 matched groups, macro AUPRC is 39.49% for m5.1 versus 38.24% for Evo 2 40B: paired difference +1.25 percentage points, bootstrap SE 1.63, 95% CI −1.93 to 4.46, two-sided bootstrap p = 0.440. Make clear that GPN-Star uses cLLR and AlphaGenome uses a different supervised score; probe comparisons are unavailable for those models. | Supported with exact frozen statistical statement and scoring qualification. |
| C7 | m5.1 used about 1,980× fewer reported training FLOPs and achieved about 2,330× higher as-deployed variant-scoring throughput than Evo 2 40B. | headline_cost_performance.svg; converted Results report approximately 1.1e21 versus 2.25e24 training FLOPs and 41 minutes versus 66 days per million variants; issue #354 is the throughput provenance source. | Verify unrounded values and arithmetic. State hardware, software, batch size, forward/reverse-complement passes, embeddings, and native context lengths. This is as-deployed throughput, not same-context or per-token efficiency. | Supported with comparability limitation; numbers require final audit. |
| C8 | Alignment-free models remain behind alignment-based GPN-Star and supervised AlphaGenome on the broader zero-shot benchmark. | figure11_leaderboard_heatmap__mendelian_llr.svg. | Explain the non-equivalent training information and score definitions. This comparison bounds the paper's claim; it is not evidence that architecture alone explains the gap. | Supported and required in Results/Discussion. |

## Result narrative and evidence order

1. **Data curation and mixture balance.** Establish why functional-region sampling is a modeling variable, then use the early specialist and upstream/CDS experiment to motivate explicit balancing.
2. **Hyperparameter transfer.** Present the failed manual scaling attempt only as the motivation for a controlled transfer recipe, then report the transfer validation.
3. **Scaling and downstream behavior.** Separate the loss-scaling result from the biological VEP result; explain the Mendelian missense exception as a limitation of zero-shot readout, not a failure of representation scaling.
4. **Continued-training mixture experiments.** Describe the three lineages and their endpoints. Preserve the chronology because the arrival of projected data explains the staged m5.1 design, while avoiding a causal claim about curriculum order.
5. **Frozen final comparison.** Report model-family differences, statistical uncertainty, training compute, and as-deployed throughput together. Separate the frozen figures from the evolving leaderboard.

## Methods coverage required by the claim ledger

- Dataset construction and immutable revisions for CDS, upstream, downstream, ncRNA, enhancer, Mendelian, and SGE data.
- Coordinate, strand, sequence extraction, repeat weighting, train/validation, and contamination controls. Genomic coordinates must remain 0-based and half-open inside MarinDNA, with conversions only at external tool boundaries.
- Qwen3-derived decoder-only architecture, nucleotide tokenization, BOS convention, 255-bp examples, model geometry, parameter counts, and objective.
- Reference hyperparameter sweep, Vizier objective/search space, AdamH/Adam settings, transfer equations, batch/token-horizon transformations, and WSD schedule.
- Parameter-scaling and continued-training designs, checkpoint selection, data-mixture weights, token exposures, FLOP accounting, and hardware.
- Mendelian and SGE benchmark construction and splits; LLR and linear-probe protocols; baseline model revisions and score mappings.
- AUPRC aggregation, uncertainty, statistical tests, multiple-comparison treatment if applicable, and the exact definition of “statistically competitive.”
- Throughput workload, native context lengths, forward/reverse-complement passes, embedding output, batch size, accelerator, software, warmup, repetitions, and summary statistic.

## Link, citation, and footnote audit

The mechanical baseline contains 30 footnotes and 107 direct external links in the section sources. These are an audit baseline, not targets.

- Scholarly support moves to references.bib and author-year Typst citations.
- Essential definitions, methods, and comparison limitations move into main prose, captions, or Methods.
- Secondary diagnostics and extended technical detail move to Supplementary Information.
- GitHub issues remain in a compact provenance table and do not serve as the sole source of a scientific claim.
- Repository, model, dataset, leaderboard, and interactive links are concentrated in the availability and provenance sections, with immutable revisions wherever possible.
- Main-text footnotes survive only when they are clearer than every inline, Methods, or supplementary alternative.

## Unresolved decisions before the first complete paper draft

- [ ] Confirm authorship and order.
- [x] Confirm the exact statistical statement supporting “competitive with Evo 2 40B,” including the test, estimate, uncertainty, and frozen values.
- [ ] Record immutable public revisions for m5.1, retained baselines, and every figure input.
- [ ] Approve the proposed main-versus-supplement dispositions in figure-inventory.md before redrawing or composing panels.
- [ ] Decide whether the title should foreground the final comparison or use a recipe-centered formulation; either title must respect the statistical result.
- [ ] Decide whether the early objective comparison is sufficiently consequential to retain in the main Introduction or belongs only in provenance/supplementary material.
- [ ] Resolve contribution, acknowledgement, funding/compute, and competing-interest statements before public posting.
