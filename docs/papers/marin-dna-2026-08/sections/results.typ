#import "../template.typ": paper-figure, supplementary-figure-ref

= Results
<results>
== Functional-region sampling
<early-mixture-experiments>
Equal upstream/CDS sampling retained performance across both region types, whereas proportional 10/90 sampling behaved similarly to CDS-only training.
Across the study, training examples came from annotation-derived coding, upstream, and downstream regions and from alignment-projected non-coding RNA and candidate cis-regulatory-element annotations. @fig-training-datasets summarizes the provenance and scale of these datasets; individual experiments used the versions and mixtures specified below and in the supplementary provenance table.

#paper-figure(
  "figures/data_provenance_training_datasets.svg",
  id: "fig-training-datasets",
  alt: "Token counts for annotation-derived CDS, upstream, and downstream datasets and alignment-projected enhancer and ncRNA datasets",
  caption: [*Training-dataset provenance and scale.*],
)
We first trained a 1.7B upstream-region specialist following the region-specialist premise of GPN-Promoter (#link("https://github.com/Open-Athena/marin-dna/issues/21")[experiment \#21]).#footnote[The upstream and CDS datasets used in these early experiments were earlier versions of those summarized in @fig-training-datasets: broadly comparable, but not identical. These models also used a 512-bp context without a BOS token, roughly twice the 255-bp context adopted for the later recipe.]
The upstream specialist and a corresponding CDS specialist (#link("https://github.com/Open-Athena/marin-dna/issues/27")[experiment \#27]) produced point estimates similar to Evo 2 40B on their matched consequence classes, although GPN-Star remained stronger.
Both specialists preceded the systematic hyperparameter-transfer recipe developed below.

#paper-figure(
  "figures/promoter_cds_specialists.svg",
  id: "fig-upstream-cds-specialists",
  alt: "Five independently scaled panels comparing upstream and CDS specialists with Evo 2 40B and GPN-Star on region-matched Mendelian variant classes",
  caption: [*Region-matched zero-shot Mendelian VEP: specialists versus generalists.* Error bars denote SE.],
)
We next compared proportional and equal sampling in one model (#link("https://github.com/Open-Athena/marin-dna/issues/13")[experiment \#13]).
Pooling the datasets without reweighting gave 10% upstream and 90% CDS examples and behaved similarly to CDS-only training.
Equal 50/50 sampling retained useful performance on both regions.
The experiment did not test enough mixture weights to establish 50/50 as optimal; region datasets differ in both size and density of learnable biological signal.

#paper-figure(
  "figures/upstream_cds_balance.svg",
  id: "fig-upstream-cds-balance",
  alt: "Promoter and missense VEP AUPRC (%) trajectories for upstream-only, balanced, proportional, and CDS-only training mixtures",
  caption: [*Upstream/CDS training-mixture comparison.* Zero-shot results; the right panel is the unweighted mean of the promoter and missense AUPRC values.],
)
== Hyperparameter transfer
<hyperparameter-transfer>
Our first attempt to scale manually lowered the learning rate as the model grew from 0.6B to 1.7B and 4B parameters (#link("https://github.com/Open-Athena/marin-dna/issues/57")[experiment \#57]). The larger models did not improve over the 0.6B model, and the early 4B run became unstable. That failure made systematic hyperparameter transfer a prerequisite: without a trustworthy optimization recipe, a model-size comparison would confound scale with tuning quality.

The annotation-derived DNA pool available at the time contained \~85B tokens. @fig-annotation-derived-training-pool shows its proportional CDS, upstream, and downstream composition.

#paper-figure(
  "figures/annotation_derived_training_pool.svg",
  id: "fig-annotation-derived-training-pool",
  alt: "Approximately 85 billion annotation-derived DNA tokens in a proportional CDS, upstream, and downstream mixture",
  caption: [*Available annotation-derived training pool.*],
)
The available pool was small relative to modern accelerator-era training corpora, and we did not have consistent access to preemptible Google TPU Research Cloud slices beyond roughly 32 H100s of peak FLOPs.
We therefore targeted O(100B)-token runs that could be reproduced at academic compute scale.
This regime permits repeated exposure to the data and may violate compute-constrained transfer assumptions at an unknown rate.

We did not identify an established transfer framework for this data-constrained regime.
Following the pattern used in #link("https://openathena.ai/blog/delphi/")[Delphi], we fit a small reference sweep with the Vizier Bayesian optimization framework and scaled the result using a Complete(d)-inspired AdamH heuristic.#footnote[Complete(d) refers to the compute-constrained hyperparameter-transfer framework described in #link("https://arxiv.org/abs/2512.22382")[Complete(d): Data-Optimizing Hyperparameter Transfer].]

#paper-figure(
  "figures/parameter_transfer_methodology_v1.svg",
  id: "fig-hyperparameter-transfer-methodology",
  alt: "Reference hyperparameter tuning and target hyperparameter transfer",
  caption: [*Hyperparameter calibration and transfer.*],
)
@fig-hyperparameter-transfer-methodology separates reference calibration from target application.
A fixed rule maps hidden width D to model geometry, and a second rule maps reference optimizer settings to a new batch size B and token horizon T.
The reference sweep tunes initialization scale, the two peak learning rates, β₁, β₂, ε, gradient clipping, and z-loss.
At the target, the two learning rates, β₂, and ε are transformed with B and T, while initialization scale, β₁, gradient clipping, and z-loss are reused unchanged.
The transfer equations, schedule, and implementation references are given in @methods-hyperparameter-transfer.

The reference sweep used \~25M-parameter models trained for 2.5B tokens with a 16k-token batch, or roughly 4e17 FLOPs per run.
We validated the transferred hyperparameters at 255M, 476M, and 1B parameters using four times as many tokens, one quarter of the batch size, and roughly 170 times the FLOPs per run.
The transferred learning rate gave the lowest final validation loss among tested settings at all three target scales (@fig-learning-rate-transfer); the less sensitive optimizer hyperparameters are shown separately in @fig-adam-transfer.

#paper-figure(
  "figures/figure1_lr_transfer.svg",
  id: "fig-learning-rate-transfer",
  alt: "Learning-rate transfer across model scales",
  caption: [*Learning-rate transfer across model scales.* Results across the 255M, 476M, and 1B validation scales. The control denotes the best configuration found in the smaller reference sweep. At each target scale, the transferred learning rate gives lower final validation loss than the control and the other tested learning rates.],
)
#paper-figure(
  "figures/figure2_beta2_epsilon_transfer.svg",
  id: "fig-adam-transfer",
  alt: "Adam beta2 and epsilon transfer across model scales",
  caption: [*Adam β₂ and ε transfer.* Results across the same scales as @fig-learning-rate-transfer.],
)
Prior biology foundation-model work has used μP-style transfer, but we are unaware of a DNA study evaluating optimizer transfer across both token horizon and batch size.
Adam β₂ showed a possible overcorrection at 1B parameters; the other transferred settings showed no qualitative failure across the tested scales.
Region-stratified results were also qualitatively consistent across CDS, upstream, and downstream sequence (#supplementary-figure-ref(<fig-region-hyperparameter-transfer>)).
These observations support using the transferred recipe for the controlled parameter-scaling comparison.
The tested grids do not establish global optimality or validate the heuristic outside the evaluated scales and training regimes.

== Parameter scaling
<parameter-scaling>
Training and validation loss decreased monotonically across eight models from 46M to 4B parameters, each trained on \~84B tokens.
The final validation losses followed a Kaplan-style power law with R² = 0.999 over the tested range.#footnote[This follows the empirical scaling-law setup from #link("https://arxiv.org/abs/2001.08361")[Kaplan et al.], where model loss is fit as a predictable function of model size, data, and compute.]
The sweep used the same training recipe at every model size, with hyperparameters set by the transfer heuristic above.
It consumed \~4.3e21 FLOPs in total; the 4B run consumed \~2.1e21 FLOPs and took approximately three weeks.#footnote[The 4B run matches the compute used at the same model scale in the data-constrained scaling study by #link("https://arxiv.org/abs/2305.16264")[Muennighoff et al.] (NeurIPS 2023).]
The validation statistic and its limitations are defined in @methods-scaling.

#paper-figure(
  "figures/figure4_loss_scaling.svg",
  id: "fig-loss-scaling",
  alt: "Loss scaling across model sizes with Kaplan power-law fits",
  caption: [*Loss scaling across model sizes.* Training and validation loss for eight models from 46M to 4B parameters, each trained on \~84B tokens; the inset shows a Kaplan power-law fit to final validation loss.],
)
Training remained stable at every scale (@fig-loss-scaling).
The WSD learning-rate schedule uses 10% warmup and 20% decay, producing the visible loss drop over the final 20% of tokens.
The scaling fit describes the tested parameter range; downstream evaluation tests whether lower validation loss corresponds to better VEP performance.

== Downstream performance
<downstream-performance>
When zero-shot LLR and linear probes are evaluated on identical variants, performance improves with scale for most variant types.
Mendelian missense is the clearest exception: zero-shot LLR peaks at 128M parameters and then deteriorates, even as linear-probe performance continues to improve.#footnote[Non-monotonic likelihood-based zero-shot variant-effect performance with increasing model scale has previously been observed in other settings. #link("https://proceedings.iclr.cc/paper_files/paper/2025/hash/62cf81a87f367758cebabce08e8d40d8-Abstract-Conference.html")[Gordon et al.] report that ESM-2 performance on protein deep-mutational-scanning benchmarks degrades beyond an intermediate model size and show that performance depends on the likelihood assigned to the wild-type sequence. #link("https://proceedings.neurips.cc/paper_files/paper/2025/hash/bdb30687f1c2255c29b11b0b45204ebe-Abstract-Conference.html")[Pugh et al.] similarly report plateaus or regressions for larger protein language models under standard likelihood-based scoring. These studies concern protein models and different evaluation regimes; we do not know whether our Mendelian missense result has the same cause.]
SGE missense improves with scale under both scoring protocols, so the regression is specific to zero-shot Mendelian missense in these experiments.

#paper-figure(
  "figures/figure5_params_vs_vep_auprc.svg",
  id: "fig-parameters-vs-vep",
  alt: "VEP performance across model parameters for Mendelian and SGE consequences, comparing zero-shot LLR and linear probes",
  caption: [*VEP performance across model scale.* Zero-shot LLR and a linear probe are compared on identical variants. Performance is measured as chromosome-weighted AUPRC; error bars denote SE.],
)
Plotting the same results against matched-region validation log-likelihood gives the same picture. For all other combinations of variant type and scoring protocol, better validation log-likelihood is associated with better downstream performance. Zero-shot Mendelian missense again points in the opposite direction: performance declines even as validation log-likelihood improves.

#paper-figure(
  "figures/figure6_loss_vs_vep_auprc.svg",
  id: "fig-loss-vs-vep",
  alt: "VEP performance versus matched-region validation log-likelihood for Mendelian and SGE consequences, comparing zero-shot LLR and linear probes",
  caption: [*VEP performance versus validation log-likelihood.* Matched-region LL (shown as −loss) across the eight parameter-scaling endpoints. Performance is measured as chromosome-weighted AUPRC. Lines are least-squares fits, and _r_ denotes Pearson correlation; error bars denote SE.],
)
Evo 2 shows the same readout divergence on the Mendelian missense benchmark: linear-probe performance improves with model size while zero-shot LLR performance declines.
The recurrence is confined to Mendelian missense in these two model families and does not show that zero-shot readouts generally deteriorate with scale.
Zero-shot LLR alone can therefore miss improvements present in frozen representations for this task.

#paper-figure(
  "figures/figure6b_marin_evo2_missense.svg",
  id: "fig-missense-readout-scaling",
  alt: "Missense VEP performance across MarinDNA and Evo 2 model scales, comparing zero-shot LLR and linear probes",
  caption: [*Missense readouts across model scale.* MarinDNA and Evo 2 are compared using zero-shot LLR and a linear probe on identical Mendelian variants. Performance is measured as chromosome-weighted AUPRC; error bars denote SE.],
)
== Five-region mixture lineages
<later-mixture-experiments>
The later mixture experiments retained the transferred optimizer settings but were exploratory.
We changed mixture constituents during training, allowed repeated exposure to the data, and selected continuations in response to observed performance gaps.
The resulting lineages do not constitute a controlled curriculum ablation.

We standardized these experiments on a 1B-parameter model.#footnote[At the time, 1B had reached a good level of zero-shot performance under our then-current evaluation. We had not established it as the optimal model size. The later linear-probe results change this judgment most: they continued to improve with scale even where zero-shot Mendelian missense did not, making a larger model a more compelling choice in retrospect.]

Our starting point was a uniform mixture of CDS, upstream, and downstream sequence. These were the three region datasets we could initially construct consistently across many species from standard genome annotations. The scaling study had instead sampled these regions in proportion to dataset size. Echoing the earlier mixture experiments, that recipe produced good CDS performance but left a large gap on the less abundant upstream tasks. We therefore switched to uniform weighting so that each functional region received meaningful exposure.

We compare a staged lineage (m5.1) with two controls trained on five-region mixtures from the beginning (m1.3 and m3.3).
m5.1 first trained for \~104B tokens on the uniform three-region mixture.
After alignment projection made multi-species ncRNA and enhancer datasets available, we added them as two equally weighted regions and continued training for another \~62B tokens.

#paper-figure(
  "figures/continued_training_data_exposures.svg",
  id: "fig-five-region-lineage",
  alt: "Training-data exposure histories for m5.1, m1.3, and m3.3 through a shared 166-billion-token horizon",
  caption: [*Training-data exposure histories.* Three recipes are compared. m5.1 trains for approximately 104B tokens on a uniform three-region mixture before adding ncRNA and enhancer data for approximately 62B tokens. The de novo m1.3 and m3.3 controls keep fixed five-region mixtures over the same displayed token horizon; m3.3 gives upstream sequence 25% weight and each other region 18.75%.],
)
The first evaluated m5.1 checkpoint after adding ncRNA and enhancer data improved on the corresponding ncRNA and distal subsets.
At the final checkpoint, m5.1 had the highest macro-average AUPRC among the three lineages under both zero-shot LLR and the linear probe, although the linear-probe trajectories were noisier.#footnote[The curves shown here use a separate probe trained within each variant subset. In a #link("https://github.com/Open-Athena/marin-dna/issues/369#issuecomment-4936655473")[separate 255M analysis on one held-out chromosome], training one probe across all subsets improved the AUPRC point estimate on several data-starved subsets, including ncRNA and distal, while hurting stronger or more specialized subsets. This makes limited labeled data one plausible contributor to the noise, but that analysis did not directly test the checkpoint-to-checkpoint variability in these 1B lineage curves.]
The controls retained stronger endpoints for some distal and ncRNA subsets.
The endpoint pattern motivates a controlled test of exposure order but does not establish a benefit from staged training.

The corresponding linear-probe trajectories appear in #supplementary-figure-ref(<fig-mixture-lineage-probe>).

#paper-figure(
  "figures/figure16_offline_lineage_llr_prototype.svg",
  id: "fig-mixture-lineage-trajectories",
  alt: "Nine-panel zero-shot Mendelian pooled AUPRC (%) trajectories with error bars along each mixture lineage",
  caption: [*Zero-shot mixture-lineage trajectories.* Mendelian AUPRC versus training tokens for three model-mixture lineages. Error bars denote SE.],
)
== Frozen comparison with external models
<leaderboard-scores>
In the frozen zero-shot snapshot, the unweighted mean across the eight consequence subsets meeting the project-wide 30-positive minimum is 39.49% AUPRC for m5.1 and 38.24% for Evo 2 40B.
The paired difference is 1.25 percentage points (bootstrap SE 1.63; 95% CI −1.93 to 4.46; two-sided bootstrap p = 0.440).
This analysis contains 16,100 variants in 1,610 1:9 matched groups; the mature-miRNA subset, with four groups, is excluded from the macro-average by the same minimum applied throughout the evaluation pipeline.
The confidence interval includes zero, so we describe m5.1 as statistically competitive with Evo 2 40B.
Its point-estimate advantage is considerably larger under linear probing, for which this paired zero-shot comparison does not apply.

m5.1's three-stage lineage consumed 166.01B tokens and 1.135e21 recorded FLOPs.
Evo 2 40B used 9.3T tokens and an estimated 2.25e24 FLOPs, giving a ratio of 1,982 that we report as approximately 1,980×.
At their native context lengths on the same GH200, the optimized m5.1 configuration extrapolates to one million variants in about 41 minutes, compared with roughly 66 days for Evo 2 40B.
The corresponding throughput ratio is approximately 2,330×.#footnote[This is an as-deployed workload comparison with forward- and reverse-complement passes and embeddings enabled. The m5.1 point uses delayed-FP8 throughput at 256 tokens and frozen BF16 AUPRC, while Evo 2 uses its original stack at 8,192 tokens. The full benchmark and quality-gate methodology is given in @methods-efficiency.]

#paper-figure(
  "figures/headline_cost_performance.svg",
  id: "fig-cost-performance",
  alt: "Zero-shot Mendelian VEP macro-average AUPRC versus variants scored per hour for MarinDNA 1B and Evo 2 1B, 7B, and 40B",
  caption: [*Variant-effect performance versus as-deployed inference throughput.* Zero-shot Mendelian macro-average AUPRC versus variants scored per hour on one GH200. Each model is evaluated at its native context length; throughput includes forward- and reverse-complement scoring and embedding output. The MarinDNA point combines the frozen BF16 AUPRC with the optimized delayed-FP8 throughput; the FP8 configuration passed the paired zero-shot quality gate reported in Methods.],
)

m5.1 outperforms Evo 2 40B on distal variants under both readouts.
Evo 2 40B remains ahead on splicing under both readouts, on promoter and synonymous variants in the zero-shot evaluation, and on missense variants under linear probing.

Within the frozen leaderboard snapshot, no single MarinDNA model has the highest point estimate across all consequences.
At least one other MarinDNA run has a higher point estimate than m5.1 in seven of eight displayed subsets, and several of those runs are region specialists.
This pattern motivates further mixture experiments aimed at combining their complementary strengths.

In the broader frozen zero-shot comparison, m5.1's macro-average point estimate remains below AlphaGenome and GPN-Star.#footnote[@fig-mendelian-leaderboard uses LLR for MarinDNA and Evo 2, calibrated LLR (cLLR) for GPN-Star, and the maximum L2 REF/ALT prediction score for AlphaGenome.]
AlphaGenome learns from functional-genomics supervision, while GPN-Star uses whole-genome alignments.
These comparisons bound the alignment-free result without isolating the effects of architecture, training data, or supervision (#link("https://github.com/Open-Athena/marin-dna/issues/397")[research question \#397]).

#paper-figure(
  "figures/figure11_leaderboard_heatmap__mendelian_llr.svg",
  id: "fig-mendelian-leaderboard",
  alt: "Mendelian VEP benchmark zero-shot AUPRC (%) heatmap across six headline models",
  caption: [*Zero-shot Mendelian leaderboard.*],
)
#paper-figure(
  "figures/figure11_leaderboard_heatmap__mendelian_probe.svg",
  id: "fig-mendelian-leaderboard-probe",
  alt: "Mendelian VEP benchmark linear-probe AUPRC (%) heatmap across four overlapping models",
  caption: [*Linear-probe Mendelian leaderboard.* AUPRC is a sample-size-weighted average across chromosomes and is not directly comparable with @fig-mendelian-leaderboard. GPN-Star and AlphaGenome are absent because no compatible probe results are available here, not because of their performance.],
)
