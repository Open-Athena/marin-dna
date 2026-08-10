#import "../template.typ": paper-figure, supplementary-figure-ref

= Results
<results>
== Early mixture experiments
<early-mixture-experiments>
We first trained a 1.7B upstream-region specialist, trying to replicate the success of GPN-Promoter (#link("https://github.com/Open-Athena/marin-dna/issues/21")[experiment \#21]).#footnote[The upstream and CDS datasets used in these early experiments were earlier versions of those summarized in #link(<fig-training-datasets>)[Fig. 2]: broadly comparable, but not identical. These models also used a 512-bp context without a BOS token, roughly twice the 255-bp context adopted for the later recipe.] Although we used reasonable defaults rather than the systematic hyperparameter-transfer recipe developed later, performance was broadly comparable to Evo 2 40B. We saw a similar pattern when training a CDS specialist (#link("https://github.com/Open-Athena/marin-dna/issues/27")[experiment \#27]). Overall, however, GPN-Star remained stronger.

#paper-figure(
  "figures/promoter_cds_specialists.svg",
  id: "fig-upstream-cds-specialists",
  alt: "Five independently scaled panels comparing upstream and CDS specialists with Evo 2 40B and GPN-Star on region-matched Mendelian variant classes",
  caption: [*Region-matched zero-shot Mendelian VEP: specialists versus generalists.* Error bars denote SE.],
)
After testing upstream and CDS specialists independently, the next experiment asked whether one model could retain both capabilities (#link("https://github.com/Open-Athena/marin-dna/issues/13")[experiment \#13]). Sampling in proportion to dataset size---10% upstream and 90% CDS---is the naive default when the two datasets are simply pooled without reweighting. In practice, it behaved similarly to CDS-only training. Equal 50/50 upstream/CDS sampling produced balanced performance across both regions. This made explicit mixture control a central axis of investigation. Even the 50/50 mixture may not be optimal: regions can differ both in size and in the density of learnable biological signal.

#paper-figure(
  "figures/upstream_cds_balance.svg",
  id: "fig-upstream-cds-balance",
  alt: "Promoter and missense VEP AUPRC (%) trajectories for upstream-only, balanced, proportional, and CDS-only training mixtures",
  caption: [*Upstream/CDS training-mixture comparison.* Zero-shot results; the right panel is the unweighted mean of the promoter and missense AUPRC values.],
)
== Hyperparameter transfer
<hyperparameter-transfer>
Our first attempt to scale manually lowered the learning rate as the model grew from 0.6B to 1.7B and 4B parameters (#link("https://github.com/Open-Athena/marin-dna/issues/57")[experiment \#57]). The larger models did not improve over the 0.6B model, and the early 4B run became unstable. That failure made systematic hyperparameter transfer a prerequisite: without a trustworthy optimization recipe, a model-size comparison would confound scale with tuning quality.

The annotation-derived DNA pool available at the time contained \~85B tokens. #link(<fig-annotation-derived-training-pool>)[Fig. 7] shows its proportional CDS, upstream, and downstream composition.

#paper-figure(
  "figures/annotation_derived_training_pool.svg",
  id: "fig-annotation-derived-training-pool",
  alt: "Approximately 85 billion annotation-derived DNA tokens in a proportional CDS, upstream, and downstream mixture",
  caption: [*Available annotation-derived training pool.*],
)
That pool is large by genomics standards but small relative to modern accelerator-era training corpora. This makes the project data-constrained in principle even though our practical constraints are messier. We train on preemptible Google TPU Research Cloud resources, do not have consistent access to slices much larger than roughly 32 H100s worth of peak FLOPs, and want the recipe to remain reproducible at academic compute scale. O(100B) tokens therefore lands in an awkward middle ground where compute-constrained methods are still relevant, even though modest epoching is possible and likely breaks their assumptions at some unknown rate.

We started with hyperparameter transfer for that reason. If a proven data-constrained transfer framework existed, we would use it. We do not know of one, so we followed the same basic pattern as #link("https://openathena.ai/blog/delphi/")[Delphi], fitting a small reference sweep with the Vizier Bayesian optimization framework and then scaling the result using a Complete(d)-inspired AdamH heuristic.#footnote[Complete(d) refers to the compute-constrained hyperparameter-transfer framework described in #link("https://arxiv.org/abs/2512.22382")[Complete(d): Data-Optimizing Hyperparameter Transfer].]

#paper-figure(
  "figures/parameter_transfer_methodology_v1.svg",
  id: "fig-hyperparameter-transfer-methodology",
  alt: "Reference hyperparameter tuning and target hyperparameter transfer",
  caption: [*Hyperparameter calibration and transfer.*],
)
#link(<fig-hyperparameter-transfer-methodology>)[Fig. 8] separates reference calibration from target application. Two heuristics sit behind that workflow and are fixed before tuning: an inherited rule maps hidden width D to model geometry,#footnote[The DNA experiment inherits its geometry rule from Marin's text-model scaling heuristic rather than fitting it on DNA. For hidden width D, it sets `layers = round(D/(55 + 4·log₂D))`, `MLP width = 4D`, and both attention-head and KV-head counts to `D/128`\; the sweep widths are selected manually. See the #link("https://github.com/marin-community/marin/blob/a638849fa837f924aaac66ff3d0c1f581dfdd49e/experiments/scaling_law_sweeps/completed_adamh.py#L133-L140")[commit-pinned geometry rule] and #link("https://github.com/marin-community/marin/blob/a638849fa837f924aaac66ff3d0c1f581dfdd49e/experiments/scaling_law_sweeps/completed_adamh.py#L211-L244")[model builder].] and a second rule maps reference optimizer settings to a new batch size B and token horizon T. The reference sweep tunes initialization scale, the two learning rates, β₁, β₂, ε, gradient clipping, and z-loss. Here, tuning a learning rate means tuning its peak value under a fixed fractional schedule; the target run reuses that schedule shape, so warmup and decay scale with the run length rather than keeping fixed absolute step counts.#footnote[Both learning rates use 10% linear warmup, remain at their peak through 80% of training, and then decay linearly to zero over the final 20%. These are fractions of the target run's total steps. See the #link("https://github.com/marin-community/marin/blob/a638849fa837f924aaac66ff3d0c1f581dfdd49e/experiments/dna/exp109_bolinas_scaling_sweep.py#L269-L290")[experiment configuration] and #link("https://github.com/marin-community/marin/blob/a638849fa837f924aaac66ff3d0c1f581dfdd49e/lib/levanter/src/levanter/optim/config.py#L283-L376")[scheduler implementation].] At the target, the two learning rates, β₂, and ε are transformed with B and T, while initialization scale, β₁, gradient clipping, and z-loss are reused unchanged.#footnote[Relative to the reference batch and token horizon (B₀, T₀), the fixed heuristic uses `AdamH LR ∝ √(B/B₀)·(T₀/T)^0.3`, `Adam LR ∝ √(r/r₀)`, `ε ∝ √(r₀/r)`, and `β₂ = clip(β₂,₀^(B/B₀))`, where `r/r₀ = (B·T₀)/(B₀·T)` and configured bounds still apply. The 0.3 exponent is inherited from Marin's text recipe rather than fitted on the DNA sweep; the Complete(d) paper proposes a 0.5 token-horizon exponent, while a later #link("https://github.com/marin-community/marin/issues/4225")[Marin text sweep] estimated \~0.28. See the #link("https://github.com/marin-community/marin/blob/a638849fa837f924aaac66ff3d0c1f581dfdd49e/experiments/scaling_law_sweeps/completed_adamh.py#L162-L209")[commit-pinned implementation].]

The reference sweep used \~25M-parameter models trained for 2.5B tokens with a 16k-token batch, or roughly 4e17 FLOPs per run. We then validated the transferred hyperparameters across 255M--1B-parameter models, with 4x as many tokens, 1/4x the batch size, and roughly 170x the FLOPs per run. The first test was whether the learning-rate prediction survived that regime. #link(<fig-learning-rate-transfer>)[Fig. 9] shows that the transferred prediction lands exactly on the best observed learning-rate setting at all three validation scales, outperforming both the unchanged reference optimum and every other target-scale sweep setting; the less sensitive optimizer hyperparameters are shown separately in #link(<fig-adam-transfer>)[Fig. 10].

#paper-figure(
  "figures/figure1_lr_transfer.svg",
  id: "fig-learning-rate-transfer",
  alt: "Learning-rate transfer across model scales",
  caption: [*Learning-rate transfer across model scales.* Results across the 255M, 476M, and 1B validation scales. The control run type indicates final loss from the optimal configuration found in the initial smaller-scale reference sweep. The predicted, optimal LR results in a better loss than both this control and all other configurations at the same scale (sweep run type), for all model sizes.],
)
#paper-figure(
  "figures/figure2_beta2_epsilon_transfer.svg",
  id: "fig-adam-transfer",
  alt: "Adam beta2 and epsilon transfer across model scales",
  caption: [*Adam β₂ and ε transfer.* Results across the same scales as @fig-learning-rate-transfer.],
)
That validation is a fairly unforgiving test. If the transferred learning rate were merely close by accident, it would be surprising for it to land correctly across all three validation scales, but the prediction remains well centered at each one. For DNA, that is a pretty cool result. Prior biology foundation-model work has used μP-style transfer, but we are not aware of a DNA result showing that a more inclusive framework like Complete(d) works across token horizon and batch size, which are the axes we keep leaning on later in ad hoc runs across epochs. The same is mostly true for the other optimizer hyperparameters too, although Adam β₂ shows some signs of being a bit aggressive at the largest scale. #supplementary-figure-ref(<fig-region-hyperparameter-transfer>) makes the same point across CDS, upstream, and downstream sequence, with no qualitative difference in transfer behavior across region types. That gives us enough confidence that the following parameter-scaling runs are at least close to optimally configured.

== Parameter scaling
<parameter-scaling>
Before asking whether better validation loss#footnote[Here, "validation loss" is best understood as a training-loss-like monitoring statistic computed on a fixed set of human training sequences, rather than a conventional estimate on held-out data. We have not yet found a satisfactory way to construct clean held-out genomic splits: genomes are phylogenetically correlated, and identifying orthologous non-coding regions by sequence alignment is difficult. See #link("https://github.com/Open-Athena/marin-dna/issues/8")[issue \#8] for split experiments and a broader discussion of why raw perplexity may not reliably track VEP performance. The meaning of lowercase differs between training and validation: during training it marks repetitive bases, whereas in these validation sets it marks non-conserved bases; in both cases lowercase positions receive 1% of the standard loss weight. In validation, this acts as a heuristic that emphasizes conserved sequence, but we did not independently design or validate it as a model-selection objective. Because this statistic informed the Vizier reference sweep and the choice of the 1B model, those decisions may not be optimal under a different validation objective. We therefore use validation loss descriptively for like-for-like comparisons and trend analysis, not as an unbiased estimate of performance on unseen sequence; the biological conclusions rely primarily on downstream VEP evaluation.] translates into better VEP performance, we first needed to check whether validation loss scaled the way it should. The parameter sweep uses the same training recipe at each model size, with all hyperparameters set by the transfer heuristic above, and then asks whether the resulting losses fit a Kaplan-style scaling law well (they do).#footnote[This follows the empirical scaling-law setup from #link("https://arxiv.org/abs/2001.08361")[Kaplan et al.], where model loss is fit as a predictable function of model size, data, and compute.] Despite this being a simple experiment conceptually, actually getting there took months --- fitting the hyperparameter transfer heuristic, running the validation experiments, and training the 4B model, which alone took about three weeks to finish. The final sweep spans 8 model sizes from 46M to 4B parameters, each trained on \~84B tokens, for \~4.3e21 FLOPs across the sweep. That puts it on par with canonical scaling-law studies in language modeling, e.g.~its \~2.1e21 FLOP 4B run matches the compute Hugging Face used at that exact model scale in their data-constrained scaling work.#footnote[See Fig. 4 of #link("https://arxiv.org/abs/2305.16264")[Muennighoff et al.], "Scaling Data-Constrained Language Models" (NeurIPS 2023).]

#paper-figure(
  "figures/figure4_loss_scaling.svg",
  id: "fig-loss-scaling",
  alt: "Loss scaling across model sizes with Kaplan power-law fits",
  caption: [*Loss scaling across model sizes.* Training and validation loss for eight models from 46M to 4B parameters, each trained on \~84B tokens; the inset shows a Kaplan power-law fit to final validation loss.],
)
The result is about as tidy as we could hope for. Training is stable at every scale, and both training and validation loss decrease monotonically and predictably, as shown in @fig-loss-scaling. We use WSD learning-rate schedules with 10% warmup and 20% decay, which causes the visible drop in both losses over the final 20% of tokens. Most importantly, the sweep gives a high-quality Kaplan scaling-law fit (R2=0.999), which makes the next question much better posed. Does lower validation loss actually correlate with better downstream VEP performance?

== Downstream performance
<downstream-performance>
The final sweep shows a mostly consistent relationship between parameter count and downstream VEP performance. When zero-shot LLR and linear probes are evaluated on identical variants, performance improves with scale for most variant types. The clearest exception is Mendelian missense: zero-shot LLR peaks at 128M parameters and then deteriorates, even as linear-probe performance continues to improve.#footnote[Non-monotonic likelihood-based zero-shot variant-effect performance with increasing model scale has previously been observed in other settings. #link("https://proceedings.iclr.cc/paper_files/paper/2025/hash/62cf81a87f367758cebabce08e8d40d8-Abstract-Conference.html")[Gordon et al.] report that ESM-2 performance on protein deep-mutational-scanning benchmarks degrades beyond an intermediate model size and show that performance depends on the likelihood assigned to the wild-type sequence. #link("https://proceedings.neurips.cc/paper_files/paper/2025/hash/bdb30687f1c2255c29b11b0b45204ebe-Abstract-Conference.html")[Pugh et al.] similarly report plateaus or regressions for larger protein language models under standard likelihood-based scoring. These studies concern protein models and different evaluation regimes; we do not know whether our Mendelian missense result has the same cause.] This is not a general failure on missense variants---the SGE missense benchmark improves with scale under both scoring protocols.

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
This divergence is not unique to MarinDNA. On the same Mendelian missense benchmark, Evo 2 also shows improving linear-probe performance alongside declining zero-shot LLR performance as model size increases. For now, this should be interpreted as a recurring pattern for Mendelian missense across these two model families---not as evidence that zero-shot readouts generally deteriorate with scale. It also cautions against using zero-shot LLR alone to judge whether scaling has improved the learned representations for this task.

#paper-figure(
  "figures/figure6b_marin_evo2_missense.svg",
  id: "fig-missense-readout-scaling",
  alt: "Missense VEP performance across MarinDNA and Evo 2 model scales, comparing zero-shot LLR and linear probes",
  caption: [*Missense readouts across model scale.* MarinDNA and Evo 2 are compared using zero-shot LLR and a linear probe on identical Mendelian variants. Performance is measured as chromosome-weighted AUPRC; error bars denote SE.],
)
== Later mixture experiments
<later-mixture-experiments>
At this point we move away from theoretically grounded, compute-constrained methods. The later experiments still rely on the transfer heuristics above, since we need learning rates and other hyperparameters for runs with very different token horizons. But the actual optimization problem becomes much more ad hoc --- we start changing mixture constituents, epoch them freely, and see whether in-flight changes can compensate for observed performance gaps.

We standardized these experiments on a 1B-parameter model.#footnote[At the time, 1B had reached a good level of zero-shot performance under our then-current evaluation. We had not established it as the optimal model size. The later linear-probe results change this judgment most: they continued to improve with scale even where zero-shot Mendelian missense did not, making a larger model a more compelling choice in retrospect.]

Our starting point was a uniform mixture of CDS, upstream, and downstream sequence. These were the three region datasets we could initially construct consistently across many species from standard genome annotations. The scaling study had instead sampled these regions in proportion to dataset size. Echoing the earlier mixture experiments, that recipe produced good CDS performance but left a large gap on the less abundant upstream tasks. We therefore switched to uniform weighting so that each functional region received meaningful exposure.

We use the internal labels m5.1, m1.3, and m3.3 for the three model-mixture lineages compared below. In this naming scheme, m denotes a mixture lineage, the leading number identifies the mixture strategy, and the suffix identifies a continuation within that lineage. m5.1 is the staged lineage: we first trained its 1B model for \~104B tokens on the uniform three-region mixture. Alignment projection then made it possible to turn human ncRNA and enhancer annotations into comparable multi-species training datasets. Once those data became available, we added them to form a uniform five-region mixture and continued training the same model for another \~62B tokens. We compare this staged history with two controls, m1.3 and m3.3, trained on five-region mixtures from the beginning.

#paper-figure(
  "figures/continued_training_data_exposures.svg",
  id: "fig-five-region-lineage",
  alt: "Training-data exposure histories for m5.1, m1.3, and m3.3 through a shared 166-billion-token horizon",
  caption: [*Training-data exposure histories.* Three recipes are compared. m5.1 trains for approximately 104B tokens on a uniform three-region mixture before adding ncRNA and enhancer data for approximately 62B tokens. The de novo m1.3 and m3.3 controls keep fixed five-region mixtures over the same displayed token horizon; m3.3 gives upstream sequence 25% weight and each other region 18.75%.],
)
For m5.1, the first evaluated checkpoint after adding ncRNA and enhancer data shows gains in variant-effect performance in the corresponding ncRNA and distal subsets. Across all eight subsets, m5.1 ultimately finishes with the highest macro-average AUPRC under both zero-shot LLR and the linear probe, although the linear-probe trajectories are visibly noisier.#footnote[The curves shown here use a separate probe trained within each variant subset. In a #link("https://github.com/Open-Athena/marin-dna/issues/369#issuecomment-4936655473")[separate 255M analysis on one held-out chromosome], training one probe across all subsets improved the AUPRC point estimate on several data-starved subsets, including ncRNA and distal, while hurting stronger or more specialized subsets. This makes limited labeled data one plausible contributor to the noise, but that analysis did not directly test the checkpoint-to-checkpoint variability in these 1B lineage curves.] Its advantage is broad but not universal: the lineages trained on five regions from the beginning retain stronger endpoints for some distal and ncRNA subsets. m5.1's strong endpoint raises the possibility that exposure order matters: learning first from the three-region mixture and introducing ncRNA and enhancer data later may be more effective than training on all five regions from the beginning, though this remains uncertain and requires further investigation.

The corresponding linear-probe trajectories appear in #supplementary-figure-ref(<fig-mixture-lineage-probe>).

#paper-figure(
  "figures/figure16_offline_lineage_llr_prototype.svg",
  id: "fig-mixture-lineage-trajectories",
  alt: "Nine-panel zero-shot Mendelian pooled AUPRC (%) trajectories with error bars along each mixture lineage",
  caption: [*Zero-shot mixture-lineage trajectories.* Mendelian AUPRC versus training tokens for three model-mixture lineages. Error bars denote SE.],
)
== Leaderboard scores
<leaderboard-scores>
The result of the previous mixture experiments is m5.1, a 1B GPT-style model evaluated alongside other models on our #link("https://openathena.ai/marin-dna/leaderboards/mendelian")[live Mendelian VEP leaderboard], where we continue to add experimental runs and baselines. In the zero-shot snapshot shown here, m5.1 comes out slightly ahead of Evo 2 40B. Its advantage is considerably larger under linear probing.

m5.1 was trained on 166B tokens (\~1.1e21 FLOPs), compared with 9.3T tokens (\~2.25e24 FLOPs) for Evo 2 40B. At their native context lengths on the same GH200, m5.1 scores one million variants in about 41 minutes, compared with roughly 66 days for Evo 2 40B---a roughly 2,330× throughput advantage.#footnote[This benchmark measures steady-state scoring with forward and reverse-complement passes and embeddings enabled. m5.1 uses a 256-token context, while Evo 2 40B uses an 8,192-token context, so this measures as-deployed throughput rather than same-context or per-token efficiency. See #link("https://github.com/Open-Athena/marin-dna/issues/354")[issue \#354] for the full methodology and results.]

Most notably, m5.1 closes Evo 2's main gap on distal variants, outperforming Evo 2 40B there under both readouts. The improvement is not uniform, however: Evo 2 40B remains ahead on splicing under both readouts, as well as on promoter and synonymous variants in the zero-shot evaluation and missense variants under linear probing.

The #link("https://openathena.ai/marin-dna/leaderboards/mendelian")[current leaderboard] also suggests there is considerable headroom: although m5.1 leads on Macro Avg among MarinDNA models, another MarinDNA run has a higher point estimate in seven of the eight displayed subsets, with m5.1 leading only on ncRNA. Several of these winners are region specialists, suggesting that further mixture refinement could recover more of their complementary strengths.

On the broader zero-shot leaderboard, m5.1 remains slightly behind AlphaGenome and substantially behind GPN-Star.#footnote[@fig-mendelian-leaderboard uses LLR for MarinDNA and Evo 2, calibrated LLR (cLLR) for GPN-Star, and the maximum L2 REF/ALT prediction score for AlphaGenome.] These are different model families: AlphaGenome learns from functional-genomics supervision, while GPN-Star uses whole-genome alignments. Further improvements to the alignment-free recipe may narrow the gap to GPN-Star, but it is not clear that they will eliminate it. Some of the remaining difference may reflect information that an alignment-free model cannot recover from unaligned sequence alone (#link("https://github.com/Open-Athena/marin-dna/issues/397")[research question \#397]).

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
