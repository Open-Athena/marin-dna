#import "../template.typ": paper-figure

= Methods
<methods>

== Study design and frozen analysis snapshot
<methods-study-design>
This manuscript reports the model, evaluation, and figure snapshot used for the published MarinDNA article rather than a continuously updated leaderboard.
The mechanical manuscript baseline and its 20 SVG assets were fixed at MarinDNA commit #link("https://github.com/Open-Athena/marin-dna/tree/3b608d39b41c2330636ec647dbb25d26b0895187/docs/papers/marin-dna-2026-08")[3b608d39b41c2330636ec647dbb25d26b0895187], whose content and figures were converted from project commit d8c4803cbbbffafb24890cd0c75134d78368d55c.
No new training run or evaluation dataset was introduced for the editorial conversion.
The live leaderboard is treated as a separate, evolving resource.

== Training data
<methods-training-data>
Training examples were drawn from functional regions across multiple animal genomes.
The initial datasets contained coding sequences (CDS), sequence upstream of annotated genes, and sequence downstream of CDS ends.
The downstream category denotes the 256 bases immediately after an annotated CDS end rather than annotated 3′ UTR intervals.
Later training mixtures added functional ncRNA regions and ENCODE V4 non-promoter candidate cis-regulatory elements, with human annotations projected to other species by sequence alignment.
All genomic coordinates inside MarinDNA use 0-based, half-open intervals, with conversion to external coordinate conventions only at tool boundaries.
The later recipe represents each example with 255 DNA bases and a beginning-of-sequence token, giving a 256-token model input.
Soft-masked repetitive bases receive 1% of the standard training-loss weight.
Training-data revisions and per-region token counts for every retained experiment will be enumerated in the supplementary provenance table.

== Model and training objective
<methods-model>
MarinDNA uses the Qwen3 decoder-only Transformer architecture with a causal next-token objective and nucleotide-level DNA tokenization.
The architecture is intentionally conventional: the experiments vary training data, optimization, and parameter scale without introducing a genomics-specific sequence operator.
The final m5.1 model has 1,120,772,224 parameters and a 255-base context.
It was first trained for approximately 104B tokens on a uniform mixture of CDS, upstream, and downstream data, then continued for approximately 62B tokens after ncRNA and enhancer data were added to form a uniform five-region mixture.
The frozen internal checkpoint is mix-v0.9-p1B-i24-exp135-m5.1-step-59158 at GCS path gs://marin-us-east5/checkpoints/dna-bolinas-mix-v0.9-p1B-i24-exp135-zoonomia-m5.1-bef41e/hf/step-59158.
The immutable public release is recorded in @availability.

== Hyperparameter calibration and transfer
<methods-hyperparameter-transfer>
A Bayesian reference sweep using Vizier trained approximately 25M-parameter models for 2.5B tokens with a 16k-token batch.
The sweep tuned initialization scale, AdamH and Adam peak learning rates, β₁, β₂, ε, gradient clipping, and z-loss against the weighted validation-loss statistic described below.
A fixed geometry rule maps hidden width to the number of layers, MLP width, attention heads, and key-value heads.
A second fixed rule transfers the reference optimizer settings to a target batch size and token horizon.
Relative to reference batch size B₀ and horizon T₀, the AdamH learning rate is multiplied by √(B/B₀)(T₀/T)^0.3.
The Adam learning rate is multiplied by √(r/r₀), ε by √(r₀/r), and β₂ is exponentiated by B/B₀ subject to configured bounds, where r/r₀ = (B T₀)/(B₀ T).
Initialization scale, β₁, gradient clipping, and z-loss are reused without transformation.
Both learning rates use linear warmup over the first 10% of steps, remain at their peak through 80% of training, and decay linearly to zero over the final 20%.
Target-scale validation used 255M-, 476M-, and 1B-parameter models with four times the token horizon and one quarter of the reference batch size.
“Best observed” refers to the lowest final validation loss among the tested target-scale settings and does not imply a global optimum.

== Parameter-scaling experiment and validation loss
<methods-scaling>
The parameter ladder contains models of approximately 46M, 76M, 128M, 255M, 476M, 1B, 2B, and 4B parameters.
Each model was trained for approximately 84.77B tokens with the transferred optimization recipe.
The final validation losses were fit with a Kaplan-style power law as a function of parameter count.
Training and validation losses weight soft-masked or non-conserved lowercase bases at 1% of the standard token weight.
The validation statistic is computed on fixed human sequences related to the training distribution and is used for controlled comparisons and model selection.
It is not interpreted as an unbiased estimate on a phylogenetically independent held-out genome set.

== Variant-effect datasets
<methods-evaluation-data>
The Mendelian benchmark compares pathogenic variants with non-rare gnomAD controls matched within consequence class.
Relative to the TraitGym design, the minimum control allele frequency is 0.1%, the consequence set includes missense and splicing variants, and matching includes potential confounders such as transcription-start-site distance and exon distance where applicable.
The frozen benchmark revision is #link("https://huggingface.co/datasets/marin-dna/evals_mendelian_traits/tree/4aed58e50c5dea0b878a665007af2ef9e5108e9f")[4aed58e50c5dea0b878a665007af2ef9e5108e9f].
The saturation-genome-editing benchmark is the v3 labeled build at #link("https://huggingface.co/datasets/marin-dna/evals_sge/tree/225d3d1ea32a4af547891b13c33b5e92a5aae849")[225d3d1ea32a4af547891b13c33b5e92a5aae849].
It retains variants labeled abnormal or normal by ClinGen/ExCALIBR-calibrated assay thresholds and groups them into missense and splicing subsets.
Variant contexts are extracted from the Ensembl release 115 GRCh38 soft-masked primary assembly.
The evaluation pipeline uses the development split for model iteration; the chromosome-disjoint test split remains reserved for the final locked evaluation.

#paper-figure(
  "figures/eval_datasets.svg",
  id: "fig-evaluation-datasets",
  alt: "Clinical Mendelian and experimental SGE benchmarks, including labels and subset counts.",
  caption: [*Variant-effect prediction benchmarks.* Counts denote the frozen analysis snapshot described in the text.],
)
The two benchmarks and their analysis subsets are summarized in @fig-evaluation-datasets.

== Zero-shot likelihood scoring
<methods-zero-shot>
For each allele, the causal model scores the variant-centered sequence on the forward strand and its reverse complement.
The raw log-likelihood ratio is defined as log P(ALT) − log P(REF).
For Mendelian and SGE evaluation, deleteriousness is oriented as the negative of the mean forward- and reverse-complement log-likelihood ratio, so larger scores indicate a larger likelihood penalty for the alternate allele.
Forward and reverse-complement scores are retained separately, but their mean is the default MarinDNA and Evo 2 score used in the headline comparison.
GPN-Star is represented by its calibrated likelihood-ratio score, whereas AlphaGenome is represented by its maximum L2 reference/alternate prediction score; these scores are not interchangeable measurements of model likelihood.

== Frozen-embedding linear probes
<methods-probes>
The same forward and reverse-complement passes used for likelihood scoring also produce last-layer hidden states for the reference and alternate alleles.
Hidden states are mean-pooled across the entire DNA window with special tokens excluded, accumulated and strand-averaged in float32, and stored in float16.
For directional Mendelian and SGE labels, the probe feature concatenates the reference embedding with the alternate-minus-reference embedding after upcasting to float32.
A separate StandardScaler plus L2-regularized logistic-regression pipeline is fit within each consequence subset.
Predictions use leave-one-chromosome-out cross-validation, with the regularization strength retuned inside each outer fold by chromosome-grouped cross-validation over a fixed 17-point grid from 1e-12 to 1e4.
Probe and zero-shot metrics are calculated on identical variant rows.

#paper-figure(
  "figures/eval_apparatus.svg",
  id: "fig-evaluation-readouts",
  alt: "Reference and alternate sequences scored using likelihoods or frozen-model embeddings.",
  caption: [*Variant-effect prediction scoring approaches.* The zero-shot likelihood score and frozen-embedding linear probe use the same reference and alternate sequence inputs.],
)
@fig-evaluation-readouts contrasts the two readouts applied to every frozen checkpoint.

== Metrics and uncertainty
<methods-statistics>
Performance is summarized by area under the precision–recall curve (AUPRC).
For matched Mendelian zero-shot analyses, AUPRC is computed within consequence subset and uncertainty is estimated by resampling matched groups with replacement, thereby preserving the one-to-many matched structure.
For probe analyses, AUPRC is first calculated per chromosome and then weighted by the number of contributing variants; uncertainty is estimated by a chromosome-cluster bootstrap.
SGE AUPRC is computed separately within each MaveDB accession and consequence subset before macro-averaging because scores are not comparable across studies.
Unless a caption states otherwise, error bars denote ±1 bootstrap standard error.
The headline comparison uses the eight consequence subsets with at least 30 positive matched groups, the project-wide minimum for inclusion in the Mendelian macro-average.
It contains 16,100 variants in 1,610 groups, each comprising one positive and nine matched controls; the ninth subset, mature miRNA, has four groups and is excluded from the macro-average.
The MarinDNA and Evo 2 score artifacts contain the same 16,140 uniquely keyed variants and agree exactly on label, subset, and match-group assignment.
For the paired comparison, 10,000 bootstrap iterations are generated with random seed 0.
Within each consequence subset, matched groups are sampled with replacement; the same sampled rows are used for both models, AUPRC is recomputed for each, and the eight within-subset differences are averaged with equal weight.
The comparison reports the observed macro-AUPRC difference, its bootstrap standard error, the 2.5th and 97.5th percentiles of the paired difference distribution, and a two-sided bootstrap p-value calculated as twice the smaller empirical probability that the paired difference is at or below zero or at or above zero.
Comparative language in the title, abstract, Results, and Discussion follows this paired uncertainty rather than the ordering of point estimates alone.

== Training compute and inference throughput
<methods-efficiency>
MarinDNA training compute is reconstructed from the recorded W&B `throughput/total_gflops` fields at the boundaries of all three segments inherited by m5.1.
The lineage contains 166,010,552,320 tokens and 1.13498e21 FLOPs.
Evo 2 40B reports 9.3T tokens and an estimated 2.25e24 FLOPs; #link("https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full#T1")[the source table] states that the estimate does not account for mixed precision or pretraining context length @brixi2026evo2.
The ratio of the reported values is 1,982, rounded to approximately 1,980-fold.

Inference throughput was measured on one 96-GB NVIDIA GH200 at USD 2.29 per hour with forward- and reverse-complement passes and pooled reference/alternate embedding output enabled.
The optimized m5.1 measurement used PyTorch 2.10.0+cu128, Transformer Engine 2.13 delayed E4M3 FP8, fused MLP and QKV projections, `torch.compile` in default mode, batch size 128, four data-loader workers, and three synchronized repetitions on 5,800 pre-materialized variants.
It achieved 1,467,776 variants per hour, equivalent to 2,452.69 seconds per million variants.
The plotted m5.1 AUPRC remains the frozen BF16 result, whereas its throughput coordinate uses this optimized FP8 configuration.
The promoted FP8 model passed a separate 10,000-draw paired zero-shot quality gate: BF16 minus FP8 macro AUPRC was 0.000355 (95% interval −0.004271 to 0.005190).
A frozen-BF16-probe compatibility check was inconclusive, so the optimized rate supports zero-shot scoring and the stated embedding-output contract but not an unconditional drop-in claim for existing BF16-trained probe classifiers.
The Evo 2 40B measurement used the NVIDIA PyTorch 25.04 container plus the Evo 2 inference package, batch size 1, and a late converged batch rate after model loading, warm-up, and tokenization; its archived rate is rounded to 630 variants per hour.
Both measurements are steady-state and exclude sequence materialization.
The resulting ratio is approximately 2,330-fold, but its precision is limited by the rounded Evo 2 rate.
The models use their native context lengths, 256 tokens for m5.1 and 8,192 tokens for Evo 2 40B, so this is an as-deployed workload measurement rather than a same-context, same-batch, or per-token architecture benchmark.
