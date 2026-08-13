= Discussion
<discussion>
The final 1B MarinDNA model reached 39.49% macro AUPRC on the frozen Mendelian benchmark, compared with 38.24% for Evo 2 40B, while using 1,982-fold fewer reported training FLOPs.
In native-context GH200 benchmarks, it scored variants at approximately 2,330 times the archived Evo 2 40B rate.
The supporting experiments show how region sampling changed performance across consequence classes and how optimizer settings transferred across model size, batch size, and token horizon.

In the upstream/CDS experiment, proportional 10/90 pooling behaved similarly to CDS-only training, whereas equal sampling retained useful performance across both region types.
The later m5.1 lineage introduced alignment-projected ncRNA and enhancer data and finished with the highest macro-average endpoint among the three frozen mixture lineages.
These results support explicit control of region exposure.
The mixture lineages were exploratory and do not establish that staged exposure is better than de novo five-region training.

The hyperparameter experiments provide complementary evidence that optimization settings can be transferred across model size, batch size, and token horizon in this regime.
The transferred learning rate was the best observed tested value at each of three target scales, and the resulting parameter ladder trained stably through 4B parameters.
The transfer rule nevertheless inherits assumptions from a text-model recipe, including its token-horizon exponent and model-geometry heuristic, and the present experiments do not establish that those choices are globally optimal for DNA.

On the frozen Mendelian evaluation, the final MarinDNA model's macro-AUPRC point estimate is 1.25 percentage points higher than Evo 2 40B, with a paired 95% bootstrap confidence interval from −1.93 to 4.46 percentage points.
The interval includes zero, so the data support statistical competitiveness without demonstrating superiority.
Point estimates differ by consequence class and readout: MarinDNA closes much of the distal-variant gap, while Evo 2 retains advantages on several coding and splicing subsets.
GPN-Star and AlphaGenome have higher macro-average point estimates in the broader zero-shot comparison.
They use alignment-derived information and functional-genomics supervision, respectively, so the comparison does not isolate architecture or training recipe.

The training-FLOP ratio uses MarinDNA's recorded lineage compute and Evo 2's published estimate, whose methodology explicitly omits mixed-precision and context-length adjustments.
The throughput measurement compares steady-state scoring on the same GH200 at each model's native context length and includes forward- and reverse-complement passes plus embedding output; MarinDNA uses an optimized delayed-FP8 implementation, while the Evo 2 denominator is rounded at source.
The throughput ratio measures the evaluated deployment configurations for this workload.
It does not estimate architecture-only efficiency at matched context length or matched tokens processed.
The FP8 configuration passed the zero-shot quality gate but was inconclusive as a drop-in source of embeddings for the existing BF16-trained probes, so the throughput result should not be generalized to probe compatibility.

The models use a short 255-base context chosen for functional-element evaluation, so these experiments do not test long-range regulatory interactions or genome-scale sequence generation.
Human variant-effect data provide the strongest available evaluation, but performance in humans does not by itself establish transfer to non-model organisms, which is an important motivation for alignment-free models.
The weighted validation loss is computed on fixed human sequences related to the training distribution and is useful for controlled comparisons.
It is not an unbiased estimate of genomic generalization on a phylogenetically independent holdout.
The zero-shot Mendelian missense regression with scale shows that likelihood-based readouts can diverge from information retained in frozen representations.

Linear-probe performance continued to improve through the 4B model, and the loss-scaling fit showed no clear saturation over the tested range.
The experiments leave open whether larger models or further mixture and optimization changes would improve the frozen benchmarks.
