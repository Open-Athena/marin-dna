= Discussion
<discussion>
MarinDNA shows that a conventional decoder-only Transformer can become a strong genomic sequence model when data construction and optimization are treated as primary modeling choices.
The final 1B model establishes the attainable performance and efficiency, while the mixture and hyperparameter-transfer experiments explain how that result was reached.

The data-mixture results indicate that proportional sampling is not a neutral default when genomic region datasets differ substantially in size.
In the upstream/CDS experiment, proportional pooling behaved similarly to CDS-only training, whereas equal sampling retained useful performance across both region types.
The later m5.1 lineage extended this principle by introducing alignment-projected ncRNA and enhancer data and finished with the strongest macro-average endpoint among the three frozen mixture lineages.
This result supports explicit control of region exposure, but it does not establish that staged exposure is intrinsically better than de novo five-region training because the lineages are not a controlled curriculum ablation.

The hyperparameter experiments provide complementary evidence that optimization settings can be transferred across model size, batch size, and token horizon in this regime.
The transferred learning rate was the best observed tested value at each of three target scales, and the resulting parameter ladder trained stably through 4B parameters.
The transfer rule nevertheless inherits assumptions from a text-model recipe, including its token-horizon exponent and model-geometry heuristic, and the present experiments do not establish that those choices are globally optimal for DNA.

On the frozen Mendelian evaluation, the final MarinDNA model is statistically competitive with Evo 2 40B rather than demonstrably superior to it.
Point estimates differ by consequence class and readout: MarinDNA closes much of the distal-variant gap, while Evo 2 retains advantages on several coding and splicing subsets.
GPN-Star and AlphaGenome remain stronger in the broader zero-shot comparison, but they use alignment-derived information and functional-genomics supervision, respectively, and therefore bound rather than directly match the alignment-free setting studied here.

The compute and throughput comparisons require similar care.
The training-FLOP ratio uses reported total training compute, whereas the throughput measurement compares steady-state scoring on the same GH200 at each model's native context length and includes forward- and reverse-complement passes plus embedding output.
It therefore measures the cost of deploying the evaluated models for this workload, not architecture-only efficiency at matched context length or matched tokens processed.

Several limitations constrain the biological interpretation.
The models use a short 255-base context chosen for functional-element evaluation, so these experiments do not test long-range regulatory interactions or genome-scale sequence generation.
Human variant-effect data provide the strongest available evaluation, but performance in humans does not by itself establish transfer to non-model organisms, which is an important motivation for alignment-free models.
The weighted validation loss is computed on fixed human sequences related to the training distribution rather than a clean phylogenetically independent holdout, so it is useful for controlled comparisons but not an unbiased estimate of genomic generalization.
Finally, the zero-shot Mendelian missense regression with scale shows that likelihood-based readouts can diverge from information retained in frozen representations.

Neither scaling nor optimization appears exhausted: linear-probe performance continued to improve through the largest evaluated model, and the loss-scaling fit showed no clear saturation over the tested range.
