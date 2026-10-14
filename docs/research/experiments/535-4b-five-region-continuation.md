# 4B uniform five-region continuation

> [!NOTE]
> **TL;DR:** The completed 4B continuation improved Mendelian development-set scores over earlier checkpoints, but its final zero-shot macro AUPRC was numerically close to the 1B m5.1 reference and its linear-probe score remained lower; this single lineage does not isolate model scale or mixture weighting.

## Findings

Continuing the proportional-three-region 4B scaling model on a uniform five-region mixture produced a usable broader-region checkpoint after 160,000 additional updates.
Final zero-shot macro AUPRC increased from 0.3485 at continuation checkpoint 275573 to 0.3894 at 375573; the 1B m5.1 reference scored 0.3945.
Frozen-embedding probe macro AUPRC increased from 0.4196 at checkpoint 255573 to 0.4420 at 375573, below m5.1's 0.4779.
These are descriptive within-protocol comparisons, not tests of equivalence or statistical superiority.

The improvements were uneven across regions.
Relative to checkpoint 275573, final zero-shot AUPRC increased most for 3′ UTR (0.1787 to 0.3462) and Distal (0.3087 to 0.3815), while Missense decreased from 0.4187 to 0.4020.
The final probe improved over checkpoint 255573 in six of eight subsets, but Promoter and 3′ UTR decreased.

## Evidence

Validation language-modeling loss is unavailable: the run deliberately omitted online validation, and no offline validation-loss evaluation is reported here.
Training-loss telemetry is not a substitute for validation loss.

The parent was the completed 4B scaling-ladder model at step 215573, with a proportional CDS/upstream/downstream training history.
Native restoration preserved model, optimizer, RNG, and step state.
The continuation sampled CDS, upstream, downstream, ncRNA exon, and non-promoter cCRE equally, retained the global batch of 1,536 sequences of 256 tokens, and added 62.91B tokens.
Including the parent's approximately 84.77B tokens, lifetime exposure was approximately 147.68B tokens.
A fresh warmup–stable–decay cycle used 16,000 warmup, 112,000 plateau, and 32,000 cooldown updates, ending at step 375573.
The same run supplied all evaluated continuation checkpoints; the final checkpoint was the planned endpoint, not selected by downstream performance.
Zero-shot evaluations cover steps 235573, 255573, 275573, and 375573; probes cover 255573 and 375573.
The intermediate exports at 295573, 315573, 335573, and 355573 were not evaluated, so these comparisons do not establish a monotone trajectory.

Evaluation used the pinned Mendelian Traits development split (odd autosomes and chromosome X), 255 bp windows, and forward/reverse-complement averaging.
The reported frame excludes mature-miRNA match groups and contains eight supported subsets, 1,610 positives, and 16,100 variants.
Zero-shot scores use `minus_llr_avg` with matched-pair AUPRC and match-group bootstrap uncertainty.
Probes use frozen pooled embeddings with `[ref, alt − ref]` features, standardized L2 logistic regression, outer leave-one-chromosome-out predictions, and inner chromosome-grouped selection of regularization strength.
Probe AUPRC is chromosome-weighted, with chromosome-cluster bootstrap uncertainty.
The two rows below use different weighting and should only be compared across models within a row.

| Development macro AUPRC ± SE | Earlier continuation checkpoint | Final 375573 | 1B m5.1 |
| --- | ---: | ---: | ---: |
| Zero-shot, matched-pair | 0.3485 ± 0.0140 (275573) | 0.3894 ± 0.0164 | 0.3945 ± 0.0155 |
| Linear probe, chromosome-weighted | 0.4196 ± 0.0240 (255573) | 0.4420 ± 0.0277 | 0.4779 ± 0.0216 |

The [metric snapshot](https://github.com/Open-Athena/marin-dna/blob/e0beff4ddbfc43a1239716c45d9443c55f3b6bcb/experiments/exp535_4b_uniform_five_region/results/final-metrics.json) retains all supported subset values and source-file hashes.
The [execution logbook](https://github.com/Open-Athena/marin-dna/blob/e0beff4ddbfc43a1239716c45d9443c55f3b6bcb/.agents/logbooks/issue-535-4b-uniform-five-region.md) records canonical artifacts and reproduction from the [evaluation snapshot](https://github.com/Open-Athena/marin-dna/commit/ce278dca7ecee8834ab3d6db64d00b373c340959).

## Limitations

One continuation seed and no matched unchanged-mixture control leave additional training, mixture changes, and the fresh learning-rate cycle confounded.
Checkpoints 255573 and 275573 were on the peak-learning-rate plateau, whereas 375573 followed cooldown from 343573; within-run gains therefore confound additional exposure with learning-rate decay, whose contribution was not separately measured.
The 4B parent had a proportional three-region history, whereas m5.1's parent used a uniform three-region mixture; model size, training history, and exposure differ.
The comparison cannot establish a causal scaling benefit, optimal mixture weights, or a general advantage of five regions over three.
Reported SEs describe each score separately; no paired inter-checkpoint or inter-model bootstrap comparison was performed.
These repeatedly inspected development-set results do not establish held-out-test performance, and no complex-trait or SGE result is claimed for this continuation.

## Related questions

- [How to optimize pretraining data mixtures?](../questions/data-mixtures.md)
- [Which genomic regions to train on, and how to find them?](../questions/training-regions.md)

## Research record

- [Experiment #535](https://github.com/Open-Athena/marin-dna/issues/535)
