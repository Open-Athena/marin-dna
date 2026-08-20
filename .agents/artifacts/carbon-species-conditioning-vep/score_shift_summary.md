# Carbon prompt-conditioning comparison

All summaries exclude the 40 mature-miRNA variants and use 16,100 variants in 1,610 complete match groups.

## Three retained approaches

Macro AUPRC averages the eight retained consequence subsets.

| approach | macro AUPRC | 95% CI | positive mean score | negative mean score | positive − negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untagged | 0.358200 | [0.338822, 0.387681] | 0.015919 | 0.002444 | 0.013475 |
| Correct mammalian | 0.355182 | [0.333180, 0.385529] | 0.015307 | 0.002407 | 0.012900 |
| Far-wrong fungal | 0.358748 | [0.337214, 0.388132] | 0.015569 | 0.002442 | 0.013127 |

## Conditioned-minus-untagged score shifts

`delta_score` is the conditioned score minus the same variant's untagged score.
The label-separation shift is the mean positive delta minus the mean delta across the nine matched negatives.
Positive label-separation shifts move pathogenic positives upward relative to their matched negatives.
Spread ratios compare the standard deviation of positive deltas with negative deltas.
Intervals are 95% match-group bootstrap intervals from 1,000 seeded draws.

| comparison | positive mean delta | negative mean delta | label-separation shift | 95% CI | positive SD / negative SD | 95% CI (log2 ratio) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Correct mammalian − untagged | -0.000612 | -0.000037 | -0.000575 | [-0.000819, -0.000335] | 2.191 | [0.829, 1.396] |
| Far-wrong fungal − untagged | -0.000350 | -0.000002 | -0.000348 | [-0.000500, -0.000190] | 1.516 | [0.292, 0.898] |
