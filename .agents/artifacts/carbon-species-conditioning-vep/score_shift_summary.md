# Carbon prompt-conditioning comparison

All summaries exclude the 40 mature-miRNA variants and use 16,100 variants in 1,610 complete match groups.

Score-level summaries rescale each raw prompt-mean LLR to a common 1,365 DNA-target-token denominator.
The multipliers are 1365/1365 for untagged, 1373/1365 for correct mammalian, and 1370/1365 for far-wrong fungal.
This removes the deterministic prompt-length scale difference; within-arm AUPRC and pairwise Pearson correlations are unchanged.

## Three retained approaches

Macro AUPRC averages the eight retained consequence subsets.

| approach | macro AUPRC | 95% CI | positive mean score | negative mean score | positive − negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untagged | 0.358200 | [0.338822, 0.387681] | 0.015919 | 0.002444 | 0.013475 |
| Correct mammalian | 0.355182 | [0.333180, 0.385529] | 0.015396 | 0.002421 | 0.012976 |
| Far-wrong fungal | 0.358748 | [0.337214, 0.388132] | 0.015626 | 0.002451 | 0.013175 |

## Conditioned-minus-untagged score shifts

`delta_score` is the DNA-token-normalized conditioned score minus the same variant's normalized untagged score.
The label-separation shift is the mean positive delta minus the mean delta across the nine matched negatives.
Positive label-separation shifts move pathogenic positives upward relative to their matched negatives.
Spread ratios compare the standard deviation of positive deltas with negative deltas.
Intervals are 95% match-group bootstrap intervals from 1,000 seeded draws.

| comparison | positive mean delta | negative mean delta | label-separation shift | 95% CI | positive SD / negative SD | 95% CI (log2 ratio) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Correct mammalian − untagged | -0.000522 | -0.000023 | -0.000499 | [-0.000742, -0.000259] | 2.179 | [0.821, 1.388] |
| Far-wrong fungal − untagged | -0.000293 | 0.000007 | -0.000300 | [-0.000449, -0.000144] | 1.507 | [0.287, 0.888] |
