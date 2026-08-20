# Carbon prompt-conditioning score shifts

`delta_score` is the conditioned score minus the same variant's untagged score.
The label-separation shift is the mean positive delta minus the mean delta across the nine matched negatives.
Positive label-separation shifts move pathogenic positives upward relative to their matched negatives.
Spread ratios compare the standard deviation of positive deltas with negative deltas.
Intervals are 95% match-group bootstrap intervals from 1,000 seeded draws.

## All development variants

| condition | positive mean delta | negative mean delta | label-separation shift | 95% CI | positive SD / negative SD | 95% CI (log2 ratio) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Correct mammalian | -0.000609 | -0.000038 | -0.000570 | [-0.000814, -0.000327] | 2.189 | [0.819, 1.413] |
| Far-wrong fungal | -0.000346 | -0.000002 | -0.000344 | [-0.000492, -0.000182] | 1.514 | [0.300, 0.898] |
