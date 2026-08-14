# Issue #459: soft VEP metrics below AUPRC resolution

## Decision

Keep AUPRC as the primary endpoint and report **plain unpaired Cohen's `d`**
beside it. For Mendelian and complex-trait evaluations, compute one `d` from
all positive and negative variants; ignore match groups. For SGE, compute
Cohen's `d` separately within each gene and report the unweighted gene macro
average, `mean(d_g)`.

Use conventional closed-form uncertainty only:

- `SE(d) = sqrt((n_pos + n_neg) / (n_pos * n_neg) + d^2 / (2 * (n_pos + n_neg - 2)))`;
- `SE(mean(d_g)) = sd(d_g) / sqrt(G)` across SGE genes, with a
  `t_(G - 1)` interval.

This deliberately treats variants as independent and does not use a bootstrap.
Cohen's `d` is dimensionless and invariant to positive score rescaling. It is
an exploratory effect-size companion, not a replacement for AUPRC.

Under the explicit no-group sensitivity analysis, variant pooled SMD detected
the home specialist earlier than row-bootstrap AUPRC on two of seven subsets,
at the same step on three, later on one, and neither detected synonymous. It
also followed the late missense reversal. The all-variant-SD gap and Student's
`t` produced identical arm rankings and detection times, so they add no signal.
Welch's unequal-variance `t` changed which subsets were early but not the
aggregate 2/3/1/1 count, and it continued rising through the missense AUPRC
decline.

Replacing exp232's contaminated cCRE arm with the independently trained
exp351-centered distal arm does not change the recommendation. On distal,
AUPRC and plain Cohen's `d` both first achieve persistent,
row-bootstrap-supported home-arm separation at step 3000 in the historical
detectability analysis. Across the resulting eight specialist subsets,
Cohen's `d` is earlier/same/later/neither on 2/4/1/1.

## Primary question: does Brier distinguish the home arm earlier?

No. Define detection at a stored checkpoint by the joint-bootstrap distribution
of

`home metric - best non-home metric`,

after orienting both AUPRC and Brier so higher is better. A step is supported
when that margin's pointwise 95% interval is above zero; the reported detection
time is the first of two consecutive supported synchronized checkpoints.

| Consequence subset | Home arm | AUPRC | `1 - calibrated Brier` |
| --- | --- | ---: | ---: |
| missense | `cds` | 1500 | 1500 |
| synonymous | `cds` | not detected | not detected |
| splicing | `cds` | 1000 | 1000 |
| 3′ UTR | `utr3` | 3500 | not detected |
| noncoding exon | `ncrna_exon` | 3500 | 4500 |
| 5′ UTR | `tss_region_and_utr5` | 1500 | 2000 |
| TSS proximal | `tss_region_and_utr5` | 1000 | 3000 |

Brier is earlier on **0/7** subsets. AUPRC is earlier on **4/7**, the metrics tie
on **2/7**, and neither detects synonymous. The bootstrap rank-first frequency
provides the per-step strength curve; it is not a p-value or posterior
probability. Brier uncertainty remains conditional on fixed out-of-fold
calibration fits.

## Alternatives under the same detectability test

No candidate robustly dominates AUPRC across the seven consequence subsets.

| Candidate | Earlier than AUPRC | Same step | AUPRC earlier | Neither |
| --- | ---: | ---: | ---: | ---: |
| Global mean gap | 2 | 2 | 2 | 1 |
| Group mean gap | 2 | 2 | 2 | 1 |
| Group SMD | 2 | 3 | 1 | 1 |
| Median / MAD | 0 | 1 | 5 | 1 |
| SoftWin | 3 | 2 | 1 | 1 |
| Calibrated log loss | 1 | 3 | 2 | 1 |
| Calibrated Brier | 0 | 2 | 4 | 1 |

SoftWin has the strongest apparent timing, detecting splicing at step 500,
3′ UTR at 1000, and noncoding exon at 3000 before AUPRC. The raw mean gaps also
lead on splicing and 3′ UTR. These candidates remain disqualified as standard
cross-model metrics because positive score rescaling changes their rankings.

Group SMD is the most interesting scale-invariant alternative for this narrow
home-arm question: it leads AUPRC by one recorded checkpoint on splicing
(500 versus 1000) and noncoding exon (3000 versus 3500), ties on three subsets,
loses on 3′ UTR, and detects neither synonymous. Calibrated log loss leads only
on 3′ UTR (2000 versus 3500), then ties on three and loses on two. This is not
enough consistency to replace AUPRC or establish a generally earlier proxy.

## No-match-group sensitivity

This sensitivity analysis discards `match_group`, resamples positive and
negative variants separately, and recomputes AUPRC from the same joint row
draws. The AUPRC timing can therefore differ from the primary cluster-bootstrap
analysis; most visibly, 3′ UTR moves from step 3500 to 4500.

| Consequence subset | Row-bootstrap AUPRC | Pooled SMD | Gap / all-variant SD | Student `t` | Welch `t` |
| --- | ---: | ---: | ---: | ---: | ---: |
| missense | 1500 | 1500 | 1500 | 1500 | 1500 |
| synonymous | not detected | not detected | not detected | not detected | not detected |
| splicing | 1000 | 500 | 500 | 500 | 500 |
| 3′ UTR | 4500 | 1500 | 1500 | 1500 | not detected |
| noncoding exon | 3500 | 4500 | 4500 | 4500 | 3000 |
| 5′ UTR | 1500 | 1500 | 1500 | 1500 | 1500 |
| TSS proximal | 1500 | 1500 | 1500 | 1500 | 1500 |

All four candidates are earlier/same/later/neither on **2/3/1/1** subsets.
That aggregate hides two useful distinctions:

- Pooled SMD, the all-variant-SD gap, and Student's `t` have identical arm
  rankings at all 63 synchronized subset/checkpoint cells and identical
  detection times. Student's `t` is pooled SMD multiplied by a fixed
  sample-size factor within each subset. The all-variant-SD gap is a monotone
  prevalence-dependent transform in this fixed-label comparison.
- Welch's `t` detects noncoding exon at step 3000 but never detects 3′ UTR. It
  also rises from 17.90 to 19.12 while missense AUPRC falls from 0.3280 to
  0.3095 between steps 4000 and 4999. Pooled SMD declines from 1.2126 to 1.1866
  over the same interval and therefore preserves this adversarial reversal.

Plain Cohen's `d` is the simplest useful statistic. It measures effect size
without turning sample size into signal. A Welch `t` p-value instead measures
evidence against equal class means and makes sample size part of the result.
For SGE, calculate `d` independently within each gene, then take the unweighted
mean across genes so large genes do not dominate. Use the conventional IID
closed-form `SE(d)` for a single gene and `sd(d_g) / sqrt(G)` for the
multi-gene macro average.

## Evidence

- The original scope was development split only: 48 existing exp232 score
  parquets, seven non-distal consequence subsets, five arms, and 1,000 joint
  `match_group` bootstrap draws. The replacement-distal extension added eight
  full offline exp351-centered predictions, then assessed six arms across the
  seven established specialist subsets plus distal. No held-out labels or
  metrics were accessed.
- All 336 exp232 AUPRC cells exactly reproduce the stored `minus_llr_avg`
  values.
- On the current 22-model `family: marin_dna` leaderboard macro-average,
  calibrated Brier has Spearman **0.9955** and Kendall **0.9654** against AUPRC,
  preserves all three AUPRC top-three models, and has four of 231 raw pairwise
  reversals. Calibrated log loss is second (Spearman 0.9864; seven reversals).
- Across the seven subset-level joint bootstraps, calibrated Brier has **zero
  confidence-supported reversals among 988 informative model pairs**. Calibrated
  log loss also has zero among 995; Brier wins the broader macro ranking and
  projection comparisons.
- Leave-one-experiment-out isotonic projection from Brier to macro AUPRC has
  MAE **0.00937** and Spearman **0.9898**, versus 0.01150 / 0.9729 for log loss
  and 0.03954 / 0.6919 for either raw mean gap.
- Positive score rescaling leaves Brier, log loss, SMD, and median/MAD model
  ranks unchanged. It changes the raw-gap leaderboard rank (mean Spearman
  0.8485 versus baseline) and also perturbs fixed-temperature SoftWin (0.9658).
  This disqualifies the raw gaps as cross-model metrics.
- On exp232 missense/CDS, AUPRC peaks at 0.3280 at step 4000 and falls to 0.3095
  at step 4999. Brier follows the degradation (0.07920 to 0.08010; lower is
  better), as does log loss weakly (0.28166 to 0.28200). The global mean gap,
  group SMD, and SoftWin continue improving through the AUPRC decline: the raw
  gap rises from 5.45 to 8.06. Those magnitude statistics hide the known late
  reversal.
- Point estimates sometimes identify a specialist earlier, but the robust
  joint home-versus-best-other result does not support the new metric. Brier is
  never earlier than AUPRC, and no metric sustains a supported synonymous home
  win. The requested evidence for an earlier *usable* specialist signal is
  therefore not established.

## Controls and limits

- FWD-only and FWD/RC-average rankings are close on the 22-model leaderboard
  (mean subset Spearman 0.953 for Brier and 0.973 for AUPRC), but materially less
  stable across only the five exp232 arms. The averaged protocol remains the
  primary readout.
- Sign reversal flips directional separation metrics. Brier and log loss retain
  their rankings because each grouped-CV calibrator can learn the reversed sign;
  they measure calibrated predictiveness, not a fixed biological direction.
- Within-group label permutation produces near-null metric values/rankings as
  expected, with sampling noise from the small five-arm and 22-model panels.
- Proper-score bootstrap intervals resample held-out per-row losses from fixed
  grouped-CV calibrators. They do not include calibrator-refit uncertainty.
- One current leaderboard model,
  `scaling-v0.5-h2944-p4B-step-215573`, disagrees with its stored AUPRC on all
  seven subsets (largest absolute difference 0.002643 on 3′UTR). The analysis
  retains and flags those rows rather than silently treating them as parity.

## Replacement distal assessment

The exp351-centered arm now has full offline Mendelian predictions at steps
500, 1500, 2000, 3000, 3500, 4000, 4500, and 4999. The augmented assessment
treats it as the distal home arm and compares it with all five uncontaminated
exp232 arms. Step 1000 is absent because no durable HF export existed; nothing
was interpolated.

| Metric | First persistent distal separation |
| --- | ---: |
| AUPRC | 3000 |
| Cohen's `d` | 3000 |

At step 3000, the distal home arm has Cohen's `d = 0.524` (conventional IID
95% interval 0.251 to 0.796), versus `d = 0.163` (-0.109 to 0.434) for the
strongest non-home arm. The historical row-bootstrap detectability rule does
not place Cohen's `d` earlier than AUPRC on distal.

Across all eight specialist subsets, plain Cohen's `d` is earlier/same/later/
neither than row-bootstrap AUPRC on 2/4/1/1. The conclusion is to report
Cohen's `d` beside AUPRC as the single scale-invariant effect-size diagnostic,
not to replace AUPRC or claim a uniformly earlier signal.

The result is still not a controlled training comparison: issue #351 reports
about 9.7 epochs for exp351-centered versus about 3.7 for its tiled counterpart.
Checkpoint step aligns optimizer progress, not distinct sequence exposure.

## Artifact index

- `exp232/point_metrics.parquet`: 48 cells × seven subsets × eight metrics.
- `exp232/pairwise_deltas.parquet`: joint arm bootstrap comparisons.
- `exp232/specialist_detectability.parquet`: per-step joint rank-first
  frequency and home-versus-best-non-home margin.
- `exp232/specialist_detection_timing.parquet`: first persistent supported
  separation under AUPRC and all seven candidate metrics.
- `exp232/metric_detection_comparison.parquet`: per-candidate earlier/tied/later
  counts relative to AUPRC.
- `exp232/ungrouped_point_metrics.parquet` and
  `ungrouped_pairwise_deltas.parquet`: AUPRC plus four no-group statistics with
  class-stratified variant-bootstrap intervals.
- `exp232/ungrouped_specialist_detectability.parquet`,
  `ungrouped_specialist_detection_timing.parquet`, and
  `ungrouped_metric_detection_comparison.parquet`: no-group home-arm evidence,
  first persistent detection, and timing counts relative to row-bootstrap
  AUPRC.
- `exp232/supported_specialist_wins.parquet`: persistent, interval-supported
  specialist timing.
- `exp232/plots/*.svg`: eight full-cross-arm trajectory panels.
- `exp232/plots/specialist_auprc_vs_brier.{svg,png}`: all-arm AUPRC and
  `1 - calibrated Brier` trajectories with the mapped home arm highlighted.
- `exp232/plots/specialist_detectability.{svg,png}` and
  `specialist_detection_timing.{svg,png}`: per-step evidence and direct
  earliest-detection comparison.
- `exp232/plots/specialist_metric_detectability_summary.{svg,png}`: all-metric
  timing heatmap and comparison counts.
- `exp232/plots/{variant_pooled_smd,variant_total_sd_gap,student_t,welch_t}.svg`
  and `specialist_ungrouped_metric_detectability_summary.{svg,png}`: no-group
  full-arm trajectories and timing comparison.
- `exp232/distributions/*.svg`: final-step POS/NEG and matched-group-difference
  ECDFs for all seven subsets.
- `leaderboard/`: 22-model ranking, controls, pairwise confidence comparisons,
  AUPRC parity, and leave-one-experiment-out projections.
- `augmented-exp232-exp351/`: the six-arm replacement-distal manifest, matched
  and no-group joint-bootstrap tables, exact AUPRC parity check,
  `cohen_d_closed_form.parquet`,
  `cohen_d_closed_form_win_probabilities.parquet`, and complete SVG/PNG
  summaries. The primary specialist plot shows AUPRC and Cohen's `d` for all
  six arms across all eight subsets. The companion win-percentage plot uses the
  independent-normal closed form for every home-versus-non-home Cohen's `d`
  comparison.
- `distal/`: exact aggregate trajectories, point-count metadata, and the
  earlier aggregate-only distal SVG and composite-panel timing table.
