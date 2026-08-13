# Issue #459: soft VEP metrics below AUPRC resolution

## Decision

Keep AUPRC as the official endpoint. Do not promote the raw global/group mean
gaps, group SMD, median/MAD separation, or fixed-temperature SoftWin into the
standard evaluation path.

Carry **grouped-CV calibrated Brier score** forward as the only candidate for a
second validation pass or an explicitly optional early-training diagnostic. It
is not ready to replace AUPRC or to drive stopping decisions: its apparent
earlier point-estimate wins usually disappear after requiring paired-bootstrap
support, and its intervals are conditional on fixed out-of-fold calibrators.

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

## Evidence

- Scope was development split only: 48 existing exp232 score parquets, seven
  non-distal consequence subsets, five arms, and 1,000 joint `match_group`
  bootstrap draws. No inference or held-out evaluation was run.
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

## Distal patch

No compatible exp326/exp351 per-variant score bundle exists, so no distal soft
metric or interval was fabricated. The aggregate-only patch preserves every
finite logged point, including duplicate resume records:

- exp232 cCRE baseline: 9 offline `evals_v2` points, final AUPRC 0.1268.
- exp326 no-exon and enhancer-only arms: 14 online `lm_eval` points each, final
  AUPRC 0.2990 and 0.2719.
- exp351 tiled and centered arms: 14 and 11 online points, final AUPRC 0.3082
  and 0.3663. The centered result remains confounded by 9.7 versus 3.7 epochs
  and both curves are still rising at step 4999.

The distal figure separates exp326 and exp351 comparisons and explicitly labels
the offline/online protocol difference.

Both composite eight-subset AUPRC panels first satisfy the two-recorded-evaluation
point-estimate rule at step **3000**. This is not a confidence-supported
all-subset result: distal has no per-variant bootstrap and synonymous has no
persistent bootstrap-supported specialist win under any metric, including
AUPRC.

## Artifact index

- `exp232/point_metrics.parquet`: 48 cells × seven subsets × eight metrics.
- `exp232/pairwise_deltas.parquet`: joint arm bootstrap comparisons.
- `exp232/specialist_detectability.parquet`: per-step joint rank-first
  frequency and home-versus-best-non-home margin.
- `exp232/specialist_detection_timing.parquet`: first persistent supported
  separation under AUPRC and all seven candidate metrics.
- `exp232/metric_detection_comparison.parquet`: per-candidate earlier/tied/later
  counts relative to AUPRC.
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
- `exp232/distributions/*.svg`: final-step POS/NEG and matched-group-difference
  ECDFs for all seven subsets.
- `leaderboard/`: 22-model ranking, controls, pairwise confidence comparisons,
  AUPRC parity, and leave-one-experiment-out projections.
- `distal/`: exact aggregate trajectories, point-count metadata, and the
  explicitly limited distal SVG and composite-panel timing table.
