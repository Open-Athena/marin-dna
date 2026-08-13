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
  version is less compelling. For missense, AUPRC, Brier, log loss, raw gaps,
  group SMD, and SoftWin all first sustain a bootstrap-supported specialist win
  at step 1500. For 3′UTR, raw gaps and SoftWin lead AUPRC (1000 versus 2000),
  but the raw metrics fail the cross-model robustness controls and Brier never
  sustains a supported two-checkpoint win. No metric, including AUPRC, sustains
  a supported synonymous specialist win. The requested evidence for an earlier
  *usable* specialist signal is therefore not established.

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

## Artifact index

- `exp232/point_metrics.parquet`: 48 cells × seven subsets × eight metrics.
- `exp232/pairwise_deltas.parquet`: joint arm bootstrap comparisons.
- `exp232/supported_specialist_wins.parquet`: persistent, interval-supported
  specialist timing.
- `exp232/plots/*.svg`: eight full-cross-arm trajectory panels.
- `exp232/distributions/*.svg`: final-step POS/NEG and matched-group-difference
  ECDFs for all seven subsets.
- `leaderboard/`: 22-model ranking, controls, pairwise confidence comparisons,
  AUPRC parity, and leave-one-experiment-out projections.
- `distal/`: exact aggregate trajectories, point-count metadata, and the
  explicitly limited distal SVG.
