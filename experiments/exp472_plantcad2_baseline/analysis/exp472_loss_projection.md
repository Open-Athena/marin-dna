# exp472 — loss trend fits and 20-epoch projection

Each of the 15 completed trials (the abandoned `lr0p001-wd1p6` is excluded) was fit
with a power law, `log10(loss) = a·log10(step) + b`, over the **pre-cooldown window
steps 64,916 → 164,928** — 10 eval points, ending at the last eval at the start of
cooldown (step 164,916 = 80% of 206,145, from `decay=0.2`). That fitted line is
extrapolated to **step 412,290**, i.e. 10 more epochs at 20,614.5 steps/epoch.

**Everything below is pre-cooldown.** The fit, the baseline, and the projection all sit
on the constant-LR part of the schedule, so the two loss columns are directly comparable
and `gain` is the improvement 10 more epochs would buy along that trend. Post-cooldown
evals appear in the top panel of the plot for context and are used nowhere else.

| # | run (lr / wd) | fit slope | R² | measured @ cooldown start | **projected @20 ep** | gain |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `lr0p0001-wd0p1` | -0.0390 | 0.993 | 1.0329 | **0.9955** | -0.0374 |
| 2 | `lr0p0001-wd0p2` | -0.0351 | 0.994 | 1.0365 | **1.0028** | -0.0337 |
| 3 | `lr0p0002-wd0p1` | -0.0298 | 0.993 | 1.0327 | **1.0041** | -0.0286 |
| 4 | `lr0p0002-wd0p2` | -0.0272 | 0.996 | 1.0415 | **1.0151** | -0.0263 |
| 5 | `lr0p0005-wd0p1` | -0.0224 | 0.996 | 1.0411 | **1.0194** | -0.0217 |
| 6 | `lr0p0001-wd0p8` | -0.0271 | 0.988 | 1.0570 | **1.0301** | -0.0270 |
| 7 | `lr0p0005-wd0p2` | -0.0189 | 0.995 | 1.0574 | **1.0388** | -0.0186 |
| 8 | `lr0p001-wd0p1` | -0.0183 | 0.994 | 1.0606 | **1.0423** | -0.0183 |
| 9 | `lr0p0002-wd0p8` | -0.0169 | 0.989 | 1.0740 | **1.0569** | -0.0171 |
| 10 | `lr0p0001-wd1p6` | -0.0184 | 0.989 | 1.0774 | **1.0586** | -0.0188 |
| 11 | `lr0p001-wd0p2` | -0.0136 | 0.993 | 1.0819 | **1.0682** | -0.0137 |
| 12 | `lr0p0002-wd1p6` | -0.0095 | 0.975 | 1.1055 | **1.0954** | -0.0100 |
| 13 | `lr0p0005-wd0p8` | -0.0092 | 0.967 | 1.1135 | **1.1039** | -0.0096 |
| 14 | `lr0p001-wd0p8` | -0.0034 | 0.914 | 1.1519 | **1.1483** | -0.0036 |
| 15 | `lr0p0005-wd1p6` | -0.0039 | 0.925 | 1.1539 | **1.1497** | -0.0042 |

## Which runs are best

- **`lr=1e-4/wd=0.1` leads** at 0.9955, ahead of `lr=1e-4/wd=0.2` (1.0028) and
  `lr=2e-4/wd=0.1` (1.0041). It has the steepest trend in the sweep (−0.0390) and the
  largest projected gain (−0.037).
- **Low LR and low WD compound.** The top five are exactly `lr ≤ 2e-4` with `wd ≤ 0.2`,
  plus `lr=5e-4/wd=0.1`.
- **Rank is set by slope, not by current standing.** At cooldown start, `lr=2e-4/wd=0.1`
  (1.0327) is marginally ahead of `lr=1e-4/wd=0.1` (1.0329), but the latter is descending
  ~30% faster and pulls clear over the next 10 epochs. Ordering by loss today gives a
  different, and for this question wrong, answer.
- **High weight decay at high LR is finished.** `lr=1e-3/wd=0.8` and `lr=5e-4/wd=1.6` are
  flat (slope ≈ −0.003, gain ≈ −0.004) with the weakest fits (R² 0.91–0.93). More epochs
  will not help them.

## Caveats

- A 2× extrapolation from a 10-point window: treat gaps below ~0.005 as noise. Ranks 2–3
  and 14–15 are within that.
- The fit assumes the constant-LR power law continues; it does not model a future cooldown,
  so a real 20-epoch run would finish below its projection by roughly the cooldown drop
  seen here (−0.04 to −0.12).
- `eval/plantcad/loss` spans only ~0.95–1.25 over the logged range, so the plot uses a
  linear y-axis with a log x-axis rather than a broken or multi-scale axis.

![loss projection](exp472_loss_projection.png)

Reproduce with `uv run --with matplotlib python project_loss.py`.
Source: W&B `eric-czech/marin`, key `eval/plantcad/loss`, 20 eval points per run.
