# exp472 — loss trend fits and 20-epoch projection

Each of the 15 completed trials (the abandoned `lr0p001-wd1p6` is excluded) was fit
with a power law, `log10(loss) = a·log10(step) + b`, over the **pre-cooldown window
steps 64,916 → 164,928** (10 eval points; the LR schedule's cooldown begins at
step 164,916 = 80% of 206,145). The line is extrapolated to **step 412,290**, i.e.
10 more epochs at 20,614.5 steps/epoch, and each run's own **measured cooldown gain**
(`final − trend(206,145)`, about −0.04 to −0.12) is added back, so the projection is
comparable to the measured 10-epoch numbers.

| # | run (lr / wd) | measured loss @10 ep | fit slope | R² | **projected loss @20 ep** |
|---:|---|---:|---:|---:|---:|
| 1 | `lr0p0001-wd0p2` | 0.9887 | -0.0351 | 0.994 | **0.9640** |
| 2 | `lr0p0002-wd0p2` | 0.9844 | -0.0272 | 0.996 | **0.9651** |
| 3 | `lr0p0005-wd0p1` | 0.9813 | -0.0224 | 0.996 | **0.9653** |
| 4 | `lr0p0002-wd0p1` | 0.9864 | -0.0298 | 0.993 | **0.9654** |
| 5 | `lr0p0001-wd0p1` | 0.9947 | -0.0390 | 0.993 | **0.9674** |
| 6 | `lr0p0001-wd0p8` | 0.9904 | -0.0271 | 0.988 | **0.9709** |
| 7 | `lr0p0005-wd0p2` | 0.9863 | -0.0189 | 0.995 | **0.9726** |
| 8 | `lr0p001-wd0p1` | 0.9898 | -0.0183 | 0.994 | **0.9765** |
| 9 | `lr0p0002-wd0p8` | 0.9956 | -0.0169 | 0.989 | **0.9831** |
| 10 | `lr0p0001-wd1p6` | 0.9985 | -0.0184 | 0.989 | **0.9849** |
| 11 | `lr0p001-wd0p2` | 0.9985 | -0.0136 | 0.993 | **0.9884** |
| 12 | `lr0p0002-wd1p6` | 1.0114 | -0.0095 | 0.975 | **1.0042** |
| 13 | `lr0p0005-wd0p8` | 1.0147 | -0.0092 | 0.967 | **1.0076** |
| 14 | `lr0p0005-wd1p6` | 1.0364 | -0.0039 | 0.925 | **1.0334** |
| 15 | `lr0p001-wd0p8` | 1.0390 | -0.0034 | 0.914 | **1.0363** |

## Which runs are best

- **`lr=1e-4/wd=0.2`, `lr=2e-4/wd=0.2`, `lr=5e-4/wd=0.1`, `lr=2e-4/wd=0.1` are a
  four-way tie at the top** — 0.9640 to 0.9654, a spread of 0.0014 that is well inside
  what this extrapolation can resolve. Any of the four is a defensible pick; **`wd=0.1–0.2`
  paired with `lr=1e-4–5e-4` is the region that matters**, not one specific cell.
- **The 10-epoch ranking is misleading.** `lr=5e-4/wd=0.1` has the best measured loss
  today (0.9813) but only 3rd-best projection, because the low-LR runs are still
  descending faster — `lr=1e-4/wd=0.2` has more than 1.5× the slope and overtakes it.
- **High weight decay is the clear loser.** `wd=0.8` and `wd=1.6` at `lr≥5e-4` are
  nearly flat (slope −0.003 to −0.009, and the weakest fits at R²≈0.91–0.93); they have
  effectively stopped improving and more epochs will not rescue them.

## Caveats

- Extrapolating 2× in steps from a 10-point window: treat differences below ~0.005 as noise.
- The fit region has constant LR, so it does not model a second cooldown; the additive
  cooldown-gain correction assumes the same benefit at 20 epochs as was measured at 10.
- `eval/plantcad/loss` spans only ~0.95–1.25 over the logged range, so the plot uses a
  linear y-axis with a log x-axis rather than a broken/multi-scale axis.

![loss projection](exp472_loss_projection.png)

Source: W&B `eric-czech/marin`, key `eval/plantcad/loss`, 20 eval points per run.
