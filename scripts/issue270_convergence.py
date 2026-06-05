"""issue #270 (cLLR subsampling pilot): how fast does the per-cell mean LLR converge?

For each pentanucleotide calibration cell (pentanuc_mut = 5mer + "_" + alt) we
have ~1000 neutral-site LLRs (FWD+RC averaged). The calibration uses the cell
*mean* as `llr_neutral_mean`. This script measures how precisely that mean is
estimated as a function of the number of neutral sites n:

  SE(n) = std over bootstrap resamples (size n, with replacement) of the mean
        ≈ s / sqrt(n),  s = within-cell SD of the LLR.

Output: a 2-panel figure (mean ± 95% CI vs n; SE(n) vs n with the s/sqrt(n)
reference) and a table of n* needed to reach SE thresholds — the input to
choosing a per-bin neutral-sampling budget for the full calibration.

Run:
    uv run python scripts/issue270_convergence.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCORES = os.environ.get(
    "ISSUE270_SCORES", "s3://oa-bolinas/scratch/issue270/pilot_scores.parquet"
)
OUT_FIG = "scratch/issue270/convergence"  # .svg + .png
SCORE_COL = "llr_avg"
SEED = 270
N_BOOT = 2000
N_GRID = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
SE_THRESHOLDS = (0.10, 0.05, 0.02)  # nats


def bootstrap_se(x: np.ndarray, n: int, n_boot: int, rng: np.random.Generator) -> float:
    """SE of the mean at sample size n via with-replacement bootstrap of x."""
    idx = rng.integers(0, len(x), size=(n_boot, n))
    return float(x[idx].mean(axis=1).std(ddof=1))


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(SCORES)
    cells = sorted(df["pentanuc_mut"].unique())
    print(f"loaded {len(df):,} scored variants; {len(cells)} cells\n")

    rows = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    for cell in cells:
        sub = df[df["pentanuc_mut"] == cell]
        x = sub[SCORE_COL].to_numpy()
        ctx = sub["context_label"].iloc[0]
        s = float(x.std(ddof=1))
        mean = float(x.mean())
        grid = [n for n in N_GRID if n <= len(x)]
        se_boot = np.array([bootstrap_se(x, n, N_BOOT, rng) for n in grid])
        se_analytic = s / np.sqrt(grid)
        curves[cell] = dict(
            n=np.array(grid),
            se_boot=se_boot,
            se_analytic=se_analytic,
            mean=mean,
            ctx=ctx,
        )
        # n* = smallest n with analytic SE below threshold (n = (s/thr)^2).
        nstar = {thr: int(np.ceil((s / thr) ** 2)) for thr in SE_THRESHOLDS}
        rows.append(
            dict(
                context=ctx,
                cell=cell,
                n_avail=len(x),
                mean=mean,
                sd=s,
                se_n1000=s / np.sqrt(1000),
                **{f"n*(SE<{t})": nstar[t] for t in SE_THRESHOLDS},
            )
        )

    table = pd.DataFrame(rows).sort_values(["context", "cell"])
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- figure ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    cmap = plt.get_cmap("tab10")
    for i, cell in enumerate(cells):
        c = curves[cell]
        color = cmap(i)
        ls = "-" if c["ctx"] == "cpg" else "--"
        # Left: mean ± 95% CI (=1.96*SE) vs n — visual convergence.
        axL.fill_between(
            c["n"],
            c["mean"] - 1.96 * c["se_boot"],
            c["mean"] + 1.96 * c["se_boot"],
            color=color,
            alpha=0.15,
        )
        axL.plot(
            c["n"],
            np.full_like(c["n"], c["mean"], dtype=float),
            color=color,
            ls=ls,
            lw=1,
            label=cell,
        )
        # Right: SE(n) bootstrap (markers) + s/sqrt(n) reference (thin line).
        axR.plot(c["n"], c["se_boot"], color=color, ls=ls, marker="o", ms=3, label=cell)
        axR.plot(c["n"], c["se_analytic"], color=color, ls=":", lw=0.8)

    axL.set(
        xscale="log",
        xlabel="n neutral sites",
        ylabel=f"cell mean {SCORE_COL} ± 95% CI",
        title="Per-cell mean LLR ± 95% CI vs n",
    )
    axL.legend(fontsize=7, ncol=2)
    for t in SE_THRESHOLDS:
        axR.axhline(t, color="gray", lw=0.5, ls="-", alpha=0.5)
        axR.text(N_GRID[-1], t, f" SE={t}", va="center", fontsize=6, color="gray")
    axR.set(
        xscale="log",
        yscale="log",
        xlabel="n neutral sites",
        ylabel=f"SE of mean {SCORE_COL} (nats)",
        title="Sampling SE(n)  (dotted = s/√n)",
    )
    axR.legend(fontsize=7, ncol=2)
    fig.suptitle(
        "cLLR neutral-baseline convergence — exp135-1B-m5.1 (solid=CpG CACGT, dashed=typical AAATG)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}.svg")
    fig.savefig(f"{OUT_FIG}.png", dpi=150)
    print(f"\nwrote {OUT_FIG}.svg / .png")


if __name__ == "__main__":
    main()
