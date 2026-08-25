"""Fit pre-cooldown loss trends for the exp472 sweep and project 10 more epochs.

Reads ``eval/plantcad/loss`` from W&B for every completed trial in the CoreWeave
state database, fits ``log10(loss) = a*log10(step) + b`` over the pre-cooldown
window, and extrapolates that trend to twice the training length. Post-cooldown evals are
plotted for context but take no part in the fit or the projection. Writes the report
table and the figure next to this file.

Usage: uv run --with matplotlib python project_loss.py
"""

import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import wandb  # noqa: E402

HERE = Path(__file__).resolve().parent
DB = HERE.parent / "scratch" / "exp472-cw" / "exp472_sweep.sqlite"
ENTITY_PROJECT = "eric-czech/marin"
METRIC = "eval/plantcad/loss"

TOTAL_STEPS = 206_145          # 10 epochs
EPOCHS = 10
STEPS_PER_EPOCH = TOTAL_STEPS / EPOCHS
DECAY_FRACTION = 0.2           # common.py: decay=0.2
COOLDOWN_STEP = int(round(TOTAL_STEPS * (1 - DECAY_FRACTION)))
FIT_LO = COOLDOWN_STEP - 100_000
FIT_HI = COOLDOWN_STEP + int(COOLDOWN_STEP * 0.01)   # include the eval just past cooldown
PROJECT_STEP = TOTAL_STEPS + 10 * STEPS_PER_EPOCH


def load_histories() -> dict[str, list[tuple[float, float]]]:
    wanted = {
        run_id: trial
        for trial, run_id in sqlite3.connect(DB).execute(
            "select trial_id, wandb_run_id from trials where status = 'completed'"
        )
    }
    api = wandb.Api(timeout=120)
    runs = api.runs(
        ENTITY_PROJECT, filters={"displayName": {"$in": list(wanted)}}, per_page=200
    )
    histories: dict[str, list[tuple[float, float]]] = {}
    for run in runs:
        rows = run.history(keys=[METRIC], samples=2000, pandas=False)
        histories[wanted[run.name]] = sorted(
            (row["_step"], row[METRIC]) for row in rows if row.get(METRIC) is not None
        )
    return histories


def fit_all(histories: dict[str, list[tuple[float, float]]]) -> list[dict]:
    fits = []
    for trial, points in histories.items():
        steps, loss = np.array(points, dtype=float).T
        window = (steps >= FIT_LO) & (steps <= FIT_HI)
        x, y = np.log10(steps[window]), np.log10(loss[window])
        design = np.vstack([x, np.ones_like(x)]).T
        (slope, intercept), *_ = np.linalg.lstsq(design, y, rcond=None)
        predicted = design @ np.array([slope, intercept])
        r2 = 1 - ((y - predicted) ** 2).sum() / ((y - y.mean()) ** 2).sum()

        def trend(step: float) -> float:
            return float(10 ** (slope * np.log10(step) + intercept))

        fits.append(
            {
                "trial": trial,
                "slope": float(slope),
                "intercept": float(intercept),
                "r2": float(r2),
                "last_precooldown_step": float(steps[window][-1]),
                "last_precooldown": float(loss[window][-1]),
                "final_post_cooldown": float(loss[-1]),
                "projected": trend(PROJECT_STEP),
                "steps": steps,
                "loss": loss,
                "window": window,
            }
        )
    return sorted(fits, key=lambda f: f["projected"])


def plot(fits: list[dict], path: Path) -> None:
    cmap = plt.get_cmap("viridis")
    colors = {f["trial"]: cmap(i / (len(fits) - 1)) for i, f in enumerate(fits)}
    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22)
    sweep = np.logspace(np.log10(9_000), np.log10(PROJECT_STEP), 300)

    ax = fig.add_subplot(grid[0, :])
    ax.axvspan(FIT_LO, FIT_HI, color="0.85", zorder=0, label="fit window (pre-cooldown)")
    ax.axvline(COOLDOWN_STEP, color="k", ls=":", lw=1)
    ax.axvline(TOTAL_STEPS, color="k", lw=0.8)
    for f in fits:
        color = colors[f["trial"]]
        ax.plot(f["steps"], f["loss"], lw=1.5, color=color)
        tail = sweep[sweep >= FIT_LO]
        ax.plot(tail, 10 ** (f["slope"] * np.log10(tail) + f["intercept"]),
                lw=0.9, ls="--", color=color, alpha=0.8)
        ax.plot(PROJECT_STEP, f["projected"], "o", ms=5, color=color)
    ax.set_xscale("log")
    ax.set_xlim(9_000, PROJECT_STEP * 1.15)
    ax.set_ylim(0.95, 1.26)
    ax.set_xlabel("training step (log)")
    ax.set_ylabel(METRIC)
    ax.set_title(
        "exp472 · observed loss (solid), pre-cooldown power-law fit extrapolated "
        "(dashed), 20-epoch projection from that trend alone (dot)"
    )
    ax.text(COOLDOWN_STEP, 1.253, "cooldown\nstarts", fontsize=8, va="top", ha="right")
    ax.text(TOTAL_STEPS, 1.253, "  10 ep end\n  (cooldown done)", fontsize=8, va="top")
    ax.text(PROJECT_STEP, 1.253, "20 ep\nprojected  ", fontsize=8, va="top", ha="right")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(grid[1, 0])
    for f in fits:
        color = colors[f["trial"]]
        ax.plot(np.log10(f["steps"][f["window"]]), np.log10(f["loss"][f["window"]]),
                "o", ms=3, color=color)
        xs = np.log10(np.array([FIT_LO, PROJECT_STEP]))
        ax.plot(xs, f["slope"] * xs + f["intercept"], lw=1, color=color)
    ax.set_xlabel("log10 step")
    ax.set_ylabel("log10 loss")
    ax.set_title("fit region in log-log space (line = least squares)", fontsize=10)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(grid[1, 1])
    ys = np.arange(len(fits))
    edge = [colors[f["trial"]] for f in fits]
    ax.scatter([f["last_precooldown"] for f in fits], ys, s=42, facecolors="none",
               edgecolors=edge, label="measured @ cooldown start (step 164,928)")
    ax.scatter([f["projected"] for f in fits], ys, s=42, marker="D",
               color=edge, label="projected @20 ep (same trend, step 412,290)")
    for i, f in enumerate(fits):
        ax.plot([f["last_precooldown"], f["projected"]], [i, i],
                color=colors[f["trial"]], lw=1, alpha=0.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([f["trial"] for f in fits], fontsize=8)
    low = min(min(f["last_precooldown"], f["projected"]) for f in fits)
    high = max(max(f["last_precooldown"], f["projected"]) for f in fits)
    pad = (high - low) * 0.10
    ax.set_xlim(low - pad, high + pad * 2.2)
    ax.set_ylim(len(fits) - 0.3, -0.7)
    ax.set_xlabel(METRIC)
    ax.set_title("ranked by projected 20-epoch loss (best at top)\nboth points are pre-cooldown, so directly comparable", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25, axis="x")

    fig.savefig(path, dpi=150, bbox_inches="tight")


def main() -> None:
    fits = fit_all(load_histories())
    plot(fits, HERE / "exp472_loss_projection.png")
    (HERE / "exp472_fits.json").write_text(
        json.dumps(
            [{k: v for k, v in f.items() if k not in ("steps", "loss", "window")}
             for f in fits],
            indent=1,
        )
    )
    for rank, f in enumerate(fits, 1):
        print(f"{rank:2d} {f['trial']:16s} slope={f['slope']:+.4f} "
              f"r2={f['r2']:.3f} precool={f['last_precooldown']:.4f} "
              f"proj20={f['projected']:.4f}")


if __name__ == "__main__":
    main()
