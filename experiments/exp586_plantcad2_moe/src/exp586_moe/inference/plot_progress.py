"""Render one optimization-history figure for each independently tuned backend."""

from pathlib import Path

import matplotlib.pyplot as plt

from exp586_moe.inference.ledger import Backend, load_trials, progress


def plot_backend(backend: Backend, ledger: Path, output: Path) -> None:
    """Plot measured attempts and the accepted throughput frontier over H100 time."""
    points = progress(load_trials(ledger))
    if not points:
        raise ValueError(f"No {backend} benchmark trials to plot")
    hours = [elapsed / 3600 for _, elapsed, _ in points]
    measured = [trial.sequences_per_second for trial, _, _ in points]
    frontier = [best for _, _, best in points]
    colors = ["#218739" if trial.accepted else "#9aa0a6" for trial, _, _ in points]

    fig, axis = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    axis.scatter(hours, measured, c=colors, s=48, zorder=3)
    axis.step(hours, frontier, where="post", color="#1769e0", linewidth=2.2)
    for (trial, elapsed, _), value in zip(points, measured, strict=True):
        if trial.accepted:
            on_right = elapsed == points[-1][1]
            axis.annotate(
                f"{trial.name}\n{value:.2f} seq/s",
                (elapsed / 3600, value),
                xytext=(-5 if on_right else 5, 7),
                textcoords="offset points",
                horizontalalignment="right" if on_right else "left",
                fontsize=8,
            )
    positive = [value for value in measured if value > 0]
    if max(positive) / min(positive) >= 20:
        axis.set_yscale("log")
        axis.set_ylim(min(positive) / 1.4, max(positive) * 1.7)
    axis.set(
        title=f"{backend.upper()}: single-H100 inference optimization",
        xlabel="Productive H100 experiment time (hours)",
        ylabel="8,192-base sequences / second",
    )
    axis.grid(alpha=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)

