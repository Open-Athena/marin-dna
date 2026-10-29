"""Render one optimization-history figure for each independently tuned backend."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from exp586_moe.inference.ledger import Backend, load_trials, progress

_DISPLAY_NAMES = {
    "full-vocab-head": "Full-vocabulary head",
    "acgt-output-head": "A/C/G/T output head",
    "exact-acgt-head-b2": "Increase batch",
    "exact-acgt-head-b4": "Increase batch",
    "exact-acgt-head-b8": "Increase batch",
    "exact-acgt-head-b16": "Increase batch",
    "exact-acgt-head-b32": "Increase batch",
    "full-vocab-eager-validated": "Full-vocabulary eager",
    "native-grouped-moe": "Grouped MoE kernels",
    "compiled-grouped-moe": "Compile model",
    "direct-hf-tokenizer": "Direct HF tokenizer",
    "reduced-hf-tokenizer": "Reduced HF tokenizer",
    "compiled-flex-expanded": "Local FlexAttention",
    "compiled-flex-global-flash": "Global FlashAttention",
    "compiled-flex-flash-b4": "Increase batch",
    "compiled-flex-flash-b8": "Increase batch",
    "naive-full-vocab-eager": "Naive eager, full vocabulary",
    "exact-acgt-head-eager": "Project only A/C/G/T",
    "exact-acgt-head-compiled": "+ compile engine",
    "compiled-reduced-tokenizer": "+ reduced HF tokenizer",
    "compiled-triton-bm64": "+ tuned Triton MoE kernel",
    "compiled-triton-bm64-b4": "Increase batch",
}
_BACKEND_NAMES = {"jax": "JAX", "torch": "PyTorch", "vllm": "vLLM"}


def plot_backend(backend: Backend, ledger: Path, output: Path) -> None:
    """Plot measured attempts and the accepted throughput frontier over H100 time."""
    points = progress(load_trials(ledger))
    if not points:
        raise ValueError(f"No {backend} benchmark trials to plot")
    hours = [elapsed / 3600 for _, elapsed, _ in points]
    measured = [
        trial.input_tokens_per_second
        if trial.input_tokens_per_second >= 1.0
        else math.nan
        for trial, _, _ in points
    ]
    frontier = [best * trial.tokens_per_sequence for trial, _, best in points]
    fig = plt.figure(figsize=(11.4, 4.8))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.8, 1), wspace=0.06)
    axis = fig.add_subplot(grid[0, 0])
    table_axis = fig.add_subplot(grid[0, 1])
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.88)
    accepted = [
        (hour, value)
        for (trial, _, _), hour, value in zip(points, hours, measured, strict=True)
        if trial.accepted
    ]
    rejected = [
        (hour, value)
        for (trial, _, _), hour, value in zip(points, hours, measured, strict=True)
        if not trial.accepted
    ]
    if accepted:
        axis.scatter(
            [elapsed for elapsed, _ in accepted],
            [value for _, value in accepted],
            color="#218739",
            s=48,
            zorder=3,
        )
    if rejected:
        axis.scatter(
            [elapsed for elapsed, _ in rejected],
            [value for _, value in rejected],
            color="#8b5a5a",
            marker="x",
            linewidths=1.8,
            s=56,
            zorder=3,
        )
    axis.step(hours, frontier, where="post", color="#1769e0", linewidth=2.2)
    frontier_trials = []
    previous_best = 0.0
    for trial, _, _ in points:
        value = trial.input_tokens_per_second
        if trial.accepted and value > previous_best:
            frontier_trials.append(trial)
            previous_best = value
    positive = [value for value in measured if math.isfinite(value) and value > 0]
    if max(positive) / min(positive) >= 20:
        axis.set_yscale("log")
        axis.set_ylim(min(positive) / 1.4, max(positive) * 1.7)
    else:
        axis.set_ylim(0, max(positive) * 1.55)
    axis.set(
        xlabel="Productive H100 experiment time (hours)",
        ylabel="Input tokens / second",
    )
    axis.set_title(
        f"{_BACKEND_NAMES[backend]}: single-H100 inference optimization", pad=28
    )
    axis.text(
        0.5,
        1.01,
        "× candidates are excluded when their output fails the correctness gate",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#6b3f3f",
    )
    axis.grid(alpha=0.22)
    axis.legend(
        handles=(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color="#218739",
                label="Passed correctness",
            ),
            Line2D(
                [],
                [],
                marker="x",
                linestyle="none",
                color="#8b5a5a",
                label="Failed correctness",
            ),
            Line2D(
                [], [], color="#1769e0", linewidth=2.2, label="Best valid throughput"
            ),
        ),
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )
    table_axis.axis("off")
    table_rows = [
        [
            _DISPLAY_NAMES.get(trial.name, trial.name.replace("-", " ")),
            str(trial.batch_size),
            f"{trial.input_tokens_per_second / 1000:,.1f}k",
        ]
        for trial in frontier_trials
    ]
    table = table_axis.table(
        cellText=table_rows,
        colLabels=("Valid speed milestone", "Batch", "tok/s"),
        colWidths=(0.58, 0.17, 0.25),
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1, 1.35)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dce1")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f5f9")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
