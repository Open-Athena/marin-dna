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
    "exact-acgt-head-b64": "Increase batch",
    "exact-acgt-head-b128": "Increase batch",
    "exact-acgt-head-b256": "Increase batch",
    "exact-acgt-head-b320-mem95": "Increase memory pool + batch",
    "exact-acgt-head-b352-mem95": "Increase memory pool + batch",
    "jax-compact-prefetch-steady-b128": "Overlap HF tokenization",
    "jax-compact-prefetch-steady-b224": "Increase batch",
    "jax-compact-prefetch-steady-b256": "Increase batch",
    "jax-specialized-topk-b224": "Specialized router top-k",
    "jax-no-router-stats-b128": "Inference-only routing",
    "jax-latest-fa4-64x128-b128": "Hopper FA4 64×128 tile",
    "jax-latest-fa4-64x128-b352": "Increase batch",
    "JAX-r6-combined-shared-b128": "Combine shared-expert GEMMs",
    "JAX-r6-combined-shared-b352": "Retune batch",
    "JAX-r6-gemm-fusion-v2-b352": "GEMM fusion v2 (short)",
    "full-vocab-eager-validated": "Full-vocabulary eager",
    "native-grouped-moe": "Grouped MoE kernels",
    "compiled-grouped-moe": "Compile model",
    "direct-hf-tokenizer": "Direct HF tokenizer",
    "reduced-hf-tokenizer": "Reduced HF tokenizer",
    "compiled-flex-expanded": "Local FlexAttention",
    "compiled-flex-global-flash": "Global FlashAttention",
    "compiled-flex-flash-b4": "Increase batch",
    "compiled-flex-flash-b8": "Increase batch",
    "compiled-flex-flash-vllm-moe-b32": "Fused MoE",
    "compiled-flex-flash-vllm-moe-b224": "Increase batch",
    "compiled-flex-flash-vllm-moe-b240": "Increase batch",
    "compiled-flex-flash-vllm-moe-b248": "Increase batch",
    "torch-prefetch-topk8-b224": "Tokenizer pipeline",
    "torch-special-topk-b128": "Specialized router top-k",
    "torch-topk-direct-sconv-b128": "Direct short convolution",
    "latest-stack-tokenizer-batch-b128": "Batched HF tokenizer",
    "torch-latest-repeat-b224": "Increase batch",
    "FA3-native-GQA-compiled-b32": "FlashAttention 3 + native GQA",
    "FA3-native-GQA-compiled-b16": "Retune batch",
    "Torch-precomputed-rope-b16": "Precompute rotary tables",
    "Torch-rope-concat-shared-b16": "Combine shared-expert gate/up",
    "Torch-rope-concat-large-b128": "Increase batch",
    "Torch-rope-concat-extended-b192": "Retune batch",
    "Torch-r5-combined-shared-b192": "Shared GEMMs (short run)",
    "Torch-r6-combined-freezing-b192": "Freeze graph (short run)",
    "Torch-r7-freeze-full-long-b240": "Freeze inference graph",
    "naive-full-vocab-eager": "Naive eager, full vocabulary",
    "exact-acgt-head-eager": "Project only A/C/G/T",
    "exact-acgt-head-compiled": "+ compile engine",
    "compiled-reduced-tokenizer": "+ reduced HF tokenizer",
    "compiled-triton-bm64": "+ tuned Triton MoE kernel",
    "compiled-triton-bm64-b4": "Increase batch",
    "compiled-triton-fa2-b32": "FlashAttention 2 + batch",
    "compiled-triton-fa2-b64-fixed": "Increase batch",
    "compiled-triton-fa2-b128-fixed": "Increase batch",
    "compiled-triton-fa2-b192-fixed": "Increase batch",
    "compiled-triton-fa2-b256-kv8g": "Right-size KV cache + batch",
    "vllm-latest-tokenize-b224": "Batch + overlap HF tokenization",
    "vllm-latest-topk-b224": "Specialized router top-k",
    "vllm-latest-topk-fa3-b224": "FlashAttention 3",
    "vllm-latest-topk-fa3-b236": "Increase batch",
    "vllm-latest-topk-fa3-b240": "Throughput peak",
    "vLLM-tuned-direct-b240-repeat1": "Direct sconv + explicit MoE tile",
    "vLLM-tuned-direct-b240-repeat2": "Long-repeat confirmation",
    "vLLM-chunked-direct-long-b240": "Chunked prefill + direct sconv",
    "vLLM-budget-262144-b240": "Limit scheduler to 262k tokens",
    "vLLM-budget-262144-boundary-b264": "Increase request batch",
    "vLLM-budget-262144-boundary-b416": "Increase request batch",
    "vLLM-budget-262144-boundary-b512": "Increase request batch",
    "vLLM-budget-winner-bf16-b512": "Transfer BF16 logits",
    "vLLM-small-budget-196608-large-b512": "Tune scheduler to 196k tokens",
    "vLLM-small-budget-196608-large-b768": "Increase request batch",
    "vLLM-r5-budget245760-b768": "245k-token scheduler (short run)",
    "vLLM-r6-async-transfer-b768": "Overlap output transfers",
    "vLLM-r7-async-batch1536-budget245760": "Increase request queue",
    "vLLM-r7-async-batch2048-budget245760": "Reach request-queue plateau",
    "vLLM-r8-prealloc-b2048-active64": "Reuse pinned output buffer",
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
    accepted_points = [
        (trial, hour, value)
        for (trial, _, _), hour, value in zip(points, hours, measured, strict=True)
        if trial.accepted
    ]
    rejected = [
        (hour, value)
        for (trial, _, _), hour, value in zip(points, hours, measured, strict=True)
        if not trial.accepted
    ]
    if accepted_points:
        axis.scatter(
            [hour for _, hour, _ in accepted_points],
            [value for _, _, value in accepted_points],
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
    if accepted_points:
        _, start_hour, start_value = accepted_points[0]
        _, best_hour, best_value = max(accepted_points, key=lambda point: point[2])
        annotations = (
            ("Start", start_hour, start_value, (8, -32 if backend == "vllm" else 12)),
            (
                "Best valid",
                best_hour,
                best_value,
                (-8, 12 if backend == "vllm" else -32),
            ),
        )
        for label, hour, value, offset in annotations:
            axis.annotate(
                f"{label}\n{value:,.0f} tok/s",
                (hour, value),
                xytext=offset,
                textcoords="offset points",
                ha="right" if offset[0] < 0 else "left",
                va="bottom" if offset[1] >= 0 else "top",
                fontsize=8.3,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#218739",
                    "alpha": 0.92,
                },
                arrowprops={"arrowstyle": "-", "color": "#218739", "linewidth": 0.8},
                zorder=4,
            )
    frontier_trials = []
    previous_best = 0.0
    for trial, _, _ in points:
        value = trial.input_tokens_per_second
        if (
            trial.accepted
            and value > previous_best
            and (
                trial.name in _DISPLAY_NAMES
                or previous_best == 0
                or value >= previous_best * 1.005
            )
        ):
            frontier_trials.append(trial)
        if trial.accepted and value > previous_best:
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
        "× candidates fail the fixed 32-sequence probability/loss/accuracy gate",
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
    table.set_fontsize(8.4 if len(table_rows) <= 15 else 8.0)
    # Keep a long optimization history inside the same figure geometry used by
    # every backend instead of changing only one backend's plot format.
    # Use the full vertical space beside the plot.  A slightly taller row makes
    # the milestone text easier to scan without changing the figure geometry.
    table.scale(1, min(1.45, 23 / (len(table_rows) + 1)))
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dce1")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f5f9")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
