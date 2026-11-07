"""Render the five matched-batch configurations from their raw process medians."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).parent
payload = json.loads((root / "comparison.json").read_text())
rows = payload["configurations"]
values = np.array([row["median_tokens_per_second"] for row in rows]) / 1e6
lower = np.array([row["min_tokens_per_second"] for row in rows]) / 1e6
upper = np.array([row["max_tokens_per_second"] for row in rows]) / 1e6
labels = [f"{row['id']}  ·  {row['label']}" for row in rows]

with plt.rc_context(
    {"svg.fonttype": "none", "svg.hashsalt": "exp586-deployment-five-configs"}
):
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(rows))
    ax.barh(positions, values, height=0.64)
    ax.errorbar(
        values,
        positions,
        xerr=np.vstack((values - lower, upper - values)),
        fmt="none",
        ecolor="black",
        capsize=0,
        linewidth=1.5,
    )
    for y, value in zip(positions, values, strict=True):
        ax.text(value + 0.045, y, f"{value:.3f}", va="center")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.80)
    ax.set_xticks(np.arange(0, 1.81, 0.3))
    ax.set_xlabel("Forward throughput (million DNA tokens/s)")
    ax.set_title("Single-H100 inference at batch 64")
    ax.set_box_aspect(1)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(root / "throughput.svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(root / "throughput.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    svg = root / "throughput.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
