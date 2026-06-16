"""Plot minus_llr_avg AUPRC vs training step for the exp166 zoonomia
1-epoch scaling experiment, comparing the two model sizes:

  - 1B  (exp166-v0.1-p1B, c127da)
  - 4B  (exp166-v0.1-p4B, 74e146)

Both are Qwen3 trained for one full epoch on `bolinas-dna/zoonomia-v1-v1`
(255 bp + BOS), evaluated offline on the matched-pair TraitGym Mendelian v2
dataset (PR #194 k=9 rebuild). Only the three HF-exported checkpoints
(steps 10000 / 20000 / 27329) are available offline, so each line has three
points — the in-training WandB AUPRC trajectory (issue body) is finer.

One panel per consequence subset, plus the `_global_` and `_macro_avg_`
aggregates. Error bars are the paired cluster-bootstrap SE. Complex traits
are omitted: that dataset was only evaluated at the final checkpoint, so it
has no training-step trajectory.

Reads metrics parquets directly from S3, no local download needed. Writes
both SVG (the artifact to upload to GitHub) and PNG (local-iteration
format) into `plots/output/exp166_zoonomia_scaling/`.

Usage:
    uv run python plots/exp166_zoonomia_scaling.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import polars as pl


class Model(NamedTuple):
    prefix: str
    label: str
    color: str


# Both runs exported the same three HF checkpoints.
STEPS: tuple[int, ...] = (10000, 20000, 27329)
MODELS: tuple[Model, ...] = (
    Model(prefix="exp166-v0.1-p1B", label="1B", color="#1f77b4"),
    Model(prefix="exp166-v0.1-p4B", label="4B", color="#d62728"),
)

# Panel order: aggregates first, then grouped by function
# (regulatory → coding → splicing/non-coding → distal → rare).
SUBSETS: tuple[str, ...] = (
    "_global_",
    "_macro_avg_",
    "tss_proximal",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "missense_variant",
    "synonymous_variant",
    "splicing",
    "non_coding_transcript_exon_variant",
    "distal",
    "mature_miRNA_variant",
)
SUBSET_LABELS: dict[str, str] = {
    "_global_": "Global",
    "_macro_avg_": "Macro-average",
    "tss_proximal": "TSS-proximal",
    "5_prime_UTR_variant": "5' UTR",
    "3_prime_UTR_variant": "3' UTR",
    "missense_variant": "Missense",
    "synonymous_variant": "Synonymous",
    "splicing": "Splicing",
    "non_coding_transcript_exon_variant": "ncRNA exon",
    "distal": "Distal enhancer",
    "mature_miRNA_variant": "mature miRNA",
}
S3_BASE = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"
DATASET = "mendelian_traits"
SCORE_TYPE = "minus_llr_avg"
NCOLS = 4
OUT_DIR = Path(__file__).parent / "output" / Path(__file__).stem
OUT_STEM = "exp166_zoonomia_scaling_auprc"


def load_all() -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    missing: list[str] = []
    total = len(MODELS) * len(STEPS)
    for model in MODELS:
        for step in STEPS:
            uri = f"{S3_BASE}/{model.prefix}-step-{step}/{DATASET}.parquet"
            try:
                df = pl.read_parquet(uri)
            except Exception as exc:
                missing.append(f"  {model.label} step-{step}: {exc}")
                continue
            parts.append(
                df.with_columns(
                    pl.lit(model.label).alias("model"),
                    pl.lit(step).alias("step"),
                )
            )
    if missing:
        print(
            f"WARNING: {len(missing)} of {total} parquets unreadable:\n"
            + "\n".join(missing),
            file=sys.stderr,
        )
    assert parts, "no parquets loaded — has the eval run finished?"
    return pl.concat(parts)


def panel_title(subset: str, head: dict[str, object]) -> str:
    n_groups = int(head["n_groups"])
    n_rows = int(head["n_rows"])
    label = SUBSET_LABELS.get(subset, subset)
    if subset == "_macro_avg_":
        # macro_avg's n_groups is the subset count, not a positive/negative split.
        return f"{label}\n(mean over {n_groups} subsets)"
    # _global_ and the per-consequence subsets: positives vs. matched negatives.
    return f"{label}\n(n={n_groups} vs {n_rows - n_groups})"


def step_label(step: int) -> str:
    """Compact tick label for a training step: 10000 -> '10k', 27329 -> '27.3k'."""
    k = step / 1000
    return f"{k:.0f}k" if k.is_integer() else f"{k:.1f}k"


def main() -> None:
    all_df = load_all()
    df = all_df.filter(
        (pl.col("score_type") == SCORE_TYPE) & (pl.col("subset").is_in(SUBSETS))
    )
    assert not df.is_empty(), (
        f"empty after filtering on {SCORE_TYPE} + subsets; "
        f"got score_types={sorted(all_df['score_type'].unique().to_list())}, "
        f"subsets={sorted(all_df['subset'].unique().to_list())}"
    )

    n_panels = len(SUBSETS)
    nrows = math.ceil(n_panels / NCOLS)
    fig, axes = plt.subplots(
        nrows, NCOLS, figsize=(4.2 * NCOLS, 3.4 * nrows), sharex=True
    )
    axes_flat = axes.flatten()
    fig.suptitle(
        "exp166 zoonomia 1-epoch scaling — Mendelian AUPRC (minus_llr_avg) "
        "vs training step",
        y=1.005,
        fontsize=14,
    )

    for i, subset in enumerate(SUBSETS):
        ax = axes_flat[i]
        sub = df.filter(pl.col("subset") == subset)
        head = sub.row(0, named=True)
        for model in MODELS:
            m = sub.filter(pl.col("model") == model.label).sort("step")
            if m.is_empty():
                continue
            ax.errorbar(
                m["step"].to_numpy(),
                m["value"].to_numpy(),
                yerr=m["se"].to_numpy(),
                marker="o",
                color=model.color,
                label=model.label,
                linewidth=1.5,
                markersize=5,
                capsize=3,
                elinewidth=1,
            )
        ax.set_title(panel_title(subset, head), fontsize=10)
        ax.set_xticks(STEPS)
        ax.set_xticklabels([step_label(s) for s in STEPS])
        if i % NCOLS == 0:
            ax.set_ylabel("AUPRC")
        if i // NCOLS == nrows - 1:
            ax.set_xlabel("Training step")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Legend in the first unused slot; hide any remaining empty axes.
    handles, labels = axes_flat[0].get_legend_handles_labels()
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].axis("off")
    if n_panels < len(axes_flat):
        axes_flat[n_panels].legend(
            handles, labels, title="Model size", loc="center", frameon=False
        )
    else:
        fig.legend(handles, labels, title="Model size", loc="lower right")

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = OUT_DIR / f"{OUT_STEM}.{ext}"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
