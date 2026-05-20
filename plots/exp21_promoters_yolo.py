"""Plot minus_llr_avg AUPRC vs training step for exp21 (animal-promoters-yolo,
1B Qwen3, 512 bp context), on the two promoter-relevant mendelian subsets:
tss_proximal + 5' UTR.

Reads metrics parquets directly from S3, no local download needed. Writes
both SVG (the artifact to upload to GitHub) and PNG (local-iteration
format) into `plots/output/exp21_promoters_yolo/`.

Usage:
    uv run python plots/exp21_promoters_yolo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

ARM_LABEL = "exp21 — 100% animal promoters (1B, 512 bp)"
ARM_COLOR = "#1f77b4"
STEPS: tuple[int, ...] = (2000, 6000, 10000, 12000, 14000, 16000, 18000, 20000, 22000)
SUBSETS: tuple[str, ...] = ("tss_proximal", "5_prime_UTR_variant")
SUBSET_LABELS: dict[str, str] = {
    "tss_proximal": "TSS-proximal (promoter)",
    "5_prime_UTR_variant": "5' UTR",
}
S3_BASE = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"
SCORE_TYPE = "minus_llr_avg"
OUT_DIR = Path(__file__).parent / "output" / Path(__file__).stem
OUT_STEM = "exp21_promoters_yolo_auprc"


def load_all() -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    missing: list[str] = []
    for step in STEPS:
        uri = f"{S3_BASE}/exp21-promoters-yolo-step-{step}/mendelian_traits.parquet"
        try:
            df = pl.read_parquet(uri)
        except Exception as exc:
            missing.append(f"  step-{step}: {exc}")
            continue
        parts.append(df.with_columns(pl.lit(step).alias("step")))
    if missing:
        print(
            f"WARNING: {len(missing)} of {len(STEPS)} parquets unreadable:\n"
            + "\n".join(missing),
            file=sys.stderr,
        )
    assert parts, "no parquets loaded — has the sweep started?"
    return pl.concat(parts)


def main() -> None:
    all_df = load_all()
    df = all_df.filter(
        (pl.col("score_type") == SCORE_TYPE) & (pl.col("subset").is_in(SUBSETS))
    )
    assert not df.is_empty(), (
        f"empty after filtering on {SCORE_TYPE} + {SUBSETS}; "
        f"got score_types={sorted(all_df['score_type'].unique().to_list())}, "
        f"subsets={sorted(all_df['subset'].unique().to_list())}"
    )

    fig, axes = plt.subplots(
        1, len(SUBSETS), figsize=(4.5 * len(SUBSETS), 4), sharex=True
    )
    fig.suptitle("Exp21 Promoters YOLO — convergence on promoter subsets", y=1.02)

    for ax, subset in zip(axes, SUBSETS):
        sub = df.filter(pl.col("subset") == subset).sort("step")
        head = sub.row(0, named=True)
        n_groups = int(head["n_groups"])
        n_rows = int(head["n_rows"])
        ax.plot(
            sub["step"].to_numpy(),
            sub["value"].to_numpy(),
            marker="o",
            color=ARM_COLOR,
            label=ARM_LABEL,
            linewidth=1.5,
            markersize=5,
        )
        ax.set_title(f"{SUBSET_LABELS[subset]}\n(n={n_groups} vs. {n_rows - n_groups})")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("AUPRC")
        ax.grid(False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[-1].legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), title="Model", frameon=False
    )
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = OUT_DIR / f"{OUT_STEM}.{ext}"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
