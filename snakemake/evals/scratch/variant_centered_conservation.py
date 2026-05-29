"""Variant-centered conservation for eval variants (issue #213, iteration 2).

For each variant, build a 255 bp window *centered on the variant* and score
its ``proportion_conserved`` against ``phyloP_447m`` (>= 2.2162; NaN as
non-conserved) -- the same scoring the zoonomia pipeline applies to its
genome-tiled anchors, but anchored on the variant instead. This yields a
continuous per-variant conservation score (vs iteration 1's binary
membership in a pre-tiled anchor).

Positives are the headline; negatives shown as contrast; broken down by
``consequence_group``.

Reuses existing pipeline outputs from S3 + the staged bigWig. Run
single-threaded:

    POLARS_MAX_THREADS=1 uv run python snakemake/evals/scratch/variant_centered_conservation.py

Stage the bigWig first (see BW_LOCAL):

    aws s3 cp s3://oa-bolinas/snakemake/zoonomia_projection_dataset/results/bigwig/phyloP_447m.bw _scratch/phyloP_447m.bw
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pyBigWig  # noqa: E402

from marin_dna.pipelines.conservation.scoring import score_windows  # noqa: E402
from marin_dna.pipelines.evals.conservation_overlap import (  # noqa: E402
    centered_windows,
)

EVALS = "s3://oa-bolinas/snakemake/evals/results/dataset_unsplit"
BW_LOCAL = Path("_scratch/phyloP_447m.bw")  # staged via aws s3 cp (10 GB)

# From snakemake/zoonomia_projection_dataset/config/config.yaml.
PHYLOP_447M_THRESHOLD = 2.2162
WINDOW_SIZE = 255  # zoonomia anchor size (255 bp DNA + BOS = 256 tokens)
PROP_CUTOFFS = [0.10, 0.20]
DATASETS = ["mendelian_traits", "complex_traits"]
OUT = Path("_scratch")


def bare_chrom_sizes(bw_path: Path) -> dict[str, int]:
    """Chromosome lengths keyed by bare Ensembl names (``chr1`` -> ``1``)."""
    bw = pyBigWig.open(str(bw_path))
    try:
        return {
            (c[3:] if c.startswith("chr") else c): int(n)
            for c, n in bw.chroms().items()
        }
    finally:
        bw.close()


def score_dataset(name: str, chrom_sizes: dict[str, int]) -> pl.DataFrame:
    """Per-variant proportion_conserved of a window centered on each variant."""
    V = pl.read_parquet(f"{EVALS}/{name}.parquet")
    assert {"chrom", "pos", "label", "consequence_group"}.issubset(V.columns)
    win = centered_windows(V, WINDOW_SIZE, chrom_sizes=chrom_sizes)
    scored = score_windows(BW_LOCAL, win, threshold=PHYLOP_447M_THRESHOLD)
    # mean phyloP with unaligned (NaN) bases counted as 0, over the full
    # window length -- consistent with proportion_conserved's NaN-as-0
    # convention. score_windows' own `mean_phylop` is nanmean (finite bases
    # only); reconstruct the NaN-as-0 mean as sum(finite) / window_len =
    # mean_phylop * n_valid_bases / (end - start). n_valid == 0 -> 0.
    scored = scored.with_columns(
        pl.when(pl.col("n_valid_bases") > 0)
        .then(
            pl.col("mean_phylop")
            * pl.col("n_valid_bases")
            / (pl.col("end") - pl.col("start"))
        )
        .otherwise(0.0)
        .alias("mean_phylop_nan0")
    )
    # All windows should be full-length except the rare near-telomere variant.
    short = scored.filter((pl.col("end") - pl.col("start")) < WINDOW_SIZE).height
    print(f"  {name}: {short} variants got a clipped (<{WINDOW_SIZE} bp) window")
    return scored


def summary_stats() -> list[pl.Expr]:
    return [
        pl.col("proportion_conserved").mean().alias("mean_prop"),
        pl.col("proportion_conserved").median().alias("median_prop"),
        # NaN-as-0 mean phyloP (unaligned bases count as 0); see score_dataset.
        pl.col("mean_phylop_nan0").mean().alias("mean_phylop"),
        *[
            (pl.col("proportion_conserved") >= c).mean().alias(f"frac>={c:.2f}")
            for c in PROP_CUTOFFS
        ],
    ]


def md_table(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in df.iter_rows(named=True):
        cells = [f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report(name: str, scored: pl.DataFrame) -> None:
    print(
        f"\n{'=' * 72}\n{name}: variant-centered {WINDOW_SIZE} bp proportion_conserved"
    )
    overall = (
        scored.group_by("label")
        .agg(n=pl.len(), *summary_stats())
        .sort("label", descending=True)
    )
    print("\noverall (positive vs negative):")
    print(md_table(overall))

    order = (
        scored.filter(pl.col("label"))["consequence_group"]
        .value_counts(sort=True)["consequence_group"]
        .to_list()
    )
    by = (
        scored.filter(pl.col("label"))
        .group_by("consequence_group")
        .agg(n_pos=pl.len(), *summary_stats())
        .with_columns(pl.col("consequence_group").cast(pl.Enum(order)).alias("_o"))
        .sort("_o")
        .drop("_o")
    )
    print("\npositives by consequence_group:")
    print(md_table(by))


def make_figure(scored: dict[str, pl.DataFrame]) -> None:
    """rows=dataset; col0 = ECDF (pos vs neg); col1 = boxplot by consequence_group."""
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(15, 9), squeeze=False)
    for i, name in enumerate(DATASETS):
        S = scored[name]
        pos = S.filter(pl.col("label"))["proportion_conserved"].to_numpy()
        neg = S.filter(~pl.col("label"))["proportion_conserved"].to_numpy()

        ax = axes[i][0]
        for arr, lab, col in [
            (pos, "positive", "#c0392b"),
            (neg, "negative", "#7f8c8d"),
        ]:
            xs = np.sort(arr)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, label=f"{lab} (n={len(xs)})", color=col, lw=2)
        ax.set_xlabel("proportion_conserved (variant-centered 255 bp)")
        ax.set_ylabel("ECDF")
        ax.set_title(f"{name} — ECDF", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[i][1]
        order = (
            S.filter(pl.col("label"))["consequence_group"]
            .value_counts(sort=True)["consequence_group"]
            .to_list()
        )
        data = [
            S.filter(pl.col("label") & (pl.col("consequence_group") == g))[
                "proportion_conserved"
            ].to_numpy()
            for g in order
        ]
        ax.boxplot(data, vert=True, showmeans=True, widths=0.6)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("proportion_conserved")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{name} — positives by consequence_group", fontsize=10)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Variant-centered {WINDOW_SIZE} bp conservation (phyloP_447m ≥ "
        f"{PHYLOP_447M_THRESHOLD}) — issue #213",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "variant_centered_conservation.png", dpi=130)
    fig.savefig(OUT / "variant_centered_conservation.svg")
    print(f"\nwrote {OUT}/variant_centered_conservation.{{png,svg}}")


def main() -> None:
    assert BW_LOCAL.exists(), f"stage the bigWig first to {BW_LOCAL}"
    chrom_sizes = bare_chrom_sizes(BW_LOCAL)
    scored = {}
    for name in DATASETS:
        scored[name] = score_dataset(name, chrom_sizes)
        scored[name].write_parquet(OUT / f"variant_centered_{name}.parquet")
    for name in DATASETS:
        report(name, scored[name])
    make_figure(scored)


if __name__ == "__main__":
    main()
