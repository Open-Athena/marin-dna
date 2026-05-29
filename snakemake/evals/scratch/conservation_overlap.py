"""How much do eval positives fall in zoonomia conservation regions? (issue #213)

For ``mendelian_traits`` and ``complex_traits`` (unsplit), reports the
fraction of variants that fall in the human regions used to build the
``zoonomia_projection_dataset``, at two granularities:

- window-level: variant base inside any kept anchor window
  (``proportion_conserved >= cutoff``) for cutoffs 0.10 and 0.20;
- base-level: variant's own nucleotide has ``phyloP_447m >= 2.2162``.

Positives (``label == True``) are the headline; negatives are shown as a
contrast. Broken down by ``consequence_group``.

Reads existing pipeline outputs from S3; writes a figure to ``_scratch/``.
Run single-threaded:

    POLARS_MAX_THREADS=1 uv run python snakemake/evals/scratch/conservation_overlap.py

The 10 GB bigWig must be staged locally first (see BW_LOCAL below):

    aws s3 cp s3://oa-bolinas/snakemake/zoonomia_projection_dataset/results/bigwig/phyloP_447m.bw _scratch/phyloP_447m.bw
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from marin_dna.pipelines.evals.conservation_overlap import (  # noqa: E402
    add_base_conservation,
    add_window_overlap,
)

EVALS = "s3://oa-bolinas/snakemake/evals/results/dataset_unsplit"
ZOON = "s3://oa-bolinas/snakemake/zoonomia_projection_dataset/results"
SCORED = f"{ZOON}/human/intervals/scored/phyloP_447m_windows.parquet"
BW_LOCAL = Path("_scratch/phyloP_447m.bw")  # staged via aws s3 cp (10 GB)

# From snakemake/zoonomia_projection_dataset/config/config.yaml.
PHYLOP_447M_THRESHOLD = 2.2162
CUTOFFS = [0.10, 0.20]
DATASETS = ["mendelian_traits", "complex_traits"]

OUT = Path("_scratch")


def window_col(cutoff: float) -> str:
    return f"in_window_min{cutoff:.2f}"


def annotate(name: str, windows_all: pl.DataFrame) -> pl.DataFrame:
    """Load one dataset and attach window-overlap + base-conservation flags."""
    V = pl.read_parquet(f"{EVALS}/{name}.parquet")
    assert {"chrom", "pos", "label", "consequence_group"}.issubset(V.columns)
    for cutoff in CUTOFFS:
        regions = windows_all.filter(pl.col("proportion_conserved") >= cutoff)
        V = add_window_overlap(V, regions, flag_col=window_col(cutoff))
    V = add_base_conservation(V, BW_LOCAL, PHYLOP_447M_THRESHOLD)
    return V


def md_table(df: pl.DataFrame, float_cols: tuple[str, ...]) -> str:
    """Render a small polars frame as a GitHub markdown table."""
    cols = df.columns
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for r in df.iter_rows(named=True):
        cells = []
        for c in cols:
            v = r[c]
            if c in float_cols and v is not None:
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


def report(name: str, V: pl.DataFrame) -> None:
    flags = [window_col(c) for c in CUTOFFS] + ["base_conserved"]
    n_pos = V.filter(pl.col("label")).height
    n_neg = V.filter(~pl.col("label")).height
    nan_pos = V.filter(pl.col("label") & pl.col("base_phylop").is_null()).height
    print(f"\n{'=' * 72}\n{name}: {V.height} rows ({n_pos} pos / {n_neg} neg)")
    print(
        f"positives with no phyloP_447m signal (NaN): {nan_pos} ({nan_pos / n_pos:.3f})"
    )

    # Overall pos vs neg, one column per flag.
    overall = (
        V.group_by("label")
        .agg(n=pl.len(), *[pl.col(f).mean().alias(f) for f in flags])
        .sort("label", descending=True)
    )
    print("\noverall fraction in-region:")
    print(md_table(overall, float_cols=tuple(flags)))

    # By consequence_group, positives only, ordered by positive count.
    order = (
        V.filter(pl.col("label"))["consequence_group"]
        .value_counts(sort=True)["consequence_group"]
        .to_list()
    )
    by = (
        V.filter(pl.col("label"))
        .group_by("consequence_group")
        .agg(n_pos=pl.len(), *[pl.col(f).mean().alias(f) for f in flags])
        .with_columns(pl.col("consequence_group").cast(pl.Enum(order)).alias("_o"))
        .sort("_o")
        .drop("_o")
    )
    print("\npositives by consequence_group:")
    print(md_table(by, float_cols=tuple(flags)))


def make_figure(annotated: dict[str, pl.DataFrame]) -> None:
    """2x2 grid: rows=dataset, cols={window@0.20, base-level}; pos vs neg bars."""
    metrics = [
        (window_col(0.20), "window-level (≥0.20 conserved)"),
        ("base_conserved", f"base-level (phyloP_447m ≥ {PHYLOP_447M_THRESHOLD})"),
    ]
    fig, axes = plt.subplots(
        len(DATASETS), len(metrics), figsize=(15, 9), squeeze=False
    )
    for i, name in enumerate(DATASETS):
        V = annotated[name]
        order = (
            V.filter(pl.col("label"))["consequence_group"]
            .value_counts(sort=True)["consequence_group"]
            .to_list()
        )
        x = range(len(order))
        for j, (flag, title) in enumerate(metrics):
            ax = axes[i][j]
            pos = [
                V.filter(pl.col("label") & (pl.col("consequence_group") == g))[
                    flag
                ].mean()
                or 0.0
                for g in order
            ]
            neg = [
                V.filter(~pl.col("label") & (pl.col("consequence_group") == g))[
                    flag
                ].mean()
                or 0.0
                for g in order
            ]
            ax.bar(
                [k - 0.2 for k in x], pos, width=0.4, label="positive", color="#c0392b"
            )
            ax.bar(
                [k + 0.2 for k in x], neg, width=0.4, label="negative", color="#7f8c8d"
            )
            ax.set_xticks(list(x))
            ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_ylabel("fraction in-region")
            ax.set_title(f"{name} — {title}", fontsize=10)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle(
        "Eval variants falling in zoonomia conservation regions (issue #213)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "conservation_overlap.png", dpi=130)
    fig.savefig(OUT / "conservation_overlap.svg")
    print(f"\nwrote {OUT}/conservation_overlap.{{png,svg}}")


def main() -> None:
    assert BW_LOCAL.exists(), (
        f"stage the bigWig first: aws s3 cp {ZOON}/bigwig/phyloP_447m.bw {BW_LOCAL}"
    )
    windows_all = (
        pl.scan_parquet(SCORED)
        .select(["chrom", "start", "end", "proportion_conserved"])
        .filter(pl.col("proportion_conserved") >= min(CUTOFFS))
        .collect()
    )
    print(f"scored windows (>= {min(CUTOFFS)} conserved): {windows_all.height:,}")

    annotated = {name: annotate(name, windows_all) for name in DATASETS}
    for name in DATASETS:
        report(name, annotated[name])
    make_figure(annotated)


if __name__ == "__main__":
    main()
