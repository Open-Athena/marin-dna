"""Figures 5 & 6, SGE worlds — params / CDS loss vs SGE AUPRC.

SGE assays only coding/splice, so Figs 5 and 6 use one panel for each of those two
variant types and omit the across-gene macro. Fig 6 matches the Mendelian scatter
style and uses CDS validation loss for both SGE consequences. AUPRC comes from
``_worlds`` SGE readers on the 8 ladder endpoints. S·Probe renders only once the
endpoints are scored+probed on SGE (skips gracefully otherwise).

Run:  uv run python -m plots.blog.figure56_sge
Out:  plots/output/blog/figure{5_params_vs_vep_auprc,6_loss_vs_vep_auprc}__sge_{llr,probe}.{svg,png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plots.blog._regions import (
    REGION_COLORS,
    SGE_VARIANT_ORDER,
    VARIANT_REGION,
    region_legend_handles,
)
from plots.blog._scaling import (
    DATA,
    LADDER_FINAL_STEP,
    REGION_LOSS_COLUMNS,
    add_relevant_region_loss,
)
from plots.blog._style.figure_style import (
    X_LABEL_PAD,
    attach_params_legend_below,
    figsize,
    palette,
)
from plots.blog._style.savefig import save_figure
from plots.blog._worlds import WORLDS
from plots.blog.figure6_loss_vs_vep_auprc import plot_loss_panel

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "blog"

SGE_TRAITS: tuple[tuple[str, str], ...] = (
    ("missense_variant", "missense"),
    ("splicing", "splicing"),
)
assert tuple(subset for subset, _ in SGE_TRAITS) == SGE_VARIANT_ORDER


def _endpoint_table(world) -> pd.DataFrame:
    """Two SGE subsets with params and corresponding-region loss at each endpoint."""
    meta = pd.read_csv(DATA)[["run_name", "params", *REGION_LOSS_COLUMNS]]
    frames = []
    for _, r in meta.iterrows():
        stem = r["run_name"].removeprefix("dna-bolinas-")
        try:
            df = world.read(f"{stem}-step-{LADDER_FINAL_STEP}").to_pandas()
        except (LookupError, FileNotFoundError, OSError):
            continue
        df = df[df["subset"].isin(SGE_VARIANT_ORDER)].copy()
        assert set(df["subset"]) == set(SGE_VARIANT_ORDER), (
            f"{stem}: expected SGE subsets {list(SGE_VARIANT_ORDER)}, "
            f"got {sorted(df['subset'].unique())}"
        )
        df["params"] = int(r["params"])
        frames.append(add_relevant_region_loss(df, r))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build(world) -> None:
    data = _endpoint_table(world)
    if data.empty:
        print(f"figure56_sge: no data for world {world.key} — skipping")
        return
    params_present = sorted({int(p) for p in data["params"].unique()})
    pal = palette(params_present)

    # Fig 5 (params → AUPRC): one panel per variant type (no macro), region-colored.
    fig5, axes = plt.subplots(1, 2, figsize=figsize(6.4, 4.2), sharex=True)
    for ax, (subset, label) in zip(axes, SGE_TRAITS, strict=True):
        color = REGION_COLORS[VARIANT_REGION[subset]]
        d = data[data["subset"] == subset].sort_values("params")
        if not d.empty:
            # Capless ±1 SE bars (drawn only where `se` is finite).
            ax.errorbar(
                d["params"],
                d["value"],
                yerr=d["se"],
                marker="o",
                linestyle="-",
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=0,
                linewidth=1.3,
                markersize=6,
                markeredgecolor="k",
                markeredgewidth=0.4,
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_title(label[:1].upper() + label[1:], fontsize=10)
        ax.set_xlabel("model params", labelpad=X_LABEL_PAD)
        ax.grid(False)
    axes[0].set_ylabel("AUPRC")

    regions_used = list(dict.fromkeys(VARIANT_REGION[s] for s, _ in SGE_TRAITS))
    handles, labels = region_legend_handles(regions_used)
    fig5.legend(
        handles,
        labels,
        title="relevant training region",
        ncol=len(handles),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=9,
        title_fontsize=9,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.6,
    )
    fig5.suptitle(
        f"Parameter scaling — AUPRC by variant type · {world.label}", fontsize=11
    )
    fig5.tight_layout(rect=(0, 0.08, 1, 0.95))
    save_figure(fig5, OUTPUT_DIR, f"figure5_params_vs_vep_auprc__{world.key}")

    # Fig 6 (CDS validation loss → AUPRC): one panel per variant type, matching
    # the Mendelian visual grammar (params-colored scatter + dotted fit + Pearson r).
    fig6, axes = plt.subplots(1, 2, figsize=figsize(6.4, 4.2), sharex=True)
    for ax, (subset, label) in zip(axes, SGE_TRAITS, strict=True):
        d = data[data["subset"] == subset].sort_values("params")
        plot_loss_panel(ax, d, pal)
        variant_counts = d["n_variants"].dropna().unique()
        assert len(variant_counts) == 1, (
            f"{world.key}/{subset}: inconsistent variant counts {variant_counts}"
        )
        n_variants = int(variant_counts[0])
        ax.set_title(f"{label[:1].upper() + label[1:]} (n={n_variants:,})", fontsize=10)
    axes[0].set_ylabel("AUPRC")
    fig6.suptitle(
        "Parameter scaling — corresponding-region validation loss vs SGE AUPRC "
        f"· {world.label}",
        fontsize=11,
        y=0.96,
    )
    fig6.tight_layout(rect=(0, 0.16, 1, 0.98))
    attach_params_legend_below(
        fig6, pal, params_present, width_scale=0.25, handlelength=0.8
    )
    save_figure(fig6, OUTPUT_DIR, f"figure6_loss_vs_vep_auprc__{world.key}")


def build_all() -> None:
    for key in ("sge_llr", "sge_probe"):
        build(WORLDS[key])


if __name__ == "__main__":
    build_all()
