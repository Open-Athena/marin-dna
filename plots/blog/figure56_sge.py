"""Figures 5 & 6, SGE worlds — params / loss vs SGE AUPRC (S·LLR, S·Probe).

SGE assays only coding/splice, so its scaling scatter is a single panel (missense /
splicing / across-gene macro) rather than the Mendelian 3-group layout in
``figure5_*`` / ``figure6_*``. Same style; params/loss from the vendored scaling CSV,
AUPRC from ``_worlds`` SGE readers on the 8 ladder endpoints. S·Probe renders only
once the endpoints are scored+probed on SGE (skips gracefully otherwise).

Run:  uv run python -m plots.blog.figure56_sge
Out:  plots/output/blog/figure{5_params_vs_vep_auprc,6_loss_vs_vep_auprc}__sge_{llr,probe}.{svg,png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from marin_dna.pipelines.evals.metrics import MACRO_AVG_SUBSET
from plots.blog._regions import REGION_COLORS, VARIANT_REGION, region_legend_handles
from plots.blog._scaling import DATA, LADDER_FINAL_STEP
from plots.blog._style.figure_style import (
    EARTH_QUAL,
    X_LABEL_PAD,
    figsize,
    palette,
)
from plots.blog._style.savefig import save_figure
from plots.blog._worlds import WORLDS

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "blog"

# Fig 5 assays only the two SGE consequences — one panel each, no macro-avg
# (per request). Panel color = the variant's training region (see VARIANT_REGION).
FIG5_SGE_TRAITS: tuple[tuple[str, str], ...] = (
    ("missense_variant", "missense"),
    ("splicing", "splicing"),
)

# (subset, label, EARTH_QUAL slot) — Fig 6 keeps the multi-line layout: missense /
# splicing colors match Figs 5–8, macro slate.
SGE_TRAITS: tuple[tuple[str, str, int], ...] = (
    ("missense_variant", "missense", 0),
    ("splicing", "splicing", 4),
    (MACRO_AVG_SUBSET, "macro avg", 3),
)


def _endpoint_table(world) -> pd.DataFrame:
    """(subset, value, params, eval_loss) for the 8 ladder endpoints in one SGE world."""
    meta = pd.read_csv(DATA)[["run_name", "params", "eval_loss"]]
    frames = []
    for _, r in meta.iterrows():
        stem = r["run_name"].removeprefix("dna-bolinas-")
        try:
            df = world.read(f"{stem}-step-{LADDER_FINAL_STEP}").to_pandas()
        except (LookupError, FileNotFoundError, OSError):
            continue
        df["params"] = int(r["params"])
        df["eval_loss"] = float(r["eval_loss"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build(world) -> None:
    data = _endpoint_table(world)
    if data.empty:
        print(f"figure56_sge: no data for world {world.key} — skipping")
        return
    color_for = {s: EARTH_QUAL[slot] for s, _, slot in SGE_TRAITS}
    params_present = sorted({int(p) for p in data["params"].unique()})
    pal = palette(params_present)

    # Fig 5 (params → AUPRC): one panel per variant type (no macro), region-colored.
    fig5, axes = plt.subplots(1, 2, figsize=figsize(6.4, 4.2), sharex=True)
    for ax, (subset, label) in zip(axes, FIG5_SGE_TRAITS, strict=True):
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

    regions_used = list(dict.fromkeys(VARIANT_REGION[s] for s, _ in FIG5_SGE_TRAITS))
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

    # Fig 6 (loss → AUPRC), single panel, params-colored markers.
    fig6, ax = plt.subplots(figsize=figsize(6.0, 4.2))
    for subset, label, _ in SGE_TRAITS:
        d = data[data["subset"] == subset].sort_values("params")
        if d.empty:
            continue
        for _, row in d.iterrows():
            ax.scatter(
                row["eval_loss"],
                row["value"],
                s=90,
                color=pal[int(row["params"])],
                edgecolors="k",
                linewidths=0.5,
                zorder=3,
            )
        ax.plot(
            d["eval_loss"],
            d["value"],
            color=color_for[subset],
            linewidth=1.0,
            alpha=0.6,
            label=label,
            zorder=2,
        )
    ax.set_xlabel("loss", labelpad=X_LABEL_PAD)
    ax.set_ylabel("AUPRC")
    ax.grid(False)
    ax.legend(fontsize=8, frameon=False, handletextpad=0.4)
    fig6.suptitle(f"Parameter scaling — SGE AUPRC vs loss · {world.label}", fontsize=11)
    fig6.tight_layout(rect=(0, 0.02, 1, 0.99))
    save_figure(fig6, OUTPUT_DIR, f"figure6_loss_vs_vep_auprc__{world.key}")


def build_all() -> None:
    for key in ("sge_llr", "sge_probe"):
        build(WORLDS[key])


if __name__ == "__main__":
    build_all()
