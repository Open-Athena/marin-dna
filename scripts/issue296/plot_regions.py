"""Figure 1 (issue #296): mean loss by region vs model scale (sanity), + strand subplot.

Main: mean loss for the reliable regions (CDS by codon position + intronic splice
+ overall; UTRs dropped) vs params. Subplot (collapsible in the issue): the splice
region split by gene strand — CDS-primed vs intron-primed donor/acceptor.

Globs the per-model Stage-2 stratum parquets, so it auto-extends as more
checkpoints are processed. Writes PNGs to plots/output/issue296/.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter  # noqa: E402

STAGE2 = Path("scratch/issue296/stage2")
OUT = Path("plots/output/issue296")


def _params_millions(model: str) -> float:
    m = re.search(r"p(\d+(?:\.\d+)?)([MB])", model)
    assert m, model
    return float(m.group(1)) * (1000.0 if m.group(2) == "B" else 1.0)


def _load(col: str) -> pl.DataFrame:
    rows = []
    for p in sorted(glob.glob(f"{STAGE2}/*/val_cds_stratum_ll_gap.parquet")):
        model = Path(p).parent.name
        d = {r["stratum"]: r[col] for r in pl.read_parquet(p).iter_rows(named=True)}
        d["params_M"] = _params_millions(model)
        rows.append(d)
    assert rows, f"no stratum parquets under {STAGE2}"
    return pl.DataFrame(rows).sort("params_M")


def _log_xaxis(ax, p: np.ndarray) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(p)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticklabels([str(int(x)) for x in p])
    ax.set_xlabel("parameters (M, log scale)")


def main() -> None:
    t = _load("mean_loss")
    p = t["params_M"].to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)

    # --- Main: mean loss by region ------------------------------------------
    regions = [
        ("all_token", "all tokens (overall)", "black", "-"),
        ("codon_1", "CDS codon pos 1", "tab:blue", "-"),
        ("codon_2", "CDS codon pos 2", "tab:cyan", "-"),
        ("codon_3", "CDS codon pos 3 (wobble)", "tab:red", "-"),
        ("codon_3_4fold", "CDS codon pos 3, 4-fold degenerate", "tab:orange", "--"),
        ("splicing", "intronic splice (≤20bp, donor+acceptor)", "tab:green", "-"),
        ("splice_donor", "splice donor", "tab:olive", "--"),
        ("splice_acceptor", "splice acceptor", "mediumseagreen", ":"),
        ("other_noncoding", "other non-coding (deep intron)", "tab:brown", "-"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for col, lab, c, ls in regions:
        ax.plot(p, t[col].to_numpy(), "o-", color=c, ls=ls, lw=2, label=lab)
    _log_xaxis(ax, p)
    ax.set_ylabel("mean loss (nats; lower = better predicted)")
    ax.set_title(
        "Fig 1 — mean loss by region vs scale (scaling-v0.5, val_cds)\n"
        "codon 1/2 + canonical splice GT/AG best-predicted; "
        "broad-splice window + deep intron lag (frozen)",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(OUT / "regions_meanloss.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "regions_meanloss.svg", bbox_inches="tight")
    plt.close(fig)

    # --- Subplot (collapsible): every strand-able region by gene strand ------
    # Same region colours as the main panel; + gene (sense) = solid/●,
    # − gene (antisense) = dashed/✕. Reads the per-(region, strand) breakdown.
    bs_rows = []
    for pp in sorted(glob.glob(f"{STAGE2}/*/val_cds_by_strand.parquet")):
        model = Path(pp).parent.name
        for r in pl.read_parquet(pp).iter_rows(named=True):
            bs_rows.append(
                {
                    "params_M": _params_millions(model),
                    "stratum": r["stratum"],
                    "gene_strand": r["gene_strand"],
                    "mean_loss": r["mean_loss"],
                }
            )
    bs = pl.DataFrame(bs_rows)
    strand_regions = [
        ("codon_1", "tab:blue"),
        ("codon_2", "tab:cyan"),
        ("codon_3", "tab:red"),
        ("codon_3_4fold", "tab:orange"),
        ("splicing", "tab:green"),
        ("splice_donor", "tab:olive"),
        ("splice_acceptor", "mediumseagreen"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for region, color in strand_regions:
        for strand, ls, mk in [("+", "-", "o"), ("-", "--", "x")]:
            sub = bs.filter(
                (pl.col("stratum") == region) & (pl.col("gene_strand") == strand)
            ).sort("params_M")
            if len(sub):
                ax.plot(
                    sub["params_M"].to_numpy(),
                    sub["mean_loss"].to_numpy(),
                    ls=ls,
                    marker=mk,
                    color=color,
                    lw=1.8,
                    ms=6,
                )
    _log_xaxis(ax, p)
    ax.set_ylabel("mean loss (nats; lower = better predicted)")
    ax.set_title(
        "Fig 1 (strand) — mean loss by region × gene strand\n"
        "+ gene = sense (solid ●), − gene = antisense (dashed ✕)",
        fontsize=11,
    )
    region_handles = [
        Line2D([0], [0], color=c, lw=3, label=r) for r, c in strand_regions
    ]
    strand_handles = [
        Line2D([0], [0], color="gray", ls="-", marker="o", label="+ gene (sense)"),
        Line2D([0], [0], color="gray", ls="--", marker="x", label="− gene (antisense)"),
    ]
    leg1 = ax.legend(
        handles=region_handles,
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.01, 0.72),
        title="region",
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=strand_handles,
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.01, 0.25),
        title="strand",
    )
    fig.savefig(OUT / "regions_meanloss_strand.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "regions_meanloss_strand.svg", bbox_inches="tight")
    plt.close(fig)
    print(
        f"[plot] wrote regions_meanloss(.svg) + regions_meanloss_strand(.svg) "
        f"({len(t)} models) → {OUT}"
    )


if __name__ == "__main__":
    main()
