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
        ("splicing", "intronic splice site (≤20bp)", "tab:green", "-"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for col, lab, c, ls in regions:
        ax.plot(p, t[col].to_numpy(), "o-", color=c, ls=ls, lw=2, label=lab)
    _log_xaxis(ax, p)
    ax.set_ylabel("mean loss (nats; lower = better predicted)")
    ax.set_title(
        "Fig 1 — mean loss by region vs scale (scaling-v0.5, val_cds)\n"
        "codon 1/2 best-predicted (improve fastest); wobble higher; "
        "splice highest & ~flat",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "regions_meanloss.png", dpi=130)
    fig.savefig(OUT / "regions_meanloss.svg")
    plt.close(fig)

    # --- Subplot (collapsible): splice region by gene strand ----------------
    strand = [
        ("splice_donor_plus", "donor (+) — CDS-primed", "tab:green", "-"),
        ("splice_acceptor_minus", "acceptor (−) — CDS-primed", "tab:olive", "-"),
        ("splice_donor_minus", "donor (−) — intron-primed", "tab:red", "--"),
        ("splice_acceptor_plus", "acceptor (+) — intron-primed", "tab:purple", "--"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for col, lab, c, ls in strand:
        ax.plot(p, t[col].to_numpy(), "o", color=c, ls=ls, lw=2, label=lab)
    _log_xaxis(ax, p)
    ax.set_ylabel("mean loss (nats; lower = better predicted)")
    ax.set_title(
        "Fig 1 (strand) — splice-site mean loss by gene strand\n"
        "CDS-primed improve with scale; intron-primed frozen",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "regions_meanloss_strand.png", dpi=130)
    fig.savefig(OUT / "regions_meanloss_strand.svg")
    plt.close(fig)
    print(
        f"[plot] wrote regions_meanloss(.svg) + regions_meanloss_strand(.svg) "
        f"({len(t)} models) → {OUT}"
    )


if __name__ == "__main__":
    main()
