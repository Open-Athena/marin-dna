"""Figure for issue #296 — stratified LL gap vs missense AUPRC across scale.

Two panels:
  A) missense VEP AUPRC (peaks at 128M) vs the codon_12 and all_token LL gaps
     (both rise; codon_12 saturates at the peak) — the main hypothesis.
  B) splice-site mean loss by gene strand — CDS-primed sites improve with scale,
     intron-primed sites are frozen.

Reads the local Stage-2 scaling parquets; writes PNG + SVG to plots/output/issue296/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter  # noqa: E402

SCRATCH = Path("scratch/issue296")
OUT = Path("plots/output/issue296")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def _log_xaxis(ax, p: np.ndarray) -> None:
    """Show exactly the model param values on a log x-axis (no decade clutter)."""
    ax.xaxis.set_major_locator(FixedLocator(list(p)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticklabels([str(int(x)) for x in p])


def main() -> None:
    g = pl.read_parquet(SCRATCH / "scaling_gap.parquet").sort("params_M")
    ll = pl.read_parquet(SCRATCH / "scaling_meanloss.parquet").sort("params_M")
    p = np.array(g["params_M"].to_list())
    auprc = np.array(g["missense_auprc"].to_list())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Panel A: the main hypothesis ---------------------------------------
    cod = np.array(g["codon_12"].to_list())
    allt = np.array(g["all_token"].to_list())
    axA.set_xscale("log")
    axA.plot(p, auprc, "o-", color="black", lw=2.2, label="missense VEP AUPRC (left)")
    axA.axvline(128, color="grey", ls=":", lw=1)
    axA.annotate(
        "AUPRC peak\n(128M)",
        (128, auprc[2]),
        textcoords="offset points",
        xytext=(6, -28),
        fontsize=9,
        color="grey",
    )
    axA.set_ylabel("missense VEP AUPRC")
    axA.set_xlabel("parameters (M, log scale)")

    axA2 = axA.twinx()
    axA2.plot(
        p,
        cod,
        "s--",
        color="tab:blue",
        lw=2,
        label=f"codon_12 gap (r={_pearson(cod, auprc):+.2f})",
    )
    axA2.plot(
        p,
        allt,
        "^--",
        color="tab:orange",
        lw=2,
        label=f"all_token gap (r={_pearson(allt, auprc):+.2f})",
    )
    axA2.set_ylabel("LL gap (conserved − non-conserved)")

    _log_xaxis(axA, p)
    axA.set_title(
        "A. Restricted gap tracks missense better\n"
        "(codon_12 saturates at the peak; all_token keeps rising)",
        fontsize=11,
    )
    lines = axA.get_lines()[:1] + axA2.get_lines()
    axA.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=9)

    # --- Panel B: strand / splice priming -----------------------------------
    axB.set_xscale("log")
    series = [
        ("splice_donor_plus", "CDS-primed donor (+)", "tab:green", "-"),
        ("splice_acceptor_minus", "CDS-primed acceptor (−)", "tab:olive", "-"),
        ("splice_donor_minus", "intron-primed donor (−)", "tab:red", "--"),
        ("splice_acceptor_plus", "intron-primed acceptor (+)", "tab:purple", "--"),
    ]
    for col, lab, c, lsty in series:
        axB.plot(
            p,
            np.array(ll[col].to_list()),
            marker="o",
            color=c,
            ls=lsty,
            lw=2,
            label=lab,
        )
    _log_xaxis(axB, p)
    axB.set_xlabel("parameters (M, log scale)")
    axB.set_ylabel("mean loss (nats; lower = better)")
    axB.set_title(
        "B. Splice sites: CDS-primed improve, intron-primed frozen", fontsize=11
    )
    axB.legend(loc="center left", fontsize=9)

    fig.suptitle(
        "Issue #296 — stratified LL gap / mean loss vs scale "
        "(scaling-v0.5, val_cds, sub-0.5B)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "scaling.png", dpi=130)
    fig.savefig(OUT / "scaling.svg")
    print(f"[plot] wrote {OUT}/scaling.png + scaling.svg")


if __name__ == "__main__":
    main()
