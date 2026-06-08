"""Issue #296 — codon_12 mean loss, conserved vs non-conserved, across scale.

For the codon_12 stratum (1st+2nd codon positions), plot the model's mean loss on
**phyloP-conserved** vs **non-conserved** bases as two lines vs parameter count.
mean loss = −mean log p, so conserved = −LL_upper, non-conserved = −LL_lower; the
vertical gap between the lines is exactly the codon_12 LL gap.

Globs the per-model Stage-2 stratum parquets, so it auto-extends as more
checkpoints are processed. Writes PNG + SVG to plots/output/issue296/.
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
STRATUM = "codon_12"


def _params_millions(model: str) -> float:
    m = re.search(r"p(\d+(?:\.\d+)?)([MB])", model)
    assert m, model
    return float(m.group(1)) * (1000.0 if m.group(2) == "B" else 1.0)


def main() -> None:
    rows = []
    for p in sorted(glob.glob(f"{STAGE2}/*/val_cds_stratum_ll_gap.parquet")):
        model = Path(p).parent.name
        r = pl.read_parquet(p).filter(pl.col("stratum") == STRATUM)
        assert len(r) == 1, (model, len(r))
        rows.append(
            {
                "params_M": _params_millions(model),
                "loss_conserved": -float(r["LL_upper"][0]),
                "loss_nonconserved": -float(r["LL_lower"][0]),
            }
        )
    assert rows, f"no stratum parquets under {STAGE2}"
    t = pl.DataFrame(rows).sort("params_M")
    p = np.array(t["params_M"].to_list())
    cons = np.array(t["loss_conserved"].to_list())
    noncons = np.array(t["loss_nonconserved"].to_list())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xscale("log")
    ax.plot(p, noncons, "o-", color="tab:gray", lw=2, label="non-conserved (lowercase)")
    ax.plot(
        p, cons, "o-", color="tab:blue", lw=2, label="conserved (uppercase / phyloP)"
    )
    ax.fill_between(p, cons, noncons, color="tab:blue", alpha=0.10)
    # annotate the gap at the largest model
    ax.annotate(
        f"LL gap = {noncons[-1] - cons[-1]:.2f}",
        (p[-1], (cons[-1] + noncons[-1]) / 2),
        textcoords="offset points",
        xytext=(-105, -4),
        fontsize=9,
        color="tab:blue",
    )

    ax.xaxis.set_major_locator(FixedLocator(list(p)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticklabels([str(int(x)) for x in p])
    ax.set_xlabel("parameters (M, log scale)")
    ax.set_ylabel("mean loss (nats; lower = better predicted)")
    ax.set_title(
        "Issue #296 — codon_12 (1st+2nd codon pos): mean loss\n"
        "conserved vs non-conserved, vs scale (scaling-v0.5, val_cds)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "codon12_loss.png", dpi=130)
    fig.savefig(OUT / "codon12_loss.svg")
    print(f"[plot] wrote {OUT}/codon12_loss.png + .svg  ({len(t)} models)")


if __name__ == "__main__":
    main()
