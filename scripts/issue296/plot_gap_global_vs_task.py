"""Figure 2 (issue #296): LL gap — global vs task-specific stratum — per variant class.

Three panels (missense / synonymous / splicing). Each overlays the **global**
`all_token` gap and the **matched** stratum gap (missense↔codon_12,
synonymous↔codon_3, splicing↔splicing) on the left axis, with that class's VEP
AUPRC (evals_v2 mendelian_traits, minus_llr_avg) on the right axis — so you can
see which gap follows the class AUPRC. The conserved-vs-non-conserved codon_12
loss decomposition (plot_codon12_loss.py) is the collapsible companion.

Globs the per-model Stage-2 stratum parquets + pulls per-model AUPRC; auto-extends.
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
METRICS = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"

# (class AUPRC subset, matched stratum, panel title)
CLASSES = [
    ("missense_variant", "codon_12", "missense"),
    ("synonymous_variant", "codon_3_4fold", "synonymous"),
    ("splicing", "splicing", "splicing"),
]


def _params_millions(model: str) -> float:
    m = re.search(r"p(\d+(?:\.\d+)?)([MB])", model)
    assert m, model
    return float(m.group(1)) * (1000.0 if m.group(2) == "B" else 1.0)


def _auprc(model: str, subset: str) -> float:
    df = pl.read_parquet(f"{METRICS}/{model}/mendelian_traits.parquet")
    r = df.filter(
        (pl.col("subset") == subset) & (pl.col("score_type") == "minus_llr_avg")
    )
    assert len(r) == 1, (model, subset, len(r))
    return float(r["value"][0])


def _collect() -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    rows, models = [], []
    for p in sorted(glob.glob(f"{STAGE2}/*/val_cds_stratum_ll_gap.parquet")):
        model = Path(p).parent.name
        models.append(model)
        d = {r["stratum"]: r["gap"] for r in pl.read_parquet(p).iter_rows(named=True)}
        d["params_M"] = _params_millions(model)
        rows.append(d)
    assert rows, f"no stratum parquets under {STAGE2}"
    gaps = pl.DataFrame(rows).sort("params_M")
    order = np.argsort([_params_millions(m) for m in models])
    models = [models[i] for i in order]
    auprc = {
        subset: np.array([_auprc(m, subset) for m in models])
        for subset, _, _ in CLASSES
    }
    return gaps, auprc


def _log_xaxis(ax, p: np.ndarray) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(p)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticklabels([str(int(x)) for x in p])
    ax.set_xlabel("parameters (M, log scale)")


def main() -> None:
    gaps, auprc = _collect()
    p = gaps["params_M"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (subset, matched, title) in zip(axes, CLASSES):
        ax.plot(
            p,
            gaps["all_token"].to_numpy(),
            "^--",
            color="tab:orange",
            lw=2,
            label="global (all_token) gap",
        )
        ax.plot(
            p,
            gaps[matched].to_numpy(),
            "s-",
            color="tab:blue",
            lw=2,
            label=f"matched ({matched}) gap",
        )
        ax.set_ylabel("LL gap (conserved − non-conserved)")
        _log_xaxis(ax, p)
        ax2 = ax.twinx()
        ax2.plot(
            p, auprc[subset], "o:", color="black", lw=2, label=f"{title} VEP AUPRC"
        )
        ax2.set_ylabel(f"{title} AUPRC")
        ax.set_title(f"{title}  (matched: {matched})", fontsize=11)
        lns = ax.get_lines() + ax2.get_lines()
        ax.legend(lns, [ln.get_label() for ln in lns], loc="upper left", fontsize=8)

    fig.suptitle(
        "Fig 2 — LL gap: global vs task-specific stratum, vs scale "
        "(scaling-v0.5, val_cds) — does the matched gap follow the class AUPRC?",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "gap_global_vs_task.png", dpi=130)
    fig.savefig(OUT / "gap_global_vs_task.svg")
    print(f"[plot] wrote gap_global_vs_task(.svg) ({len(p)} models) → {OUT}")


if __name__ == "__main__":
    main()
