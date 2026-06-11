"""issue #302 — iteration 20: the inverse of iter17's drop-common cut. Restrict the Mendelian
missense NEGATIVES to progressively higher MAF cutoffs and watch the AUPRC-vs-scale shape.

iter17 removed the common benigns and the degradation stayed. The inverse view (GB's ask):
keep ONLY the more-common benigns — negatives with AF > {0.1%, 1%, 5%, 10%} — and plot the
missense AUPRC across the ladder, one line per cutoff. All Mendelian missense negatives already
have AF > 0.1% (so >0.1% = the full/default negative set, 5220); >1%/>5%/>10% thin to the
common tail (1842/872/631).

Expectation from iter17: the hard negatives (driving the degradation) are the rarer-but-
conserved benigns, so restricting to common benigns should WEAKEN the decline (and raise the
level — common benigns are easier; iter17's AUROC pos-vs-common 0.89 > pos-vs-rare 0.82).
Score = minus_llr_avg. CPU; reads/writes S3.

CAVEAT: higher cutoff -> fewer negatives -> higher prevalence (0.10->0.48) -> higher AUPRC
baseline. Compare the across-scale SHAPE within each line, NOT the levels across lines.

Run: OMP_NUM_THREADS=2 uv run --group genome-s3 python scripts/issue302/iter20_maf_cutoff_gradient.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score as ap

SCORES_S3 = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/scores"
OUT_S3 = "s3://oa-bolinas/analysis/issue302/probe_out"
OUT = Path("scratch/issue302/figs")
LADDER = [
    ("scaling-v0.5-h640-p46M-step-215573", 46),
    ("scaling-v0.5-h768-p76M-step-215573", 76),
    ("scaling-v0.5-h896-p128M-step-215573", 128),
    ("scaling-v0.5-h1152-p255M-step-215573", 255),
    ("scaling-v0.5-h1408-p476M-step-215573", 476),
    ("scaling-v0.5-h1920-p1B-step-215573", 1120),
    ("scaling-v0.5-h2432-p2B-step-215573", 2270),
    ("scaling-v0.5-h2944-p4B-step-215573", 4020),
]
CUTS = [(0.001, ">0.1% (all negs)"), (0.01, ">1%"), (0.05, ">5%"), (0.10, ">10%")]


def main() -> None:
    rows = []
    ncounts = {}
    for sdir, params in LADDER:
        m = (
            pl.read_parquet(f"{SCORES_S3}/{sdir}/mendelian_traits.parquet")
            .filter(pl.col("subset") == "missense_variant")
            .with_columns(
                pl.col("AF").fill_null(0.0),
                (-(pl.col("llr_fwd") + pl.col("llr_rc")) / 2).alias("mll"),
            )
        )
        pos = m.filter(pl.col("label") == 1)["mll"].to_numpy()
        for cut, lab in CUTS:
            neg = m.filter((pl.col("label") == 0) & (pl.col("AF") > cut))[
                "mll"
            ].to_numpy()
            y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            rows.append(
                {
                    "params": params,
                    "cut": lab,
                    "auprc": float(ap(y, np.r_[pos, neg])),
                    "n_neg": len(neg),
                    "prev": len(pos) / (len(pos) + len(neg)),
                }
            )
            ncounts[lab] = len(neg)
        print(
            f"{params:>5}M | "
            + "  ".join(
                f"{lab}={[r for r in rows if r['params'] == params and r['cut'] == lab][0]['auprc']:.3f}"
                for _, lab in CUTS
            )
        )

    res = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.write_parquet("scratch/issue302/maf_cutoff_gradient.parquet")
    c = res.to_pandas()

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    cmap = plt.get_cmap("plasma")
    for i, (_cut, lab) in enumerate(CUTS):
        s = c[c.cut == lab].sort_values("params")
        prev = s["prev"].iloc[0]
        ax.plot(
            s["params"],
            s["auprc"],
            "o-",
            color=cmap(i / (len(CUTS) - 1)),
            lw=2.5,
            label=f"neg AF {lab}  (n={ncounts[lab]}, prev={prev:.2f})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("params (M, log)")
    ax.set_ylabel("Mendelian missense AUPRC (minus_llr)")
    ax.set_title(
        "Restricting negatives to higher-MAF benigns weakens the degradation\n(the hard, over-called benigns are the rarer-but-conserved ones)"
    )
    ax.legend(fontsize=8, title="negative set (more stringent ↓)")
    ax.grid(alpha=0.3)
    # annotate per-line peak->4B drop
    note = "peak→4B drop:  " + "  ".join(
        f"{lab.split()[0]} {c[c.cut == lab]['auprc'].max() - c[(c.cut == lab) & (c.params == 4020)]['auprc'].iloc[0]:+.3f}"
        for _, lab in CUTS
    )
    ax.text(
        0.5,
        -0.18,
        note,
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color="dimgray",
    )
    fig.tight_layout()
    fig.savefig(OUT / "maf_cutoff_gradient.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "maf_cutoff_gradient.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'maf_cutoff_gradient'}.{{png,svg}}")

    import s3fs

    fs = s3fs.S3FileSystem()
    for p in (
        "scratch/issue302/maf_cutoff_gradient.parquet",
        str(OUT / "maf_cutoff_gradient.png"),
        str(OUT / "maf_cutoff_gradient.svg"),
    ):
        fs.put(p, f"{OUT_S3.replace('s3://', '')}/{Path(p).name}")
    print(f"  uploaded -> {OUT_S3}")


if __name__ == "__main__":
    main()
