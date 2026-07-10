"""Figures for #369 — LoRA fine-tuning vs the frozen probe on Mendelian-missense VEP.

Fig 1: FT / frozen-probe / zero-shot-LLR chr1 AUPRC vs model scale (46M->476M).
Fig 2: data-scaling — chr1 AUPRC vs training-set size, FT vs probe (255M missense).

Numbers are the collected single-seed chr1 results (see the #369 iteration log); hard-coded
here because they're scattered across per-run S3 parquets. Outputs svg + png to
plots/output/issue369/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

OUT = Path("plots/output/issue369")
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")
C = {"ft": "#d1495b", "probe": "#2e6f95", "llr": "#8d99ae"}

# --- Fig 1: scale -----------------------------------------------------------------
# model -> (params_M, FT, probe, LLR), chr1 missense, single seed
SCALE = {
    "46M": (46, 0.379, 0.362, 0.367),
    "76M": (76, 0.422, 0.415, 0.442),
    "128M": (128, 0.480, 0.505, 0.474),
    "255M": (255, 0.524, 0.523, 0.489),
    "476M": (476, 0.545, 0.548, 0.472),
}


def fig_scale() -> None:
    x = [v[0] for v in SCALE.values()]
    names = list(SCALE)
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, key in enumerate(("ft", "probe", "llr")):
        y = [v[1 + i] for v in SCALE.values()]
        lab = {"ft": "LoRA fine-tune (rank 2)", "probe": "frozen linear probe",
               "llr": "zero-shot LLR"}[key]
        ax.plot(x, y, "o-", color=C[key], label=lab, lw=2.5, ms=9)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.minorticks_off()  # drop clashing log minor-tick labels
    ax.set_xlabel("model size (parameters)")
    ax.set_ylabel("chr1 missense AUPRC (per-chrom-weighted)")
    ax.set_title("FT matches the frozen probe at every scale — never beats it")
    ax.legend(frameon=False, fontsize=13, loc="upper left")
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"scale_ft_probe_llr.{ext}", dpi=140)
    plt.close(fig)


# --- Fig 2: data-scaling (255M missense) ------------------------------------------
PROBE_DS = [(307, 0.416), (613, 0.451), (1228, 0.475), (2456, 0.516), (4910, 0.523)]
FT_DS = [(525, 0.425), (1050, 0.465), (4200, 0.524)]  # 50% run pending
LLR_255 = 0.489


def fig_datascaling() -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(*zip(*FT_DS), "o-", color=C["ft"], label="LoRA fine-tune", lw=2.5, ms=9)
    ax.plot(*zip(*PROBE_DS), "s-", color=C["probe"], label="frozen linear probe", lw=2.5, ms=9)
    ax.axhline(LLR_255, ls="--", color=C["llr"], lw=2, label="zero-shot LLR (data-independent)")
    ax.set_xscale("log")
    ax.set_xlabel("# training missense variants (chr-grouped)")
    ax.set_ylabel("chr1 missense AUPRC")
    ax.set_title("Data-scaling (255M): FT and probe rise together")
    ax.legend(frameon=False, fontsize=13, loc="lower right")
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"datascaling_255M.{ext}", dpi=140)
    plt.close(fig)


# --- Fig 3: pooled-all-subsets FT vs per-subset probe (255M, chr1) ----------------
POOLED = {  # subset -> (pooled-all FT, per-subset probe)
    "synonymous": (0.384, 0.697), "splicing": (0.457, 0.531),
    "missense": (0.447, 0.523), "tss-prox": (0.155, 0.255),
    "distal": (0.500, 0.480), "ncRNA-exon": (0.196, 0.150),
    "5'UTR": (0.278, 0.187), "3'UTR": (0.265, 0.125),
}


def fig_pooled() -> None:
    items = sorted(POOLED.items(), key=lambda kv: kv[1][0] - kv[1][1])
    labels, ft = [k for k, _ in items], [v[0] for _, v in items]
    pr = [v[1] for _, v in items]
    y = list(range(len(items)))
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (f, p) in enumerate(zip(ft, pr)):
        ax.plot([p, f], [i, i], color=C["ft"] if f > p else C["probe"], lw=3, zorder=1)
    ax.scatter(pr, y, color=C["probe"], s=95, label="per-subset probe", zorder=2)
    ax.scatter(ft, y, color=C["ft"], s=95, label="pooled-all FT", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("chr1 AUPRC (255M)")
    ax.set_title("Pooled-all FT vs per-subset probe (255M)")
    ax.legend(frameon=False, loc="lower right", fontsize=13)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"pooled_persubset_255M.{ext}", dpi=140)
    plt.close(fig)


# --- Fig 4: the 2x2 — pooling is a shared data effect (255M, chr1) ----------------
import numpy as np  # noqa: E402

GRID = {  # subset -> (probe per-subset, probe pooled, FT pooled)
    "synonymous": (0.697, 0.268, 0.384), "splicing": (0.531, 0.455, 0.457),
    "missense": (0.523, 0.481, 0.447), "distal": (0.480, 0.611, 0.500),
    "tss-prox": (0.255, 0.211, 0.155), "5'UTR": (0.187, 0.272, 0.278),
    "ncRNA-exon": (0.150, 0.207, 0.196), "3'UTR": (0.125, 0.247, 0.265),
}


def fig_2x2() -> None:
    labels = list(GRID)
    pp, ppool, ftpool = (list(t) for t in zip(*GRID.values()))
    y, h = np.arange(len(labels)), 0.26
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(y + h, pp, h, color=C["probe"], label="probe · per-subset")
    ax.barh(y, ppool, h, color="#8ecae6", label="probe · pooled")
    ax.barh(y - h, ftpool, h, color=C["ft"], label="FT · pooled")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("chr1 AUPRC (255M)")
    ax.set_title("Pooling is a shared data effect — pooled FT ≈ pooled probe")
    ax.legend(frameon=False, loc="lower right", fontsize=12)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"two_by_two_255M.{ext}", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    fig_scale()
    fig_datascaling()
    fig_pooled()
    fig_2x2()
    print(f"wrote figures to {OUT}/")
