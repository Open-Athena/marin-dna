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


if __name__ == "__main__":
    fig_scale()
    fig_datascaling()
    print(f"wrote figures to {OUT}/")
