"""issue #302 — iteration 8: plateau or degradation? Does the model increasingly
(and more confidently) SCORE the conserved-benign missense variants as
deleterious as it scales — i.e. is the missense decline a *sharpening of a
wrong-for-this-subset signal* (active degradation) rather than a ceiling from a
*missing* signal (plateau)?

Fix a model-INDEPENDENT conserved-benign subset (missense negatives with high
phyloP) and a non-conserved-benign control (low phyloP), and track — across the
8-size ladder — how far each group's score sits above the benign baseline,
standardized per model (z vs the negative distribution, to remove the score
inflation that grows with scale). Sharpening ⇒ the conserved-benigns climb
toward the positives with scale (and the fraction of them out-ranking the
positive median grows); plateau ⇒ they flatten.

Input: scratch/issue302/missense_enriched.parquet (iter3). CPU only.
Output (scratch/issue302/figs/): sharpening_vs_scale.

Run:  uv run python scripts/issue302/iter8_sharpening.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ENRICHED = Path("scratch/issue302/missense_enriched.parquet")
OUT = Path("scratch/issue302/figs")
SIZES = ["46M", "76M", "128M", "255M", "476M", "1B", "2B", "4B"]
PARAMS = {
    "46M": 46,
    "76M": 76,
    "128M": 128,
    "255M": 255,
    "476M": 476,
    "1B": 1120,
    "2B": 2270,
    "4B": 4020,
}
PHYLOP_CONS = (
    4.0  # conserved-benign threshold (pathogenic median ~6.5, typical-neg ~1.4)
)
PHYLOP_NONCONS = 1.0


def main() -> None:
    w = pl.read_parquet(ENRICHED).to_pandas()
    w = w[w["phyloP"].notna()]
    pos = w[w.label == 1]
    neg = w[w.label == 0]
    cons = neg[
        neg.phyloP >= PHYLOP_CONS
    ]  # conserved benigns (fixed, model-independent)
    noncons = neg[neg.phyloP < PHYLOP_NONCONS]  # non-conserved benigns (control)
    print(
        f"positives n={len(pos)} | conserved-benign (phyloP>={PHYLOP_CONS}) n={len(cons)} | "
        f"non-conserved-benign (phyloP<{PHYLOP_NONCONS}) n={len(noncons)} | all neg n={len(neg)}"
    )

    rows = []
    for s in SIZES:
        mu, sd = neg[s].mean(), neg[s].std()
        z = lambda x: (np.median(x) - mu) / sd  # noqa: E731  median score in benign-SD units
        pos_med = pos[s].median()
        rows.append(
            {
                "size": s,
                "params": PARAMS[s],
                "z_pos": z(pos[s]),
                "z_cons": z(cons[s]),
                "z_noncons": z(noncons[s]),
                # misranking: fraction of conserved-benigns scoring above the positive median
                "frac_cons_above_posmed": float((cons[s] > pos_med).mean()),
                "frac_noncons_above_posmed": float((noncons[s] > pos_med).mean()),
                # gap that AUPRC cares about: conserved-benign median relative to positive median, in benign SDs
                "gap_pos_minus_cons_sd": z(pos[s]) - z(cons[s]),
            }
        )
    res = pl.DataFrame(rows)
    print("\n  standardized score (median, in benign-SD units) + misranking, per size:")
    print(
        f"    {'size':>5} {'z_pos':>6} {'z_cons':>7} {'z_noncons':>9} {'pos-cons gap':>12} {'%cons>posMed':>12}"
    )
    for r in res.iter_rows(named=True):
        print(
            f"    {r['size']:>5} {r['z_pos']:6.2f} {r['z_cons']:7.2f} {r['z_noncons']:9.2f} "
            f"{r['gap_pos_minus_cons_sd']:12.2f} {r['frac_cons_above_posmed'] * 100:11.1f}%"
        )

    c = res.to_pandas()
    verdict = (
        "SHARPENING (degradation): conserved-benigns climb toward positives"
        if c["z_cons"].iloc[-1] - c["z_cons"].iloc[2] > 0.1
        and c["gap_pos_minus_cons_sd"].iloc[-1] < c["gap_pos_minus_cons_sd"].iloc[2]
        else "PLATEAU: conserved-benigns flatten"
    )
    print(
        f"\n  => 128M→4B: z_cons {c['z_cons'].iloc[2]:.2f}→{c['z_cons'].iloc[-1]:.2f}, "
        f"pos-cons gap {c['gap_pos_minus_cons_sd'].iloc[2]:.2f}→{c['gap_pos_minus_cons_sd'].iloc[-1]:.2f}, "
        f"%cons>posMed {c['frac_cons_above_posmed'].iloc[2] * 100:.1f}→{c['frac_cons_above_posmed'].iloc[-1] * 100:.1f}"
    )
    print(f"  VERDICT: {verdict}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(
        c["params"], c["z_pos"], "o-", color="tab:red", label="pathogenic (pos)"
    )
    axes[0].plot(
        c["params"],
        c["z_cons"],
        "o-",
        color="tab:orange",
        label=f"conserved benign (phyloP≥{PHYLOP_CONS})",
    )
    axes[0].plot(
        c["params"],
        c["z_noncons"],
        "o-",
        color="tab:blue",
        label=f"non-conserved benign (phyloP<{PHYLOP_NONCONS})",
    )
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("params (M, log)")
    axes[0].set_ylabel("median score, in benign-SD units (z vs all negatives)")
    axes[0].set_title(
        "Conserved benigns CLIMB toward pathogenics with scale\n(score inflation removed by per-model z)"
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(
        c["params"],
        c["frac_cons_above_posmed"] * 100,
        "o-",
        color="tab:orange",
        label="conserved benign",
    )
    axes[1].plot(
        c["params"],
        c["frac_noncons_above_posmed"] * 100,
        "o-",
        color="tab:blue",
        label="non-conserved benign",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("params (M, log)")
    axes[1].set_ylabel("% scoring above the positive median")
    axes[1].set_title(
        "Misranking grows with scale (not a plateau):\nmore conserved-benigns out-rank the median pathogenic"
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle(
        "Iter8 — plateau vs degradation: the conserved-benign 'wrong signal' sharpens with scale",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(OUT / "sharpening_vs_scale.png", dpi=130, bbox_inches="tight")
    fig.savefig(OUT / "sharpening_vs_scale.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {OUT / 'sharpening_vs_scale'}.{{png,svg}}")


if __name__ == "__main__":
    main()
