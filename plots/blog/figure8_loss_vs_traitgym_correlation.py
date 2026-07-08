"""Figure 8 (M·LLR): loss ↔ VEP AUPRC correlation across training, by scale (new eval).

Redo of Eric's Fig 8 (1×2: mean-ρ bars + per-trait ρ heatmap) on the **new** offline
eval. For each scale we take the scored HF-checkpoint trajectory of per-trait AUPRC
(``blog_metrics``) and the validation loss at those steps (PCHIP/interp of the
vendored per-step ``eval/loss`` history), then Spearman ρ(−loss, AUPRC) — negated so
"loss-drop tracks AUPRC-gain" reads positive.

**Cross-scale fairness (issue #365):** the offline eval only scores the saved HF
checkpoints, and the scales have uneven cadences (128M late-only, 1B early+late, 4B
wider). A ρ computed over each scale's *own* steps is not comparable across scales
(e.g. 1B missense ρ = −0.02 full vs −0.71 on the shared window). So ρ here is computed
on the **intersection of scored steps common to all rendered scales** — an apples-to-
apples comparison. The full-cadence trajectory *shapes* live in Fig 7.

Run:  uv run python -m plots.blog.figure8_loss_vs_traitgym_correlation
Out:  plots/output/blog/figure8_loss_vs_traitgym_correlation__mendelian_llr.{svg,png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from marin_dna.pipelines.evals.blog_metrics import read_llr_metrics
from plots.blog._style.figure_style import DIVERGING_CMAP, figsize
from plots.blog._style.savefig import save_figure

HISTORY = Path(__file__).resolve().parent / "data" / "parameter_scaling_history.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "blog"

# (label, wandb run_name (loss history key), evals_v2 stem, intended HF steps).
SCALES: tuple[tuple[str, str, str, tuple[int, ...]], ...] = (
    (
        "128M",
        "dna-bolinas-scaling-v0.5-h896-p128M",
        "scaling-v0.5-h896-p128M",
        (160000, 170000, 180000, 190000, 200000, 210000, 215573),
    ),
    (
        "1B",
        "dna-bolinas-scaling-v0.5-h1920-p1B",
        "scaling-v0.5-h1920-p1B",
        (
            10000,
            20000,
            140000,
            150000,
            160000,
            170000,
            180000,
            190000,
            200000,
            210000,
            215573,
        ),
    ),
    (
        "4B",
        "dna-bolinas-scaling-v0.5-h2944-p4B",
        "scaling-v0.5-h2944-p4B",
        (
            80000,
            90000,
            100000,
            110000,
            120000,
            130000,
            140000,
            150000,
            160000,
            170000,
            180000,
            190000,
            200000,
            210000,
            215573,
        ),
    ),
)
# Six variant subsets (Eric's Fig 8 set), display order = heatmap columns.
TRAITS: tuple[tuple[str, str], ...] = (
    ("missense_variant", "missense"),
    ("tss_proximal", "promoter"),
    ("5_prime_UTR_variant", "5' UTR"),
    ("3_prime_UTR_variant", "3' UTR"),
    ("splicing", "splicing"),
    ("synonymous_variant", "synonymous"),
)


def _loss_at(run_name: str, steps: np.ndarray) -> np.ndarray:
    """Validation loss at ``steps`` — linear interp of the vendored eval/loss history
    on log10(step) (no extrapolation beyond the history's [0, 215573] range)."""
    h = pd.read_csv(HISTORY)
    h = h[(h["run_name"] == run_name) & (h["metric"] == "eval/loss")].sort_values(
        "step"
    )
    xs = np.log10(h["step"].clip(lower=1).to_numpy())
    return np.interp(np.log10(np.clip(steps, 1, None)), xs, h["value"].to_numpy())


def _scored(stem: str, steps: tuple[int, ...]) -> dict[int, dict[str, float]]:
    """{step: {subset: auprc}} for the steps that are scored (skips unscored)."""
    out: dict[int, dict[str, float]] = {}
    for s in steps:
        try:
            df = read_llr_metrics(f"{stem}-step-{s}", "mendelian_traits")
        except (LookupError, FileNotFoundError, OSError):
            continue
        out[s] = {r["subset"]: r["value"] for r in df.iter_rows(named=True)}
    return out


def build() -> None:
    # Collect scored AUPRC per scale, then the intersection of scored steps.
    scored = {label: _scored(stem, steps) for label, _run, stem, steps in SCALES}
    present = [
        (label, run) for label, run, _stem, _steps in SCALES if len(scored[label]) >= 2
    ]
    if not present:
        print("figure8: no scored scales yet — skipping")
        return
    common = set.intersection(*(set(scored[label]) for label, _ in present))
    common = sorted(common)
    if len(common) < 3:
        print(
            f"figure8: intersection too small ({len(common)} steps) — rendering anyway"
        )

    labels = [label for label, _ in present]
    trait_keys = [t for t, _ in TRAITS]
    rho = np.full((len(labels), len(TRAITS)), np.nan)
    for i, (label, run) in enumerate(present):
        steps = np.array(common, dtype=float)
        loss = _loss_at(run, steps)
        for j, t in enumerate(trait_keys):
            y = np.array([scored[label][int(s)].get(t, np.nan) for s in common])
            if np.isfinite(y).sum() >= 3:
                rho[i, j] = spearmanr(
                    -loss[np.isfinite(y)], y[np.isfinite(y)]
                ).statistic
    mean_rho = np.nanmean(rho, axis=1)

    fig, (ax_bar, ax_hm) = plt.subplots(
        1, 2, figsize=figsize(12.0, 4.2), gridspec_kw={"width_ratios": [1, 2.3]}
    )
    y = np.arange(len(labels))
    ax_bar.barh(y, mean_rho, color="#7e8a45", edgecolor="k", linewidth=0.5)
    ax_bar.axvline(0, color="0.5", linewidth=0.8)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("mean ρ (−loss, AUPRC)")
    ax_bar.set_title("mean across variants", fontsize=10)
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)

    im = ax_hm.imshow(rho, cmap=DIVERGING_CMAP, vmin=-1, vmax=1, aspect="auto")
    ax_hm.set_xticks(range(len(TRAITS)))
    ax_hm.set_xticklabels(
        [lab for _, lab in TRAITS], rotation=30, ha="right", fontsize=9
    )
    ax_hm.set_yticks(y)
    ax_hm.set_yticklabels([])
    for i in range(len(labels)):
        for j in range(len(TRAITS)):
            if np.isfinite(rho[i, j]):
                ax_hm.text(
                    j,
                    i,
                    f"{rho[i, j]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(rho[i, j]) > 0.55 else "black",
                )
    ax_hm.set_title("per-variant ρ", fontsize=10)
    cb = fig.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.02)
    cb.set_label("Spearman ρ", fontsize=9)

    fig.suptitle(
        f"Loss ↔ VEP AUPRC correlation across training · shared {common[0] // 1000}k–{common[-1] // 1000}k steps",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, OUTPUT_DIR, "figure8_loss_vs_traitgym_correlation__mendelian_llr")


if __name__ == "__main__":
    build()
