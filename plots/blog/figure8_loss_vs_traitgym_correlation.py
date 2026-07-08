"""Figure 8: loss ↔ VEP AUPRC correlation across training, by scale — all four worlds.

Redo of Eric's Fig 8 (1×2: mean-ρ bars + per-trait ρ heatmap) on the **new** eval,
for every metric-world (Mendelian/SGE × zero-shot-LLR/probe). Per scale: the scored
HF-checkpoint trajectory of each trait's AUPRC (``_worlds`` readers) and the
validation loss at those steps (interp of the vendored per-step ``eval/loss``
history), then Spearman ρ(−loss, AUPRC) — negated so "loss-drop tracks AUPRC-gain"
reads positive.

**Cross-scale fairness (issue #365):** the offline eval only scores the saved HF
checkpoints, and the scales have uneven cadences (128M late-only, 1B early+late, 4B
wider). A ρ over each scale's *own* steps isn't comparable across scales (e.g. 1B
missense ρ = −0.02 full vs −0.71 on the shared window). So ρ is computed on the
**intersection of scored steps common to all rendered scales**. Full-cadence *shapes*
live in Fig 7.

Run:  uv run python -m plots.blog.figure8_loss_vs_traitgym_correlation
Out:  plots/output/blog/figure8_loss_vs_traitgym_correlation__{world}.{svg,png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from plots.blog._style.figure_style import DIVERGING_CMAP, figsize
from plots.blog._style.savefig import save_figure
from plots.blog._worlds import WORLDS, World

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


def _loss_at(run_name: str, steps: np.ndarray) -> np.ndarray:
    """Validation loss at ``steps`` — linear interp of the vendored eval/loss history
    on log10(step) (no extrapolation beyond the history's [0, 215573] range)."""
    h = pd.read_csv(HISTORY)
    h = h[(h["run_name"] == run_name) & (h["metric"] == "eval/loss")].sort_values(
        "step"
    )
    xs = np.log10(h["step"].clip(lower=1).to_numpy())
    return np.interp(np.log10(np.clip(steps, 1, None)), xs, h["value"].to_numpy())


def _scored(
    stem: str, steps: tuple[int, ...], world: World
) -> dict[int, dict[str, float]]:
    """{step: {subset: auprc}} for the steps that are scored (skips unscored)."""
    out: dict[int, dict[str, float]] = {}
    for s in steps:
        try:
            df = world.read(f"{stem}-step-{s}")
        except (LookupError, FileNotFoundError, OSError):
            continue
        out[s] = {r["subset"]: r["value"] for r in df.iter_rows(named=True)}
    return out


def correlations(world: World) -> tuple[dict[str, float], int]:
    """Per-scale mean Spearman ρ(−loss, AUPRC) over the world's traits, computed on
    the **shared-step intersection** (the same quantity ``build`` renders as the bar
    panel). Returns ``({scale_label: mean_rho}, n_common_steps)``. Exposed for
    verification/reporting so the correlation claim can be cited with a number."""
    trait_keys = [t for t, _, _ in world.traits]
    scored = {label: _scored(stem, steps, world) for label, _run, stem, steps in SCALES}
    present = [
        (label, run) for label, run, _stem, _steps in SCALES if len(scored[label]) >= 2
    ]
    if not present:
        return {}, 0
    common = sorted(set.intersection(*(set(scored[label]) for label, _ in present)))
    out: dict[str, float] = {}
    for label, run in present:
        loss = _loss_at(run, np.array(common, dtype=float))
        rhos = []
        for t in trait_keys:
            y = np.array([scored[label][int(s)].get(t, np.nan) for s in common])
            ok = np.isfinite(y)
            if ok.sum() >= 3:
                rhos.append(spearmanr(-loss[ok], y[ok]).statistic)
        out[label] = float(np.nanmean(rhos)) if rhos else float("nan")
    return out, len(common)


def build(world: World) -> None:
    trait_keys = [t for t, _, _ in world.traits]
    trait_labels = [lab for _, lab, _ in world.traits]

    scored = {label: _scored(stem, steps, world) for label, _run, stem, steps in SCALES}
    present = [
        (label, run) for label, run, _stem, _steps in SCALES if len(scored[label]) >= 2
    ]
    if not present:
        print(f"figure8: no scored scales for world {world.key} — skipping")
        return
    common = sorted(set.intersection(*(set(scored[label]) for label, _ in present)))

    labels = [label for label, _ in present]
    rho = np.full((len(labels), len(trait_keys)), np.nan)
    for i, (label, run) in enumerate(present):
        loss = _loss_at(run, np.array(common, dtype=float))
        for j, t in enumerate(trait_keys):
            y = np.array([scored[label][int(s)].get(t, np.nan) for s in common])
            ok = np.isfinite(y)
            if ok.sum() >= 3:
                rho[i, j] = spearmanr(-loss[ok], y[ok]).statistic
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
    ax_hm.set_xticks(range(len(trait_keys)))
    ax_hm.set_xticklabels(trait_labels, rotation=30, ha="right", fontsize=9)
    ax_hm.set_yticks(y)
    ax_hm.set_yticklabels([])
    for i in range(len(labels)):
        for j in range(len(trait_keys)):
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

    span = f"{common[0] // 1000}k–{common[-1] // 1000}k" if common else "n/a"
    fig.suptitle(
        f"Loss ↔ VEP AUPRC correlation · {world.label} · shared {span} steps",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, OUTPUT_DIR, f"figure8_loss_vs_traitgym_correlation__{world.key}")


def build_all() -> None:
    for world in WORLDS.values():
        build(world)


if __name__ == "__main__":
    build_all()
