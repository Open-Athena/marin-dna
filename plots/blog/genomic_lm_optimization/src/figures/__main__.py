"""Rebuild every figure in the set.

Usage:
    uv run src/figures/__main__.py
    # or, equivalently, from the `src` directory:
    uv run python -m figures
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable so `figures.*` and `utils.*` resolve when this file is
# run directly as a script (its own dir, not `src`, is otherwise on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figures import data  # noqa: E402
from figures.figure1_lr_transfer import build as figure1  # noqa: E402
from figures.figure2_beta2_epsilon_transfer import build as figure2  # noqa: E402
from figures.figure3_region_hyper_transfer import build as figure3  # noqa: E402
from figures.figure4_loss_scaling import build as figure4  # noqa: E402
from figures.figure5_params_vs_vep_auprc import build as figure5  # noqa: E402
from figures.figure6_loss_vs_vep_auprc import build as figure6  # noqa: E402

# Figures 7 & 8 (loss_vs_traitgym_curves / _correlation) are wandb-fed standalone
# scripts — run separately, not part of this CSV-only build.
from figures.figure9_upstream_mix_auprc import build as figure9  # noqa: E402
from figures.figure10_lineage_vep_trajectory import build as figure10  # noqa: E402
from figures.figure11_leaderboard_heatmap import build as figure11  # noqa: E402
from utils.figure_style import palette  # noqa: E402


def main() -> None:
    transfer_df = data.load_transfer()
    scaling_results = data.load_scaling_results()
    scaling_history = data.load_scaling_history()
    mixture_df = data.load_mixture()
    mixture_history_df = data.load_mixture_history()
    leaderboard_df = data.load_leaderboard()

    # Per-sweep palettes so each figure's colors span the full viridis range.
    # Sharing a single palette across both compresses the transfer scales (only 3
    # of 8 positions) into a narrow color band that's hard to distinguish.
    transfer_params = sorted({int(p) for p in transfer_df["params"].dropna().unique()})
    transfer_palette = palette(transfer_params)
    scaling_params = sorted(
        {int(p) for p in scaling_results["params"].dropna().unique()}
    )
    scaling_palette = palette(scaling_params)

    figure1(transfer_df, transfer_palette, transfer_params)
    figure2(transfer_df, transfer_palette, transfer_params)
    figure3(transfer_df, transfer_palette, transfer_params)
    figure4(scaling_history, scaling_results, scaling_palette)
    figure5()
    figure6(scaling_results, scaling_palette)
    figure9(mixture_df)
    figure10(mixture_df, mixture_history_df)
    figure11(leaderboard_df)


if __name__ == "__main__":
    main()
