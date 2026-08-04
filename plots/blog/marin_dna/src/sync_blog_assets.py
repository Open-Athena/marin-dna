"""Copy explicitly selected generated SVGs into the blog asset directory."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ASSET_NAMES = (
    "headline_cost_performance",
    "figure1_lr_transfer",
    "figure2_beta2_epsilon_transfer",
    "figure3_region_hyper_transfer",
    "figure4_loss_scaling",
    "figure5_params_vs_vep_auprc",
    "figure6_loss_vs_vep_auprc",
    "figure6b_marin_evo2_missense",
    "figure9_upstream_mix_auprc",
    "figure10_lineage_vep_trajectory",
    "figure16_offline_lineage_llr_prototype",
    "figure16_offline_lineage_probe_prototype",
    "figure11_leaderboard_heatmap",
    "figure11_leaderboard_heatmap__mendelian_llr",
    "figure11_leaderboard_heatmap__mendelian_probe",
)
CURRENT_LEADERBOARD_NAMES = frozenset(
    {
        "figure11_leaderboard_heatmap__mendelian_llr",
        "figure11_leaderboard_heatmap__mendelian_probe",
    }
)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[2]
REPO_SRC = REPO_ROOT / "src"
assert (REPO_SRC / "marin_dna" / "blog_figure_typography.py").is_file(), REPO_SRC
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from marin_dna.blog_figure_typography import sync_article_figure_width  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "plots" / "output" / "blog" / "marin_dna"
CURRENT_LEADERBOARD_OUTPUT_DIR = REPO_ROOT / "plots" / "output" / "blog"
BLOG_ASSET_DIR = (
    REPO_ROOT
    / "blog"
    / "marin-dna"
    / "static"
    / "assets"
    / "images"
    / "blog"
    / "marin-dna"
)
BLOG_ARTICLE = REPO_ROOT / "blog" / "marin-dna" / "content" / "blog" / "marin-dna.md"
ARTICLE_FIGURE_IDS = {
    "headline_cost_performance": "fig-cost-performance",
    "figure1_lr_transfer": "fig-learning-rate-transfer",
    "figure2_beta2_epsilon_transfer": "fig-adam-transfer",
    "figure3_region_hyper_transfer": "fig-region-hyperparameter-transfer",
    "figure4_loss_scaling": "fig-loss-scaling",
    "figure5_params_vs_vep_auprc": "fig-parameters-vs-vep",
    "figure6_loss_vs_vep_auprc": "fig-loss-vs-vep",
    "figure6b_marin_evo2_missense": "fig-missense-readout-scaling",
    "figure16_offline_lineage_llr_prototype": "fig-mixture-lineage-trajectories",
    "figure16_offline_lineage_probe_prototype": "fig-mixture-lineage-probe",
    "figure11_leaderboard_heatmap__mendelian_llr": "fig-mendelian-leaderboard",
    "figure11_leaderboard_heatmap__mendelian_probe": (
        "fig-mendelian-leaderboard-probe"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", choices=ASSET_NAMES)
    args = parser.parse_args()

    BLOG_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        source_dir = (
            CURRENT_LEADERBOARD_OUTPUT_DIR
            if name in CURRENT_LEADERBOARD_NAMES
            else OUTPUT_DIR
        )
        source = source_dir / f"{name}.svg"
        destination = BLOG_ASSET_DIR / f"{name}.svg"
        if not source.is_file():
            raise FileNotFoundError(f"generate {source} before syncing it")
        shutil.copyfile(source, destination)
        figure_id = ARTICLE_FIGURE_IDS.get(name)
        if figure_id is not None:
            frame_width = sync_article_figure_width(
                BLOG_ARTICLE, figure_id, destination
            )
            print(f"Synced {figure_id} to {frame_width:.1f}px")
        print(f"Copied {source} -> {destination}")

    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "python",
            "src/marin_dna/blog_workspace.py",
            "normalize-figures",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
