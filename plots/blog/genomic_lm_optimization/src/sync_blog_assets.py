"""Copy explicitly selected generated SVGs into the blog asset directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ASSET_NAMES = (
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
OUTPUT_DIR = REPO_ROOT / "plots" / "output" / "blog" / "genomic_lm_optimization"
CURRENT_LEADERBOARD_OUTPUT_DIR = REPO_ROOT / "plots" / "output" / "blog"
BLOG_ASSET_DIR = (
    REPO_ROOT
    / "blog"
    / "genomic-lm-optimization"
    / "static"
    / "assets"
    / "images"
    / "blog"
    / "genomic-lm-optimization"
)


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
        print(f"Copied {source} -> {destination}")


if __name__ == "__main__":
    main()
