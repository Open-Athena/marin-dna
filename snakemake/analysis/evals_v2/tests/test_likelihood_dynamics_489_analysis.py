from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from marin_dna_evals.likelihood_dynamics_489_analysis import (
    CONTROL_TERMS,
    _lowest_fraction_mask,
    _rank_bins,
    analyze_likelihood_dynamics_489,
)
from marin_dna_evals.likelihood_dynamics_489_figure import (
    plot_conservation_auprc_489,
    plot_controlled_conservation_489,
    plot_future_loss_deciles_489,
    plot_score_distributions_489,
    plot_selection_jaccard_489,
    plot_trajectory_groups_489,
)

CHECKPOINTS = [f"checkpoint-{index}" for index in range(5)]
CUMULATIVE_TOKENS = [100, 200, 300, 400, 500]
REGIONS = ["cds", "upstream", "downstream", "ncrna", "enhancer"]


def _write_synthetic_inputs(
    tmp_path: Path,
) -> tuple[
    dict[tuple[str, str], Path],
    dict[tuple[str, str], Path],
]:
    atom_paths: dict[tuple[str, str], Path] = {}
    manifest_paths: dict[tuple[str, str], Path] = {}
    window_size = 20
    n_windows = 4
    for region_index, region in enumerate(REGIONS):
        row_index = np.repeat(np.arange(n_windows, dtype=np.int32), window_size)
        target_pos = np.tile(np.arange(window_size, dtype=np.int16), n_windows)
        token_index = row_index.astype(np.int64) * window_size + target_pos
        conserved = (target_pos + row_index + region_index) % 4 == 0
        repeat = target_pos == 3
        base_nll = (
            1.8
            + 0.03 * target_pos
            + 0.04 * row_index
            + 0.02 * region_index
            - 0.25 * conserved
        )
        learning_rate = 0.015 * (1 + (target_pos % 5))
        for checkpoint_order, (checkpoint, cumulative_tokens) in enumerate(
            zip(CHECKPOINTS, CUMULATIVE_TOKENS, strict=True)
        ):
            nll = base_nll - checkpoint_order * learning_rate
            entropy = (
                1.2
                + 0.01 * target_pos
                - 0.10 * conserved
                - checkpoint_order * learning_rate / 2
            )
            frame = pd.DataFrame(
                {
                    "token_index": token_index,
                    "row_index": row_index,
                    "region": region,
                    "chrom": np.where(row_index % 2 == 0, "1", "2"),
                    "genomic_pos": row_index.astype(np.int64) * 10_000 + target_pos,
                    "target_pos": target_pos,
                    "is_conserved": conserved,
                    "is_repeat": repeat,
                    "is_ambiguous": False,
                    "is_scorable": True,
                    "window_gc": 0.35 + 0.05 * row_index,
                    "kmer7_nll": 1.1 + 0.02 * target_pos,
                    "nll": nll.astype(np.float32),
                    "entropy_4nuc": entropy.astype(np.float32),
                }
            )
            atom_path = tmp_path / f"{checkpoint}-{region}.parquet"
            manifest_path = tmp_path / f"{checkpoint}-{region}.manifest.json"
            frame.to_parquet(atom_path, index=False)
            manifest_path.write_text(
                json.dumps(
                    {
                        "artifact_schema_version": "v1",
                        "scope": "full",
                        "dataset": {
                            "region": region,
                            "hf_repo": f"marin-dna/{region}",
                            "hf_revision": "a" * 40,
                        },
                        "checkpoint": {
                            "name": checkpoint,
                            "order": checkpoint_order,
                            "cumulative_tokens": cumulative_tokens,
                        },
                        "atom_manifest": {
                            "token_identity": [
                                "region",
                                "row_index",
                                "target_pos",
                            ],
                            "n_positions": len(frame),
                            "n_scorable": len(frame),
                        },
                    }
                )
                + "\n"
            )
            atom_paths[(checkpoint, region)] = atom_path
            manifest_paths[(checkpoint, region)] = manifest_path
    return atom_paths, manifest_paths


def _run_analysis(
    tmp_path: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    atom_paths, manifest_paths = _write_synthetic_inputs(tmp_path)
    return analyze_likelihood_dynamics_489(
        atom_paths,
        manifest_paths,
        checkpoints=CHECKPOINTS,
        cumulative_tokens=CUMULATIVE_TOKENS,
        regions=REGIONS,
        primary_start=2,
        primary_end_exclusive=18,
        block_bp=10_000,
        bootstrap_replicates=20,
        top_fraction=0.25,
        n_bins=4,
        seed=489,
    )


def test_ranked_masks_have_exact_sizes_and_stable_ties() -> None:
    values = np.asarray([2.0, 1.0, 1.0, 3.0, 0.5, 4.0, 2.0, 5.0])
    token_id = np.asarray([7, 6, 5, 4, 3, 2, 1, 0])
    selected = _lowest_fraction_mask(values, token_id, fraction=0.25)
    assert selected.sum() == 2
    assert selected.tolist() == [False, False, True, False, True, False, False, False]

    bins = _rank_bins(values, token_id, n_bins=4)
    assert np.bincount(bins, minlength=5)[1:].tolist() == [2, 2, 2, 2]
    assert bins[4] == 1
    assert bins[2] == 1


def test_full_analysis_contract(tmp_path: Path) -> None:
    frames, manifest = _run_analysis(tmp_path)

    assert set(frames) == {
        "population",
        "conservation_auprc",
        "trajectory_groups",
        "selection_jaccard",
        "future_loss_deciles",
        "distributions",
        "controlled_contrasts",
    }
    assert len(frames["population"]) == 5
    assert len(frames["conservation_auprc"]) == 6 * 2 * 5
    assert len(frames["selection_jaccard"]) == 6 * 2 * 5
    assert len(frames["future_loss_deciles"]) == 6 * 4 * 2 * 4
    assert len(frames["distributions"]) == 6 * 5 * 2 * 2
    assert len(frames["controlled_contrasts"]) == 5 * 5 * 2 * len(CONTROL_TERMS)
    assert frames["conservation_auprc"]["auprc"].between(0, 1).all()
    assert frames["selection_jaccard"]["jaccard"].between(0, 1).all()
    assert set(frames["trajectory_groups"]["group"]) <= {
        "low_to_low",
        "low_to_high",
        "high_to_low",
        "high_to_high",
    }
    assert set(frames["controlled_contrasts"]["term"]) == set(CONTROL_TERMS)
    assert manifest["validation"]["passed"] is True
    assert manifest["validation"]["n_input_cells"] == 25


def test_all_figures_render_from_analysis_outputs(tmp_path: Path) -> None:
    frames, _ = _run_analysis(tmp_path)
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path

    outputs = {
        "auprc": tmp_path / "auprc.svg",
        "trajectory": tmp_path / "trajectory.svg",
        "jaccard": tmp_path / "jaccard.svg",
        "future": tmp_path / "future.svg",
        "distributions": tmp_path / "distributions.svg",
        "controlled": tmp_path / "controlled.svg",
    }
    plot_conservation_auprc_489(paths["conservation_auprc"], outputs["auprc"])
    plot_trajectory_groups_489(paths["trajectory_groups"], outputs["trajectory"])
    plot_selection_jaccard_489(paths["selection_jaccard"], outputs["jaccard"])
    plot_future_loss_deciles_489(paths["future_loss_deciles"], outputs["future"])
    plot_score_distributions_489(paths["distributions"], outputs["distributions"])
    plot_controlled_conservation_489(
        paths["controlled_contrasts"],
        outputs["controlled"],
    )
    for output in outputs.values():
        assert output.stat().st_size > 2_000
