"""Issue-configurable paired specialist home-rank trajectory analysis."""

import hashlib

from marin_dna_evals.home_rank import (
    first_persistent_checkpoint,
    joint_auprc_home_rank_probability,
)


def get_home_rank_inputs(_wildcards):
    cfg = config["home_rank"]
    return [
        f"results/scores/{cfg['model_name_template'].format(arm=arm, step=step)}/"
        f"{cfg['dataset']}.parquet"
        for step in cfg["checkpoints"]
        for arm in cfg["arms"]
    ]


rule compute_home_rank:
    input:
        get_home_rank_inputs,
    output:
        trajectory="results/analysis/home_rank_trajectory.parquet",
        persistence="results/analysis/home_rank_persistence.parquet",
    params:
        cfg=lambda _wc: config["home_rank"],
    run:
        cfg = params.cfg
        arms = tuple(cfg["arms"])
        checkpoints = tuple(cfg["checkpoints"])
        home_by_subset = {
            subset: arm
            for arm, subsets in cfg["home_subsets"].items()
            for subset in subsets
        }
        assert set(cfg["home_subsets"]) == set(
            arms
        ), "home_subsets keys must exactly match home_rank arms"
        assert len(home_by_subset) == sum(
            len(subsets) for subsets in cfg["home_subsets"].values()
        ), "each home subset must map to exactly one arm"
        dataset_cfg = get_dataset_config(cfg["dataset"])
        score_type = f"{dataset_cfg['score_protocol']}_avg"
        assert score_type == cfg["score_type"], (
            f"configured score_type {cfg['score_type']!r} does not match "
            f"dataset protocol {score_type!r}"
        )
        transform = SCORE_PROTOCOLS[dataset_cfg["score_protocol"]]
        excluded_subsets = tuple(
            dataset_cfg.get("exclude_complete_match_groups_with_subsets", [])
        )
        rows = []
        for step in checkpoints:
            frames = {}
            for arm in arms:
                model = cfg["model_name_template"].format(arm=arm, step=step)
                path = f"results/scores/{model}/{cfg['dataset']}.parquet"
                frames[arm] = pd.read_parquet(path)
            reference = frames[arms[0]]
            identity_columns = list(REQUIRED_VARIANT_COLUMNS)
            for arm in arms[1:]:
                pd.testing.assert_frame_equal(
                    reference[identity_columns].reset_index(drop=True),
                    frames[arm][identity_columns].reset_index(drop=True),
                    check_dtype=True,
                    obj=f"variant rows for {arms[0]} and {arm} at step {step}",
                )
            keep_index = reference.index
            if excluded_subsets:
                filtered = exclude_complete_match_groups_with_subsets(
                    reference, excluded_subsets
                )
                keep_index = filtered.index
            reference = reference.loc[keep_index].reset_index(drop=True)
            arm_scores = {
                arm: transform(
                    (
                        frames[arm].loc[keep_index, "llr_fwd"].to_numpy()
                        + frames[arm].loc[keep_index, "llr_rc"].to_numpy()
                    )
                    / 2
                )
                for arm in arms
            }
            present_subsets = set(reference["subset"])
            missing_subsets = set(home_by_subset) - present_subsets
            assert (
                not missing_subsets
            ), f"home subsets absent after exclusions: {sorted(missing_subsets)}"
            for subset, home_arm in home_by_subset.items():
                subset_mask = reference["subset"].eq(subset).to_numpy()
                subset_dataset = reference.loc[
                    subset_mask, ["label", "match_group"]
                ].reset_index(drop=True)
                subset_scores = {
                    arm: np.asarray(score)[subset_mask]
                    for arm, score in arm_scores.items()
                }
                seed_material = f"{cfg['bootstrap_seed']}:{step}:{subset}".encode()
                cell_seed = int.from_bytes(
                    hashlib.sha256(seed_material).digest()[:8], "little"
                )
                result = joint_auprc_home_rank_probability(
                    subset_dataset,
                    subset_scores,
                    home_arm,
                    n_bootstrap=cfg["n_bootstrap"],
                    rng=cell_seed,
                )
                row = {
                    "dataset": cfg["dataset"],
                    "split": config["split"],
                    "score_type": score_type,
                    "subset": subset,
                    "home_arm": home_arm,
                    "checkpoint": step,
                    "point_winners": "|".join(result["point_winners"]),
                    "home_is_point_winner": result["home_is_point_winner"],
                    "home_rank_first_probability": result[
                        "home_rank_first_probability"
                    ],
                    "n_bootstrap": result["n_bootstrap"],
                    "n_bootstrap_valid": result["n_bootstrap_valid"],
                    "n_groups": result["n_groups"],
                    "n_rows": result["n_rows"],
                    "n_pos": result["n_pos"],
                }
                row.update(
                    {f"{arm}_auprc": result["point_auprc"][arm] for arm in arms}
                )
                rows.append(row)
        trajectory = pd.DataFrame(rows)
        trajectory.to_parquet(output.trajectory, index=False)
        persistence_rows = []
        for subset, home_arm in home_by_subset.items():
            subset_trajectory = trajectory.loc[
                trajectory["subset"].eq(subset)
            ].sort_values("checkpoint")
            first_checkpoint = first_persistent_checkpoint(
                subset_trajectory["checkpoint"].tolist(),
                subset_trajectory["home_rank_first_probability"].tolist(),
                threshold=cfg["persistence_threshold"],
                consecutive=cfg["persistence_consecutive"],
            )
            terminal = subset_trajectory.iloc[-1]
            persistence_rows.append(
                {
                    "dataset": cfg["dataset"],
                    "split": config["split"],
                    "score_type": score_type,
                    "subset": subset,
                    "home_arm": home_arm,
                    "first_persistent_checkpoint": first_checkpoint,
                    "persistence_threshold": cfg["persistence_threshold"],
                    "persistence_consecutive": cfg["persistence_consecutive"],
                    "terminal_checkpoint": int(terminal["checkpoint"]),
                    "terminal_home_rank_first_probability": terminal[
                        "home_rank_first_probability"
                    ],
                    "terminal_home_is_point_winner": terminal[
                        "home_is_point_winner"
                    ],
                    "terminal_point_winners": terminal["point_winners"],
                }
            )
        persistence = pd.DataFrame(persistence_rows)
        persistence["first_persistent_checkpoint"] = persistence[
            "first_persistent_checkpoint"
        ].astype("Int64")
        persistence.to_parquet(output.persistence, index=False)
        print(
            f"[evals_v2] home-rank trajectory: {len(trajectory)} cells; "
            f"persistence summary: {len(persistence)} subsets"
        )
