from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parents[1]


def test_far_wrong_task_adds_only_the_fungal_scoring_arm() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config/full_development.yaml").read_text())
    task = (PROJECT_ROOT / "sky/run-gh200-far-wrong.yaml").read_text()

    assert config["experiment_id"] == "CARBON-SC-003-FULL"
    assert config["analysis"]["conditions"] == [
        "untagged",
        "correct",
        "far_wrong",
    ]
    assert config["conditions"]["far_wrong"] == "fungi"
    assert config["metrics"]["comparisons"]["far_wrong_minus_untagged"] == [
        "far_wrong",
        "untagged",
    ]
    assert config["metrics"]["comparisons"]["far_wrong_minus_correct"] == [
        "far_wrong",
        "correct",
    ]
    assert config["analysis"]["output_namespace"] == "full_development"
    assert "results/full_development/metrics/Carbon-3B/far_wrong.parquet" in task
    assert "snakemake all" not in task
    assert "2400s" in task


def test_historical_two_arm_task_has_exact_targets_and_pinned_inputs() -> None:
    task = (PROJECT_ROOT / "sky/run-gh200-full.yaml").read_text()
    stage = (PROJECT_ROOT / "sky/stage-gh200-full.sh").read_text()

    expected_targets = [
        "scores/Carbon-3B/untagged.parquet",
        "metrics/Carbon-3B/untagged.parquet",
        "scores/Carbon-3B/correct.parquet",
        "metrics/Carbon-3B/correct.parquet",
        "paired/Carbon-3B/correct_minus_untagged.parquet",
    ]
    for target in expected_targets:
        assert f"results/full_development/{target}" in task
    assert "snakemake all" not in task
    assert "far_wrong" not in task
    assert "4200s" in task

    snapshot = "snapshots/carbon-conditioning-vep-full-three-arm-20260820"
    assert snapshot in stage
    assert '"$snapshot_prefix/windows/mendelian.parquet"' in stage
    assert '"$snapshot_prefix/windows/exclusions.parquet"' in stage
    assert "200980cdc6d96dce810b3d5119d7e811fcd615b77437bb512b6a0f0de5bbafe5" in stage
    assert "c09520428a35a1b8d15e03b7afec65039c925ed8556ec04ce509e2735dd1f3b3" in stage
    assert "3cbaa4ae48043fa1aa1932220717a02cc597c7bbfb8e750acd9b2b18301a6b04" in stage
    assert '"$canonical_prefix/windows/mendelian.parquet"' not in stage
