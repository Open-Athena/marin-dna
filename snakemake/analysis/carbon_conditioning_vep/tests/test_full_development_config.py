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
    assert config["analysis"]["output_namespace"] == "full_development"
    assert "results/full_development/metrics/Carbon-3B/far_wrong.parquet" in task
    assert "snakemake all" not in task
    assert "2400s" in task
