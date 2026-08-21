from __future__ import annotations

from pathlib import Path

from launch import execution_environment, launch_command


def test_audit_launch_forwards_hf_and_wandb_and_self_terminates() -> None:
    command = launch_command(
        "audit",
        "a" * 40,
        1234,
        hf_repo_id="gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover",
        prior_cost_usd=10.232556777459978,
    )
    assert command[4] == "sky/audit.yaml"
    assert "HF_REPO_ID=gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover" in command
    assert "EXP479_PRIOR_COST_USD=10.232556777459978" in command
    assert command.count("--secret") == 2
    assert "HF_TOKEN" in command
    assert "WANDB_API_KEY" in command
    assert "--down" in command


def test_audit_environment_requires_both_credentials(monkeypatch: object) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-hf")  # type: ignore[attr-defined]
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb")  # type: ignore[attr-defined]
    environment = execution_environment("audit")
    assert environment["HF_TOKEN"] == "test-hf"
    assert environment["WANDB_API_KEY"] == "test-wandb"


def test_stability_launch_forwards_hf_and_wandb_and_self_terminates() -> None:
    command = launch_command(
        "stability",
        "b" * 40,
        2345,
        hf_repo_id="gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover",
        prior_cost_usd=12.0,
    )
    assert command[4] == "sky/stability.yaml"
    assert "HF_REPO_ID=gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover" in command
    assert "EXP479_PRIOR_COST_USD=12.0" in command
    assert command.count("--secret") == 2
    assert "HF_TOKEN" in command
    assert "WANDB_API_KEY" in command
    assert "--down" in command


def test_dependency_launch_forwards_hf_and_wandb_and_self_terminates() -> None:
    command = launch_command(
        "dependency",
        "c" * 40,
        3456,
        hf_repo_id="gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover",
        prior_cost_usd=23.0,
        retry_until_up=True,
    )
    assert command[4] == "sky/dependency.yaml"
    assert "HF_REPO_ID=gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover" in command
    assert "EXP479_PRIOR_COST_USD=23.0" in command
    assert command.count("--secret") == 2
    assert "HF_TOKEN" in command
    assert "WANDB_API_KEY" in command
    assert "--retry-until-up" in command
    assert "--down" in command


def test_dependency_environment_requires_both_credentials(monkeypatch: object) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-hf")  # type: ignore[attr-defined]
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb")  # type: ignore[attr-defined]
    environment = execution_environment("dependency")
    assert environment["HF_TOKEN"] == "test-hf"
    assert environment["WANDB_API_KEY"] == "test-wandb"


def test_calibration_launch_is_single_arm_secret_forwarding_and_self_terminating() -> None:
    command = launch_command(
        "calibration",
        "d" * 40,
        4567,
        hf_repo_id="gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover",
        prior_cost_usd=24.7340,
    )
    assert command[4] == "sky/calibration.yaml"
    assert "HF_REPO_ID=gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover" in command
    assert "EXP479_PRIOR_COST_USD=24.734" in command
    assert command.count("--secret") == 2
    assert "HF_TOKEN" in command
    assert "WANDB_API_KEY" in command
    assert "--down" in command


def test_calibration_environment_requires_both_credentials(monkeypatch: object) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-hf")  # type: ignore[attr-defined]
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb")  # type: ignore[attr-defined]
    environment = execution_environment("calibration")
    assert environment["HF_TOKEN"] == "test-hf"
    assert environment["WANDB_API_KEY"] == "test-wandb"


def test_calibration_sky_stage_has_no_learning_rate_sweep() -> None:
    stage = Path("sky/calibration.yaml").read_text(encoding="utf-8")
    assert stage.count("uv run --locked exp479 causal-calibration") == 1
    assert "3e-6" not in stage
    assert "1e-5" not in stage
    assert "3e-5" not in stage


def test_longrun_launch_forwards_only_wandb_and_self_terminates() -> None:
    command = launch_command(
        "longrun",
        "e" * 40,
        5678,
        prior_cost_usd=25.26241970350875,
        retry_until_up=True,
    )
    assert command[4] == "sky/longrun.yaml"
    assert "EXP479_PRIOR_COST_USD=25.26241970350875" in command
    assert command.count("--secret") == 1
    assert "WANDB_API_KEY" in command
    assert "HF_TOKEN" not in command
    assert not any(value.startswith("HF_REPO_ID=") for value in command)
    assert "--retry-until-up" in command
    assert "--down" in command


def test_longrun_environment_requires_wandb_but_not_hf(
    monkeypatch: object,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)  # type: ignore[attr-defined]
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb")  # type: ignore[attr-defined]
    environment = execution_environment("longrun")
    assert environment["WANDB_API_KEY"] == "test-wandb"
    assert "HF_TOKEN" not in environment


def test_longrun_sky_stage_is_one_retained_configuration_without_hf() -> None:
    stage = Path("sky/longrun.yaml").read_text(encoding="utf-8")
    assert stage.count("uv run --locked exp479 causal-longrun") == 1
    assert "WANDB_API_KEY" in stage
    assert "HF_TOKEN" not in stage
    assert "HF_REPO_ID" not in stage
    assert "causal-longrun-lr1e-5" in stage
    assert "finalize-local" in stage
