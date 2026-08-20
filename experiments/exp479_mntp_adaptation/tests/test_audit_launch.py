from __future__ import annotations

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
    )
    assert command[4] == "sky/dependency.yaml"
    assert "HF_REPO_ID=gonzalobenegas/marin-dna-exp479-mntp-m5.1-spillover" in command
    assert "EXP479_PRIOR_COST_USD=23.0" in command
    assert command.count("--secret") == 2
    assert "HF_TOKEN" in command
    assert "WANDB_API_KEY" in command
    assert "--down" in command


def test_dependency_environment_requires_both_credentials(monkeypatch: object) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-hf")  # type: ignore[attr-defined]
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb")  # type: ignore[attr-defined]
    environment = execution_environment("dependency")
    assert environment["HF_TOKEN"] == "test-hf"
    assert environment["WANDB_API_KEY"] == "test-wandb"
