from __future__ import annotations

from launch import execution_environment, launch_command


def test_preflight_launch_is_commit_pinned_lambda_gh200_and_self_terminating() -> None:
    commit = "a" * 40
    command = launch_command("preflight", commit, 1234)
    assert command[:5] == ["sky", "launch", "-c", "dna-exp479-gh200", "sky/preflight.yaml"]
    assert command[command.index("--git-ref") + 1] == commit
    assert "EXP479_INSTANCE_START_UNIX=1234" in command
    assert "--down" in command
    assert "--yes" in command


def test_pilot_launch_forwards_private_publication_secrets() -> None:
    command = launch_command("pilot", "a" * 40, 1234)
    assert command[4] == "sky/pilot.yaml"
    assert "HF_REPO_ID=marin-dna/marin-dna-exp479-mntp-m5.1" in command
    assert command.count("--secret") == 2
    assert "HF_TOKEN" in command
    assert "WANDB_API_KEY" in command


def test_pilot_launch_can_resume_into_private_spillover() -> None:
    command = launch_command(
        "pilot",
        "a" * 40,
        1234,
        hf_repo_id="person/spillover",
        resume_hf_repo_id="org/original",
        checkpoint_upload_steps=(400, 800),
        prior_cost_usd=2.7,
    )
    assert "HF_REPO_ID=person/spillover" in command
    assert "RESUME_HF_REPO_ID=org/original" in command
    assert "CHECKPOINT_UPLOAD_STEPS=400 800" in command
    assert "EXP479_PRIOR_COST_USD=2.7" in command


def test_pilot_dry_run_is_explicitly_forwarded_to_sky() -> None:
    command = launch_command("pilot", "a" * 40, 1234, dry_run=True)
    assert command[-1] == "--dryrun"


def test_non_pilot_execution_environment_does_not_require_secrets(
    monkeypatch: object,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)  # type: ignore[attr-defined]
    monkeypatch.delenv("WANDB_API_KEY", raising=False)  # type: ignore[attr-defined]
    environment = execution_environment("preflight")
    assert "HF_TOKEN" not in environment
    assert "WANDB_API_KEY" not in environment


def test_diagnostics_launch_only_forwards_hugging_face_secret() -> None:
    command = launch_command(
        "diagnostics",
        "a" * 40,
        1234,
        hf_repo_id="person/spillover",
        prior_cost_usd=9.56,
    )
    assert command[4] == "sky/diagnostics.yaml"
    assert "HF_REPO_ID=person/spillover" in command
    assert "EXP479_PRIOR_COST_USD=9.56" in command
    assert command.count("--secret") == 1
    assert "HF_TOKEN" in command
    assert "WANDB_API_KEY" not in command
    assert "--down" in command


def test_diagnostics_environment_only_requires_hugging_face(
    monkeypatch: object,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")  # type: ignore[attr-defined]
    monkeypatch.delenv("WANDB_API_KEY", raising=False)  # type: ignore[attr-defined]
    environment = execution_environment("diagnostics")
    assert environment["HF_TOKEN"] == "test-token"
    assert "WANDB_API_KEY" not in environment
