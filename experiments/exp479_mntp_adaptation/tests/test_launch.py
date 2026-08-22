from __future__ import annotations

from pathlib import Path

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


def test_loss_normalization_launch_uses_only_wandb_and_retained_artifacts() -> None:
    command = launch_command(
        "loss-normalization",
        "a" * 40,
        1234,
        prior_cost_usd=26.1237,
    )
    assert command[4] == "sky/loss-normalization.yaml"
    assert "EXP479_PRIOR_COST_USD=26.1237" in command
    assert command.count("--secret") == 1
    assert "WANDB_API_KEY" in command
    assert "HF_TOKEN" not in command
    assert "HF_REPO_ID=marin-dna/marin-dna-exp479-mntp-m5.1" not in command
    assert "--down" in command


def test_source_validation_launch_uses_public_hf_and_wandb_only() -> None:
    command = launch_command(
        "source-validation",
        "a" * 40,
        1234,
        prior_cost_usd=26.4688,
    )
    assert command[4] == "sky/source-validation.yaml"
    assert "EXP479_PRIOR_COST_USD=26.4688" in command
    assert command.count("--secret") == 1
    assert "WANDB_API_KEY" in command
    assert "HF_TOKEN" not in command
    assert "--down" in command


def test_mntp_longrun_launch_is_wandb_only_and_self_terminating() -> None:
    command = launch_command(
        "mntp-longrun",
        "a" * 40,
        1234,
        prior_cost_usd=28.307954,
        retry_until_up=True,
    )
    assert command[4] == "sky/mntp-longrun.yaml"
    assert "EXP479_PRIOR_COST_USD=28.307954" in command
    assert command.count("--secret") == 1
    assert "WANDB_API_KEY" in command
    assert "HF_TOKEN" not in command
    assert "--retry-until-up" in command
    assert "--down" in command


def test_mntp_longrun_environment_requires_wandb_but_not_hf(
    monkeypatch: object,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)  # type: ignore[attr-defined]
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb")  # type: ignore[attr-defined]
    environment = execution_environment("mntp-longrun")
    assert environment["WANDB_API_KEY"] == "test-wandb"
    assert "HF_TOKEN" not in environment


def test_mntp_longrun_sky_stage_runs_one_corrected_arm_without_hf() -> None:
    stage = Path("sky/mntp-longrun.yaml").read_text(encoding="utf-8")
    assert stage.count("uv run --locked exp479 mntp-longrun") == 1
    assert stage.count("uv run --locked pytest") == 1
    assert "WANDB_API_KEY" in stage
    assert "HF_TOKEN" not in stage
    assert "HF_REPO_ID" not in stage
    assert "mntp-longrun-lr1e-5-corrected" in stage
    assert "--vep-batch-size 1024" in stage
    assert "--dependency-batch-size 1024" in stage
    assert "finalize-local" in stage


def test_mntp_dependency_uses_only_retained_wandb_checkpoint() -> None:
    command = launch_command(
        "mntp-dependency",
        "a" * 40,
        1234,
        prior_cost_usd=32.289179,
        retry_until_up=True,
    )
    assert command[4] == "sky/mntp-dependency.yaml"
    assert "EXP479_PRIOR_COST_USD=32.289179" in command
    assert command.count("--secret") == 1
    assert "WANDB_API_KEY" in command
    assert "HF_TOKEN" not in command
    assert "HF_REPO_ID=marin-dna/marin-dna-exp479-mntp-m5.1" not in command
    assert "--down" in command

    stage = Path("sky/mntp-dependency.yaml").read_text(encoding="utf-8")
    assert stage.count("uv run --locked exp479 mntp-dependency") == 1
    assert stage.count("uv run --locked pytest") == 1
    assert "WANDB_API_KEY" in stage
    assert "HF_TOKEN" not in stage
    assert "finalize-local" in stage


def test_paired_nucleotide_gate_uses_a10_and_wandb_only() -> None:
    command = launch_command(
        "paired-nucleotide-gate",
        "a" * 40,
        1234,
        prior_cost_usd=32.289179,
        retry_until_up=True,
    )
    assert command[:5] == [
        "sky",
        "launch",
        "-c",
        "dna-exp479-a10",
        "sky/paired-nucleotide-gate.yaml",
    ]
    assert command.count("--secret") == 1
    assert "WANDB_API_KEY" in command
    assert "HF_TOKEN" not in command
    assert "--down" in command

    stage = Path("sky/paired-nucleotide-gate.yaml").read_text(encoding="utf-8")
    assert stage.count("uv run --locked exp479 paired-nucleotide-gate") == 1
    assert stage.count("uv run --locked pytest") == 1
    assert "accelerators: A10:1" in stage
    assert "mntp-dependency" not in stage


def test_attention_anneal_diagnostic_uses_one_a10g_and_no_training() -> None:
    command = launch_command(
        "attention-anneal-diagnostic",
        "a" * 40,
        1234,
        prior_cost_usd=32.473494,
        retry_until_up=True,
    )
    assert command[:5] == [
        "sky",
        "launch",
        "-c",
        "dna-exp479-anneal-a10",
        "sky/attention-anneal-diagnostic.yaml",
    ]
    assert "EXP479_INSTANCE_PRICE_PER_HOUR_USD=1.006" in command
    assert command.count("--secret") == 1
    assert "WANDB_API_KEY" in command
    assert "HF_TOKEN" not in command
    assert "--down" in command

    stage = Path("sky/attention-anneal-diagnostic.yaml").read_text(encoding="utf-8")
    assert stage.count("uv run --locked exp479 attention-anneal-diagnostic") == 1
    assert stage.count("uv run --locked pytest") == 1
    assert "cloud: aws" in stage
    assert "region: us-east-2" in stage
    assert "accelerators: A10G:1" in stage
    assert "use_spot: false" in stage
    assert "disk_size: 80" in stage
    assert "WANDB_API_KEY" in stage
    assert "HF_TOKEN" not in stage
    assert "lora-mntp" not in stage
    assert "vep" not in stage.lower()
    assert "nuc-dep" not in stage


def test_lora_mntp_uses_one_a10_wandb_and_no_downstream_evaluation() -> None:
    command = launch_command(
        "lora-mntp",
        "a" * 40,
        1234,
        prior_cost_usd=32.473494,
        retry_until_up=True,
    )
    assert command[:5] == [
        "sky",
        "launch",
        "-c",
        "dna-exp479-lora-a10",
        "sky/lora-mntp.yaml",
    ]
    assert "EXP479_INSTANCE_PRICE_PER_HOUR_USD=1.006" in command
    assert command.count("--secret") == 1
    assert "WANDB_API_KEY" in command
    assert "HF_TOKEN" not in command
    assert "--down" in command

    stage = Path("sky/lora-mntp.yaml").read_text(encoding="utf-8")
    assert stage.count("uv run --locked exp479 lora-mntp") == 1
    assert stage.count("uv run --locked pytest") == 1
    assert "cloud: aws" in stage
    assert "region: us-east-2" in stage
    assert "accelerators: A10G:1" in stage
    assert "use_spot: false" in stage
    assert "disk_size: 80" in stage
    assert "--batch-size 64" in stage
    assert "WANDB_API_KEY" in stage
    assert "HF_TOKEN" not in stage
    assert "vep" not in stage.lower()
    assert "nuc-dep" not in stage
