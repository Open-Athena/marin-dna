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


def test_localized_attention_diagnostic_uses_one_a10g_and_no_training() -> None:
    command = launch_command(
        "localized-attention-diagnostic",
        "a" * 40,
        1234,
        prior_cost_usd=34.995183,
        retry_until_up=True,
    )
    assert command[:5] == [
        "sky",
        "launch",
        "-c",
        "dna-exp479-localized-attention-a10",
        "sky/localized-attention-diagnostic.yaml",
    ]
    assert "EXP479_INSTANCE_PRICE_PER_HOUR_USD=1.006" in command
    assert command.count("--secret") == 1
    assert "WANDB_API_KEY" in command
    assert "HF_TOKEN" not in command
    assert "--down" in command

    stage = Path("sky/localized-attention-diagnostic.yaml").read_text(encoding="utf-8")
    assert stage.count("uv run --locked exp479 localized-attention-diagnostic") == 1
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


def test_bico_attention_diagnostic_uses_one_gh200_and_no_training() -> None:
    command = launch_command(
        "bico-attention-diagnostic",
        "a" * 40,
        1234,
        prior_cost_usd=40.354899,
        retry_until_up=True,
    )
    assert command[:5] == [
        "sky",
        "launch",
        "-c",
        "dna-exp479-bico-attention-gh200",
        "sky/bico-attention-diagnostic.yaml",
    ]
    assert "EXP479_INSTANCE_PRICE_PER_HOUR_USD=2.29" in command
    assert command.count("--secret") == 1
    assert "WANDB_API_KEY" in command
    assert "HF_TOKEN" not in command
    assert "--down" in command

    stage = Path("sky/bico-attention-diagnostic.yaml").read_text(encoding="utf-8")
    assert stage.count("uv run --locked exp479 bico-attention-diagnostic") == 1
    assert stage.count("uv run --locked pytest") == 1
    assert "cloud: lambda" in stage
    assert "region:" not in stage
    assert "accelerators: GH200:1" in stage
    assert "use_spot: false" in stage
    assert "disk_size: 80" in stage
    assert "WANDB_API_KEY" in stage
    assert "HF_TOKEN" not in stage
    assert "lora-mntp" not in stage
    assert "vep" not in stage.lower()
    assert "nuc-dep" not in stage
    assert "timeout --signal=TERM 3600" in stage
    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in stage


def test_bico_lora_uses_maximal_no_accumulation_gh200_batch() -> None:
    command = launch_command(
        "bico-lora-mntp",
        "a" * 40,
        1234,
        prior_cost_usd=40.354899,
        retry_until_up=True,
    )
    assert command[:5] == [
        "sky",
        "launch",
        "-c",
        "dna-exp479-bico-lora-gh200",
        "sky/bico-lora-mntp.yaml",
    ]
    assert "EXP479_INSTANCE_PRICE_PER_HOUR_USD=2.29" in command
    assert command.count("--secret") == 1
    assert "WANDB_API_KEY" in command
    assert "HF_TOKEN" not in command
    assert "--down" in command

    stage = Path("sky/bico-lora-mntp.yaml").read_text(encoding="utf-8")
    assert stage.count("uv run --locked exp479 bico-attention-diagnostic") == 1
    assert stage.count("uv run --locked exp479 bico-lora-preflight") == 1
    assert stage.count("uv run --locked exp479 bico-lora-mntp") == 1
    assert stage.count("uv run --locked pytest") == 1
    assert "cloud: lambda" in stage
    assert "accelerators: GH200:1" in stage
    assert "for candidate in 1024 512 256 128 64 32 16 8" in stage
    assert "while [ $((upper - lower)) -gt 1 ]" in stage
    assert 'run_candidate "$selected"' in stage
    assert '--batch-size "$selected"' in stage
    assert "accumulate" not in stage.lower()
    assert "WANDB_API_KEY" in stage
    assert "HF_TOKEN" not in stage
    assert "vep" not in stage.lower()
    assert "nuc-dep" not in stage
    assert "timeout --signal=TERM 12000" in stage


def test_bico_lora_resume_rechecks_only_selected_batch_before_training() -> None:
    command = launch_command(
        "bico-lora-resume",
        "a" * 40,
        1234,
        prior_cost_usd=40.85832745128643,
        retry_until_up=True,
    )
    assert command[:5] == [
        "sky",
        "launch",
        "-c",
        "dna-exp479-bico-lora-resume-gh200",
        "sky/bico-lora-resume.yaml",
    ]
    assert "EXP479_INSTANCE_PRICE_PER_HOUR_USD=2.29" in command
    assert command.count("--secret") == 1
    assert "WANDB_API_KEY" in command
    assert "HF_TOKEN" not in command

    stage = Path("sky/bico-lora-resume.yaml").read_text(encoding="utf-8")
    assert stage.count("uv run --locked pytest") == 1
    assert stage.count("uv run --locked exp479 bico-lora-preflight") == 1
    assert stage.count("uv run --locked exp479 bico-lora-mntp") == 1
    assert "bico-attention-diagnostic" not in stage
    assert stage.count("--batch-size 94") == 3
    assert "accumulate" not in stage.lower()
    assert "cloud: lambda" in stage
    assert "accelerators: GH200:1" in stage
    assert "HF_TOKEN" not in stage
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


def test_lora_reload_audit_uses_retained_wandb_artifacts_only() -> None:
    command = launch_command(
        "lora-reload-audit",
        "a" * 40,
        1234,
        prior_cost_usd=34.0,
        retry_until_up=True,
    )
    assert command[:5] == [
        "sky",
        "launch",
        "-c",
        "dna-exp479-lora-reload-a10",
        "sky/lora-reload-audit.yaml",
    ]
    assert "EXP479_INSTANCE_PRICE_PER_HOUR_USD=1.006" in command
    assert command.count("--secret") == 1
    assert "WANDB_API_KEY" in command
    assert "HF_TOKEN" not in command
    assert "--down" in command

    stage = Path("sky/lora-reload-audit.yaml").read_text(encoding="utf-8")
    assert stage.count("uv run --locked exp479 lora-reload-audit") == 1
    assert stage.count("uv run --locked pytest") == 1
    assert "cloud: aws" in stage
    assert "region: us-east-2" in stage
    assert "accelerators: A10G:1" in stage
    assert "use_spot: false" in stage
    assert "disk_size: 80" in stage
    assert "WANDB_API_KEY" in stage
    assert "HF_TOKEN" not in stage
    assert "vep" not in stage.lower()
    assert "timeout --signal=TERM 7200" in stage
    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in stage
    assert "nuc-dep" not in stage


def test_gated_lora_uses_one_a10g_wandb_and_no_downstream_evaluation() -> None:
    command = launch_command(
        "gated-lora-mntp",
        "a" * 40,
        1234,
        prior_cost_usd=37.091314,
        retry_until_up=True,
    )
    assert command[:5] == [
        "sky",
        "launch",
        "-c",
        "dna-exp479-gated-lora-a10",
        "sky/gated-lora-mntp.yaml",
    ]
    assert "EXP479_INSTANCE_PRICE_PER_HOUR_USD=1.006" in command
    assert command.count("--secret") == 1
    assert "WANDB_API_KEY" in command
    assert "HF_TOKEN" not in command
    assert "--down" in command

    stage = Path("sky/gated-lora-mntp.yaml").read_text(encoding="utf-8")
    assert stage.count("uv run --locked exp479 gated-lora-mntp") == 1
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
    assert "timeout --signal=TERM 14400" in stage
    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in stage


def test_two_pass_gate_uses_one_a10g_wandb_and_no_model_updates() -> None:
    command = launch_command(
        "two-pass-information-gate",
        "a" * 40,
        1234,
        prior_cost_usd=40.0,
        retry_until_up=True,
    )
    assert command[:5] == [
        "sky",
        "launch",
        "-c",
        "dna-exp479-two-pass-a10",
        "sky/two-pass-information-gate.yaml",
    ]
    assert "EXP479_INSTANCE_PRICE_PER_HOUR_USD=1.006" in command
    assert command.count("--secret") == 1
    assert "WANDB_API_KEY" in command
    assert "HF_TOKEN" not in command
    assert "--down" in command

    stage = Path("sky/two-pass-information-gate.yaml").read_text(encoding="utf-8")
    assert stage.count("uv run --locked exp479 two-pass-information-gate") == 1
    assert stage.count("uv run --locked pytest") == 1
    assert "--batch-size 64" in stage
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
    assert "timeout --signal=TERM 3600" in stage
    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in stage


def test_two_pass_vep_uses_one_lambda_gh200_and_no_external_upload() -> None:
    command = launch_command(
        "two-pass-vep",
        "a" * 40,
        1234,
        prior_cost_usd=39.5,
        retry_until_up=True,
    )
    assert command[:5] == [
        "sky",
        "launch",
        "-c",
        "dna-exp479-two-pass-vep-gh200",
        "sky/two-pass-vep.yaml",
    ]
    assert "EXP479_INSTANCE_PRICE_PER_HOUR_USD=2.29" in command
    assert command.count("--secret") == 1
    assert "WANDB_API_KEY" in command
    assert "HF_TOKEN" not in command
    assert "--down" in command

    stage = Path("sky/two-pass-vep.yaml").read_text(encoding="utf-8")
    assert stage.count("uv run --locked exp479 two-pass-vep") == 1
    assert stage.count("uv run --locked pytest") == 1
    assert "cloud: lambda" in stage
    assert "accelerators: GH200:1" in stage
    assert "disk_size: 256" in stage
    assert "--batch-size 1024" in stage
    assert "--n-bootstrap 1000" in stage
    assert "WANDB_API_KEY" in stage
    assert "HF_TOKEN" not in stage
    assert "nuc-dep" not in stage
    assert "upload" not in stage.lower()
    assert "timeout --signal=TERM 7200" in stage
    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in stage
