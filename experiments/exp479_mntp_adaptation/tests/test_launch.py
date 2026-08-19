from __future__ import annotations

from launch import launch_command


def test_preflight_launch_is_commit_pinned_lambda_gh200_and_self_terminating() -> None:
    commit = "a" * 40
    command = launch_command("preflight", commit, 1234)
    assert command[:5] == ["sky", "launch", "-c", "dna-exp479-gh200", "sky/preflight.yaml"]
    assert command[command.index("--git-ref") + 1] == commit
    assert "EXP479_INSTANCE_START_UNIX=1234" in command
    assert "--down" in command
    assert "--yes" in command
