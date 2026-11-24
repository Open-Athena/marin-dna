from pathlib import Path

import pytest
from marin_dna_evals.sweep_identity import check_clusters


@pytest.fixture(autouse=True)
def clean_launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in ("SKYPILOT_CONFIG", "SKYPILOT_PROJECT_CONFIG", "SKYPILOT_GLOBAL_CONFIG"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("source", [".sky.yaml", ".sky/config.yaml", "explicit.yaml"])
def test_rejects_bolinas_in_local_sources(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = Path(source)
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        'aws: {remote_identity: "BolinasJob"} # same role, different YAML\n'
    )
    if source == "explicit.yaml":
        monkeypatch.setenv("SKYPILOT_PROJECT_CONFIG", source)
    with pytest.raises(ValueError, match="lacks EC2 teardown"):
        check_clusters(["evals-v2-a"])


def test_cluster_pattern_order_and_config_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Path("user.yaml").write_text("aws: {remote_identity: BolinasJob}")
    monkeypatch.setenv("SKYPILOT_GLOBAL_CONFIG", "user.yaml")
    Path(".sky.yaml").write_text(
        "aws:\n  remote_identity:\n    - evals-v2-a: OtherRole\n    - 'evals-*': BolinasJob\n"
    )
    check_clusters(["evals-v2-a"])
    with pytest.raises(ValueError, match="evals-v2-b may use BolinasJob"):
        check_clusters(["evals-v2-a", "evals-v2-b"])


def test_missing_explicit_config_fails_before_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    check_clusters(["evals-v2-a"])
    monkeypatch.setenv("SKYPILOT_PROJECT_CONFIG", "missing.yaml")
    with pytest.raises(FileNotFoundError):
        check_clusters(["evals-v2-a"])


@pytest.mark.parametrize("selection", ["active_workspace: default", ""])
def test_workspace_identity_overrides_global(selection: str) -> None:
    Path(".sky.yaml").write_text(
        f"{selection}\naws: {{remote_identity: OtherRole}}\n"
        "workspaces: {default: {aws: {remote_identity: BolinasJob}}}\n"
    )
    with pytest.raises(ValueError, match="may use BolinasJob"):
        check_clusters(["evals-v2-a"])


@pytest.mark.parametrize("project_override", [False, True])
def test_workspace_display_name_and_conflicting_project_role(
    project_override: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Path("server.yaml").write_text(
        "active_workspace: dna\naws: {remote_identity: OtherRole}\n"
        "workspaces: {ws-123: {name: dna, aws: {remote_identity: BolinasJob}}}\n"
    )
    monkeypatch.setenv("SKYPILOT_GLOBAL_CONFIG", "server.yaml")
    if project_override:
        Path(".sky.yaml").write_text(
            "workspaces: {ws-123: {name: dna, aws: {remote_identity: OtherRole}}}\n"
        )
    with pytest.raises(ValueError, match="may use BolinasJob"):
        check_clusters(["evals-v2-a"])
