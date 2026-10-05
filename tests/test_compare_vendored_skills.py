import json
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[1]
    / ".agents/skills/maintain-vendored-skills/scripts/compare_upstream.py"
)
SKILL_ROOT = Path(__file__).parents[1] / ".agents/skills"
MANIFEST_ROOT = SKILL_ROOT / "maintain-vendored-skills/references"


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def _make_upstream(tmp_path: Path) -> tuple[Path, str]:
    upstream = tmp_path / "upstream"
    skill = upstream / ".agents/skills/demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("demo\n")
    _run(["git", "init", "-q"], upstream)
    _run(["git", "add", "."], upstream)
    _run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
        upstream,
    )
    commit = _run(["git", "rev-parse", "HEAD"], upstream).stdout.strip()
    return upstream, commit


def test_custom_manifest_reports_exact_and_changed_vendors(tmp_path: Path) -> None:
    upstream, commit = _make_upstream(tmp_path)
    local_root = tmp_path / "local"
    local_skill = local_root / ".agents/skills/demo"
    local_skill.mkdir(parents=True)
    local_file = local_skill / "SKILL.md"
    local_file.write_text("demo\n")

    manifest = tmp_path / "vendor.json"
    manifest.write_text(
        json.dumps(
            {
                "upstream_repo": "example/upstream",
                "commit": commit,
                "unchanged": ["demo"],
                "adapted": [],
                "local": [],
            }
        )
    )
    command = [
        sys.executable,
        str(SCRIPT),
        "--manifest",
        str(manifest),
        "--upstream-root",
        str(upstream),
        "--repo-root",
        str(local_root),
    ]

    generated_cache = upstream / ".agents/skills/demo/__pycache__"
    generated_cache.mkdir()
    (generated_cache / "noise.pyc").write_bytes(b"generated")

    exact = subprocess.run(command, check=False, capture_output=True, text=True)
    assert exact.returncode == 0
    assert "# example/upstream vendored-skill deltas" in exact.stdout
    assert "`demo`: content/type/mode-identical" in exact.stdout

    upstream_file = upstream / ".agents/skills/demo/SKILL.md"
    upstream_file.write_text("dirty\n")
    dirty = subprocess.run(command, check=False, capture_output=True, text=True)
    assert dirty.returncode == 1
    assert "upstream skill tree is dirty" in dirty.stdout
    assert "Input provenance: Git HEAD" in dirty.stdout
    upstream_file.write_text("demo\n")

    local_file.chmod(0o755)
    mode_changed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert mode_changed.returncode == 1
    assert "mode=100644 type=file" in mode_changed.stdout
    assert "mode=100755 type=file" in mode_changed.stdout
    local_file.chmod(0o644)

    local_file.unlink()
    local_file.symlink_to("demo\n")
    type_changed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert type_changed.returncode == 1
    assert "mode=100644 type=file" in type_changed.stdout
    assert "mode=120000 type=symlink" in type_changed.stdout
    local_file.unlink()

    local_file.write_text("changed\n")
    changed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert changed.returncode == 1
    assert "example/upstream/.agents/skills/demo/SKILL.md" in changed.stdout
    assert "Open-Athena/marin-dna/.agents/skills/demo/SKILL.md" in changed.stdout
    assert "demo is classified unchanged but has a diff" in changed.stdout


def test_every_repository_skill_has_one_provenance_classification() -> None:
    classified: list[str] = []
    for path in MANIFEST_ROOT.glob("*manifest.json"):
        manifest = json.loads(path.read_text())
        classified.extend(manifest["unchanged"])
        classified.extend(item["name"] for item in manifest["adapted"])
        classified.extend(manifest["local"])

    actual = sorted(path.parent.name for path in SKILL_ROOT.glob("*/SKILL.md"))
    assert sorted(classified) == actual
    assert len(classified) == len(set(classified))
