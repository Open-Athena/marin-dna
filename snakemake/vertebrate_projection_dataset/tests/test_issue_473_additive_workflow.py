"""Issue #473 must remain isolated from the shared S3-backed workflow."""

from pathlib import Path


def test_issue_473_uses_a_standalone_additive_entrypoint() -> None:
    root = Path(__file__).parents[1]
    shared = (root / "workflow" / "Snakefile").read_text()
    issue = (root / "workflow" / "Issue473.smk").read_text()

    shared_includes = (
        "rules/common.smk",
        "rules/anchors.smk",
        "rules/staging.smk",
        "rules/projection.smk",
        "rules/dataset.smk",
    )
    issue_includes = (
        "rules/issue_473.smk",
        "rules/issue_473_fixed.smk",
        "rules/issue_473_trace.smk",
    )
    for rule_module in shared_includes:
        assert f'include: "{rule_module}"' in shared
        assert f'include: "{rule_module}"' in issue
    for rule_module in issue_includes:
        assert rule_module not in shared
        assert f'include: "{rule_module}"' in issue

    for launcher_name in ("issue_473.yaml", "issue_473_full_v2.yaml"):
        launcher = (root / "sky" / launcher_name).read_text()
        assert "--snakefile workflow/Issue473.smk" in launcher
