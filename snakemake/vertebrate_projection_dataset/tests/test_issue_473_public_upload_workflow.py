from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parents[1]


def test_public_upload_workflow_is_pinned_and_public_only() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "config/issue_473_public_upload.yaml").read_text()
    )
    rules = (PROJECT_ROOT / "workflow/rules/issue_473_public_upload.smk").read_text()
    launcher = (PROJECT_ROOT / "sky/issue_473_hf_public_upload.yaml").read_text()

    assert len(config["publication_artifact_commit"]) == 40
    assert len(config["publication_artifact_config_sha256"]) == 64
    assert config["publication_artifact_generation"] == "issue_473_publication_v3"
    assert "upload_public_validated_dataset" in rules
    assert "upload_private_validated_dataset" not in rules
    assert 'DRY_RUN: "1"' in launcher
    assert 'ALLOW_PUBLIC_HF_UPLOAD: "0"' in launcher
    assert 'test "$DRY_RUN" = "1"' in launcher
    assert 'else\n    test "$ALLOW_PUBLIC_HF_UPLOAD" = "1"' in launcher
