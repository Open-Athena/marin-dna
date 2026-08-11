from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parents[1]
REPOSITORY_ROOT = Path(__file__).parents[3]
STORAGE_PREFIX = "s3://oa-bolinas/snakemake/vertebrate_projection_dataset/"


def test_default_profile_uses_canonical_s3_storage() -> None:
    profile = yaml.safe_load(
        (PROJECT_ROOT / "workflow/profiles/default/config.yaml").read_text()
    )
    assert profile["default-storage-provider"] == "s3"
    assert profile["default-storage-prefix"] == STORAGE_PREFIX


def test_hf_worker_uses_snakemake_storage_instead_of_snapshot_copy() -> None:
    worker = (PROJECT_ROOT / "sky/hf.yaml").read_text()
    for obsolete in [
        "DATA_SNAPSHOT_SHA",
        "ARTIFACT_S3_PREFIX",
        "aws s3 cp",
        "aws s3 sync",
    ]:
        assert obsolete not in worker
    assert "--profile workflow/profiles/default" in worker


def test_ci_dry_run_explicitly_disables_remote_storage() -> None:
    workflow = (REPOSITORY_ROOT / ".github/workflows/test.yml").read_text()
    assert "snakemake -n --quiet all --default-storage-provider none" in workflow
