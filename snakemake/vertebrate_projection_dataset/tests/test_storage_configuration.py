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
    assert 'PIPELINE_COMMIT_SHA: ""' in worker
    assert 'test -n "$PIPELINE_COMMIT_SHA"' in worker


def test_ci_dry_run_explicitly_disables_remote_storage() -> None:
    workflow = (REPOSITORY_ROOT / ".github/workflows/test.yml").read_text()
    assert "snakemake -n --quiet all --default-storage-provider none" in workflow


def test_results_are_producer_keyed_and_verification_receipts_are_local() -> None:
    common = (PROJECT_ROOT / "workflow/rules/common.smk").read_text()
    staging = (PROJECT_ROOT / "workflow/rules/staging.smk").read_text()
    projection = (PROJECT_ROOT / "workflow/rules/projection.smk").read_text()
    dataset = (PROJECT_ROOT / "workflow/rules/dataset.smk").read_text()

    assert "PIPELINE_COMMIT = resolve_pipeline_commit()" in common
    assert "PIPELINE_CONFIG_SHA256 = hash_pipeline_config(config)" in common
    assert (
        'f"results/{PIPELINE_VERSION}/{PIPELINE_COMMIT}/'
        '{PIPELINE_CONFIG_SHA256}/{TIER}"' in common
    )
    assert "multiz_mirror.done" not in staging
    assert "local(HAL_VALIDATION)" in staging
    assert projection.count("validation=local(HAL_VALIDATION)") == 3
    assert 'temp(local(f"{RESULTS}/upload.done/{{region}}"))' in dataset
    assert 'local(expand(f"{RESULTS}/upload.done/{{region}}"' in dataset
