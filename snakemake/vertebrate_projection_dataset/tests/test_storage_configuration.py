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


def test_projection_memory_pools_cover_default_jobs_with_worker_headroom() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config/config.yaml").read_text())
    profile = yaml.safe_load(
        (PROJECT_ROOT / "workflow/profiles/default/config.yaml").read_text()
    )
    worker = yaml.safe_load((PROJECT_ROOT / "sky/project.yaml").read_text())
    largest_job_mb = max(config["liftover_mem_mb"], config["table_mem_mb"], 30000)
    worker_pool_mb = int(worker["envs"]["MEM_MB"])
    assert int(profile["resources"]["mem_mb"]) >= largest_job_mb
    assert worker_pool_mb >= largest_job_mb
    # Conservatively allow four GiB beyond the scheduler pool for the runtime.
    minimum_worker_mib = float(str(worker["resources"]["memory"]).rstrip("+")) * 1024
    assert minimum_worker_mib >= worker_pool_mb + 4096


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
    projection = (PROJECT_ROOT / "workflow/rules/projection.smk").read_text()
    dataset = (PROJECT_ROOT / "workflow/rules/dataset.smk").read_text()

    assert "PIPELINE_COMMIT = resolve_pipeline_commit()" in common
    assert "PIPELINE_CONFIG_SHA256 = hash_pipeline_config(RESOLVED_CONFIG)" in common
    assert "chain_assets_sha256" in common and "genome_assets_sha256" in common
    assert (
        'f"results/{PIPELINE_VERSION}/{PIPELINE_COMMIT}/'
        '{PIPELINE_CONFIG_SHA256}/{TIER}"' in common
    )
    assert not (PROJECT_ROOT / "workflow/Snakefile.chains").exists()
    assert not (PROJECT_ROOT / "workflow/rules/staging.smk").exists()
    assert "PROJECTION_REQUESTS" in common
    assert "prepare_chain_requests" in projection
    assert "liftOver -minMatch=0.95 -multiple" in projection
    assert "validate_genome_source" in projection
    assert "validate_genome_dictionary" in projection
    for recipe in (PROJECT_ROOT / "sky").glob("*.yaml"):
        assert "Snakefile.chains" not in recipe.read_text()
    for obsolete in ("halLiftover", "hal2fasta", "HAL_PATH", "multiz_candidates"):
        assert obsolete not in projection
    assert 'temp(local(f"{RESULTS}/upload.done/{{region}}"))' in dataset
    assert 'local(expand(f"{RESULTS}/upload.done/{{region}}"' in dataset
