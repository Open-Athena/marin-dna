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
    assert "PROJECTION_REQUESTS" in common
    assert "build_projection_requests" in projection
    assert "write_hal_request_bed6" in projection
    assert "write_maf_request_candidates" in projection
    assert "write_hal_bed6" not in projection
    assert "write_maf_candidates" not in projection
    assert 'temp(local(f"{RESULTS}/upload.done/{{region}}"))' in dataset
    assert 'local(expand(f"{RESULTS}/upload.done/{{region}}"' in dataset


def test_gpn_star_profile_is_pinned_additive_and_ec2_only() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "config/gpn_star_p.yaml").read_text())
    manifest = PROJECT_ROOT / config["gpn_entropy_manifest"]
    manifest_rows = manifest.read_text().splitlines()
    snakefile = (PROJECT_ROOT / "workflow/gpn_star.Snakefile").read_text()
    anchors = (PROJECT_ROOT / "workflow/rules/gpn_star_anchors.smk").read_text()
    worker = (PROJECT_ROOT / "sky/gpn_star_project.yaml").read_text()

    assert config["pipeline_version"] == "gpn-star-p-uniform-v1"
    assert config["gpn_dataset_repo"] == "songlab/gpn-star-scores"
    assert (
        config["gpn_dataset_revision"]
        == "5c799b2ec6aa089f0caa8294ae72adb4510f81ae"
    )
    assert config["gpn_score_set"] == "gpn-star-hg38-p243-200m"
    assert config["gpn_entropy_cutoff"] == 0.081001
    assert config["gpn_min_selected_bases"] == 51
    assert config["expected_windows_ge_20pct"] == 1_627_410
    assert config["assignment_arms"] == [
        "cds",
        "utr3",
        "tss_region_and_utr5",
        "ncrna_exon",
        "enhancer",
        "background",
    ]
    assert len(manifest_rows) == 25
    assert set(row.split("\t")[0] for row in manifest_rows[1:]) == set(
        config["standard_chroms"]
    )
    assert all(len(row.split("\t")[4]) == 64 for row in manifest_rows[1:])

    assert 'include: "rules/gpn_star_anchors.smk"' in snakefile
    assert 'include: "rules/projection.smk"' in snakefile
    assert 'include: "rules/dataset.smk"' not in snakefile
    assert "temp(local(" in anchors
    assert 'labels=local(f"{RESULTS}/anchors/labels.parquet")' in anchors
    assert 'selected=local(f"{RESULTS}/anchors/selected.parquet")' in anchors
    assert "retries: 3" in anchors
    assert "c6id.12xlarge" in worker
    assert "workflow/gpn_star.Snakefile" in worker
    assert "--profile workflow/profiles/default" in worker
    assert 'tests|all|all_projection)' in worker
    assert "uv run --locked pytest" in worker
