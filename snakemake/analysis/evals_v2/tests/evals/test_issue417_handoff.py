from __future__ import annotations

import pytest
from marin_dna_evals.issue417_handoff import (
    parse_sky_status_json,
    validate_hf_export_listing,
)

from scripts.issue417_wait_and_launch_eval import _object_stat_command

PREFIX = "gs://bucket/checkpoints/run/hf/step-4999"


def _listing(*, model_size: int = 1_019_422_904, extra: str = "") -> str:
    return (
        f"1366  2026-08-02T03:46:57Z  {PREFIX}/config.json\n"
        f"{model_size}  2026-08-02T03:46:56Z  {PREFIX}/model.safetensors\n"
        f"1693  2026-08-02T03:46:57Z  {PREFIX}/tokenizer.json\n"
        f"289  2026-08-02T03:46:57Z  {PREFIX}/tokenizer_config.json\n"
        f"{extra}"
        "TOTAL: 4 objects, 1019426252 bytes (972.2 MiB)\n"
    )


def test_validate_hf_export_listing_accepts_exact_complete_export() -> None:
    sizes = validate_hf_export_listing(PREFIX, _listing())

    assert sizes == {
        "config.json": 1366,
        "model.safetensors": 1_019_422_904,
        "tokenizer.json": 1693,
        "tokenizer_config.json": 289,
    }


def test_validate_hf_export_listing_rejects_missing_object() -> None:
    listing = _listing().replace(
        f"289  2026-08-02T03:46:57Z  {PREFIX}/tokenizer_config.json\n",
        "",
    )

    with pytest.raises(AssertionError, match="missing objects"):
        validate_hf_export_listing(PREFIX, listing)


def test_validate_hf_export_listing_rejects_unexpected_object() -> None:
    extra = f"12  2026-08-02T03:46:58Z  {PREFIX}/partial.tmp\n"

    with pytest.raises(AssertionError, match="unexpected objects"):
        validate_hf_export_listing(PREFIX, _listing(extra=extra))


def test_validate_hf_export_listing_rejects_small_model() -> None:
    with pytest.raises(AssertionError, match="unexpectedly small"):
        validate_hf_export_listing(PREFIX, _listing(model_size=999_999_999))


def test_parse_sky_status_json_accepts_missing_cluster_warning() -> None:
    output = "Cluster(s) not found: dna417-cds-vep.\n[]\n"

    assert parse_sky_status_json(output) == []


def test_parse_sky_status_json_accepts_existing_cluster() -> None:
    output = 'Enabled Infra: aws\n[{"name": "dna417-cds-vep", "status": "UP"}]\n'

    assert parse_sky_status_json(output) == [{"name": "dna417-cds-vep", "status": "UP"}]


def test_object_stat_command_routes_cloud_uri_to_its_native_cli() -> None:
    assert _object_stat_command("gs://bucket/checkpoint/model.safetensors") == [
        "gsutil",
        "stat",
        "gs://bucket/checkpoint/model.safetensors",
    ]
    assert _object_stat_command("s3://bucket/results/metrics/model/eval.parquet") == [
        "aws",
        "s3api",
        "head-object",
        "--bucket",
        "bucket",
        "--key",
        "results/metrics/model/eval.parquet",
    ]


def test_object_stat_command_rejects_unsupported_or_incomplete_uri() -> None:
    with pytest.raises(AssertionError, match="unsupported"):
        _object_stat_command("https://bucket/results/file")
    with pytest.raises(AssertionError, match="no key"):
        _object_stat_command("s3://bucket")
