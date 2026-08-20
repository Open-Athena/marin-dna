"""Contracts for the pinned evals_v2 Sky GPU runtime and parity gate."""

from __future__ import annotations

import ast
import hashlib
import re
import tomllib
from pathlib import Path

import pandas as pd
import pytest
import yaml
from marin_dna_evals import gpu_runtime_validation
from marin_dna_evals.gpu_runtime_validation import (
    RuntimeSpec,
    Tolerance,
    compare_score_frames,
    load_validation_spec,
    read_verified_parquet,
    validate_runtime_metadata,
)

PROJECT_ROOT = Path(__file__).parents[2]
VALIDATION_CONFIG = PROJECT_ROOT / "config" / "gpu_runtime_validation.yaml"


def test_runtime_gate_has_no_optimization_removable_assertions() -> None:
    source_path = Path(gpu_runtime_validation.__file__)
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    assertion_lines = [
        node.lineno for node in ast.walk(module) if isinstance(node, ast.Assert)
    ]
    assert assertion_lines == []


def test_runtime_contract_matches_project_and_sky_configuration() -> None:
    spec = load_validation_spec(VALIDATION_CONFIG)
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    with (PROJECT_ROOT / "sky" / "run.yaml").open(encoding="utf-8") as handle:
        sky = yaml.safe_load(handle)

    assert (
        f"torch=={spec.runtime.pytorch_version}" in project["project"]["dependencies"]
    )
    assert sky["resources"]["image_id"] == spec.runtime.image_id
    setup = sky["setup"].replace("\\\n", " ")
    run = sky["run"].replace("\\\n", " ")
    path_export = setup.index(
        'export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"'
    )
    uv_install = setup.index("if ! command -v uv")
    uv_version = setup.index('if [[ "$(uv --version)"')
    assert path_export < uv_install < uv_version
    assert re.search(r"\bevals-gpu-runtime-check\b[^\n]*\bsmoke\b", setup)
    assert re.search(r"\bevals-gpu-runtime-check\b[^\n]*\bparity\b", run)
    assert spec.parity.split == "train"
    assert spec.parity.dataset_filename == "train.parquet"


def test_validate_runtime_metadata_strips_local_torch_suffix() -> None:
    spec = RuntimeSpec(
        image_id="ami-test",
        image_name="test",
        operating_system="Ubuntu",
        driver_version="595.71.05",
        pytorch_version="2.13.0",
        compiled_cuda_version="13.0",
        device_name="NVIDIA A10G",
    )
    observed = validate_runtime_metadata(
        spec,
        pytorch_version="2.13.0+cu130",
        compiled_cuda_version="13.0",
        driver_version="595.71.05",
        device_name="NVIDIA A10G",
    )
    assert observed["pytorch_version"] == "2.13.0"


def test_validate_runtime_metadata_rejects_implicit_cuda_change() -> None:
    spec = load_validation_spec(VALIDATION_CONFIG).runtime
    with pytest.raises(AssertionError, match="GPU runtime mismatch"):
        validate_runtime_metadata(
            spec,
            pytorch_version=spec.pytorch_version,
            compiled_cuda_version="13.1",
            driver_version=spec.driver_version,
            device_name=spec.device_name,
        )


def _score_frame(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chrom": ["1", "3"],
            "pos": [100, 200],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "llr_fwd": [0.1 + offset, -0.2],
            "jsd_fwd": [0.01, 0.02],
        }
    )


def test_compare_score_frames_reports_tolerance_bounded_differences() -> None:
    tolerances = {
        "llr_fwd": Tolerance(rtol=0.0, atol=0.005),
        "jsd_fwd": Tolerance(rtol=0.0, atol=0.0001),
    }
    report = compare_score_frames(
        _score_frame(offset=0.004),
        _score_frame(),
        identity_columns=("chrom", "pos", "ref", "alt"),
        tolerances=tolerances,
    )
    assert report["llr_fwd"]["max_absolute_difference"] == pytest.approx(0.004)
    assert report["llr_fwd"]["n_outside_tolerance"] == 0


def test_compare_score_frames_rejects_out_of_tolerance_difference() -> None:
    with pytest.raises(AssertionError, match="score parity failed"):
        compare_score_frames(
            _score_frame(offset=0.006),
            _score_frame(),
            identity_columns=("chrom", "pos", "ref", "alt"),
            tolerances={"llr_fwd": Tolerance(rtol=0.0, atol=0.005)},
        )


def test_compare_score_frames_rejects_variant_misalignment() -> None:
    candidate = _score_frame()
    candidate.loc[0, "pos"] = 101
    with pytest.raises(AssertionError, match="variants do not match"):
        compare_score_frames(
            candidate,
            _score_frame(),
            identity_columns=("chrom", "pos", "ref", "alt"),
            tolerances={"llr_fwd": Tolerance(rtol=0.0, atol=0.005)},
        )


def test_read_verified_parquet_rejects_changed_baseline(tmp_path: Path) -> None:
    path = tmp_path / "baseline.parquet"
    _score_frame().to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pd.testing.assert_frame_equal(
        read_verified_parquet(str(path), digest), _score_frame()
    )
    with pytest.raises(AssertionError, match="baseline checksum mismatch"):
        read_verified_parquet(str(path), "0" * 64)


def test_inference_scores_before_opening_s3_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = load_validation_spec(VALIDATION_CONFIG).parity
    variants = _score_frame().loc[:, list(spec.identity_columns)]
    events: list[str] = []

    class DatasetStub:
        def to_pandas(self) -> pd.DataFrame:
            return variants

    def fake_hf_hub_download(**kwargs: object) -> str:
        assert kwargs["repo_id"] == spec.dataset_repo
        assert kwargs["filename"] == "train.parquet"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["revision"] == spec.dataset_revision
        return str(tmp_path / "train.parquet")

    def fake_load_dataset(*args: object, **kwargs: object) -> DatasetStub:
        assert args == ("parquet",)
        assert kwargs["data_files"] == {"train": str(tmp_path / "train.parquet")}
        return DatasetStub()

    def fake_compute_variant_scores(**kwargs: object) -> pd.DataFrame:
        events.append("score")
        return pd.DataFrame(
            {column: [0.0, 0.0] for column in spec.tolerances},
        )

    def fake_read_verified_parquet(uri: str, sha256: str) -> pd.DataFrame:
        events.append("baseline")
        return pd.DataFrame()

    monkeypatch.setattr(gpu_runtime_validation, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(gpu_runtime_validation, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        gpu_runtime_validation, "compute_variant_scores", fake_compute_variant_scores
    )
    monkeypatch.setattr(
        gpu_runtime_validation, "read_verified_parquet", fake_read_verified_parquet
    )
    monkeypatch.setattr(
        gpu_runtime_validation,
        "compare_score_frames",
        lambda *args, **kwargs: {},
    )

    report = gpu_runtime_validation.run_inference_parity(spec, checkpoint_path=tmp_path)

    assert events == ["score", "baseline"]
    assert report["n_rows"] == len(variants)
