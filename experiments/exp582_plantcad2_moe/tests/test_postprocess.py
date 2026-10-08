from pathlib import Path

import pytest

from exp582_moe import postprocess
from exp582_moe.postprocess import PostprocessConfig, digest


def test_postprocess_requires_pinned_durable_inputs(tmp_path: Path):
    path = tmp_path / "value"
    path.write_bytes(b"exp582")
    assert (
        digest(path)
        == "1d9b5c8bd7b6228a8ac98eb19e07a65cf68ac0cbdde45aab092360c8ad657324"
    )
    config = PostprocessConfig(
        condition="random",
        run_id="exp582-final-eval-postprocess-smoke-v1",
        cluster="cw-rno2a",
        requests_uri="s3://example/requests",
        requests_sha256="a" * 64,
        artifact_root="s3://example/artifacts",
        analysis_wheel_uri="s3://example/analysis.whl",
        analysis_wheel_sha256="b" * 64,
        output_prefix="s3://example/output",
    )
    assert config.nodes == 1
    assert config.skip_probes is False


def test_postprocess_can_skip_probes():
    config = PostprocessConfig(
        condition="pretrained",
        run_id="exp582-final-eval-postprocess-smoke-no-probes-v1",
        cluster="cw-rno2a",
        requests_uri="s3://example/requests",
        requests_sha256="a" * 64,
        artifact_root="s3://example/artifacts",
        analysis_wheel_uri="s3://example/analysis.whl",
        analysis_wheel_sha256="b" * 64,
        output_prefix="s3://example/output",
        skip_probes=True,
    )
    assert config.skip_probes is True


def test_postprocess_probe_only_requires_reduced_analysis():
    values = {
        "condition": "pretrained",
        "run_id": "exp582-final-eval-probes-smoke-v1",
        "cluster": "cw-rno2a",
        "requests_uri": "s3://example/requests",
        "requests_sha256": "a" * 64,
        "artifact_root": "s3://example/artifacts",
        "analysis_wheel_uri": "s3://example/analysis.whl",
        "analysis_wheel_sha256": "b" * 64,
        "output_prefix": "s3://example/output",
        "probe_only": True,
    }
    with pytest.raises(ValueError, match="reduced analysis root"):
        PostprocessConfig(**values)
    config = PostprocessConfig(**values, reduced_analysis_root="s3://example/reduced")
    assert config.probe_only is True
    with pytest.raises(ValueError, match="cannot skip"):
        PostprocessConfig(
            **values,
            reduced_analysis_root="s3://example/reduced",
            skip_probes=True,
        )


def test_collect_reduced_predictions_validates_receipt(monkeypatch, tmp_path: Path):
    config = PostprocessConfig(
        condition="random",
        run_id="exp582-final-eval-probes-smoke-v1",
        cluster="cw-rno2a",
        requests_uri="s3://example/requests",
        requests_sha256="a" * 64,
        artifact_root="s3://example/artifacts",
        analysis_wheel_uri="s3://example/analysis.whl",
        analysis_wheel_sha256="b" * 64,
        output_prefix="s3://example/output",
        probe_only=True,
        reduced_analysis_root="s3://example/reduced",
    )
    receipt = {
        "condition": "pretrained",
        "requests_sha256": "a" * 64,
        "analysis_wheel_sha256": "b" * 64,
        "source_artifacts": "s3://example/artifacts",
        "probes_included": False,
        "files": {},
    }
    monkeypatch.setattr(
        postprocess,
        "read_bytes",
        lambda uri: __import__("json").dumps(receipt).encode(),
    )
    with pytest.raises(ValueError, match="does not match"):
        postprocess.collect_reduced_predictions(config, tmp_path)


def test_collect_reduced_predictions_accepts_separately_pinned_reduction_wheel(
    monkeypatch, tmp_path: Path
):
    config = PostprocessConfig(
        condition="random",
        run_id="exp582-final-eval-probes-smoke-v1",
        cluster="cw-rno2a",
        requests_uri="s3://example/requests",
        requests_sha256="a" * 64,
        artifact_root="s3://example/artifacts",
        analysis_wheel_uri="s3://example/probes.whl",
        analysis_wheel_sha256="b" * 64,
        output_prefix="s3://example/output",
        probe_only=True,
        reduced_analysis_root="s3://example/reduced",
        reduced_analysis_wheel_sha256="c" * 64,
    )
    receipt = {
        "condition": "random",
        "requests_sha256": "a" * 64,
        "analysis_wheel_sha256": "c" * 64,
        "source_artifacts": "s3://example/artifacts",
        "probes_included": False,
        "files": {},
    }
    monkeypatch.setattr(
        postprocess,
        "read_bytes",
        lambda uri: __import__("json").dumps(receipt).encode(),
    )
    with pytest.raises(ValueError, match="omits context-8192"):
        postprocess.collect_reduced_predictions(config, tmp_path)


def test_distributed_postprocess_has_one_writer(monkeypatch):
    calls = []
    monkeypatch.setattr(
        postprocess,
        "run_postprocess",
        lambda config: calls.append(("run", config)),
    )
    config = object()
    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_INDEX", "1")
    postprocess.run_postprocess_distributed(config)
    assert calls == []

    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_INDEX", "0")
    postprocess.run_postprocess_distributed(config)
    assert calls == [("run", config)]

    calls.clear()
    monkeypatch.delenv("IRIS_MULTIGPU_PROCESS_INDEX")
    monkeypatch.setattr(postprocess.jax, "process_index", lambda: 1)
    postprocess.run_postprocess_distributed(config)
    assert calls == []
