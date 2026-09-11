"""Bounded recovery contracts; no network or GPU operations."""

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from botocore.exceptions import ClientError

spec = importlib.util.spec_from_file_location(
    "finish_local_a10g", Path(__file__).with_name("finish-local-a10g.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.writes = []
        self.fail_after = None

    def head_object(self, **kwargs):
        if kwargs["Key"] not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return self.objects[kwargs["Key"]]

    def put_object(self, **kwargs):
        if self.fail_after == len(self.writes):
            raise RuntimeError("injected transfer failure")
        assert kwargs["Bucket"] == "oa-bolinas"
        assert kwargs["IfNoneMatch"] == "*"
        payload = kwargs["Body"].read()
        digest = base64.b64encode(hashlib.sha256(payload).digest()).decode()
        assert digest == kwargs["ChecksumSHA256"]
        self.objects[kwargs["Key"]] = {
            "ContentLength": len(payload),
            "ChecksumSHA256": digest,
        }
        self.writes.append(kwargs["Key"])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    work, project = tmp_path / "work", tmp_path / "project"
    work.mkdir()
    (project / "config").mkdir(parents=True)
    (project / "workflow/profiles/default").mkdir(parents=True)
    monkeypatch.setattr(module, "WORK", work)
    monkeypatch.setattr(module, "PROJECT", project)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    stage = json.loads(
        (Path(module.__file__).parent / "step-10000-staged.json").read_text()
    )
    receipt = {
        "completed": True,
        "exit_status": 0,
        "model": module.MODEL,
        "split": "train",
        "cohort_sizes": module.COHORTS,
        "publication": "local_only",
        "source_commit": "a" * 40,
        "checkpoint_sha256": stage["verified_objects"]["model.safetensors"]["sha256"],
        "harness_sha256": "b" * 64,
    }
    (work / "a10g-run-receipt.json").write_text(json.dumps(receipt))
    (project / "config/config.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": module.MODEL, "rag_harness": {"sha256": "b" * 64}}]}
        )
    )
    (project / "workflow/profiles/default/config.yaml").write_text(
        yaml.safe_dump(
            {
                "default-storage-provider": "s3",
                "default-storage-prefix": "s3://oa-bolinas/snakemake/analysis/evals_v2/",
            }
        )
    )
    for kind in ("scores", "metrics"):
        for cohort in module.COHORTS:
            path = project / f"results/{kind}/{module.MODEL}/{cohort}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fixture {kind} {cohort}".encode())
    monkeypatch.setattr(
        module.pq,
        "ParquetFile",
        lambda p: SimpleNamespace(
            metadata=SimpleNamespace(num_rows=module.COHORTS[p.stem]),
            schema_arrow=SimpleNamespace(
                names=["emb_ref", "emb_alt", "llr_fwd", "llr_rc"]
            ),
        ),
    )
    commands = []
    monkeypatch.setattr(
        module.subprocess, "run", lambda cmd, **kw: commands.append((cmd, kw))
    )
    client = FakeS3()
    monkeypatch.setattr(module.boto3, "client", lambda *args, **kwargs: client)
    return client, commands, work, project


def test_retry_preserves_completed_objects_and_uses_standard_profile(setup):
    client, commands, work, _ = setup
    client.fail_after = 2
    with pytest.raises(RuntimeError, match="injected"):
        module.deliver()
    assert len(client.writes) == 2
    assert not (work / "a10g-s3-receipt.json").exists()
    client.fail_after = None
    module.deliver()
    module.deliver()
    assert len(client.writes) == 6
    receipt = json.loads((work / "a10g-s3-receipt.json").read_text())
    assert receipt["canonical_s3_verified"] and len(receipt["files"]) == 6
    assert "--dry-run" in commands[-1][0]
    assert "--workflow-profile" not in commands[-1][0]


def test_conflicting_object_is_never_overwritten(setup):
    client, _, _, _ = setup
    key = f"snakemake/analysis/evals_v2/results/scores/{module.MODEL}/mendelian_traits.parquet"
    client.objects[key] = {"ContentLength": 3, "ChecksumSHA256": "wrong"}
    with pytest.raises(ValueError, match="Existing canonical output differs"):
        module.deliver()
    assert client.writes == []


def test_missing_output_prevents_any_transfer(setup):
    client, _, _, project = setup
    (project / f"results/metrics/{module.MODEL}/sge.parquet").unlink()
    with pytest.raises(AssertionError):
        module.deliver()
    assert client.writes == []


def test_failure_still_stops_worker(monkeypatch):
    calls = []
    monkeypatch.setattr(module.sys, "argv", ["finish-local-a10g.py", "--stop-after"])
    monkeypatch.setattr(
        module.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )

    def fail():
        raise RuntimeError("injected completion failure")

    monkeypatch.setattr(module, "deliver", fail)
    with pytest.raises(RuntimeError, match="injected"):
        module.main()
    assert calls == [["sudo", "shutdown", "-h", "now"]]
