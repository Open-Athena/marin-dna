"""Run registered development VEP with local inputs and durable Iris staging.

Run in the locked evals_v2 environment on the organization's shared H100.
Only input downloads use scoped, short-lived AWS read URLs.
Outputs use the worker's normal CoreWeave storage provider and must subsequently
be copied, with checksum verification, into the canonical evals_v2 AWS paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import boto3
import pandas as pd
import torch
import yaml

MODEL = "dna-exp550-rag46m-five-regions-v1-step-10000"
COHORTS = {"mendelian_traits": 16140, "complex_traits": 11630, "sge": 23853}
BUCKET = "marin-us-east-02a"
BASELINE = "eddd3a6e1f4d379b9d2aeb9a8745e60ea87221bf"
ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / "snakemake/analysis/evals_v2"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    os.chdir(PROJECT)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == os.environ["GIT_COMMIT"]
    # Reuse the completed numerical gate only while all evaluation code and
    # locked dependencies are unchanged from its actual source checkout.
    subprocess.run(
        [
            "git", "diff", "--quiet", BASELINE, commit, "--",
            "snakemake/analysis/evals_v2", "src/marin_dna", "pyproject.toml",
            "uv.lock", ".agents/artifacts/issue-550/evaluation/recheck-final-checkpoint.py",
        ],
        cwd=ROOT,
        check=True,
    )
    artifacts = ROOT / ".agents/artifacts/issue-550/evaluation"
    stage = json.loads((artifacts / "step-10000-staged.json").read_text())
    parity = json.loads((artifacts / "step-10000-h100-parity.json").read_text())
    assert stage["applied"] and stage["exit_status"] == 0
    assert stage["model"] == parity["registered_model"] == MODEL
    assert parity["source_commit"] == BASELINE
    assert parity["completed"] and parity["parity_passed"]
    assert torch.__version__ == parity["runtime"]["torch"]
    assert torch.cuda.get_device_name() == parity["runtime"]["device"]
    torch.set_num_threads(2)

    config = yaml.safe_load(Path("config/config.yaml").read_text())
    registered = next(m for m in config["models"] if m["name"] == MODEL)
    assert config["split"] == "train"
    assert set(registered["datasets"]) == set(COHORTS)
    assert registered["gcs_path"] == stage["source"] == parity["checkpoint"]
    checkpoint = Path("results/checkpoints") / MODEL
    checkpoint.mkdir(parents=True, exist_ok=False)
    expected = {
        name: (checkpoint / name, meta["sha256"])
        for name, meta in stage["verified_objects"].items()
    }
    harness = Path("/tmp/issue550-combined-development.parquet")
    expected["harness"] = (harness, registered["rag_harness"]["sha256"])
    entries = json.loads(os.environ.pop("ISSUE550_INPUT_URLS"))
    assert len(entries) == len(expected)
    assert {entry["name"] for entry in entries} == set(expected)
    for entry in entries:
        path, sha256 = expected[entry["name"]]
        assert entry["sha256"] == sha256
        try:
            with urllib.request.urlopen(entry["url"], timeout=120) as source:
                with path.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        except Exception as exc:
            raise RuntimeError(
                f"Input download failed for {entry['name']}: {type(exc).__name__}"
            ) from None
        assert digest(path) == sha256, f"SHA-256 mismatch: {entry['name']}"
    del entries
    (checkpoint / ".snakemake_timestamp").touch()

    # Only transport location changes; the registered harness SHA-256 and all
    # model, cohort, precision, bootstrap, and scoring contracts stay fixed.
    registered["rag_harness"]["uri"] = str(harness)
    overlay = Path("/tmp/issue550-local-inputs.yaml")
    overlay.write_text(yaml.safe_dump({"models": config["models"]}))
    prefix = f"marin/MarinDNA/exp550_rag_five_regions/evaluation-staging/{MODEL}/{commit}"
    client = boto3.client("s3")
    canary = f"{prefix}/storage-preflight.txt"
    client.put_object(Bucket=BUCKET, Key=canary, Body=b"issue550-storage-ready\n")
    assert client.get_object(Bucket=BUCKET, Key=canary)["Body"].read() == b"issue550-storage-ready\n"
    receipt: dict = {
        "model": MODEL, "consumer_commit": commit, "parity_source_commit": BASELINE,
        "checkpoint_sha256": stage["verified_objects"]["model.safetensors"]["sha256"],
        "harness_sha256": registered["rag_harness"]["sha256"],
        "split": "train", "cohort_sizes": COHORTS,
        "staging_prefix": f"s3://{BUCKET}/{prefix}/",
        "canonical_prefix": "s3://oa-bolinas/snakemake/analysis/evals_v2/",
        "canonical_published": False, "started_at_unix": time.time(), "files": {},
    }

    def save_receipt() -> None:
        data = (json.dumps(receipt, indent=2) + "\n").encode()
        client.put_object(Bucket=BUCKET, Key=f"{prefix}/receipt.json", Body=data)

    def upload(path: Path) -> None:
        sha256 = digest(path)
        key = f"{prefix}/{path.as_posix()}"
        client.upload_file(str(path), BUCKET, key, ExtraArgs={"Metadata": {"sha256": sha256}})
        head = client.head_object(Bucket=BUCKET, Key=key)
        assert head["ContentLength"] == path.stat().st_size
        assert head["Metadata"]["sha256"] == sha256
        receipt["files"][path.as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256}
        save_receipt()

    save_receipt()
    command = [
        sys.executable, "-m", "snakemake", "--workflow-profile", "none",
        "--cores", "4", "--rerun-incomplete", "--configfiles",
        "config/config.yaml", "config/rag_issue550/fp32.yaml", str(overlay),
    ]
    scores = [Path(f"results/scores/{MODEL}/{name}.parquet") for name in COHORTS]
    metrics = [Path(f"results/metrics/{MODEL}/{name}.parquet") for name in COHORTS]
    # Dry-run explicit cells after all inputs and durable storage are ready.
    subprocess.run(command + ["--dry-run", "--"] + [str(p) for p in metrics], check=True)
    subprocess.run(command + ["--"] + [str(p) for p in scores], check=True)
    for name, path in zip(COHORTS, scores, strict=True):
        frame = pd.read_parquet(path)
        assert len(frame) == COHORTS[name]
        assert {"emb_ref", "emb_alt", "llr_fwd", "llr_rc"} <= set(frame.columns)
        del frame
        upload(path)
    receipt["scores_complete"] = True
    save_receipt()
    print("SCORES_DURABLE " + receipt["staging_prefix"], flush=True)
    subprocess.run(command + ["--"] + [str(p) for p in metrics], check=True)
    for path in metrics:
        upload(path)
    receipt["completed"] = True
    receipt["finished_at_unix"] = time.time()
    save_receipt()
    print("VEP_DEVELOPMENT_COMPLETE " + json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
