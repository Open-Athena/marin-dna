"""Recover this completed local run into the standard Snakemake S3 outputs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import boto3
import pyarrow.parquet as pq
import yaml

MODEL = "dna-exp550-rag46m-five-regions-v1-step-10000"
COHORTS = {"mendelian_traits": 16140, "complex_traits": 11630, "sge": 23853}
ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / "snakemake/analysis/evals_v2"
WORK = Path("/opt/issue550")


def deliver() -> None:
    receipt = json.loads((WORK / "a10g-run-receipt.json").read_text())
    assert receipt["completed"] and receipt["exit_status"] == 0
    assert receipt["model"] == MODEL and receipt["split"] == "train"
    assert receipt["cohort_sizes"] == COHORTS
    assert receipt["publication"] == "local_only"
    stage = json.loads((Path(__file__).parent / "step-10000-staged.json").read_text())
    assert (
        receipt["checkpoint_sha256"]
        == stage["verified_objects"]["model.safetensors"]["sha256"]
    )
    config = yaml.safe_load((PROJECT / "config/config.yaml").read_text())
    registered = next(m for m in config["models"] if m["name"] == MODEL)
    assert receipt["harness_sha256"] == registered["rag_harness"]["sha256"]
    subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            receipt["source_commit"],
            "HEAD",
            "--",
            "snakemake/analysis/evals_v2",
            "src/marin_dna",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=ROOT,
        check=True,
    )
    profile = yaml.safe_load(
        (PROJECT / "workflow/profiles/default/config.yaml").read_text()
    )
    assert profile["default-storage-provider"] == "s3"
    prefix = profile["default-storage-prefix"]
    assert prefix == "s3://oa-bolinas/snakemake/analysis/evals_v2/"
    destination = urlsplit(prefix)
    # Reuse the existing, tested conditional-upload checksum checks.
    spec = importlib.util.spec_from_file_location(
        "issue550_publisher", Path(__file__).with_name("publish-shared-results.py")
    )
    assert spec and spec.loader
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    files = {}
    for kind in ("scores", "metrics"):
        for cohort, count in COHORTS.items():
            relative = f"results/{kind}/{MODEL}/{cohort}.parquet"
            path = PROJECT / relative
            assert path.is_file() and not path.is_symlink()
            size = path.stat().st_size
            assert 0 < size <= 256 * 1024**2
            parquet = pq.ParquetFile(path)
            assert parquet.metadata.num_rows > 0
            if kind == "scores":
                assert parquet.metadata.num_rows == count
                assert {"emb_ref", "emb_alt", "llr_fwd", "llr_rc"} <= set(
                    parquet.schema_arrow.names
                )
            with path.open("rb") as stream:
                sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            files[relative] = {"bytes": size, "sha256": sha256}
    client = boto3.client("s3", region_name="us-east-2")
    for relative, meta in files.items():
        if not publisher.existing_matches(client, relative, meta):
            with (PROJECT / relative).open("rb") as stream:
                client.put_object(
                    Bucket=destination.netloc,
                    Key=destination.path.lstrip("/") + relative,
                    Body=stream,
                    ContentLength=meta["bytes"],
                    IfNoneMatch="*",
                    ChecksumSHA256=base64.b64encode(
                        bytes.fromhex(meta["sha256"])
                    ).decode(),
                    Metadata={"sha256": meta["sha256"]},
                )
        assert publisher.existing_matches(client, relative, meta)
        print("SNAKEMAKE_OUTPUT_SAVED " + prefix + relative, flush=True)
    receipt["canonical_s3_verified"] = True
    receipt["canonical_prefix"] = prefix
    receipt["files"] = files
    (WORK / "a10g-s3-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    command = [
        sys.executable,
        "-m",
        "snakemake",
        "--dry-run",
        "--cores",
        "2",
        "--local-storage-prefix",
        str(WORK / "storage"),
        "--configfiles",
        "config/config.yaml",
        str(WORK / "a10g-bf16.yaml"),
        "--",
        *[f"results/metrics/{MODEL}/{cohort}.parquet" for cohort in COHORTS],
    ]
    subprocess.run(command, cwd=PROJECT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-after", action="store_true")
    args = parser.parse_args()
    try:
        deliver()
    finally:
        if args.stop_after:
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)


if __name__ == "__main__":
    main()
