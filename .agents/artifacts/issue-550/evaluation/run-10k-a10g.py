"""Run the registered 10k development targets with measured A10G BF16 settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

MODEL = "dna-exp550-rag46m-five-regions-v1-step-10000"
COHORTS = {"mendelian_traits": 16140, "complex_traits": 11630, "sge": 23853}
ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / "snakemake/analysis/evals_v2"
WORK = Path("/opt/issue550")
STORAGE = WORK / "storage"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    os.chdir(PROJECT)
    sweep = json.loads((WORK / "batch-sweep/summary.json").read_text())
    assert sweep["completed"] and sweep["synthetic_only"]
    subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            sweep["source_commit"],
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
    batch = sweep["selected_batch"]
    selected = next(m for m in sweep["measurements"] if m["batch"] == batch)
    assert selected["finite_outputs"] and selected["prefix_cache"]
    assert selected["bf16"] and selected["compiled"] and selected["embeddings"]
    # Include 30% inference margin and 30 minutes for metric jobs and uploads.
    estimated_seconds = sweep["projected_51623_variant_hours"] * 3600 * 1.3 + 1800
    assert time.time() + estimated_seconds < 1789164043, (
        "Runtime exceeds 6 p.m. NYC shutdown"
    )
    config = yaml.safe_load(Path("config/config.yaml").read_text())
    assert config["split"] == "train"
    registered = next(m for m in config["models"] if m["name"] == MODEL)
    assert set(registered["datasets"]) == set(COHORTS)
    registered["batch_size"] = batch
    registered["eval_accumulation_steps"] = 8
    inference = config["inference"]
    assert (
        inference["bf16"]
        and inference["torch_compile"]
        and inference["return_embeddings"]
    )
    inference["num_workers"] = 2
    overlay = WORK / "a10g-bf16.yaml"
    overlay.write_text(
        yaml.safe_dump({"models": config["models"], "inference": inference})
    )
    cache = (
        STORAGE
        / "s3/oa-bolinas/snakemake/analysis/evals_v2/results/checkpoints"
        / MODEL
    )
    if args.local_only:
        cache = PROJECT / "results/checkpoints" / MODEL
    cache.mkdir(parents=True, exist_ok=True)
    stage = json.loads(
        (
            ROOT / ".agents/artifacts/issue-550/evaluation/step-10000-staged.json"
        ).read_text()
    )
    assert stage["applied"] and stage["exit_status"] == 0 and stage["model"] == MODEL
    assert registered["gcs_path"] == stage["source"]
    for name, meta in stage["verified_objects"].items():
        source = WORK / "checkpoint" / name
        with source.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == meta["sha256"]
        shutil.copyfile(source, cache / name)
    shutil.copyfile(
        WORK / "checkpoint/.snakemake_timestamp", cache / ".snakemake_timestamp"
    )
    command = [
        sys.executable,
        "-m",
        "snakemake",
        "--cores",
        "2",
        "--rerun-incomplete",
        "--keep-storage-local-copies",
        "--local-storage-prefix",
        str(STORAGE),
        "--configfiles",
        "config/config.yaml",
        str(overlay),
    ]
    if args.local_only:
        command[3:3] = ["--workflow-profile", "none"]
    targets = [f"results/metrics/{MODEL}/{name}.parquet" for name in COHORTS]
    subprocess.run(command + ["--dry-run", "--"] + targets, check=True)
    if not args.execute:
        return
    receipt = {
        "model": MODEL,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "split": "train",
        "cohort_sizes": COHORTS,
        "batch": batch,
        "bf16": True,
        "torch_compile": True,
        "prefix_cache": True,
        "left_padded_tokens": 10240,
        "human_variant_position": 10112,
        "embeddings": True,
        "started_at_unix": time.time(),
        "checkpoint_sha256": stage["verified_objects"]["model.safetensors"]["sha256"],
        "harness_sha256": registered["rag_harness"]["sha256"],
        "completed": False,
        "publication": "local_only" if args.local_only else "canonical_s3",
    }
    receipt_path = WORK / "a10g-run-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    result = subprocess.run(command + ["--"] + targets, check=False)
    receipt["exit_status"] = result.returncode
    receipt["finished_at_unix"] = time.time()
    receipt["completed"] = result.returncode == 0
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    result.check_returncode()
    print("DEVELOPMENT_VEP_COMPLETE " + json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
