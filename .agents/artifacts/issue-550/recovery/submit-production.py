"""Submit the frozen issue-550 run after verified public inputs are committed."""

from __future__ import annotations

import argparse
import json
import netrc
import os
import re
import subprocess
from pathlib import Path

REGIONS = {
    "cds": 581256,
    "tss_utr5": 112740,
    "utr3": 131550,
    "ncrna": 56396,
    "enhancer": 228064,
}
PROJECT = "experiments/exp550_rag_five_regions"
MANIFEST = f"{PROJECT}/config/verified-public-datasets.json"
PILOT_CODE = "c0585d0e117075026b4384d552d59b52009d257e"
JOB_NAME = "dna-exp550-rag46m-five-regions-v1-20260910"
VERSION = "2026.09.10.5"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    ):
        raise SystemExit("Submission requires a clean committed snapshot")
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            PILOT_CODE,
            commit,
            "--",
            f"{PROJECT}/src",
            f"{PROJECT}/uv.lock",
            f"{PROJECT}/pyproject.toml",
        ],
        check=True,
    )
    payload = Path(MANIFEST).read_bytes()
    if payload != subprocess.check_output(["git", "show", f"{commit}:{MANIFEST}"]):
        raise SystemExit("The manifest differs from the committed snapshot")
    datasets = json.loads(payload)
    assert set(datasets) == set(REGIONS)
    for region, rows in REGIONS.items():
        spec = datasets[region]
        assert spec["repo_id"] == f"marin-dna/rag-five-regions-v1-{region}"
        assert re.fullmatch(r"[0-9a-f]{40}", spec["revision"])
        assert re.fullmatch(r"[0-9a-f]{64}", spec["release_manifest_sha256"])
        assert spec["train_rows"] == rows and spec["validation_rows"] == 400
        assert spec["anonymous_verified"] is True
    description = {
        "commit": commit,
        "manifest": MANIFEST,
        "job_name": JOB_NAME,
        "version": VERSION,
        "region": "us-east5",
        "accelerator": "v6e-8",
        "per_device": 5,
        "updates": 100000,
        "batch_documents": 200,
        "wandb": "gonzalobenegas/marin",
        "datasets": datasets,
    }
    print(json.dumps(description, indent=2), flush=True)
    if not args.run:
        return
    token = os.environ.get("WANDB_API_KEY")
    if not token:
        auth = netrc.netrc().authenticators("api.wandb.ai")
        token = auth[2] if auth else None
    if not token:
        raise SystemExit("W&B credential unavailable")
    command = [
        "/tmp/issue550-iris-client/bin/iris",
        "--cluster",
        "marin",
        "job",
        "run",
        "--no-wait",
        "--user",
        "gonzalo",
        "--priority",
        "batch",
        "--job-name",
        JOB_NAME,
        "--cpu",
        "1",
        "--memory",
        "2G",
        "--region",
        "us-east5",
        "--extra=tpu",
        "--no-sync",
        "--max-retries",
        "2",
    ]
    for pattern in (
        r"^docs/",
        r"^snakemake/",
        r"^tests/",
        r"^scripts/",
        r"^\.agents/",
        r"^src/",
    ):
        command.extend(["--exclude", pattern])
    environment = {
        "WANDB_API_KEY": token,
        "WANDB_ENTITY": "gonzalobenegas",
        "WANDB_PROJECT": "marin",
        "MARIN_PREFIX": "gs://marin-us-east5/MarinDNA/exp550_rag_five_regions",
        "UV_PROJECT": f"/app/{PROJECT}",
        "GIT_COMMIT": commit,
    }
    for name, value in environment.items():
        command.extend(["-e", name, value])
    command.extend(
        [
            "--",
            "bash",
            "-lc",
            f"export PATH=$HOME/.local/bin:$PATH && curl -LsSf https://astral.sh/uv/0.11.31/install.sh | sh && cd /app/{PROJECT} && uv sync --locked --extra tpu --link-mode symlink && exec uv run --locked python -m marin_dna_exp550.launch --dataset-manifest config/verified-public-datasets.json --per-device 5 --region us-east5 --version {VERSION} --run",
        ]
    )
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    output = (result.stdout + result.stderr).replace(token, "[REDACTED]")
    Path("/tmp/issue550-production-submission.log").write_text(output)
    print(output)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
