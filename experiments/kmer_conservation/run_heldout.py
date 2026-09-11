"""Execute the published selection once, preserving the exact command receipt."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-commit", required=True)
    parser.add_argument("--mmseqs", type=Path, required=True)
    args = parser.parse_args()
    assert re.fullmatch(r"[a-f0-9]{40}", args.selection_commit)
    assert not list((args.root / "results").glob("heldout-*")), (
        "Held-out results already exist"
    )
    selection_bytes = args.selection.read_bytes()
    cfg = json.loads(selection_bytes)
    assert cfg["decision"] != "PENDING COMPLETED DEVELOPMENT MATRIX"
    fixture = json.loads((args.root / "data/fixture_manifest.json").read_text())
    assert cfg["contexts_sha256"] == fixture["contexts_sha256"]
    common = ["--root", str(args.root), "--split", "heldout"]
    frozen = ["--frozen-selection", str(args.selection)]
    commands = []
    for width, k, divisor, mask in cfg["allowed_settings"]:
        commands.append(
            [
                "uv",
                "run",
                "--locked",
                "kmer-screen",
                *common,
                *frozen,
                "--width",
                str(width),
                "--k",
                str(k),
                "--divisor",
                str(divisor),
                *(["--mask"] if mask else []),
            ]
        )
    for width, k, divisor, mask, hashes, method, rows in cfg["allowed_sketches"]:
        commands.append(
            [
                "uv",
                "run",
                "--locked",
                "python",
                "-m",
                "kmer_conservation.sketch",
                *common,
                *frozen,
                "--width",
                str(width),
                "--k",
                str(k),
                "--divisor",
                str(divisor),
                "--hashes",
                str(hashes),
                "--method",
                method,
                "--rows",
                str(rows),
                *(["--mask"] if mask else []),
            ]
        )
    for width, k, divisor in cfg["linclust"]:
        commands.append(
            [
                "uv",
                "run",
                "--locked",
                "python",
                "-m",
                "kmer_conservation.linclust",
                *common,
                *frozen,
                "--width",
                str(width),
                "--k",
                str(k),
                "--divisor",
                str(divisor),
                "--mmseqs",
                str(args.mmseqs),
            ]
        )
    commands.append(
        [
            "uv",
            "run",
            "--locked",
            "python",
            "-m",
            "kmer_conservation.diagnostics",
            *common,
            "--mode",
            "union",
        ]
    )
    for width, k in [(255, 9), (1024, 13)]:
        commands.append(
            [
                "uv",
                "run",
                "--locked",
                "python",
                "-m",
                "kmer_conservation.diagnostics",
                *common,
                "--mode",
                "verify",
                "--width",
                str(width),
                "--k",
                str(k),
            ]
        )
    receipt = {
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "selection_commit": args.selection_commit,
        "commands": commands,
        "started_epoch": time.time(),
        "completed": [],
    }
    path = args.root / "heldout_execution.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "2",
    }
    for number, command in enumerate(commands):
        subprocess.run(command, check=True, env=env)
        receipt["completed"].append(
            {"command_index": number, "ended_epoch": time.time()}
        )
        path.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt["ended_epoch"] = time.time()
    path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
