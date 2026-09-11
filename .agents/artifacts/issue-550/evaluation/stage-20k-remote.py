# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3==1.41.5", "PyYAML==6.0.3"]
# ///
"""Stage 20k on the existing free coordinator when the shared VM lacks headroom.

Only small metadata/RPC requests run locally. Checkpoint bytes stay remote.
The coordinator uses its normal GCS provider; five-minute, object-specific S3
PUT capabilities require both source MD5 and computed SHA-256 and forbid
replacement. Capabilities and raw remote stderr are never printed or persisted.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import boto3
import yaml
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[4]
MODEL = "dna-exp550-rag46m-five-regions-v1-step-20000"
TASK = "/gonzalo/dna-exp550-rag46m-five-regions-v1-20260911-europe-memory/0"
IRIS = "/tmp/issue550-iris-client/bin/iris"
BUCKET = "oa-bolinas"
PREFIX = f"snakemake/analysis/evals_v2/results/checkpoints/{MODEL}/"
FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")

PREPARE = r"""
import base64, hashlib, json
from pathlib import Path
import gcsfs

root = Path('/tmp/issue550-20k-stage')
root.mkdir(exist_ok=True)
fs = gcsfs.GCSFileSystem(version_aware=True)
source = payload['source'].removeprefix('gs://').rstrip('/')
info = {name: fs.info(source + '/' + name) for name in payload['files']}
assert sum(int(meta['size']) for meta in info.values()) <= 200_000_000
objects, verified = {}, {}
for name, meta in info.items():
    generation, expected_md5 = meta['generation'], meta['md5Hash']
    md5, sha256, size = hashlib.md5(), hashlib.sha256(), 0
    with fs.open(source + '/' + name, 'rb', generation=generation,
                 block_size=1024**2, cache_type='none') as src:
        with (root / name).open('wb') as dst:
            while chunk := src.read(1024**2):
                size += len(chunk)
                assert size <= int(meta['size'])
                md5.update(chunk)
                sha256.update(chunk)
                dst.write(chunk)
    actual_md5 = base64.b64encode(md5.digest()).decode()
    assert size == int(meta['size']) and actual_md5 == expected_md5
    objects[name] = {
        'name': meta['name'], 'generation': generation, 'size': size,
        'md5_hash': expected_md5, 'crc32c_hash': meta['crc32c'],
    }
    verified[name] = {'size': size, 'md5': actual_md5, 'sha256': sha256.hexdigest()}
config = json.loads((root / 'config.json').read_text())
for key, expected in payload['expected_config'].items():
    assert config.get(key) == expected, key
print(json.dumps({'source_objects': objects, 'verified_objects': verified}))
"""

UPLOAD = r"""
import hashlib, json
from pathlib import Path
import requests

root = Path('/tmp/issue550-20k-stage')
for item in payload:
    path = root / item['name']
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256']
    with path.open('rb') as stream:
        response = requests.put(item['url'], data=stream, headers=item['headers'], timeout=180)
    assert response.status_code == 200, 'Conditional checkpoint upload failed'
print(json.dumps({'uploaded': [item['name'] for item in payload]}))
"""


def remote(code: str, payload: dict | list) -> dict:
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    script = (
        "import base64,json\npayload=json.loads(base64.b64decode("
        + repr(encoded)
        + "))\ntry:\n"
        + "\n".join("    " + line for line in code.splitlines())
        + "\nexcept Exception as exc:\n    print(json.dumps({'error_type': type(exc).__name__}))\n"
    )
    result = subprocess.run(
        [
            IRIS,
            "--cluster",
            "marin",
            "task",
            "exec",
            TASK,
            "--timeout",
            "300",
            "--",
            "/app/.venv/bin/python",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=330,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("Remote staging RPC failed; raw output suppressed")
    report = json.loads(result.stdout)
    if "error_type" in report:
        raise RuntimeError("Remote staging failed: " + report["error_type"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    config = yaml.safe_load(
        (ROOT / "snakemake/analysis/evals_v2/config/config.yaml").read_text()
    )
    model = next(m for m in config["models"] if m["name"] == MODEL)
    assert model["gcs_path"].endswith("/2026.09.10.9/hf/step-20000")
    report = {
        "model": MODEL,
        "consumer_commit": commit,
        "source": model["gcs_path"],
        "destination": f"s3://{BUCKET}/{PREFIX}",
        "remote_task": TASK,
        "started_at": datetime.now(UTC).isoformat(),
        "applied": False,
        "exit_status": 1,
    }
    report.update(
        remote(
            PREPARE,
            {
                "source": report["source"],
                "files": FILES,
                "expected_config": {
                    "model_type": "qwen3",
                    "vocab_size": 8,
                    "hidden_size": 640,
                    "intermediate_size": 2560,
                    "num_hidden_layers": 7,
                    "num_attention_heads": 5,
                    "num_key_value_heads": 5,
                    "head_dim": 128,
                    "max_position_embeddings": 10240,
                    "tie_word_embeddings": False,
                },
            },
        )
    )
    client = boto3.client(
        "s3", region_name="us-east-2", config=Config(signature_version="s3v4")
    )
    listed = client.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX, MaxKeys=20)
    existing = {obj["Key"][len(PREFIX) :] for obj in listed.get("Contents", [])}
    assert not listed.get("IsTruncated") and existing <= set(FILES) | {
        ".snakemake_timestamp"
    }
    pending = []
    for name, meta in report["verified_objects"].items():
        checksum = base64.b64encode(bytes.fromhex(meta["sha256"])).decode()
        if name in existing:
            head = client.head_object(
                Bucket=BUCKET, Key=PREFIX + name, ChecksumMode="ENABLED"
            )
            assert (
                head["ContentLength"] == meta["size"]
                and head["ChecksumSHA256"] == checksum
            )
            continue
        params = {
            "Bucket": BUCKET,
            "Key": PREFIX + name,
            "ContentMD5": meta["md5"],
            "ChecksumSHA256": checksum,
            "IfNoneMatch": "*",
        }
        pending.append(
            {
                "name": name,
                "sha256": meta["sha256"],
                "url": client.generate_presigned_url(
                    "put_object", Params=params, ExpiresIn=300
                ),
                "headers": {
                    "Content-MD5": meta["md5"],
                    "x-amz-checksum-sha256": checksum,
                    "If-None-Match": "*",
                },
            }
        )
    if args.apply:
        if pending:
            remote(UPLOAD, pending)
        for name, meta in report["verified_objects"].items():
            head = client.head_object(
                Bucket=BUCKET, Key=PREFIX + name, ChecksumMode="ENABLED"
            )
            assert head["ContentLength"] == meta["size"]
            assert (
                head["ChecksumSHA256"]
                == base64.b64encode(bytes.fromhex(meta["sha256"])).decode()
            )
        if ".snakemake_timestamp" not in existing:
            client.put_object(
                Bucket=BUCKET,
                Key=PREFIX + ".snakemake_timestamp",
                Body=b"",
                IfNoneMatch="*",
            )
        report["applied"] = True
    report.update(exit_status=0, finished_at=datetime.now(UTC).isoformat())
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 -- redact provider errors
        # Neither signed URLs nor provider error strings belong in logs.
        print(json.dumps({"error_type": type(exc).__name__}))
        raise SystemExit(1) from None
