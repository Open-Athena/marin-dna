"""issue #270: S3 <-> local staging for the scoring job, run as a SEPARATE process.

Run BEFORE (`--mode download`) and AFTER (`--mode upload`) `issue270_score_llr.py`
so the scoring process itself never initializes s3fs/fsspec in its parent. That
preserves DataLoader-fork safety for the lazy S3 genome reads — the same property
evals_v2 relies on (only the genome touches S3, lazily, inside each worker).
Pulling the checkpoint/variants in-process in the scoring parent would spin up
fsspec's async loop, whose background thread does not survive fork() -> the
workers deadlock on the genome byte-range reads.

    uv run python scripts/issue270_fetch.py --mode download \
        --ckpt-s3 s3://.../checkpoints/<model> --variants-s3 s3://.../pilot_variants.parquet
    uv run python scripts/issue270_fetch.py --mode upload \
        --scores-local pilot_scores.parquet --scores-s3 s3://.../pilot_scores.parquet
"""

from __future__ import annotations

import argparse
import os

import fsspec


def _key(s3_uri: str) -> str:
    assert s3_uri.startswith("s3://"), s3_uri
    return s3_uri[len("s3://") :]


def download(
    ckpt_s3: str, variants_s3: str, ckpt_local: str, variants_local: str
) -> None:
    fs = fsspec.filesystem("s3")
    os.makedirs(ckpt_local, exist_ok=True)
    prefix = _key(ckpt_s3).rstrip("/")
    for entry in fs.ls(prefix, detail=True):
        if entry["type"] != "file" or entry["size"] == 0:
            continue  # skip the dir marker and the 0-byte .snakemake_timestamp
        name = entry["name"].rstrip("/").split("/")[-1]
        fs.get_file(entry["name"], os.path.join(ckpt_local, name))
    fs.get_file(_key(variants_s3), variants_local)
    print(f"downloaded checkpoint -> {ckpt_local}/ ({sorted(os.listdir(ckpt_local))})")
    print(f"downloaded variants  -> {variants_local}")


def upload(scores_local: str, scores_s3: str) -> None:
    fsspec.filesystem("s3").put_file(scores_local, _key(scores_s3))
    print(f"uploaded {scores_local} -> {scores_s3}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["download", "upload"], required=True)
    ap.add_argument("--ckpt-s3")
    ap.add_argument("--variants-s3")
    ap.add_argument("--ckpt-local", default="ckpt")
    ap.add_argument("--variants-local", default="pilot_variants.parquet")
    ap.add_argument("--scores-local", default="pilot_scores.parquet")
    ap.add_argument("--scores-s3")
    args = ap.parse_args()

    if args.mode == "download":
        assert args.ckpt_s3 and args.variants_s3, (
            "download needs --ckpt-s3 and --variants-s3"
        )
        download(args.ckpt_s3, args.variants_s3, args.ckpt_local, args.variants_local)
    else:
        assert args.scores_s3, "upload needs --scores-s3"
        upload(args.scores_local, args.scores_s3)


if __name__ == "__main__":
    main()
