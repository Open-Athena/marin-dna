"""Observable Framework data loader: embedding-UMAP plots → one zip blob.

Emits a zip on stdout containing every materialized embedding-UMAP SVG
(``{model}/{region,conservation}.svg``) plus a ``manifest.json`` describing them
(model display names). The page (``src/interpretation/embedding-umap.md``) reads
it via ``FileAttachment("../data/umap.zip").zip()`` and renders the region +
conservation panels per model.

Artifact keys come from the evals_v2 ``umap_embeddings`` config (issue #246), not
S3 enumeration: the dashboard CI role has S3 ``GetObject`` but no ``ListBucket``.
Each candidate SVG is fetched; ones not yet on S3 are skipped — a 404, or
(without ListBucket) a 403 for a missing key. We assert at least one was fetched,
so an IAM/path misconfiguration fails the build loudly instead of silently
shipping an empty section. Mirrors ``nuc_dep.zip.py``.
"""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from typing import Any

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

from marin_dna.pipelines.evals.interpretation_catalog import (
    S3_BUCKET,
    UMAP_PLOT_PREFIX,
    load_umap_block,
    model_display_map,
    umap_candidates,
)

# Codes meaning "not a readable present key" → not materialized yet, skip.
# Without ListBucket, S3 answers a missing key with 403 (not 404), so both.
_SKIP_CODES = {"NoSuchKey", "404", "403", "AccessDenied"}


def _s3_client() -> Any:
    """S3 client on the bucket's region. ``BOLINAS_S3_ANON=1`` → unsigned reads;
    else the standard cred chain (env → ``~/.aws`` → IMDS / GitHub OIDC in CI)."""
    if os.environ.get("BOLINAS_S3_ANON") in ("1", "true"):
        return boto3.client(
            "s3", region_name="us-east-2", config=Config(signature_version=UNSIGNED)
        )
    return boto3.client("s3", region_name="us-east-2")


def main() -> None:
    block = load_umap_block()
    candidates = umap_candidates(block, model_displays=model_display_map())
    s3 = _s3_client()

    buf = io.BytesIO()
    manifest: list[dict[str, Any]] = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for cand in candidates:
            # `svg` is the canonical relative path; the S3 key prepends the prefix.
            key = f"{UMAP_PLOT_PREFIX}/{cand['svg']}"
            try:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in _SKIP_CODES:
                    print(
                        f"  ! umap skip (not materialized): {key} [{code}]",
                        file=sys.stderr,
                    )
                    continue
                raise
            zf.writestr(cand["svg"], obj["Body"].read())
            manifest.append(cand)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    assert manifest, (
        f"fetched 0 embedding-UMAP SVGs from s3://{S3_BUCKET}/{UMAP_PLOT_PREFIX}/ "
        f"— expected ≥1. Check the evals_v2 `umap_embeddings` config and that the "
        f"dashboard IAM role can read the plots/umap prefix."
    )
    print(
        f"[umap] zipped {len(manifest)} SVG(s) of {len(candidates)} candidate(s)",
        file=sys.stderr,
    )
    sys.stdout.buffer.write(buf.getvalue())


if __name__ == "__main__":
    main()
