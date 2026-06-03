"""Observable Framework data loader: nucleotide-dependency maps → one zip blob.

Emits a zip on stdout containing every materialized dependency-map heatmap SVG
(``{combine}/{locus}/{model}.svg``) plus a ``manifest.json`` describing them
(per-locus metadata, UCSC links, paper references). The page
(``src/interpretation/nucleotide-dependency.md``) reads it via
``FileAttachment("../data/nuc_dep.zip").zip()`` and renders a selector over the
manifest, pulling the selected SVG from the same archive.

Artifact keys come from the evals_v2 ``nuc_dep`` config (issue #237/#238), not
S3 enumeration: the dashboard CI role has S3 ``GetObject`` but no
``ListBucket``. Each candidate SVG is fetched; ones not yet on S3 are skipped —
a 404, or (without ListBucket) a 403 for a missing key. We assert at least one
was fetched, so an IAM/path misconfiguration fails the build loudly instead of
silently shipping an empty section.
"""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

from marin_dna.pipelines.evals.interpretation_catalog import (
    NUC_DEP_PLOT_PREFIX,
    S3_BUCKET,
    load_nuc_dep_block,
    model_display_map,
    nuc_dep_candidates,
    s3_key_for,
)

# Codes meaning "not a readable present key" → not materialized yet, skip.
# Without ListBucket, S3 answers a missing key with 403 (not 404), so both.
_SKIP_CODES = {"NoSuchKey", "404", "403", "AccessDenied"}


def _s3_client() -> Any:
    """S3 client on the bucket's region. ``BOLINAS_S3_ANON=1`` → unsigned reads
    (mirrors ``leaderboard._storage_options``); else the standard cred chain
    (env → ``~/.aws`` → IMDS / GitHub OIDC in CI)."""
    if os.environ.get("BOLINAS_S3_ANON") in ("1", "true"):
        return boto3.client(
            "s3", region_name="us-east-2", config=Config(signature_version=UNSIGNED)
        )
    return boto3.client("s3", region_name="us-east-2")


def main() -> None:
    block = load_nuc_dep_block()
    refs_dir = Path(__file__).resolve().parent.parent / "interpretation" / "refs"
    candidates = nuc_dep_candidates(
        block, model_displays=model_display_map(), refs_dir=refs_dir
    )
    s3 = _s3_client()

    buf = io.BytesIO()
    manifest: list[dict[str, Any]] = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for cand in candidates:
            key = s3_key_for(cand["combine"], cand["locus"], cand["model"])
            try:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in _SKIP_CODES:
                    print(
                        f"  ! nuc_dep skip (not materialized): {key} [{code}]",
                        file=sys.stderr,
                    )
                    continue
                raise
            zf.writestr(cand["svg"], obj["Body"].read())
            manifest.append(cand)
        # Bundle the committed paper-reference screenshots into the same zip:
        # Observable's build only copies statically-referenced files, not ones
        # behind a runtime `<img src>`, so we serve them from the archive like
        # the SVGs. The manifest's `paper.image` is the zip key (e.g.
        # `refs/LDLR.png`); read it back from `refs_dir`.
        seen_imgs: set[str] = set()
        for cand in manifest:
            img = (cand.get("paper") or {}).get("image")
            if img and img not in seen_imgs:
                src = refs_dir / Path(img).name
                if src.is_file():
                    zf.writestr(img, src.read_bytes())
                    seen_imgs.add(img)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    assert manifest, (
        f"fetched 0 nucleotide-dependency SVGs from "
        f"s3://{S3_BUCKET}/{NUC_DEP_PLOT_PREFIX}/ — expected ≥1. Check the "
        f"evals_v2 `nuc_dep` config and that the dashboard IAM role can read the "
        f"plots/nuc_dep prefix."
    )
    print(
        f"[nuc_dep] zipped {len(manifest)} SVG(s) of {len(candidates)} candidate(s)",
        file=sys.stderr,
    )
    sys.stdout.buffer.write(buf.getvalue())


if __name__ == "__main__":
    main()
