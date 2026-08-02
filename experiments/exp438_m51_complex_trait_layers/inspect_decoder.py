"""Inspect the decoder-space neighborhood of a post-hoc SAE candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from safetensors import safe_open

from common import D_SAE, ISSUE, sha256_file, write_json

EXPECTED_D_MODEL = 1_920


def decoder_neighborhood(
    weights_path: Path, feature_id: int, *, top_n: int
) -> pl.DataFrame:
    assert weights_path.is_file() and 0 <= feature_id < D_SAE and 0 < top_n <= D_SAE
    with safe_open(weights_path, framework="np") as handle:
        assert "W_dec" in handle.keys()
        decoder = handle.get_tensor("W_dec")
    assert decoder.shape == (D_SAE, EXPECTED_D_MODEL)
    assert np.isfinite(decoder).all()
    norms = np.linalg.norm(decoder, axis=1)
    assert np.all(norms > 0)
    cosine = (decoder @ decoder[feature_id]) / (norms * norms[feature_id])
    order = np.argsort(cosine)[::-1][:top_n]
    assert int(order[0]) == feature_id and np.isclose(cosine[feature_id], 1.0)
    return pl.DataFrame(
        {
            "rank": np.arange(1, top_n + 1, dtype=np.uint32),
            "feature_id": order.astype(np.uint32),
            "cosine": cosine[order],
            "decoder_norm": norms[order],
        }
    )


def run(
    weights_path: Path,
    output_dir: Path,
    *,
    feature_id: int,
    top_n: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    neighbors = decoder_neighborhood(weights_path, feature_id, top_n=top_n)
    neighbors.write_parquet(
        output_dir / "decoder_neighbors.parquet", compression="zstd"
    )
    metadata = {
        "issue": ISSUE,
        "analysis_status": "post_hoc_descriptive",
        "feature_id": feature_id,
        "weights_bytes": weights_path.stat().st_size,
        "weights_sha256": sha256_file(weights_path),
        "nearest_other_feature": neighbors.row(1, named=True),
        "top_n": top_n,
    }
    write_json(output_dir / "results.json", metadata)
    (output_dir / "RESULTS.md").write_text(
        "# SAE decoder-neighborhood audit\n\n"
        f"Feature: block 19 / 25M / {feature_id}\n\n"
        "This is a post-hoc geometric description, not biological validation.\n\n"
        "```json\n" + json.dumps(metadata, indent=2, sort_keys=True) + "\n```\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-id", type=int, default=1662)
    parser.add_argument("--top-n", type=int, default=200)
    args = parser.parse_args()
    run(
        args.weights,
        args.output_dir,
        feature_id=args.feature_id,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
