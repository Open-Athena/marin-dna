"""Freeze decoder-neighborhood candidates across independently trained SAEs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import polars as pl
import torch
from safetensors.torch import load_file

ISSUE = 432
D_IN = 1_920
D_SAE = 15_360
TOP_K = 32
REFERENCE_QUERIES = (
    ("splice_acceptor", 11_698),
    ("splice_donor", 11_681),
    ("stop_creation_positive", 3_312),
    ("stop_creation_negative", 4_281),
    ("synonymous_degeneracy", 6_072),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def normalized_decoder(sae_dir: Path) -> torch.Tensor:
    weights_path = sae_dir / "sae_weights.safetensors"
    cfg_path = sae_dir / "cfg.json"
    assert weights_path.is_file() and cfg_path.is_file()
    cfg = json.loads(cfg_path.read_text())
    assert cfg["d_in"] == D_IN and cfg["d_sae"] == D_SAE
    assert cfg["normalize_activations"] == "none"
    weights = load_file(weights_path, device="cpu")["W_dec"].float()
    assert weights.shape == (D_SAE, D_IN) and torch.isfinite(weights).all()
    norms = torch.linalg.vector_norm(weights, dim=1)
    assert torch.all(norms > 0)
    normalized = weights / norms[:, None]
    torch.testing.assert_close(
        torch.linalg.vector_norm(normalized, dim=1),
        torch.ones(D_SAE),
        rtol=1e-5,
        atol=1e-6,
    )
    return normalized


def decoder_neighbors(
    reference_sae: Path,
    candidate_sae: Path,
    *,
    dictionary_name: str,
    top_k: int = TOP_K,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    assert dictionary_name and 0 < top_k < D_SAE
    reference = normalized_decoder(reference_sae)
    candidate = normalized_decoder(candidate_sae)
    rows: list[dict[str, Any]] = []
    for concept, query_feature in REFERENCE_QUERIES:
        similarities = candidate @ reference[query_feature]
        values, indices = torch.topk(similarities, k=top_k, largest=True, sorted=True)
        candidate_rows = candidate.index_select(0, indices)
        reverse_similarities = candidate_rows @ reference.T
        reverse_values, reverse_indices = reverse_similarities.max(dim=1)
        for rank, (feature_id, cosine, reverse_id, reverse_cosine) in enumerate(
            zip(indices, values, reverse_indices, reverse_values, strict=True), start=1
        ):
            rows.append(
                {
                    "dictionary": dictionary_name,
                    "concept": concept,
                    "reference_feature_id": query_feature,
                    "candidate_feature_id": int(feature_id),
                    "candidate_rank": rank,
                    "decoder_cosine": float(cosine),
                    "candidate_nearest_reference_feature_id": int(reverse_id),
                    "candidate_nearest_reference_cosine": float(reverse_cosine),
                    "mutual_nearest": int(reverse_id) == query_feature and rank == 1,
                }
            )
    frame = pl.DataFrame(rows).sort(["concept", "candidate_rank"])
    assert frame.height == len(REFERENCE_QUERIES) * top_k
    assert frame.select("candidate_feature_id").n_unique() <= frame.height
    assert frame["decoder_cosine"].is_finite().all()
    summary = {
        "issue": ISSUE,
        "experiment_commit": os.environ.get("EXPERIMENT_COMMIT", ""),
        "dictionary": dictionary_name,
        "top_k": top_k,
        "reference_queries": [
            {"concept": concept, "feature_id": feature_id}
            for concept, feature_id in REFERENCE_QUERIES
        ],
        "reference_sae": {
            "path": str(reference_sae),
            "weights_sha256": sha256_file(reference_sae / "sae_weights.safetensors"),
            "cfg_sha256": sha256_file(reference_sae / "cfg.json"),
        },
        "candidate_sae": {
            "path": str(candidate_sae),
            "weights_sha256": sha256_file(candidate_sae / "sae_weights.safetensors"),
            "cfg_sha256": sha256_file(candidate_sae / "cfg.json"),
        },
        "unique_candidate_features": frame.select("candidate_feature_id").n_unique(),
        "top1": frame.filter(pl.col("candidate_rank") == 1)
        .select(
            "concept",
            "reference_feature_id",
            "candidate_feature_id",
            "decoder_cosine",
            "candidate_nearest_reference_feature_id",
            "candidate_nearest_reference_cosine",
            "mutual_nearest",
        )
        .to_dicts(),
    }
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-sae", type=Path, required=True)
    parser.add_argument("--candidate-sae", type=Path, required=True)
    parser.add_argument("--dictionary-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    assert not args.output_dir.exists()
    frame, summary = decoder_neighbors(
        args.reference_sae,
        args.candidate_sae,
        dictionary_name=args.dictionary_name,
        top_k=args.top_k,
    )
    args.output_dir.mkdir(parents=True)
    candidates_path = args.output_dir / "decoder_candidates.parquet"
    frame.write_parquet(candidates_path)
    summary["artifacts"] = {
        candidates_path.name: {
            "bytes": candidates_path.stat().st_size,
            "sha256": sha256_file(candidates_path),
        }
    }
    write_json(args.output_dir / "manifest.json", summary)
    print(json.dumps(summary["top1"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
