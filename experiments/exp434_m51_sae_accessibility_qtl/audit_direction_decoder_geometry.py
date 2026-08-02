"""Audit encoder/decoder geometry of correlated block-19 dsQTL hits.

This post-hoc descriptive pass asks whether the response-family redundancy found
by ``cross_reference_direction_hits.py`` is explained by copied or split SAE
directions. It reads only the selected rows/columns from the exact pinned
safetensors checkpoint and never runs the language model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr

ISSUE = 434
ARM = "block19-25m"
EXPECTED_WEIGHTS_BYTES = 236_060_560
EXPECTED_WEIGHTS_SHA256 = (
    "e4f10ba59f10be943dbdc33f469f986f598c5e34fcba42577efad27717231533"
)
COMPONENT_THRESHOLD = 0.7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def assert_commit(value: str) -> None:
    assert len(value) == 40
    assert all(character in "0123456789abcdef" for character in value)


def verify_manifest_artifact(
    root: Path, manifest: dict[str, Any], relative_path: str
) -> None:
    expected = manifest["artifacts"][relative_path]
    path = root / relative_path
    assert path.is_file(), path
    assert path.stat().st_size == expected["bytes"]
    assert sha256_file(path) == expected["sha256"]


def safetensor_array(path: Path, name: str) -> np.memmap:
    """Return a read-only memory map for one little-endian F32 tensor."""
    with path.open("rb") as handle:
        header_bytes = handle.read(8)
        assert len(header_bytes) == 8
        (header_length,) = struct.unpack("<Q", header_bytes)
        assert 0 < header_length < path.stat().st_size
        header = json.loads(handle.read(header_length))
    tensor = header[name]
    assert tensor["dtype"] == "F32"
    shape = tuple(int(size) for size in tensor["shape"])
    start, stop = (int(offset) for offset in tensor["data_offsets"])
    assert stop - start == int(np.prod(shape)) * 4
    return np.memmap(
        path,
        dtype="<f4",
        mode="r",
        offset=8 + header_length + start,
        shape=shape,
    )


def cosine_matrix(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert vectors.ndim == 2 and vectors.shape[0] >= 2
    assert np.isfinite(vectors).all()
    norms = np.linalg.norm(vectors, axis=1)
    assert np.all(norms > 0)
    normalized = vectors / norms[:, None]
    cosine = normalized @ normalized.T
    assert np.isfinite(cosine).all()
    assert np.allclose(np.diag(cosine), 1, atol=1e-10)
    return cosine, norms


def geometry_summary(cosine: np.ndarray) -> dict[str, Any]:
    assert cosine.ndim == 2 and cosine.shape[0] == cosine.shape[1]
    eigenvalues = np.clip(np.linalg.eigvalsh(cosine), 0, None)
    off_diagonal = cosine[np.triu_indices(cosine.shape[0], k=1)]
    absolute = np.abs(off_diagonal)
    return {
        "features": cosine.shape[0],
        "effective_rank": float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()),
        "top_eigenvalue_share": float(eigenvalues[-1] / eigenvalues.sum()),
        "pairwise_cosine": {
            "minimum": float(off_diagonal.min()),
            "median": float(np.median(off_diagonal)),
            "maximum": float(off_diagonal.max()),
        },
        "absolute_pairwise_cosine_quantiles": {
            str(quantile): float(np.quantile(absolute, quantile))
            for quantile in (0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "absolute_cosine_threshold_counts": {
            str(threshold): int(np.count_nonzero(absolute >= threshold))
            for threshold in (0.5, 0.7, 0.8, 0.9)
        },
    }


def response_geometry_association(
    pairs: pl.DataFrame, geometry_column: str
) -> dict[str, float]:
    response = pairs["absolute_response_pearson"].to_numpy()
    geometry = pairs[geometry_column].to_numpy()
    linear = pearsonr(response, geometry)
    rank = spearmanr(response, geometry)
    return {
        "pearson_r": float(linear.statistic),
        "pearson_p": float(linear.pvalue),
        "spearman_rho": float(rank.statistic),
        "spearman_p": float(rank.pvalue),
    }


def build_pair_table(
    *,
    feature_ids: list[int],
    decoder_cosine: np.ndarray,
    encoder_cosine: np.ndarray,
    response_pairs: pl.DataFrame,
    components: pl.DataFrame,
) -> pl.DataFrame:
    assert feature_ids == sorted(feature_ids)
    assert (
        decoder_cosine.shape
        == encoder_cosine.shape
        == (
            len(feature_ids),
            len(feature_ids),
        )
    )
    component = components.filter(
        pl.col("absolute_pearson_threshold") == COMPONENT_THRESHOLD
    ).select("feature_id", "component_index", "component_size")
    assert component.height == len(feature_ids)
    giant = set(
        component.filter(pl.col("component_size") == pl.col("component_size").max())[
            "feature_id"
        ].to_list()
    )
    assert len(giant) == 36

    rows = []
    for left_index, left in enumerate(feature_ids):
        for right_index in range(left_index + 1, len(feature_ids)):
            right = feature_ids[right_index]
            if left in giant and right in giant:
                category = "within_giant_component"
            elif left in giant or right in giant:
                category = "giant_to_singleton"
            else:
                category = "between_singletons"
            rows.append(
                {
                    "feature_id_left": left,
                    "feature_id_right": right,
                    "pair_category": category,
                    "decoder_cosine": decoder_cosine[left_index, right_index],
                    "absolute_decoder_cosine": abs(
                        decoder_cosine[left_index, right_index]
                    ),
                    "encoder_cosine": encoder_cosine[left_index, right_index],
                    "absolute_encoder_cosine": abs(
                        encoder_cosine[left_index, right_index]
                    ),
                }
            )
    pairs = pl.DataFrame(rows).join(
        response_pairs.select(
            "feature_id_left",
            "feature_id_right",
            pl.col("pearson").alias("response_pearson"),
            pl.col("absolute_pearson").alias("absolute_response_pearson"),
            pl.col("spearman").alias("response_spearman"),
            pl.col("absolute_spearman").alias("absolute_response_spearman"),
        ),
        on=["feature_id_left", "feature_id_right"],
        how="inner",
    )
    assert pairs.height == len(feature_ids) * (len(feature_ids) - 1) // 2
    assert pairs.null_count().to_numpy().sum() == 0
    return pairs.sort("absolute_response_pearson", descending=True)


def run(
    *, cross_reference_root: Path, weights_path: Path, output_dir: Path
) -> dict[str, Any]:
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    run_id = os.environ.get("RUN_ID", "")
    assert run_id

    manifest_path = cross_reference_root / "manifest.json"
    cross_manifest = json.loads(manifest_path.read_text())
    assert cross_manifest["issue"] == ISSUE
    assert cross_manifest["analysis"] == (
        "post-hoc cross-dataset response-family triage"
    )
    for relative in (
        "qtl_hits.parquet",
        "forward_pair_correlations.parquet",
        "forward_response_components.parquet",
    ):
        verify_manifest_artifact(cross_reference_root, cross_manifest, relative)
    assert weights_path.stat().st_size == EXPECTED_WEIGHTS_BYTES
    assert sha256_file(weights_path) == EXPECTED_WEIGHTS_SHA256

    qtl_hits = pl.read_parquet(cross_reference_root / "qtl_hits.parquet")
    feature_ids = (
        qtl_hits.filter(pl.col("orientation") == "forward")
        .sort("feature_id")["feature_id"]
        .to_list()
    )
    assert len(feature_ids) == 45
    response_pairs = pl.read_parquet(
        cross_reference_root / "forward_pair_correlations.parquet"
    )
    components = pl.read_parquet(
        cross_reference_root / "forward_response_components.parquet"
    )

    decoder_full = safetensor_array(weights_path, "W_dec")
    encoder_full = safetensor_array(weights_path, "W_enc")
    assert decoder_full.shape == (15_360, 1_920)
    assert encoder_full.shape == (1_920, 15_360)
    decoder_vectors = np.asarray(decoder_full[feature_ids], dtype=np.float64)
    encoder_vectors = np.asarray(encoder_full[:, feature_ids].T, dtype=np.float64)
    decoder_cosine, decoder_norms = cosine_matrix(decoder_vectors)
    encoder_cosine, encoder_norms = cosine_matrix(encoder_vectors)

    pairs = build_pair_table(
        feature_ids=feature_ids,
        decoder_cosine=decoder_cosine,
        encoder_cosine=encoder_cosine,
        response_pairs=response_pairs,
        components=components,
    )
    component = components.filter(
        pl.col("absolute_pearson_threshold") == COMPONENT_THRESHOLD
    ).select("feature_id", "component_index", "component_size")
    features = (
        pl.DataFrame(
            {
                "feature_id": feature_ids,
                "decoder_norm": decoder_norms,
                "encoder_norm": encoder_norms,
            }
        )
        .join(component, on="feature_id")
        .join(
            qtl_hits.filter(pl.col("orientation") == "forward").select(
                "feature_id", "pearson", "spearman", "pearson_q", "spearman_q"
            ),
            on="feature_id",
        )
        .sort("feature_id")
    )
    giant_indices = np.array(
        [index for index, size in enumerate(features["component_size"]) if size == 36]
    )
    assert giant_indices.size == 36

    category_summary = (
        pairs.group_by("pair_category")
        .agg(
            pl.len().alias("pairs"),
            pl.col("absolute_response_pearson")
            .median()
            .alias("median_absolute_response_pearson"),
            pl.col("absolute_decoder_cosine")
            .median()
            .alias("median_absolute_decoder_cosine"),
            pl.col("absolute_encoder_cosine")
            .median()
            .alias("median_absolute_encoder_cosine"),
            pl.col("absolute_decoder_cosine")
            .max()
            .alias("maximum_absolute_decoder_cosine"),
            pl.col("absolute_encoder_cosine")
            .max()
            .alias("maximum_absolute_encoder_cosine"),
        )
        .sort("pair_category")
    )
    summary = {
        "features": len(feature_ids),
        "response_component": {
            "threshold": COMPONENT_THRESHOLD,
            "giant_component_features": int(giant_indices.size),
            "other_components": 9,
        },
        "decoder": {
            "all_hits": geometry_summary(decoder_cosine),
            "giant_component": geometry_summary(
                decoder_cosine[np.ix_(giant_indices, giant_indices)]
            ),
        },
        "encoder": {
            "all_hits": geometry_summary(encoder_cosine),
            "giant_component": geometry_summary(
                encoder_cosine[np.ix_(giant_indices, giant_indices)]
            ),
        },
        "absolute_response_vs_geometry": {
            "decoder": response_geometry_association(pairs, "absolute_decoder_cosine"),
            "encoder": response_geometry_association(pairs, "absolute_encoder_cosine"),
        },
        "pair_categories": category_summary.to_dicts(),
    }

    output_dir.mkdir(parents=True)
    artifacts = {
        "feature_geometry.parquet": features,
        "pair_geometry.parquet": pairs,
        "pair_category_summary.parquet": category_summary,
    }
    for name, frame in artifacts.items():
        frame.write_parquet(output_dir / name)
    write_json(output_dir / "summary.json", summary)

    artifact_names = (*artifacts.keys(), "summary.json")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": run_id,
        "experiment_commit": experiment_commit,
        "analysis": "post-hoc QTL response-family SAE geometry audit",
        "inputs": {
            "cross_reference_manifest_sha256": sha256_file(manifest_path),
            "weights": {
                "arm": ARM,
                "bytes": EXPECTED_WEIGHTS_BYTES,
                "sha256": EXPECTED_WEIGHTS_SHA256,
            },
        },
        "summary": summary,
        "artifacts": {
            name: {
                "bytes": (output_dir / name).stat().st_size,
                "sha256": sha256_file(output_dir / name),
            }
            for name in artifact_names
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-reference-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = run(
        cross_reference_root=args.cross_reference_root,
        weights_path=args.weights,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
