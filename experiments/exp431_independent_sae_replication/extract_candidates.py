"""Extract decoder-neighborhood SAE profiles for one #431 panel split."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from huggingface_hub import snapshot_download
from marin_dna.model.sae import load_frozen_m51

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP429_DIR = REPO_ROOT / "experiments" / "exp429_variant_feature_map"
if str(EXP429_DIR) not in sys.path:
    sys.path.insert(0, str(EXP429_DIR))

from extract_perturbations import (
    MODEL_ID,
    MODEL_REVISION,
    ORIENTATIONS,
    build_state_table,
    extract_orientation,
    sha256_file,
    validate_design,
)
from sample_panel import assert_current_commit
from spatial import read_sae_provenance

ISSUE = 431
BLOCK_INDEX = 9
SPATIAL_RADIUS = 15


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_candidates(
    candidate_dir: Path, sae_path: Path
) -> tuple[pl.DataFrame, list[int], dict[str, Any]]:
    manifest_path = candidate_dir / "manifest.json"
    table_path = candidate_dir / "decoder_candidates.parquet"
    assert manifest_path.is_file() and table_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["issue"] == ISSUE
    assert manifest["artifacts"][table_path.name]["sha256"] == sha256_file(table_path)
    assert manifest["candidate_sae"]["weights_sha256"] == sha256_file(
        sae_path / "sae_weights.safetensors"
    )
    table = pl.read_parquet(table_path)
    required = {
        "dictionary",
        "concept",
        "reference_feature_id",
        "candidate_feature_id",
        "candidate_rank",
        "decoder_cosine",
    }
    assert required <= set(table.columns)
    feature_ids = sorted(set(table["candidate_feature_id"].to_list()))
    assert feature_ids and min(feature_ids) >= 0 and max(feature_ids) < 15_360
    assert len(feature_ids) == manifest["unique_candidate_features"]
    return table, feature_ids, manifest


def extract_candidates(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    candidate_dir: Path,
    sae_path: Path,
    output_dir: Path,
    dictionary_name: str,
    batch_size: int,
) -> dict[str, Any]:
    """Extract FWD/RC 31-position profiles for the frozen candidate union."""

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert dictionary_name and batch_size > 0 and not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_current_commit(experiment_commit)
    started = time.monotonic()
    panel_manifest = json.loads(panel_manifest_path.read_text())
    panel = pl.read_parquet(panel_path)
    validate_design(panel, panel_manifest, panel_path=panel_path)
    split_values = set(panel["source_split"].unique())
    assert len(split_values) == 1
    source_split = next(iter(split_values))
    candidate_table, feature_ids, candidate_manifest = load_candidates(
        candidate_dir, sae_path
    )
    assert set(candidate_table["dictionary"].unique()) == {dictionary_name}
    states, reference_indices, alternate_indices = build_state_table(panel)
    assert panel.height < states.height <= 2 * panel.height
    sae_provenance = read_sae_provenance(sae_path, block_index=BLOCK_INDEX)

    output_dir.mkdir(parents=True)
    states.write_parquet(output_dir / "perturbation_states.parquet")
    candidate_table.write_parquet(output_dir / "decoder_candidates.parquet")
    np.save(output_dir / "feature_ids.npy", np.asarray(feature_ids, dtype=np.int64))
    np.save(output_dir / "reference_state_indices.npy", reference_indices)
    np.save(output_dir / "alternate_state_indices.npy", alternate_indices)

    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    frozen = load_frozen_m51(checkpoint, device="cuda", dtype=torch.bfloat16)
    from sae_lens.saes.sae import SAE

    sae = SAE.load_from_disk(sae_path, device="cuda", dtype="float32").eval()
    sae.requires_grad_(False)
    feature_tensor = torch.tensor(feature_ids, dtype=torch.long, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    orientation_results = {
        orientation: extract_orientation(
            states,
            frozen=frozen,
            sae=sae,
            feature_ids=feature_tensor,
            orientation=orientation,
            block_index=BLOCK_INDEX,
            batch_size=batch_size,
            radius=SPATIAL_RADIUS,
            output_dir=output_dir,
        )
        for orientation in ORIENTATIONS
    }
    result = {
        "issue": ISSUE,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_commit": experiment_commit,
        "dictionary": dictionary_name,
        "source_split": source_split,
        "elapsed_seconds": time.monotonic() - started,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "reported_block": BLOCK_INDEX + 1,
            "implementation_block_index": BLOCK_INDEX,
            "dtype": "bfloat16",
        },
        "sae": sae_provenance,
        "candidates": {
            "manifest_sha256": sha256_file(candidate_dir / "manifest.json"),
            "table_sha256": sha256_file(candidate_dir / "decoder_candidates.parquet"),
            "feature_count": len(feature_ids),
            "query_count": candidate_table.select("reference_feature_id").n_unique(),
            "top_k": candidate_manifest["top_k"],
        },
        "design": {
            "manifest_sha256": sha256_file(panel_manifest_path),
            "panel_sha256": sha256_file(panel_path),
            "paired_rows": panel.height,
            "unique_states": states.height,
        },
        "orientation_outputs": orientation_results,
    }
    write_json(output_dir / "results.json", result)
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {**result, "artifacts": artifacts}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dictionary-name", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = extract_candidates(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        candidate_dir=args.candidate_dir,
        sae_path=args.sae,
        output_dir=args.output_dir,
        dictionary_name=args.dictionary_name,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
