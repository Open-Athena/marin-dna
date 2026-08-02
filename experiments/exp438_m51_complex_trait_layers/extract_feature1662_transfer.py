"""Extract fixed feature 1662 from the untouched complex-traits test split."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from marin_dna.data.genome import Genome
from marin_dna.model.sae import load_frozen_m51
from sae_lens.load_model import HookedProxyLM
from sae_lens.saes.sae import SAE

from common import (
    MODEL_ID,
    MODEL_REVISION,
    assert_commit,
    sha256_file,
    write_json,
)
from extract_focal import (
    HOOK_NAMES,
    ORIENTATIONS,
    arm_label,
    batch_sequences,
    extract_raw_focal,
    model_path,
    read_model_provenance,
)
from transfer_common import (
    BLOCK_INDEX,
    EXPECTED_ROWS,
    FEATURE_ID,
    validate_test_panel,
)

SCHEMA = pa.schema(
    [
        pa.field("panel_row", pa.uint32(), nullable=False),
        pa.field("ref_activation", pa.float32(), nullable=False),
        pa.field("alt_activation", pa.float32(), nullable=False),
        pa.field("delta", pa.float32(), nullable=False),
    ]
)


def dense_feature_table(
    panel_rows: np.ndarray, ref: np.ndarray, alt: np.ndarray
) -> pa.Table:
    assert panel_rows.dtype == np.uint32
    assert ref.shape == alt.shape == panel_rows.shape
    assert ref.dtype == alt.dtype == np.float32
    assert np.isfinite(ref).all() and np.isfinite(alt).all()
    assert np.all(ref >= 0) and np.all(alt >= 0)
    return pa.Table.from_arrays(
        [
            pa.array(panel_rows, type=pa.uint32()),
            pa.array(ref, type=pa.float32()),
            pa.array(alt, type=pa.float32()),
            pa.array(alt - ref, type=pa.float32()),
        ],
        schema=SCHEMA,
    )


def extract(
    panel_path: Path,
    fasta_path: Path,
    models_root: Path,
    output_dir: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert batch_size > 0 and not output_dir.exists()
    assert fasta_path.is_file() and Path(f"{fasta_path}.fai").is_file()
    assert Path(f"{fasta_path}.gzi").is_file()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    panel_manifest = validate_test_panel(panel_path)
    frame = pl.read_parquet(panel_path).with_row_index("panel_row")
    assert frame.height == EXPECTED_ROWS
    output_dir.mkdir(parents=True)
    started = time.monotonic()

    label = arm_label(BLOCK_INDEX)
    path = model_path(block_index=BLOCK_INDEX, models_root=models_root)
    provenance = read_model_provenance(path, block_index=BLOCK_INDEX)
    sae = SAE.load_from_disk(path, device="cuda", dtype="float32")
    sae.requires_grad_(False)
    sae.eval()

    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    frozen = load_frozen_m51(checkpoint, device="cuda", dtype=torch.bfloat16)
    frozen.model.config.use_cache = False
    model = HookedProxyLM(frozen.model, frozen.tokenizer, hook_names=list(HOOK_NAMES))
    chroms = set(frame["chrom"].cast(pl.String).unique().to_list())
    genome = Genome(fasta_path, subset_chroms=chroms)
    assert set(genome.chroms) == chroms
    torch.cuda.reset_peak_memory_stats()

    artifacts: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    for orientation in ORIENTATIONS:
        path = output_dir / f"feature1662_{orientation}.parquet"
        writer = pq.ParquetWriter(path, SCHEMA, compression="zstd")
        try:
            for offset in range(0, frame.height, batch_size):
                stop = min(offset + batch_size, frame.height)
                indices = list(range(offset, stop))
                sequences = batch_sequences(
                    frame, indices, genome=genome, orientation=orientation
                )
                raw = extract_raw_focal(
                    sequences, tokenizer=frozen.tokenizer, model=model
                )[BLOCK_INDEX]
                features = sae.encode(raw)
                assert features.shape[0] == 2 * len(indices)
                selected = features[:, FEATURE_ID]
                assert torch.isfinite(selected).all() and torch.all(selected >= 0)
                panel_rows = frame["panel_row"].slice(offset, stop - offset).to_numpy()
                writer.write_table(
                    dense_feature_table(
                        panel_rows,
                        selected[0::2].cpu().numpy().astype(np.float32, copy=False),
                        selected[1::2].cpu().numpy().astype(np.float32, copy=False),
                    )
                )
                if offset == 0 or stop == frame.height or stop % (batch_size * 25) == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "test_transfer",
                                "orientation": orientation,
                                "processed": stop,
                                "total": frame.height,
                            }
                        ),
                        flush=True,
                    )
        finally:
            writer.close()
        observed = pl.read_parquet(path)
        assert observed.height == EXPECTED_ROWS
        assert observed["panel_row"].to_list() == list(range(EXPECTED_ROWS))
        assert observed.select(pl.all().exclude("panel_row").is_finite().all()).row(
            0
        ) == (
            True,
            True,
            True,
        )
        summaries[orientation] = {
            "rows": observed.height,
            "nonzero_response": int((observed["delta"] != 0).sum()),
        }
        artifacts[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": 438,
        "run_id": os.environ.get("RUN_ID", ""),
        "experiment_commit": experiment_commit,
        "analysis_status": "preregistered_untouched_test_transfer",
        "elapsed_seconds": time.monotonic() - started,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16"},
        "sae": {"arm": label, "feature_id": FEATURE_ID, "provenance": provenance},
        "panel": panel_manifest,
        "protocol": {
            "responses": ["forward_abs_delta", "reverse_complement_abs_delta"],
            "compile_llm": False,
            "batch_size_variants": batch_size,
        },
        "outputs": summaries,
        "artifacts": artifacts,
    }
    write_json(output_dir / "results.json", result)
    result["artifacts"]["results.json"] = {
        "bytes": (output_dir / "results.json").stat().st_size,
        "sha256": sha256_file(output_dir / "results.json"),
    }
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    extract(
        args.panel,
        args.fasta,
        args.models_root,
        args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
