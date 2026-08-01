"""Persist all focal m5.1 SAE ref/alt activations for issue 420.

Dataset positions are VCF-style 1-based. They are converted exactly once to
0-based coordinates at the FASTA boundary; every interval thereafter is
0-based, half-open.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from marin_dna.data.dna import reverse_complement
from marin_dna.data.genome import Genome
from marin_dna.model.sae import (
    M51_HIDDEN_SIZE,
    M51GenomicWindow,
    load_frozen_m51,
    run_m51_with_activations,
)
from sae_lens.saes.sae import SAE
from torch.utils.data import DataLoader, Dataset

from analysis import (
    BLOCK_INDEX,
    CONTEXT_BP,
    D_SAE,
    FOCAL_INDEX,
    ISSUE,
    MODEL_ID,
    MODEL_REVISION,
    WINDOW_BP,
    _sha256,
    _validate_panel,
    variant_sequences,
)

ORIENTATIONS = ("forward", "reverse_complement")

SPARSE_SCHEMA = pa.schema(
    [
        pa.field("row_index", pa.uint32(), nullable=False),
        pa.field("feature_id", pa.uint32(), nullable=False),
        pa.field("ref_activation", pa.float32(), nullable=False),
        pa.field("alt_activation", pa.float32(), nullable=False),
        pa.field("delta", pa.float32(), nullable=False),
    ]
)
CONTEXT_SCHEMA = pa.schema(
    [
        pa.field("row_index", pa.uint32(), nullable=False),
        pa.field("ref_context", pa.string(), nullable=False),
        pa.field("alt_context", pa.string(), nullable=False),
        pa.field("flank_gc_count", pa.uint8(), nullable=False),
        pa.field("flank_gc_bin", pa.uint8(), nullable=False),
    ]
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_sae_provenance(sae_path: Path) -> dict[str, Any]:
    """Read and validate the model/SAE version contract."""

    required = (
        sae_path / "cfg.json",
        sae_path / "runner_cfg.json",
        sae_path / "sae_weights.safetensors",
        sae_path / "sparsity.safetensors",
    )
    for path in required:
        assert path.is_file(), path
    cfg = json.loads(required[0].read_text())
    runner = json.loads(required[1].read_text())
    metadata = cfg["metadata"]
    assert metadata["model_name"] == MODEL_ID
    assert metadata["model_revision"] == MODEL_REVISION
    assert metadata["block_index"] == BLOCK_INDEX
    assert metadata["report_block"] == BLOCK_INDEX + 1
    assert cfg["architecture"] == "jumprelu"
    assert cfg["d_in"] == M51_HIDDEN_SIZE
    assert cfg["d_sae"] == D_SAE
    assert runner["model_name"] == MODEL_ID
    assert runner["model_from_pretrained_kwargs"]["revision"] == MODEL_REVISION
    assert runner["sae"]["d_in"] == cfg["d_in"]
    assert runner["sae"]["d_sae"] == cfg["d_sae"]
    assert runner["seed"] == metadata["seed"]
    assert runner["training_tokens"] > 0
    return {
        "architecture": cfg["architecture"],
        "d_in": cfg["d_in"],
        "d_sae": cfg["d_sae"],
        "training_architecture": metadata["training_architecture"],
        "training_tokens": runner["training_tokens"],
        "seed": runner["seed"],
        "metadata": metadata,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in required
        },
    }


class VariantSequenceDataset(Dataset[dict[str, Any]]):
    """Load validated forward ref/alt windows with one FASTA handle per worker."""

    def __init__(self, frame: pl.DataFrame, fasta_path: Path) -> None:
        self.rows = frame.select("chrom", "pos", "ref", "alt").to_dicts()
        self.fasta_path = fasta_path
        self.chroms = {str(value) for value in frame["chrom"].unique()}
        self._genome: Genome | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._genome is None:
            self._genome = Genome(self.fasta_path, subset_chroms=self.chroms)
            assert set(self._genome.chroms) == self.chroms
        row = self.rows[index]
        pos0 = int(row["pos"]) - 1
        assert pos0 >= FOCAL_INDEX
        start = pos0 - FOCAL_INDEX
        end = pos0 + FOCAL_INDEX + 1
        assert start >= 0 and end - start == WINDOW_BP
        reference = self._genome(str(row["chrom"]), start, end, "+").upper()
        ref_sequence, alt_sequence = variant_sequences(
            reference, str(row["ref"]), str(row["alt"])
        )
        return {
            "row_index": index,
            "chrom": str(row["chrom"]),
            "start": start,
            "end": end,
            "ref_sequence": ref_sequence,
            "alt_sequence": alt_sequence,
        }


def collate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assert records
    return records


def sparse_union_table(
    ref_features: np.ndarray,
    alt_features: np.ndarray,
    row_indices: np.ndarray,
) -> pa.Table:
    """Encode the nonzero union of ref/alt activations as a long Arrow table."""

    assert ref_features.shape == alt_features.shape
    assert ref_features.ndim == 2
    assert row_indices.shape == (ref_features.shape[0],)
    assert np.isfinite(ref_features).all() and np.isfinite(alt_features).all()
    assert (ref_features >= 0).all() and (alt_features >= 0).all()
    nonzero = (ref_features != 0) | (alt_features != 0)
    local_row, feature_id = np.nonzero(nonzero)
    refs = ref_features[local_row, feature_id].astype(np.float32, copy=False)
    alts = alt_features[local_row, feature_id].astype(np.float32, copy=False)
    return pa.Table.from_arrays(
        [
            pa.array(row_indices[local_row], type=pa.uint32()),
            pa.array(feature_id, type=pa.uint32()),
            pa.array(refs, type=pa.float32()),
            pa.array(alts, type=pa.float32()),
            pa.array(alts - refs, type=pa.float32()),
        ],
        schema=SPARSE_SCHEMA,
    )


def context_table(records: list[dict[str, Any]]) -> pa.Table:
    """Return 41-bp contexts and a focal-base-excluded GC control."""

    radius = CONTEXT_BP // 2
    rows: list[int] = []
    refs: list[str] = []
    alts: list[str] = []
    gc_counts: list[int] = []
    gc_bins: list[int] = []
    for record in records:
        ref_sequence = record["ref_sequence"]
        alt_sequence = record["alt_sequence"]
        ref_context = ref_sequence[FOCAL_INDEX - radius : FOCAL_INDEX + radius + 1]
        alt_context = alt_sequence[FOCAL_INDEX - radius : FOCAL_INDEX + radius + 1]
        assert len(ref_context) == len(alt_context) == CONTEXT_BP
        assert sum(a != b for a, b in zip(ref_context, alt_context, strict=True)) == 1
        flank = ref_context[:radius] + ref_context[radius + 1 :]
        assert len(flank) == CONTEXT_BP - 1
        gc_count = sum(base in "GC" for base in flank)
        rows.append(int(record["row_index"]))
        refs.append(ref_context)
        alts.append(alt_context)
        gc_counts.append(gc_count)
        gc_bins.append(min(gc_count // 8, 4))
    return pa.Table.from_arrays(
        [
            pa.array(rows, type=pa.uint32()),
            pa.array(refs, type=pa.string()),
            pa.array(alts, type=pa.string()),
            pa.array(gc_counts, type=pa.uint8()),
            pa.array(gc_bins, type=pa.uint8()),
        ],
        schema=CONTEXT_SCHEMA,
    )


def _activation_tables(
    features: torch.Tensor,
    row_indices: np.ndarray,
) -> dict[str, pa.Table]:
    assert features.shape == (4 * len(row_indices), D_SAE)
    return {
        "forward": sparse_union_table(
            features[0::4].cpu().numpy(),
            features[1::4].cpu().numpy(),
            row_indices,
        ),
        "reverse_complement": sparse_union_table(
            features[2::4].cpu().numpy(),
            features[3::4].cpu().numpy(),
            row_indices,
        ),
    }


def extract_all_features(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    fasta_path: Path,
    sae_path: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    compile_model: bool,
) -> dict[str, Any]:
    """Run one versioned FWD+RC prediction loop and write sparse activations."""

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert panel_path.is_file() and panel_manifest_path.is_file()
    assert fasta_path.is_file() and Path(f"{fasta_path}.fai").is_file()
    assert Path(f"{fasta_path}.gzi").is_file() and sae_path.is_dir()
    assert batch_size > 0 and num_workers >= 0 and not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert len(experiment_commit) == 40
    assert all(character in "0123456789abcdef" for character in experiment_commit)
    started = time.monotonic()

    panel_manifest = json.loads(panel_manifest_path.read_text())
    assert panel_manifest["panel_sha256"] == _sha256(panel_path)
    frame = pl.read_parquet(panel_path)
    _validate_panel(frame)
    sae_provenance = read_sae_provenance(sae_path)
    output_dir.mkdir(parents=True, exist_ok=False)

    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    frozen = load_frozen_m51(checkpoint, device="cuda", dtype=torch.bfloat16)
    frozen.model.config.use_cache = False
    torch.set_float32_matmul_precision("high")
    if compile_model:
        frozen = replace(
            frozen,
            model=torch.compile(frozen.model, mode="reduce-overhead", fullgraph=False),
        )
    sae = SAE.load_from_disk(sae_path, device="cuda", dtype="float32")
    sae.requires_grad_(False)
    sae.eval()
    assert not sae.training
    assert all(not parameter.requires_grad for parameter in sae.parameters())
    assert sae.cfg.architecture() == sae_provenance["architecture"]
    assert sae.cfg.d_in == M51_HIDDEN_SIZE and sae.cfg.d_sae == D_SAE

    dataset = VariantSequenceDataset(frame, fasta_path)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_records,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)

    sparse_paths = {
        orientation: output_dir / f"sae_activations_{orientation}.parquet"
        for orientation in ORIENTATIONS
    }
    contexts_path = output_dir / "variant_contexts.parquet"
    sparse_rows = {orientation: 0 for orientation in ORIENTATIONS}
    processed = 0
    inference_seconds = 0.0
    writers = {
        orientation: pq.ParquetWriter(path, SPARSE_SCHEMA, compression="zstd")
        for orientation, path in sparse_paths.items()
    }
    try:
        with pq.ParquetWriter(
            contexts_path, CONTEXT_SCHEMA, compression="zstd"
        ) as context_writer:
            for batch_number, records in enumerate(loader, start=1):
                sequences: list[str] = []
                windows: list[M51GenomicWindow] = []
                for record in records:
                    ref_sequence = record["ref_sequence"]
                    alt_sequence = record["alt_sequence"]
                    rc_ref = reverse_complement(ref_sequence)
                    rc_alt = reverse_complement(alt_sequence)
                    assert rc_ref[FOCAL_INDEX] == reverse_complement(
                        ref_sequence[FOCAL_INDEX]
                    )
                    assert rc_alt[FOCAL_INDEX] == reverse_complement(
                        alt_sequence[FOCAL_INDEX]
                    )
                    sequences.extend((ref_sequence, alt_sequence, rc_ref, rc_alt))
                    forward_window = M51GenomicWindow(
                        chrom=record["chrom"],
                        start=int(record["start"]),
                        end=int(record["end"]),
                        strand="+",
                    )
                    reverse_window = M51GenomicWindow(
                        chrom=record["chrom"],
                        start=int(record["start"]),
                        end=int(record["end"]),
                        strand="-",
                    )
                    windows.extend(
                        (forward_window, forward_window, reverse_window, reverse_window)
                    )
                encoded = frozen.tokenizer(
                    sequences,
                    add_special_tokens=True,
                    padding=False,
                    truncation=False,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to("cuda", non_blocking=True)
                attention_mask = encoded["attention_mask"].to("cuda", non_blocking=True)
                torch.cuda.synchronize()
                inference_started = time.monotonic()
                _, activation_batch = run_m51_with_activations(
                    frozen,
                    input_ids,
                    attention_mask,
                    windows,
                    block_index=BLOCK_INDEX,
                )
                raw = activation_batch.activations[:, FOCAL_INDEX, :].float()
                with torch.inference_mode():
                    features = sae.encode(raw)
                torch.cuda.synchronize()
                inference_seconds += time.monotonic() - inference_started
                assert features.shape == (4 * len(records), D_SAE)
                assert torch.isfinite(features).all() and torch.all(features >= 0)
                row_indices = np.asarray(
                    [record["row_index"] for record in records], dtype=np.uint32
                )
                tables = _activation_tables(features, row_indices)
                for orientation, table in tables.items():
                    writers[orientation].write_table(table)
                    sparse_rows[orientation] += table.num_rows
                context_writer.write_table(context_table(records))
                processed += len(records)
                if (
                    batch_number == 1
                    or processed == frame.height
                    or batch_number % 25 == 0
                ):
                    print(
                        json.dumps(
                            {
                                "processed": processed,
                                "total": frame.height,
                                "inference_seconds": inference_seconds,
                                "sparse_rows": sparse_rows,
                            }
                        ),
                        flush=True,
                    )
                del input_ids, attention_mask, raw, features
    finally:
        for writer in writers.values():
            writer.close()
    assert processed == frame.height

    output_summaries: list[dict[str, Any]] = []
    for orientation, path in sparse_paths.items():
        summary = (
            pl.scan_parquet(path)
            .select(
                pl.len().alias("rows"),
                pl.col("row_index").n_unique().alias("variants_with_nonzero"),
                pl.col("feature_id").n_unique().alias("features"),
                pl.col("delta").is_nan().sum().alias("nan_deltas"),
            )
            .collect()
        )
        assert summary["rows"].item() == sparse_rows[orientation]
        assert summary["variants_with_nonzero"].item() == frame.height
        assert summary["nan_deltas"].item() == 0
        output_summaries.append(
            {
                "orientation": orientation,
                "path": path.name,
                "rows": sparse_rows[orientation],
                "variants_with_nonzero": summary["variants_with_nonzero"].item(),
                "features": summary["features"].item(),
                "schema": str(SPARSE_SCHEMA),
            }
        )
    contexts = pl.read_parquet(contexts_path)
    assert contexts.height == contexts["row_index"].n_unique() == frame.height
    assert contexts["row_index"].to_list() == list(range(frame.height))
    assert contexts.null_count().sum_horizontal().sum() == 0
    assert contexts["flank_gc_count"].min() >= 0
    assert contexts["flank_gc_count"].max() <= CONTEXT_BP - 1

    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": os.environ.get("RUN_ID", ""),
        "experiment_commit": experiment_commit,
        "elapsed_seconds": time.monotonic() - started,
        "inference_seconds": inference_seconds,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "performance": {
            "dtype": "bfloat16",
            "torch_compile": compile_model,
            "torch_compile_mode": "reduce-overhead" if compile_model else None,
            "use_cache": False,
            "batch_size_variants": batch_size,
            "sequences_per_full_batch": 4 * batch_size,
            "dataloader_workers": num_workers,
            "prefetch_factor": 2 if num_workers > 0 else None,
            "inference_mode": True,
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "reported_block": BLOCK_INDEX + 1,
            "implementation_block_index": BLOCK_INDEX,
        },
        "sae": {"path": str(sae_path), **sae_provenance},
        "panel": {
            "path": str(panel_path),
            "sha256": _sha256(panel_path),
            "manifest_sha256": _sha256(panel_manifest_path),
            "rows": frame.height,
            "match_groups": frame["match_group"].n_unique(),
        },
        "protocol": {
            "coordinate_system": "0-based half-open after pos0 = pos1 - 1",
            "window_bp": WINDOW_BP,
            "focal_index": FOCAL_INDEX,
            "context_bp": CONTEXT_BP,
            "orientations": list(ORIENTATIONS),
            "sparse_representation": "union of nonzero ref/alt activations",
        },
        "outputs": output_summaries,
    }
    _write_json(output_dir / "results.json", result)
    artifact_files = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        **result,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in artifact_files
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--torch-compile", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    manifest = extract_all_features(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        fasta_path=args.fasta,
        sae_path=args.sae,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        compile_model=args.torch_compile,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
