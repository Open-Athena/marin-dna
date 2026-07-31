"""Extract versioned m5.1 SAE variant effects for the issue 422 panel.

Input positions are 1-based. They are converted exactly once to 0-based,
half-open intervals at the FASTA boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

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

ISSUE = 422
WINDOW_BP = 255
FOCAL_INDEX = 127
MODEL_ID = "marin-dna/marin-dna-exp135-m5.1"
MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
DEFAULT_BLOCK_INDEX = 9
ORIENTATIONS = ("forward", "reverse_complement")
NUCLEOTIDES = frozenset("ACGT")
EXPECTED_ROWS = 17_920
EXPECTED_CLASSES = 35
EXPECTED_SPLIT_COUNTS = {
    "discovery": 35 * 256,
    "validation": 35 * 128,
    "test": 35 * 128,
}

assert WINDOW_BP == 2 * FOCAL_INDEX + 1
assert M51_HIDDEN_SIZE == 1_920

SPARSE_SCHEMA = pa.schema(
    [
        pa.field("panel_row", pa.uint32(), nullable=False),
        pa.field("feature_id", pa.uint32(), nullable=False),
        pa.field("ref_activation", pa.float32(), nullable=False),
        pa.field("alt_activation", pa.float32(), nullable=False),
        pa.field("delta", pa.float32(), nullable=False),
    ]
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_sae_provenance(sae_path: Path, *, block_index: int) -> dict[str, Any]:
    """Read and validate the version contract embedded in a saved SAE."""

    cfg_path = sae_path / "cfg.json"
    runner_path = sae_path / "runner_cfg.json"
    weights_path = sae_path / "sae_weights.safetensors"
    sparsity_path = sae_path / "sparsity.safetensors"
    for path in (cfg_path, runner_path, weights_path, sparsity_path):
        assert path.is_file(), path

    cfg = json.loads(cfg_path.read_text())
    runner = json.loads(runner_path.read_text())
    metadata = cfg["metadata"]
    assert metadata["model_name"] == MODEL_ID
    assert metadata["model_revision"] == MODEL_REVISION
    assert metadata["block_index"] == block_index
    assert metadata["report_block"] == block_index + 1
    assert cfg["d_in"] == M51_HIDDEN_SIZE
    assert cfg["d_sae"] > 0
    assert cfg["architecture"] == "jumprelu"
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
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (cfg_path, runner_path, weights_path, sparsity_path)
        },
    }


def variant_sequences(reference_sequence: str, ref: str, alt: str) -> tuple[str, str]:
    """Validate a centered reference window and return ref/alt sequences."""

    reference_sequence = reference_sequence.upper()
    ref = ref.upper()
    alt = alt.upper()
    assert len(reference_sequence) == WINDOW_BP
    assert set(reference_sequence) <= NUCLEOTIDES
    assert len(ref) == len(alt) == 1
    assert ref in NUCLEOTIDES and alt in NUCLEOTIDES and ref != alt
    assert reference_sequence[FOCAL_INDEX] == ref
    alternate = (
        reference_sequence[:FOCAL_INDEX] + alt + reference_sequence[FOCAL_INDEX + 1 :]
    )
    assert len(alternate) == WINDOW_BP
    assert alternate[FOCAL_INDEX] == alt
    assert sum(a != b for a, b in zip(reference_sequence, alternate, strict=True)) == 1
    return reference_sequence, alternate


def validate_panel(
    frame: pl.DataFrame, manifest: dict[str, Any], panel_path: Path
) -> None:
    """Fail loudly on a changed or malformed frozen panel."""

    required = {
        "panel_row",
        "chrom",
        "pos",
        "ref",
        "alt",
        "consequence",
        "consequence_cre",
        "block_id",
        "split",
        "sample_hash",
    }
    assert required <= set(frame.columns), required - set(frame.columns)
    assert manifest["output"]["sha256"] == sha256_file(panel_path)
    assert manifest["output"]["rows"] == frame.height == EXPECTED_ROWS
    assert manifest["output"]["classes"] == EXPECTED_CLASSES
    assert frame["panel_row"].to_list() == list(range(frame.height))
    assert (
        frame.select(pl.struct("chrom", "pos", "ref", "alt").n_unique()).item()
        == frame.height
    )
    assert frame["chrom"].unique().to_list() == ["21"]
    assert frame["consequence_cre"].n_unique() == EXPECTED_CLASSES
    assert frame.null_count().sum_horizontal().sum() == 0
    assert frame.filter(pl.col("pos") < 1).is_empty()
    assert frame.filter(pl.col("ref") == pl.col("alt")).is_empty()
    assert frame.filter(
        ~pl.col("ref").is_in(sorted(NUCLEOTIDES))
        | ~pl.col("alt").is_in(sorted(NUCLEOTIDES))
    ).is_empty()
    observed_split_counts = dict(frame.group_by("split").len().iter_rows())
    assert observed_split_counts == EXPECTED_SPLIT_COUNTS
    expected_split = (
        pl.when((pl.col("block_id") % 5) <= 2)
        .then(pl.lit("discovery"))
        .when((pl.col("block_id") % 5) == 3)
        .then(pl.lit("validation"))
        .otherwise(pl.lit("test"))
    )
    assert frame.filter(expected_split != pl.col("split")).is_empty()


def sparse_union_table(
    ref_features: np.ndarray,
    alt_features: np.ndarray,
    panel_rows: np.ndarray,
) -> pa.Table:
    """Encode the nonzero union of ref/alt SAE activations as a long Arrow table."""

    assert ref_features.shape == alt_features.shape
    assert ref_features.ndim == 2
    assert panel_rows.shape == (ref_features.shape[0],)
    assert np.isfinite(ref_features).all() and np.isfinite(alt_features).all()
    assert (ref_features >= 0).all() and (alt_features >= 0).all()
    nonzero = (ref_features != 0) | (alt_features != 0)
    row_index, feature_id = np.nonzero(nonzero)
    refs = ref_features[row_index, feature_id].astype(np.float32, copy=False)
    alts = alt_features[row_index, feature_id].astype(np.float32, copy=False)
    deltas = alts - refs
    return pa.Table.from_arrays(
        [
            pa.array(panel_rows[row_index], type=pa.uint32()),
            pa.array(feature_id, type=pa.uint32()),
            pa.array(refs, type=pa.float32()),
            pa.array(alts, type=pa.float32()),
            pa.array(deltas, type=pa.float32()),
        ],
        schema=SPARSE_SCHEMA,
    )


def encode_sae_features(sae: SAE, raw: torch.Tensor) -> torch.Tensor:
    """Encode with a frozen SAE without constructing an autograd graph."""

    assert not sae.training
    assert all(not parameter.requires_grad for parameter in sae.parameters())
    with torch.inference_mode():
        features = sae.encode(raw)
    assert not features.requires_grad
    return features


def extract_variant_batch(
    frame: pl.DataFrame,
    indices: Sequence[int],
    *,
    genome: Genome,
    frozen: Any,
    sae: SAE,
    orientation: Literal["forward", "reverse_complement"],
    block_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw deltas plus ref and alt sparse-feature activations."""

    sequences: list[str] = []
    windows: list[M51GenomicWindow] = []
    for index in indices:
        row = frame.row(index, named=True)
        pos0 = int(row["pos"]) - 1
        assert pos0 >= FOCAL_INDEX
        start = pos0 - FOCAL_INDEX
        end = pos0 + FOCAL_INDEX + 1
        assert start >= 0 and end - start == WINDOW_BP
        reference = genome(row["chrom"], start, end, "+").upper()
        ref_sequence, alt_sequence = variant_sequences(
            reference, row["ref"], row["alt"]
        )
        strand = "+"
        if orientation == "reverse_complement":
            ref_sequence = reverse_complement(ref_sequence)
            alt_sequence = reverse_complement(alt_sequence)
            strand = "-"
            assert ref_sequence[FOCAL_INDEX] == reverse_complement(row["ref"])
            assert alt_sequence[FOCAL_INDEX] == reverse_complement(row["alt"])
        sequences.extend((ref_sequence, alt_sequence))
        window = M51GenomicWindow(
            chrom=row["chrom"], start=start, end=end, strand=strand
        )
        windows.extend((window, window))

    encoded = frozen.tokenizer(
        sequences,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")
    _, activation_batch = run_m51_with_activations(
        frozen,
        input_ids,
        attention_mask,
        windows,
        block_index=block_index,
    )
    raw = activation_batch.activations[:, FOCAL_INDEX, :].float()
    features = encode_sae_features(sae, raw)
    assert raw.shape == (2 * len(indices), M51_HIDDEN_SIZE)
    assert features.shape == (2 * len(indices), sae.cfg.d_sae)
    assert torch.isfinite(raw).all() and torch.isfinite(features).all()
    assert torch.all(features >= 0)
    return (
        (raw[1::2] - raw[0::2]).cpu().numpy(),
        features[0::2].cpu().numpy(),
        features[1::2].cpu().numpy(),
    )


def extract_orientation(
    frame: pl.DataFrame,
    *,
    genome: Genome,
    frozen: Any,
    sae: SAE,
    orientation: Literal["forward", "reverse_complement"],
    block_index: int,
    batch_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Write one orientation's sparse SAE activations and dense raw deltas."""

    raw_path = output_dir / f"raw_delta_{orientation}.npy"
    sparse_path = output_dir / f"sae_activations_{orientation}.parquet"
    raw = np.lib.format.open_memmap(
        raw_path,
        mode="w+",
        dtype=np.float32,
        shape=(frame.height, M51_HIDDEN_SIZE),
    )
    sparse_rows = 0
    with pq.ParquetWriter(sparse_path, SPARSE_SCHEMA, compression="zstd") as writer:
        for offset in range(0, frame.height, batch_size):
            stop = min(offset + batch_size, frame.height)
            indices = list(range(offset, stop))
            raw_batch, ref_features, alt_features = extract_variant_batch(
                frame,
                indices,
                genome=genome,
                frozen=frozen,
                sae=sae,
                orientation=orientation,
                block_index=block_index,
            )
            raw[offset:stop] = raw_batch
            panel_rows = frame["panel_row"].slice(offset, stop - offset).to_numpy()
            sparse = sparse_union_table(ref_features, alt_features, panel_rows)
            writer.write_table(sparse)
            sparse_rows += sparse.num_rows
            if offset == 0 or stop == frame.height or stop % (batch_size * 25) == 0:
                print(
                    json.dumps(
                        {
                            "orientation": orientation,
                            "processed": stop,
                            "total": frame.height,
                            "sparse_rows": sparse_rows,
                        }
                    ),
                    flush=True,
                )
    raw.flush()
    assert np.isfinite(raw).all()
    sparse_summary = (
        pl.scan_parquet(sparse_path)
        .select(
            pl.len().alias("rows"),
            pl.col("panel_row").n_unique().alias("variants_with_nonzero"),
            pl.col("feature_id").n_unique().alias("features"),
            pl.col("delta").is_nan().sum().alias("nan_deltas"),
        )
        .collect()
    )
    assert sparse_summary["rows"].item() == sparse_rows
    assert sparse_summary["nan_deltas"].item() == 0
    return {
        "orientation": orientation,
        "raw": {
            "path": raw_path.name,
            "shape": [frame.height, M51_HIDDEN_SIZE],
            "dtype": "float32",
        },
        "sae": {
            "path": sparse_path.name,
            "rows": sparse_rows,
            "variants_with_nonzero": sparse_summary["variants_with_nonzero"].item(),
            "features": sparse_summary["features"].item(),
            "schema": str(SPARSE_SCHEMA),
        },
    }


def evaluate(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    fasta_path: Path,
    sae_path: Path,
    output_dir: Path,
    batch_size: int,
    block_index: int,
) -> dict[str, Any]:
    """Run the versioned GPU extraction and write a hash-complete manifest."""

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert batch_size > 0 and block_index >= 0
    assert panel_path.is_file() and panel_manifest_path.is_file()
    assert fasta_path.is_file() and Path(f"{fasta_path}.fai").is_file()
    assert Path(f"{fasta_path}.gzi").is_file() and sae_path.is_dir()
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert len(experiment_commit) == 40
    assert all(character in "0123456789abcdef" for character in experiment_commit)

    started = time.monotonic()
    panel_manifest = json.loads(panel_manifest_path.read_text())
    frame = pl.read_parquet(panel_path)
    validate_panel(frame, panel_manifest, panel_path)
    sae_provenance = read_sae_provenance(sae_path, block_index=block_index)
    output_dir.mkdir(parents=True, exist_ok=False)

    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    frozen = load_frozen_m51(checkpoint, device="cuda", dtype=torch.bfloat16)
    sae = SAE.load_from_disk(sae_path, device="cuda", dtype="float32")
    sae.requires_grad_(False)
    sae.eval()
    assert not sae.training
    assert all(not parameter.requires_grad for parameter in sae.parameters())
    assert sae.cfg.architecture() == sae_provenance["architecture"]
    assert sae.cfg.d_in == sae_provenance["d_in"] == M51_HIDDEN_SIZE
    assert sae.cfg.d_sae == sae_provenance["d_sae"]
    genome = Genome(fasta_path, subset_chroms={"21"})
    assert set(genome.chroms) == {"21"}

    orientation_results = [
        extract_orientation(
            frame,
            genome=genome,
            frozen=frozen,
            sae=sae,
            orientation=orientation,
            block_index=block_index,
            batch_size=batch_size,
            output_dir=output_dir,
        )
        for orientation in ORIENTATIONS
    ]
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": os.environ.get("RUN_ID", ""),
        "experiment_commit": experiment_commit,
        "elapsed_seconds": time.monotonic() - started,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "reported_block": block_index + 1,
            "implementation_block_index": block_index,
            "hidden_size": M51_HIDDEN_SIZE,
        },
        "sae": {
            "path": str(sae_path),
            **sae_provenance,
        },
        "panel": {
            "path": str(panel_path),
            "sha256": sha256_file(panel_path),
            "rows": frame.height,
            "classes": frame["consequence_cre"].n_unique(),
            "source": panel_manifest["source"],
            "sampling": panel_manifest["sampling"],
        },
        "protocol": {
            "coordinate_system": "0-based half-open after pos0 = pos1 - 1",
            "window_bp": WINDOW_BP,
            "focal_index": FOCAL_INDEX,
            "orientations": list(ORIENTATIONS),
            "batch_size": batch_size,
        },
        "outputs": orientation_results,
    }
    write_json(output_dir / "results.json", result)
    artifact_files = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        **result,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artifact_files
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-index", type=int, default=DEFAULT_BLOCK_INDEX)
    args = parser.parse_args()
    manifest = evaluate(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        fasta_path=args.fasta,
        sae_path=args.sae,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        block_index=args.block_index,
    )
    print(json.dumps(manifest["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
