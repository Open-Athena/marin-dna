"""Extract selected-feature position profiles around issue 429 variants.

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

from sample_panel import assert_current_commit

ISSUE = 429
WINDOW_BP = 255
FOCAL_INDEX = 127
SPATIAL_RADIUS = 15
MODEL_ID = "marin-dna/marin-dna-exp135-m5.1"
MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
DEFAULT_BLOCK_INDEX = 9
ORIENTATIONS = ("forward", "reverse_complement")
NUCLEOTIDES = frozenset("ACGT")
EXPECTED_CLASSES = 11
EXPECTED_ROWS = 22_528
EXPECTED_SELECTED_FEATURES = 18
EXPECTED_SPLIT_COUNTS = {
    "discovery": 11 * 1_024,
    "validation": 11 * 512,
    "test": 11 * 512,
}
EXPECTED_CONSEQUENCES = frozenset(
    {
        "missense_variant",
        "synonymous_variant",
        "stop_gained",
        "stop_lost",
        "start_lost",
        "splice_region_variant",
        "splice_polypyrimidine_tract_variant",
        "splice_donor_region_variant",
        "splice_donor_variant",
        "splice_acceptor_variant",
        "splice_donor_5th_base_variant",
    }
)

assert WINDOW_BP == 2 * FOCAL_INDEX + 1
assert M51_HIDDEN_SIZE == 1_920


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
    assert set(frame["consequence_cre"].unique()) == EXPECTED_CONSEQUENCES
    assert set(manifest["sampling"]["target_classes"]) == EXPECTED_CONSEQUENCES
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


def encode_selected_features(
    sae: SAE, raw: torch.Tensor, feature_ids: torch.Tensor
) -> torch.Tensor:
    """Encode only frozen feature IDs with the exact JumpReLU formula."""

    assert not sae.training and raw.shape[-1] == sae.cfg.d_in
    assert feature_ids.ndim == 1 and feature_ids.dtype == torch.long
    assert sae.cfg.architecture() == "jumprelu"
    assert sae.cfg.normalize_activations == "none"
    assert not sae.hook_z_reshaping_mode
    with torch.inference_mode():
        sae_in = sae.process_sae_in(raw)
        weights = sae.W_enc.index_select(1, feature_ids)
        bias = sae.b_enc.index_select(0, feature_ids)
        threshold = sae.threshold.index_select(0, feature_ids)
        hidden_pre = sae_in @ weights + bias
        base_acts = sae.activation_fn(hidden_pre)
        selected = base_acts * (hidden_pre > threshold).to(base_acts.dtype)
    assert selected.shape == (*raw.shape[:-1], len(feature_ids))
    assert torch.isfinite(selected).all() and torch.all(selected >= 0)
    return selected


def extract_spatial_batch(
    frame: pl.DataFrame,
    indices: Sequence[int],
    *,
    genome: Genome,
    frozen: Any,
    sae: SAE,
    feature_ids: torch.Tensor,
    orientation: Literal["forward", "reverse_complement"],
    block_index: int,
    radius: int,
    validate_subset: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ref and alt selected-feature activations around each edit."""

    sequences: list[str] = []
    windows: list[M51GenomicWindow] = []
    for index in indices:
        row = frame.row(index, named=True)
        pos0 = int(row["pos"]) - 1
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
        frozen, input_ids, attention_mask, windows, block_index=block_index
    )
    position_slice = slice(FOCAL_INDEX - radius, FOCAL_INDEX + radius + 1)
    raw = activation_batch.activations[:, position_slice, :].float()
    selected = encode_selected_features(sae, raw, feature_ids)
    if validate_subset:
        with torch.inference_mode():
            expected = sae.encode(raw[:1, :, :]).index_select(-1, feature_ids)
        torch.testing.assert_close(selected[:1, :, :], expected, rtol=1e-6, atol=1e-5)
    assert selected.shape == (2 * len(indices), 2 * radius + 1, len(feature_ids))
    return selected[0::2].cpu().numpy(), selected[1::2].cpu().numpy()


def extract_orientation(
    frame: pl.DataFrame,
    *,
    genome: Genome,
    frozen: Any,
    sae: SAE,
    feature_ids: torch.Tensor,
    orientation: Literal["forward", "reverse_complement"],
    block_index: int,
    batch_size: int,
    radius: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Write dense ref and alt position profiles for one orientation."""

    shape = (frame.height, 2 * radius + 1, len(feature_ids))
    ref_path = output_dir / f"spatial_ref_{orientation}.npy"
    alt_path = output_dir / f"spatial_alt_{orientation}.npy"
    ref = np.lib.format.open_memmap(ref_path, mode="w+", dtype=np.float32, shape=shape)
    alt = np.lib.format.open_memmap(alt_path, mode="w+", dtype=np.float32, shape=shape)
    for offset in range(0, frame.height, batch_size):
        stop = min(offset + batch_size, frame.height)
        ref_batch, alt_batch = extract_spatial_batch(
            frame,
            list(range(offset, stop)),
            genome=genome,
            frozen=frozen,
            sae=sae,
            feature_ids=feature_ids,
            orientation=orientation,
            block_index=block_index,
            radius=radius,
            validate_subset=offset == 0,
        )
        ref[offset:stop] = ref_batch
        alt[offset:stop] = alt_batch
        if offset == 0 or stop == frame.height or stop % (batch_size * 25) == 0:
            print(
                json.dumps(
                    {
                        "orientation": orientation,
                        "processed": stop,
                        "total": frame.height,
                        "features": len(feature_ids),
                        "positions": 2 * radius + 1,
                    }
                ),
                flush=True,
            )
    ref.flush()
    alt.flush()
    assert np.isfinite(ref).all() and np.isfinite(alt).all()
    return {
        "orientation": orientation,
        "ref": {"path": ref_path.name, "shape": list(shape), "dtype": "float32"},
        "alt": {"path": alt_path.name, "shape": list(shape), "dtype": "float32"},
    }


def evaluate(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    fasta_path: Path,
    sae_path: Path,
    analysis_dir: Path,
    output_dir: Path,
    batch_size: int,
    block_index: int,
    radius: int,
) -> dict[str, Any]:
    """Extract commit-pinned position profiles for validation-selected features."""

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert batch_size > 0 and block_index >= 0 and 0 < radius < FOCAL_INDEX
    assert panel_path.is_file() and panel_manifest_path.is_file()
    assert fasta_path.is_file() and Path(f"{fasta_path}.fai").is_file()
    assert Path(f"{fasta_path}.gzi").is_file() and sae_path.is_dir()
    analysis_manifest_path = analysis_dir / "manifest.json"
    selected_path = analysis_dir / "selected_individual_features.parquet"
    assert analysis_manifest_path.is_file() and selected_path.is_file()
    assert not output_dir.exists()
    spatial_commit = os.environ.get("SPATIAL_COMMIT", "")
    assert_current_commit(spatial_commit)

    started = time.monotonic()
    panel_manifest = json.loads(panel_manifest_path.read_text())
    frame = pl.read_parquet(panel_path)
    validate_panel(frame, panel_manifest, panel_path)
    analysis_manifest = json.loads(analysis_manifest_path.read_text())
    selected_metadata = analysis_manifest["artifacts"][selected_path.name]
    assert sha256_file(selected_path) == selected_metadata["sha256"]
    assert analysis_manifest["panel_sha256"] == sha256_file(panel_path)
    selected_frame = pl.read_parquet(selected_path)
    required = {"class", "orientation", "transform", "dimension"}
    assert required <= set(selected_frame.columns)
    feature_id_values = sorted(selected_frame["dimension"].unique().to_list())
    assert len(feature_id_values) == EXPECTED_SELECTED_FEATURES
    assert feature_id_values[0] >= 0

    sae_provenance = read_sae_provenance(sae_path, block_index=block_index)
    assert feature_id_values[-1] < sae_provenance["d_sae"]
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    frozen = load_frozen_m51(checkpoint, device="cuda", dtype=torch.bfloat16)
    sae = SAE.load_from_disk(sae_path, device="cuda", dtype="float32")
    sae.requires_grad_(False)
    sae.eval()
    assert all(not parameter.requires_grad for parameter in sae.parameters())
    assert sae.cfg.architecture() == sae_provenance["architecture"]
    assert sae.cfg.d_in == sae_provenance["d_in"] == M51_HIDDEN_SIZE
    assert sae.cfg.d_sae == sae_provenance["d_sae"]
    feature_ids = torch.tensor(feature_id_values, dtype=torch.long, device="cuda")
    genome = Genome(fasta_path, subset_chroms={"21"})
    assert set(genome.chroms) == {"21"}
    torch.cuda.reset_peak_memory_stats()

    orientation_results = [
        extract_orientation(
            frame,
            genome=genome,
            frozen=frozen,
            sae=sae,
            feature_ids=feature_ids,
            orientation=orientation,
            block_index=block_index,
            batch_size=batch_size,
            radius=radius,
            output_dir=output_dir,
        )
        for orientation in ORIENTATIONS
    ]
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": os.environ.get("RUN_ID", ""),
        "spatial_commit": spatial_commit,
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
            "reported_block": block_index + 1,
            "implementation_block_index": block_index,
        },
        "sae": sae_provenance,
        "panel": {
            "sha256": sha256_file(panel_path),
            "rows": frame.height,
            "classes": frame["consequence_cre"].n_unique(),
        },
        "selection": {
            "analysis_manifest_sha256": sha256_file(analysis_manifest_path),
            "analysis_commit": analysis_manifest["analysis_commit"],
            "selected_artifact_sha256": sha256_file(selected_path),
            "selected_rows": selected_frame.height,
            "feature_ids": feature_id_values,
            "feature_count": len(feature_id_values),
            "rule": "unique IDs after discovery ranking and validation transform selection; no test reranking",
        },
        "protocol": {
            "coordinate_system": "0-based half-open after pos0 = pos1 - 1",
            "window_bp": WINDOW_BP,
            "focal_index": FOCAL_INDEX,
            "spatial_radius": radius,
            "relative_positions": list(range(-radius, radius + 1)),
            "orientations": list(ORIENTATIONS),
            "batch_size": batch_size,
            "subset_encoder_validation": "full 31-position equality check against SAE encode on the first item per orientation with rtol=1e-6 and atol=1e-5",
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
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-index", type=int, default=DEFAULT_BLOCK_INDEX)
    parser.add_argument("--radius", type=int, default=SPATIAL_RADIUS)
    args = parser.parse_args()
    manifest = evaluate(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        fasta_path=args.fasta,
        sae_path=args.sae,
        analysis_dir=args.analysis_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        block_index=args.block_index,
        radius=args.radius,
    )
    print(json.dumps(manifest["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
