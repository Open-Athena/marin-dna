"""Extract selected SAE feature profiles for issue #429 causal perturbations."""

from __future__ import annotations

import argparse
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
from marin_dna.model.sae import (
    M51_HIDDEN_SIZE,
    M51GenomicWindow,
    load_frozen_m51,
    run_m51_with_activations,
)

from analyze import sha256_file, write_json
from sample_panel import assert_current_commit

ISSUE = 429
WINDOW_BP = 255
FOCAL_INDEX = 127
SPATIAL_RADIUS = 15
MODEL_ID = "marin-dna/marin-dna-exp135-m5.1"
MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
DEFAULT_BLOCK_INDEX = 9
ORIENTATIONS = ("forward", "reverse_complement")
FEATURE_IDS = (3312, 4281, 6072, 11681, 11698)
EXPECTED_CLASSES = {
    "splice_acceptor_variant",
    "splice_donor_5th_base_variant",
    "stop_gained",
    "synonymous_variant",
}

assert WINDOW_BP == 2 * FOCAL_INDEX + 1
assert M51_HIDDEN_SIZE == 1_920


def build_state_table(
    panel: pl.DataFrame,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Deduplicate repeated reference states while preserving paired row indices."""

    required = {
        "chrom",
        "window_start0",
        "window_end0",
        "reference_sequence",
        "alternate_sequence",
    }
    assert required <= set(panel.columns)
    state_index: dict[tuple[str, int, int, str], int] = {}
    state_rows: list[dict[str, Any]] = []
    reference_indices = np.empty(panel.height, dtype=np.int64)
    alternate_indices = np.empty(panel.height, dtype=np.int64)
    for row_index, row in enumerate(panel.iter_rows(named=True)):
        for state, sequence_column, output in (
            ("reference", "reference_sequence", reference_indices),
            ("alternate", "alternate_sequence", alternate_indices),
        ):
            sequence = str(row[sequence_column])
            key = (
                str(row["chrom"]),
                int(row["window_start0"]),
                int(row["window_end0"]),
                sequence,
            )
            index = state_index.get(key)
            if index is None:
                index = len(state_rows)
                state_index[key] = index
                state_rows.append(
                    {
                        "state_index": index,
                        "first_seen_as": state,
                        "chrom": key[0],
                        "window_start0": key[1],
                        "window_end0": key[2],
                        "sequence": key[3],
                    }
                )
            output[row_index] = index
    states = pl.DataFrame(state_rows)
    assert states["state_index"].to_list() == list(range(states.height))
    assert reference_indices.min() >= 0 and alternate_indices.min() >= 0
    assert reference_indices.max() < states.height
    assert alternate_indices.max() < states.height
    return states, reference_indices, alternate_indices


def validate_design(
    panel: pl.DataFrame,
    manifest: dict[str, Any],
    *,
    panel_path: Path,
) -> None:
    """Validate the commit-pinned perturbation panel and its sequence pairs."""

    required = {
        "perturbation_row",
        "perturbation_type",
        "class",
        "feature_id",
        "context_group",
        "chrom",
        "window_start0",
        "window_end0",
        "edit_distance",
        "reference_sequence",
        "alternate_sequence",
    }
    assert required <= set(panel.columns), required - set(panel.columns)
    assert manifest["artifacts"][panel_path.name]["sha256"] == sha256_file(panel_path)
    assert manifest["rows"] == panel.height and panel.height > 0
    assert panel["perturbation_row"].to_list() == list(range(panel.height))
    assert set(panel["class"].unique()) == EXPECTED_CLASSES
    assert set(panel["feature_id"].unique()) == set(FEATURE_IDS) - {4281}
    context_groups = set(panel["context_group"].unique())
    declared_context_group = manifest["protocol"].get("context_group")
    if declared_context_group is None:
        assert context_groups == set(manifest["protocol"]["context_groups"])
    else:
        assert context_groups == {declared_context_group}
    for row in panel.select(
        "window_start0",
        "window_end0",
        "edit_distance",
        "reference_sequence",
        "alternate_sequence",
    ).iter_rows(named=True):
        reference = row["reference_sequence"]
        alternate = row["alternate_sequence"]
        assert row["window_end0"] - row["window_start0"] == WINDOW_BP
        assert len(reference) == len(alternate) == WINDOW_BP
        observed_distance = sum(
            ref != alt for ref, alt in zip(reference, alternate, strict=True)
        )
        assert observed_distance == row["edit_distance"]
        assert 1 <= observed_distance <= 3


def encode_selected_features(
    sae: Any, raw: torch.Tensor, feature_ids: torch.Tensor
) -> torch.Tensor:
    """Encode frozen feature IDs with the exact exported JumpReLU formula."""

    assert not sae.training and raw.shape[-1] == sae.cfg.d_in
    assert feature_ids.ndim == 1 and feature_ids.dtype == torch.long
    assert sae.cfg.architecture() == "jumprelu"
    assert sae.cfg.normalize_activations == "none"
    assert not sae.hook_z_reshaping_mode
    with torch.inference_mode():
        full_identity = len(feature_ids) == sae.cfg.d_sae and torch.equal(
            feature_ids, torch.arange(sae.cfg.d_sae, device=feature_ids.device)
        )
        if full_identity:
            selected = sae.encode(raw)
        else:
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


def assert_selected_features_match_full(
    selected: torch.Tensor, expected: torch.Tensor
) -> None:
    """Validate subset GEMM output against full SAE encoding.

    Selecting encoder columns changes the GEMM reduction shape, so fp32 outputs
    can differ by a few ulps. The JumpReLU support must still match exactly;
    only active-value accumulation drift receives a floating-point tolerance.
    """

    assert selected.shape == expected.shape
    assert torch.equal(selected > 0, expected > 0)
    torch.testing.assert_close(selected, expected, rtol=1e-5, atol=5e-5)


def extract_state_batch(
    states: pl.DataFrame,
    indices: Sequence[int],
    *,
    frozen: Any,
    sae: Any,
    feature_ids: torch.Tensor,
    orientation: Literal["forward", "reverse_complement"],
    block_index: int,
    radius: int,
    validate_subset: bool,
) -> np.ndarray:
    """Return selected-feature profiles for unique designed sequence states."""

    sequences: list[str] = []
    windows: list[M51GenomicWindow] = []
    for index in indices:
        row = states.row(index, named=True)
        sequence = row["sequence"]
        strand = "+"
        if orientation == "reverse_complement":
            sequence = reverse_complement(sequence)
            strand = "-"
        sequences.append(sequence)
        windows.append(
            M51GenomicWindow(
                chrom=row["chrom"],
                start=int(row["window_start0"]),
                end=int(row["window_end0"]),
                strand=strand,
            )
        )

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
            expected = sae.encode(raw[:1]).index_select(-1, feature_ids)
        assert_selected_features_match_full(selected[:1], expected)
    assert selected.shape == (len(indices), 2 * radius + 1, len(feature_ids))
    return selected.cpu().numpy()


def extract_orientation(
    states: pl.DataFrame,
    *,
    frozen: Any,
    sae: Any,
    feature_ids: torch.Tensor,
    orientation: Literal["forward", "reverse_complement"],
    block_index: int,
    batch_size: int,
    radius: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Write selected-feature profiles for all unique states in one orientation."""

    shape = (states.height, 2 * radius + 1, len(feature_ids))
    path = output_dir / f"state_activations_{orientation}.npy"
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    for offset in range(0, states.height, batch_size):
        stop = min(offset + batch_size, states.height)
        output[offset:stop] = extract_state_batch(
            states,
            list(range(offset, stop)),
            frozen=frozen,
            sae=sae,
            feature_ids=feature_ids,
            orientation=orientation,
            block_index=block_index,
            radius=radius,
            validate_subset=offset == 0,
        )
        if offset == 0 or stop == states.height or stop % (batch_size * 25) == 0:
            print(
                json.dumps(
                    {
                        "orientation": orientation,
                        "processed": stop,
                        "total": states.height,
                    }
                ),
                flush=True,
            )
    output.flush()
    assert np.isfinite(output).all()
    return {"path": path.name, "shape": list(shape), "dtype": "float32"}


def evaluate(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    sae_path: Path,
    output_dir: Path,
    batch_size: int,
    block_index: int,
    radius: int,
) -> dict[str, Any]:
    """Extract a commit-pinned, deduplicated causal perturbation response map."""

    from sae_lens.saes.sae import SAE

    from spatial import read_sae_provenance

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert batch_size > 0 and block_index >= 0 and 0 < radius < FOCAL_INDEX
    assert panel_path.is_file() and panel_manifest_path.is_file()
    assert sae_path.is_dir() and not output_dir.exists()
    extraction_commit = os.environ.get("PERTURBATION_EXTRACTION_COMMIT", "")
    assert_current_commit(extraction_commit)
    started = time.monotonic()

    panel_manifest = json.loads(panel_manifest_path.read_text())
    panel = pl.read_parquet(panel_path)
    validate_design(panel, panel_manifest, panel_path=panel_path)
    states, reference_indices, alternate_indices = build_state_table(panel)
    assert panel.height < states.height <= 2 * panel.height
    sae_provenance = read_sae_provenance(sae_path, block_index=block_index)
    assert max(FEATURE_IDS) < sae_provenance["d_sae"]

    output_dir.mkdir(parents=True, exist_ok=False)
    states.write_parquet(output_dir / "perturbation_states.parquet")
    np.save(output_dir / "reference_state_indices.npy", reference_indices)
    np.save(output_dir / "alternate_state_indices.npy", alternate_indices)
    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    frozen = load_frozen_m51(checkpoint, device="cuda", dtype=torch.bfloat16)
    sae = SAE.load_from_disk(sae_path, device="cuda", dtype="float32")
    sae.requires_grad_(False)
    sae.eval()
    assert all(not parameter.requires_grad for parameter in sae.parameters())
    feature_ids = torch.tensor(FEATURE_IDS, dtype=torch.long, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    orientation_results = {
        orientation: extract_orientation(
            states,
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
    }
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": os.environ.get("RUN_ID", ""),
        "perturbation_extraction_commit": extraction_commit,
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
        "design": {
            "manifest_sha256": sha256_file(panel_manifest_path),
            "panel_sha256": sha256_file(panel_path),
            "design_commit": panel_manifest["design_commit"],
            "paired_rows": panel.height,
        },
        "deduplication": {
            "paired_sequence_states": 2 * panel.height,
            "unique_sequence_states": states.height,
            "saved_forward_fraction": 1 - states.height / (2 * panel.height),
        },
        "protocol": {
            "window_bp": WINDOW_BP,
            "focal_index": FOCAL_INDEX,
            "spatial_radius": radius,
            "relative_positions": list(range(-radius, radius + 1)),
            "orientations": list(ORIENTATIONS),
            "feature_ids": list(FEATURE_IDS),
            "batch_size": batch_size,
            "base_model_dtype": "bfloat16",
            "sae_dtype": "float32",
            "torch_compile": False,
            "torch_compile_reason": "pinned dynamic hook-cache path is validated in eager mode",
        },
        "outputs": orientation_results,
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
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-index", type=int, default=DEFAULT_BLOCK_INDEX)
    parser.add_argument("--radius", type=int, default=SPATIAL_RADIUS)
    args = parser.parse_args()
    manifest = evaluate(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        sae_path=args.sae,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        block_index=args.block_index,
        radius=args.radius,
    )
    print(json.dumps(manifest["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
