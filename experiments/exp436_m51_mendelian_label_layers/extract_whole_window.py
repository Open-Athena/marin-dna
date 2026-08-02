"""Extract exact float32 whole-window mean SAE codes for issue 436.

Each allele is encoded at every one of the 255 nucleotide positions after BOS.
The non-negative JumpReLU code is then averaged across positions.  Ref and alt
matrices remain separate so every downstream response is formed in float32.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from huggingface_hub import snapshot_download
from marin_dna.data.genome import Genome
from marin_dna.model.sae import M51_HIDDEN_SIZE, load_frozen_m51
from sae_lens.load_model import HookedProxyLM
from sae_lens.saes.sae import SAE

from extract_focal import (
    BLOCK_INDICES,
    BUDGETS,
    D_SAE,
    EXPECTED_ROWS,
    HOOK_NAMES,
    ISSUE,
    MODEL_ID,
    MODEL_REVISION,
    ORIENTATIONS,
    WINDOW_BP,
    arm_label,
    batch_sequences,
    model_path,
    read_model_provenance,
    sha256_file,
    validate_panel,
    write_json,
)
from train import assert_commit

POOLING = "whole_window_mean"
ALLELES = ("ref", "alt")


@torch.inference_mode()
def extract_raw_windows(
    sequences: Sequence[str],
    *,
    tokenizer: Any,
    model: HookedProxyLM,
) -> dict[int, torch.Tensor]:
    """Return every nucleotide residual, explicitly excluding the BOS token."""

    encoded = tokenizer(
        list(sequences),
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_tensors="pt",
    )
    tokens = encoded["input_ids"].to("cuda")
    assert tokens.shape == (len(sequences), WINDOW_BP + 1)
    output, cache = model.run_with_cache(
        tokens,
        names_filter=list(HOOK_NAMES),
        stop_at_layer=max(BLOCK_INDICES) + 1,
    )
    assert output is None and set(cache) == set(HOOK_NAMES)
    raw: dict[int, torch.Tensor] = {}
    for block_index, hook_name in zip(BLOCK_INDICES, HOOK_NAMES, strict=True):
        captured = cache[hook_name]
        assert captured.shape == (
            len(sequences),
            WINDOW_BP + 1,
            M51_HIDDEN_SIZE,
        )
        nucleotide_residuals = captured[:, 1:, :].float()
        assert nucleotide_residuals.shape == (
            len(sequences),
            WINDOW_BP,
            M51_HIDDEN_SIZE,
        )
        assert torch.isfinite(nucleotide_residuals).all()
        raw[block_index] = nucleotide_residuals
    return raw


@torch.inference_mode()
def mean_sae_code(raw: torch.Tensor, sae: SAE) -> torch.Tensor:
    """Encode each position, then mean-pool the non-negative sparse code."""

    assert raw.ndim == 3 and raw.shape[1:] == (WINDOW_BP, M51_HIDDEN_SIZE)
    encoded = sae.encode(raw)
    assert encoded.shape == (raw.shape[0], WINDOW_BP, D_SAE)
    assert encoded.dtype == torch.float32
    assert torch.isfinite(encoded).all() and torch.all(encoded >= 0)
    pooled = encoded.mean(dim=1, dtype=torch.float32)
    assert pooled.shape == (raw.shape[0], D_SAE)
    assert pooled.dtype == torch.float32
    assert torch.isfinite(pooled).all() and torch.all(pooled >= 0)
    return pooled


def matrix_relative(arm: str, orientation: str, allele: str) -> Path:
    assert orientation in ORIENTATIONS and allele in ALLELES
    return Path(arm) / f"sae_{POOLING}_{orientation}_{allele}.npy"


def create_matrix(path: Path, rows: int = EXPECTED_ROWS) -> np.memmap:
    assert rows > 0 and not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(rows, D_SAE),
    )


def write_paired_batch(
    ref: np.memmap,
    alt: np.memmap,
    pooled: torch.Tensor,
    *,
    offset: int,
    stop: int,
) -> None:
    assert pooled.shape == (2 * (stop - offset), D_SAE)
    values = pooled.cpu().numpy()
    assert values.dtype == np.float32
    ref[offset:stop, :] = values[0::2, :]
    alt[offset:stop, :] = values[1::2, :]


def summarize_matrix(
    path: Path, *, rows: int = EXPECTED_ROWS, chunk_rows: int = 128
) -> dict[str, Any]:
    matrix = np.load(path, mmap_mode="r")
    assert matrix.shape == (rows, D_SAE) and matrix.dtype == np.float32
    nonzero_rows = 0
    feature_seen = np.zeros(D_SAE, dtype=bool)
    minimum = float("inf")
    maximum = float("-inf")
    for offset in range(0, rows, chunk_rows):
        block = np.asarray(matrix[offset : offset + chunk_rows])
        assert np.isfinite(block).all() and np.all(block >= 0)
        nonzero_rows += int(np.count_nonzero(np.any(block != 0, axis=1)))
        feature_seen |= np.any(block != 0, axis=0)
        minimum = min(minimum, float(block.min()))
        maximum = max(maximum, float(block.max()))
    assert nonzero_rows == rows
    return {
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "rows_with_nonzero": nonzero_rows,
        "features_observed": int(feature_seen.sum()),
        "minimum": minimum,
        "maximum": maximum,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def extract(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    fasta_path: Path,
    block01_models_root: Path,
    existing_models_root: Path,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert batch_size > 0
    assert panel_path.is_file() and panel_manifest_path.is_file()
    assert fasta_path.is_file() and Path(f"{fasta_path}.fai").is_file()
    assert Path(f"{fasta_path}.gzi").is_file()
    assert block01_models_root.is_dir() and existing_models_root.is_dir()
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    started = time.monotonic()

    panel_manifest = json.loads(panel_manifest_path.read_text())
    frame = pl.read_parquet(panel_path)
    validate_panel(frame, panel_manifest, panel_path)
    frame = frame.with_row_index("panel_row")
    assert frame["panel_row"].to_list() == list(range(frame.height))
    output_dir.mkdir(parents=True)

    provenance: dict[str, Any] = {}
    saes: dict[str, SAE] = {}
    for block_index in BLOCK_INDICES:
        for budget in BUDGETS:
            label = arm_label(block_index, budget)
            path = model_path(
                block_index=block_index,
                budget=budget,
                block01_models_root=block01_models_root,
                existing_models_root=existing_models_root,
            )
            provenance[label] = read_model_provenance(
                path, block_index=block_index, budget=budget
            )
            sae = SAE.load_from_disk(path, device="cuda", dtype="float32")
            sae.requires_grad_(False)
            sae.eval()
            assert sae.cfg.architecture() == "jumprelu"
            assert all(not parameter.requires_grad for parameter in sae.parameters())
            saes[label] = sae

    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    frozen = load_frozen_m51(checkpoint, device="cuda", dtype=torch.bfloat16)
    frozen.model.config.use_cache = False
    model = HookedProxyLM(frozen.model, frozen.tokenizer, hook_names=list(HOOK_NAMES))
    chroms = set(frame["chrom"].cast(pl.String).unique().to_list())
    genome = Genome(fasta_path, subset_chroms=chroms)
    assert set(genome.chroms) == chroms
    torch.cuda.reset_peak_memory_stats()

    for orientation in ORIENTATIONS:
        matrices: dict[str, tuple[np.memmap, np.memmap]] = {}
        for label in saes:
            matrices[label] = (
                create_matrix(output_dir / matrix_relative(label, orientation, "ref")),
                create_matrix(output_dir / matrix_relative(label, orientation, "alt")),
            )
        try:
            for offset in range(0, frame.height, batch_size):
                stop = min(offset + batch_size, frame.height)
                indices = list(range(offset, stop))
                sequences = batch_sequences(
                    frame, indices, genome=genome, orientation=orientation
                )
                raw_layers = extract_raw_windows(
                    sequences, tokenizer=frozen.tokenizer, model=model
                )
                for block_index in BLOCK_INDICES:
                    raw = raw_layers[block_index]
                    for budget in BUDGETS:
                        label = arm_label(block_index, budget)
                        pooled = mean_sae_code(raw, saes[label])
                        ref_matrix, alt_matrix = matrices[label]
                        write_paired_batch(
                            ref_matrix,
                            alt_matrix,
                            pooled,
                            offset=offset,
                            stop=stop,
                        )
                        del pooled
                    del raw
                del raw_layers
                if offset == 0 or stop == frame.height or stop % (batch_size * 25) == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "extract_whole_window",
                                "orientation": orientation,
                                "processed": stop,
                                "total": frame.height,
                            }
                        ),
                        flush=True,
                    )
        finally:
            for ref_matrix, alt_matrix in matrices.values():
                ref_matrix.flush()
                alt_matrix.flush()
            del matrices
            gc.collect()

    artifacts: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    for label in sorted(saes):
        outputs[label] = {}
        for orientation in ORIENTATIONS:
            outputs[label][orientation] = {}
            for allele in ALLELES:
                relative = matrix_relative(label, orientation, allele)
                summary = summarize_matrix(output_dir / relative)
                outputs[label][orientation][allele] = summary
                artifacts[str(relative)] = {
                    "bytes": summary["bytes"],
                    "sha256": summary["sha256"],
                }

    elapsed = time.monotonic() - started
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": os.environ.get("RUN_ID", ""),
        "experiment_commit": experiment_commit,
        "elapsed_seconds": elapsed,
        "variants_per_second_including_both_orientations": frame.height / elapsed,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "reported_blocks": [index + 1 for index in BLOCK_INDICES],
            "hidden_size": M51_HIDDEN_SIZE,
            "dtype": "bfloat16",
            "use_cache": False,
            "compile_llm": False,
        },
        "saes": provenance,
        "panel": {
            "path": str(panel_path),
            "sha256": sha256_file(panel_path),
            "rows": frame.height,
            "match_groups": frame["match_group"].n_unique(),
            "subsets": sorted(frame["subset"].unique().to_list()),
            "dataset": panel_manifest["dataset"],
        },
        "protocol": {
            "coordinate_boundary": "VCF pos1 -> pos0 = pos1 - 1",
            "window_bp": WINDOW_BP,
            "bos_excluded": True,
            "encoded_token_indices_with_bos": [1, WINDOW_BP],
            "pooling": POOLING,
            "pool_after_sae_encode": True,
            "orientations": list(ORIENTATIONS),
            "batch_size_variants": batch_size,
            "matrix_dtype": "float32",
            "ref_alt_stored_separately": True,
            "quantized": False,
        },
        "outputs": outputs,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--block01-models-root", type=Path, required=True)
    parser.add_argument("--existing-models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result = extract(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        fasta_path=args.fasta,
        block01_models_root=args.block01_models_root,
        existing_models_root=args.existing_models_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
