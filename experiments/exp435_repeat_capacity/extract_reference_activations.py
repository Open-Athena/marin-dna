"""Extract layer-resolved sparse focal activations for the frozen repeat panel."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from marin_dna.data.dna import reverse_complement
from marin_dna.model.sae import M51_HIDDEN_SIZE, load_frozen_m51
from sae_lens.load_model import HookedProxyLM
from sae_lens.saes.sae import SAE

from common import assert_commit, write_json
from extract_common import (
    BLOCK_INDICES,
    CONTEXTS,
    D_SAE,
    EXTRACTION_RUN_ID,
    FOCAL_INDEX,
    HOOK_NAMES,
    ISSUE,
    MODEL_ID,
    MODEL_REVISION,
    ORIENTATIONS,
    PANEL_ARCHIVE_MANIFEST_SHA256,
    PANEL_RUN_ID,
    SPARSE_SCHEMA,
    TRAINING_TOKENS,
    WINDOW_BP,
    arm_label,
    read_model_provenance,
    sha256_file,
    sparse_activation_table,
)

assert M51_HIDDEN_SIZE == 1_920


def validate_panel_archive(
    panel_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], pl.DataFrame]:
    archive_path = panel_root / "archive_manifest.json"
    assert archive_path.is_file()
    assert sha256_file(archive_path) == PANEL_ARCHIVE_MANIFEST_SHA256
    archive = json.loads(archive_path.read_text())
    assert archive["issue"] == ISSUE and archive["run_id"] == PANEL_RUN_ID
    assert archive["analysis_status"] == "outcome_blind_reference_panel"
    for relative, expected in archive["artifacts"].items():
        path = panel_root / relative
        assert path.is_file() and path.stat().st_size == expected["bytes"]
        assert sha256_file(path) == expected["sha256"]

    panel_dir = panel_root / "panel"
    manifest_path = panel_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["issue"] == ISSUE and manifest["run_id"] == PANEL_RUN_ID
    assert manifest["analysis_status"] == "outcome_blind_reference_panel"
    assert manifest["contexts"] == CONTEXTS
    assert manifest["window_bp"] == WINDOW_BP
    assert manifest["focal_index"] == FOCAL_INDEX
    for relative, expected in manifest["artifacts"].items():
        path = panel_dir / relative
        assert path.is_file() and path.stat().st_size == expected["bytes"]
        assert sha256_file(path) == expected["sha256"]

    frame = pl.read_parquet(panel_dir / "contexts.parquet").sort("context_id")
    assert frame.height == CONTEXTS
    assert frame["context_id"].to_list() == list(range(CONTEXTS))
    assert frame["context_id"].n_unique() == CONTEXTS
    assert frame["sequence"].str.len_chars().unique().to_list() == [WINDOW_BP]
    assert frame.filter((pl.col("end0") - pl.col("start0")) != WINDOW_BP).is_empty()
    assert frame.filter(pl.col("pos0") != pl.col("start0") + FOCAL_INDEX).is_empty()
    assert frame.filter(
        ~pl.col("is_repeat") & (pl.col("repeat_fraction") != 0)
    ).is_empty()
    return archive, manifest, frame


def model_path(*, block_index: int, models_root: Path) -> Path:
    path = models_root / arm_label(block_index)
    assert path.is_dir(), path
    return path


@torch.inference_mode()
def extract_raw_focal(
    sequences: Sequence[str], *, tokenizer: Any, model: HookedProxyLM
) -> dict[int, torch.Tensor]:
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
        assert captured.shape == (len(sequences), WINDOW_BP + 1, M51_HIDDEN_SIZE)
        focal = captured[:, FOCAL_INDEX + 1, :].float()
        assert torch.isfinite(focal).all()
        raw[block_index] = focal
    return raw


def extract(
    *,
    panel_root: Path,
    models_root: Path,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert batch_size > 0 and not output_dir.exists() and models_root.is_dir()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    assert os.environ.get("RUN_ID") == EXTRACTION_RUN_ID
    started = time.monotonic()
    archive, panel_manifest, frame = validate_panel_archive(panel_root)
    output_dir.mkdir(parents=True)

    provenance: dict[str, Any] = {}
    saes: dict[str, SAE] = {}
    for block_index in BLOCK_INDICES:
        label = arm_label(block_index)
        path = model_path(block_index=block_index, models_root=models_root)
        provenance[label] = read_model_provenance(path, block_index=block_index)
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
    torch.cuda.reset_peak_memory_stats()

    summaries: dict[str, Any] = {
        label: {"sparse_rows": {orientation: 0 for orientation in ORIENTATIONS}}
        for label in saes
    }
    for orientation in ORIENTATIONS:
        writers: dict[str, pq.ParquetWriter] = {}
        try:
            for label in saes:
                arm_dir = output_dir / label
                arm_dir.mkdir(parents=True, exist_ok=True)
                writers[label] = pq.ParquetWriter(
                    arm_dir / f"sae_focal_{orientation}.parquet",
                    SPARSE_SCHEMA,
                    compression="zstd",
                )
            for offset in range(0, frame.height, batch_size):
                stop = min(offset + batch_size, frame.height)
                sequences = frame["sequence"].slice(offset, stop - offset).to_list()
                if orientation == "reverse_complement":
                    sequences = [reverse_complement(sequence) for sequence in sequences]
                else:
                    assert orientation == "forward"
                raw_layers = extract_raw_focal(
                    sequences, tokenizer=frozen.tokenizer, model=model
                )
                context_ids = (
                    frame["context_id"].slice(offset, stop - offset).to_numpy()
                )
                for block_index in BLOCK_INDICES:
                    label = arm_label(block_index)
                    features = saes[label].encode(raw_layers[block_index])
                    assert features.shape == (len(sequences), D_SAE)
                    assert torch.isfinite(features).all() and torch.all(features >= 0)
                    table = sparse_activation_table(features.cpu().numpy(), context_ids)
                    writers[label].write_table(table)
                    summaries[label]["sparse_rows"][orientation] += table.num_rows
                if offset == 0 or stop == frame.height or stop % (batch_size * 25) == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "extract_repeat_reference",
                                "orientation": orientation,
                                "processed": stop,
                                "total": frame.height,
                            }
                        ),
                        flush=True,
                    )
        finally:
            for writer in writers.values():
                writer.close()

    torch.cuda.synchronize()
    artifacts: dict[str, Any] = {}
    for label, summary in summaries.items():
        for orientation in ORIENTATIONS:
            relative = Path(label) / f"sae_focal_{orientation}.parquet"
            path = output_dir / relative
            observed = (
                pl.scan_parquet(path)
                .select(
                    pl.len().alias("rows"),
                    pl.col("context_id").n_unique().alias("contexts_with_nonzero"),
                    pl.col("feature_id").n_unique().alias("features"),
                    pl.col("activation").sum().alias("activation_sum"),
                    pl.col("activation").is_nan().sum().alias("nan_activations"),
                )
                .collect()
            )
            rows = int(observed["rows"].item())
            assert rows == summary["sparse_rows"][orientation]
            assert observed["nan_activations"].item() == 0
            summary[orientation] = {
                "rows": rows,
                "contexts_with_nonzero": int(observed["contexts_with_nonzero"].item()),
                "features": int(observed["features"].item()),
                "activation_sum": float(observed["activation_sum"].item()),
                "mean_nonzero_per_context": rows / CONTEXTS,
                "nonzero_slot_fraction": rows / (CONTEXTS * D_SAE),
            }
            artifacts[str(relative)] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    elapsed = time.monotonic() - started
    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": EXTRACTION_RUN_ID,
        "analysis_status": "frozen_reference_sae_extraction",
        "experiment_commit": experiment_commit,
        "elapsed_seconds": elapsed,
        "sequences_per_second": (2 * CONTEXTS) / elapsed,
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
            "implementation_block_indices": list(BLOCK_INDICES),
            "hidden_size": M51_HIDDEN_SIZE,
            "dtype": "bfloat16",
            "use_cache": False,
            "torch_compile": False,
            "torch_compile_reason": (
                "the pinned dynamic SAE-Lens hook-cache path is validated in eager mode"
            ),
        },
        "saes": provenance,
        "panel": {
            "run_id": PANEL_RUN_ID,
            "archive_manifest_sha256": PANEL_ARCHIVE_MANIFEST_SHA256,
            "archive_objects_excluding_manifest": archive[
                "object_count_excluding_this_manifest"
            ],
            "panel_manifest_sha256": sha256_file(
                panel_root / "panel" / "manifest.json"
            ),
            "panel_experiment_commit": panel_manifest["experiment_commit"],
            "contexts": CONTEXTS,
        },
        "protocol": {
            "window_bp": WINDOW_BP,
            "focal_index_after_bos_removal": FOCAL_INDEX,
            "captured_token_index_with_bos": FOCAL_INDEX + 1,
            "orientations": list(ORIENTATIONS),
            "batch_size_sequences": batch_size,
            "shared_forward_layers": len(BLOCK_INDICES),
            "saes_per_layer": 1,
            "training_tokens_per_sae": TRAINING_TOKENS,
            "panel_loading": "in-memory compact parquet; no FASTA or dataloader",
            "tokenizer_parallelism": True,
            "sparse_storage": "one row per nonzero focal activation",
        },
        "outputs": summaries,
        "artifacts": artifacts,
    }
    result_path = output_dir / "results.json"
    write_json(result_path, result)
    result["artifacts"]["results.json"] = {
        "bytes": result_path.stat().st_size,
        "sha256": sha256_file(result_path),
    }
    write_json(output_dir / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    result = extract(
        panel_root=args.panel_root,
        models_root=args.models_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
