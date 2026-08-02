"""Extract block-1/10/19 focal SAE activations for issue #440."""

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
from marin_dna.model.sae import M51_HIDDEN_SIZE, load_frozen_m51
from sae_lens.load_model import HookedProxyLM
from sae_lens.saes.sae import SAE

from build_panel import FOCAL_INDEX, REFERENCE_CLASSES, WINDOW_BP

ISSUE = 440
EXTRACTION_RUN_ID = "dna-exp440-reference-state-focal-seed288-r1"
PANEL_RUN_ID = "dna-exp440-reference-state-panel-r1"
PANEL_SHA256 = "e06d3a513ad437d5202a41187123598f18631e5d7f8d84ba322b7830b837e063"
PANEL_MANIFEST_SHA256 = (
    "82caf694ed74050cea9cedab829e2d31fa6b5668dbe08cd1dc2cbfc0c68a9c68"
)
MODEL_ID = "marin-dna/marin-dna-exp135-m5.1"
MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
TRAINING_TOKENS = 25_000_200
D_SAE = 15_360
BLOCK_INDICES = (0, 9, 18)
HOOK_NAMES = tuple(f"model.layers.{index}" for index in BLOCK_INDICES)
ORIENTATIONS = ("forward", "reverse_complement")

EXPECTED_SAE_ARTIFACTS = {
    "block01-25m": {
        "cfg.json": (
            1_286,
            "e96f615d346068a7f2b32343474bf2d08615aa9bedc167bf529984d12db71b4c",
        ),
        "runner_cfg.json": (
            6_184,
            "2373149c73663b87f3da77186ba46fc06fc4e4261617ad9f9833eb010f847d83",
        ),
        "sae_weights.safetensors": (
            236_060_560,
            "97b7c23c76abc1a38a45fd3dcea241285811c2e810bcc371c314af9bb332a353",
        ),
        "sparsity.safetensors": (
            61_520,
            "39f50d1bd29bb6ef1e3cc091271193a90e3625d07ebf120c04492d5398717098",
        ),
    },
    "block10-25m": {
        "cfg.json": (
            1_283,
            "cfc161c05921a787cdcd6a369c9416a00e356030d31d2c9c0a78b6bbdccd51b9",
        ),
        "runner_cfg.json": (
            12_049,
            "5082e318b95bdef98446556ff94b768dc1ca123da0df4a7bc79de71b065cf554",
        ),
        "sae_weights.safetensors": (
            236_060_560,
            "606b81e2cc34ad7225de0fbaf5e673e688c4f990fc748cb59223316893e826b6",
        ),
        "sparsity.safetensors": (
            61_520,
            "e6a2776c487d6a84de0fc0b5c093560611bb5252cc5f88cc09322f8d00c03082",
        ),
    },
    "block19-25m": {
        "cfg.json": (
            1_287,
            "8825220f296bea463f266bda9e0497be3ebcc956f8f846109178aaa45ff06848",
        ),
        "runner_cfg.json": (
            12_053,
            "d36576c21a33a4b64a507559266dff757156b5032a277dfbd68498fc3bfa62a8",
        ),
        "sae_weights.safetensors": (
            236_060_560,
            "e4f10ba59f10be943dbdc33f469f986f598c5e34fcba42577efad27717231533",
        ),
        "sparsity.safetensors": (
            61_520,
            "ef641aeb1be378356881a81563a9886d81ae0edc511d1ff5669d8ed71990d465",
        ),
    },
}

SPARSE_SCHEMA = pa.schema(
    [
        pa.field("panel_row", pa.uint32(), nullable=False),
        pa.field("feature_id", pa.uint32(), nullable=False),
        pa.field("activation", pa.float32(), nullable=False),
    ]
)

assert WINDOW_BP == 2 * FOCAL_INDEX + 1
assert M51_HIDDEN_SIZE == 1_920


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


def arm_label(block_index: int) -> str:
    assert block_index in BLOCK_INDICES
    return f"block{block_index + 1:02d}-25m"


def model_path(*, block_index: int, models_root: Path) -> Path:
    path = models_root / arm_label(block_index)
    assert path.is_dir(), path
    return path


def read_model_provenance(
    path: Path,
    *,
    block_index: int,
    expected_artifacts: dict[str, tuple[int, str]] | None = None,
) -> dict[str, Any]:
    paths = {
        name: path / name
        for name in (
            "cfg.json",
            "runner_cfg.json",
            "sae_weights.safetensors",
            "sparsity.safetensors",
        )
    }
    assert all(item.is_file() for item in paths.values())
    expected = expected_artifacts or EXPECTED_SAE_ARTIFACTS[arm_label(block_index)]
    assert set(expected) == set(paths)
    for name, item in paths.items():
        expected_bytes, expected_sha256 = expected[name]
        assert item.stat().st_size == expected_bytes
        assert sha256_file(item) == expected_sha256

    cfg = json.loads(paths["cfg.json"].read_text())
    runner = json.loads(paths["runner_cfg.json"].read_text())
    metadata = cfg["metadata"]
    assert metadata["model_name"] == MODEL_ID
    assert metadata["model_revision"] == MODEL_REVISION
    assert metadata["block_index"] == block_index
    assert metadata["report_block"] == block_index + 1
    assert metadata["training_tokens"] == TRAINING_TOKENS
    assert cfg["architecture"] == "jumprelu"
    assert cfg["d_in"] == M51_HIDDEN_SIZE and cfg["d_sae"] == D_SAE
    assert runner["model_name"] == MODEL_ID
    assert runner["model_from_pretrained_kwargs"]["revision"] == MODEL_REVISION
    assert runner["training_tokens"] == TRAINING_TOKENS
    assert runner["sae"]["d_sae"] == D_SAE
    return {
        "architecture": cfg["architecture"],
        "d_in": cfg["d_in"],
        "d_sae": cfg["d_sae"],
        "training_tokens": TRAINING_TOKENS,
        "metadata": metadata,
        "files": {
            name: {"bytes": item.stat().st_size, "sha256": sha256_file(item)}
            for name, item in paths.items()
        },
    }


def validate_panel(
    frame: pl.DataFrame, manifest: dict[str, Any], panel_path: Path
) -> None:
    required = {
        "panel_row",
        "name",
        "reference_class",
        "chrom",
        "start",
        "end",
        "sequence",
    }
    assert required <= set(frame.columns), required - set(frame.columns)
    assert manifest["issue"] == ISSUE and manifest["run_id"] == PANEL_RUN_ID
    assert manifest["window_bp"] == WINDOW_BP
    assert manifest["focal_index"] == FOCAL_INDEX
    assert manifest["panel"]["sha256"] == PANEL_SHA256 == sha256_file(panel_path)
    assert manifest["rows"] == frame.height == 14_336
    assert manifest["classes"] == list(REFERENCE_CLASSES)
    assert frame["panel_row"].to_list() == list(range(frame.height))
    assert frame["name"].n_unique() == frame.height
    assert (
        frame.select(pl.struct("chrom", "start", "end").n_unique()).item()
        == frame.height
    )
    assert frame.filter(pl.col("start") < 0).is_empty()
    assert frame.filter(pl.col("end") - pl.col("start") != WINDOW_BP).is_empty()
    assert frame.filter(pl.col("sequence").str.len_chars() != WINDOW_BP).is_empty()
    assert frame.filter(pl.col("sequence").str.contains("[^ACGT]")).is_empty()
    observed_counts = dict(
        frame.group_by("reference_class").len().sort("reference_class").iter_rows()
    )
    assert observed_counts == manifest["class_counts"]
    assert set(observed_counts) == set(REFERENCE_CLASSES)
    assert set(observed_counts.values()) == {2_048}


def load_panel(panel_path: Path, manifest_path: Path) -> pl.DataFrame:
    assert panel_path.is_file() and manifest_path.is_file()
    assert sha256_file(manifest_path) == PANEL_MANIFEST_SHA256
    manifest = json.loads(manifest_path.read_text())
    frame = pl.read_parquet(panel_path)
    validate_panel(frame, manifest, panel_path)
    return frame


def oriented_sequences(
    frame: pl.DataFrame,
    *,
    offset: int,
    length: int,
    orientation: Literal["forward", "reverse_complement"],
) -> list[str]:
    sequences = frame["sequence"].slice(offset, length).to_list()
    if orientation == "reverse_complement":
        sequences = [reverse_complement(sequence) for sequence in sequences]
    else:
        assert orientation == "forward"
    return sequences


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


def sparse_activation_table(features: np.ndarray, panel_rows: np.ndarray) -> pa.Table:
    assert features.ndim == 2 and features.shape[1] == D_SAE
    assert panel_rows.shape == (features.shape[0],)
    assert np.isfinite(features).all() and np.all(features >= 0)
    row_index, feature_id = np.nonzero(features)
    activations = features[row_index, feature_id].astype(np.float32, copy=False)
    return pa.Table.from_arrays(
        [
            pa.array(panel_rows[row_index], type=pa.uint32()),
            pa.array(feature_id, type=pa.uint32()),
            pa.array(activations, type=pa.float32()),
        ],
        schema=SPARSE_SCHEMA,
    )


def extract(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    models_root: Path,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert batch_size > 0 and models_root.is_dir() and not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    assert os.environ.get("RUN_ID") == EXTRACTION_RUN_ID
    started = time.monotonic()
    frame = load_panel(panel_path, panel_manifest_path)
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
                length = stop - offset
                sequences = oriented_sequences(
                    frame,
                    offset=offset,
                    length=length,
                    orientation=orientation,
                )
                raw_layers = extract_raw_focal(
                    sequences, tokenizer=frozen.tokenizer, model=model
                )
                panel_rows = frame["panel_row"].slice(offset, length).to_numpy()
                for block_index in BLOCK_INDICES:
                    label = arm_label(block_index)
                    features = saes[label].encode(raw_layers[block_index])
                    assert features.shape == (length, D_SAE)
                    assert torch.isfinite(features).all() and torch.all(features >= 0)
                    table = sparse_activation_table(features.cpu().numpy(), panel_rows)
                    writers[label].write_table(table)
                    summaries[label]["sparse_rows"][orientation] += table.num_rows
                if offset == 0 or stop == frame.height or stop % (batch_size * 25) == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "extract_reference_focal",
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
                    pl.col("panel_row").n_unique().alias("contexts_with_nonzero"),
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
                "mean_nonzero_per_context": rows / frame.height,
                "nonzero_slot_fraction": rows / (frame.height * D_SAE),
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
        "analysis_status": "frozen_reference_state_sae_extraction",
        "experiment_commit": experiment_commit,
        "elapsed_seconds": elapsed,
        "sequences_per_second_including_both_orientations": (
            2 * frame.height / elapsed
        ),
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
            "panel_sha256": PANEL_SHA256,
            "manifest_sha256": PANEL_MANIFEST_SHA256,
            "rows": frame.height,
            "class_counts": dict(frame.group_by("reference_class").len().iter_rows()),
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
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    result = extract(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        models_root=args.models_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
