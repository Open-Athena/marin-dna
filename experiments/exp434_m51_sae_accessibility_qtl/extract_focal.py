"""Extract block-1/10/19 focal SAE responses for issue #434's QTL panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
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

from build_panel import FOCAL_INDEX, WINDOW_BP

ISSUE = 434
MODEL_ID = "marin-dna/marin-dna-exp135-m5.1"
MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
BUDGET = 25_000_200
D_SAE = 15_360
BLOCK_INDICES = (0, 9, 18)
HOOK_NAMES = tuple(f"model.layers.{index}" for index in BLOCK_INDICES)
ORIENTATIONS = ("forward", "reverse_complement")
NUCLEOTIDES = frozenset("ACGT")

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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def read_model_provenance(path: Path, *, block_index: int) -> dict[str, Any]:
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
    cfg = json.loads(paths["cfg.json"].read_text())
    runner = json.loads(paths["runner_cfg.json"].read_text())
    metadata = cfg["metadata"]
    assert metadata["model_name"] == MODEL_ID
    assert metadata["model_revision"] == MODEL_REVISION
    assert metadata["block_index"] == block_index
    assert metadata["report_block"] == block_index + 1
    assert metadata["training_tokens"] == BUDGET
    assert cfg["architecture"] == "jumprelu"
    assert cfg["d_in"] == M51_HIDDEN_SIZE and cfg["d_sae"] == D_SAE
    assert runner["model_name"] == MODEL_ID
    assert runner["model_from_pretrained_kwargs"]["revision"] == MODEL_REVISION
    assert runner["training_tokens"] == BUDGET
    assert runner["sae"]["d_sae"] == D_SAE
    return {
        "architecture": cfg["architecture"],
        "d_in": cfg["d_in"],
        "d_sae": cfg["d_sae"],
        "training_tokens": BUDGET,
        "metadata": metadata,
        "files": {
            name: {"bytes": item.stat().st_size, "sha256": sha256_file(item)}
            for name, item in paths.items()
        },
    }


def validate_panel(frame: pl.DataFrame, manifest: dict[str, Any], path: Path) -> None:
    required = {
        "panel_row",
        "chrom",
        "pos",
        "ref",
        "alt",
        "label",
        "effect",
        "dataset",
        "official_split",
        "ref_sequence",
        "alt_sequence",
    }
    assert required <= set(frame.columns), required - set(frame.columns)
    assert manifest["panel"]["sha256"] == sha256_file(path)
    assert manifest["rows"] == frame.height
    assert frame["panel_row"].to_list() == list(range(frame.height))
    assert frame.filter(pl.col("pos") < 1).is_empty()
    assert frame.filter(pl.col("ref") == pl.col("alt")).is_empty()
    assert frame.filter(
        ~pl.col("ref").is_in(sorted(NUCLEOTIDES))
        | ~pl.col("alt").is_in(sorted(NUCLEOTIDES))
    ).is_empty()
    observed = dict(frame.group_by("dataset").agg(pl.col("label").sum()).iter_rows())
    assert observed == manifest["dataset_positives"]
    assert frame.filter(pl.col("label") & pl.col("effect").is_null()).is_empty()
    assert frame.filter(
        (pl.col("ref_sequence").str.len_chars() != WINDOW_BP)
        | (pl.col("alt_sequence").str.len_chars() != WINDOW_BP)
    ).is_empty()
    assert frame.filter(
        pl.col("ref_sequence").str.slice(FOCAL_INDEX, 1) != pl.col("ref")
    ).is_empty()
    assert frame.filter(
        pl.col("alt_sequence").str.slice(FOCAL_INDEX, 1) != pl.col("alt")
    ).is_empty()


def batch_sequences(
    frame: pl.DataFrame,
    *,
    offset: int,
    length: int,
    orientation: Literal["forward", "reverse_complement"],
) -> list[str]:
    batch = frame.slice(offset, length)
    refs = batch["ref_sequence"].to_list()
    alts = batch["alt_sequence"].to_list()
    if orientation == "reverse_complement":
        refs = [reverse_complement(sequence) for sequence in refs]
        alts = [reverse_complement(sequence) for sequence in alts]
    else:
        assert orientation == "forward"
    sequences = [sequence for pair in zip(refs, alts, strict=True) for sequence in pair]
    assert len(sequences) == 2 * length
    return sequences


@torch.inference_mode()
def extract_raw_focal(
    sequences: list[str],
    *,
    tokenizer: Any,
    model: HookedProxyLM,
) -> dict[int, torch.Tensor]:
    encoded = tokenizer(
        sequences,
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
        assert focal.shape == (len(sequences), M51_HIDDEN_SIZE)
        assert torch.isfinite(focal).all()
        raw[block_index] = focal
    return raw


def sparse_union_table(
    ref_features: np.ndarray,
    alt_features: np.ndarray,
    panel_rows: np.ndarray,
) -> pa.Table:
    assert ref_features.shape == alt_features.shape
    assert ref_features.ndim == 2
    assert panel_rows.shape == (ref_features.shape[0],)
    assert np.isfinite(ref_features).all() and np.isfinite(alt_features).all()
    assert (ref_features >= 0).all() and (alt_features >= 0).all()
    row_index, feature_id = np.nonzero((ref_features != 0) | (alt_features != 0))
    refs = ref_features[row_index, feature_id].astype(np.float32, copy=False)
    alts = alt_features[row_index, feature_id].astype(np.float32, copy=False)
    return pa.Table.from_arrays(
        [
            pa.array(panel_rows[row_index], type=pa.uint32()),
            pa.array(feature_id, type=pa.uint32()),
            pa.array(refs, type=pa.float32()),
            pa.array(alts, type=pa.float32()),
            pa.array(alts - refs, type=pa.float32()),
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
    assert batch_size > 0
    assert panel_path.is_file() and panel_manifest_path.is_file()
    assert models_root.is_dir()
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    started = time.monotonic()

    manifest = json.loads(panel_manifest_path.read_text())
    frame = pl.read_parquet(panel_path)
    validate_panel(frame, manifest, panel_path)
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

    arm_summaries: dict[str, Any] = {
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
                sequences = batch_sequences(
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
                    assert features.shape == (2 * length, D_SAE)
                    assert torch.isfinite(features).all() and torch.all(features >= 0)
                    table = sparse_union_table(
                        features[0::2].cpu().numpy(),
                        features[1::2].cpu().numpy(),
                        panel_rows,
                    )
                    writers[label].write_table(table)
                    arm_summaries[label]["sparse_rows"][orientation] += table.num_rows
                if offset == 0 or stop == frame.height or stop % (batch_size * 25) == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "extract_focal",
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

    artifacts: dict[str, Any] = {}
    for label, summary in arm_summaries.items():
        for orientation in ORIENTATIONS:
            relative = Path(label) / f"sae_focal_{orientation}.parquet"
            path = output_dir / relative
            observed = (
                pl.scan_parquet(path)
                .select(
                    pl.len().alias("rows"),
                    pl.col("panel_row").n_unique().alias("variants_with_nonzero"),
                    pl.col("feature_id").n_unique().alias("features"),
                    pl.col("delta").is_nan().sum().alias("nan_deltas"),
                )
                .collect()
            )
            assert observed["rows"].item() == summary["sparse_rows"][orientation]
            assert observed["nan_deltas"].item() == 0
            summary[orientation] = {
                "rows": observed["rows"].item(),
                "variants_with_nonzero": observed["variants_with_nonzero"].item(),
                "features": observed["features"].item(),
            }
            artifacts[str(relative)] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
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
            "implementation_block_indices": list(BLOCK_INDICES),
            "hidden_size": M51_HIDDEN_SIZE,
            "dtype": "bfloat16",
            "use_cache": False,
            "compile_llm": False,
        },
        "saes": provenance,
        "panel": {
            "scope": manifest["scope"],
            "sha256": sha256_file(panel_path),
            "rows": frame.height,
            "dataset_rows": dict(frame.group_by("dataset").len().iter_rows()),
            "dataset_positives": dict(
                frame.group_by("dataset").agg(pl.col("label").sum()).iter_rows()
            ),
            "dataset_revisions": {
                item["name"]: item["revision"] for item in manifest["datasets"]
            },
        },
        "protocol": {
            "materialized_sequences": True,
            "window_bp": WINDOW_BP,
            "focal_index_after_bos_removal": FOCAL_INDEX,
            "captured_token_index_with_bos": FOCAL_INDEX + 1,
            "orientations": list(ORIENTATIONS),
            "batch_size_variants": batch_size,
            "shared_forward_layers": len(BLOCK_INDICES),
            "saes_per_layer": 1,
            "sae_training_activations": BUDGET,
        },
        "outputs": arm_summaries,
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
    print(json.dumps(result["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
