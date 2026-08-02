"""Extract focal SAE ref/alt features for issue 438's complex-traits panel.

Input positions are VCF-style 1-based coordinates. They are converted exactly
once to 0-based, half-open coordinates at the FASTA boundary.
"""

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
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from marin_dna.data.dna import reverse_complement
from marin_dna.data.genome import Genome
from marin_dna.model.sae import M51_HIDDEN_SIZE, load_frozen_m51
from sae_lens.load_model import HookedProxyLM
from sae_lens.saes.sae import SAE

from common import (
    D_SAE,
    EXPECTED_SAE_ARTIFACTS,
    ISSUE,
    MODEL_ID,
    MODEL_REVISION,
    TRAINING_TOKENS,
    assert_commit,
    sha256_file,
    write_json,
)
from prepare_panel import (
    EXPECTED_ROWS,
)
from prepare_panel import (
    validate_panel as validate_pinned_panel,
)

WINDOW_BP = 255
FOCAL_INDEX = 127
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


def arm_label(block_index: int, budget: int = TRAINING_TOKENS) -> str:
    assert block_index in BLOCK_INDICES and budget == TRAINING_TOKENS
    return f"block{block_index + 1:02d}-{budget // 1_000_000}m"


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


def variant_sequences(reference: str, ref: str, alt: str) -> tuple[str, str]:
    reference, ref, alt = reference.upper(), ref.upper(), alt.upper()
    assert len(reference) == WINDOW_BP and set(reference) <= NUCLEOTIDES
    assert len(ref) == len(alt) == 1 and ref != alt
    assert ref in NUCLEOTIDES and alt in NUCLEOTIDES
    assert reference[FOCAL_INDEX] == ref
    alternate = reference[:FOCAL_INDEX] + alt + reference[FOCAL_INDEX + 1 :]
    assert len(alternate) == WINDOW_BP and alternate[FOCAL_INDEX] == alt
    assert sum(a != b for a, b in zip(reference, alternate, strict=True)) == 1
    return reference, alternate


def batch_sequences(
    frame: pl.DataFrame,
    indices: Sequence[int],
    *,
    genome: Genome,
    orientation: Literal["forward", "reverse_complement"],
) -> list[str]:
    sequences: list[str] = []
    for index in indices:
        row = frame.row(index, named=True)
        pos0 = int(row["pos"]) - 1
        start, end = pos0 - FOCAL_INDEX, pos0 + FOCAL_INDEX + 1
        assert start >= 0 and end - start == WINDOW_BP
        reference = genome(str(row["chrom"]), start, end, "+").upper()
        ref_sequence, alt_sequence = variant_sequences(
            reference, str(row["ref"]), str(row["alt"])
        )
        if orientation == "reverse_complement":
            ref_sequence = reverse_complement(ref_sequence)
            alt_sequence = reverse_complement(alt_sequence)
            assert ref_sequence[FOCAL_INDEX] == reverse_complement(str(row["ref"]))
            assert alt_sequence[FOCAL_INDEX] == reverse_complement(str(row["alt"]))
        else:
            assert orientation == "forward"
        sequences.extend((ref_sequence, alt_sequence))
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


def sparse_union_table(
    ref_features: np.ndarray,
    alt_features: np.ndarray,
    panel_rows: np.ndarray,
) -> pa.Table:
    assert ref_features.shape == alt_features.shape
    assert ref_features.ndim == 2 and panel_rows.shape == (ref_features.shape[0],)
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
    fasta_path: Path,
    models_root: Path,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert batch_size > 0 and not output_dir.exists()
    assert panel_path.is_file() and panel_manifest_path.is_file()
    assert fasta_path.is_file() and Path(f"{fasta_path}.fai").is_file()
    assert Path(f"{fasta_path}.gzi").is_file() and models_root.is_dir()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    started = time.monotonic()

    manifest = json.loads(panel_manifest_path.read_text())
    pinned = validate_pinned_panel(panel_path)
    assert pinned == manifest
    frame = pl.read_parquet(panel_path).with_row_index("panel_row")
    assert frame["panel_row"].to_list() == list(range(EXPECTED_ROWS))
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
    chroms = set(frame["chrom"].cast(pl.String).unique().to_list())
    genome = Genome(fasta_path, subset_chroms=chroms)
    assert set(genome.chroms) == chroms
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
                indices = list(range(offset, stop))
                sequences = batch_sequences(
                    frame, indices, genome=genome, orientation=orientation
                )
                raw_layers = extract_raw_focal(
                    sequences, tokenizer=frozen.tokenizer, model=model
                )
                panel_rows = frame["panel_row"].slice(offset, stop - offset).to_numpy()
                for block_index in BLOCK_INDICES:
                    label = arm_label(block_index)
                    features = saes[label].encode(raw_layers[block_index])
                    assert features.shape == (2 * len(indices), D_SAE)
                    assert torch.isfinite(features).all() and torch.all(features >= 0)
                    table = sparse_union_table(
                        features[0::2].cpu().numpy(),
                        features[1::2].cpu().numpy(),
                        panel_rows,
                    )
                    writers[label].write_table(table)
                    summaries[label]["sparse_rows"][orientation] += table.num_rows
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
    for label, summary in summaries.items():
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

    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": os.environ.get("RUN_ID", ""),
        "experiment_commit": experiment_commit,
        "elapsed_seconds": time.monotonic() - started,
        "variants_per_second_including_both_orientations": frame.height
        / (time.monotonic() - started),
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
            "sha256": sha256_file(panel_path),
            "rows": frame.height,
            "match_groups": frame["match_group"].n_unique(),
            "subsets": sorted(frame["subset"].unique().to_list()),
            "dataset": manifest["dataset"],
            "revision": manifest["revision"],
        },
        "protocol": {
            "coordinate_boundary": "VCF pos1 -> pos0 = pos1 - 1",
            "window_bp": WINDOW_BP,
            "focal_index_after_bos_removal": FOCAL_INDEX,
            "captured_token_index_with_bos": FOCAL_INDEX + 1,
            "orientations": list(ORIENTATIONS),
            "batch_size_variants": batch_size,
            "shared_forward_layers": len(BLOCK_INDICES),
            "saes_per_layer": 1,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = extract(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        fasta_path=args.fasta,
        models_root=args.models_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
