"""Extract blocks 1/10/19 25M SAE responses on issue #422's frozen panel."""

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

import polars as pl
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from marin_dna.data.dna import reverse_complement
from marin_dna.data.genome import Genome
from marin_dna.model.sae import M51_HIDDEN_SIZE, load_frozen_m51
from sae_lens.load_model import HookedProxyLM
from sae_lens.saes.sae import SAE

from extract import (
    EXPECTED_ROWS,
    FOCAL_INDEX,
    MODEL_ID,
    MODEL_REVISION,
    ORIENTATIONS,
    SPARSE_SCHEMA,
    WINDOW_BP,
    sha256_file,
    sparse_union_table,
    validate_panel,
    variant_sequences,
    write_json,
)

ISSUE = 422
D_SAE = 15_360
TRAINING_TOKENS = 25_000_200
BLOCK_INDICES = (0, 9, 18)
HOOK_NAMES = tuple(f"model.layers.{index}" for index in BLOCK_INDICES)


def arm_label(block_index: int) -> str:
    assert block_index in BLOCK_INDICES
    return f"block{block_index + 1:02d}-25m"


def assert_commit(value: str) -> None:
    assert len(value) == 40
    assert all(character in "0123456789abcdef" for character in value)


def read_model_provenance(path: Path, *, block_index: int) -> dict[str, Any]:
    files = {
        name: path / name
        for name in (
            "cfg.json",
            "runner_cfg.json",
            "sae_weights.safetensors",
            "sparsity.safetensors",
        )
    }
    assert all(item.is_file() for item in files.values())
    cfg = json.loads(files["cfg.json"].read_text())
    runner = json.loads(files["runner_cfg.json"].read_text())
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
            for name, item in files.items()
        },
    }


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
        start = pos0 - FOCAL_INDEX
        end = pos0 + FOCAL_INDEX + 1
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
        assert captured.shape == (
            len(sequences),
            WINDOW_BP + 1,
            M51_HIDDEN_SIZE,
        )
        focal = captured[:, FOCAL_INDEX + 1, :].float()
        assert focal.shape == (len(sequences), M51_HIDDEN_SIZE)
        assert torch.isfinite(focal).all()
        raw[block_index] = focal
    return raw


def extract(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    fasta_path: Path,
    sae_paths: dict[int, Path],
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert batch_size > 0
    assert panel_path.is_file() and panel_manifest_path.is_file()
    assert fasta_path.is_file() and Path(f"{fasta_path}.fai").is_file()
    assert Path(f"{fasta_path}.gzi").is_file()
    assert set(sae_paths) == set(BLOCK_INDICES)
    assert all(path.is_dir() for path in sae_paths.values())
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    started = time.monotonic()

    panel_manifest = json.loads(panel_manifest_path.read_text())
    frame = pl.read_parquet(panel_path)
    validate_panel(frame, panel_manifest, panel_path)
    assert frame["panel_row"].to_list() == list(range(EXPECTED_ROWS))
    output_dir.mkdir(parents=True)

    provenance: dict[str, Any] = {}
    saes: dict[int, SAE] = {}
    for block_index in BLOCK_INDICES:
        label = arm_label(block_index)
        provenance[label] = read_model_provenance(
            sae_paths[block_index], block_index=block_index
        )
        sae = SAE.load_from_disk(sae_paths[block_index], device="cuda", dtype="float32")
        sae.requires_grad_(False)
        sae.eval()
        assert sae.cfg.architecture() == "jumprelu"
        assert sae.cfg.d_in == M51_HIDDEN_SIZE and sae.cfg.d_sae == D_SAE
        assert all(not parameter.requires_grad for parameter in sae.parameters())
        saes[block_index] = sae

    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    frozen = load_frozen_m51(checkpoint, device="cuda", dtype=torch.bfloat16)
    frozen.model.config.use_cache = False
    model = HookedProxyLM(frozen.model, frozen.tokenizer, hook_names=list(HOOK_NAMES))
    genome = Genome(fasta_path, subset_chroms={"21"})
    assert set(genome.chroms) == {"21"}
    torch.cuda.reset_peak_memory_stats()

    summaries: dict[str, Any] = {
        arm_label(block_index): {
            "sparse_rows": {orientation: 0 for orientation in ORIENTATIONS}
        }
        for block_index in BLOCK_INDICES
    }
    for orientation in ORIENTATIONS:
        writers: dict[int, pq.ParquetWriter] = {}
        try:
            for block_index in BLOCK_INDICES:
                arm_dir = output_dir / arm_label(block_index)
                arm_dir.mkdir(parents=True, exist_ok=True)
                writers[block_index] = pq.ParquetWriter(
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
                    features = saes[block_index].encode(raw_layers[block_index])
                    assert features.shape == (2 * len(indices), D_SAE)
                    assert torch.isfinite(features).all() and torch.all(features >= 0)
                    table = sparse_union_table(
                        features[0::2].cpu().numpy(),
                        features[1::2].cpu().numpy(),
                        panel_rows,
                    )
                    writers[block_index].write_table(table)
                    summaries[arm_label(block_index)]["sparse_rows"][orientation] += (
                        table.num_rows
                    )
                if offset == 0 or stop == frame.height or stop % (batch_size * 25) == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "extract_multilayer",
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
    for block_index in BLOCK_INDICES:
        label = arm_label(block_index)
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
            assert (
                observed["rows"].item() == summaries[label]["sparse_rows"][orientation]
            )
            assert observed["nan_deltas"].item() == 0
            summaries[label][orientation] = {
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
            "path": str(panel_path),
            "sha256": sha256_file(panel_path),
            "manifest_sha256": sha256_file(panel_manifest_path),
            "rows": frame.height,
            "classes": frame["consequence_cre"].n_unique(),
            "source": panel_manifest["source"],
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
            "training_tokens_per_sae": TRAINING_TOKENS,
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
    parser.add_argument("--block01-sae", type=Path, required=True)
    parser.add_argument("--block10-sae", type=Path, required=True)
    parser.add_argument("--block19-sae", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = extract(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        fasta_path=args.fasta,
        sae_paths={0: args.block01_sae, 9: args.block10_sae, 18: args.block19_sae},
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(json.dumps(result["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
