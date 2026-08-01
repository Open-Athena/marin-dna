"""Extract all eight issue-426 SAE arms on the fixed chr21 variant panel.

Panel positions are 1-based and are converted exactly once to 0-based,
half-open intervals at the FASTA boundary.
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

from train import (
    ARM_TO_BLOCK,
    ARM_TO_HOOK,
    BLOCK_INDICES,
    BUDGETS,
    D_SAE,
    HOOK_NAMES,
    ISSUE,
    MODEL_ID,
    MODEL_REVISION,
    arm_label,
    assert_commit,
    sha256_file,
    write_json,
)

WINDOW_BP = 255
FOCAL_INDEX = 127
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


def read_model_provenance(
    model_path: Path, *, block_index: int, budget: int
) -> dict[str, Any]:
    paths = {
        name: model_path / name
        for name in (
            "cfg.json",
            "runner_cfg.json",
            "sae_weights.safetensors",
            "sparsity.safetensors",
        )
    }
    assert all(path.is_file() for path in paths.values())
    cfg = json.loads(paths["cfg.json"].read_text())
    runner = json.loads(paths["runner_cfg.json"].read_text())
    metadata = cfg["metadata"]
    assert metadata["model_name"] == MODEL_ID
    assert metadata["model_revision"] == MODEL_REVISION
    assert metadata["block_index"] == block_index
    assert metadata["report_block"] == block_index + 1
    assert metadata["training_tokens"] == budget
    assert cfg["architecture"] == "jumprelu"
    assert cfg["d_in"] == M51_HIDDEN_SIZE and cfg["d_sae"] == D_SAE
    assert runner["model_name"] == MODEL_ID
    assert runner["model_from_pretrained_kwargs"]["revision"] == MODEL_REVISION
    assert runner["training_tokens"] == budget
    assert runner["sae"]["d_sae"] == D_SAE
    return {
        "architecture": cfg["architecture"],
        "d_in": cfg["d_in"],
        "d_sae": cfg["d_sae"],
        "training_tokens": budget,
        "metadata": metadata,
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }


def variant_sequences(reference_sequence: str, ref: str, alt: str) -> tuple[str, str]:
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
    assert alternate[FOCAL_INDEX] == alt and len(alternate) == WINDOW_BP
    assert sum(a != b for a, b in zip(reference_sequence, alternate, strict=True)) == 1
    return reference_sequence, alternate


def validate_panel(
    frame: pl.DataFrame, manifest: dict[str, Any], panel_path: Path
) -> None:
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
    assert dict(frame.group_by("split").len().iter_rows()) == EXPECTED_SPLIT_COUNTS
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
    assert ref_features.shape == alt_features.shape
    assert ref_features.ndim == 2
    assert panel_rows.shape == (ref_features.shape[0],)
    assert np.isfinite(ref_features).all() and np.isfinite(alt_features).all()
    assert (ref_features >= 0).all() and (alt_features >= 0).all()
    nonzero = (ref_features != 0) | (alt_features != 0)
    row_index, feature_id = np.nonzero(nonzero)
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
        assert pos0 >= FOCAL_INDEX
        start = pos0 - FOCAL_INDEX
        end = pos0 + FOCAL_INDEX + 1
        assert start >= 0 and end - start == WINDOW_BP
        reference = genome(row["chrom"], start, end, "+").upper()
        ref_sequence, alt_sequence = variant_sequences(
            reference, row["ref"], row["alt"]
        )
        if orientation == "reverse_complement":
            ref_sequence = reverse_complement(ref_sequence)
            alt_sequence = reverse_complement(alt_sequence)
            assert ref_sequence[FOCAL_INDEX] == reverse_complement(row["ref"])
            assert alt_sequence[FOCAL_INDEX] == reverse_complement(row["alt"])
        else:
            assert orientation == "forward"
        sequences.extend((ref_sequence, alt_sequence))
    return sequences


@torch.inference_mode()
def extract_raw_layers(
    sequences: Sequence[str],
    *,
    tokenizer: Any,
    model: HookedProxyLM,
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
    for name, block_index in ARM_TO_BLOCK.items():
        captured = cache[ARM_TO_HOOK[name]]
        assert captured.shape == (len(sequences), WINDOW_BP + 1, M51_HIDDEN_SIZE)
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
    models_root: Path,
    output_dir: Path,
    batch_size: int,
    compile_llm: bool,
) -> dict[str, Any]:
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert batch_size > 0
    assert panel_path.is_file() and panel_manifest_path.is_file()
    assert fasta_path.is_file() and Path(f"{fasta_path}.fai").is_file()
    assert Path(f"{fasta_path}.gzi").is_file() and models_root.is_dir()
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    started = time.monotonic()
    panel_manifest = json.loads(panel_manifest_path.read_text())
    frame = pl.read_parquet(panel_path)
    validate_panel(frame, panel_manifest, panel_path)
    output_dir.mkdir(parents=True)

    provenance: dict[str, Any] = {}
    saes: dict[str, SAE] = {}
    for block_index in BLOCK_INDICES:
        for budget in BUDGETS:
            label = arm_label(block_index, budget)
            path = models_root / label
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
    model = HookedProxyLM(frozen.model, frozen.tokenizer, hook_names=list(HOOK_NAMES))
    if compile_llm:
        model.run_with_cache = torch.compile(  # type: ignore[method-assign]
            model.run_with_cache, mode="reduce-overhead"
        )
    genome = Genome(fasta_path, subset_chroms={"21"})
    assert set(genome.chroms) == {"21"}
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
                    arm_dir / f"sae_activations_{orientation}.parquet",
                    SPARSE_SCHEMA,
                    compression="zstd",
                )
            for offset in range(0, frame.height, batch_size):
                stop = min(offset + batch_size, frame.height)
                indices = list(range(offset, stop))
                sequences = batch_sequences(
                    frame, indices, genome=genome, orientation=orientation
                )
                raw_layers = extract_raw_layers(
                    sequences, tokenizer=frozen.tokenizer, model=model
                )
                panel_rows = frame["panel_row"].slice(offset, stop - offset).to_numpy()
                for block_index in BLOCK_INDICES:
                    raw = raw_layers[block_index]
                    for budget in BUDGETS:
                        label = arm_label(block_index, budget)
                        features = saes[label].encode(raw)
                        assert features.shape == (2 * len(indices), D_SAE)
                        assert torch.isfinite(features).all() and torch.all(
                            features >= 0
                        )
                        table = sparse_union_table(
                            features[0::2].cpu().numpy(),
                            features[1::2].cpu().numpy(),
                            panel_rows,
                        )
                        writers[label].write_table(table)
                        arm_summaries[label]["sparse_rows"][orientation] += (
                            table.num_rows
                        )
                if offset == 0 or stop == frame.height or stop % (batch_size * 25) == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "extract",
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
            relative = Path(label) / f"sae_activations_{orientation}.parquet"
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
            "compile_llm": compile_llm,
        },
        "models": provenance,
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
            "focal_index_after_bos_removal": FOCAL_INDEX,
            "captured_token_index_with_bos": FOCAL_INDEX + 1,
            "orientations": list(ORIENTATIONS),
            "batch_size_variants": batch_size,
            "shared_forward_layers": len(BLOCK_INDICES),
            "saes_per_layer": len(BUDGETS),
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
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    manifest = extract(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        fasta_path=args.fasta,
        models_root=args.models_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        compile_llm=not args.no_compile,
    )
    print(json.dumps(manifest["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
