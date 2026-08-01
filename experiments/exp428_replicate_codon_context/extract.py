"""Extract the three preregistered SAE features for ref/alt and FWD/RC.

Panel positions are 1-based. They are converted exactly once to 0-based,
half-open coordinates at the FASTA boundary.
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

LOCAL_THREAD_LIMITS = {
    "POLARS_MAX_THREADS": 2,
    "RAYON_NUM_THREADS": 2,
    "OMP_NUM_THREADS": 1,
    "MKL_NUM_THREADS": 1,
    "OPENBLAS_NUM_THREADS": 1,
    "NUMEXPR_NUM_THREADS": 1,
}
for variable, limit in LOCAL_THREAD_LIMITS.items():
    configured = int(os.environ.get(variable, limit))
    assert configured > 0, (variable, configured)
    os.environ[variable] = str(min(configured, limit))

import numpy as np
import polars as pl
import torch
from huggingface_hub import snapshot_download
from marin_dna.data.dna import reverse_complement
from marin_dna.data.genome import Genome
from marin_dna.model.sae import M51_HIDDEN_SIZE, load_frozen_m51
from sae_lens.load_model import HookedProxyLM
from sae_lens.saes.sae import SAE

from panel import assert_current_commit, sha256_file, write_json

ISSUE = 428
MODEL_ID = "marin-dna/marin-dna-exp135-m5.1"
MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
SOURCE_EXPERIMENT_COMMIT = "c39f38815cfaebe58e6a0f5648856334a0427fd6"
HOOK_NAME = "model.layers.18"
BLOCK_INDEX = 18
REPORT_BLOCK = 19
D_SAE = 15_360
WINDOW_BP = 255
FOCAL_INDEX = 127
CAPTURED_TOKEN_INDEX = FOCAL_INDEX + 1
ORIENTATIONS = ("forward", "reverse_complement")
NUCLEOTIDES = frozenset("ACGT")

SAE_SPECS: dict[str, dict[str, Any]] = {
    "block19-5m": {
        "training_tokens": 5_000_550,
        "feature_ids": (11_064, 12_658),
        "file_sha256": {
            "cfg.json": "9a1d9848bb76f47b250bbba81bc2ae685243e7436699334e73cf6ae2ad9fa15c",
            "runner_cfg.json": "e30aaa053d1b2625c7b86918b320fa1a5fb5ab9d0d62f6a2fef984127aaae620",
            "sae_weights.safetensors": "a35abcd7d8b9098b3574bff1270cd177117b687ade5845471403900b46f00971",
            "sparsity.safetensors": "22b53e0455988b6824cd9f3dd54ba050b15b13a617ce033df2d3b78724a10fa2",
        },
    },
    "block19-25m": {
        "training_tokens": 25_000_200,
        "feature_ids": (13_637,),
        "file_sha256": {
            "cfg.json": "8825220f296bea463f266bda9e0497be3ebcc956f8f846109178aaa45ff06848",
            "runner_cfg.json": "d36576c21a33a4b64a507559266dff757156b5032a277dfbd68498fc3bfa62a8",
            "sae_weights.safetensors": "e4f10ba59f10be943dbdc33f469f986f598c5e34fcba42577efad27717231533",
            "sparsity.safetensors": "ef641aeb1be378356881a81563a9886d81ae0edc511d1ff5669d8ed71990d465",
        },
    },
}

FEATURE_COLUMNS = (
    "f11064_5m",
    "f12658_5m",
    "f13637_25m",
)
FEATURE_LOCATIONS = {
    "f11064_5m": ("block19-5m", 11_064),
    "f12658_5m": ("block19-5m", 12_658),
    "f13637_25m": ("block19-25m", 13_637),
}

assert WINDOW_BP == 2 * FOCAL_INDEX + 1
assert M51_HIDDEN_SIZE == 1_920


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_file_hashes(
    root: Path, expected: dict[str, str]
) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for name, expected_sha256 in expected.items():
        path = root / name
        assert path.is_file(), path
        sha256 = sha256_file(path)
        assert sha256 == expected_sha256, (path, sha256, expected_sha256)
        observed[name] = {"bytes": path.stat().st_size, "sha256": sha256}
    return observed


def read_model_provenance(models_root: Path) -> dict[str, Any]:
    source_manifest_path = models_root.parent / "manifest.json"
    assert source_manifest_path.is_file()
    source_manifest = json.loads(source_manifest_path.read_text())
    assert source_manifest["experiment_commit"] == SOURCE_EXPERIMENT_COMMIT
    provenance: dict[str, Any] = {}
    for label, spec in SAE_SPECS.items():
        model_path = models_root / label
        files = verify_file_hashes(model_path, spec["file_sha256"])
        cfg = json.loads((model_path / "cfg.json").read_text())
        runner = json.loads((model_path / "runner_cfg.json").read_text())
        metadata = cfg["metadata"]
        assert metadata["model_name"] == MODEL_ID
        assert metadata["model_revision"] == MODEL_REVISION
        assert metadata["hook_name"] == HOOK_NAME
        assert metadata["block_index"] == BLOCK_INDEX
        assert metadata["report_block"] == REPORT_BLOCK
        assert metadata["training_tokens"] == spec["training_tokens"]
        assert cfg["architecture"] == "jumprelu"
        assert cfg["normalize_activations"] == "none"
        assert cfg["d_in"] == M51_HIDDEN_SIZE and cfg["d_sae"] == D_SAE
        assert runner["model_name"] == MODEL_ID
        assert runner["model_from_pretrained_kwargs"]["revision"] == MODEL_REVISION
        assert runner["training_tokens"] == spec["training_tokens"]
        assert runner["sae"]["d_sae"] == D_SAE
        provenance[label] = {
            "architecture": "jumprelu",
            "block_index": BLOCK_INDEX,
            "report_block": REPORT_BLOCK,
            "training_tokens": spec["training_tokens"],
            "feature_ids": list(spec["feature_ids"]),
            "metadata": metadata,
            "files": files,
        }
    return {
        "source_experiment_commit": SOURCE_EXPERIMENT_COMMIT,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "saes": provenance,
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
        "consequence_cre",
        "block_id",
        "split",
        "matching_stratum",
        "consensus_codon_position",
        "transcript_substitution",
    }
    assert required <= set(frame.columns), required - set(frame.columns)
    assert manifest["issue"] == ISSUE
    assert manifest["output"]["sha256"] == sha256_file(panel_path)
    assert manifest["output"]["rows"] == frame.height > 0
    assert frame["panel_row"].to_list() == list(range(frame.height))
    assert (
        frame.select(pl.struct("chrom", "pos", "ref", "alt").n_unique()).item()
        == frame.height
    )
    assert frame["chrom"].unique().to_list() == ["21"]
    assert set(frame["consequence_cre"].unique()) == {
        "missense_variant",
        "synonymous_variant",
    }
    assert frame.select(sorted(required)).null_count().sum_horizontal().sum() == 0
    assert frame.filter(pl.col("pos") <= FOCAL_INDEX).is_empty()
    assert frame.filter(pl.col("ref") == pl.col("alt")).is_empty()
    assert frame.filter(
        ~pl.col("ref").is_in(sorted(NUCLEOTIDES))
        | ~pl.col("alt").is_in(sorted(NUCLEOTIDES))
    ).is_empty()
    balance = frame.group_by(["split", "matching_stratum", "consequence_cre"]).len()
    for split in ("discovery", "validation", "test"):
        strata = frame.filter(pl.col("split") == split)["matching_stratum"].unique()
        assert len(strata) > 0
        for stratum in strata:
            observed = balance.filter(
                (pl.col("split") == split) & (pl.col("matching_stratum") == stratum)
            )
            assert observed.height == 2 and observed["len"].n_unique() == 1
    assert (
        frame.group_by("block_id").agg(pl.col("split").n_unique())["split"].max() == 1
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
def extract_raw_layer(
    sequences: Sequence[str], *, tokenizer: Any, model: HookedProxyLM
) -> torch.Tensor:
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
        names_filter=[HOOK_NAME],
        stop_at_layer=REPORT_BLOCK,
    )
    assert output is None and set(cache) == {HOOK_NAME}
    captured = cache[HOOK_NAME]
    assert captured.shape == (len(sequences), WINDOW_BP + 1, M51_HIDDEN_SIZE)
    focal = captured[:, CAPTURED_TOKEN_INDEX, :].float()
    assert focal.shape == (len(sequences), M51_HIDDEN_SIZE)
    assert torch.isfinite(focal).all()
    return focal


def selected_feature_values(
    encoded: torch.Tensor, feature_ids: Sequence[int]
) -> np.ndarray:
    assert encoded.ndim == 2 and encoded.shape[1] == D_SAE
    assert torch.isfinite(encoded).all() and torch.all(encoded >= 0)
    selected = encoded[:, list(feature_ids)].float().cpu().numpy()
    assert selected.shape == (encoded.shape[0], len(feature_ids))
    assert np.isfinite(selected).all() and (selected >= 0).all()
    return selected


def dense_output_frame(
    panel_rows: np.ndarray, values: dict[str, np.ndarray]
) -> pl.DataFrame:
    assert set(values) == set(FEATURE_COLUMNS)
    columns: dict[str, Any] = {"panel_row": panel_rows.astype(np.uint32)}
    for name in FEATURE_COLUMNS:
        matrix = values[name]
        assert matrix.shape == (len(panel_rows), 2)
        assert np.isfinite(matrix).all() and (matrix >= 0).all()
        ref = matrix[:, 0].astype(np.float32, copy=False)
        alt = matrix[:, 1].astype(np.float32, copy=False)
        columns[f"{name}_ref"] = ref
        columns[f"{name}_alt"] = alt
        columns[f"{name}_delta"] = (alt - ref).astype(np.float32, copy=False)
    frame = pl.DataFrame(columns)
    assert frame.height == len(panel_rows)
    assert frame["panel_row"].n_unique() == frame.height
    return frame


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
    assert batch_size > 0
    assert panel_path.is_file() and panel_manifest_path.is_file()
    assert fasta_path.is_file() and Path(f"{fasta_path}.fai").is_file()
    assert Path(f"{fasta_path}.gzi").is_file() and models_root.is_dir()
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_current_commit(experiment_commit)
    started = time.monotonic()
    panel_manifest = json.loads(panel_manifest_path.read_text())
    frame = pl.read_parquet(panel_path)
    validate_panel(frame, panel_manifest, panel_path)
    provenance = read_model_provenance(models_root)
    output_dir.mkdir(parents=True)

    saes: dict[str, SAE] = {}
    for label in SAE_SPECS:
        sae = SAE.load_from_disk(models_root / label, device="cuda", dtype="float32")
        sae.requires_grad_(False)
        sae.eval()
        assert sae.cfg.architecture() == "jumprelu"
        assert all(not parameter.requires_grad for parameter in sae.parameters())
        saes[label] = sae

    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    frozen = load_frozen_m51(checkpoint, device="cuda", dtype=torch.bfloat16)
    frozen.model.config.use_cache = False
    model = HookedProxyLM(frozen.model, frozen.tokenizer, hook_names=[HOOK_NAME])
    genome = Genome(fasta_path, subset_chroms={"21"})
    assert set(genome.chroms) == {"21"}
    torch.cuda.reset_peak_memory_stats()

    output_metadata: dict[str, Any] = {}
    for orientation in ORIENTATIONS:
        collected = {
            name: np.empty((frame.height, 2), dtype=np.float32)
            for name in FEATURE_COLUMNS
        }
        for offset in range(0, frame.height, batch_size):
            stop = min(offset + batch_size, frame.height)
            indices = list(range(offset, stop))
            raw = extract_raw_layer(
                batch_sequences(frame, indices, genome=genome, orientation=orientation),
                tokenizer=frozen.tokenizer,
                model=model,
            )
            by_model: dict[str, np.ndarray] = {}
            for label, spec in SAE_SPECS.items():
                encoded = saes[label].encode(raw)
                by_model[label] = selected_feature_values(encoded, spec["feature_ids"])
            for name, (label, feature_id) in FEATURE_LOCATIONS.items():
                local_index = SAE_SPECS[label]["feature_ids"].index(feature_id)
                selected = by_model[label][:, local_index]
                collected[name][offset:stop, 0] = selected[0::2]
                collected[name][offset:stop, 1] = selected[1::2]
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
        output = dense_output_frame(frame["panel_row"].to_numpy(), collected)
        output_path = output_dir / f"selected_features_{orientation}.parquet"
        output.write_parquet(output_path, compression="zstd")
        observed = pl.read_parquet(output_path)
        assert observed.equals(output)
        output_metadata[orientation] = {
            "path_name": output_path.name,
            "rows": output.height,
            "columns": output.columns,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
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
            "hook_name": HOOK_NAME,
            "implementation_block_index": BLOCK_INDEX,
            "reported_block": REPORT_BLOCK,
            "hidden_size": M51_HIDDEN_SIZE,
            "dtype": "bfloat16",
            "use_cache": False,
            "compile_llm": False,
            "compile_note": "eager run_with_cache is the known-correct hook path",
        },
        "models": provenance,
        "feature_protocol_sha256": sha256_json(
            {name: list(location) for name, location in FEATURE_LOCATIONS.items()}
        ),
        "panel": {
            "path_name": panel_path.name,
            "sha256": sha256_file(panel_path),
            "manifest_sha256": sha256_file(panel_manifest_path),
            "rows": frame.height,
        },
        "protocol": {
            "coordinate_system": "0-based half-open after pos0 = pos1 - 1",
            "window_bp": WINDOW_BP,
            "focal_index_after_bos_removal": FOCAL_INDEX,
            "captured_token_index_with_bos": CAPTURED_TOKEN_INDEX,
            "orientations": list(ORIENTATIONS),
            "batch_size_variants": batch_size,
            "selected_features": {
                name: {"sae": label, "feature_id": feature_id}
                for name, (label, feature_id) in FEATURE_LOCATIONS.items()
            },
        },
        "outputs": output_metadata,
    }
    write_json(output_dir / "manifest.json", result)
    result["manifest_sha256"] = sha256_file(output_dir / "manifest.json")
    write_json(output_dir / "results.json", result)
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
    print(json.dumps(result["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
