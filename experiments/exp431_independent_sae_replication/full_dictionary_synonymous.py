"""Preregistered whole-dictionary sensitivity analysis for synonymous variants.

Discovery and validation are deliberately separate from the final test command.
The search command has no test-path argument, so its job can omit the test panel
while freezing the selected feature and response view.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from huggingface_hub import snapshot_download
from marin_dna.model.sae import load_frozen_m51

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP429_DIR = REPO_ROOT / "experiments" / "exp429_variant_feature_map"
if str(EXP429_DIR) not in sys.path:
    sys.path.insert(0, str(EXP429_DIR))

from extract_perturbations import (
    MODEL_ID,
    MODEL_REVISION,
    ORIENTATIONS,
    build_state_table,
    extract_orientation,
    sha256_file,
    validate_design,
)
from sample_panel import assert_current_commit
from spatial import read_sae_provenance

from analyze_replication import (
    BOOTSTRAPS,
    DISCOVERY_SHORTLIST,
    MIN_SUPPORT_CONTEXTS,
    SPATIAL_RADIUS,
    VIEW_NAMES,
    bootstrap_mean,
    context_effects,
    response_views,
)
from decoder_neighbors import D_SAE, normalized_decoder
from extract_candidates import validate_state_table

ISSUE = 431
BLOCK_INDEX = 9
CLASS_NAME = "synonymous_variant"
CONCEPT = "synonymous_degeneracy"
REFERENCE_FEATURE_ID = 6_072
FEATURE_CHUNK = 256


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def synonymous_panel(panel: pl.DataFrame) -> pl.DataFrame:
    """Return only synonymous source designs with a new contiguous row index."""

    assert "perturbation_row" in panel.columns and "class" in panel.columns
    result = (
        panel.filter(pl.col("class") == CLASS_NAME)
        .rename({"perturbation_row": "original_perturbation_row"})
        .with_row_index("perturbation_row")
    )
    assert result.height > 0
    assert set(result["class"].unique()) == {CLASS_NAME}
    assert result["perturbation_row"].to_list() == list(range(result.height))
    return result


def extract(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    sae_path: Path,
    output_dir: Path,
    dictionary_name: str,
    source_split: str,
    stage: str,
    feature_ids: list[int],
    batch_size: int,
) -> dict[str, Any]:
    """Extract FWD/RC state profiles for the synonymous-only panel."""

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert stage in {"search", "test"}
    assert source_split in {"discovery", "validation", "test"}
    assert (stage == "search") == (source_split in {"discovery", "validation"})
    assert dictionary_name and batch_size > 0 and not output_dir.exists()
    assert feature_ids == sorted(set(feature_ids))
    assert feature_ids and min(feature_ids) >= 0 and max(feature_ids) < D_SAE
    if stage == "search":
        assert feature_ids == list(range(D_SAE))
    else:
        assert len(feature_ids) == 1
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_current_commit(experiment_commit)
    started = time.monotonic()

    full_manifest = json.loads(panel_manifest_path.read_text())
    full_panel = pl.read_parquet(panel_path)
    validate_design(full_panel, full_manifest, panel_path=panel_path)
    assert set(full_panel["source_split"].unique()) == {source_split}
    panel = synonymous_panel(full_panel)
    assert panel.select("source_panel_row").n_unique() >= MIN_SUPPORT_CONTEXTS
    states, reference_indices, alternate_indices = build_state_table(panel)
    validate_state_table(panel, states, reference_indices, alternate_indices)
    sae_provenance = read_sae_provenance(sae_path, block_index=BLOCK_INDEX)
    assert sae_provenance["d_sae"] == D_SAE

    output_dir.mkdir(parents=True)
    panel_path_out = output_dir / "synonymous_panel.parquet"
    panel.write_parquet(panel_path_out)
    states.write_parquet(output_dir / "perturbation_states.parquet")
    np.save(output_dir / "feature_ids.npy", np.asarray(feature_ids, dtype=np.int64))
    np.save(output_dir / "reference_state_indices.npy", reference_indices)
    np.save(output_dir / "alternate_state_indices.npy", alternate_indices)

    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    frozen = load_frozen_m51(checkpoint, device="cuda", dtype=torch.bfloat16)
    from sae_lens.saes.sae import SAE

    sae = SAE.load_from_disk(sae_path, device="cuda", dtype="float32").eval()
    sae.requires_grad_(False)
    feature_tensor = torch.tensor(feature_ids, dtype=torch.long, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    orientation_results = {
        orientation: extract_orientation(
            states,
            frozen=frozen,
            sae=sae,
            feature_ids=feature_tensor,
            orientation=orientation,
            block_index=BLOCK_INDEX,
            batch_size=batch_size,
            radius=SPATIAL_RADIUS,
            output_dir=output_dir,
        )
        for orientation in ORIENTATIONS
    }
    result = {
        "issue": ISSUE,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_commit": experiment_commit,
        "dictionary": dictionary_name,
        "stage": stage,
        "source_split": source_split,
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
            "reported_block": BLOCK_INDEX + 1,
            "implementation_block_index": BLOCK_INDEX,
            "dtype": "bfloat16",
        },
        "sae": sae_provenance,
        "design": {
            "full_panel_manifest_sha256": sha256_file(panel_manifest_path),
            "full_panel_sha256": sha256_file(panel_path),
            "synonymous_panel_sha256": sha256_file(panel_path_out),
            "paired_rows": panel.height,
            "source_contexts": panel.select("source_panel_row").n_unique(),
            "unique_states": states.height,
        },
        "protocol": {
            "concept": CONCEPT,
            "orientations": list(ORIENTATIONS),
            "spatial_radius": SPATIAL_RADIUS,
            "feature_count": len(feature_ids),
            "batch_size": batch_size,
            "base_model_dtype": "bfloat16",
            "sae_dtype": "float32",
            "torch_compile": False,
        },
        "orientation_outputs": orientation_results,
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


def load_extraction(
    path: Path,
    *,
    expected_split: str,
) -> tuple[
    pl.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """Load and hash-check one extraction while keeping activation arrays mapped."""

    manifest_path = path / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["issue"] == ISSUE and manifest["source_split"] == expected_split
    for name, metadata in manifest["artifacts"].items():
        artifact = path / name
        assert artifact.is_file() and artifact.stat().st_size == metadata["bytes"]
        assert sha256_file(artifact) == metadata["sha256"]
    panel = pl.read_parquet(path / "synonymous_panel.parquet")
    feature_ids = np.load(path / "feature_ids.npy")
    reference_indices = np.load(path / "reference_state_indices.npy")
    alternate_indices = np.load(path / "alternate_state_indices.npy")
    assert panel.height == len(reference_indices) == len(alternate_indices)
    arrays = {
        orientation: np.load(
            path / f"state_activations_{orientation}.npy", mmap_mode="r"
        )
        for orientation in ORIENTATIONS
    }
    expected_shape = (manifest["design"]["unique_states"], 31, len(feature_ids))
    assert all(array.shape == expected_shape for array in arrays.values())
    return panel, feature_ids, reference_indices, alternate_indices, arrays, manifest


def deltas_for_columns(
    arrays: dict[str, np.ndarray],
    reference_indices: np.ndarray,
    alternate_indices: np.ndarray,
    columns: slice | np.ndarray,
) -> dict[str, np.ndarray]:
    """Materialize paired deltas only for a bounded feature column subset."""

    result: dict[str, np.ndarray] = {}
    for orientation, array in arrays.items():
        selected = np.asarray(array[:, :, columns])
        result[orientation] = selected[alternate_indices] - selected[reference_indices]
    assert result["forward"].ndim == 3
    assert result["forward"].shape == result["reverse_complement"].shape
    return result


def rank_feature_views(
    panel: pl.DataFrame,
    feature_ids: np.ndarray,
    views: dict[str, np.ndarray],
) -> pl.DataFrame:
    """Score each feature/view by equal-context synonymous contrast."""

    rows: list[dict[str, Any]] = []
    for view_name in VIEW_NAMES:
        _, effects = context_effects(panel, views[view_name], concept=CONCEPT)
        means = effects.mean(axis=0)
        support = (np.abs(effects) > 1e-8).sum(axis=0)
        assert len(means) == len(feature_ids)
        rows.extend(
            {
                "feature_id": int(feature_id),
                "view": view_name,
                "effect": float(effect),
                "abs_effect": float(abs(effect)),
                "support_contexts": int(feature_support),
            }
            for feature_id, effect, feature_support in zip(
                feature_ids, means, support, strict=True
            )
        )
    return pl.DataFrame(rows)


def score_full_dictionary(
    panel: pl.DataFrame,
    feature_ids: np.ndarray,
    reference_indices: np.ndarray,
    alternate_indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    *,
    chunk_size: int = FEATURE_CHUNK,
) -> pl.DataFrame:
    """Score a full dictionary in bounded feature chunks."""

    assert np.array_equal(feature_ids, np.arange(D_SAE))
    assert chunk_size > 0
    frames: list[pl.DataFrame] = []
    for start in range(0, D_SAE, chunk_size):
        stop = min(start + chunk_size, D_SAE)
        deltas = deltas_for_columns(
            arrays, reference_indices, alternate_indices, slice(start, stop)
        )
        views = response_views(deltas["forward"], deltas["reverse_complement"])
        frames.append(rank_feature_views(panel, feature_ids[start:stop], views))
        print(
            json.dumps({"scored_features": stop, "total_features": D_SAE}), flush=True
        )
    result = pl.concat(frames).sort(
        ["abs_effect", "feature_id", "view"], descending=[True, False, False]
    )
    assert result.height == D_SAE * len(VIEW_NAMES)
    return result


def decoder_relationship(
    reference_sae: Path,
    candidate_sae: Path,
    feature_id: int,
) -> dict[str, Any]:
    """Measure the frozen candidate's geometry relative to reference feature 6072."""

    reference = normalized_decoder(reference_sae)
    candidate = normalized_decoder(candidate_sae)
    similarities = candidate @ reference[REFERENCE_FEATURE_ID]
    selected_cosine = float(similarities[feature_id])
    global_rank = int((similarities > similarities[feature_id]).sum()) + 1
    reverse = candidate[feature_id] @ reference.T
    nearest_cosine, nearest_id = reverse.max(dim=0)
    return {
        "reference_feature_id": REFERENCE_FEATURE_ID,
        "decoder_cosine": selected_cosine,
        "decoder_global_rank": global_rank,
        "candidate_nearest_reference_feature_id": int(nearest_id),
        "candidate_nearest_reference_cosine": float(nearest_cosine),
        "mutual_nearest": global_rank == 1 and int(nearest_id) == REFERENCE_FEATURE_ID,
    }


def select(
    *,
    discovery_dir: Path,
    validation_dir: Path,
    reference_sae: Path,
    candidate_sae: Path,
    output_dir: Path,
    dictionary_name: str,
    chunk_size: int = FEATURE_CHUNK,
) -> dict[str, Any]:
    """Use discovery and validation only to freeze one feature/view."""

    assert dictionary_name and not output_dir.exists()
    discovery = load_extraction(discovery_dir, expected_split="discovery")
    validation = load_extraction(validation_dir, expected_split="validation")
    assert discovery[5]["dictionary"] == validation[5]["dictionary"] == dictionary_name
    assert discovery[5]["stage"] == validation[5]["stage"] == "search"
    assert np.array_equal(discovery[1], validation[1])
    discovery_scores = score_full_dictionary(
        discovery[0],
        discovery[1],
        discovery[2],
        discovery[3],
        discovery[4],
        chunk_size=chunk_size,
    )
    shortlist = (
        discovery_scores.filter(pl.col("support_contexts") >= MIN_SUPPORT_CONTEXTS)
        .head(DISCOVERY_SHORTLIST)
        .rename(
            {
                "effect": "discovery_effect",
                "abs_effect": "discovery_abs_effect",
                "support_contexts": "discovery_support_contexts",
            }
        )
    )
    assert shortlist.height == DISCOVERY_SHORTLIST

    selected_feature_ids = np.asarray(
        sorted(set(shortlist["feature_id"].to_list())), dtype=np.int64
    )
    feature_columns = np.searchsorted(validation[1], selected_feature_ids)
    assert np.array_equal(validation[1][feature_columns], selected_feature_ids)
    deltas = deltas_for_columns(
        validation[4], validation[2], validation[3], feature_columns
    )
    validation_views = response_views(deltas["forward"], deltas["reverse_complement"])
    validation_scores = rank_feature_views(
        validation[0], selected_feature_ids, validation_views
    ).rename(
        {
            "effect": "validation_effect",
            "abs_effect": "validation_abs_effect",
            "support_contexts": "validation_support_contexts",
        }
    )
    candidates = (
        shortlist.join(
            validation_scores,
            on=["feature_id", "view"],
            how="inner",
            validate="1:1",
        )
        .with_columns(
            (
                (pl.col("discovery_effect") * pl.col("validation_effect") > 0)
                & (pl.col("validation_support_contexts") >= MIN_SUPPORT_CONTEXTS)
            ).alias("validation_confirmed")
        )
        .sort(
            [
                "validation_confirmed",
                "validation_abs_effect",
                "discovery_abs_effect",
                "feature_id",
                "view",
            ],
            descending=[True, True, True, False, False],
        )
    )
    eligible = candidates.filter(pl.col("validation_confirmed"))
    selection: dict[str, Any]
    if eligible.is_empty():
        selection = {
            "status": "no_validation_confirmed_candidate",
            "concept": CONCEPT,
        }
    else:
        row = eligible.row(0, named=True)
        feature_id = int(row["feature_id"])
        geometry = decoder_relationship(reference_sae, candidate_sae, feature_id)
        selection = {
            "status": "selected",
            "concept": CONCEPT,
            "feature_id": feature_id,
            "view": str(row["view"]),
            "expected_sign": int(math.copysign(1, float(row["discovery_effect"]))),
            "discovery_effect": float(row["discovery_effect"]),
            "discovery_support_contexts": int(row["discovery_support_contexts"]),
            "validation_effect": float(row["validation_effect"]),
            "validation_support_contexts": int(row["validation_support_contexts"]),
            **geometry,
        }

    output_dir.mkdir(parents=True)
    discovery_path = output_dir / "discovery_full_scores.parquet"
    candidates_path = output_dir / "validation_candidates.parquet"
    selection_path = output_dir / "selection.parquet"
    discovery_scores.write_parquet(discovery_path)
    candidates.write_parquet(candidates_path)
    pl.DataFrame([selection]).write_parquet(selection_path)
    result = {
        "issue": ISSUE,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_commit": os.environ.get("EXPERIMENT_COMMIT", ""),
        "dictionary": dictionary_name,
        "selection": selection,
        "protocol": {
            "search_space": (f"all {D_SAE} features x {len(VIEW_NAMES)} frozen views"),
            "feature_chunk": chunk_size,
            "discovery_shortlist": DISCOVERY_SHORTLIST,
            "minimum_support_contexts": MIN_SUPPORT_CONTEXTS,
            "validation": (
                "same discovery sign, then largest absolute validation effect"
            ),
            "test_data_read": False,
        },
        "inputs": {
            "discovery_manifest_sha256": sha256_file(discovery_dir / "manifest.json"),
            "validation_manifest_sha256": sha256_file(validation_dir / "manifest.json"),
            "reference_sae_weights_sha256": sha256_file(
                reference_sae / "sae_weights.safetensors"
            ),
            "candidate_sae_weights_sha256": sha256_file(
                candidate_sae / "sae_weights.safetensors"
            ),
        },
    }
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (discovery_path, candidates_path, selection_path)
    }
    manifest = {**result, "artifacts": artifacts}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def test_selection(
    *,
    test_dir: Path,
    selection_dir: Path,
    output_dir: Path,
    dictionary_name: str,
    bootstraps: int = BOOTSTRAPS,
) -> dict[str, Any]:
    """Read the held-out test split once for the frozen feature/view."""

    assert dictionary_name and bootstraps > 0 and not output_dir.exists()
    selection_manifest_path = selection_dir / "manifest.json"
    selection_manifest = json.loads(selection_manifest_path.read_text())
    assert selection_manifest["dictionary"] == dictionary_name
    selection = selection_manifest["selection"]
    assert selection["status"] == "selected"
    test = load_extraction(test_dir, expected_split="test")
    assert test[5]["dictionary"] == dictionary_name and test[5]["stage"] == "test"
    feature_id = int(selection["feature_id"])
    assert test[1].tolist() == [feature_id]
    deltas = deltas_for_columns(test[4], test[2], test[3], slice(0, 1))
    views = response_views(deltas["forward"], deltas["reverse_complement"])
    view_name = str(selection["view"])
    source_ids, effects = context_effects(test[0], views[view_name], concept=CONCEPT)
    values = effects[:, 0]
    mean, ci_low, ci_high = bootstrap_mean(
        values, seed=ISSUE * 10_000 + 99, samples=bootstraps
    )
    expected_sign = int(selection["expected_sign"])
    replicated = ci_low > 0 if expected_sign > 0 else ci_high < 0
    spatial_kind = "local_peak" if "local_peak" in view_name else "focal"
    component_views = (
        ("fwd_local_peak", "rc_local_peak")
        if spatial_kind == "local_peak"
        else ("fwd_focal", "rc_focal")
    )
    components: dict[str, dict[str, float]] = {}
    component_values: dict[str, np.ndarray] = {}
    for component_index, component_view in enumerate(component_views):
        component_sources, component_effects = context_effects(
            test[0], views[component_view], concept=CONCEPT
        )
        assert np.array_equal(source_ids, component_sources)
        component_values[component_view] = component_effects[:, 0]
        comp_mean, comp_low, comp_high = bootstrap_mean(
            component_effects[:, 0],
            seed=ISSUE * 100_000 + 990 + component_index,
            samples=bootstraps,
        )
        components[component_view] = {
            "mean": comp_mean,
            "ci_low": comp_low,
            "ci_high": comp_high,
        }

    output_dir.mkdir(parents=True)
    contexts_path = output_dir / "test_context_effects.parquet"
    profile_path = output_dir / "test_codon_profile.parquet"
    pl.DataFrame(
        {
            "source_panel_row": source_ids,
            "effect": values,
            "fwd_effect": component_values[component_views[0]],
            "rc_effect": component_values[component_views[1]],
        }
    ).write_parquet(contexts_path)
    scalar = views[view_name][:, 0]
    (
        test[0]
        .with_columns(pl.Series("score", scalar))
        .group_by(["expected_consequence", "alternate_codon"])
        .agg(pl.col("score").mean().alias("mean_score"), pl.len().alias("rows"))
        .sort(["expected_consequence", "alternate_codon"])
        .write_parquet(profile_path)
    )
    result = {
        "issue": ISSUE,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_commit": os.environ.get("EXPERIMENT_COMMIT", ""),
        "dictionary": dictionary_name,
        "selection": selection,
        "test": {
            "effect": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "replicated": replicated,
            "contexts": len(values),
            "bootstrap_samples": bootstraps,
            "fwd_component": components[component_views[0]],
            "rc_component": components[component_views[1]],
        },
        "inputs": {
            "selection_manifest_sha256": sha256_file(selection_manifest_path),
            "test_extraction_manifest_sha256": sha256_file(test_dir / "manifest.json"),
        },
    }
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (contexts_path, profile_path)
    }
    manifest = {**result, "artifacts": artifacts}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--panel", type=Path, required=True)
    extract_parser.add_argument("--panel-manifest", type=Path, required=True)
    extract_parser.add_argument("--sae", type=Path, required=True)
    extract_parser.add_argument("--output-dir", type=Path, required=True)
    extract_parser.add_argument("--dictionary-name", required=True)
    extract_parser.add_argument("--source-split", required=True)
    extract_parser.add_argument("--stage", choices=("search", "test"), required=True)
    extract_parser.add_argument("--feature-id", type=int, action="append")
    extract_parser.add_argument("--selection-manifest", type=Path)
    extract_parser.add_argument("--batch-size", type=int, default=8)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--discovery-dir", type=Path, required=True)
    select_parser.add_argument("--validation-dir", type=Path, required=True)
    select_parser.add_argument("--reference-sae", type=Path, required=True)
    select_parser.add_argument("--candidate-sae", type=Path, required=True)
    select_parser.add_argument("--output-dir", type=Path, required=True)
    select_parser.add_argument("--dictionary-name", required=True)
    select_parser.add_argument("--chunk-size", type=int, default=FEATURE_CHUNK)

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--test-dir", type=Path, required=True)
    test_parser.add_argument("--selection-dir", type=Path, required=True)
    test_parser.add_argument("--output-dir", type=Path, required=True)
    test_parser.add_argument("--dictionary-name", required=True)
    test_parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "extract":
        if args.selection_manifest is not None:
            selection = json.loads(args.selection_manifest.read_text())["selection"]
            assert selection["status"] == "selected"
            feature_ids = [int(selection["feature_id"])]
        else:
            feature_ids = args.feature_id or list(range(D_SAE))
        result = extract(
            panel_path=args.panel,
            panel_manifest_path=args.panel_manifest,
            sae_path=args.sae,
            output_dir=args.output_dir,
            dictionary_name=args.dictionary_name,
            source_split=args.source_split,
            stage=args.stage,
            feature_ids=feature_ids,
            batch_size=args.batch_size,
        )
    elif args.command == "select":
        result = select(
            discovery_dir=args.discovery_dir,
            validation_dir=args.validation_dir,
            reference_sae=args.reference_sae,
            candidate_sae=args.candidate_sae,
            output_dir=args.output_dir,
            dictionary_name=args.dictionary_name,
            chunk_size=args.chunk_size,
        )
    else:
        result = test_selection(
            test_dir=args.test_dir,
            selection_dir=args.selection_dir,
            output_dir=args.output_dir,
            dictionary_name=args.dictionary_name,
            bootstraps=args.bootstraps,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
