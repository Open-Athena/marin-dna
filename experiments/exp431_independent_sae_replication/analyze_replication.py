"""Select and test semantic feature transfer across one independent SAE."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ISSUE = 431
SPLITS = ("discovery", "validation", "test")
SPATIAL_RADIUS = 15
MIN_SUPPORT_CONTEXTS = 16
DISCOVERY_SHORTLIST = 16
BOOTSTRAPS = 2_000
CONCEPTS = (
    "splice_acceptor",
    "splice_donor",
    "stop_creation",
    "synonymous_degeneracy",
)
CLASS_BY_CONCEPT = {
    "splice_acceptor": "splice_acceptor_variant",
    "splice_donor": "splice_donor_5th_base_variant",
    "stop_creation": "stop_gained",
    "synonymous_degeneracy": "synonymous_variant",
}
QUERY_BY_CONCEPT = {
    "splice_acceptor": ("splice_acceptor",),
    "splice_donor": ("splice_donor",),
    "stop_creation": ("stop_creation_positive", "stop_creation_negative"),
    "synonymous_degeneracy": ("synonymous_degeneracy",),
}
VIEW_NAMES = (
    "fwd_focal",
    "rc_focal",
    "mean_focal",
    "maxabs_focal",
    "fwd_local_peak",
    "rc_local_peak",
    "mean_local_peak",
    "maxabs_local_peak",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def signed_peak(values: np.ndarray, *, axis: int = 1) -> np.ndarray:
    """Select the largest absolute value while preserving its sign."""

    assert values.ndim == 3 and axis == 1
    indices = np.abs(values).argmax(axis=axis)
    selected = np.take_along_axis(values, indices[:, None, :], axis=axis)[:, 0, :]
    assert selected.shape == (values.shape[0], values.shape[2])
    return selected


def response_views(
    delta_fwd: np.ndarray, delta_rc: np.ndarray
) -> dict[str, np.ndarray]:
    """Construct frozen focal and local signed views after genomic RC alignment."""

    assert delta_fwd.shape == delta_rc.shape
    assert delta_fwd.ndim == 3 and delta_fwd.shape[1] == 2 * SPATIAL_RADIUS + 1
    assert np.isfinite(delta_fwd).all() and np.isfinite(delta_rc).all()
    rc = delta_rc[:, ::-1, :]
    mean_profile = 0.5 * (delta_fwd + rc)
    focal = SPATIAL_RADIUS
    fwd_focal = delta_fwd[:, focal, :]
    rc_focal = rc[:, focal, :]
    focal_stack = np.stack((fwd_focal, rc_focal), axis=1)
    local_stack = np.concatenate((delta_fwd, rc), axis=1)
    views = {
        "fwd_focal": fwd_focal,
        "rc_focal": rc_focal,
        "mean_focal": mean_profile[:, focal, :],
        "maxabs_focal": signed_peak(focal_stack),
        "fwd_local_peak": signed_peak(delta_fwd),
        "rc_local_peak": signed_peak(rc),
        "mean_local_peak": signed_peak(mean_profile),
        "maxabs_local_peak": signed_peak(local_stack),
    }
    assert tuple(views) == VIEW_NAMES
    assert all(value.shape == fwd_focal.shape for value in views.values())
    return views


def target_mask(frame: pl.DataFrame, concept: str) -> np.ndarray:
    assert concept in CONCEPTS
    if concept == "splice_acceptor":
        return frame["relative_position"].is_in((-1, 0)).to_numpy()
    if concept == "splice_donor":
        return frame["relative_position"].is_in((-4, -3)).to_numpy()
    if concept == "stop_creation":
        return (frame["expected_consequence"] == "stop_gained").to_numpy()
    return (frame["expected_consequence"] == "synonymous_variant").to_numpy()


def context_effects(
    panel: pl.DataFrame,
    scores: np.ndarray,
    *,
    concept: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute target-minus-other effects with equal weight per source context."""

    assert scores.ndim == 2 and scores.shape[0] == panel.height
    class_name = CLASS_BY_CONCEPT[concept]
    rows = panel.filter(pl.col("class") == class_name)
    indices = rows["perturbation_row"].to_numpy()
    assert np.array_equal(np.sort(indices), np.unique(indices))
    class_scores = scores[indices]
    is_target = target_mask(rows, concept)
    source_ids = rows["source_panel_row"].to_numpy()
    effects: list[np.ndarray] = []
    ordered_sources = np.unique(source_ids)
    for source_id in ordered_sources:
        source = source_ids == source_id
        target = source & is_target
        other = source & ~is_target
        assert target.any() and other.any()
        effects.append(
            class_scores[target].mean(axis=0) - class_scores[other].mean(axis=0)
        )
    result = np.stack(effects)
    assert result.shape == (len(ordered_sources), scores.shape[1])
    assert np.isfinite(result).all()
    return ordered_sources, result


def bootstrap_mean(
    values: np.ndarray,
    *,
    seed: int,
    samples: int = BOOTSTRAPS,
) -> tuple[float, float, float]:
    assert values.ndim == 1 and len(values) >= 2 and samples > 0
    assert np.isfinite(values).all()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def load_split(
    *,
    split: str,
    panel_root: Path,
    extraction_root: Path,
) -> tuple[pl.DataFrame, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    assert split in SPLITS
    panel_path = panel_root / split / "perturbation_panel.parquet"
    panel_manifest_path = panel_root / split / "manifest.json"
    extraction_dir = extraction_root / split
    extraction_manifest_path = extraction_dir / "manifest.json"
    for path in (panel_path, panel_manifest_path, extraction_manifest_path):
        assert path.is_file(), path
    panel_manifest = json.loads(panel_manifest_path.read_text())
    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    assert panel_manifest["artifacts"][panel_path.name]["sha256"] == sha256_file(
        panel_path
    )
    assert extraction_manifest["source_split"] == split
    assert extraction_manifest["design"]["panel_sha256"] == sha256_file(panel_path)
    panel = pl.read_parquet(panel_path)
    feature_ids = np.load(extraction_dir / "feature_ids.npy")
    reference_indices = np.load(extraction_dir / "reference_state_indices.npy")
    alternate_indices = np.load(extraction_dir / "alternate_state_indices.npy")
    assert feature_ids.ndim == 1 and len(np.unique(feature_ids)) == len(feature_ids)
    arrays: dict[str, np.ndarray] = {}
    for orientation in ("forward", "reverse_complement"):
        path = extraction_dir / f"state_activations_{orientation}.npy"
        assert extraction_manifest["artifacts"][path.name]["sha256"] == sha256_file(
            path
        )
        states = np.load(path, mmap_mode="r")
        assert states.shape[1:] == (2 * SPATIAL_RADIUS + 1, len(feature_ids))
        arrays[orientation] = np.asarray(
            states[alternate_indices] - states[reference_indices]
        )
    views = response_views(arrays["forward"], arrays["reverse_complement"])
    return panel, feature_ids, views, extraction_manifest


def candidate_feature_ids(candidate_table: pl.DataFrame, concept: str) -> set[int]:
    queries = QUERY_BY_CONCEPT[concept]
    return set(
        candidate_table.filter(pl.col("concept").is_in(queries))[
            "candidate_feature_id"
        ].to_list()
    )


def decoder_metadata(
    candidate_table: pl.DataFrame, concept: str, feature_id: int
) -> dict[str, Any]:
    rows = candidate_table.filter(
        pl.col("concept").is_in(QUERY_BY_CONCEPT[concept])
        & (pl.col("candidate_feature_id") == feature_id)
    ).sort("decoder_cosine", descending=True)
    assert rows.height >= 1
    row = rows.row(0, named=True)
    return {
        "reference_query": row["concept"],
        "reference_feature_id": int(row["reference_feature_id"]),
        "decoder_rank": int(row["candidate_rank"]),
        "decoder_cosine": float(row["decoder_cosine"]),
        "mutual_nearest": bool(row["mutual_nearest"]),
    }


def rank_discovery(
    candidate_table: pl.DataFrame,
    feature_ids: np.ndarray,
    views: dict[str, np.ndarray],
    panel: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]]:
    feature_column = {
        int(feature_id): index for index, feature_id in enumerate(feature_ids)
    }
    effects_by_concept_view: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for concept in CONCEPTS:
        allowed = candidate_feature_ids(candidate_table, concept)
        assert allowed
        for view_name, scores in views.items():
            source_ids, effects = context_effects(panel, scores, concept=concept)
            effects_by_concept_view[(concept, view_name)] = (source_ids, effects)
            for feature_id in allowed:
                column = feature_column[feature_id]
                values = effects[:, column]
                support = int((np.abs(values) > 1e-8).sum())
                if support < MIN_SUPPORT_CONTEXTS:
                    continue
                rows.append(
                    {
                        "concept": concept,
                        "feature_id": feature_id,
                        "view": view_name,
                        "discovery_effect": float(values.mean()),
                        "discovery_abs_effect": float(abs(values.mean())),
                        "discovery_support_contexts": support,
                    }
                )
    ranked = pl.DataFrame(rows).sort(
        ["concept", "discovery_abs_effect"], descending=[False, True]
    )
    shortlist = ranked.group_by("concept", maintain_order=True).head(
        DISCOVERY_SHORTLIST
    )
    return shortlist, effects_by_concept_view


def analyze(
    *,
    panel_root: Path,
    extraction_root: Path,
    output_dir: Path,
    dictionary_name: str,
    bootstraps: int = BOOTSTRAPS,
) -> dict[str, Any]:
    assert dictionary_name and bootstraps > 0 and not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    loaded = {
        split: load_split(
            split=split, panel_root=panel_root, extraction_root=extraction_root
        )
        for split in SPLITS
    }
    feature_ids = loaded["discovery"][1]
    assert all(np.array_equal(feature_ids, loaded[split][1]) for split in SPLITS)
    candidate_table = pl.read_parquet(
        extraction_root / "discovery" / "decoder_candidates.parquet"
    )
    assert set(candidate_table["dictionary"].unique()) == {dictionary_name}
    shortlist, _ = rank_discovery(
        candidate_table,
        feature_ids,
        loaded["discovery"][2],
        loaded["discovery"][0],
    )
    feature_column = {
        int(feature_id): index for index, feature_id in enumerate(feature_ids)
    }
    selection_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    profile_frames: list[pl.DataFrame] = []
    validation_rows: list[dict[str, Any]] = []
    for concept_index, concept in enumerate(CONCEPTS):
        candidates = shortlist.filter(pl.col("concept") == concept)
        eligible: list[dict[str, Any]] = []
        for candidate in candidates.iter_rows(named=True):
            feature_id = int(candidate["feature_id"])
            view_name = str(candidate["view"])
            _, effects = context_effects(
                loaded["validation"][0],
                loaded["validation"][2][view_name],
                concept=concept,
            )
            values = effects[:, feature_column[feature_id]]
            validation_effect = float(values.mean())
            support = int((np.abs(values) > 1e-8).sum())
            same_sign = validation_effect * float(candidate["discovery_effect"]) > 0
            row = {
                **candidate,
                "validation_effect": validation_effect,
                "validation_abs_effect": abs(validation_effect),
                "validation_support_contexts": support,
                "validation_same_sign": same_sign,
            }
            validation_rows.append(row)
            if same_sign and support >= MIN_SUPPORT_CONTEXTS:
                eligible.append(row)
        if not eligible:
            selection_rows.append(
                {
                    "dictionary": dictionary_name,
                    "concept": concept,
                    "status": "no_validation_confirmed_candidate",
                    "replicated": False,
                }
            )
            continue
        selected = max(eligible, key=lambda row: row["validation_abs_effect"])
        feature_id = int(selected["feature_id"])
        view_name = str(selected["view"])
        source_ids, test_effects = context_effects(
            loaded["test"][0], loaded["test"][2][view_name], concept=concept
        )
        values = test_effects[:, feature_column[feature_id]]
        mean, ci_low, ci_high = bootstrap_mean(
            values, seed=ISSUE * 10_000 + concept_index, samples=bootstraps
        )
        expected_sign = int(math.copysign(1, float(selected["discovery_effect"])))
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
                loaded["test"][0],
                loaded["test"][2][component_view],
                concept=concept,
            )
            assert np.array_equal(source_ids, component_sources)
            values_component = component_effects[:, feature_column[feature_id]]
            component_values[component_view] = values_component
            comp_mean, comp_low, comp_high = bootstrap_mean(
                values_component,
                seed=ISSUE * 100_000 + concept_index * 10 + component_index,
                samples=bootstraps,
            )
            components[component_view] = {
                "mean": comp_mean,
                "ci_low": comp_low,
                "ci_high": comp_high,
            }
        decoder = decoder_metadata(candidate_table, concept, feature_id)
        selection_rows.append(
            {
                "dictionary": dictionary_name,
                "concept": concept,
                "status": "tested",
                "feature_id": feature_id,
                "view": view_name,
                "expected_sign": expected_sign,
                "discovery_effect": float(selected["discovery_effect"]),
                "validation_effect": float(selected["validation_effect"]),
                "test_effect": mean,
                "test_ci_low": ci_low,
                "test_ci_high": ci_high,
                "replicated": replicated,
                **decoder,
                "fwd_component": components[component_views[0]],
                "rc_component": components[component_views[1]],
            }
        )
        for row_index, source_id in enumerate(source_ids):
            context_rows.append(
                {
                    "dictionary": dictionary_name,
                    "concept": concept,
                    "feature_id": feature_id,
                    "view": view_name,
                    "source_panel_row": int(source_id),
                    "effect": float(values[row_index]),
                    "fwd_effect": float(
                        component_values[component_views[0]][row_index]
                    ),
                    "rc_effect": float(component_values[component_views[1]][row_index]),
                }
            )
        test_panel = loaded["test"][0]
        class_panel = test_panel.filter(pl.col("class") == CLASS_BY_CONCEPT[concept])
        row_indices = class_panel["perturbation_row"].to_numpy()
        scalar = loaded["test"][2][view_name][row_indices, feature_column[feature_id]]
        profile = class_panel.with_columns(pl.Series("score", scalar))
        grouping = (
            ["relative_position"]
            if concept.startswith("splice_")
            else ["expected_consequence", "alternate_codon"]
        )
        profile_frames.append(
            profile.group_by(grouping)
            .agg(pl.col("score").mean().alias("mean_score"), pl.len().alias("rows"))
            .with_columns(
                pl.lit(dictionary_name).alias("dictionary"),
                pl.lit(concept).alias("concept"),
                pl.lit(feature_id).alias("feature_id"),
                pl.lit(view_name).alias("view"),
            )
        )

    selections = pl.DataFrame(selection_rows)
    assert selections.height == len(CONCEPTS)
    tested = selections.filter(pl.col("status") == "tested")
    output_dir.mkdir(parents=True)
    selections_path = output_dir / "selected_features.parquet"
    shortlist_path = output_dir / "discovery_shortlist.parquet"
    validation_path = output_dir / "validation_candidates.parquet"
    contexts_path = output_dir / "test_context_effects.parquet"
    profiles_path = output_dir / "test_profiles.parquet"
    selections.write_parquet(selections_path)
    shortlist.write_parquet(shortlist_path)
    pl.DataFrame(validation_rows).write_parquet(validation_path)
    pl.DataFrame(context_rows).write_parquet(contexts_path)
    if profile_frames:
        pl.concat(profile_frames, how="diagonal_relaxed").write_parquet(profiles_path)
    else:
        pl.DataFrame().write_parquet(profiles_path)
    results = {
        "issue": ISSUE,
        "experiment_commit": experiment_commit,
        "dictionary": dictionary_name,
        "protocol": {
            "candidate_search": "top-32 positive decoder-cosine neighborhood per frozen #429 query",
            "discovery_shortlist_per_concept": DISCOVERY_SHORTLIST,
            "minimum_support_contexts": MIN_SUPPORT_CONTEXTS,
            "validation": "same discovery sign, then largest absolute validation effect",
            "test": "single frozen feature/view; equal-weight context mean; context bootstrap",
            "bootstrap_samples": bootstraps,
        },
        "concepts_tested": tested.height,
        "concepts_replicated": int(tested["replicated"].sum()) if tested.height else 0,
        "results": selections.to_dicts(),
        "inputs": {
            split: {
                "panel_manifest_sha256": sha256_file(
                    panel_root / split / "manifest.json"
                ),
                "extraction_manifest_sha256": sha256_file(
                    extraction_root / split / "manifest.json"
                ),
            }
            for split in SPLITS
        },
    }
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (
            selections_path,
            shortlist_path,
            validation_path,
            contexts_path,
            profiles_path,
        )
    }
    manifest = {**results, "artifacts": artifacts}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dictionary-name", required=True)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    args = parser.parse_args()
    result = analyze(
        panel_root=args.panel_root,
        extraction_root=args.extraction_root,
        output_dir=args.output_dir,
        dictionary_name=args.dictionary_name,
        bootstraps=args.bootstraps,
    )
    print(json.dumps(result["results"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
