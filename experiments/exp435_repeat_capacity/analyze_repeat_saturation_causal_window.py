"""Post-result causal-window sensitivity for repeat-feature saturation."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from analyze_repeat_saturation import (
    ALPHA,
    build_context_effects,
    build_feature_summary,
    build_planned_tests,
    build_view_summary,
)
from common import ISSUE, assert_commit, sha256_file, write_json
from saturation_common import MIN_CONTEXTS, VIEW_KEYS

RUN_ID = "dna-exp435-repeat-saturation-causal-window-r1"
INPUT_RUN_ID = "dna-exp435-repeat-saturation-r1"
INPUT_ARCHIVE_MANIFEST_SHA256 = (
    "76fd74a79afaca4f9b988d65c74358d6a2b2442ab4cd65911a081168ac2e238f"
)


def verify_input_archive(root: Path) -> tuple[dict[str, Any], pl.DataFrame]:
    """Verify the exact full-window saturation archive and load responses."""

    manifest_path = root / "archive_manifest.json"
    assert manifest_path.is_file()
    assert sha256_file(manifest_path) == INPUT_ARCHIVE_MANIFEST_SHA256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["issue"] == ISSUE and manifest["run_id"] == INPUT_RUN_ID
    assert manifest["analysis_status"] == "post_hoc_repeat_motif_saturation"
    for relative, expected in manifest["artifacts"].items():
        path = root / relative
        assert path.is_file() and path.stat().st_size == expected["bytes"]
        assert sha256_file(path) == expected["sha256"]
    responses = pl.read_parquet(root / "extraction/mutation_responses.parquet")
    return manifest, responses


def select_causal_window(responses: pl.DataFrame) -> pl.DataFrame:
    """Keep focal/upstream model positions after verifying downstream invariance."""

    required = {"model_offset", "abs_delta", "orientation", "feature_id", "block"}
    assert required <= set(responses.columns)
    downstream = responses.filter(pl.col("model_offset") > 0)
    assert downstream.height > 0
    assert downstream["abs_delta"].max() == 0
    assert int((downstream["abs_delta"] > 0).sum()) == 0
    causal = responses.filter(pl.col("model_offset") <= 0)
    assert causal.height + downstream.height == responses.height
    assert causal["abs_delta"].min() >= 0
    return causal


def analyze(input_root: Path, output_dir: Path) -> dict[str, Any]:
    assert input_root.is_dir() and not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    assert os.environ.get("RUN_ID") == RUN_ID
    input_manifest, responses = verify_input_archive(input_root)
    causal = select_causal_window(responses)
    output_dir.mkdir(parents=True)

    effects = build_context_effects(causal)
    tests = build_planned_tests(effects)
    views = build_view_summary(effects, tests)
    features = build_feature_summary(views)
    tables = {
        "context_effects.parquet": effects,
        "planned_tests.parquet": tests,
        "view_summary.parquet": views,
        "feature_summary.parquet": features,
    }
    for name, frame in tables.items():
        frame.write_parquet(output_dir / name, compression="zstd")

    result: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": RUN_ID,
        "analysis_status": "post_result_causal_window_sensitivity",
        "experiment_commit": experiment_commit,
        "input": {
            "run_id": INPUT_RUN_ID,
            "archive_manifest_sha256": INPUT_ARCHIVE_MANIFEST_SHA256,
            "analysis_status": input_manifest["analysis_status"],
        },
        "rationale": (
            "the completed full-window intervention showed exact zero response "
            "for every model-downstream mutation under the causal mask"
        ),
        "protocol": {
            "filter": "model_offset <= 0",
            "downstream_rows_verified_zero": responses.filter(
                pl.col("model_offset") > 0
            ).height,
            "causal_rows": causal.height,
            "minimum_contexts": MIN_CONTEXTS,
            "alpha": ALPHA,
            "multiple_testing": (
                "unchanged from primary: BH separately within layer, test family, "
                "and test type across prespecified feature-by-orientation views"
            ),
        },
        "summary": {
            "views": len(VIEW_KEYS),
            "features": len(VIEW_KEYS) // 2,
            "motif_loss_supported_views": int(views["motif_loss_supported"].sum()),
            "motif_specificity_supported_views": int(
                views["motif_specificity_supported"].sum()
            ),
            "strand_stable_motif_loss_features": int(
                features["strand_stable_motif_loss"].sum()
            ),
            "strand_stable_motif_specificity_features": int(
                features["strand_stable_motif_specificity"].sum()
            ),
        },
    }
    result_path = output_dir / "results.json"
    write_json(result_path, result)
    results_md = output_dir / "RESULTS.md"
    results_md.write_text(
        "# Causal-window saturation sensitivity\n\n"
        "This post-result sensitivity retains only model offsets <= 0 after "
        "verifying exact zero response at every downstream offset. The original "
        "full-window analysis remains primary.\n\n"
        f"Summary: `{json.dumps(result['summary'], sort_keys=True)}`\n"
    )
    artifact_paths = [
        *(output_dir / name for name in tables),
        result_path,
        results_md,
    ]
    manifest = {
        **result,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artifact_paths
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input_root, args.output_dir)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
