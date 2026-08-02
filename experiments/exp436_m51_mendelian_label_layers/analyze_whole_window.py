"""Run the frozen whole-window mean Mendelian SAE association scan."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy

from analyze_focal import (
    DESCRIPTIVE_ONLY_SUBSETS,
    EXPECTED_INFERENTIAL_SUBSETS,
    FDR_THRESHOLD,
    MIN_CLASS_SIZE,
    MIN_NONZERO_SUPPORT,
    PRIMARY_RESPONSES,
    RESPONSES,
    analyze_target,
    correct_family,
    family_summary,
    parse_arm,
    response_matrix,
    target_definitions,
)
from extract_focal import (
    BUDGETS,
    D_SAE,
    EXPECTED_GROUPS,
    EXPECTED_ROWS,
    ISSUE,
    ORIENTATIONS,
    sha256_file,
    write_json,
)
from extract_whole_window import POOLING, matrix_relative
from train import assert_commit


def verify_extraction(root: Path, manifest: dict[str, Any]) -> None:
    assert manifest["protocol"]["pooling"] == POOLING
    assert manifest["protocol"]["matrix_dtype"] == "float32"
    assert manifest["protocol"]["bos_excluded"] is True
    assert manifest["protocol"]["quantized"] is False
    for relative, expected in manifest["artifacts"].items():
        path = root / relative
        assert path.is_file(), path
        assert path.stat().st_size == expected["bytes"], path
        assert sha256_file(path) == expected["sha256"], path


def load_pair(root: Path, arm: str, orientation: str) -> tuple[np.memmap, np.memmap]:
    ref = np.load(root / matrix_relative(arm, orientation, "ref"), mmap_mode="r")
    alt = np.load(root / matrix_relative(arm, orientation, "alt"), mmap_mode="r")
    assert ref.shape == alt.shape == (EXPECTED_ROWS, D_SAE)
    assert ref.dtype == alt.dtype == np.float32
    return ref, alt


def analyze(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    extraction_root: Path,
    output_dir: Path,
    ap_chunk_size: int,
) -> dict[str, Any]:
    assert not output_dir.exists() and ap_chunk_size > 0
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    started = time.monotonic()

    panel_manifest = json.loads(panel_manifest_path.read_text())
    extraction_manifest = json.loads((extraction_root / "manifest.json").read_text())
    assert panel_manifest["panel_sha256"] == sha256_file(panel_path)
    assert extraction_manifest["panel"]["sha256"] == panel_manifest["panel_sha256"]
    assert extraction_manifest["panel"]["rows"] == EXPECTED_ROWS
    assert extraction_manifest["panel"]["match_groups"] == EXPECTED_GROUPS
    verify_extraction(extraction_root, extraction_manifest)

    panel = pl.read_parquet(panel_path).with_row_index("panel_row")
    assert panel["panel_row"].to_list() == list(range(EXPECTED_ROWS))
    targets = target_definitions(panel)
    output_dir.mkdir(parents=True)
    (output_dir / "families").mkdir()
    summaries: dict[str, Any] = {}
    top_frames: list[pl.DataFrame] = []
    artifacts: dict[str, Any] = {}

    expected_arms = {
        f"block{block:02d}-{budget // 1_000_000}m"
        for block in (1, 10, 19)
        for budget in BUDGETS
    }
    arms = sorted(extraction_manifest["outputs"])
    assert set(arms) == expected_arms
    for arm in arms:
        block, budget = parse_arm(arm)
        summaries[arm] = {}
        for orientation in ORIENTATIONS:
            print(
                json.dumps(
                    {
                        "stage": "load_whole_window_pair",
                        "arm": arm,
                        "orientation": orientation,
                    }
                ),
                flush=True,
            )
            ref, alt = load_pair(extraction_root, arm, orientation)
            delta: np.ndarray | None = None
            summaries[arm][orientation] = {}
            for response_name in RESPONSES:
                response, delta = response_matrix(
                    response_name, ref=ref, alt=alt, delta=delta
                )
                target_frames = [
                    analyze_target(
                        response,
                        target,
                        arm=arm,
                        block=block,
                        budget=budget,
                        orientation=orientation,
                        response_name=response_name,
                        ap_chunk_size=ap_chunk_size,
                        pooling=POOLING,
                    )
                    for target in targets
                ]
                family = correct_family(target_frames)
                relative = (
                    Path("families") / arm / orientation / f"{response_name}.parquet"
                )
                path = output_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                family.write_parquet(path, compression="zstd")
                artifacts[str(relative)] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                summaries[arm][orientation][response_name] = family_summary(family)
                top_frames.append(
                    family.sort(["minimum_q", "target", "feature_id"]).head(25)
                )
                print(
                    json.dumps(
                        {
                            "stage": "family_complete",
                            "arm": arm,
                            "orientation": orientation,
                            "response": response_name,
                            **summaries[arm][orientation][response_name],
                        }
                    ),
                    flush=True,
                )
                del target_frames, family
                if response_name == "abs_delta":
                    del response
                gc.collect()
            del ref, alt, delta
            gc.collect()

    top_hits = pl.concat(top_frames, how="vertical").sort(
        ["minimum_q", "arm", "orientation", "response", "target", "feature_id"]
    )
    top_hits_path = output_dir / "top_hits.parquet"
    top_hits.write_parquet(top_hits_path, compression="zstd")
    artifacts["top_hits.parquet"] = {
        "bytes": top_hits_path.stat().st_size,
        "sha256": sha256_file(top_hits_path),
    }

    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": os.environ.get("RUN_ID", ""),
        "experiment_commit": experiment_commit,
        "elapsed_seconds": time.monotonic() - started,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "scipy": scipy.__version__,
        "input": {
            "extraction_run_id": extraction_manifest["run_id"],
            "extraction_experiment_commit": extraction_manifest["experiment_commit"],
            "extraction_manifest_sha256": sha256_file(
                extraction_root / "manifest.json"
            ),
            "panel_sha256": panel_manifest["panel_sha256"],
            "rows": EXPECTED_ROWS,
            "match_groups": EXPECTED_GROUPS,
        },
        "protocol": {
            "layers_reported": [1, 10, 19],
            "budgets": list(BUDGETS),
            "orientations": list(ORIENTATIONS),
            "pooling": POOLING,
            "responses_in_execution_order": list(RESPONSES),
            "primary_responses": sorted(PRIMARY_RESPONSES),
            "minimum_nonzero_support": MIN_NONZERO_SUPPORT,
            "minimum_class_size": MIN_CLASS_SIZE,
            "inferential_subsets": sorted(EXPECTED_INFERENTIAL_SUBSETS),
            "descriptive_only_subsets": sorted(DESCRIPTIVE_ONLY_SUBSETS),
            "tests": ["Welch t", "Mann-Whitney U"],
            "effect_sizes": [
                "standardized mean difference",
                "rank-biserial correlation",
            ],
            "descriptive_metric": "AUPRC in raw and sign-reversed direction",
            "bh_family": (
                "layer x budget x orientation x whole-window mean x response x "
                "statistic; all eligible overall and within-subset feature-target pairs"
            ),
            "fdr_threshold": FDR_THRESHOLD,
            "ap_chunk_size": ap_chunk_size,
            "uses_all_variants": True,
            "uses_chromosome_split": False,
            "uses_label_for_feature_support": False,
        },
        "summaries": summaries,
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
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ap-chunk-size", type=int, default=128)
    args = parser.parse_args()
    result = analyze(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        extraction_root=args.extraction_root,
        output_dir=args.output_dir,
        ap_chunk_size=args.ap_chunk_size,
    )
    print(json.dumps(result["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
