"""Analyze signed SAE deltas for the dsQTL-positive direction pilot."""

from __future__ import annotations

import argparse
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
from scipy import stats

from analyze import (
    ALPHAGENOME_CANDIDATES,
    ARMS,
    FDR_THRESHOLD,
    direction_associations,
    load_delta,
    verify_artifacts,
)
from build_panel import EXPECTED_POSITIVES, SCOPE_DSQTL_DIRECTION_PILOT
from extract_focal import (
    ISSUE,
    ORIENTATIONS,
    assert_commit,
    sha256_file,
    write_json,
)

DATASET = "dsqtl"
TOP_K = 25
BASELINE_DIRECTION_SCORERS = {
    "chrombpnet_atac_logfc": "chrombpnet_atac_logfc",
    "chrombpnet_dnase_logfc": "chrombpnet_dnase_logfc",
    "enformer_dnase_local_logfc": "enformer_dnase_local_logfc",
}


def baseline_direction_correlations(panel: pl.DataFrame) -> pl.DataFrame:
    effect = panel["effect"].to_numpy()
    assert len(effect) == EXPECTED_POSITIVES[DATASET]
    assert np.isfinite(effect).all() and np.std(effect) > 0
    rows: list[dict[str, Any]] = []
    for scorer, column in BASELINE_DIRECTION_SCORERS.items():
        score = panel[column].to_numpy()
        assert np.isfinite(score).all() and np.std(score) > 0
        pearson = stats.pearsonr(score, effect)
        spearman = stats.spearmanr(score, effect)
        rows.append(
            {
                "dataset": DATASET,
                "scorer": scorer,
                "n": len(effect),
                "pearson": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            }
        )
    return pl.DataFrame(rows)


def top_direction_hits(frame: pl.DataFrame, *, top_k: int = TOP_K) -> pl.DataFrame:
    assert frame["outcome"].unique().to_list() == ["direction"]
    outputs: list[pl.DataFrame] = []
    for metric in ("pearson", "spearman"):
        outputs.append(
            frame.sort(pl.col(metric).abs(), descending=True)
            .head(top_k)
            .with_row_index("rank", offset=1)
            .with_columns(pl.lit(metric).alias("ranking_metric"))
        )
    return pl.concat(outputs, how="vertical")


def analyze_direction_pilot(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    extraction_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert panel_path.is_file() and panel_manifest_path.is_file()
    assert extraction_root.is_dir() and not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    started = time.monotonic()

    panel_manifest = json.loads(panel_manifest_path.read_text())
    extraction_manifest = json.loads((extraction_root / "manifest.json").read_text())
    assert panel_manifest["scope"] == SCOPE_DSQTL_DIRECTION_PILOT
    assert extraction_manifest["panel"]["scope"] == SCOPE_DSQTL_DIRECTION_PILOT
    assert panel_manifest["panel"]["sha256"] == sha256_file(panel_path)
    assert extraction_manifest["panel"]["sha256"] == sha256_file(panel_path)
    assert extraction_manifest["panel"]["rows"] == EXPECTED_POSITIVES[DATASET]
    verify_artifacts(extraction_root, extraction_manifest)

    panel = pl.read_parquet(panel_path)
    assert panel.height == EXPECTED_POSITIVES[DATASET]
    assert panel["dataset"].unique().to_list() == [DATASET]
    assert panel["label"].all()
    effect = panel["effect"].to_numpy()
    assert np.isfinite(effect).all()

    output_dir.mkdir(parents=True)
    family_dir = output_dir / "families"
    family_dir.mkdir()
    artifacts: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    candidate_frames: list[pl.DataFrame] = []
    top_frames: list[pl.DataFrame] = []
    all_features = np.arange(extraction_manifest["saes"][ARMS[0]]["d_sae"])

    for arm in ARMS:
        assert extraction_manifest["saes"][arm]["d_sae"] == len(all_features)
        for orientation in ORIENTATIONS:
            relative = f"{arm}/sae_focal_{orientation}.parquet"
            signed = load_delta(extraction_root / relative, rows=panel.height)
            direction = direction_associations(
                signed,
                effect,
                all_features,
                dataset=DATASET,
                arm=arm,
                orientation=orientation,
            )
            name = f"{DATASET}__{arm}__{orientation}__direction.parquet"
            path = family_dir / name
            direction.write_parquet(path, compression="zstd")
            artifacts[str(path.relative_to(output_dir))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": direction.height,
            }
            summary = {
                "dataset": DATASET,
                "arm": arm,
                "report_block": int(arm.removeprefix("block").split("-")[0]),
                "orientation": orientation,
                "features_tested": direction.height,
                "pearson_q05": direction.filter(
                    pl.col("pearson_q") < FDR_THRESHOLD
                ).height,
                "spearman_q05": direction.filter(
                    pl.col("spearman_q") < FDR_THRESHOLD
                ).height,
                "concordant_q05": direction.filter(
                    (pl.col("pearson_q") < FDR_THRESHOLD)
                    & (pl.col("spearman_q") < FDR_THRESHOLD)
                ).height,
                "max_abs_pearson": direction["pearson"].abs().max(),
                "max_abs_spearman": direction["spearman"].abs().max(),
            }
            summaries.append(summary)
            top_frames.append(top_direction_hits(direction))
            candidates = ALPHAGENOME_CANDIDATES.get(arm, ())
            if candidates:
                candidate_frames.append(
                    direction.filter(pl.col("feature_id").is_in(candidates))
                )
            print(json.dumps({"stage": "family_complete", **summary}), flush=True)

    summary_frame = pl.DataFrame(summaries, infer_schema_length=None)
    summary_path = output_dir / "family_summary.parquet"
    summary_frame.write_parquet(summary_path, compression="zstd")
    artifacts[summary_path.name] = {
        "bytes": summary_path.stat().st_size,
        "sha256": sha256_file(summary_path),
        "rows": summary_frame.height,
    }
    top = pl.concat(top_frames, how="vertical")
    top_path = output_dir / "top_direction_features.parquet"
    top.write_parquet(top_path, compression="zstd")
    artifacts[top_path.name] = {
        "bytes": top_path.stat().st_size,
        "sha256": sha256_file(top_path),
        "rows": top.height,
    }
    candidates = pl.concat(candidate_frames, how="vertical")
    candidate_path = output_dir / "alphagenome_candidate_overlap.parquet"
    candidates.write_parquet(candidate_path, compression="zstd")
    artifacts[candidate_path.name] = {
        "bytes": candidate_path.stat().st_size,
        "sha256": sha256_file(candidate_path),
        "rows": candidates.height,
    }
    baselines = baseline_direction_correlations(panel)
    baseline_path = output_dir / "official_baseline_direction_sanity.parquet"
    baselines.write_parquet(baseline_path, compression="zstd")
    artifacts[baseline_path.name] = {
        "bytes": baseline_path.stat().st_size,
        "sha256": sha256_file(baseline_path),
        "rows": baselines.height,
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
            "scope": SCOPE_DSQTL_DIRECTION_PILOT,
            "panel_sha256": sha256_file(panel_path),
            "extraction_manifest_sha256": sha256_file(
                extraction_root / "manifest.json"
            ),
            "extraction_run_id": extraction_manifest["run_id"],
            "rows": panel.height,
        },
        "protocol": {
            "layers_reported": [1, 10, 19],
            "sae_training_activations": 25_000_200,
            "orientations": list(ORIENTATIONS),
            "response": "activation_alt - activation_ref",
            "population": "all 559 official dsQTL causal positives",
            "metrics": ["Pearson", "Spearman"],
            "minimum_nonzero_support": 10,
            "bh_family": "layer x orientation x statistic",
            "fdr_threshold": FDR_THRESHOLD,
            "alphagenome_candidates_fixed_before_qtl_outcomes": (
                ALPHAGENOME_CANDIDATES
            ),
        },
        "summaries": summaries,
        "baseline_sanity": baselines.to_dicts(),
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
    args = parser.parse_args()
    print(
        json.dumps(
            analyze_direction_pilot(
                panel_path=args.panel,
                panel_manifest_path=args.panel_manifest,
                extraction_root=args.extraction_root,
                output_dir=args.output_dir,
            )["artifacts"],
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
