"""Analyze the preregistered untouched-test transfer of feature 1662."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import average_precision_score

from common import ISSUE, sha256_file, write_json
from extract_focal import ORIENTATIONS
from transfer_common import (
    EXPECTED_ROWS,
    FEATURE_ID,
    bh_adjust,
    validate_test_panel,
)

TARGETS = ("missense_variant", "overall")


def standardized_mean_difference(positive: np.ndarray, negative: np.ndarray) -> float:
    pooled_variance = (
        (positive.size - 1) * positive.var(ddof=1)
        + (negative.size - 1) * negative.var(ddof=1)
    ) / (positive.size + negative.size - 2)
    assert pooled_variance > 0
    return float((positive.mean() - negative.mean()) / np.sqrt(pooled_variance))


def load_response(path: Path) -> np.ndarray:
    assert path.is_file()
    frame = pl.read_parquet(path)
    assert frame.height == EXPECTED_ROWS
    assert frame["panel_row"].to_list() == list(range(EXPECTED_ROWS))
    response = frame["delta"].abs().to_numpy()
    assert response.shape == (EXPECTED_ROWS,) and np.isfinite(response).all()
    return response


def association_row(
    labels: np.ndarray,
    response: np.ndarray,
    *,
    orientation: str,
    target: str,
) -> dict[str, float | int | str]:
    assert labels.dtype == bool and labels.shape == response.shape
    positive, negative = response[labels], response[~labels]
    assert positive.size >= 30 and negative.size >= 30
    welch = stats.ttest_ind(positive, negative, equal_var=False)
    mann = stats.mannwhitneyu(positive, negative, alternative="two-sided")
    return {
        "feature_id": FEATURE_ID,
        "target": target,
        "orientation": orientation,
        "n": labels.size,
        "n_positive": positive.size,
        "prevalence": float(labels.mean()),
        "auprc": float(average_precision_score(labels, response)),
        "positive_mean": float(positive.mean()),
        "negative_mean": float(negative.mean()),
        "welch_p": float(welch.pvalue),
        "mann_whitney_p": float(mann.pvalue),
        "standardized_mean_difference": standardized_mean_difference(
            positive, negative
        ),
        "rank_biserial": float(
            2 * mann.statistic / (positive.size * negative.size) - 1
        ),
        "nonzero_support": int(np.count_nonzero(response)),
    }


def analyze(
    panel_path: Path, extraction_root: Path, output_dir: Path
) -> dict[str, Any]:
    assert not output_dir.exists() and extraction_root.is_dir()
    panel_manifest = validate_test_panel(panel_path)
    panel = pl.read_parquet(panel_path)
    rows: list[dict[str, float | int | str]] = []
    for orientation in ORIENTATIONS:
        response = load_response(extraction_root / f"feature1662_{orientation}.parquet")
        for target in TARGETS:
            if target == "overall":
                mask = np.ones(EXPECTED_ROWS, dtype=bool)
            else:
                mask = (panel["subset"] == target).to_numpy()
            labels = panel["label"].to_numpy()[mask]
            rows.append(
                association_row(
                    labels,
                    response[mask],
                    orientation=orientation,
                    target=target,
                )
            )
    result_frame = pl.DataFrame(rows)
    assert result_frame.height == 4
    result_frame = result_frame.with_columns(
        pl.Series("welch_q", bh_adjust(result_frame["welch_p"].to_numpy())),
        pl.Series(
            "mann_whitney_q",
            bh_adjust(result_frame["mann_whitney_p"].to_numpy()),
        ),
    ).sort("target", "orientation")
    primary = result_frame.filter(pl.col("target") == "missense_variant")
    strict_success = bool(
        primary.select(
            (
                (pl.col("standardized_mean_difference") > 0)
                & (pl.col("rank_biserial") > 0)
                & (pl.col("welch_q") < 0.05)
                & (pl.col("mann_whitney_q") < 0.05)
            ).all()
        ).item()
    )
    output_dir.mkdir(parents=True)
    association_path = output_dir / "associations.parquet"
    result_frame.write_parquet(association_path, compression="zstd")
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "analysis_status": "preregistered_untouched_test_transfer",
        "feature_id": FEATURE_ID,
        "primary_target": "missense_variant",
        "secondary_target": "overall",
        "strict_primary_replication": strict_success,
        "panel": panel_manifest,
        "multiple_testing": "BH across 2 orientations x 2 targets per statistic",
        "rows": result_frame.to_dicts(),
        "artifacts": {
            association_path.name: {
                "bytes": association_path.stat().st_size,
                "sha256": sha256_file(association_path),
            }
        },
    }
    write_json(output_dir / "results.json", result)
    result["artifacts"]["results.json"] = {
        "bytes": (output_dir / "results.json").stat().st_size,
        "sha256": sha256_file(output_dir / "results.json"),
    }
    write_json(output_dir / "manifest.json", result)
    (output_dir / "RESULTS.md").write_text(
        "# Feature 1662 untouched-test transfer\n\n"
        f"Strict primary replication: **{strict_success}**\n\n"
        "```json\n" + json.dumps(result["rows"], indent=2) + "\n```\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.panel, args.extraction_root, args.output_dir)
    print(json.dumps(result["rows"], indent=2))


if __name__ == "__main__":
    main()
