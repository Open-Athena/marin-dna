"""Run the preregistered focal Mendelian SAE feature-association scan.

The analysis uses every eligible variant and feature. Eligibility depends only on
the response support, never on the Mendelian label. BH correction is performed
within each layer x budget x orientation x focal x response x statistic family
across the overall target and every adequately supported predefined subset.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import scipy
from scipy import stats

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
from train import assert_commit

MIN_NONZERO_SUPPORT = 10
MIN_CLASS_SIZE = 40
FDR_THRESHOLD = 0.05
RESPONSES = ("abs_delta", "delta", "ref_activation", "alt_activation")
PRIMARY_RESPONSES = frozenset(("abs_delta", "delta"))
EXPECTED_INFERENTIAL_SUBSETS = frozenset(
    (
        "3_prime_UTR_variant",
        "5_prime_UTR_variant",
        "distal",
        "missense_variant",
        "non_coding_transcript_exon_variant",
        "splicing",
        "synonymous_variant",
        "tss_proximal",
    )
)
DESCRIPTIVE_ONLY_SUBSETS = frozenset(("mature_miRNA_variant",))


@dataclass(frozen=True)
class Target:
    kind: str
    name: str
    indices: np.ndarray
    labels: np.ndarray


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return adjusted
    assert np.all((values[valid] >= 0) & (values[valid] <= 1))
    positions = np.flatnonzero(valid)
    order = np.argsort(values[valid], kind="stable")
    ranked = values[valid][order]
    scaled = ranked * ranked.size / np.arange(1, ranked.size + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    destination = positions[order]
    adjusted[destination] = monotone
    return adjusted


def _average_precision_from_sorted(
    sorted_scores: np.ndarray,
    sorted_labels: np.ndarray,
    positives: int,
) -> np.ndarray:
    assert sorted_scores.shape == sorted_labels.shape
    assert sorted_scores.ndim == 2 and positives > 0
    cumulative_positives = np.cumsum(sorted_labels, axis=0, dtype=np.int64)
    group_end = np.empty(sorted_scores.shape, dtype=bool)
    group_end[-1, :] = True
    group_end[:-1, :] = sorted_scores[:-1, :] != sorted_scores[1:, :]
    result = np.empty(sorted_scores.shape[1], dtype=np.float64)
    for column in range(sorted_scores.shape[1]):
        ends = np.flatnonzero(group_end[:, column])
        true_positives = cumulative_positives[ends, column]
        positives_in_group = np.diff(np.concatenate(([0], true_positives)))
        precision = true_positives / (ends + 1)
        result[column] = np.sum((positives_in_group / positives) * precision)
    return result


def average_precision_both_directions(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores)
    assert labels.ndim == 1 and scores.ndim == 2
    assert scores.shape[0] == labels.size and chunk_size > 0
    assert np.isfinite(scores).all()
    positives = int(labels.sum())
    assert 0 < positives < labels.size
    raw = np.empty(scores.shape[1], dtype=np.float64)
    negated = np.empty(scores.shape[1], dtype=np.float64)
    for start in range(0, scores.shape[1], chunk_size):
        stop = min(start + chunk_size, scores.shape[1])
        block = scores[:, start:stop]
        ascending_order = np.argsort(block, axis=0, kind="stable")
        ascending_scores = np.take_along_axis(block, ascending_order, axis=0)
        ascending_labels = labels[ascending_order]
        negated[start:stop] = _average_precision_from_sorted(
            ascending_scores, ascending_labels, positives
        )
        descending_scores = ascending_scores[::-1, :]
        descending_labels = ascending_labels[::-1, :]
        raw[start:stop] = _average_precision_from_sorted(
            descending_scores, descending_labels, positives
        )
    return raw, negated


def target_definitions(frame: pl.DataFrame) -> list[Target]:
    assert frame.height == EXPECTED_ROWS
    labels = frame["label"].cast(pl.UInt8).to_numpy()
    overall_positive = int(labels.sum())
    assert overall_positive * 10 == labels.size
    targets = [
        Target(
            kind="overall",
            name="overall",
            indices=np.arange(frame.height, dtype=np.int64),
            labels=labels,
        )
    ]
    inferential: set[str] = set()
    descriptive: set[str] = set()
    subsets = sorted(frame["subset"].cast(pl.String).unique().to_list())
    subset_values = frame["subset"].cast(pl.String).to_numpy()
    for subset in subsets:
        indices = np.flatnonzero(subset_values == subset)
        subset_labels = labels[indices]
        positives = int(subset_labels.sum())
        negatives = int(subset_labels.size - positives)
        assert positives * 10 == subset_labels.size
        if positives >= MIN_CLASS_SIZE and negatives >= MIN_CLASS_SIZE:
            inferential.add(subset)
            targets.append(
                Target(
                    kind="subset",
                    name=subset,
                    indices=indices,
                    labels=subset_labels,
                )
            )
        else:
            descriptive.add(subset)
    assert inferential == EXPECTED_INFERENTIAL_SUBSETS
    assert descriptive == DESCRIPTIVE_ONLY_SUBSETS
    return targets


def load_dense_pair(path: Path, *, rows: int) -> tuple[np.ndarray, np.ndarray, int]:
    assert path.is_file()
    table = pq.read_table(
        path,
        columns=[
            "panel_row",
            "feature_id",
            "ref_activation",
            "alt_activation",
        ],
        memory_map=True,
    )
    panel_rows = table["panel_row"].to_numpy(zero_copy_only=False)
    feature_ids = table["feature_id"].to_numpy(zero_copy_only=False)
    ref_values = table["ref_activation"].to_numpy(zero_copy_only=False)
    alt_values = table["alt_activation"].to_numpy(zero_copy_only=False)
    assert panel_rows.dtype == np.uint32 and feature_ids.dtype == np.uint32
    assert ref_values.dtype == np.float32 and alt_values.dtype == np.float32
    assert panel_rows.size == feature_ids.size == ref_values.size == alt_values.size
    assert panel_rows.size > 0
    assert int(panel_rows.max()) < rows and int(feature_ids.max()) < D_SAE
    ordered = (panel_rows[1:] > panel_rows[:-1]) | (
        (panel_rows[1:] == panel_rows[:-1]) & (feature_ids[1:] > feature_ids[:-1])
    )
    assert ordered.all()
    assert np.isfinite(ref_values).all() and np.isfinite(alt_values).all()
    assert np.all(ref_values >= 0) and np.all(alt_values >= 0)
    assert np.all((ref_values != 0) | (alt_values != 0))
    ref = np.zeros((rows, D_SAE), dtype=np.float32)
    alt = np.zeros((rows, D_SAE), dtype=np.float32)
    ref[panel_rows, feature_ids] = ref_values
    alt[panel_rows, feature_ids] = alt_values
    sparse_rows = int(panel_rows.size)
    del table, panel_rows, feature_ids, ref_values, alt_values, ordered
    gc.collect()
    return ref, alt, sparse_rows


def analyze_target(
    response: np.ndarray,
    target: Target,
    *,
    arm: str,
    block: int,
    budget: int,
    orientation: str,
    response_name: str,
    ap_chunk_size: int,
    pooling: str = "focal",
) -> pl.DataFrame:
    assert response.shape == (EXPECTED_ROWS, D_SAE)
    assert orientation in ORIENTATIONS and response_name in RESPONSES
    assert pooling in ("focal", "whole_window_mean")
    matrix = response if target.kind == "overall" else response[target.indices, :]
    labels = target.labels
    assert matrix.shape[0] == labels.size
    support = np.count_nonzero(matrix, axis=0)
    minimum = matrix.min(axis=0)
    maximum = matrix.max(axis=0)
    eligible = np.flatnonzero((support >= MIN_NONZERO_SUPPORT) & (minimum != maximum))
    assert eligible.size > 0
    values = matrix[:, eligible]
    positive = values[labels == 1, :]
    negative = values[labels == 0, :]
    n_positive = positive.shape[0]
    n_negative = negative.shape[0]
    assert n_positive >= MIN_CLASS_SIZE and n_negative >= MIN_CLASS_SIZE

    mean_positive = positive.mean(axis=0, dtype=np.float64)
    mean_negative = negative.mean(axis=0, dtype=np.float64)
    variance_positive = positive.var(axis=0, ddof=1, dtype=np.float64)
    variance_negative = negative.var(axis=0, ddof=1, dtype=np.float64)
    mean_difference = mean_positive - mean_negative
    pooled_sd = np.sqrt((variance_positive + variance_negative) / 2)
    standardized_mean_difference = np.divide(
        mean_difference,
        pooled_sd,
        out=np.full(mean_difference.shape, np.nan, dtype=np.float64),
        where=pooled_sd > 0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        welch = stats.ttest_ind(
            positive,
            negative,
            axis=0,
            equal_var=False,
            nan_policy="raise",
        )
        mann_whitney = stats.mannwhitneyu(
            positive,
            negative,
            axis=0,
            alternative="two-sided",
            method="asymptotic",
        )
    welch_statistic = np.asarray(welch.statistic, dtype=np.float64)
    welch_p = np.asarray(welch.pvalue, dtype=np.float64)
    u_statistic = np.asarray(mann_whitney.statistic, dtype=np.float64)
    mann_whitney_p = np.asarray(mann_whitney.pvalue, dtype=np.float64)
    rank_biserial = 2 * u_statistic / (n_positive * n_negative) - 1
    auprc, auprc_negated = average_precision_both_directions(
        labels, values, chunk_size=ap_chunk_size
    )
    best_is_raw = auprc >= auprc_negated

    rows = eligible.size
    return pl.DataFrame(
        {
            "arm": [arm] * rows,
            "block": np.full(rows, block, dtype=np.uint8),
            "budget": np.full(rows, budget, dtype=np.uint32),
            "orientation": [orientation] * rows,
            "pooling": [pooling] * rows,
            "response": [response_name] * rows,
            "response_role": [
                "primary"
                if response_name in PRIMARY_RESPONSES
                else "contextual_control"
            ]
            * rows,
            "target_kind": [target.kind] * rows,
            "target": [target.name] * rows,
            "feature_id": eligible.astype(np.uint32),
            "n": np.full(rows, labels.size, dtype=np.uint32),
            "n_positive": np.full(rows, n_positive, dtype=np.uint32),
            "n_negative": np.full(rows, n_negative, dtype=np.uint32),
            "prevalence": np.full(rows, n_positive / labels.size, dtype=np.float64),
            "nonzero_support": support[eligible].astype(np.uint32),
            "mean_positive": mean_positive,
            "mean_negative": mean_negative,
            "mean_difference": mean_difference,
            "standardized_mean_difference": standardized_mean_difference,
            "welch_statistic": welch_statistic,
            "welch_p": welch_p,
            "u_statistic": u_statistic,
            "rank_biserial": rank_biserial,
            "mann_whitney_p": mann_whitney_p,
            "auprc": auprc,
            "auprc_negated": auprc_negated,
            "best_auprc": np.maximum(auprc, auprc_negated),
            "best_auprc_direction": np.where(best_is_raw, "higher", "lower"),
        }
    )


def correct_family(frames: list[pl.DataFrame]) -> pl.DataFrame:
    assert frames
    family = pl.concat(frames, how="vertical")
    welch_q = bh_adjust(family["welch_p"].to_numpy())
    mann_whitney_q = bh_adjust(family["mann_whitney_p"].to_numpy())
    minimum_q = np.fmin(welch_q, mann_whitney_q)
    return family.with_columns(
        pl.Series("welch_q", welch_q),
        pl.Series("mann_whitney_q", mann_whitney_q),
        pl.Series("minimum_q", minimum_q),
    )


def response_matrix(
    name: str,
    *,
    ref: np.ndarray,
    alt: np.ndarray,
    delta: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    assert name in RESPONSES
    if name == "ref_activation":
        return ref, delta
    if name == "alt_activation":
        return alt, delta
    if delta is None:
        delta = alt - ref
        assert delta.dtype == np.float32 and np.isfinite(delta).all()
    if name == "delta":
        return delta, delta
    assert name == "abs_delta"
    return np.abs(delta), delta


def verify_input_artifacts(root: Path, manifest: dict[str, Any]) -> None:
    for relative, expected in manifest["artifacts"].items():
        path = root / relative
        assert path.is_file(), path
        assert path.stat().st_size == expected["bytes"], path
        assert sha256_file(path) == expected["sha256"], path


def family_summary(frame: pl.DataFrame) -> dict[str, Any]:
    welch_q = frame["welch_q"].to_numpy()
    mann_q = frame["mann_whitney_q"].to_numpy()
    minimum_q = frame["minimum_q"].to_numpy()
    finite = minimum_q[np.isfinite(minimum_q)]
    assert finite.size > 0
    return {
        "eligible_feature_target_pairs": frame.height,
        "targets": frame["target"].n_unique(),
        "features": frame["feature_id"].n_unique(),
        "welch_discoveries_q05": int(np.count_nonzero(welch_q <= FDR_THRESHOLD)),
        "mann_whitney_discoveries_q05": int(np.count_nonzero(mann_q <= FDR_THRESHOLD)),
        "union_discoveries_q05": int(np.count_nonzero(minimum_q <= FDR_THRESHOLD)),
        "minimum_q": float(finite.min()),
        "maximum_best_auprc": float(frame["best_auprc"].max()),
    }


def parse_arm(arm: str) -> tuple[int, int]:
    block_text, budget_text = arm.split("-", maxsplit=1)
    block = int(block_text.removeprefix("block"))
    budget = int(budget_text.removesuffix("m")) * 1_000_000
    if budget == 5_000_000:
        budget = BUDGETS[0]
    elif budget == 25_000_000:
        budget = BUDGETS[1]
    assert block in (1, 10, 19) and budget in BUDGETS
    return block, budget


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
    verify_input_artifacts(extraction_root, extraction_manifest)

    frame = pl.read_parquet(panel_path).with_row_index("panel_row")
    assert frame["panel_row"].to_list() == list(range(EXPECTED_ROWS))
    targets = target_definitions(frame)
    output_dir.mkdir(parents=True)
    families_dir = output_dir / "families"
    families_dir.mkdir()
    summaries: dict[str, Any] = {}
    top_frames: list[pl.DataFrame] = []
    artifacts: dict[str, Any] = {}

    expected_arms = {
        f"block{block:02d}-{budget // 1_000_000}m"
        for block in (1, 10, 19)
        for budget in (5_000_000, 25_000_000)
    }
    arms = sorted(extraction_manifest["outputs"])
    assert set(arms) == expected_arms
    for arm in arms:
        block, budget = parse_arm(arm)
        summaries[arm] = {}
        for orientation in ORIENTATIONS:
            sparse_path = extraction_root / arm / f"sae_focal_{orientation}.parquet"
            print(
                json.dumps(
                    {
                        "stage": "load_dense",
                        "arm": arm,
                        "orientation": orientation,
                        "path": str(sparse_path),
                    }
                ),
                flush=True,
            )
            ref, alt, sparse_rows = load_dense_pair(sparse_path, rows=EXPECTED_ROWS)
            expected_sparse_rows = extraction_manifest["outputs"][arm][orientation][
                "rows"
            ]
            assert sparse_rows == expected_sparse_rows
            delta: np.ndarray | None = None
            summaries[arm][orientation] = {}
            for response_name in RESPONSES:
                response, delta = response_matrix(
                    response_name, ref=ref, alt=alt, delta=delta
                )
                print(
                    json.dumps(
                        {
                            "stage": "analyze_family",
                            "arm": arm,
                            "orientation": orientation,
                            "response": response_name,
                            "targets": len(targets),
                        }
                    ),
                    flush=True,
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
            "pooling": "focal",
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
                "layer x budget x orientation x focal x response x statistic; "
                "all eligible overall and within-subset feature-target pairs"
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
