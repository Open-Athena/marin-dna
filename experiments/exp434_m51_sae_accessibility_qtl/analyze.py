"""Sparse complete-family accessibility-QTL associations for issue #434."""

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
from scipy import sparse, stats
from sklearn.metrics import average_precision_score

from build_panel import EXPECTED_POSITIVES, EXPECTED_ROWS
from extract_focal import (
    D_SAE,
    ISSUE,
    ORIENTATIONS,
    assert_commit,
    sha256_file,
    write_json,
)

ARMS = ("block01-25m", "block10-25m", "block19-25m")
DATASETS = ("caqtl", "dsqtl")
MIN_CAUSALITY_SUPPORT = 32
MIN_DIRECTION_SUPPORT = 10
FDR_THRESHOLD = 0.05
ALPHAGENOME_CANDIDATES = {
    "block10-25m": (11_137,),
    "block19-25m": (219, 11_928),
}


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=np.float64)
    assert values.ndim == 1 and len(values) > 0
    assert np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all()
    order = np.argsort(values, kind="stable")
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(ranked)
    output[order] = np.minimum(ranked, 1.0)
    return output


def load_delta(path: Path, *, rows: int) -> sparse.csr_matrix:
    frame = pl.read_parquet(path, columns=["panel_row", "feature_id", "delta"])
    assert frame.null_count().sum_horizontal().sum() == 0
    panel_row = frame["panel_row"].to_numpy().astype(np.int32, copy=False)
    feature_id = frame["feature_id"].to_numpy().astype(np.int32, copy=False)
    delta = frame["delta"].to_numpy().astype(np.float32, copy=False)
    assert np.isfinite(delta).all()
    assert (panel_row >= 0).all() and (panel_row < rows).all()
    assert (feature_id >= 0).all() and (feature_id < D_SAE).all()
    keep = delta != 0
    matrix = sparse.coo_matrix(
        (delta[keep], (panel_row[keep], feature_id[keep])),
        shape=(rows, D_SAE),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    assert matrix.nnz > 0
    return matrix


def absolute_response(matrix: sparse.spmatrix) -> sparse.csc_matrix:
    output = matrix.tocsc(copy=True)
    output.data = np.abs(output.data)
    output.eliminate_zeros()
    assert (output.data > 0).all()
    return output


def binary_moments(
    matrix: sparse.spmatrix, labels: np.ndarray
) -> tuple[np.ndarray, ...]:
    matrix = matrix.tocsc()
    labels = np.asarray(labels, dtype=bool)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    assert 1 < n1 < len(labels) - 1
    indicator = labels.astype(np.float64)
    sums1 = np.asarray(matrix.T @ indicator).ravel().astype(np.float64)
    total = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    squared = matrix.copy()
    squared.data = squared.data.astype(np.float64) ** 2
    squares1 = np.asarray(squared.T @ indicator).ravel().astype(np.float64)
    total_squares = np.asarray(squared.sum(axis=0)).ravel().astype(np.float64)
    sums0 = total - sums1
    squares0 = total_squares - squares1
    mean1 = sums1 / n1
    mean0 = sums0 / n0
    var1 = np.maximum((squares1 - sums1 * sums1 / n1) / (n1 - 1), 0.0)
    var0 = np.maximum((squares0 - sums0 * sums0 / n0) / (n0 - 1), 0.0)
    difference = mean1 - mean0
    a = var1 / n1
    b = var0 / n0
    se = np.sqrt(a + b)
    statistic = np.divide(difference, se, out=np.zeros_like(difference), where=se > 0)
    denominator = a * a / (n1 - 1) + b * b / (n0 - 1)
    degrees = np.divide(
        (a + b) ** 2,
        denominator,
        out=np.full_like(denominator, np.inf),
        where=denominator > 0,
    )
    pvalue = 2.0 * stats.t.sf(np.abs(statistic), degrees)
    deterministic = (se == 0) & (difference != 0)
    pvalue[deterministic] = 0.0
    pooled_variance = ((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2)
    standardized = np.divide(
        difference,
        np.sqrt(pooled_variance),
        out=np.zeros_like(difference),
        where=pooled_variance > 0,
    )
    assert np.isfinite(pvalue).all()
    return mean1, mean0, difference, standardized, statistic, pvalue


def sparse_rank_deviations(
    matrix: sparse.spmatrix,
) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
    matrix = matrix.tocsc()
    rows, columns = matrix.shape
    deviations = np.empty(matrix.nnz, dtype=np.float32)
    zero_ranks = np.empty(columns, dtype=np.float64)
    tie_terms = np.empty(columns, dtype=np.float64)
    for column in range(columns):
        start, stop = matrix.indptr[column : column + 2]
        values = matrix.data[start:stop]
        negative = values < 0
        positive = values > 0
        assert np.all(negative | positive)
        negative_count = int(np.count_nonzero(negative))
        zero_count = rows - len(values)
        zero_rank = negative_count + (zero_count + 1.0) / 2.0
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[negative] = stats.rankdata(values[negative], method="average")
        ranks[positive] = (
            stats.rankdata(values[positive], method="average")
            + negative_count
            + zero_count
        )
        deviations[start:stop] = (ranks - zero_rank).astype(np.float32)
        zero_ranks[column] = zero_rank
        _, counts = np.unique(values, return_counts=True)
        tie_terms[column] = zero_count**3 - zero_count + np.sum(counts**3 - counts)
    output = sparse.csc_matrix(
        (deviations, matrix.indices.copy(), matrix.indptr.copy()), shape=matrix.shape
    )
    return output, zero_ranks, tie_terms


def mann_whitney_binary(
    matrix: sparse.spmatrix, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=bool)
    n = len(labels)
    n1 = int(labels.sum())
    n0 = n - n1
    deviations, zero_ranks, tie_terms = sparse_rank_deviations(matrix)
    rank_sum = (
        zero_ranks * n1 + np.asarray(deviations.T @ labels.astype(np.float64)).ravel()
    )
    u = rank_sum - n1 * (n1 + 1.0) / 2.0
    rank_biserial = 2.0 * u / (n1 * n0) - 1.0
    variance = n1 * n0 / 12.0 * (n + 1.0 - tie_terms / (n * (n - 1.0)))
    centered = u - n1 * n0 / 2.0
    z = np.divide(
        centered,
        np.sqrt(np.maximum(variance, 0.0)),
        out=np.zeros_like(centered),
        where=variance > 0,
    )
    pvalue = 2.0 * stats.norm.sf(np.abs(z))
    assert np.isfinite(pvalue).all() and np.isfinite(rank_biserial).all()
    return rank_biserial, pvalue


def sparse_average_precision(matrix: sparse.spmatrix, labels: np.ndarray) -> np.ndarray:
    matrix = matrix.tocsc()
    labels = np.asarray(labels, dtype=np.uint8)
    positives = int(labels.sum())
    assert 0 < positives < len(labels)
    output = np.empty(matrix.shape[1], dtype=np.float64)
    for column in range(matrix.shape[1]):
        start, stop = matrix.indptr[column : column + 2]
        rows = matrix.indices[start:stop]
        values = matrix.data[start:stop]
        assert (values > 0).all()
        scores = np.concatenate((values, np.asarray([0], dtype=values.dtype)))
        unique_scores, inverse = np.unique(scores, return_inverse=True)
        zero_group = int(inverse[-1])
        groups = inverse[:-1]
        sizes = np.bincount(groups, minlength=len(unique_scores)).astype(np.float64)
        positive_counts = np.bincount(
            groups, weights=labels[rows], minlength=len(unique_scores)
        ).astype(np.float64)
        sizes[zero_group] += matrix.shape[0] - len(values)
        positive_counts[zero_group] += positives - int(labels[rows].sum())
        nonempty = np.flatnonzero(sizes > 0)[::-1]
        cumulative_positive = np.cumsum(positive_counts[nonempty])
        cumulative_rows = np.cumsum(sizes[nonempty])
        precision = cumulative_positive / cumulative_rows
        output[column] = np.sum(positive_counts[nonempty] * precision) / positives
    assert np.isfinite(output).all() and ((0 <= output) & (output <= 1)).all()
    return output


def _correlation_pvalue(effect: np.ndarray, n: int) -> np.ndarray:
    clipped = np.clip(effect, -1.0, 1.0)
    denominator = np.maximum(1.0 - clipped * clipped, 0.0)
    statistic = np.divide(
        clipped * np.sqrt(n - 2),
        np.sqrt(denominator),
        out=np.full_like(clipped, np.inf),
        where=denominator > 0,
    )
    pvalue = 2.0 * stats.t.sf(np.abs(statistic), n - 2)
    pvalue[np.abs(clipped) == 1] = 0.0
    return pvalue


def sparse_correlations(
    matrix: sparse.spmatrix, effect: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = matrix.tocsc()
    effect = np.asarray(effect, dtype=np.float64)
    n = len(effect)
    assert matrix.shape[0] == n and n >= 3
    assert np.isfinite(effect).all() and np.std(effect) > 0

    centered = effect - effect.mean()
    sum_x = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    squared = matrix.copy()
    squared.data = squared.data.astype(np.float64) ** 2
    sum_x2 = np.asarray(squared.sum(axis=0)).ravel().astype(np.float64)
    ssx = np.maximum(sum_x2 - sum_x * sum_x / n, 0.0)
    ssy = float(centered @ centered)
    numerator = np.asarray(matrix.T @ centered).ravel()
    pearson = np.divide(
        numerator,
        np.sqrt(ssx * ssy),
        out=np.zeros_like(numerator),
        where=ssx > 0,
    )
    pearson_p = _correlation_pvalue(pearson, n)

    deviations, _, tie_terms = sparse_rank_deviations(matrix)
    effect_rank = stats.rankdata(effect, method="average")
    effect_rank -= effect_rank.mean()
    rank_ssy = float(effect_rank @ effect_rank)
    rank_ssx = np.maximum((n**3 - n - tie_terms) / 12.0, 0.0)
    rank_numerator = np.asarray(deviations.T @ effect_rank).ravel()
    spearman = np.divide(
        rank_numerator,
        np.sqrt(rank_ssx * rank_ssy),
        out=np.zeros_like(rank_numerator),
        where=rank_ssx > 0,
    )
    spearman_p = _correlation_pvalue(spearman, n)
    assert np.isfinite(pearson).all() and np.isfinite(spearman).all()
    return pearson, pearson_p, spearman, spearman_p


def causality_associations(
    matrix: sparse.spmatrix,
    labels: np.ndarray,
    feature_ids: np.ndarray,
    *,
    dataset: str,
    arm: str,
    orientation: str,
) -> pl.DataFrame:
    support = np.asarray(matrix.getnnz(axis=0)).ravel()
    eligible = support >= MIN_CAUSALITY_SUPPORT
    matrix = matrix[:, eligible].tocsc()
    selected = feature_ids[eligible]
    assert len(selected) > 0
    mean1, mean0, difference, standardized, welch_t, welch_p = binary_moments(
        matrix, labels
    )
    rank_biserial, mwu_p = mann_whitney_binary(matrix, labels)
    auprc = sparse_average_precision(matrix, labels)
    return pl.DataFrame(
        {
            "dataset": dataset,
            "arm": arm,
            "report_block": int(arm.removeprefix("block").split("-")[0]),
            "orientation": orientation,
            "outcome": "causality",
            "feature_id": selected.astype(np.uint32),
            "n": np.uint32(len(labels)),
            "n_positive": np.uint32(np.sum(labels)),
            "prevalence": float(np.mean(labels)),
            "nonzero_support": support[eligible].astype(np.uint32),
            "mean_positive": mean1,
            "mean_negative": mean0,
            "mean_difference": difference,
            "standardized_mean_difference": standardized,
            "welch_t": welch_t,
            "welch_p": welch_p,
            "welch_q": benjamini_hochberg(welch_p),
            "rank_biserial": rank_biserial,
            "mann_whitney_p": mwu_p,
            "mann_whitney_q": benjamini_hochberg(mwu_p),
            "official_auprc": auprc,
        }
    )


def direction_associations(
    matrix: sparse.spmatrix,
    effect: np.ndarray,
    feature_ids: np.ndarray,
    *,
    dataset: str,
    arm: str,
    orientation: str,
) -> pl.DataFrame:
    support = np.asarray(matrix.getnnz(axis=0)).ravel()
    eligible = support >= MIN_DIRECTION_SUPPORT
    matrix = matrix[:, eligible].tocsc()
    selected = feature_ids[eligible]
    assert len(selected) > 0
    pearson, pearson_p, spearman, spearman_p = sparse_correlations(matrix, effect)
    sums = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    squared = matrix.copy()
    squared.data = squared.data.astype(np.float64) ** 2
    sums_squared = np.asarray(squared.sum(axis=0)).ravel().astype(np.float64)
    centered_sums_squared = np.maximum(
        sums_squared - sums * sums / matrix.shape[0], 0.0
    )
    estimable = centered_sums_squared > 0
    assert estimable.any()
    return pl.DataFrame(
        {
            "dataset": dataset,
            "arm": arm,
            "report_block": int(arm.removeprefix("block").split("-")[0]),
            "orientation": orientation,
            "outcome": "direction",
            "feature_id": selected[estimable].astype(np.uint32),
            "n": np.uint32(len(effect)),
            "nonzero_support": support[eligible][estimable].astype(np.uint32),
            "pearson": pearson[estimable],
            "pearson_p": pearson_p[estimable],
            "pearson_q": benjamini_hochberg(pearson_p[estimable]),
            "spearman": spearman[estimable],
            "spearman_p": spearman_p[estimable],
            "spearman_q": benjamini_hochberg(spearman_p[estimable]),
        }
    )


def baseline_metrics(panel: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    scorers = {
        "chrombpnet_atac": ("chrombpnet_atac_ips", "chrombpnet_atac_logfc"),
        "chrombpnet_dnase": ("chrombpnet_dnase_ips", "chrombpnet_dnase_logfc"),
        "enformer_dnase": (
            "enformer_dnase_local_logfc",
            "enformer_dnase_local_logfc",
        ),
    }
    for dataset in DATASETS:
        frame = panel.filter(pl.col("dataset") == dataset)
        labels = frame["label"].to_numpy().astype(bool)
        effect = frame["effect"].to_numpy()
        for scorer, (causality_col, direction_col) in scorers.items():
            causality = frame[causality_col].to_numpy()
            direction = frame[direction_col].to_numpy()
            assert np.isfinite(causality).all() and np.isfinite(direction).all()
            rows.append(
                {
                    "dataset": dataset,
                    "scorer": scorer,
                    "n": len(labels),
                    "n_positive": int(labels.sum()),
                    "causality_auprc": float(
                        average_precision_score(labels, np.abs(causality))
                    ),
                    "direction_pearson": float(
                        stats.pearsonr(direction[labels], effect[labels]).statistic
                    ),
                    "direction_spearman": float(
                        stats.spearmanr(direction[labels], effect[labels]).statistic
                    ),
                }
            )
    return pl.DataFrame(rows)


def verify_artifacts(root: Path, manifest: dict[str, Any]) -> None:
    for relative, expected in manifest["artifacts"].items():
        path = root / relative
        assert path.is_file(), path
        assert path.stat().st_size == expected["bytes"], path
        assert sha256_file(path) == expected["sha256"], path


def analyze(
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
    assert panel_manifest["panel"]["sha256"] == sha256_file(panel_path)
    assert extraction_manifest["panel"]["sha256"] == sha256_file(panel_path)
    assert extraction_manifest["panel"]["rows"] == EXPECTED_ROWS
    verify_artifacts(extraction_root, extraction_manifest)
    panel = pl.read_parquet(panel_path)
    assert panel.height == EXPECTED_ROWS

    output_dir.mkdir(parents=True)
    family_dir = output_dir / "families"
    family_dir.mkdir()
    artifacts: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    candidate_frames: list[pl.DataFrame] = []
    for arm in ARMS:
        for orientation in ORIENTATIONS:
            relative = f"{arm}/sae_focal_{orientation}.parquet"
            signed_full = load_delta(extraction_root / relative, rows=EXPECTED_ROWS)
            all_features = np.arange(D_SAE, dtype=np.int32)
            for dataset in DATASETS:
                indices = np.flatnonzero(panel["dataset"].to_numpy() == dataset)
                dataset_panel = panel.filter(pl.col("dataset") == dataset)
                labels = dataset_panel["label"].to_numpy().astype(bool)
                assert int(labels.sum()) == EXPECTED_POSITIVES[dataset]
                signed = signed_full[indices, :].tocsr()

                causality = causality_associations(
                    absolute_response(signed),
                    labels,
                    all_features,
                    dataset=dataset,
                    arm=arm,
                    orientation=orientation,
                )
                causality_name = f"{dataset}__{arm}__{orientation}__causality.parquet"
                causality_path = family_dir / causality_name
                causality.write_parquet(causality_path, compression="zstd")
                artifacts[str(causality_path.relative_to(output_dir))] = {
                    "bytes": causality_path.stat().st_size,
                    "sha256": sha256_file(causality_path),
                    "rows": causality.height,
                }

                positive_matrix = signed[labels, :].tocsr()
                effect = dataset_panel.filter(pl.col("label"))["effect"].to_numpy()
                assert len(effect) == labels.sum() and np.isfinite(effect).all()
                direction = direction_associations(
                    positive_matrix,
                    effect,
                    all_features,
                    dataset=dataset,
                    arm=arm,
                    orientation=orientation,
                )
                direction_name = f"{dataset}__{arm}__{orientation}__direction.parquet"
                direction_path = family_dir / direction_name
                direction.write_parquet(direction_path, compression="zstd")
                artifacts[str(direction_path.relative_to(output_dir))] = {
                    "bytes": direction_path.stat().st_size,
                    "sha256": sha256_file(direction_path),
                    "rows": direction.height,
                }

                summaries.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "report_block": int(arm.removeprefix("block").split("-")[0]),
                        "orientation": orientation,
                        "causality_features": causality.height,
                        "causality_welch_q05": causality.filter(
                            pl.col("welch_q") < FDR_THRESHOLD
                        ).height,
                        "causality_mwu_q05": causality.filter(
                            pl.col("mann_whitney_q") < FDR_THRESHOLD
                        ).height,
                        "causality_concordant_q05": causality.filter(
                            (pl.col("welch_q") < FDR_THRESHOLD)
                            & (pl.col("mann_whitney_q") < FDR_THRESHOLD)
                        ).height,
                        "best_causality_auprc": causality["official_auprc"].max(),
                        "direction_features": direction.height,
                        "direction_pearson_q05": direction.filter(
                            pl.col("pearson_q") < FDR_THRESHOLD
                        ).height,
                        "direction_spearman_q05": direction.filter(
                            pl.col("spearman_q") < FDR_THRESHOLD
                        ).height,
                        "direction_concordant_q05": direction.filter(
                            (pl.col("pearson_q") < FDR_THRESHOLD)
                            & (pl.col("spearman_q") < FDR_THRESHOLD)
                        ).height,
                        "max_abs_direction_pearson": direction["pearson"].abs().max(),
                        "max_abs_direction_spearman": direction["spearman"].abs().max(),
                    }
                )
                candidates = ALPHAGENOME_CANDIDATES.get(arm, ())
                if candidates:
                    candidate_frames.extend(
                        [
                            causality.filter(pl.col("feature_id").is_in(candidates)),
                            direction.filter(pl.col("feature_id").is_in(candidates)),
                        ]
                    )
                print(
                    json.dumps({"stage": "family_complete", **summaries[-1]}),
                    flush=True,
                )

    summary = pl.DataFrame(summaries, infer_schema_length=None)
    summary_path = output_dir / "family_summary.parquet"
    summary.write_parquet(summary_path, compression="zstd")
    artifacts[summary_path.name] = {
        "bytes": summary_path.stat().st_size,
        "sha256": sha256_file(summary_path),
        "rows": summary.height,
    }
    baselines = baseline_metrics(panel)
    baseline_path = output_dir / "official_baseline_sanity.parquet"
    baselines.write_parquet(baseline_path, compression="zstd")
    artifacts[baseline_path.name] = {
        "bytes": baseline_path.stat().st_size,
        "sha256": sha256_file(baseline_path),
        "rows": baselines.height,
    }
    candidates = pl.concat(candidate_frames, how="diagonal_relaxed")
    candidate_path = output_dir / "alphagenome_candidate_overlap.parquet"
    candidates.write_parquet(candidate_path, compression="zstd")
    artifacts[candidate_path.name] = {
        "bytes": candidate_path.stat().st_size,
        "sha256": sha256_file(candidate_path),
        "rows": candidates.height,
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
            "panel_sha256": sha256_file(panel_path),
            "extraction_manifest_sha256": sha256_file(
                extraction_root / "manifest.json"
            ),
            "extraction_run_id": extraction_manifest["run_id"],
            "rows": panel.height,
            "dataset_rows": dict(panel.group_by("dataset").len().iter_rows()),
            "dataset_positives": dict(
                panel.group_by("dataset").agg(pl.col("label").sum()).iter_rows()
            ),
        },
        "protocol": {
            "layers_reported": [1, 10, 19],
            "sae_training_activations": 25_000_200,
            "orientations": list(ORIENTATIONS),
            "uses_all_official_variants": True,
            "uses_discovery_test_split": False,
            "causality_response": "abs(activation_alt - activation_ref)",
            "causality_tests": ["Welch t", "Mann-Whitney U"],
            "causality_effect_sizes": [
                "standardized mean difference",
                "rank-biserial correlation",
            ],
            "causality_descriptive_metric": "official AUPRC",
            "direction_response": "activation_alt - activation_ref",
            "direction_population": "official causal variants only",
            "direction_metrics": ["Pearson", "Spearman"],
            "minimum_causality_nonzero_support": MIN_CAUSALITY_SUPPORT,
            "minimum_direction_nonzero_support_among_causal": MIN_DIRECTION_SUPPORT,
            "bh_family": "dataset x layer x orientation x outcome x statistic",
            "fdr_threshold": FDR_THRESHOLD,
            "alphagenome_candidates_fixed_before_qtl_outcomes": ALPHAGENOME_CANDIDATES,
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
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
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
