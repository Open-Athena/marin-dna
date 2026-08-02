"""Complete-family multi-layer associations for issue #422 consequences."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from scipy import sparse, stats

ISSUE = 422
D_SAE = 15_360
EXPECTED_ROWS = 17_920
EXPECTED_CLASSES = 35
EXPECTED_PER_CLASS = 512
ARMS = ("block01-25m", "block10-25m", "block19-25m")
ORIENTATIONS = ("forward", "reverse_complement")
RESPONSES = ("absolute", "signed")
MIN_SUPPORT = 32
CHANCE_AUPRC = 1.0 / EXPECTED_CLASSES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    flat = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    assert len(flat) > 0
    assert np.isfinite(flat).all() and ((0 <= flat) & (flat <= 1)).all()
    order = np.argsort(flat, kind="stable")
    adjusted = flat[order] * len(flat) / np.arange(1, len(flat) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output.reshape(pvalues.shape)


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


def transform_response(
    matrix: sparse.csr_matrix, response: Literal["absolute", "signed"]
) -> sparse.csr_matrix:
    if response == "signed":
        return matrix
    assert response == "absolute"
    output = matrix.copy()
    output.data = np.abs(output.data)
    output.eliminate_zeros()
    return output


def class_indicator(labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    classes = sorted(str(value) for value in np.unique(labels))
    assert len(classes) == EXPECTED_CLASSES
    lookup = {value: index for index, value in enumerate(classes)}
    encoded = np.asarray([lookup[str(value)] for value in labels], dtype=np.int16)
    counts = np.bincount(encoded, minlength=EXPECTED_CLASSES)
    assert np.all(counts == EXPECTED_PER_CLASS)
    indicator = np.zeros((len(labels), EXPECTED_CLASSES), dtype=np.float64)
    indicator[np.arange(len(labels)), encoded] = 1.0
    return indicator, classes


def group_moments(
    matrix: sparse.spmatrix, indicator: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = matrix.tocsc()
    sums = np.asarray(matrix.T @ indicator, dtype=np.float64)
    squared = matrix.copy()
    squared.data = squared.data.astype(np.float64) ** 2
    sum_squares = np.asarray(squared.T @ indicator, dtype=np.float64)
    total_sum = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    total_squares = np.asarray(squared.sum(axis=0)).ravel().astype(np.float64)
    assert sums.shape == sum_squares.shape == (matrix.shape[1], EXPECTED_CLASSES)
    return sums, sum_squares, total_sum, total_squares


def welch_one_vs_rest(
    matrix: sparse.spmatrix, indicator: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Welch p-values, mean differences, and pooled standardized differences."""

    sums, sum_squares, total_sum, total_squares = group_moments(matrix, indicator)
    n1 = float(EXPECTED_PER_CLASS)
    n0 = float(EXPECTED_ROWS - EXPECTED_PER_CLASS)
    pos_mean = sums / n1
    neg_sum = total_sum[:, None] - sums
    neg_squares = total_squares[:, None] - sum_squares
    neg_mean = neg_sum / n0
    pos_var = np.maximum((sum_squares - sums * sums / n1) / (n1 - 1), 0.0)
    neg_var = np.maximum((neg_squares - neg_sum * neg_sum / n0) / (n0 - 1), 0.0)
    difference = pos_mean - neg_mean
    a = pos_var / n1
    b = neg_var / n0
    standard_error = np.sqrt(a + b)
    statistic = np.divide(
        difference,
        standard_error,
        out=np.zeros_like(difference),
        where=standard_error > 0,
    )
    denominator = a * a / (n1 - 1) + b * b / (n0 - 1)
    degrees = np.divide(
        (a + b) ** 2,
        denominator,
        out=np.full_like(denominator, np.inf),
        where=denominator > 0,
    )
    pvalue = 2.0 * stats.t.sf(np.abs(statistic), degrees)
    deterministic_difference = (standard_error == 0) & (difference != 0)
    pvalue[deterministic_difference] = 0.0
    pooled_variance = ((n1 - 1) * pos_var + (n0 - 1) * neg_var) / (n1 + n0 - 2)
    standardized = np.divide(
        difference,
        np.sqrt(pooled_variance),
        out=np.zeros_like(difference),
        where=pooled_variance > 0,
    )
    assert np.isfinite(pvalue).all() and np.isfinite(difference).all()
    assert np.isfinite(standardized).all()
    return pvalue, difference, standardized


def sparse_rank_deviations(
    matrix: sparse.spmatrix,
) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
    """Represent tied ranks by sparse deviations from the implicit-zero rank."""

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
    assert np.isfinite(output.data).all()
    assert np.isfinite(zero_ranks).all() and np.isfinite(tie_terms).all()
    return output, zero_ranks, tie_terms


def mann_whitney_one_vs_rest(
    deviations: sparse.csc_matrix,
    zero_ranks: np.ndarray,
    tie_terms: np.ndarray,
    indicator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymptotic two-sided MW tests without continuity correction."""

    n1 = float(EXPECTED_PER_CLASS)
    n0 = float(EXPECTED_ROWS - EXPECTED_PER_CLASS)
    rank_sum = zero_ranks[:, None] * n1 + np.asarray(
        deviations.T @ indicator, dtype=np.float64
    )
    u = rank_sum - n1 * (n1 + 1.0) / 2.0
    rank_biserial = 2.0 * u / (n1 * n0) - 1.0
    variance = (
        n1
        * n0
        / 12.0
        * (
            EXPECTED_ROWS
            + 1.0
            - tie_terms[:, None] / (EXPECTED_ROWS * (EXPECTED_ROWS - 1.0))
        )
    )
    centered = u - n1 * n0 / 2.0
    statistic = np.divide(
        centered,
        np.sqrt(np.maximum(variance, 0.0)),
        out=np.zeros_like(centered),
        where=variance > 0,
    )
    pvalue = 2.0 * stats.norm.sf(np.abs(statistic))
    assert np.isfinite(pvalue).all() and np.isfinite(rank_biserial).all()
    return pvalue, rank_biserial, rank_sum


def directional_average_precision(
    matrix: sparse.spmatrix,
    encoded_labels: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Compute tie-aware direction-aligned AP for every feature and class."""

    matrix = matrix.tocsc()
    assert direction.shape == (matrix.shape[1], EXPECTED_CLASSES)
    class_totals = np.bincount(encoded_labels, minlength=EXPECTED_CLASSES).astype(
        np.float64
    )
    output = np.empty(direction.shape, dtype=np.float64)
    for column in range(matrix.shape[1]):
        start, stop = matrix.indptr[column : column + 2]
        rows = matrix.indices[start:stop]
        values = matrix.data[start:stop]
        scores = np.concatenate((values, np.asarray([0], dtype=values.dtype)))
        unique_scores, inverse = np.unique(scores, return_inverse=True)
        zero_group = int(inverse[-1])
        nonzero_groups = inverse[:-1]
        group_sizes = np.bincount(nonzero_groups, minlength=len(unique_scores)).astype(
            np.float64
        )
        group_sizes[zero_group] += matrix.shape[0] - len(values)
        group_class = np.zeros((len(unique_scores), EXPECTED_CLASSES), dtype=np.float64)
        np.add.at(group_class, (nonzero_groups, encoded_labels[rows]), 1.0)
        nonzero_class = np.bincount(encoded_labels[rows], minlength=EXPECTED_CLASSES)
        group_class[zero_group] += class_totals - nonzero_class

        def average_precision(
            order: np.ndarray,
            *,
            counts: np.ndarray = group_class,
            sizes: np.ndarray = group_sizes,
        ) -> np.ndarray:
            ordered_class = counts[order]
            cumulative_true = np.cumsum(ordered_class, axis=0)
            cumulative_rows = np.cumsum(sizes[order])[:, None]
            precision = cumulative_true / cumulative_rows
            return np.sum(ordered_class * precision, axis=0) / class_totals

        descending = np.arange(len(unique_scores) - 1, -1, -1)
        positive_ap = average_precision(descending)
        negative_ap = average_precision(descending[::-1])
        output[column] = np.where(direction[column] >= 0, positive_ap, negative_ap)
    assert np.isfinite(output).all() and ((0 <= output) & (output <= 1)).all()
    return output


def welch_omnibus(
    matrix: sparse.spmatrix, indicator: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Welch one-way ANOVA where every class has estimable variance."""

    sums, sum_squares, _, _ = group_moments(matrix, indicator)
    n = float(EXPECTED_PER_CLASS)
    means = sums / n
    variances = np.maximum((sum_squares - sums * sums / n) / (n - 1), 0.0)
    valid = np.all(variances > 0, axis=1)
    statistic = np.zeros(matrix.shape[1], dtype=np.float64)
    pvalue = np.ones(matrix.shape[1], dtype=np.float64)
    if valid.any():
        valid_variance = variances[valid]
        weights = n / valid_variance
        total_weight = weights.sum(axis=1)
        weighted_mean = (weights * means[valid]).sum(axis=1) / total_weight
        numerator = (weights * (means[valid] - weighted_mean[:, None]) ** 2).sum(
            axis=1
        ) / (EXPECTED_CLASSES - 1)
        correction_sum = ((1.0 - weights / total_weight[:, None]) ** 2 / (n - 1)).sum(
            axis=1
        )
        correction = (
            1.0
            + 2.0 * (EXPECTED_CLASSES - 2) / (EXPECTED_CLASSES**2 - 1) * correction_sum
        )
        f = numerator / correction
        df2 = (EXPECTED_CLASSES**2 - 1) / (3.0 * correction_sum)
        statistic[valid] = f
        pvalue[valid] = stats.f.sf(f, EXPECTED_CLASSES - 1, df2)
    assert np.isfinite(statistic).all() and np.isfinite(pvalue).all()
    return pvalue, statistic, valid


def kruskal_omnibus(
    rank_sum: np.ndarray, tie_terms: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = 12.0 / (EXPECTED_ROWS * (EXPECTED_ROWS + 1.0)) * np.sum(
        rank_sum * rank_sum / EXPECTED_PER_CLASS, axis=1
    ) - 3.0 * (EXPECTED_ROWS + 1.0)
    correction = 1.0 - tie_terms / (EXPECTED_ROWS**3 - EXPECTED_ROWS)
    valid = correction > 0
    statistic = np.divide(
        raw,
        correction,
        out=np.zeros_like(raw),
        where=valid,
    )
    statistic = np.maximum(statistic, 0.0)
    pvalue = np.ones_like(statistic)
    pvalue[valid] = stats.chi2.sf(statistic[valid], EXPECTED_CLASSES - 1)
    assert np.isfinite(statistic).all() and np.isfinite(pvalue).all()
    return pvalue, statistic, valid


def analyze(
    *,
    panel_path: Path,
    extraction_root: Path,
    extraction_manifest_path: Path,
    output_dir: Path,
    analysis_commit: str,
) -> dict[str, Any]:
    assert len(analysis_commit) == 40 and all(
        character in "0123456789abcdef" for character in analysis_commit
    )
    assert panel_path.is_file() and extraction_root.is_dir()
    assert extraction_manifest_path.is_file() and not output_dir.exists()
    started = time.monotonic()
    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    assert extraction_manifest["panel"]["sha256"] == sha256_file(panel_path)
    assert extraction_manifest["panel"]["rows"] == EXPECTED_ROWS
    panel = pl.read_parquet(panel_path)
    assert panel.height == EXPECTED_ROWS
    labels = panel["consequence_cre"].to_numpy()
    indicator, classes = class_indicator(labels)
    encoded_labels = indicator.argmax(axis=1).astype(np.int16)

    output_dir.mkdir(parents=True)
    family_dir = output_dir / "families"
    family_dir.mkdir()
    artifacts: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    total_hypotheses = 0
    for arm in ARMS:
        report_block = int(arm.removeprefix("block").split("-")[0])
        for orientation in ORIENTATIONS:
            relative = f"{arm}/sae_focal_{orientation}.parquet"
            path = extraction_root / relative
            assert path.is_file(), path
            expected = extraction_manifest["artifacts"][relative]
            assert expected["bytes"] == path.stat().st_size
            assert expected["sha256"] == sha256_file(path)
            signed_matrix = load_delta(path, rows=EXPECTED_ROWS)
            for response in RESPONSES:
                matrix = transform_response(signed_matrix, response).tocsc()
                support = np.asarray(matrix.getnnz(axis=0)).ravel()
                eligible = support >= MIN_SUPPORT
                feature_ids = np.flatnonzero(eligible).astype(np.int32)
                matrix = matrix[:, eligible].tocsc()
                assert len(feature_ids) > 0

                welch_p, mean_difference, standardized = welch_one_vs_rest(
                    matrix, indicator
                )
                deviations, zero_ranks, tie_terms = sparse_rank_deviations(matrix)
                mwu_p, rank_biserial, rank_sum = mann_whitney_one_vs_rest(
                    deviations, zero_ranks, tie_terms, indicator
                )
                welch_q = benjamini_hochberg(welch_p)
                mwu_q = benjamini_hochberg(mwu_p)
                auprc = directional_average_precision(
                    matrix, encoded_labels, mean_difference
                )
                total_hypotheses += 2 * welch_p.size

                features = np.repeat(feature_ids, EXPECTED_CLASSES)
                consequences = np.tile(np.asarray(classes), len(feature_ids))
                one_vs_rest = pl.DataFrame(
                    {
                        "arm": np.repeat(arm, welch_p.size),
                        "report_block": np.repeat(report_block, welch_p.size),
                        "orientation": np.repeat(orientation, welch_p.size),
                        "response": np.repeat(response, welch_p.size),
                        "feature_id": features,
                        "feature_support": np.repeat(
                            support[eligible], EXPECTED_CLASSES
                        ),
                        "consequence": consequences,
                        "positive_rows": np.repeat(EXPECTED_PER_CLASS, welch_p.size),
                        "prevalence": np.repeat(CHANCE_AUPRC, welch_p.size),
                        "mean_difference": mean_difference.reshape(-1),
                        "standardized_mean_difference": standardized.reshape(-1),
                        "welch_pvalue": welch_p.reshape(-1),
                        "welch_qvalue": welch_q.reshape(-1),
                        "rank_biserial": rank_biserial.reshape(-1),
                        "mwu_pvalue": mwu_p.reshape(-1),
                        "mwu_qvalue": mwu_q.reshape(-1),
                        "direction_aligned_auprc": auprc.reshape(-1),
                    }
                )
                one_name = f"{arm}__{orientation}__{response}__one_vs_rest.parquet"
                one_path = family_dir / one_name
                one_vs_rest.write_parquet(one_path, compression="zstd")
                artifacts[str(one_path.relative_to(output_dir))] = {
                    "bytes": one_path.stat().st_size,
                    "sha256": sha256_file(one_path),
                    "rows": one_vs_rest.height,
                }

                welch_omnibus_p, welch_f, welch_valid = welch_omnibus(matrix, indicator)
                kruskal_p, kruskal_h, kruskal_valid = kruskal_omnibus(
                    rank_sum, tie_terms
                )
                welch_omnibus_q = benjamini_hochberg(welch_omnibus_p)
                kruskal_q = benjamini_hochberg(kruskal_p)
                total_hypotheses += 2 * len(feature_ids)
                omnibus = pl.DataFrame(
                    {
                        "arm": np.repeat(arm, len(feature_ids)),
                        "report_block": np.repeat(report_block, len(feature_ids)),
                        "orientation": np.repeat(orientation, len(feature_ids)),
                        "response": np.repeat(response, len(feature_ids)),
                        "feature_id": feature_ids,
                        "feature_support": support[eligible],
                        "welch_anova_valid": welch_valid,
                        "welch_anova_f": welch_f,
                        "welch_anova_pvalue": welch_omnibus_p,
                        "welch_anova_qvalue": welch_omnibus_q,
                        "kruskal_valid": kruskal_valid,
                        "kruskal_h": kruskal_h,
                        "kruskal_pvalue": kruskal_p,
                        "kruskal_qvalue": kruskal_q,
                    }
                )
                omnibus_name = f"{arm}__{orientation}__{response}__omnibus.parquet"
                omnibus_path = family_dir / omnibus_name
                omnibus.write_parquet(omnibus_path, compression="zstd")
                artifacts[str(omnibus_path.relative_to(output_dir))] = {
                    "bytes": omnibus_path.stat().st_size,
                    "sha256": sha256_file(omnibus_path),
                    "rows": omnibus.height,
                }

                summaries.append(
                    {
                        "arm": arm,
                        "report_block": report_block,
                        "orientation": orientation,
                        "response": response,
                        "eligible_features": len(feature_ids),
                        "ineligible_features": D_SAE - len(feature_ids),
                        "one_vs_rest_pairs": welch_p.size,
                        "welch_q_lt_0_05": int(np.count_nonzero(welch_q < 0.05)),
                        "mwu_q_lt_0_05": int(np.count_nonzero(mwu_q < 0.05)),
                        "concordant_q_lt_0_05": int(
                            np.count_nonzero((welch_q < 0.05) & (mwu_q < 0.05))
                        ),
                        "best_auprc": float(auprc.max()),
                        "welch_omnibus_valid_features": int(
                            np.count_nonzero(welch_valid)
                        ),
                        "welch_omnibus_q_lt_0_05": int(
                            np.count_nonzero(welch_omnibus_q < 0.05)
                        ),
                        "kruskal_q_lt_0_05": int(np.count_nonzero(kruskal_q < 0.05)),
                    }
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
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": os.environ.get("RUN_ID", ""),
        "analysis_commit": analysis_commit,
        "elapsed_seconds": time.monotonic() - started,
        "panel": {
            "path": str(panel_path),
            "rows": EXPECTED_ROWS,
            "classes": EXPECTED_CLASSES,
            "rows_per_class": EXPECTED_PER_CLASS,
            "sha256": sha256_file(panel_path),
        },
        "extraction": {
            "manifest_sha256": sha256_file(extraction_manifest_path),
            "experiment_commit": extraction_manifest["experiment_commit"],
        },
        "protocol": {
            "layers": [1, 10, 19],
            "orientations": list(ORIENTATIONS),
            "responses": {
                "primary": "abs(alt_activation - ref_activation)",
                "secondary": "alt_activation - ref_activation",
            },
            "minimum_global_nonzero_support": MIN_SUPPORT,
            "one_vs_rest": [
                "Welch t-test + pooled standardized mean difference",
                "Mann-Whitney U without continuity correction + rank-biserial effect",
                "direction-aligned tie-aware AUPRC (descriptive)",
            ],
            "omnibus": [
                "Welch one-way ANOVA where every class variance is estimable; invalid sparse features retained with p=1",
                "tie-corrected Kruskal-Wallis",
            ],
            "multiple_testing": "separate BH over every eligible feature x 35 consequences for Welch and MW, and over every eligible feature for each omnibus statistic, within layer x orientation x response",
            "selection": "none; every eligible one-vs-rest and omnibus result is archived",
            "population": "all frozen 17,920 variants; historical splits ignored",
            "chance_auprc": CHANCE_AUPRC,
        },
        "hypotheses": total_hypotheses,
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
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--extraction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-commit", required=True)
    args = parser.parse_args()
    result = analyze(
        panel_path=args.panel,
        extraction_root=args.extraction_root,
        extraction_manifest_path=args.extraction_manifest,
        output_dir=args.output_dir,
        analysis_commit=args.analysis_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
