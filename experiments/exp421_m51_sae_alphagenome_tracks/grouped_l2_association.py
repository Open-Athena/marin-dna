"""Complete-family SAE associations with grouped AlphaGenome L2 outcomes.

This is the registered second pass for issue #421.  It uses every eligible
feature and all Mendelian variants, keeps layers and orientations separate,
and corrects the complete feature-by-outcome family at each resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import sparse, stats

ISSUE = 421
D_SAE = 15_360
EXPECTED_ROWS = 16_140
EXPECTED_TRACKS = 4_430
EXPECTED_TARGETS = {
    "overall": 1,
    "assay": 7,
    "biosample": 714,
    "assay_biosample": 1_411,
}
ARMS = ("block01-25m", "block10-25m", "block19-25m")
ORIENTATIONS = ("forward", "reverse_complement")
KEYS = ("chrom", "pos", "ref", "alt")
MIN_SUPPORT = 32
TOP_ROWS_PER_TARGET = 50
DETAILED_EFFECT_THRESHOLD = 0.05


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
    """Return exact BH q-values for one complete, finite hypothesis family."""

    flat = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    assert len(flat) > 0
    assert np.isfinite(flat).all() and ((0 <= flat) & (flat <= 1)).all()
    order = np.argsort(flat, kind="stable")
    adjusted = flat[order] * len(flat) / np.arange(1, len(flat) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output.reshape(pvalues.shape)


def correlation_pvalues(correlation: np.ndarray, rows: int) -> np.ndarray:
    """Two-sided correlation p-values using the usual t approximation."""

    assert rows >= 3 and np.isfinite(correlation).all()
    squared = np.minimum(np.asarray(correlation, dtype=np.float64) ** 2, 1.0)
    denominator = np.maximum(1.0 - squared, np.finfo(np.float64).tiny)
    statistic = np.abs(correlation) * np.sqrt((rows - 2) / denominator)
    output = 2.0 * stats.t.sf(statistic, df=rows - 2)
    output[squared >= 1.0] = 0.0
    assert np.isfinite(output).all() and ((0 <= output) & (output <= 1)).all()
    return output


def load_abs_delta(path: Path, *, rows: int) -> sparse.csr_matrix:
    frame = pl.read_parquet(path, columns=["panel_row", "feature_id", "delta"])
    assert frame.null_count().sum_horizontal().sum() == 0
    panel_row = frame["panel_row"].to_numpy().astype(np.int32, copy=False)
    feature_id = frame["feature_id"].to_numpy().astype(np.int32, copy=False)
    values = np.abs(frame["delta"].to_numpy()).astype(np.float32, copy=False)
    assert (panel_row >= 0).all() and (panel_row < rows).all()
    assert (feature_id >= 0).all() and (feature_id < D_SAE).all()
    assert np.isfinite(values).all()
    keep = values > 0
    matrix = sparse.coo_matrix(
        (values[keep], (panel_row[keep], feature_id[keep])),
        shape=(rows, D_SAE),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    assert matrix.shape == (rows, D_SAE)
    assert matrix.nnz > 0 and (matrix.data > 0).all()
    return matrix


def sparse_pearson(
    matrix: sparse.spmatrix, outcomes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlations and centered sums of squares for sparse features."""

    matrix = matrix.tocsc()
    assert matrix.shape[0] == outcomes.shape[0]
    assert np.isfinite(matrix.data).all() and np.isfinite(outcomes).all()
    rows = matrix.shape[0]
    y = outcomes.astype(np.float64, copy=False)
    y_centered = y - y.mean(axis=0, dtype=np.float64)
    y_ss = np.einsum("ij,ij->j", y_centered, y_centered)
    x_sum = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    squared = matrix.copy()
    squared.data = squared.data.astype(np.float64) ** 2
    x_ss = np.asarray(squared.sum(axis=0)).ravel() - x_sum * x_sum / rows
    numerator = np.asarray(matrix.T @ y_centered, dtype=np.float64)
    denominator = np.sqrt(np.maximum(x_ss[:, None] * y_ss[None, :], 0.0))
    correlation = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    correlation = np.clip(correlation, -1.0, 1.0)
    assert np.isfinite(correlation).all()
    return correlation, x_ss


def positive_sparse_rank_deviations(
    matrix: sparse.spmatrix,
) -> tuple[sparse.csc_matrix, np.ndarray]:
    """Encode tied ranks as sparse deviations from each feature's zero rank.

    All explicit values must be positive.  This makes the implicit zero group
    the lowest tied group, so Spearman covariance can be evaluated with a
    sparse-dense product after the outcome ranks are centered.
    """

    matrix = matrix.tocsc()
    rows, columns = matrix.shape
    assert matrix.nnz > 0 and np.isfinite(matrix.data).all()
    assert (matrix.data > 0).all()
    data = np.empty(matrix.nnz, dtype=np.float32)
    centered_ss = np.empty(columns, dtype=np.float64)
    mean_rank = (rows + 1.0) / 2.0
    for column in range(columns):
        start, stop = matrix.indptr[column : column + 2]
        values = matrix.data[start:stop]
        zero_count = rows - len(values)
        zero_rank = (zero_count + 1.0) / 2.0
        nonzero_ranks = stats.rankdata(values, method="average") + zero_count
        data[start:stop] = (nonzero_ranks - zero_rank).astype(np.float32)
        centered_ss[column] = zero_count * (zero_rank - mean_rank) ** 2 + np.sum(
            (nonzero_ranks - mean_rank) ** 2
        )
    deviations = sparse.csc_matrix(
        (data, matrix.indices.copy(), matrix.indptr.copy()), shape=matrix.shape
    )
    assert np.isfinite(deviations.data).all() and (centered_ss >= 0).all()
    return deviations, centered_ss


def sparse_rank_correlation(
    deviations: sparse.csc_matrix,
    feature_rank_ss: np.ndarray,
    outcomes: np.ndarray,
) -> np.ndarray:
    """Spearman correlations from sparse feature-rank deviations."""

    assert deviations.shape[0] == outcomes.shape[0]
    ranked = np.empty(outcomes.shape, dtype=np.float32)
    for column in range(outcomes.shape[1]):
        ranked[:, column] = stats.rankdata(
            outcomes[:, column], method="average"
        ).astype(np.float32)
    ranked -= ranked.mean(axis=0, dtype=np.float64).astype(np.float32)
    outcome_ss = np.einsum(
        "ij,ij->j", ranked.astype(np.float64), ranked.astype(np.float64)
    )
    numerator = np.asarray(deviations.T @ ranked, dtype=np.float64)
    denominator = np.sqrt(
        np.maximum(feature_rank_ss[:, None] * outcome_ss[None, :], 0.0)
    )
    correlation = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    correlation = np.clip(correlation, -1.0, 1.0)
    assert np.isfinite(correlation).all()
    return correlation


def _max_groups(
    tracks: np.ndarray,
    mapping: pl.DataFrame,
    *,
    resolution: str,
) -> tuple[np.ndarray, pl.DataFrame]:
    assert tracks.shape == (EXPECTED_ROWS, EXPECTED_TRACKS)
    mapping = mapping.with_row_index("track_index")
    if resolution == "overall":
        records = [
            {
                "target_id": "all_tracks",
                "target_name": "maximum across all AlphaGenome L2 tracks",
                "assay": None,
                "canonical_biosample_id": None,
                "canonical_biosample_name": None,
                "canonical_biosample_type": None,
                "track_count": EXPECTED_TRACKS,
            }
        ]
        values = tracks.max(axis=1, keepdims=True)
    else:
        if resolution == "assay":
            group_columns = ["assay"]
        elif resolution == "biosample":
            group_columns = ["canonical_biosample_id"]
        else:
            assert resolution == "assay_biosample"
            group_columns = ["assay", "canonical_biosample_id"]
        grouped = mapping.group_by(group_columns, maintain_order=False).agg(
            pl.col("track_index").sort(),
            pl.len().alias("track_count"),
            pl.col("canonical_biosample_name").first(),
            pl.col("canonical_biosample_type").first(),
        )
        grouped = grouped.sort(group_columns)
        columns: list[np.ndarray] = []
        records = []
        for row in grouped.iter_rows(named=True):
            indices = np.asarray(row["track_index"], dtype=np.int32)
            columns.append(tracks[:, indices].max(axis=1))
            assay = row.get("assay")
            biosample_id = row.get("canonical_biosample_id")
            if resolution == "assay":
                target_id = str(assay)
                target_name = str(assay)
            elif resolution == "biosample":
                target_id = str(biosample_id)
                target_name = str(row["canonical_biosample_name"])
            else:
                target_id = f"{assay}|{biosample_id}"
                target_name = f"{assay} | {row['canonical_biosample_name']}"
            records.append(
                {
                    "target_id": target_id,
                    "target_name": target_name,
                    "assay": assay,
                    "canonical_biosample_id": biosample_id,
                    "canonical_biosample_name": row.get("canonical_biosample_name"),
                    "canonical_biosample_type": row.get("canonical_biosample_type"),
                    "track_count": int(row["track_count"]),
                }
            )
        values = np.column_stack(columns)
    catalog = pl.DataFrame(records, infer_schema_length=None).with_row_index(
        "target_index"
    )
    assert values.shape == (EXPECTED_ROWS, EXPECTED_TARGETS[resolution])
    assert catalog.height == values.shape[1]
    assert np.isfinite(values).all() and (values >= 0).all()
    assert catalog["track_count"].sum() == EXPECTED_TRACKS
    return values.astype(np.float32, copy=False), catalog


def load_grouped_outcomes(
    *, panel: pl.DataFrame, alphagenome_uri: str, mapping: pl.DataFrame
) -> dict[str, tuple[np.ndarray, pl.DataFrame]]:
    """Align the score table to the panel and form the four frozen max views."""

    assert panel.height == EXPECTED_ROWS
    assert mapping.height == mapping["track_id"].n_unique() == EXPECTED_TRACKS
    track_ids = mapping["track_id"].to_list()
    score = pl.read_parquet(alphagenome_uri)
    assert score.height == EXPECTED_ROWS
    assert set(track_ids) <= set(score.columns)
    assert score.select(pl.struct(KEYS).n_unique()).item() == EXPECTED_ROWS
    indexed = panel.with_row_index("panel_row")
    metadata = [*KEYS, "label", "subset", "match_group"]
    aligned = (
        indexed.select(["panel_row", *metadata])
        .join(
            score.select([*metadata, *track_ids]),
            on=list(KEYS),
            how="inner",
            suffix="_score",
            validate="1:1",
        )
        .sort("panel_row")
    )
    assert aligned.height == EXPECTED_ROWS
    assert aligned["panel_row"].to_list() == list(range(EXPECTED_ROWS))
    assert (
        aligned["label"].cast(pl.Boolean) == aligned["label_score"].cast(pl.Boolean)
    ).all()
    assert (aligned["subset"] == aligned["subset_score"]).all()
    assert (aligned["match_group"] == aligned["match_group_score"]).all()
    tracks = aligned.select(track_ids).to_numpy().astype(np.float32, copy=False)
    assert tracks.shape == (EXPECTED_ROWS, EXPECTED_TRACKS)
    assert np.isfinite(tracks).all() and (tracks >= 0).all()
    return {
        resolution: _max_groups(tracks, mapping, resolution=resolution)
        for resolution in EXPECTED_TARGETS
    }


def family_tables(
    *,
    correlation: np.ndarray,
    feature_ids: np.ndarray,
    support: np.ndarray,
    target_catalog: pl.DataFrame,
    family: dict[str, str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Correct a complete family, summarize all tests, and retain interpretable rows."""

    assert correlation.shape == (len(feature_ids), target_catalog.height)
    pvalues = correlation_pvalues(correlation, EXPECTED_ROWS)
    qvalues = benjamini_hochberg(pvalues)
    detailed: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    metadata = target_catalog.to_dicts()
    for target_index, target in enumerate(metadata):
        effects = correlation[:, target_index]
        q = qvalues[:, target_index]
        p = pvalues[:, target_index]
        order = np.lexsort((feature_ids, -np.abs(effects)))
        top = order[: min(TOP_ROWS_PER_TARGET, len(order))]
        material = np.flatnonzero(
            (q < 0.05) & (np.abs(effects) >= DETAILED_EFFECT_THRESHOLD)
        )
        selected = np.unique(np.concatenate((top, material)))
        for local in selected:
            detailed.append(
                {
                    **family,
                    **target,
                    "feature_id": int(feature_ids[local]),
                    "feature_support": int(support[local]),
                    "effect": float(effects[local]),
                    "pvalue": float(p[local]),
                    "qvalue": float(q[local]),
                    "q_lt_0_05": bool(q[local] < 0.05),
                    "abs_effect_ge_0_05": bool(
                        abs(effects[local]) >= DETAILED_EFFECT_THRESHOLD
                    ),
                    "absolute_effect_rank": int(
                        np.flatnonzero(order == local).item() + 1
                    ),
                }
            )
        winner = int(order[0])
        summaries.append(
            {
                **family,
                **target,
                "eligible_features": len(feature_ids),
                "hypotheses": len(feature_ids),
                "q_lt_0_05": int(np.count_nonzero(q < 0.05)),
                "q_lt_0_05_and_abs_effect_ge_0_05": int(
                    np.count_nonzero(
                        (q < 0.05) & (np.abs(effects) >= DETAILED_EFFECT_THRESHOLD)
                    )
                ),
                "max_abs_effect": float(abs(effects[winner])),
                "winning_feature_id": int(feature_ids[winner]),
                "winning_effect": float(effects[winner]),
                "winning_pvalue": float(p[winner]),
                "winning_qvalue": float(q[winner]),
                "minimum_qvalue": float(q.min()),
            }
        )
    return (
        pl.DataFrame(detailed, infer_schema_length=None),
        pl.DataFrame(summaries, infer_schema_length=None),
    )


def analyze(
    *,
    panel_path: Path,
    activation_root: Path,
    extraction_manifest_path: Path,
    alphagenome_uri: str,
    track_mapping_path: Path,
    mapping_manifest_path: Path,
    output_dir: Path,
    analysis_commit: str,
) -> dict[str, Any]:
    assert len(analysis_commit) == 40 and all(
        c in "0123456789abcdef" for c in analysis_commit
    )
    assert panel_path.is_file() and activation_root.is_dir()
    assert extraction_manifest_path.is_file()
    assert track_mapping_path.is_file() and mapping_manifest_path.is_file()
    assert not output_dir.exists()
    started = time.monotonic()

    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    assert extraction_manifest["panel"]["sha256"] == sha256_file(panel_path)
    assert extraction_manifest["panel"]["rows"] == EXPECTED_ROWS
    mapping_manifest = json.loads(mapping_manifest_path.read_text())
    mapping_artifacts = mapping_manifest["artifacts"]
    mapping_relative = next(
        name for name in mapping_artifacts if name.endswith("track_mapping.parquet")
    )
    assert mapping_artifacts[mapping_relative]["sha256"] == sha256_file(
        track_mapping_path
    )

    panel = pl.read_parquet(panel_path)
    assert panel.height == EXPECTED_ROWS
    mapping = pl.read_parquet(track_mapping_path).sort("track_id")
    outcomes = load_grouped_outcomes(
        panel=panel, alphagenome_uri=alphagenome_uri, mapping=mapping
    )

    output_dir.mkdir(parents=True)
    table_dir = output_dir / "families"
    table_dir.mkdir()
    catalogs = []
    for resolution, (_, catalog) in outcomes.items():
        catalogs.append(catalog.with_columns(pl.lit(resolution).alias("resolution")))
    pl.concat(catalogs, how="diagonal_relaxed").write_parquet(
        output_dir / "target_catalog.parquet", compression="zstd"
    )

    all_summaries: list[pl.DataFrame] = []
    artifacts: dict[str, dict[str, Any]] = {}
    family_count = 0
    hypothesis_count = 0
    for arm in ARMS:
        report_block = int(arm.removeprefix("block").split("-")[0])
        for orientation in ORIENTATIONS:
            relative = f"{arm}/sae_focal_{orientation}.parquet"
            activation_path = activation_root / relative
            assert activation_path.is_file(), activation_path
            expected_artifact = extraction_manifest["artifacts"][relative]
            assert expected_artifact["bytes"] == activation_path.stat().st_size
            assert expected_artifact["sha256"] == sha256_file(activation_path)
            matrix = load_abs_delta(activation_path, rows=EXPECTED_ROWS)
            feature_support = np.asarray(matrix.getnnz(axis=0)).ravel()
            eligible = feature_support >= MIN_SUPPORT
            feature_ids = np.flatnonzero(eligible).astype(np.int32)
            assert len(feature_ids) > 0
            raw = matrix[:, eligible].tocsc()
            log = raw.copy()
            log.data = np.log1p(log.data).astype(np.float32)
            rank_deviations, feature_rank_ss = positive_sparse_rank_deviations(raw)

            for resolution, (target_values, target_catalog) in outcomes.items():
                computations = []
                raw_r, _ = sparse_pearson(raw, target_values)
                computations.append(("pearson_abs_delta", raw_r))
                log_r, _ = sparse_pearson(log, target_values)
                computations.append(("pearson_log1p_abs_delta", log_r))
                rho = sparse_rank_correlation(
                    rank_deviations, feature_rank_ss, target_values
                )
                computations.append(("spearman_abs_delta", rho))
                for statistic, correlation in computations:
                    family = {
                        "arm": arm,
                        "report_block": str(report_block),
                        "orientation": orientation,
                        "resolution": resolution,
                        "statistic": statistic,
                    }
                    detailed, summary = family_tables(
                        correlation=correlation,
                        feature_ids=feature_ids,
                        support=feature_support[eligible],
                        target_catalog=target_catalog,
                        family=family,
                    )
                    filename = (
                        f"{arm}__{orientation}__{resolution}__{statistic}.parquet"
                    )
                    path = table_dir / filename
                    detailed.write_parquet(path, compression="zstd")
                    artifacts[str(path.relative_to(output_dir))] = {
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "rows": detailed.height,
                    }
                    all_summaries.append(summary)
                    family_count += 1
                    hypothesis_count += correlation.size
                    print(
                        json.dumps(
                            {
                                "stage": "family_complete",
                                **family,
                                "eligible_features": len(feature_ids),
                                "hypotheses": correlation.size,
                            }
                        ),
                        flush=True,
                    )

    summary = pl.concat(all_summaries, how="diagonal_relaxed")
    assert summary.height > 0
    summary_path = output_dir / "family_summary.parquet"
    summary.write_parquet(summary_path, compression="zstd")
    target_path = output_dir / "target_catalog.parquet"
    for path in (summary_path, target_path):
        artifacts[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": pl.scan_parquet(path).select(pl.len()).collect().item(),
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
            "sha256": sha256_file(panel_path),
        },
        "extraction": {
            "manifest_sha256": sha256_file(extraction_manifest_path),
            "experiment_commit": extraction_manifest["experiment_commit"],
        },
        "alphagenome": {
            "uri": alphagenome_uri,
            "score": "exported L2_DIFF_LOG1P; no second outcome transform",
            "tracks": EXPECTED_TRACKS,
            "mapping_manifest_sha256": sha256_file(mapping_manifest_path),
            "target_counts": EXPECTED_TARGETS,
        },
        "protocol": {
            "layers": [1, 10, 19],
            "orientations": list(ORIENTATIONS),
            "primary_feature_response": "abs(alt_activation - ref_activation)",
            "scale_sensitivity": "log1p(abs(alt_activation - ref_activation))",
            "minimum_global_nonzero_support": MIN_SUPPORT,
            "statistics": [
                "Pearson on raw abs(delta)",
                "Pearson on log1p(abs(delta))",
                "Spearman (identical under the monotone raw/log1p feature transform)",
            ],
            "multiple_testing": "BH across every eligible feature x outcome within layer x resolution x orientation x statistic",
            "selection": "none before inference; detailed artifacts retain top 50 per outcome plus q<0.05 and abs(effect)>=0.05 rows",
            "population": "all 16,140 Mendelian variants; no label/subset/split selection",
        },
        "families": family_count,
        "hypotheses": hypothesis_count,
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
    parser.add_argument("--activation-root", type=Path, required=True)
    parser.add_argument("--extraction-manifest", type=Path, required=True)
    parser.add_argument("--alphagenome-uri", required=True)
    parser.add_argument("--track-mapping", type=Path, required=True)
    parser.add_argument("--mapping-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-commit", required=True)
    args = parser.parse_args()
    manifest = analyze(
        panel_path=args.panel,
        activation_root=args.activation_root,
        extraction_manifest_path=args.extraction_manifest,
        alphagenome_uri=args.alphagenome_uri,
        track_mapping_path=args.track_mapping,
        mapping_manifest_path=args.mapping_manifest,
        output_dir=args.output_dir,
        analysis_commit=args.analysis_commit,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
