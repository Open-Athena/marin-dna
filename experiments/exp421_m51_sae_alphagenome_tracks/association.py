"""Globally associate absolute SAE variant responses with AlphaGenome tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl
from scipy import sparse, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ISSUE = 421
D_SAE = 15_360
ORIENTATIONS = ("forward", "reverse_complement")
KEYS = ["chrom", "pos", "ref", "alt"]
PANEL_METADATA = [*KEYS, "label", "subset", "match_group", "split"]
ASSAYS = (
    "ATAC",
    "DNASE",
    "CHIP_TF",
    "CHIP_HISTONE",
    "CAGE",
    "PROCAP",
    "RNA_SEQ",
)
EXPECTED_COUNTS = {
    "ATAC": 167,
    "DNASE": 305,
    "CHIP_TF": 1_617,
    "CHIP_HISTONE": 1_116,
    "CAGE": 546,
    "PROCAP": 12,
    "RNA_SEQ": 667,
}
MIN_NONZERO_PER_SPLIT = 32
DISCOVERY_PER_ASSAY = 64
VALIDATED_PER_ASSAY = 10
TRACK_CHUNK = 256
BOOTSTRAPS = 1_000
SEED = 421_1


def seed(*parts: Any) -> int:
    value = "|".join(str(part) for part in (SEED, *parts))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    assert x.ndim == y.ndim == 1 and x.shape == y.shape
    assert np.isfinite(x).all() and np.isfinite(y).all()
    x_centered = x.astype(np.float64) - x.mean(dtype=np.float64)
    y_centered = y.astype(np.float64) - y.mean(dtype=np.float64)
    denominator = np.sqrt(
        np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)
    )
    if denominator == 0:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denominator)


def pearson_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    assert x.ndim == y.ndim == 1 and x.shape == y.shape
    assert np.isfinite(x).all() and np.isfinite(y).all()
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return 1.0
    pvalue = float(stats.pearsonr(x, y).pvalue)
    assert np.isfinite(pvalue)
    return pvalue


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return rank correlation and p-value, with constants mapped to the null."""

    assert x.ndim == y.ndim == 1 and x.shape == y.shape
    assert np.isfinite(x).all() and np.isfinite(y).all()
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0, 1.0
    result = stats.spearmanr(x, y)
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    assert np.isfinite(rho) and np.isfinite(pvalue)
    return rho, pvalue


def group_center(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    assert values.ndim == groups.ndim == 1 and values.shape == groups.shape
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    means = np.bincount(inverse, weights=values) / counts
    output = values - means[inverse]
    assert np.isfinite(output).all()
    return output


def group_bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    random_seed: int,
    samples: int = BOOTSTRAPS,
) -> tuple[float, float]:
    """Bootstrap correlation using per-group sufficient statistics."""

    assert x.shape == y.shape == groups.shape
    _, inverse = np.unique(groups, return_inverse=True)
    group_count = int(inverse.max()) + 1
    counts = np.bincount(inverse).astype(np.float64)
    sx = np.bincount(inverse, weights=x)
    sy = np.bincount(inverse, weights=y)
    sx2 = np.bincount(inverse, weights=x * x)
    sy2 = np.bincount(inverse, weights=y * y)
    sxy = np.bincount(inverse, weights=x * y)
    rng = np.random.default_rng(random_seed)
    draws = rng.integers(0, group_count, size=(samples, group_count))
    n = counts[draws].sum(axis=1)
    x_sum = sx[draws].sum(axis=1)
    y_sum = sy[draws].sum(axis=1)
    numerator = sxy[draws].sum(axis=1) - x_sum * y_sum / n
    x_ss = sx2[draws].sum(axis=1) - x_sum * x_sum / n
    y_ss = sy2[draws].sum(axis=1) - y_sum * y_sum / n
    correlations = numerator / np.sqrt(x_ss * y_ss)
    assert np.isfinite(correlations).all()
    low, high = np.quantile(correlations, [0.025, 0.975])
    return float(low), float(high)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    assert pvalues.ndim == 1 and len(pvalues) > 0
    assert np.isfinite(pvalues).all() and ((0 <= pvalues) & (pvalues <= 1)).all()
    order = np.argsort(pvalues, kind="stable")
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(ranked)
    output[order] = np.minimum(ranked, 1)
    return output


def load_feature_matrix(path: Path, *, rows: int) -> sparse.csr_matrix:
    frame = pl.read_parquet(path, columns=["row_index", "feature_id", "delta"])
    assert frame.null_count().sum_horizontal().sum() == 0
    row_index = frame["row_index"].to_numpy().astype(np.int32, copy=False)
    feature_id = frame["feature_id"].to_numpy().astype(np.int32, copy=False)
    signed = frame["delta"].to_numpy().astype(np.float32, copy=False)
    values = np.log1p(np.abs(signed)).astype(np.float32, copy=False)
    keep = values > 0
    matrix = sparse.coo_matrix(
        (values[keep], (row_index[keep], feature_id[keep])),
        shape=(rows, D_SAE),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    assert matrix.shape == (rows, D_SAE) and matrix.nnz > rows
    assert np.isfinite(matrix.data).all() and (matrix.data > 0).all()
    return matrix


def _track_assay(track_id: str) -> str:
    for assay in ASSAYS:
        if track_id.startswith(f"{assay}_"):
            return assay
    raise AssertionError(track_id)


def align_alphagenome(
    panel: pl.DataFrame,
    *,
    uri: str,
    track_ids: list[str],
) -> np.ndarray:
    alpha = pl.read_parquet(uri)
    assert alpha.height == panel.height
    assert set(track_ids) <= set(alpha.columns)
    assert len(track_ids) == len(set(track_ids)) == 4_430
    assert alpha.select(pl.struct(KEYS).n_unique()).item() == alpha.height
    indexed = panel.select(PANEL_METADATA).with_row_index("row_index")
    aligned = indexed.join(
        alpha, on=KEYS, how="inner", suffix="_ag", validate="1:1"
    ).sort("row_index")
    assert aligned.height == panel.height
    assert aligned["row_index"].to_list() == list(range(panel.height))
    assert (aligned["label"].cast(pl.Boolean) == aligned["label_ag"]).all()
    assert (aligned["subset"] == aligned["subset_ag"]).all()
    assert (aligned["match_group"] == aligned["match_group_ag"]).all()
    tracks = aligned.select(track_ids)
    assert tracks.null_count().sum_horizontal().sum() == 0
    matrix = tracks.to_numpy().astype(np.float32, copy=False)
    assert matrix.shape == (panel.height, 4_430)
    assert np.isfinite(matrix).all() and (matrix >= 0).all()
    return matrix


def feature_support(
    matrix: sparse.csr_matrix, splits: np.ndarray
) -> dict[str, np.ndarray]:
    output = {
        split: np.asarray(matrix[splits == split].getnnz(axis=0)).ravel()
        for split in ("discovery", "validation", "test")
    }
    assert all(values.shape == (D_SAE,) for values in output.values())
    return output


def screen_discovery(
    matrix: sparse.csr_matrix,
    tracks: np.ndarray,
    track_ids: list[str],
    rows: np.ndarray,
    support: dict[str, np.ndarray],
    *,
    orientation: str,
) -> list[dict[str, Any]]:
    """Keep the top discovery correlations within each assay."""

    x = matrix[rows]
    x_mean = np.asarray(x.mean(axis=0)).ravel().astype(np.float64)
    x_second = np.asarray(x.multiply(x).mean(axis=0)).ravel().astype(np.float64)
    x_std = np.sqrt(np.maximum(x_second - x_mean * x_mean, 0))
    eligible = (
        (support["discovery"] >= MIN_NONZERO_PER_SPLIT)
        & (support["validation"] >= MIN_NONZERO_PER_SPLIT)
        & (support["test"] >= MIN_NONZERO_PER_SPLIT)
        & (x_std > 0)
    )
    assert eligible.sum() > 100
    candidates: list[dict[str, Any]] = []
    for assay in ASSAYS:
        assay_indices = np.asarray(
            [
                index
                for index, track_id in enumerate(track_ids)
                if _track_assay(track_id) == assay
            ],
            dtype=np.int32,
        )
        assert len(assay_indices) == EXPECTED_COUNTS[assay]
        assay_candidates: list[dict[str, Any]] = []
        for offset in range(0, len(assay_indices), TRACK_CHUNK):
            indices = assay_indices[offset : offset + TRACK_CHUNK]
            y = tracks[np.ix_(rows, indices)].astype(np.float32, copy=True)
            y -= y.mean(axis=0, dtype=np.float64).astype(np.float32)
            y_std = y.std(axis=0, dtype=np.float64)
            valid_tracks = y_std > 0
            y[:, valid_tracks] /= y_std[valid_tracks].astype(np.float32)
            y[:, ~valid_tracks] = 0
            cross = np.asarray(x.T @ y, dtype=np.float64) / len(rows)
            correlation = np.divide(
                cross,
                x_std[:, None],
                out=np.zeros_like(cross),
                where=x_std[:, None] > 0,
            )
            correlation[~eligible] = 0
            flat = np.abs(correlation).ravel()
            keep = min(DISCOVERY_PER_ASSAY, int(np.count_nonzero(flat)))
            if keep == 0:
                continue
            selected = np.argpartition(flat, -keep)[-keep:]
            for flat_index in selected:
                feature_id, local_track = np.unravel_index(
                    flat_index, correlation.shape
                )
                global_track = int(indices[local_track])
                assay_candidates.append(
                    {
                        "orientation": orientation,
                        "assay": assay,
                        "feature_id": int(feature_id),
                        "track_index": global_track,
                        "track_id": track_ids[global_track],
                        "discovery_r": float(correlation[feature_id, local_track]),
                        "discovery_support": int(support["discovery"][feature_id]),
                        "validation_support": int(support["validation"][feature_id]),
                        "test_support": int(support["test"][feature_id]),
                    }
                )
        assay_candidates.sort(key=lambda row: -abs(row["discovery_r"]))
        unique: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for row in assay_candidates:
            key = (row["feature_id"], row["track_index"])
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
            if len(unique) == DISCOVERY_PER_ASSAY:
                break
        assert len(unique) == DISCOVERY_PER_ASSAY, (assay, len(unique))
        candidates.extend(unique)
    return candidates


def validate_candidates(
    candidates: list[dict[str, Any]],
    matrix: sparse.csr_matrix,
    tracks: np.ndarray,
    rows: np.ndarray,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for candidate in candidates:
        x = matrix[rows, candidate["feature_id"]].toarray().ravel()
        y = tracks[rows, candidate["track_index"]]
        validation_r = pearson(x, y)
        validation_rho, _ = spearman(x, y)
        enriched.append(
            {
                **candidate,
                "validation_r": validation_r,
                "validation_rho": validation_rho,
                "validation_direction_consistent": bool(
                    np.sign(validation_r) == np.sign(candidate["discovery_r"])
                ),
            }
        )
    selected: list[dict[str, Any]] = []
    for assay in ASSAYS:
        rows_for_assay = [
            row
            for row in enriched
            if row["assay"] == assay and row["validation_direction_consistent"]
        ]
        rows_for_assay.sort(key=lambda row: -abs(row["validation_r"]))
        assert len(rows_for_assay) >= VALIDATED_PER_ASSAY, assay
        selected.extend(rows_for_assay[:VALIDATED_PER_ASSAY])
    return selected


def test_candidates(
    selected: list[dict[str, Any]],
    matrix: sparse.csr_matrix,
    tracks: np.ndarray,
    rows: np.ndarray,
    groups: np.ndarray,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate in selected:
        x = matrix[rows, candidate["feature_id"]].toarray().ravel().astype(np.float64)
        y = tracks[rows, candidate["track_index"]].astype(np.float64)
        test_r = pearson(x, y)
        test_rho, spearman_pvalue = spearman(x, y)
        low, high = group_bootstrap_correlation(
            x,
            y,
            groups,
            random_seed=seed(
                candidate["orientation"], candidate["feature_id"], candidate["track_id"]
            ),
        )
        x_centered = group_center(x, groups)
        y_centered = group_center(y, groups)
        centered_r = pearson(x_centered, y_centered)
        centered_rho, _ = spearman(x_centered, y_centered)
        pvalue = pearson_pvalue(x, y)
        output.append(
            {
                **candidate,
                "test_r": test_r,
                "test_rho": test_rho,
                "test_r_ci95_low": low,
                "test_r_ci95_high": high,
                "test_group_centered_r": centered_r,
                "test_group_centered_rho": centered_rho,
                "test_pvalue": pvalue,
                "test_spearman_pvalue": spearman_pvalue,
            }
        )
    qvalues = benjamini_hochberg(
        np.asarray([row["test_pvalue"] for row in output], dtype=np.float64)
    )
    spearman_qvalues = benjamini_hochberg(
        np.asarray([row["test_spearman_pvalue"] for row in output], dtype=np.float64)
    )
    for row, qvalue, spearman_qvalue in zip(
        output, qvalues, spearman_qvalues, strict=True
    ):
        row["test_qvalue"] = float(qvalue)
        row["test_spearman_qvalue"] = float(spearman_qvalue)
        row["test_discovery"] = bool(qvalue < 0.05 and abs(row["test_r"]) >= 0.1)
        row["spearman_confirmation"] = bool(
            spearman_qvalue < 0.05
            and abs(row["test_rho"]) >= 0.1
            and np.sign(row["test_rho"]) == np.sign(row["test_r"])
        )
    return output


def top_contexts(
    tested: list[dict[str, Any]],
    matrices: dict[str, sparse.csr_matrix],
    tracks: np.ndarray,
    panel: pl.DataFrame,
    test_rows: np.ndarray,
    *,
    pairs: int = 20,
    examples: int = 3,
) -> pl.DataFrame:
    output: list[dict[str, Any]] = []
    ranked = sorted(tested, key=lambda row: -abs(row["test_r"]))[:pairs]
    for pair_rank, row in enumerate(ranked, start=1):
        x = matrices[row["orientation"]][test_rows, row["feature_id"]].toarray().ravel()
        y = tracks[test_rows, row["track_index"]]
        joint = x * y
        order = np.argsort(-joint, kind="stable")[:examples]
        for example_rank, local_row in enumerate(order, start=1):
            global_row = int(test_rows[local_row])
            metadata = panel.row(global_row, named=True)
            output.append(
                {
                    "pair_rank": pair_rank,
                    "example_rank": example_rank,
                    "orientation": row["orientation"],
                    "feature_id": row["feature_id"],
                    "track_id": row["track_id"],
                    "feature_magnitude": float(x[local_row]),
                    "alphagenome_l2": float(y[local_row]),
                    **{column: metadata[column] for column in PANEL_METADATA},
                }
            )
    return pl.DataFrame(output)


def assay_summary(tested: pl.DataFrame) -> pl.DataFrame:
    return (
        tested.group_by("orientation", "assay")
        .agg(
            pl.len().alias("tested_pairs"),
            pl.col("test_discovery").sum().alias("discoveries"),
            pl.col("spearman_confirmation").sum().alias("spearman_confirmations"),
            pl.col("test_r").abs().max().alias("max_abs_test_r"),
            pl.col("test_rho").abs().max().alias("max_abs_test_rho"),
            pl.col("test_group_centered_r").abs().max().alias("max_abs_centered_r"),
            pl.col("test_r").abs().median().alias("median_abs_test_r"),
        )
        .sort("assay", "orientation")
    )


def plot_results(tested: pl.DataFrame, summary: pl.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    colors = {
        "ATAC": "#0072B2",
        "DNASE": "#56B4E9",
        "CHIP_TF": "#D55E00",
        "CHIP_HISTONE": "#CC79A7",
        "CAGE": "#009E73",
        "PROCAP": "#F0E442",
        "RNA_SEQ": "#000000",
    }
    for assay in ASSAYS:
        rows = tested.filter(pl.col("assay") == assay)
        axes[0].scatter(
            rows["validation_r"],
            rows["test_r"],
            s=22,
            alpha=0.7,
            color=colors[assay],
            label=assay,
        )
    axes[0].axhline(0, color="grey", linewidth=0.8)
    axes[0].axvline(0, color="grey", linewidth=0.8)
    axes[0].set_xlabel("Validation Pearson r")
    axes[0].set_ylabel("Held-out Pearson r")
    axes[0].set_title("Selected global feature–track pairs")
    axes[0].legend(fontsize=7, ncol=2)

    for assay in ASSAYS:
        rows = tested.filter(pl.col("assay") == assay)
        axes[1].scatter(
            rows["test_r"],
            rows["test_rho"],
            s=22,
            alpha=0.7,
            color=colors[assay],
        )
    bounds = (-1.0, 1.0)
    axes[1].plot(bounds, bounds, color="grey", linewidth=0.8, linestyle="--")
    axes[1].set_xlim(bounds)
    axes[1].set_ylim(bounds)
    axes[1].set_xlabel("Held-out Pearson r")
    axes[1].set_ylabel("Held-out Spearman rho")
    axes[1].set_title("Linear versus monotonic association")

    assays = list(ASSAYS)
    x = np.arange(len(assays))
    width = 0.36
    for offset, orientation in (
        (-width / 2, "forward"),
        (width / 2, "reverse_complement"),
    ):
        rows = summary.filter(pl.col("orientation") == orientation).sort(
            pl.col("assay").replace_strict(assays, list(range(len(assays))))
        )
        axes[2].bar(
            x + offset,
            rows["max_abs_test_r"],
            width,
            label="FWD" if orientation == "forward" else "RC",
        )
    axes[2].set_xticks(x, assays, rotation=45, ha="right")
    axes[2].set_ylabel("Maximum held-out |r|")
    axes[2].set_title("Strongest Pearson pair by assay")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("exp421: global |ΔSAE| ↔ AlphaGenome L2 associations")
    figure.savefig(output_dir / "association_overview.png", dpi=180)
    figure.savefig(output_dir / "association_overview.svg")
    plt.close(figure)


def markdown(tested: pl.DataFrame, summary: pl.DataFrame) -> str:
    lines = [
        "# exp421 global SAE–AlphaGenome associations",
        "",
        "Primary feature variable: `log1p(|activation_alt - activation_ref|)`. AlphaGenome values are the exported unsigned `L2_DIFF_LOG1P` scores and are not transformed again. All variants are pooled without label/subset stratification; group-centering is reported only as a sensitivity analysis.",
        "",
        "## Assay overview",
        "",
        "| orientation | assay | tested | Pearson discoveries | Spearman confirmations | max abs(r) | max abs(rho) | max centered abs(r) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dicts():
        lines.append(
            f"| {row['orientation']} | {row['assay']} | {row['tested_pairs']} | "
            f"{row['discoveries']} | {row['spearman_confirmations']} | "
            f"{row['max_abs_test_r']:.3f} | {row['max_abs_test_rho']:.3f} | "
            f"{row['max_abs_centered_r']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strongest held-out pairs",
            "",
            "Track names use the current AlphaGenome metadata endpoint. Counts match the score export exactly, but the original May 2026 metadata snapshot was not preserved.",
            "",
            "| orientation | feature | track | name | biosample | discovery r | validation r / rho | test r (95% group-bootstrap CI) | test rho | centered r / rho | Pearson q | Spearman q |",
            "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in tested.sort(pl.col("test_r").abs(), descending=True).head(25).to_dicts():
        name = str(row.get("name") or "").replace("|", "/")[:60]
        biosample = str(row.get("biosample_name") or "").replace("|", "/")[:40]
        lines.append(
            f"| {row['orientation']} | {row['feature_id']} | `{row['track_id']}` | "
            f"{name} | {biosample} | {row['discovery_r']:.3f} | "
            f"{row['validation_r']:.3f} / {row['validation_rho']:.3f} | "
            f"{row['test_r']:.3f} "
            f"[{row['test_r_ci95_low']:.3f}, {row['test_r_ci95_high']:.3f}] | "
            f"{row['test_rho']:.3f} | {row['test_group_centered_r']:.3f} / "
            f"{row['test_group_centered_rho']:.3f} | {row['test_qvalue']:.2g} | "
            f"{row['test_spearman_qvalue']:.2g} |"
        )
    lines.append("")
    return "\n".join(lines)


def analyze(
    *,
    panel_path: Path,
    extraction_dir: Path,
    alphagenome_uri: str,
    metadata_path: Path,
    metadata_manifest_path: Path,
    output_dir: Path,
    extraction_commit: str,
    analysis_commit: str,
) -> dict[str, Any]:
    assert panel_path.is_file() and extraction_dir.is_dir()
    assert metadata_path.is_file() and metadata_manifest_path.is_file()
    assert not output_dir.exists()
    extraction_manifest_path = extraction_dir / "manifest.json"
    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    assert extraction_manifest["experiment_commit"] == extraction_commit
    assert extraction_manifest["panel"]["sha256"] == sha256(panel_path)
    for name, metadata in extraction_manifest["artifacts"].items():
        path = extraction_dir / name
        assert path.stat().st_size == metadata["bytes"]
        assert sha256(path) == metadata["sha256"]
    metadata_manifest = json.loads(metadata_manifest_path.read_text())
    assert metadata_manifest["artifact"]["sha256"] == sha256(metadata_path)
    assert metadata_manifest["rows"] == 4_430

    panel = pl.read_parquet(panel_path)
    assert panel.height == 16_140
    assert panel.select(pl.struct(KEYS).n_unique()).item() == panel.height
    metadata = pl.read_parquet(metadata_path)
    assert metadata.height == metadata["track_id"].n_unique() == 4_430
    track_ids = metadata["track_id"].to_list()
    assert all(
        _track_assay(track_id) == assay
        for track_id, assay in zip(track_ids, metadata["assay"], strict=True)
    )
    tracks = align_alphagenome(panel, uri=alphagenome_uri, track_ids=track_ids)
    splits = panel["split"].to_numpy()
    discovery_rows = np.flatnonzero(splits == "discovery")
    validation_rows = np.flatnonzero(splits == "validation")
    test_rows = np.flatnonzero(splits == "test")
    groups = panel["match_group"].to_numpy()

    matrices: dict[str, sparse.csr_matrix] = {}
    discovery_rows_out: list[dict[str, Any]] = []
    tested_rows: list[dict[str, Any]] = []
    for orientation in ORIENTATIONS:
        matrix = load_feature_matrix(
            extraction_dir / f"sae_activations_{orientation}.parquet",
            rows=panel.height,
        )
        matrices[orientation] = matrix
        support = feature_support(matrix, splits)
        candidates = screen_discovery(
            matrix,
            tracks,
            track_ids,
            discovery_rows,
            support,
            orientation=orientation,
        )
        discovery_rows_out.extend(candidates)
        selected = validate_candidates(candidates, matrix, tracks, validation_rows)
        tested_rows.extend(
            test_candidates(
                selected,
                matrix,
                tracks,
                test_rows,
                groups[test_rows],
            )
        )

    tested = pl.DataFrame(tested_rows).join(
        metadata, on=["track_id", "assay"], how="left", validate="m:1"
    )
    assert tested.height == len(ORIENTATIONS) * len(ASSAYS) * VALIDATED_PER_ASSAY
    assert tested["name"].null_count() == 0
    discovery = pl.DataFrame(discovery_rows_out)
    summary = assay_summary(tested)
    contexts = top_contexts(
        tested_rows,
        matrices,
        tracks,
        panel,
        test_rows,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    tables = {
        "discovery_candidates": discovery,
        "tested_pairs": tested,
        "assay_summary": summary,
        "top_variants": contexts,
    }
    for name, table in tables.items():
        assert table.height > 0
        table.write_parquet(output_dir / f"{name}.parquet", compression="zstd")
    plot_results(tested, summary, output_dir)
    (output_dir / "RESULTS.md").write_text(markdown(tested, summary))

    track_column_hash = hashlib.sha256("\n".join(track_ids).encode()).hexdigest()
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "analysis_commit": analysis_commit,
        "extraction_commit": extraction_commit,
        "panel": {"rows": panel.height, "sha256": sha256(panel_path)},
        "alphagenome": {
            "uri": alphagenome_uri,
            "rows": panel.height,
            "tracks": len(track_ids),
            "track_column_sha256": track_column_hash,
            "score": "CenterMaskScorer(width=None, L2_DIFF_LOG1P), forward API call",
        },
        "metadata": {
            "manifest_sha256": sha256(metadata_manifest_path),
            "artifact_sha256": sha256(metadata_path),
            "alphagenome_version": metadata_manifest["alphagenome_version"],
            "historical_snapshot_caveat": metadata_manifest["caveat"],
        },
        "extraction_manifest_sha256": sha256(extraction_manifest_path),
        "protocol": {
            "feature_variable": "log1p(abs(alt_activation - ref_activation))",
            "track_variable": "exported AlphaGenome L2_DIFF_LOG1P (no second transform)",
            "primary_population": "all variants pooled; no label/subset stratification",
            "primary_association": "Pearson correlation",
            "secondary_association": "Spearman correlation on frozen validation/test candidates",
            "sensitivity": "within-match-group centered Pearson and Spearman correlations on selected test pairs",
            "minimum_nonzero_rows_per_split": MIN_NONZERO_PER_SPLIT,
            "discovery_candidates_per_assay_orientation": DISCOVERY_PER_ASSAY,
            "validation_selected_per_assay_orientation": VALIDATED_PER_ASSAY,
            "held_out_multiple_testing": "Benjamini-Hochberg across all fixed test pairs",
            "test_discovery_threshold": "q < 0.05 and abs(test_r) >= 0.10",
            "group_bootstraps": BOOTSTRAPS,
            "splits": {
                "discovery": "chr5/7/9/13/15/17/19/21",
                "validation": "chr1/3",
                "test": "chr11/X",
            },
        },
    }
    write_json(output_dir / "results.json", result)
    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        **result,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--alphagenome-uri", required=True)
    parser.add_argument("--track-metadata", type=Path, required=True)
    parser.add_argument("--track-metadata-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extraction-commit", required=True)
    parser.add_argument("--analysis-commit", required=True)
    args = parser.parse_args()
    result = analyze(
        panel_path=args.panel,
        extraction_dir=args.extraction_dir,
        alphagenome_uri=args.alphagenome_uri,
        metadata_path=args.track_metadata,
        metadata_manifest_path=args.track_metadata_manifest,
        output_dir=args.output_dir,
        extraction_commit=args.extraction_commit,
        analysis_commit=args.analysis_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
