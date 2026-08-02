"""Cross-reference block-19 dsQTL direction hits with existing #421/#422 scans.

This is a post-hoc descriptive analysis. It quantifies response redundancy among
the complete-family QTL discoveries, then joins every discovered feature to the
existing AlphaGenome L2 and broad consequence result tables. It does not select
new QTL features or alter the original multiple-testing families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import rankdata

ISSUE = 434
ARM = "block19-25m"
ORIENTATIONS = ("forward", "reverse_complement")
RESPONSES = ("absolute", "signed")
CORRELATION_THRESHOLDS = (0.5, 0.7, 0.8, 0.9)
ALPHA_STATISTICS = ("pearson", "spearman")
ALPHA_MINIMUM_ABSOLUTE_EFFECT = 0.05
CONSEQUENCE_MINIMUM_ABSOLUTE_RANK_BISERIAL = 0.1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def assert_commit(value: str) -> None:
    assert len(value) == 40
    assert all(character in "0123456789abcdef" for character in value)


def load_manifest(root: Path, *, issue: int) -> dict[str, Any]:
    path = root / "manifest.json"
    assert path.is_file()
    manifest = json.loads(path.read_text())
    assert manifest["issue"] == issue
    return manifest


def verify_manifest_artifact(
    root: Path, manifest: dict[str, Any], relative_path: str
) -> None:
    expected = manifest["artifacts"][relative_path]
    path = root / relative_path
    assert path.is_file(), path
    assert path.stat().st_size == expected["bytes"]
    assert sha256_file(path) == expected["sha256"]


def load_qtl_hits(
    association_root: Path, association_manifest: dict[str, Any]
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for orientation in ORIENTATIONS:
        relative = f"families/dsqtl__{ARM}__{orientation}__direction.parquet"
        verify_manifest_artifact(association_root, association_manifest, relative)
        frame = pl.read_parquet(association_root / relative)
        assert frame["orientation"].unique().to_list() == [orientation]
        assert frame["arm"].unique().to_list() == [ARM]
        frames.append(
            frame.filter((pl.col("pearson_q") < 0.05) & (pl.col("spearman_q") < 0.05))
        )
    hits = pl.concat(frames).sort("orientation", "feature_id")
    assert hits.height == 47
    assert dict(hits.group_by("orientation").len().iter_rows()) == {
        "forward": 45,
        "reverse_complement": 2,
    }
    return hits


def dense_delta_matrix(
    *, panel_rows: int, sparse: pl.DataFrame, feature_ids: list[int]
) -> np.ndarray:
    assert panel_rows > 0 and feature_ids
    assert len(feature_ids) == len(set(feature_ids))
    lookup = {feature_id: index for index, feature_id in enumerate(feature_ids)}
    selected = sparse.join(
        pl.DataFrame(
            {
                "feature_id": feature_ids,
                "matrix_column": list(range(len(feature_ids))),
            }
        ),
        on="feature_id",
        how="inner",
    ).select("panel_row", "matrix_column", "delta")
    matrix = np.zeros((panel_rows, len(feature_ids)), dtype=np.float64)
    for panel_row, matrix_column, delta in selected.iter_rows():
        assert 0 <= panel_row < panel_rows
        matrix[panel_row, matrix_column] = delta
    assert np.isfinite(matrix).all()
    assert all(np.any(matrix[:, lookup[feature_id]] != 0) for feature_id in feature_ids)
    return matrix


def _component_members(
    correlation: np.ndarray, feature_ids: list[int], *, threshold: float
) -> list[list[int]]:
    assert correlation.shape == (len(feature_ids), len(feature_ids))
    assert 0 < threshold <= 1
    parent = list(range(len(feature_ids)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(feature_ids)):
        for right in range(left + 1, len(feature_ids)):
            if abs(correlation[left, right]) >= threshold:
                union(left, right)
    groups: dict[int, list[int]] = {}
    for index, feature_id in enumerate(feature_ids):
        groups.setdefault(find(index), []).append(feature_id)
    return sorted(
        (sorted(members) for members in groups.values()),
        key=lambda members: (-len(members), members),
    )


def response_redundancy(
    matrix: np.ndarray, feature_ids: list[int]
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    pearson = np.corrcoef(matrix, rowvar=False)
    spearman = np.corrcoef(rankdata(matrix, axis=0), rowvar=False)
    for correlation in (pearson, spearman):
        assert correlation.shape == (len(feature_ids), len(feature_ids))
        assert np.isfinite(correlation).all()

    def correlation_summary(correlation: np.ndarray) -> dict[str, float]:
        eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0, None)
        return {
            "effective_rank": float(
                eigenvalues.sum() ** 2 / np.square(eigenvalues).sum()
            ),
            "top_eigenvalue_share": float(eigenvalues[-1] / eigenvalues.sum()),
        }

    pair_rows = []
    for left in range(len(feature_ids)):
        for right in range(left + 1, len(feature_ids)):
            pair_rows.append(
                {
                    "feature_id_left": feature_ids[left],
                    "feature_id_right": feature_ids[right],
                    "pearson": pearson[left, right],
                    "absolute_pearson": abs(pearson[left, right]),
                    "spearman": spearman[left, right],
                    "absolute_spearman": abs(spearman[left, right]),
                }
            )
    pairs = pl.DataFrame(pair_rows).sort("absolute_pearson", descending=True)
    component_rows = []
    threshold_summary = {}
    for threshold in CORRELATION_THRESHOLDS:
        components = _component_members(pearson, feature_ids, threshold=threshold)
        threshold_summary[str(threshold)] = {
            "components": len(components),
            "component_sizes": [len(members) for members in components],
        }
        for component_index, members in enumerate(components, start=1):
            for feature_id in members:
                component_rows.append(
                    {
                        "absolute_pearson_threshold": threshold,
                        "component_index": component_index,
                        "component_minimum_feature_id": min(members),
                        "component_size": len(members),
                        "feature_id": feature_id,
                    }
                )
    components = pl.DataFrame(component_rows).sort(
        "absolute_pearson_threshold", "component_index", "feature_id"
    )
    summary = {
        "features": len(feature_ids),
        "pearson": correlation_summary(pearson),
        "spearman": correlation_summary(spearman),
        "absolute_pairwise_correlation_quantiles": {
            statistic: {
                str(quantile): float(pairs[f"absolute_{statistic}"].quantile(quantile))
                for quantile in (0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
            }
            for statistic in ("pearson", "spearman")
        },
        "component_metric": "absolute Pearson correlation",
        "threshold_components": threshold_summary,
    }
    return pairs, components, summary


def _load_alpha_statistic(
    root: Path, manifest: dict[str, Any], statistic: str
) -> pl.DataFrame:
    assert statistic in ALPHA_STATISTICS
    frames = []
    pattern = f"families/{ARM}__*__*__{statistic}_abs_delta.parquet"
    for path in sorted(root.glob(pattern)):
        relative = str(path.relative_to(root))
        verify_manifest_artifact(root, manifest, relative)
        frame = pl.read_parquet(path).with_columns(
            pl.col(column).cast(pl.String)
            for column in (
                "target_name",
                "group_axis",
                "assay",
                "tissue_group",
                "cell_lineage",
            )
        )
        frames.append(frame)
    assert frames
    return pl.concat(frames, how="diagonal_relaxed")


def alpha_overlap(
    *, root: Path, manifest: dict[str, Any], qtl_hits: pl.DataFrame
) -> pl.DataFrame:
    keys = ["feature_id", "orientation", "resolution", "target_id"]
    metadata = [
        "target_name",
        "group_axis",
        "assay",
        "tissue_group",
        "cell_lineage",
        "track_count",
    ]
    pearson = _load_alpha_statistic(root, manifest, "pearson").select(
        *keys,
        *metadata,
        pl.col("effect").alias("pearson_effect"),
        pl.col("pvalue").alias("pearson_p"),
        pl.col("qvalue").alias("pearson_q"),
    )
    spearman = _load_alpha_statistic(root, manifest, "spearman").select(
        *keys,
        pl.col("effect").alias("spearman_effect"),
        pl.col("pvalue").alias("spearman_p"),
        pl.col("qvalue").alias("spearman_q"),
    )
    overlap = (
        pearson.join(spearman, on=keys)
        .filter(
            (pl.col("pearson_q") < 0.05)
            & (pl.col("spearman_q") < 0.05)
            & (pl.col("pearson_effect").abs() >= ALPHA_MINIMUM_ABSOLUTE_EFFECT)
            & (pl.col("spearman_effect").abs() >= ALPHA_MINIMUM_ABSOLUTE_EFFECT)
            & (pl.col("pearson_effect").sign() == pl.col("spearman_effect").sign())
        )
        .join(
            qtl_hits.select(
                "feature_id",
                "orientation",
                pl.col("pearson").alias("qtl_pearson"),
                pl.col("spearman").alias("qtl_spearman"),
            ),
            on=["feature_id", "orientation"],
        )
        .sort("orientation", "feature_id", "resolution", "target_id")
    )
    return overlap


def consequence_overlap(
    *, root: Path, manifest: dict[str, Any], qtl_hits: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    frames = []
    for orientation in ORIENTATIONS:
        for response in RESPONSES:
            relative = f"families/{ARM}__{orientation}__{response}__one_vs_rest.parquet"
            verify_manifest_artifact(root, manifest, relative)
            frames.append(pl.read_parquet(root / relative))
    complete = (
        pl.concat(frames)
        .join(
            qtl_hits.select("feature_id", "orientation"),
            on=["feature_id", "orientation"],
        )
        .with_columns(pl.col("rank_biserial").abs().alias("absolute_rank_biserial"))
        .sort("orientation", "feature_id", "consequence", "response")
    )
    strongest = (
        complete.sort("absolute_rank_biserial", descending=True)
        .unique(["feature_id", "orientation"], keep="first")
        .sort("orientation", "feature_id")
    )
    return complete, strongest


def run(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    association_root: Path,
    extraction_root: Path,
    alpha_root: Path,
    consequence_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_commit(experiment_commit)
    run_id = os.environ.get("RUN_ID", "")
    assert run_id

    panel_manifest = json.loads(panel_manifest_path.read_text())
    panel = pl.read_parquet(panel_path)
    assert panel.height == 559
    assert panel_manifest["scope"] == "dsqtl-positive-direction-pilot"
    assert panel_manifest["panel"]["sha256"] == sha256_file(panel_path)
    association_manifest = load_manifest(association_root, issue=ISSUE)
    extraction_manifest = load_manifest(extraction_root, issue=ISSUE)
    alpha_manifest = load_manifest(alpha_root, issue=421)
    consequence_manifest = load_manifest(consequence_root, issue=422)

    qtl_hits = load_qtl_hits(association_root, association_manifest)
    forward_hits = qtl_hits.filter(pl.col("orientation") == "forward")
    forward_feature_ids = forward_hits["feature_id"].to_list()
    extraction_relative = f"{ARM}/sae_focal_forward.parquet"
    verify_manifest_artifact(extraction_root, extraction_manifest, extraction_relative)
    sparse_forward = pl.read_parquet(extraction_root / extraction_relative)
    matrix = dense_delta_matrix(
        panel_rows=panel.height,
        sparse=sparse_forward,
        feature_ids=forward_feature_ids,
    )
    pairs, components, redundancy = response_redundancy(matrix, forward_feature_ids)
    alpha = alpha_overlap(root=alpha_root, manifest=alpha_manifest, qtl_hits=qtl_hits)
    consequences, strongest_consequences = consequence_overlap(
        root=consequence_root,
        manifest=consequence_manifest,
        qtl_hits=qtl_hits,
    )

    consequence_evidence = consequences.filter(
        (pl.col("welch_qvalue") < 0.05)
        & (pl.col("mwu_qvalue") < 0.05)
        & (
            pl.col("absolute_rank_biserial")
            >= CONSEQUENCE_MINIMUM_ABSOLUTE_RANK_BISERIAL
        )
    )
    forward_strongest = strongest_consequences.filter(
        pl.col("orientation") == "forward"
    )
    strongest_counts = {
        f"{consequence}|{response}": count
        for consequence, response, count in forward_strongest.group_by(
            "consequence", "response"
        )
        .len()
        .sort("len", descending=True)
        .iter_rows()
    }
    summary = {
        "qtl_hits": {
            orientation: qtl_hits.filter(pl.col("orientation") == orientation).height
            for orientation in ORIENTATIONS
        },
        "forward_response_redundancy": redundancy,
        "alphagenome_concordant_overlap": {
            "rows": alpha.height,
            "feature_orientation_pairs": alpha.select("feature_id", "orientation")
            .unique()
            .height,
            "criteria": (
                "Pearson q<0.05 and Spearman q<0.05, same effect direction, "
                f"both |effect|>={ALPHA_MINIMUM_ABSOLUTE_EFFECT}"
            ),
        },
        "consequence_overlap": {
            "feature_orientation_pairs_with_both_q_and_abs_rank_biserial_ge_0_1": (
                consequence_evidence.select("feature_id", "orientation").unique().height
            ),
            "forward_strongest_consequence_counts": strongest_counts,
        },
    }

    output_dir.mkdir(parents=True)
    artifacts = {
        "qtl_hits.parquet": qtl_hits,
        "forward_pair_correlations.parquet": pairs,
        "forward_response_components.parquet": components,
        "alphagenome_overlap.parquet": alpha,
        "consequence_overlap.parquet": consequences,
        "strongest_consequence_by_hit.parquet": strongest_consequences,
    }
    for name, frame in artifacts.items():
        frame.write_parquet(output_dir / name)
    write_json(output_dir / "summary.json", summary)

    artifact_names = (*artifacts.keys(), "summary.json")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": run_id,
        "experiment_commit": experiment_commit,
        "analysis": "post-hoc cross-dataset response-family triage",
        "inputs": {
            "panel_sha256": sha256_file(panel_path),
            "panel_manifest_sha256": sha256_file(panel_manifest_path),
            "association_manifest_sha256": sha256_file(
                association_root / "manifest.json"
            ),
            "extraction_manifest_sha256": sha256_file(
                extraction_root / "manifest.json"
            ),
            "alphagenome_manifest_sha256": sha256_file(alpha_root / "manifest.json"),
            "consequence_manifest_sha256": sha256_file(
                consequence_root / "manifest.json"
            ),
        },
        "summary": summary,
        "artifacts": {
            name: {
                "bytes": (output_dir / name).stat().st_size,
                "sha256": sha256_file(output_dir / name),
            }
            for name in artifact_names
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--association-root", type=Path, required=True)
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--alphagenome-root", type=Path, required=True)
    parser.add_argument("--consequence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = run(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        association_root=args.association_root,
        extraction_root=args.extraction_root,
        alpha_root=args.alphagenome_root,
        consequence_root=args.consequence_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
