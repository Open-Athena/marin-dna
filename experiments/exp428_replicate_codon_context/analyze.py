"""Score the frozen SAE hypotheses and sequence baseline for experiment 428."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOCAL_THREAD_LIMITS = {
    "POLARS_MAX_THREADS": 2,
    "RAYON_NUM_THREADS": 2,
    "OMP_NUM_THREADS": 1,
    "MKL_NUM_THREADS": 1,
    "OPENBLAS_NUM_THREADS": 1,
    "NUMEXPR_NUM_THREADS": 1,
}
for variable, limit in LOCAL_THREAD_LIMITS.items():
    configured = int(os.environ.get(variable, limit))
    assert configured > 0, (variable, configured)
    os.environ[variable] = str(min(configured, limit))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from marin_dna.data.genome import Genome
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from extract import FEATURE_COLUMNS, ORIENTATIONS
from panel import assert_current_commit, sha256_file, write_json

ISSUE = 428
SEED = 428
BOOTSTRAPS = 1_000
BASELINE_WINDOW_BP = 31
BASELINE_FOCAL_INDEX = 15
BASELINE_CS = (0.01, 0.1, 1.0, 10.0, 100.0)
POSITIVE_CLASS = "missense_variant"
NUCLEOTIDES = "ACGT"

FEATURE_SPECS = {
    "f11064_5m": {
        "feature_id": 11_064,
        "sae": "block19-5m",
        "direction": 1,
        "registered_view": "max_abs",
        "role": "primary",
    },
    "f12658_5m": {
        "feature_id": 12_658,
        "sae": "block19-5m",
        "direction": 1,
        "registered_view": "signed_mean",
        "role": "secondary",
    },
    "f13637_25m": {
        "feature_id": 13_637,
        "sae": "block19-25m",
        "direction": 1,
        "registered_view": "signed_mean",
        "role": "secondary",
    },
}
SENSITIVITY_VIEWS = (
    "forward",
    "reverse_complement",
    "coding_aligned",
    "anti_aligned",
    "signed_mean",
    "max_abs",
)

assert set(FEATURE_SPECS) == set(FEATURE_COLUMNS)
assert BASELINE_WINDOW_BP == 2 * BASELINE_FOCAL_INDEX + 1


def matched_auc(scores: np.ndarray, positive: np.ndarray, strata: np.ndarray) -> float:
    """Pair-weighted AUROC within the fixed matching strata."""
    assert scores.shape == positive.shape == strata.shape and scores.ndim == 1
    assert np.isfinite(scores).all()
    numerator = 0.0
    comparable_pairs = 0
    for stratum in np.unique(strata):
        selected = strata == stratum
        selected_positive = positive[selected]
        positive_count = int(selected_positive.sum())
        negative_count = int((~selected_positive).sum())
        if positive_count == 0 or negative_count == 0:
            continue
        pairs = positive_count * negative_count
        numerator += pairs * roc_auc_score(selected_positive, scores[selected])
        comparable_pairs += pairs
    assert comparable_pairs > 0
    return float(numerator / comparable_pairs)


def stratified_block_resample_indices(
    strata: np.ndarray, blocks: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Resample genomic blocks independently within each fixed label stratum."""
    assert strata.shape == blocks.shape and strata.ndim == 1
    sampled_groups: list[np.ndarray] = []
    for stratum in np.unique(strata):
        stratum_rows = np.flatnonzero(strata == stratum)
        stratum_blocks = np.unique(blocks[stratum_rows])
        assert len(stratum_blocks) > 0
        rows_by_block = [
            stratum_rows[blocks[stratum_rows] == block] for block in stratum_blocks
        ]
        sampled_groups.append(
            np.concatenate(
                [
                    rows_by_block[index]
                    for index in rng.integers(
                        0, len(rows_by_block), size=len(rows_by_block)
                    )
                ]
            )
        )
    output = np.concatenate(sampled_groups)
    assert len(output) > 0
    return output


def bootstrap_matched_aucs(
    score_matrix: np.ndarray,
    score_names: list[str],
    positive: np.ndarray,
    matching_strata: np.ndarray,
    blocks: np.ndarray,
    *,
    samples: int = BOOTSTRAPS,
    seed: int = SEED * 1_000 + 1,
) -> pl.DataFrame:
    assert score_matrix.shape == (len(positive), len(score_names))
    assert positive.shape == matching_strata.shape == blocks.shape
    assert samples > 0 and np.isfinite(score_matrix).all()
    bootstrap_strata = np.asarray(
        [
            f"{int(label)}|{stratum}"
            for label, stratum in zip(positive, matching_strata, strict=True)
        ]
    )
    rng = np.random.default_rng(seed)
    values = np.empty((samples, len(score_names)), dtype=np.float64)
    for sample in range(samples):
        indices = stratified_block_resample_indices(bootstrap_strata, blocks, rng)
        for column in range(len(score_names)):
            values[sample, column] = matched_auc(
                score_matrix[indices, column],
                positive[indices],
                matching_strata[indices],
            )
    rows = []
    for column, name in enumerate(score_names):
        low, high = np.quantile(values[:, column], [0.025, 0.975])
        rows.append(
            {
                "score_column": name,
                "bootstrap_samples": samples,
                "conditional_auc_ci_low": float(low),
                "conditional_auc_ci_high": float(high),
                "bootstrap_mean": float(values[:, column].mean()),
                "bootstrap_std": float(values[:, column].std(ddof=1)),
            }
        )
    return pl.DataFrame(rows)


def feature_score_views(
    forward: np.ndarray,
    reverse_complement_values: np.ndarray,
    strand: np.ndarray,
    *,
    direction: int,
) -> dict[str, np.ndarray]:
    assert forward.shape == reverse_complement_values.shape == strand.shape
    assert direction in {-1, 1}
    assert set(np.unique(strand)) <= {"+", "-"}
    signed_forward = direction * forward
    signed_reverse = direction * reverse_complement_values
    return {
        "forward": signed_forward,
        "reverse_complement": signed_reverse,
        "coding_aligned": np.where(strand == "+", signed_forward, signed_reverse),
        "anti_aligned": np.where(strand == "+", signed_reverse, signed_forward),
        "signed_mean": direction * (forward + reverse_complement_values) / 2,
        "max_abs": direction
        * np.maximum(np.abs(forward), np.abs(reverse_complement_values)),
    }


def encode_positional_contexts(
    contexts: list[str], alternate_bases: list[str]
) -> np.ndarray:
    assert len(contexts) == len(alternate_bases) > 0
    base_index = {base: index for index, base in enumerate(NUCLEOTIDES)}
    design = np.zeros((len(contexts), BASELINE_WINDOW_BP * 4 + 4), dtype=np.float32)
    for row, (context, alt) in enumerate(zip(contexts, alternate_bases, strict=True)):
        assert len(context) == BASELINE_WINDOW_BP
        assert set(context) <= set(NUCLEOTIDES) and alt in NUCLEOTIDES
        for position, base in enumerate(context):
            design[row, position * 4 + base_index[base]] = 1
        design[row, BASELINE_WINDOW_BP * 4 + base_index[alt]] = 1
    assert np.all(design[:, : BASELINE_WINDOW_BP * 4].sum(axis=1) == BASELINE_WINDOW_BP)
    assert np.all(design[:, BASELINE_WINDOW_BP * 4 :].sum(axis=1) == 1)
    return design


def baseline_design(frame: pl.DataFrame, fasta_path: Path) -> np.ndarray:
    genome = Genome(fasta_path, subset_chroms={"21"})
    assert set(genome.chroms) == {"21"}
    contexts: list[str] = []
    for row in frame.iter_rows(named=True):
        pos0 = int(row["pos"]) - 1
        start = pos0 - BASELINE_FOCAL_INDEX
        end = pos0 + BASELINE_FOCAL_INDEX + 1
        assert start >= 0 and end - start == BASELINE_WINDOW_BP
        context = genome(row["chrom"], start, end, "+").upper()
        assert context[BASELINE_FOCAL_INDEX] == row["ref"]
        contexts.append(context)
    return encode_positional_contexts(contexts, frame["alt"].to_list())


def fit_sequence_baseline(
    design: np.ndarray,
    positive: np.ndarray,
    split: np.ndarray,
    matching_strata: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    assert design.shape[0] == len(positive) == len(split) == len(matching_strata)
    discovery = np.flatnonzero(split == "discovery")
    validation = np.flatnonzero(split == "validation")
    test = np.flatnonzero(split == "test")
    assert len(discovery) > 0 and len(validation) > 0 and len(test) > 0
    candidates: list[dict[str, float]] = []
    selected_model: LogisticRegression | None = None
    selected_c = 0.0
    selected_validation_auc = -np.inf
    for c_value in BASELINE_CS:
        model = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=2_000,
            random_state=SEED,
        )
        model.fit(design[discovery], positive[discovery])
        validation_scores = model.predict_proba(design[validation])[:, 1]
        validation_auc = matched_auc(
            validation_scores,
            positive[validation],
            matching_strata[validation],
        )
        candidates.append({"c": c_value, "validation_conditional_auc": validation_auc})
        if validation_auc > selected_validation_auc:
            selected_validation_auc = validation_auc
            selected_c = c_value
            selected_model = model
    assert selected_model is not None and selected_c in BASELINE_CS
    scores = selected_model.predict_proba(design)[:, 1]
    assert scores.shape == positive.shape and np.isfinite(scores).all()
    return scores, {
        "design": "31-bp positional reference one-hot plus alternate-base one-hot",
        "features": design.shape[1],
        "regularization_candidates": list(BASELINE_CS),
        "candidate_validation_metrics": candidates,
        "selected_c": selected_c,
        "selected_validation_conditional_auc": selected_validation_auc,
        "fit_rows": len(discovery),
        "selection_rows": len(validation),
        "untouched_test_rows": len(test),
        "refit_after_validation": False,
        "coefficient_l2_norm": float(np.linalg.norm(selected_model.coef_)),
        "iterations": selected_model.n_iter_.tolist(),
    }


def load_inputs(
    panel_path: Path,
    panel_manifest_path: Path,
    extraction_dir: Path,
) -> tuple[pl.DataFrame, dict[str, Any], dict[str, Any]]:
    panel_manifest = json.loads(panel_manifest_path.read_text())
    extraction_manifest_path = extraction_dir / "manifest.json"
    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    assert panel_manifest["issue"] == extraction_manifest["issue"] == ISSUE
    assert panel_manifest["output"]["sha256"] == sha256_file(panel_path)
    assert extraction_manifest["panel"]["sha256"] == sha256_file(panel_path)
    panel = pl.read_parquet(panel_path)
    assert panel.height == panel_manifest["output"]["rows"]
    frame = panel
    for orientation in ORIENTATIONS:
        metadata = extraction_manifest["outputs"][orientation]
        path = extraction_dir / metadata["path_name"]
        assert sha256_file(path) == metadata["sha256"]
        activations = pl.read_parquet(path)
        assert activations.height == panel.height
        assert activations["panel_row"].n_unique() == panel.height
        activations = activations.rename(
            {
                column: f"{column}_{orientation}"
                for column in activations.columns
                if column != "panel_row"
            }
        )
        frame = frame.join(activations, on="panel_row", how="inner", validate="1:1")
        assert frame.height == panel.height
    return frame, panel_manifest, extraction_manifest


def make_scores(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, str]]:
    score_columns: dict[str, np.ndarray] = {}
    score_metadata: dict[str, str] = {}
    strand = frame["consensus_strand"].to_numpy()
    for feature, spec in FEATURE_SPECS.items():
        forward = frame[f"{feature}_delta_forward"].to_numpy()
        reverse = frame[f"{feature}_delta_reverse_complement"].to_numpy()
        views = feature_score_views(
            forward, reverse, strand, direction=spec["direction"]
        )
        assert set(views) == set(SENSITIVITY_VIEWS)
        for view, values in views.items():
            column = f"score__{feature}__{view}"
            score_columns[column] = values.astype(np.float32)
            score_metadata[column] = f"{feature}|{view}"
    output = frame.with_columns(
        [pl.Series(name, values) for name, values in score_columns.items()]
    )
    assert output.height == frame.height
    return output, score_metadata


def evaluate_scores(
    frame: pl.DataFrame, score_metadata: dict[str, str]
) -> pl.DataFrame:
    positive = frame["consequence_cre"].to_numpy() == POSITIVE_CLASS
    matching_strata = frame["matching_stratum"].to_numpy()
    split = frame["split"].to_numpy()
    rows: list[dict[str, Any]] = []
    for score_column, descriptor in score_metadata.items():
        feature, view = descriptor.split("|")
        spec = FEATURE_SPECS.get(feature)
        for split_name in ("discovery", "validation", "test"):
            selected = split == split_name
            scores = frame[score_column].to_numpy()[selected]
            labels = positive[selected]
            strata = matching_strata[selected]
            rows.append(
                {
                    "score_column": score_column,
                    "feature": feature,
                    "view": view,
                    "split": split_name,
                    "role": "baseline" if spec is None else spec["role"],
                    "registered": feature == "sequence_baseline"
                    or view == spec["registered_view"],
                    "rows": int(selected.sum()),
                    "positives": int(labels.sum()),
                    "overall_auc": float(roc_auc_score(labels, scores)),
                    "conditional_auc": matched_auc(scores, labels, strata),
                    "conditional_auc_ci_low": None,
                    "conditional_auc_ci_high": None,
                }
            )
    return pl.DataFrame(rows)


def add_bootstrap_intervals(
    metrics: pl.DataFrame,
    frame: pl.DataFrame,
    score_metadata: dict[str, str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    test = frame["split"].to_numpy() == "test"
    names = list(score_metadata)
    score_matrix = np.column_stack([frame[name].to_numpy()[test] for name in names])
    intervals = bootstrap_matched_aucs(
        score_matrix,
        names,
        frame["consequence_cre"].to_numpy()[test] == POSITIVE_CLASS,
        frame["matching_stratum"].to_numpy()[test],
        frame["block_id"].to_numpy()[test],
    )
    updated = metrics.join(intervals, on="score_column", how="left", validate="m:1")
    updated = updated.with_columns(
        pl.when(pl.col("split") != "test")
        .then(None)
        .otherwise(pl.col("conditional_auc_ci_low_right"))
        .alias("conditional_auc_ci_low"),
        pl.when(pl.col("split") != "test")
        .then(None)
        .otherwise(pl.col("conditional_auc_ci_high_right"))
        .alias("conditional_auc_ci_high"),
    ).drop(
        "conditional_auc_ci_low_right",
        "conditional_auc_ci_high_right",
        "bootstrap_samples",
        "bootstrap_mean",
        "bootstrap_std",
    )
    return updated, intervals


def stratified_test_metrics(
    frame: pl.DataFrame, score_metadata: dict[str, str]
) -> pl.DataFrame:
    test = frame.filter(pl.col("split") == "test")
    rows: list[dict[str, Any]] = []
    for score_column, descriptor in score_metadata.items():
        feature, view = descriptor.split("|")
        for group_column in ("consensus_codon_position", "matching_stratum"):
            for group_value in test[group_column].unique().sort():
                selected = test.filter(pl.col(group_column) == group_value)
                positive = selected["consequence_cre"].to_numpy() == POSITIVE_CLASS
                if positive.sum() == 0 or (~positive).sum() == 0:
                    continue
                rows.append(
                    {
                        "feature": feature,
                        "view": view,
                        "group_type": group_column,
                        "group": str(group_value),
                        "rows": selected.height,
                        "positives": int(positive.sum()),
                        "auc": float(
                            roc_auc_score(positive, selected[score_column].to_numpy())
                        ),
                    }
                )
    return pl.DataFrame(rows)


def orientation_diagnostics(frame: pl.DataFrame) -> pl.DataFrame:
    rows = []
    test = frame.filter(pl.col("split") == "test")
    for feature in FEATURE_SPECS:
        forward = test[f"{feature}_delta_forward"].to_numpy()
        reverse = test[f"{feature}_delta_reverse_complement"].to_numpy()
        both_nonzero = (forward != 0) & (reverse != 0)
        pearson = (
            float(np.corrcoef(forward, reverse)[0, 1])
            if np.std(forward) > 0 and np.std(reverse) > 0
            else None
        )
        sign_agreement = (
            float(
                np.mean(
                    np.sign(forward[both_nonzero]) == np.sign(reverse[both_nonzero])
                )
            )
            if both_nonzero.any()
            else None
        )
        rows.append(
            {
                "feature": feature,
                "rows": test.height,
                "pearson_fwd_rc": pearson,
                "nonzero_fwd": int(np.count_nonzero(forward)),
                "nonzero_rc": int(np.count_nonzero(reverse)),
                "both_nonzero": int(np.count_nonzero(both_nonzero)),
                "same_sign_among_both_nonzero": sign_agreement,
            }
        )
    return pl.DataFrame(rows)


def plot_registered_metrics(metrics: pl.DataFrame, output_dir: Path) -> list[Path]:
    selected = metrics.filter(
        (pl.col("split") == "test") & pl.col("registered")
    ).with_columns(
        (
            pl.col("feature") + pl.lit(" · ") + pl.col("view").str.replace_all("_", " ")
        ).alias("label")
    )
    data = selected.to_pandas().sort_values("conditional_auc")
    sns.set_theme(style="whitegrid", context="talk")
    graph = sns.relplot(
        data=data,
        x="conditional_auc",
        y="label",
        hue="role",
        style="role",
        kind="scatter",
        s=140,
        height=5.4,
        aspect=1.45,
    )
    axis = graph.ax
    labels = [tick.get_text() for tick in axis.get_yticklabels()]
    positions = {label: index for index, label in enumerate(labels)}
    for row in data.itertuples():
        axis.hlines(
            positions[row.label],
            row.conditional_auc_ci_low,
            row.conditional_auc_ci_high,
            color="0.35",
            linewidth=2,
            zorder=1,
        )
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
    graph.set_axis_labels("Pair-weighted conditional AUROC", "")
    graph.figure.suptitle("Frozen SAE replication on held-out chr21 blocks", y=1.03)
    graph.figure.text(
        0.01,
        0.01,
        "Intervals: 95% label × matched-stratum genomic-block bootstrap CI (1,000 replicates)",
        fontsize=9,
    )
    graph.figure.tight_layout(rect=(0, 0.05, 1, 1))
    paths = [
        output_dir / "registered_metrics.svg",
        output_dir / "registered_metrics.png",
    ]
    for path in paths:
        graph.figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(graph.figure)
    return paths


def plot_orientation_scatter(frame: pl.DataFrame, output_dir: Path) -> list[Path]:
    test = frame.filter(pl.col("split") == "test")
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_SPECS:
        for row in test.select(
            "consequence_cre",
            f"{feature}_delta_forward",
            f"{feature}_delta_reverse_complement",
        ).iter_rows(named=True):
            rows.append(
                {
                    "feature": feature,
                    "consequence": row["consequence_cre"],
                    "forward_delta": row[f"{feature}_delta_forward"],
                    "reverse_complement_delta": row[
                        f"{feature}_delta_reverse_complement"
                    ],
                }
            )
    graph = sns.relplot(
        data=pl.DataFrame(rows).to_pandas(),
        x="forward_delta",
        y="reverse_complement_delta",
        col="feature",
        hue="consequence",
        kind="scatter",
        alpha=0.3,
        s=25,
        height=4.2,
        aspect=1,
        facet_kws={"sharex": False, "sharey": False},
    )
    graph.set_axis_labels("FWD alt − ref", "RC alt − ref")
    graph.set_titles("{col_name}")
    graph.figure.suptitle("Orientation sensitivity on held-out test blocks", y=1.03)
    paths = [
        output_dir / "orientation_scatter.svg",
        output_dir / "orientation_scatter.png",
    ]
    for path in paths:
        graph.figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(graph.figure)
    return paths


def analyze(
    *,
    panel_path: Path,
    panel_manifest_path: Path,
    extraction_dir: Path,
    fasta_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert panel_path.is_file() and panel_manifest_path.is_file()
    assert extraction_dir.is_dir() and fasta_path.is_file()
    assert Path(f"{fasta_path}.fai").is_file() and Path(f"{fasta_path}.gzi").is_file()
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_current_commit(experiment_commit)
    started = time.monotonic()
    frame, panel_manifest, extraction_manifest = load_inputs(
        panel_path, panel_manifest_path, extraction_dir
    )
    assert extraction_manifest["experiment_commit"] == experiment_commit
    output_dir.mkdir(parents=True)
    frame, score_metadata = make_scores(frame)

    positive = frame["consequence_cre"].to_numpy() == POSITIVE_CLASS
    baseline_scores, baseline_metadata = fit_sequence_baseline(
        baseline_design(frame, fasta_path),
        positive,
        frame["split"].to_numpy(),
        frame["matching_stratum"].to_numpy(),
    )
    baseline_column = "score__sequence_baseline__positional31_alt"
    frame = frame.with_columns(pl.Series(baseline_column, baseline_scores))
    score_metadata[baseline_column] = "sequence_baseline|positional31_alt"

    metrics = evaluate_scores(frame, score_metadata)
    metrics, intervals = add_bootstrap_intervals(metrics, frame, score_metadata)
    strata_metrics = stratified_test_metrics(frame, score_metadata)
    orientation = orientation_diagnostics(frame)

    primary = metrics.filter(
        (pl.col("split") == "test")
        & (pl.col("feature") == "f11064_5m")
        & (pl.col("view") == "max_abs")
    )
    assert primary.height == 1
    primary_row = primary.row(0, named=True)
    primary_success = primary_row["conditional_auc_ci_low"] > 0.5

    artifact_paths = {
        "scores.parquet": output_dir / "scores.parquet",
        "metrics.parquet": output_dir / "metrics.parquet",
        "bootstrap_intervals.parquet": output_dir / "bootstrap_intervals.parquet",
        "stratified_test_metrics.parquet": output_dir
        / "stratified_test_metrics.parquet",
        "orientation_diagnostics.parquet": output_dir
        / "orientation_diagnostics.parquet",
    }
    frame.write_parquet(artifact_paths["scores.parquet"], compression="zstd")
    metrics.write_parquet(artifact_paths["metrics.parquet"], compression="zstd")
    intervals.write_parquet(
        artifact_paths["bootstrap_intervals.parquet"], compression="zstd"
    )
    strata_metrics.write_parquet(
        artifact_paths["stratified_test_metrics.parquet"], compression="zstd"
    )
    orientation.write_parquet(
        artifact_paths["orientation_diagnostics.parquet"], compression="zstd"
    )
    for path in plot_registered_metrics(metrics, output_dir):
        artifact_paths[path.name] = path
    for path in plot_orientation_scatter(frame, output_dir):
        artifact_paths[path.name] = path

    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "issue": ISSUE,
        "run_id": os.environ.get("RUN_ID", ""),
        "experiment_commit": experiment_commit,
        "elapsed_seconds": time.monotonic() - started,
        "inputs": {
            "panel_sha256": sha256_file(panel_path),
            "panel_manifest_sha256": sha256_file(panel_manifest_path),
            "panel_experiment_commit": panel_manifest["experiment_commit"],
            "extraction_manifest_sha256": sha256_file(extraction_dir / "manifest.json"),
            "extraction_experiment_commit": extraction_manifest["experiment_commit"],
            "fasta_sha256": sha256_file(fasta_path),
        },
        "registered_features": FEATURE_SPECS,
        "baseline": baseline_metadata,
        "bootstrap": {
            "samples": BOOTSTRAPS,
            "seed": SEED * 1_000 + 1,
            "strata": "label × codon-position/transcript-substitution",
            "resampling_unit": "1 Mb genomic block within each label-stratum",
        },
        "primary": {
            **primary_row,
            "success_criterion": "95% CI lower bound > 0.5",
            "success": primary_success,
        },
        "registered_test_metrics": metrics.filter(
            (pl.col("split") == "test") & pl.col("registered")
        ).to_dicts(),
        "orientation_diagnostics": orientation.to_dicts(),
        "artifacts": {
            name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in artifact_paths.items()
        },
    }
    write_json(output_dir / "manifest.json", result)
    result["manifest_sha256"] = sha256_file(output_dir / "manifest.json")
    write_json(output_dir / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        panel_path=args.panel,
        panel_manifest_path=args.panel_manifest,
        extraction_dir=args.extraction_dir,
        fasta_path=args.fasta,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["primary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
