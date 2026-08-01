"""Test whether simple allele or local-sequence cues explain the coding contrast."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from marin_dna.data.genome import Genome
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from analyze import (
    column_values,
    make_views,
    sha256_file,
    stratified_block_resample_indices,
    write_json,
)
from pairwise import (
    BOOTSTRAPS,
    NEGATIVE_CLASS,
    PAIR_CLASSES,
    PAIRWISE_CHANCE,
    POSITIVE_CLASS,
    bootstrap_pairwise_metrics,
    load_selected_activations,
)
from train import BLOCK_INDICES, BUDGETS, D_SAE, ISSUE, arm_label, assert_commit

CONTEXT_RADIUS = 15
CONTEXT_BP = 2 * CONTEXT_RADIUS + 1
C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
RANDOM_SEED = 4_261
TARGET_SAE_ROWS = (
    ("block19-5m", "signed_mean"),
    ("block19-25m", "signed_mean"),
    ("block19-5m", "max_abs"),
)


def extract_contexts(panel: pl.DataFrame, fasta_path: Path) -> np.ndarray:
    """Extract 31-bp reference contexts; panel positions are 1-based."""
    assert fasta_path.is_file()
    assert Path(f"{fasta_path}.fai").is_file() and Path(f"{fasta_path}.gzi").is_file()
    assert panel["chrom"].unique().to_list() == ["21"]
    genome = Genome(fasta_path, subset_chroms={"21"})
    assert set(genome.chroms) == {"21"}
    contexts: list[str] = []
    for row in panel.iter_rows(named=True):
        pos0 = int(row["pos"]) - 1
        assert pos0 >= CONTEXT_RADIUS
        start = pos0 - CONTEXT_RADIUS
        end = pos0 + CONTEXT_RADIUS + 1
        context = genome(row["chrom"], start, end, "+").upper()
        assert len(context) == CONTEXT_BP
        assert set(context) <= set("ACGT")
        assert context[CONTEXT_RADIUS] == row["ref"]
        contexts.append(context)
    return np.asarray(contexts)


def make_baseline_designs(
    contexts: np.ndarray, alternate: np.ndarray
) -> dict[str, np.ndarray]:
    assert contexts.ndim == alternate.ndim == 1
    assert len(contexts) == len(alternate) and len(contexts) > 0
    assert all(len(context) == CONTEXT_BP for context in contexts)
    assert set(alternate.tolist()) <= set("ACGT")
    designs: dict[str, np.ndarray] = {}
    for k in (1, 3, 5, 7):
        half = k // 2
        categories = np.asarray(
            [
                f"{context[CONTEXT_RADIUS - half : CONTEXT_RADIUS + half + 1]}>{alt}"
                for context, alt in zip(contexts, alternate, strict=True)
            ],
            dtype=object,
        )
        designs[f"centered_{k}mer_alt"] = categories.reshape(-1, 1)
    designs["positional_31bp_alt"] = np.asarray(
        [
            list(context) + [alt]
            for context, alt in zip(contexts, alternate, strict=True)
        ],
        dtype=object,
    )
    assert set(designs) == {
        "centered_1mer_alt",
        "centered_3mer_alt",
        "centered_5mer_alt",
        "centered_7mer_alt",
        "positional_31bp_alt",
    }
    assert all(design.shape[0] == len(contexts) for design in designs.values())
    return designs


def matched_substitution_auc(
    scores: np.ndarray, positive: np.ndarray, substitutions: np.ndarray
) -> float:
    """Pair-weighted AUROC using only positive/negative pairs with the same allele change."""
    assert scores.shape == positive.shape == substitutions.shape
    numerator = 0.0
    comparable_pairs = 0
    for substitution in np.unique(substitutions):
        selected = substitutions == substitution
        selected_positive = positive[selected]
        if not selected_positive.any() or selected_positive.all():
            continue
        positive_count = int(selected_positive.sum())
        negative_count = int((~selected_positive).sum())
        pairs = positive_count * negative_count
        numerator += pairs * roc_auc_score(selected_positive, scores[selected])
        comparable_pairs += pairs
    assert comparable_pairs > 0
    return float(numerator / comparable_pairs)


def bootstrap_matched_substitution_auc(
    scores: np.ndarray,
    positive: np.ndarray,
    substitutions: np.ndarray,
    blocks: np.ndarray,
    *,
    seed: int,
    samples: int = BOOTSTRAPS,
) -> tuple[float, float]:
    assert scores.shape == positive.shape == substitutions.shape == blocks.shape
    strata = np.asarray(
        [
            f"{int(is_positive)}|{substitution}"
            for is_positive, substitution in zip(positive, substitutions, strict=True)
        ]
    )
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indices = stratified_block_resample_indices(strata, blocks, rng)
        values.append(
            matched_substitution_auc(
                scores[indices], positive[indices], substitutions[indices]
            )
        )
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def fit_baseline(
    design: np.ndarray,
    positive: np.ndarray,
    split: np.ndarray,
    substitutions: np.ndarray,
    blocks: np.ndarray,
    *,
    method: str,
) -> tuple[dict[str, Any], np.ndarray]:
    assert design.shape[0] == len(positive) == len(split) == len(substitutions)
    discovery = np.flatnonzero(split == "discovery")
    validation = np.flatnonzero(split == "validation")
    test = np.flatnonzero(split == "test")
    assert positive[discovery].sum() == 256
    assert positive[validation].sum() == positive[test].sum() == 128
    selected_model = None
    selected_c = None
    selected_validation_auc = -np.inf
    selected_validation_ap = -np.inf
    for c_value in C_VALUES:
        model = make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            LogisticRegression(
                C=c_value,
                solver="liblinear",
                max_iter=2_000,
                random_state=RANDOM_SEED,
            ),
        )
        model.fit(design[discovery], positive[discovery])
        validation_scores = model.predict_proba(design[validation])[:, 1]
        validation_auc = float(roc_auc_score(positive[validation], validation_scores))
        validation_ap = float(
            average_precision_score(positive[validation], validation_scores)
        )
        if validation_auc > selected_validation_auc:
            selected_model = model
            selected_c = c_value
            selected_validation_auc = validation_auc
            selected_validation_ap = validation_ap
    assert selected_model is not None and selected_c is not None
    test_scores = selected_model.predict_proba(design[test])[:, 1]
    confidence = bootstrap_pairwise_metrics(
        test_scores,
        positive[test],
        blocks[test],
        seed=RANDOM_SEED + sum(map(ord, method)),
    )
    matched_low, matched_high = bootstrap_matched_substitution_auc(
        test_scores,
        positive[test],
        substitutions[test],
        blocks[test],
        seed=RANDOM_SEED + 10_000 + sum(map(ord, method)),
    )
    encoder = selected_model.named_steps["onehotencoder"]
    return (
        {
            "method": method,
            "selected_c": selected_c,
            "input_columns": design.shape[1],
            "encoded_features": len(encoder.get_feature_names_out()),
            "validation_auroc": selected_validation_auc,
            "validation_average_precision": selected_validation_ap,
            "test_auroc": float(roc_auc_score(positive[test], test_scores)),
            "test_average_precision": float(
                average_precision_score(positive[test], test_scores)
            ),
            **confidence,
            "test_matched_substitution_auroc": matched_substitution_auc(
                test_scores, positive[test], substitutions[test]
            ),
            "test_matched_substitution_auroc_ci95_low": matched_low,
            "test_matched_substitution_auroc_ci95_high": matched_high,
        },
        test_scores,
    )


def sae_substitution_controls(
    *,
    extraction_dir: Path,
    extraction_manifest: dict[str, Any],
    pairwise_metrics: pl.DataFrame,
    selected_rows: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    substitutions: np.ndarray,
    blocks: np.ndarray,
) -> pl.DataFrame:
    test = np.flatnonzero(split == "test")
    positive = labels[test] == POSITIVE_CLASS
    result_rows: list[dict[str, Any]] = []
    for block_index in BLOCK_INDICES:
        for budget in BUDGETS:
            arm = arm_label(block_index, budget)
            paths = {
                "forward": extraction_dir / arm / "sae_activations_forward.parquet",
                "reverse_complement": extraction_dir
                / arm
                / "sae_activations_reverse_complement.parquet",
            }
            for path in paths.values():
                relative = str(path.relative_to(extraction_dir))
                assert relative in extraction_manifest["artifacts"]
                assert (
                    sha256_file(path)
                    == extraction_manifest["artifacts"][relative]["sha256"]
                )
            forward = load_selected_activations(
                paths["forward"], selected_rows=selected_rows, columns=D_SAE
            )
            reverse = load_selected_activations(
                paths["reverse_complement"],
                selected_rows=selected_rows,
                columns=D_SAE,
            )
            views = make_views(forward.delta, reverse.delta)
            for view_name, matrix in views.items():
                selected = pairwise_metrics.filter(
                    (pl.col("arm") == arm) & (pl.col("view") == view_name)
                )
                assert selected.height == 1
                row = selected.row(0, named=True)
                scores = int(row["direction"]) * column_values(
                    matrix, test, int(row["feature_id"])
                )
                np.testing.assert_allclose(
                    roc_auc_score(positive, scores), row["test_auroc"], atol=1e-12
                )
                low, high = bootstrap_matched_substitution_auc(
                    scores,
                    positive,
                    substitutions[test],
                    blocks[test],
                    seed=RANDOM_SEED
                    + block_index * 100_000
                    + budget // 1_000
                    + sum(map(ord, view_name)),
                )
                result_rows.append(
                    {
                        "arm": arm,
                        "reported_block": block_index + 1,
                        "block_index": block_index,
                        "budget": budget,
                        "view": view_name,
                        "feature_id": row["feature_id"],
                        "direction": row["direction"],
                        "test_auroc": row["test_auroc"],
                        "test_auroc_ci95_low": row["test_auroc_ci95_low"],
                        "test_auroc_ci95_high": row["test_auroc_ci95_high"],
                        "test_matched_substitution_auroc": matched_substitution_auc(
                            scores, positive, substitutions[test]
                        ),
                        "test_matched_substitution_auroc_ci95_low": low,
                        "test_matched_substitution_auroc_ci95_high": high,
                    }
                )
            del forward, reverse, views
            print(json.dumps({"stage": "sae_control", "arm": arm}), flush=True)
    output = pl.DataFrame(result_rows).sort(["view", "budget", "reported_block"])
    assert output.height == len(BLOCK_INDICES) * len(BUDGETS) * 4
    return output


def plot_controls(
    baseline_metrics: pl.DataFrame,
    sae_metrics: pl.DataFrame,
    output_dir: Path,
) -> None:
    method_labels = {
        "centered_1mer_alt": "allele change only",
        "centered_3mer_alt": "centered 3-mer + alt",
        "centered_5mer_alt": "centered 5-mer + alt",
        "centered_7mer_alt": "centered 7-mer + alt",
        "positional_31bp_alt": "positional 31 bp + alt",
    }
    selected_sae = []
    for arm, view in TARGET_SAE_ROWS:
        row = sae_metrics.filter((pl.col("arm") == arm) & (pl.col("view") == view))
        assert row.height == 1
        selected_sae.append(row.row(0, named=True))
    plot_rows: list[dict[str, Any]] = []
    for row in baseline_metrics.iter_rows(named=True):
        method = method_labels[row["method"]]
        for comparison, prefix in (
            ("overall", "test_auroc"),
            ("within same ref→alt", "test_matched_substitution_auroc"),
        ):
            plot_rows.append(
                {
                    "method": method,
                    "comparison": comparison,
                    "auroc": row[prefix],
                    "low": row[f"{prefix}_ci95_low"],
                    "high": row[f"{prefix}_ci95_high"],
                }
            )
    for row in selected_sae:
        view_label = "signed mean" if row["view"] == "signed_mean" else "max abs"
        method = f"SAE {row['arm']} {view_label}"
        for comparison, prefix in (
            ("overall", "test_auroc"),
            ("within same ref→alt", "test_matched_substitution_auroc"),
        ):
            plot_rows.append(
                {
                    "method": method,
                    "comparison": comparison,
                    "auroc": row[prefix],
                    "low": row[f"{prefix}_ci95_low"],
                    "high": row[f"{prefix}_ci95_high"],
                }
            )
    method_order = list(method_labels.values()) + [
        "SAE block19-5m signed mean",
        "SAE block19-25m signed mean",
        "SAE block19-5m max abs",
    ]
    plot_frame = pl.DataFrame(plot_rows).to_pandas()
    plot_frame["method"] = (
        plot_frame["method"]
        .astype("category")
        .cat.set_categories(method_order, ordered=True)
    )
    grid = sns.relplot(
        data=plot_frame,
        x="auroc",
        y="method",
        hue="comparison",
        style="comparison",
        kind="scatter",
        s=80,
        palette={"overall": "#4C78A8", "within same ref→alt": "#E45756"},
        height=6.2,
        aspect=1.25,
    )
    axis = grid.ax
    for row in plot_rows:
        axis.errorbar(
            row["auroc"],
            row["method"],
            xerr=np.asarray(
                [
                    [max(0.0, row["auroc"] - row["low"])],
                    [max(0.0, row["high"] - row["auroc"])],
                ]
            ),
            fmt="none",
            capsize=2.5,
            linewidth=1,
            color=("#4C78A8" if row["comparison"] == "overall" else "#E45756"),
            alpha=0.65,
        )
    axis.axvline(PAIRWISE_CHANCE, color="black", linestyle="--", linewidth=1)
    axis.set_xlim(0.38, 0.9)
    grid.set_axis_labels("Held-out missense vs synonymous AUROC", "")
    grid.figure.suptitle(
        "Simple sequence controls versus SAE features\n"
        "error bars = genomic-block bootstrap 95% CI",
        y=1.04,
    )
    for suffix in ("png", "svg"):
        grid.figure.savefig(
            output_dir / f"sequence_controls.{suffix}",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(grid.figure)


def analyze_controls(
    *,
    extraction_dir: Path,
    extraction_manifest_path: Path,
    pairwise_metrics_path: Path,
    panel_path: Path,
    fasta_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert extraction_dir.is_dir() and extraction_manifest_path.is_file()
    assert pairwise_metrics_path.is_file() and panel_path.is_file()
    assert fasta_path.is_file() and not output_dir.exists()
    analysis_commit = os.environ.get("ANALYSIS_COMMIT", "")
    assert_commit(analysis_commit)
    extraction_manifest = json.loads(extraction_manifest_path.read_text())
    assert extraction_manifest["issue"] == ISSUE
    assert sha256_file(panel_path) == extraction_manifest["panel"]["sha256"]

    panel = pl.read_parquet(panel_path)
    pair_panel = panel.filter(pl.col("consequence_cre").is_in(PAIR_CLASSES))
    assert pair_panel.height == 1_024
    selected_rows = pair_panel["panel_row"].to_numpy().astype(np.int64)
    assert np.all(selected_rows[:-1] < selected_rows[1:])
    labels = pair_panel["consequence_cre"].to_numpy()
    positive = labels == POSITIVE_CLASS
    split = pair_panel["split"].to_numpy()
    blocks = pair_panel["block_id"].to_numpy()
    substitutions = (
        pair_panel["ref"] + pl.Series([">"] * pair_panel.height) + pair_panel["alt"]
    ).to_numpy()
    assert set(np.unique(labels)) == {POSITIVE_CLASS, NEGATIVE_CLASS}
    assert len(np.unique(substitutions)) == 12

    contexts = extract_contexts(pair_panel, fasta_path)
    designs = make_baseline_designs(contexts, pair_panel["alt"].to_numpy())
    baseline_rows: list[dict[str, Any]] = []
    for method, design in designs.items():
        row, _ = fit_baseline(
            design,
            positive,
            split,
            substitutions,
            blocks,
            method=method,
        )
        baseline_rows.append(row)
    baseline_metrics = pl.DataFrame(baseline_rows).sort("method")

    pairwise_metrics = pl.read_parquet(pairwise_metrics_path)
    assert pairwise_metrics.height == len(BLOCK_INDICES) * len(BUDGETS) * 4
    sae_metrics = sae_substitution_controls(
        extraction_dir=extraction_dir,
        extraction_manifest=extraction_manifest,
        pairwise_metrics=pairwise_metrics,
        selected_rows=selected_rows,
        labels=labels,
        split=split,
        substitutions=substitutions,
        blocks=blocks,
    )

    output_dir.mkdir(parents=True)
    baseline_metrics.write_parquet(output_dir / "baseline_metrics.parquet")
    sae_metrics.write_parquet(output_dir / "sae_substitution_controls.parquet")
    plot_controls(baseline_metrics, sae_metrics, output_dir)
    best_baseline = baseline_metrics.sort(
        ["validation_auroc", "method"], descending=[True, False]
    ).row(0, named=True)
    result = {
        "analysis_commit": analysis_commit,
        "issue": ISSUE,
        "extraction_manifest_sha256": sha256_file(extraction_manifest_path),
        "pairwise_metrics_sha256": sha256_file(pairwise_metrics_path),
        "panel_sha256": sha256_file(panel_path),
        "fasta": {
            "path_name": fasta_path.name,
            "bytes": fasta_path.stat().st_size,
            "sha256": sha256_file(fasta_path),
        },
        "rows": pair_panel.height,
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
        "best_validation_selected_baseline": best_baseline,
        "protocol": {
            "context_radius_bp": CONTEXT_RADIUS,
            "c_values": list(C_VALUES),
            "block_bootstraps": BOOTSTRAPS,
            "matched_auc": "pair-weighted AUROC restricted to positive/negative pairs sharing the exact ref>alt substitution",
            "feature_selection": "fit discovery; choose logistic C by validation AUROC; report held-out genomic blocks without refitting",
            "status": "exploratory control specified after the direct coding contrast",
        },
    }
    write_json(output_dir / "results.json", result)
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {**result, "artifacts": artifacts}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-dir", type=Path, required=True)
    parser.add_argument("--extraction-manifest", type=Path, required=True)
    parser.add_argument("--pairwise-metrics", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = analyze_controls(
        extraction_dir=args.extraction_dir,
        extraction_manifest_path=args.extraction_manifest,
        pairwise_metrics_path=args.pairwise_metrics,
        panel_path=args.panel,
        fasta_path=args.fasta,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
