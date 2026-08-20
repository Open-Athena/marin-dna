"""Conservation classification from issue #478 per-base model statistics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from marin_dna_evals.analysis_478 import load_rc_averaged_atoms

_BLOCK_FACTOR = 1_000_000


@dataclass(frozen=True)
class RegionScores:
    """Eligible non-repeat positions and model statistics for one region."""

    region: str
    labels: np.ndarray
    blocks: np.ndarray
    nll_by_model: dict[str, np.ndarray]
    entropy_by_model: dict[str, np.ndarray]
    chrom_by_code: dict[int, str]


@dataclass(frozen=True)
class ScoreSpec:
    """One ranking statistic evaluated as a conservation classifier."""

    statistic: str
    model_from: str
    model_to: str


def _matrix(frame: pd.DataFrame, column: str, width: int) -> np.ndarray:
    values = np.stack(frame[column].to_numpy())
    assert values.shape == (len(frame), width), (
        f"{column} has shape {values.shape}, expected {(len(frame), width)}"
    )
    return values


def _chromosome_code(accession: str) -> int:
    match = re.fullmatch(r"NC_(\d+)\.\d+", accession)
    assert match is not None, f"unexpected RefSeq accession {accession!r}"
    return int(match.group(1))


def _score_specs(model_order: list[str]) -> list[ScoreSpec]:
    specs = [
        ScoreSpec(statistic, model, model)
        for statistic in ("loss", "entropy")
        for model in model_order
    ]
    specs.extend(
        ScoreSpec("loss_delta", smaller, larger)
        for smaller, larger in combinations(model_order, 2)
    )
    return specs


def _score_values(data: RegionScores, spec: ScoreSpec) -> np.ndarray:
    if spec.statistic == "loss":
        return -data.nll_by_model[spec.model_from]
    if spec.statistic == "entropy":
        return -data.entropy_by_model[spec.model_from]
    if spec.statistic == "loss_delta":
        return data.nll_by_model[spec.model_from] - data.nll_by_model[spec.model_to]
    raise ValueError(f"unknown statistic {spec.statistic!r}")


def _load_region_scores(
    joined_path: str | Path,
    atom_paths: dict[tuple[str, str], str | Path],
    *,
    region: str,
    model_order: list[str],
    window_size: int,
    primary_start: int,
    primary_end_exclusive: int,
    block_bp: int,
) -> RegionScores:
    joined = pd.read_parquet(joined_path)
    assert len(joined) > 0
    assert primary_start >= 0
    assert primary_start < primary_end_exclusive <= window_size
    expected_ids = joined["window_id"]
    span = slice(primary_start, primary_end_exclusive)

    conserved = _matrix(joined, "is_conserved", window_size)[:, span].astype(
        bool, copy=False
    )
    repeat = _matrix(joined, "is_repeat", window_size)[:, span].astype(bool, copy=False)
    ambiguous = _matrix(joined, "is_ambiguous", window_size)[:, span].astype(
        bool, copy=False
    )
    eligible = ~repeat & ~ambiguous
    labels = conserved[eligible]
    assert labels.any() and (~labels).any()

    chrom_by_code = {
        _chromosome_code(accession): accession
        for accession in joined["chrom"].drop_duplicates()
    }
    chrom_codes = joined["chrom"].map(_chromosome_code).to_numpy(dtype=np.int64)
    offsets = np.arange(
        primary_start,
        primary_end_exclusive,
        dtype=np.int64,
    )
    genomic_positions = joined["start"].to_numpy(dtype=np.int64)[:, None] + offsets
    block_numbers = genomic_positions // block_bp
    block_matrix = chrom_codes[:, None] * _BLOCK_FACTOR + block_numbers
    blocks = block_matrix[eligible]

    nll_by_model: dict[str, np.ndarray] = {}
    entropy_by_model: dict[str, np.ndarray] = {}
    for model in model_order:
        nll, entropy = load_rc_averaged_atoms(
            atom_paths[(model, "fwd")],
            atom_paths[(model, "rc")],
            expected_ids,
            width=window_size,
        )
        nll_values = nll[:, span][eligible].astype(np.float32, copy=False)
        entropy_values = entropy[:, span][eligible].astype(np.float32, copy=False)
        assert np.isfinite(nll_values).all()
        assert np.isfinite(entropy_values).all()
        nll_by_model[model] = nll_values
        entropy_by_model[model] = entropy_values

    return RegionScores(
        region=region,
        labels=labels,
        blocks=blocks,
        nll_by_model=nll_by_model,
        entropy_by_model=entropy_by_model,
        chrom_by_code=chrom_by_code,
    )


def _scope_score(
    data: list[RegionScores],
    spec: ScoreSpec,
) -> np.ndarray:
    if len(data) == 1:
        return _score_values(data[0], spec)
    return np.concatenate([_score_values(item, spec) for item in data])


def _evaluate_scope(
    data: list[RegionScores],
    *,
    scope: str,
    specs: list[ScoreSpec],
    block_bp: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = (
        data[0].labels
        if len(data) == 1
        else np.concatenate([item.labels for item in data])
    )
    blocks = (
        data[0].blocks
        if len(data) == 1
        else np.concatenate([item.blocks for item in data])
    )
    prevalence = float(labels.mean())
    n_positions = len(labels)
    n_conserved = int(labels.sum())
    unique_blocks = np.unique(blocks)

    block_order = np.argsort(blocks, kind="stable")
    ordered_blocks = blocks[block_order]
    ordered_labels = labels[block_order]
    block_ids, starts = np.unique(ordered_blocks, return_index=True)
    ends = np.r_[starts[1:], len(ordered_blocks)]
    chrom_by_code = {
        code: accession
        for item in data
        for code, accession in item.chrom_by_code.items()
    }

    metrics: list[dict[str, Any]] = []
    block_metrics: list[dict[str, Any]] = []
    for spec in specs:
        score = _scope_score(data, spec)
        assert len(score) == n_positions and np.isfinite(score).all()
        auprc = float(average_precision_score(labels, score))
        metrics.append(
            {
                "scope": scope,
                "statistic": spec.statistic,
                "model_from": spec.model_from,
                "model_to": spec.model_to,
                "orientation": "fwd_rc_mean",
                "auprc": auprc,
                "prevalence": prevalence,
                "auprc_minus_prevalence": auprc - prevalence,
                "n_positions": n_positions,
                "n_conserved": n_conserved,
                "n_blocks": len(unique_blocks),
            }
        )

        ordered_score = score[block_order]
        for block_id, start, end in zip(block_ids, starts, ends, strict=True):
            block_labels = ordered_labels[start:end]
            block_n_conserved = int(block_labels.sum())
            block_n = len(block_labels)
            if block_n_conserved == 0 or block_n_conserved == block_n:
                continue
            block_auprc = float(
                average_precision_score(block_labels, ordered_score[start:end])
            )
            chrom_code, block_number = divmod(int(block_id), _BLOCK_FACTOR)
            block_prevalence = block_n_conserved / block_n
            block_metrics.append(
                {
                    "scope": scope,
                    "statistic": spec.statistic,
                    "model_from": spec.model_from,
                    "model_to": spec.model_to,
                    "orientation": "fwd_rc_mean",
                    "chrom": chrom_by_code[chrom_code],
                    "block_start": block_number * block_bp,
                    "block_end": (block_number + 1) * block_bp,
                    "auprc": block_auprc,
                    "prevalence": block_prevalence,
                    "auprc_minus_prevalence": block_auprc - block_prevalence,
                    "n_positions": block_n,
                    "n_conserved": block_n_conserved,
                }
            )
    return metrics, block_metrics


def analyze_conservation_classification_478(
    joined_paths: dict[str, str | Path],
    atom_paths: dict[tuple[str, str, str], str | Path],
    *,
    model_order: list[str],
    window_size: int,
    primary_start: int,
    primary_end_exclusive: int,
    block_bp: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate model uncertainty and loss deltas as conservation classifiers."""
    assert model_order
    assert block_bp > 0
    regions = list(joined_paths)
    region_data = {
        region: _load_region_scores(
            joined_paths[region],
            {
                (model, orientation): atom_paths[(model, region, orientation)]
                for model in model_order
                for orientation in ("fwd", "rc")
            },
            region=region,
            model_order=model_order,
            window_size=window_size,
            primary_start=primary_start,
            primary_end_exclusive=primary_end_exclusive,
            block_bp=block_bp,
        )
        for region in regions
    }
    specs = _score_specs(model_order)
    metric_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    scopes = [("global", list(region_data.values()))]
    scopes.extend((region, [region_data[region]]) for region in regions)
    for scope, data in scopes:
        scope_metrics, scope_blocks = _evaluate_scope(
            data,
            scope=scope,
            specs=specs,
            block_bp=block_bp,
        )
        metric_rows.extend(scope_metrics)
        block_rows.extend(scope_blocks)

    metrics = pd.DataFrame(metric_rows)
    block_metrics = pd.DataFrame(block_rows)
    manifest = {
        "analysis": "conservation_classification_478",
        "positive_class": "conserved",
        "eligible_positions": "central span, non-repeat, non-ambiguous",
        "span": [primary_start, primary_end_exclusive],
        "coordinate_system": "0-based half-open",
        "orientation": "mean of FWD and genomically realigned RC",
        "metric": "sklearn.metrics.average_precision_score",
        "prevalence_baseline": True,
        "score_directions": {
            "loss": "negative NLL; lower loss ranks as more conserved",
            "entropy": "negative 4-nucleotide entropy; lower entropy ranks as more conserved",
            "loss_delta": "smaller-model NLL minus larger-model NLL",
        },
        "model_order": model_order,
        "model_pairs": len(list(combinations(model_order, 2))),
        "block_bp": block_bp,
        "block_metrics": (
            "within-block AUPRC variation; not a confidence interval for pooled AUPRC"
        ),
        "counts": {
            row["scope"]: {
                "n_positions": int(row["n_positions"]),
                "n_conserved": int(row["n_conserved"]),
                "n_blocks": int(row["n_blocks"]),
            }
            for row in metrics[
                (metrics["statistic"] == "loss")
                & (metrics["model_from"] == model_order[0])
            ].to_dict("records")
        },
    }
    return metrics, block_metrics, manifest
