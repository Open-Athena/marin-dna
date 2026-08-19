"""Block-bootstrap analysis for conservation × repeat predictability (#478)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Score:
    kind: str
    model_from: str
    model_to: str
    values: np.ndarray
    fraction_positive: bool


def _matrix(frame: pd.DataFrame, column: str, width: int) -> np.ndarray:
    values = np.stack(frame[column].to_numpy()).astype(np.float32, copy=False)
    assert values.shape == (len(frame), width), (
        f"{column} has shape {values.shape}, expected {(len(frame), width)}"
    )
    return values


def load_rc_averaged_atoms(
    fwd_path: str | Path,
    rc_path: str | Path,
    window_ids: pd.Series,
    *,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load separate orientations and average after their genomic realignment."""
    fwd, rc = pd.read_parquet(fwd_path), pd.read_parquet(rc_path)
    expected_ids = window_ids.tolist()
    assert fwd["window_id"].tolist() == expected_ids
    assert rc["window_id"].tolist() == expected_ids
    nll = (_matrix(fwd, "nll", width) + _matrix(rc, "nll", width)) / 2
    entropy = (
        _matrix(fwd, "entropy_4nuc", width) + _matrix(rc, "entropy_4nuc", width)
    ) / 2
    assert np.isfinite(nll).all() and np.isfinite(entropy).all()
    return nll, entropy


def make_scores(
    nll_by_model: dict[str, np.ndarray],
    entropy_46m: np.ndarray,
    model_order: list[str],
) -> list[Score]:
    """Prespecified absolute, entropy, endpoint, and adjacent-rung scores."""
    assert list(nll_by_model) == model_order
    scores = [
        Score("absolute_nll", model, model, nll_by_model[model], False)
        for model in model_order
    ]
    scores.append(
        Score(
            "predictive_entropy_46m",
            model_order[0],
            model_order[0],
            entropy_46m,
            False,
        )
    )
    scores.append(
        Score(
            "endpoint_delta",
            model_order[0],
            model_order[-1],
            nll_by_model[model_order[0]] - nll_by_model[model_order[-1]],
            True,
        )
    )
    scores.extend(
        Score(
            "adjacent_delta",
            smaller,
            larger,
            nll_by_model[smaller] - nll_by_model[larger],
            True,
        )
        for smaller, larger in pairwise(model_order)
    )
    return scores


def _block_bootstrap_mean(
    values: np.ndarray,
    blocks: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    unique, inverse = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    assert (counts > 0).all()
    draws = rng.integers(0, len(unique), size=(replicates, len(unique)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high), len(unique)


def _summary_row(
    values: np.ndarray,
    blocks: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    assert len(values) > 0 and np.isfinite(values).all()
    ci_low, ci_high, n_blocks = _block_bootstrap_mean(
        values, blocks, replicates=replicates, rng=rng
    )
    q10, q50, q90 = np.quantile(values, [0.1, 0.5, 0.9])
    return {
        "n_positions": len(values),
        "n_blocks": n_blocks,
        "mean": float(values.mean(dtype=np.float64)),
        "sd": float(values.std(dtype=np.float64)),
        "q10": float(q10),
        "median": float(q50),
        "q90": float(q90),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def summarize_primary(
    scores: list[Score],
    *,
    region: str,
    conserved: np.ndarray,
    repeat: np.ndarray,
    ambiguous: np.ndarray,
    block: np.ndarray,
    positions: np.ndarray,
    primary_start: int,
    primary_end_exclusive: int,
    replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for span, position_mask in (
        (
            "central_32_222",
            (positions >= primary_start) & (positions < primary_end_exclusive),
        ),
        ("all_255", np.ones_like(positions, dtype=bool)),
    ):
        base_valid = position_mask & ~ambiguous
        for score in scores:
            flat_score = score.values.reshape(-1)
            assert len(flat_score) == len(base_valid)
            for is_conserved in (False, True):
                for is_repeat in (False, True):
                    mask = (
                        base_valid
                        & (conserved == is_conserved)
                        & (repeat == is_repeat)
                        & np.isfinite(flat_score)
                    )
                    if not mask.any():
                        continue
                    values = flat_score[mask].astype(np.float64, copy=False)
                    row: dict[str, Any] = {
                        "analysis_family": "primary",
                        "span": span,
                        "region": region,
                        "feature": "all",
                        "conserved": is_conserved,
                        "repeat": is_repeat,
                        "score_kind": score.kind,
                        "model_from": score.model_from,
                        "model_to": score.model_to,
                        **_summary_row(
                            values,
                            block[mask],
                            replicates=replicates,
                            rng=rng,
                        ),
                    }
                    row["fraction_positive"] = (
                        float((values > 0).mean())
                        if score.fraction_positive
                        else np.nan
                    )
                    rows.append(row)
    return rows


def summarize_cds_secondary(
    scores: list[Score],
    *,
    conserved: np.ndarray,
    repeat: np.ndarray,
    ambiguous: np.ndarray,
    block: np.ndarray,
    positions: np.ndarray,
    codon_position: np.ndarray,
    codon_strand: np.ndarray,
    splice_class: np.ndarray,
    splice_strand: np.ndarray,
    primary_start: int,
    primary_end_exclusive: int,
    replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """CDS-only, central-span secondary diagnostics."""
    rows: list[dict[str, Any]] = []
    base_valid = (
        (positions >= primary_start) & (positions < primary_end_exclusive) & ~ambiguous
    )
    feature_sets = (
        (
            "secondary_codon",
            codon_position,
            codon_strand,
            ((1, "codon_1"), (2, "codon_2"), (3, "codon_3")),
        ),
        (
            "secondary_splice",
            splice_class,
            splice_strand,
            ((1, "splice_donor_2bp"), (2, "splice_acceptor_2bp")),
        ),
    )
    for family, labels, strands, levels in feature_sets:
        for label, feature in levels:
            for strand_code, strand_name in ((1, "plus"), (-1, "minus")):
                feature_mask = (labels == label) & (strands == strand_code)
                for score in scores:
                    flat_score = score.values.reshape(-1)
                    for is_conserved in (False, True):
                        for is_repeat in (False, True):
                            mask = (
                                base_valid
                                & feature_mask
                                & (conserved == is_conserved)
                                & (repeat == is_repeat)
                                & np.isfinite(flat_score)
                            )
                            if not mask.any():
                                continue
                            values = flat_score[mask].astype(np.float64, copy=False)
                            row: dict[str, Any] = {
                                "analysis_family": family,
                                "span": "central_32_222",
                                "region": "cds",
                                "feature": feature,
                                "feature_strand": strand_name,
                                "conserved": is_conserved,
                                "repeat": is_repeat,
                                "score_kind": score.kind,
                                "model_from": score.model_from,
                                "model_to": score.model_to,
                                **_summary_row(
                                    values,
                                    block[mask],
                                    replicates=replicates,
                                    rng=rng,
                                ),
                            }
                            row["fraction_positive"] = (
                                float((values > 0).mean())
                                if score.fraction_positive
                                else np.nan
                            )
                            rows.append(row)
    return rows


def _controlled_fit(
    y: np.ndarray,
    *,
    conserved: np.ndarray,
    repeat: np.ndarray,
    window_gc: np.ndarray,
    kmer_nll: np.ndarray,
    positions: np.ndarray,
    blocks: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """OLS controls with 10-Mb-block bootstrap confidence intervals."""
    valid = np.isfinite(y) & np.isfinite(window_gc) & np.isfinite(kmer_nll)
    y = y[valid].astype(np.float64, copy=False)
    cons = conserved[valid].astype(np.float64)
    rep = repeat[valid].astype(np.float64)
    gc = window_gc[valid].astype(np.float64) - 0.5
    kmer = kmer_nll[valid].astype(np.float64) - np.log(4)
    pos = (positions[valid].astype(np.float64) - 127.0) / 127.0
    block = blocks[valid]
    names = [
        "intercept",
        "conserved",
        "repeat",
        "conserved_x_repeat",
        "window_gc",
        "window_gc_sq",
        "kmer7_nll",
        "kmer7_nll_sq",
        "target_position",
        "target_position_sq",
        "target_position_cu",
    ]
    design = np.column_stack(
        [
            np.ones(len(y)),
            cons,
            rep,
            cons * rep,
            gc,
            gc**2,
            kmer,
            kmer**2,
            pos,
            pos**2,
            pos**3,
        ]
    )
    beta = np.linalg.lstsq(design, y, rcond=None)[0]

    unique = np.unique(block)
    xtx = np.empty((len(unique), len(names), len(names)), dtype=np.float64)
    xty = np.empty((len(unique), len(names)), dtype=np.float64)
    for index, block_id in enumerate(unique):
        subset = block == block_id
        x_block = design[subset]
        xtx[index] = x_block.T @ x_block
        xty[index] = x_block.T @ y[subset]
    draws = rng.integers(0, len(unique), size=(replicates, len(unique)))
    boot = np.empty((replicates, len(names)), dtype=np.float64)
    for index, draw in enumerate(draws):
        draw_xtx = xtx[draw].sum(axis=0)
        draw_xty = xty[draw].sum(axis=0)
        boot[index] = np.linalg.lstsq(draw_xtx, draw_xty, rcond=None)[0]

    rows = []
    for index, name in enumerate(names):
        low, high = np.quantile(boot[:, index], [0.025, 0.975])
        rows.append(
            {
                "term": name,
                "estimate": float(beta[index]),
                "ci_low": float(low),
                "ci_high": float(high),
                "n_positions": len(y),
                "n_blocks": len(unique),
            }
        )
    return rows


def analyze_predictability_478(
    joined_paths: dict[str, str | Path],
    atom_paths: dict[tuple[str, str, str], str | Path],
    *,
    model_order: list[str],
    window_size: int,
    primary_start: int,
    primary_end_exclusive: int,
    block_bp: int,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Analyze all prespecified scores, controls, and CDS-only diagnostics."""
    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, Any]] = []
    controlled_rows: list[dict[str, Any]] = []
    exclusions: dict[str, Any] = {}
    for region, joined_path in joined_paths.items():
        joined = pd.read_parquet(joined_path)
        assert (joined["region"] == region).all()
        assert ((joined["end"] - joined["start"]) == window_size).all()
        n_windows = len(joined)
        positions = np.tile(np.arange(window_size, dtype=np.int16), n_windows)
        conserved = (
            _matrix(joined, "is_conserved", window_size).reshape(-1).astype(bool)
        )
        repeat = _matrix(joined, "is_repeat", window_size).reshape(-1).astype(bool)
        ambiguous = (
            _matrix(joined, "is_ambiguous", window_size).reshape(-1).astype(bool)
        )
        kmer_nll = _matrix(joined, "kmer7_nll", window_size).reshape(-1)
        window_gc = np.repeat(
            joined["window_gc"].to_numpy(dtype=np.float32), window_size
        )
        block_labels = (
            joined["chrom"].astype(str)
            + ":"
            + (joined["start"] // block_bp).astype(str)
        )
        block = np.repeat(pd.factorize(block_labels)[0], window_size)

        nll_by_model: dict[str, np.ndarray] = {}
        entropy_46m: np.ndarray | None = None
        for model in model_order:
            nll, entropy = load_rc_averaged_atoms(
                atom_paths[(model, region, "fwd")],
                atom_paths[(model, region, "rc")],
                joined["window_id"],
                width=window_size,
            )
            nll_by_model[model] = nll
            if model == model_order[0]:
                entropy_46m = entropy
        assert entropy_46m is not None
        scores = make_scores(nll_by_model, entropy_46m, model_order)
        summary_rows.extend(
            summarize_primary(
                scores,
                region=region,
                conserved=conserved,
                repeat=repeat,
                ambiguous=ambiguous,
                block=block,
                positions=positions,
                primary_start=primary_start,
                primary_end_exclusive=primary_end_exclusive,
                replicates=bootstrap_replicates,
                rng=rng,
            )
        )

        central = (
            (positions >= primary_start)
            & (positions < primary_end_exclusive)
            & ~ambiguous
            & np.isfinite(kmer_nll)
        )
        controlled_scores = [
            score
            for score in scores
            if score.kind in {"predictive_entropy_46m", "endpoint_delta"}
            or (
                score.kind == "absolute_nll"
                and score.model_from in {model_order[0], model_order[-1]}
            )
        ]
        for score in controlled_scores:
            for row in _controlled_fit(
                score.values.reshape(-1)[central],
                conserved=conserved[central],
                repeat=repeat[central],
                window_gc=window_gc[central],
                kmer_nll=kmer_nll[central],
                positions=positions[central],
                blocks=block[central],
                replicates=bootstrap_replicates,
                rng=rng,
            ):
                controlled_rows.append(
                    {
                        "region": region,
                        "score_kind": score.kind,
                        "model_from": score.model_from,
                        "model_to": score.model_to,
                        **row,
                    }
                )
            secondary_scores = [
                score
                for score in scores
                if score.kind in {"predictive_entropy_46m", "endpoint_delta"}
                or (score.kind == "absolute_nll" and score.model_from == model_order[0])
            ]

        if region == "cds":
            summary_rows.extend(
                summarize_cds_secondary(
                    secondary_scores,
                    conserved=conserved,
                    repeat=repeat,
                    ambiguous=ambiguous,
                    block=block,
                    positions=positions,
                    codon_position=_matrix(
                        joined, "codon_position", window_size
                    ).reshape(-1),
                    codon_strand=_matrix(joined, "codon_strand", window_size).reshape(
                        -1
                    ),
                    splice_class=_matrix(joined, "splice_class", window_size).reshape(
                        -1
                    ),
                    splice_strand=_matrix(joined, "splice_strand", window_size).reshape(
                        -1
                    ),
                    primary_start=primary_start,
                    primary_end_exclusive=primary_end_exclusive,
                    replicates=bootstrap_replicates,
                    rng=rng,
                )
            )
        exclusions[region] = {
            "n_windows": n_windows,
            "n_positions": n_windows * window_size,
            "n_ambiguous": int(ambiguous.sum()),
            "n_kmer_control_missing": int((~np.isfinite(kmer_nll)).sum()),
            "n_edge_positions_excluded_primary": n_windows
            * (window_size - (primary_end_exclusive - primary_start)),
        }

    summary = pd.DataFrame(summary_rows)
    controlled = pd.DataFrame(controlled_rows)
    manifest = {
        "seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_unit": f"genomic block ({block_bp} bp)",
        "primary_span": [primary_start, primary_end_exclusive],
        "models": model_order,
        "exclusions": exclusions,
        "control_formula": (
            "score ~ conserved * repeat + window_gc + window_gc^2 + "
            "kmer7_nll + kmer7_nll^2 + position + position^2 + position^3"
        ),
        "secondary_scope": "CDS only; codon position and canonical 2-bp splice sites",
    }
    return summary, controlled, manifest
