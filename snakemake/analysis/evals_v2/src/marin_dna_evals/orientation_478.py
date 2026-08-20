"""Single-orientation sensitivity analysis for issue #478."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from marin_dna_evals.analysis_478 import (
    _controlled_fit,
    _matrix,
    make_scores,
    summarize_cds_secondary,
    summarize_primary,
)

ORIENTATIONS = ("fwd", "rc")
AGREEMENT_SCORE_KINDS = (
    "absolute_nll_46m",
    "predictive_entropy_46m",
    "endpoint_delta",
)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.array_equal(left, right):
        return 1.0
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _load_orientation_atom(
    path: str | Path,
    window_ids: pd.Series,
    *,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(path)
    assert frame["window_id"].tolist() == window_ids.tolist()
    nll = _matrix(frame, "nll", width)
    entropy = _matrix(frame, "entropy_4nuc", width)
    assert np.isfinite(nll).all() and np.isfinite(entropy).all()
    return nll, entropy


def _sample_spearman(
    left: np.ndarray,
    right: np.ndarray,
    *,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[float, int]:
    n = len(left)
    if n > sample_size:
        indices = rng.choice(n, size=sample_size, replace=False)
        left = left[indices]
        right = right[indices]
    left_rank = pd.Series(left).rank(method="average").to_numpy()
    right_rank = pd.Series(right).rank(method="average").to_numpy()
    return _safe_correlation(left_rank, right_rank), len(left)


def _tail_overlap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    fraction: float,
    largest: bool,
) -> float:
    assert 0 < fraction < 1
    k = max(1, int(np.ceil(len(left) * fraction)))
    if largest:
        left_indices = np.argpartition(left, len(left) - k)[-k:]
        right_indices = np.argpartition(right, len(right) - k)[-k:]
    else:
        left_indices = np.argpartition(left, k - 1)[:k]
        right_indices = np.argpartition(right, k - 1)[:k]
    return float(np.intersect1d(left_indices, right_indices).size / k)


def orientation_agreement_row(
    left: np.ndarray,
    right: np.ndarray,
    *,
    score_kind: str,
    top_fraction: float,
    rank_sample_size: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Summarize whether one score preserves another per-base candidate score."""
    assert left.shape == right.shape and left.ndim == 1 and len(left) > 1
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid].astype(np.float64, copy=False)
    right = right[valid].astype(np.float64, copy=False)
    assert len(left) > 1

    pearson = _safe_correlation(left, right)
    spearman, rank_n = _sample_spearman(
        left,
        right,
        sample_size=rank_sample_size,
        rng=rng,
    )
    difference = left - right
    return {
        "n_positions": len(left),
        "pearson": pearson,
        "spearman_sample": spearman,
        "spearman_sample_n": rank_n,
        "mean_left": float(left.mean(dtype=np.float64)),
        "mean_right": float(right.mean(dtype=np.float64)),
        "mean_left_minus_right": float(difference.mean(dtype=np.float64)),
        "mae": float(np.abs(difference).mean(dtype=np.float64)),
        "top_fraction": top_fraction,
        "top_fraction_overlap": _tail_overlap(
            left,
            right,
            fraction=top_fraction,
            largest=True,
        ),
        "bottom_fraction_overlap": _tail_overlap(
            left,
            right,
            fraction=top_fraction,
            largest=False,
        ),
        "sign_agreement": (
            float(((left > 0) == (right > 0)).mean())
            if score_kind == "endpoint_delta"
            else float("nan")
        ),
    }


def _agreement_rows(
    candidates: dict[str, dict[str, np.ndarray]],
    *,
    region: str,
    conserved: np.ndarray,
    repeat: np.ndarray,
    ambiguous: np.ndarray,
    positions: np.ndarray,
    primary_start: int,
    primary_end_exclusive: int,
    top_fraction: float,
    rank_sample_size: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    central = (
        (positions >= primary_start) & (positions < primary_end_exclusive) & ~ambiguous
    )
    strata: list[tuple[str, str, np.ndarray]] = [
        ("all", "all", np.ones_like(central, dtype=bool))
    ]
    strata.extend(
        (
            "conserved" if is_conserved else "other",
            "repeat" if is_repeat else "nonrepeat",
            (conserved == is_conserved) & (repeat == is_repeat),
        )
        for is_conserved in (False, True)
        for is_repeat in (False, True)
    )

    rows: list[dict[str, Any]] = []
    for score_kind in AGREEMENT_SCORE_KINDS:
        fwd = candidates["fwd"][score_kind].reshape(-1)
        rc = candidates["rc"][score_kind].reshape(-1)
        mean = (fwd.astype(np.float64) + rc) / 2
        comparisons = (
            ("fwd_vs_rc", "fwd", "rc", fwd, rc),
            ("fwd_vs_mean", "fwd", "fwd_rc_mean", fwd, mean),
            ("rc_vs_mean", "rc", "fwd_rc_mean", rc, mean),
        )
        for comparison, left_name, right_name, left, right in comparisons:
            for conservation, repeat_status, stratum in strata:
                mask = central & stratum
                rows.append(
                    {
                        "region": region,
                        "span": "central_32_222",
                        "conservation": conservation,
                        "repeat_status": repeat_status,
                        "score_kind": score_kind,
                        "comparison": comparison,
                        "left": left_name,
                        "right": right_name,
                        **orientation_agreement_row(
                            left[mask],
                            right[mask],
                            score_kind=score_kind,
                            top_fraction=top_fraction,
                            rank_sample_size=rank_sample_size,
                            rng=rng,
                        ),
                    }
                )
    return rows


def analyze_orientation_sensitivity_478(
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
    top_fraction: float,
    rank_sample_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Analyze FWD-only and RC-only scores without rerunning inference."""
    summary_rng = np.random.default_rng(seed)
    agreement_rng = np.random.default_rng(seed + 1)
    summary_rows: list[dict[str, Any]] = []
    controlled_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
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
            joined["window_gc"].to_numpy(dtype=np.float32),
            window_size,
        )
        block_labels = (
            joined["chrom"].astype(str)
            + ":"
            + (joined["start"] // block_bp).astype(str)
        )
        block = np.repeat(pd.factorize(block_labels)[0], window_size)
        central = (
            (positions >= primary_start)
            & (positions < primary_end_exclusive)
            & ~ambiguous
            & np.isfinite(kmer_nll)
        )

        candidates: dict[str, dict[str, np.ndarray]] = {}
        for orientation in ORIENTATIONS:
            nll_by_model: dict[str, np.ndarray] = {}
            entropy_46m: np.ndarray | None = None
            for model in model_order:
                nll, entropy = _load_orientation_atom(
                    atom_paths[(model, region, orientation)],
                    joined["window_id"],
                    width=window_size,
                )
                nll_by_model[model] = nll
                if model == model_order[0]:
                    entropy_46m = entropy
            assert entropy_46m is not None
            scores = make_scores(nll_by_model, entropy_46m, model_order)

            orientation_summary = summarize_primary(
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
                rng=summary_rng,
            )
            for row in orientation_summary:
                row["orientation"] = orientation
            summary_rows.extend(orientation_summary)

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
                    rng=summary_rng,
                ):
                    controlled_rows.append(
                        {
                            "orientation": orientation,
                            "region": region,
                            "score_kind": score.kind,
                            "model_from": score.model_from,
                            "model_to": score.model_to,
                            **row,
                        }
                    )

            if region == "cds":
                secondary_scores = [
                    score
                    for score in scores
                    if score.kind in {"predictive_entropy_46m", "endpoint_delta"}
                    or (
                        score.kind == "absolute_nll"
                        and score.model_from == model_order[0]
                    )
                ]
                secondary_rows = summarize_cds_secondary(
                    secondary_scores,
                    conserved=conserved,
                    repeat=repeat,
                    ambiguous=ambiguous,
                    block=block,
                    positions=positions,
                    codon_position=_matrix(
                        joined,
                        "codon_position",
                        window_size,
                    ).reshape(-1),
                    codon_strand=_matrix(
                        joined,
                        "codon_strand",
                        window_size,
                    ).reshape(-1),
                    splice_class=_matrix(
                        joined,
                        "splice_class",
                        window_size,
                    ).reshape(-1),
                    splice_strand=_matrix(
                        joined,
                        "splice_strand",
                        window_size,
                    ).reshape(-1),
                    primary_start=primary_start,
                    primary_end_exclusive=primary_end_exclusive,
                    replicates=bootstrap_replicates,
                    rng=summary_rng,
                )
                for row in secondary_rows:
                    row["orientation"] = orientation
                summary_rows.extend(secondary_rows)

            candidates[orientation] = {
                "absolute_nll_46m": nll_by_model[model_order[0]],
                "predictive_entropy_46m": entropy_46m,
                "endpoint_delta": (
                    nll_by_model[model_order[0]] - nll_by_model[model_order[-1]]
                ),
            }

        agreement_rows.extend(
            _agreement_rows(
                candidates,
                region=region,
                conserved=conserved,
                repeat=repeat,
                ambiguous=ambiguous,
                positions=positions,
                primary_start=primary_start,
                primary_end_exclusive=primary_end_exclusive,
                top_fraction=top_fraction,
                rank_sample_size=rank_sample_size,
                rng=agreement_rng,
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

    manifest = {
        "seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_unit": f"genomic block ({block_bp} bp)",
        "primary_span": [primary_start, primary_end_exclusive],
        "orientations": list(ORIENTATIONS),
        "primary_analysis": "FWD/RC mean; these are single-orientation sensitivities",
        "models": model_order,
        "agreement_score_kinds": list(AGREEMENT_SCORE_KINDS),
        "agreement_comparisons": [
            "fwd_vs_rc",
            "fwd_vs_mean",
            "rc_vs_mean",
        ],
        "agreement_span": "central_32_222",
        "top_fraction": top_fraction,
        "rank_sample_size": rank_sample_size,
        "exclusions": exclusions,
        "control_formula": (
            "score ~ conserved * repeat + window_gc + window_gc^2 + "
            "kmer7_nll + kmer7_nll^2 + position + position^2 + position^3"
        ),
        "secondary_scope": "CDS only; codon position and canonical 2-bp splice sites",
    }
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(controlled_rows),
        pd.DataFrame(agreement_rows),
        manifest,
    )
