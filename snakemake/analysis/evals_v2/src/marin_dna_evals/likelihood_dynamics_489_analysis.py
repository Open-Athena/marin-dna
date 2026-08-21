"""Statistical summaries for likelihood dynamics through m1.3 (issue #489)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

CONTROL_TERMS = (
    "intercept",
    "conserved",
    "window_gc",
    "window_gc_sq",
    "kmer7_nll",
    "kmer7_nll_sq",
    "target_position",
    "target_position_sq",
    "target_position_cu",
)


@dataclass(frozen=True)
class RegionData:
    """One region's primary nonrepeat central-span population."""

    region: str
    token_id: np.ndarray
    row_index: np.ndarray
    target_pos: np.ndarray
    conserved: np.ndarray
    window_gc: np.ndarray
    kmer7_nll: np.ndarray
    block: np.ndarray
    n_blocks: int
    nll: np.ndarray
    entropy: np.ndarray

    @property
    def n_positions(self) -> int:
        return len(self.token_id)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest(
    path: str | Path,
    *,
    checkpoint: str,
    checkpoint_order: int,
    cumulative_tokens: int,
    region: str,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    assert payload["artifact_schema_version"] == "v1"
    assert payload["scope"] == "full"
    assert payload["dataset"]["region"] == region
    checkpoint_payload = payload["checkpoint"]
    assert checkpoint_payload["name"] == checkpoint
    assert checkpoint_payload["order"] == checkpoint_order
    assert checkpoint_payload["cumulative_tokens"] == cumulative_tokens
    atom_manifest = payload["atom_manifest"]
    assert atom_manifest["token_identity"] == [
        "region",
        "row_index",
        "target_pos",
    ]
    assert atom_manifest["n_positions"] == atom_manifest["n_scorable"]
    return {
        "checkpoint": checkpoint,
        "checkpoint_order": checkpoint_order,
        "cumulative_tokens": cumulative_tokens,
        "region": region,
        "hf_repo": payload["dataset"]["hf_repo"],
        "hf_revision": payload["dataset"]["hf_revision"],
        "n_positions": atom_manifest["n_positions"],
        "sha256": _sha256(path),
    }


def _load_region(
    atom_paths: dict[tuple[str, str], str | Path],
    manifest_paths: dict[tuple[str, str], str | Path],
    *,
    region: str,
    checkpoints: list[str],
    cumulative_tokens: list[int],
    primary_start: int,
    primary_end_exclusive: int,
    block_bp: int,
) -> tuple[RegionData, dict[str, Any], list[dict[str, Any]]]:
    common_columns = [
        "token_index",
        "row_index",
        "region",
        "chrom",
        "genomic_pos",
        "target_pos",
        "is_conserved",
        "is_repeat",
        "is_ambiguous",
        "is_scorable",
        "window_gc",
        "kmer7_nll",
        "nll",
        "entropy_4nuc",
    ]
    first = pd.read_parquet(
        atom_paths[(checkpoints[0], region)], columns=common_columns
    )
    assert len(first) > 0
    assert (first["region"].astype(str) == region).all()
    assert not first.duplicated(["row_index", "target_pos"]).any()
    assert first["token_index"].is_monotonic_increasing
    primary = (
        (first["target_pos"] >= primary_start)
        & (first["target_pos"] < primary_end_exclusive)
        & first["is_scorable"]
        & ~first["is_ambiguous"]
        & ~first["is_repeat"]
    ).to_numpy(dtype=bool)
    assert primary.any()
    token_id_all = first["token_index"].to_numpy(dtype=np.int64)
    row_index_all = first["row_index"].to_numpy(dtype=np.int32)
    target_pos_all = first["target_pos"].to_numpy(dtype=np.int16)
    token_id = token_id_all[primary]
    row_index = row_index_all[primary]
    target_pos = target_pos_all[primary]
    conserved = first["is_conserved"].to_numpy(dtype=bool)[primary]
    window_gc = first["window_gc"].to_numpy(dtype=np.float32)[primary]
    kmer7_nll = first["kmer7_nll"].to_numpy(dtype=np.float32)[primary]
    block_labels = (
        first.loc[primary, "chrom"].astype(str)
        + ":"
        + (first.loc[primary, "genomic_pos"] // block_bp).astype(str)
    )
    block, block_values = pd.factorize(block_labels, sort=True)
    block = block.astype(np.int32, copy=False)

    nll_rows: list[np.ndarray] = []
    entropy_rows: list[np.ndarray] = []
    input_cells: list[dict[str, Any]] = []
    for checkpoint_order, (checkpoint, tokens) in enumerate(
        zip(checkpoints, cumulative_tokens, strict=True)
    ):
        input_cells.append(
            _validate_manifest(
                manifest_paths[(checkpoint, region)],
                checkpoint=checkpoint,
                checkpoint_order=checkpoint_order,
                cumulative_tokens=tokens,
                region=region,
            )
        )
        if checkpoint_order == 0:
            scores = first
        else:
            scores = pd.read_parquet(
                atom_paths[(checkpoint, region)],
                columns=[
                    "token_index",
                    "row_index",
                    "region",
                    "target_pos",
                    "nll",
                    "entropy_4nuc",
                ],
            )
            assert np.array_equal(
                scores["token_index"].to_numpy(dtype=np.int64),
                token_id_all,
            ), f"{checkpoint}/{region}: token_index changed"
            assert np.array_equal(
                scores["row_index"].to_numpy(dtype=np.int32),
                row_index_all,
            ), f"{checkpoint}/{region}: row_index changed"
            assert np.array_equal(
                scores["target_pos"].to_numpy(dtype=np.int16),
                target_pos_all,
            ), f"{checkpoint}/{region}: target_pos changed"
            assert (scores["region"].astype(str) == region).all()
        nll = scores["nll"].to_numpy(dtype=np.float32)[primary]
        entropy = scores["entropy_4nuc"].to_numpy(dtype=np.float32)[primary]
        assert np.isfinite(nll).all() and (nll >= 0).all()
        assert np.isfinite(entropy).all() and (entropy >= 0).all()
        nll_rows.append(nll)
        entropy_rows.append(entropy)

    n_total = len(first)
    population = {
        "region": region,
        "n_windows": int(first["row_index"].nunique()),
        "n_positions_cached": n_total,
        "n_primary_span": int(
            (
                (first["target_pos"] >= primary_start)
                & (first["target_pos"] < primary_end_exclusive)
            ).sum()
        ),
        "n_repeat_excluded": int(
            (
                (first["target_pos"] >= primary_start)
                & (first["target_pos"] < primary_end_exclusive)
                & first["is_repeat"]
            ).sum()
        ),
        "n_ambiguous_or_unscorable_excluded": int(
            (
                (first["target_pos"] >= primary_start)
                & (first["target_pos"] < primary_end_exclusive)
                & (~first["is_scorable"] | first["is_ambiguous"])
            ).sum()
        ),
        "n_primary_nonrepeat_scorable": int(primary.sum()),
        "n_conserved": int(conserved.sum()),
        "prevalence": float(conserved.mean()),
        "n_blocks": len(block_values),
    }
    data = RegionData(
        region=region,
        token_id=token_id,
        row_index=row_index,
        target_pos=target_pos,
        conserved=conserved,
        window_gc=window_gc,
        kmer7_nll=kmer7_nll,
        block=block,
        n_blocks=len(block_values),
        nll=np.stack(nll_rows),
        entropy=np.stack(entropy_rows),
    )
    del first
    return data, population, input_cells


def _combine_regions(regions: list[RegionData]) -> RegionData:
    block_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    block_offset = 0
    row_offset = 0
    for data in regions:
        block_parts.append(data.block + block_offset)
        row_parts.append(data.row_index.astype(np.int64) + row_offset)
        block_offset += data.n_blocks
        row_offset += int(data.row_index.max()) + 1
    n_positions = sum(data.n_positions for data in regions)
    return RegionData(
        region="global",
        token_id=np.arange(n_positions, dtype=np.int64),
        row_index=np.concatenate(row_parts),
        target_pos=np.concatenate([data.target_pos for data in regions]),
        conserved=np.concatenate([data.conserved for data in regions]),
        window_gc=np.concatenate([data.window_gc for data in regions]),
        kmer7_nll=np.concatenate([data.kmer7_nll for data in regions]),
        block=np.concatenate(block_parts),
        n_blocks=block_offset,
        nll=np.concatenate([data.nll for data in regions], axis=1),
        entropy=np.concatenate([data.entropy for data in regions], axis=1),
    )


def _summary(
    values: np.ndarray,
    blocks: np.ndarray,
    *,
    n_block_levels: int,
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    assert len(values) > 0 and np.isfinite(values).all()
    block_sums = np.bincount(
        blocks,
        weights=values,
        minlength=n_block_levels,
    )
    block_counts = np.bincount(blocks, minlength=n_block_levels)
    active = block_counts > 0
    block_sums = block_sums[active]
    block_counts = block_counts[active]
    assert len(block_counts) > 0
    draws = rng.integers(
        0,
        len(block_counts),
        size=(bootstrap_replicates, len(block_counts)),
    )
    bootstrap_means = block_sums[draws].sum(axis=1) / block_counts[draws].sum(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    q10, q25, median, q75, q90 = np.quantile(
        values,
        [0.10, 0.25, 0.50, 0.75, 0.90],
    )
    return {
        "n_positions": len(values),
        "n_blocks": len(block_counts),
        "mean": float(values.mean()),
        "sd": float(values.std()),
        "q10": float(q10),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "q90": float(q90),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def _classification_rows(
    data: RegionData,
    *,
    checkpoints: list[str],
    cumulative_tokens: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = data.conserved
    assert labels.any() and (~labels).any()
    for checkpoint_order, (checkpoint, tokens) in enumerate(
        zip(checkpoints, cumulative_tokens, strict=True)
    ):
        for statistic, values in (
            ("loss", data.nll[checkpoint_order]),
            ("entropy", data.entropy[checkpoint_order]),
        ):
            auprc = average_precision_score(labels, -values)
            rows.append(
                {
                    "scope": data.region,
                    "statistic": statistic,
                    "checkpoint": checkpoint,
                    "checkpoint_order": checkpoint_order,
                    "cumulative_tokens": tokens,
                    "auprc": float(auprc),
                    "prevalence": float(labels.mean()),
                    "auprc_minus_prevalence": float(auprc - labels.mean()),
                    "n_positions": len(labels),
                    "n_conserved": int(labels.sum()),
                }
            )
    return rows


def _trajectory_groups(data: RegionData) -> tuple[np.ndarray, float, float]:
    early_mean = float(data.nll[0].mean(dtype=np.float64))
    terminal_mean = float(data.nll[-1].mean(dtype=np.float64))
    early_low = data.nll[0] <= early_mean
    terminal_low = data.nll[-1] <= terminal_mean
    groups = np.full(data.n_positions, "high_to_high", dtype=object)
    groups[early_low & terminal_low] = "low_to_low"
    groups[early_low & ~terminal_low] = "low_to_high"
    groups[~early_low & terminal_low] = "high_to_low"
    return groups, early_mean, terminal_mean


def _trajectory_rows(
    data: RegionData,
    groups: np.ndarray,
    *,
    checkpoints: list[str],
    cumulative_tokens: list[int],
    early_mean: float | None,
    terminal_mean: float | None,
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in ("low_to_low", "low_to_high", "high_to_low", "high_to_high"):
        mask = groups == group
        if not mask.any():
            continue
        frequency = float(mask.mean())
        conservation_prevalence = float(data.conserved[mask].mean())
        for checkpoint_order, (checkpoint, tokens) in enumerate(
            zip(checkpoints, cumulative_tokens, strict=True)
        ):
            rows.append(
                {
                    "scope": data.region,
                    "group": group,
                    "checkpoint": checkpoint,
                    "checkpoint_order": checkpoint_order,
                    "cumulative_tokens": tokens,
                    "frequency": frequency,
                    "conservation_prevalence": conservation_prevalence,
                    "early_region_mean_nll": early_mean,
                    "terminal_region_mean_nll": terminal_mean,
                    **_summary(
                        data.nll[checkpoint_order, mask],
                        data.block[mask],
                        n_block_levels=data.n_blocks,
                        bootstrap_replicates=bootstrap_replicates,
                        rng=rng,
                    ),
                }
            )
    return rows


def _lowest_fraction_mask(
    values: np.ndarray,
    token_id: np.ndarray,
    *,
    fraction: float,
) -> np.ndarray:
    assert 0 < fraction < 1
    n_selected = max(1, int(np.floor(len(values) * fraction)))
    order = np.lexsort((token_id, values))
    selected = np.zeros(len(values), dtype=bool)
    selected[order[:n_selected]] = True
    assert selected.sum() == n_selected
    return selected


def _checkpoint_pairs(n_checkpoints: int) -> list[tuple[int, int, str]]:
    pairs = [(index, index + 1, "adjacent") for index in range(n_checkpoints - 1)]
    endpoint = (0, n_checkpoints - 1)
    if endpoint not in {(left, right) for left, right, _ in pairs}:
        pairs.append((*endpoint, "endpoint"))
    return pairs


def _selection_rows(
    regions: list[RegionData],
    *,
    checkpoints: list[str],
    cumulative_tokens: list[int],
    fraction: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pairs = _checkpoint_pairs(len(checkpoints))
    selected_by_region: dict[tuple[str, str, int], np.ndarray] = {}
    for data in regions:
        for statistic, matrix in (("loss", data.nll), ("entropy", data.entropy)):
            for checkpoint_order in range(len(checkpoints)):
                selected_by_region[(data.region, statistic, checkpoint_order)] = (
                    _lowest_fraction_mask(
                        matrix[checkpoint_order],
                        data.token_id,
                        fraction=fraction,
                    )
                )
            for left, right, comparison in pairs:
                first = selected_by_region[(data.region, statistic, left)]
                second = selected_by_region[(data.region, statistic, right)]
                intersection = int((first & second).sum())
                union = int((first | second).sum())
                rows.append(
                    {
                        "scope": data.region,
                        "statistic": statistic,
                        "comparison": comparison,
                        "checkpoint_from": checkpoints[left],
                        "checkpoint_to": checkpoints[right],
                        "checkpoint_order_from": left,
                        "checkpoint_order_to": right,
                        "cumulative_tokens_from": cumulative_tokens[left],
                        "cumulative_tokens_to": cumulative_tokens[right],
                        "n_positions": data.n_positions,
                        "n_selected_from": int(first.sum()),
                        "n_selected_to": int(second.sum()),
                        "intersection": intersection,
                        "union": union,
                        "jaccard": intersection / union,
                    }
                )

    for statistic in ("loss", "entropy"):
        for left, right, comparison in pairs:
            selections = [
                (
                    selected_by_region[(data.region, statistic, left)],
                    selected_by_region[(data.region, statistic, right)],
                )
                for data in regions
            ]
            intersection = sum(
                int((first & second).sum()) for first, second in selections
            )
            union = sum(int((first | second).sum()) for first, second in selections)
            rows.append(
                {
                    "scope": "global",
                    "statistic": statistic,
                    "comparison": comparison,
                    "checkpoint_from": checkpoints[left],
                    "checkpoint_to": checkpoints[right],
                    "checkpoint_order_from": left,
                    "checkpoint_order_to": right,
                    "cumulative_tokens_from": cumulative_tokens[left],
                    "cumulative_tokens_to": cumulative_tokens[right],
                    "n_positions": sum(data.n_positions for data in regions),
                    "n_selected_from": sum(first.sum() for first, _ in selections),
                    "n_selected_to": sum(second.sum() for _, second in selections),
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union,
                }
            )
    return rows


def _rank_bins(
    values: np.ndarray,
    token_id: np.ndarray,
    *,
    n_bins: int,
) -> np.ndarray:
    assert n_bins >= 2 and len(values) >= n_bins
    order = np.lexsort((token_id, values))
    bins = np.empty(len(values), dtype=np.int8)
    bins[order] = np.minimum(
        n_bins,
        np.arange(len(values), dtype=np.int64) * n_bins // len(values) + 1,
    )
    assert bins.min() == 1 and bins.max() == n_bins
    return bins


def _future_loss_rows(
    regions: list[RegionData],
    *,
    checkpoints: list[str],
    cumulative_tokens: list[int],
    n_bins: int,
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    assignments: dict[tuple[str, int], np.ndarray] = {}
    for data in regions:
        for current in range(len(checkpoints) - 1):
            assignments[(data.region, current)] = _rank_bins(
                data.nll[current],
                data.token_id,
                n_bins=n_bins,
            )
            for horizon, future in (
                ("next", current + 1),
                ("terminal", len(checkpoints) - 1),
            ):
                reduction = data.nll[current] - data.nll[future]
                for decile in range(1, n_bins + 1):
                    mask = assignments[(data.region, current)] == decile
                    summary = _summary(
                        reduction[mask],
                        data.block[mask],
                        n_block_levels=data.n_blocks,
                        bootstrap_replicates=bootstrap_replicates,
                        rng=rng,
                    )
                    rows.append(
                        {
                            "scope": data.region,
                            "horizon": horizon,
                            "current_checkpoint": checkpoints[current],
                            "future_checkpoint": checkpoints[future],
                            "current_checkpoint_order": current,
                            "future_checkpoint_order": future,
                            "current_cumulative_tokens": cumulative_tokens[current],
                            "future_cumulative_tokens": cumulative_tokens[future],
                            "current_loss_bin": decile,
                            "n_bins": n_bins,
                            "mean_current_nll": float(
                                data.nll[current, mask].mean(dtype=np.float64)
                            ),
                            "mean_future_nll": float(
                                data.nll[future, mask].mean(dtype=np.float64)
                            ),
                            "fraction_positive_reduction": float(
                                (reduction[mask] > 0).mean()
                            ),
                            **summary,
                        }
                    )

    for current in range(len(checkpoints) - 1):
        for horizon, future in (
            ("next", current + 1),
            ("terminal", len(checkpoints) - 1),
        ):
            for decile in range(1, n_bins + 1):
                values: list[np.ndarray] = []
                current_values: list[np.ndarray] = []
                future_values: list[np.ndarray] = []
                blocks: list[np.ndarray] = []
                block_offset = 0
                for data in regions:
                    mask = assignments[(data.region, current)] == decile
                    values.append(data.nll[current, mask] - data.nll[future, mask])
                    current_values.append(data.nll[current, mask])
                    future_values.append(data.nll[future, mask])
                    blocks.append(data.block[mask] + block_offset)
                    block_offset += data.n_blocks
                reduction = np.concatenate(values)
                summary = _summary(
                    reduction,
                    np.concatenate(blocks),
                    n_block_levels=block_offset,
                    bootstrap_replicates=bootstrap_replicates,
                    rng=rng,
                )
                rows.append(
                    {
                        "scope": "global",
                        "horizon": horizon,
                        "current_checkpoint": checkpoints[current],
                        "future_checkpoint": checkpoints[future],
                        "current_checkpoint_order": current,
                        "future_checkpoint_order": future,
                        "current_cumulative_tokens": cumulative_tokens[current],
                        "future_cumulative_tokens": cumulative_tokens[future],
                        "current_loss_bin": decile,
                        "n_bins": n_bins,
                        "mean_current_nll": float(
                            np.concatenate(current_values).mean()
                        ),
                        "mean_future_nll": float(np.concatenate(future_values).mean()),
                        "fraction_positive_reduction": float((reduction > 0).mean()),
                        **summary,
                    }
                )
    return rows


def _distribution_rows(
    data: RegionData,
    *,
    checkpoints: list[str],
    cumulative_tokens: list[int],
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint_order, (checkpoint, tokens) in enumerate(
        zip(checkpoints, cumulative_tokens, strict=True)
    ):
        for statistic, matrix in (("loss", data.nll), ("entropy", data.entropy)):
            for conserved in (False, True):
                mask = data.conserved == conserved
                rows.append(
                    {
                        "scope": data.region,
                        "statistic": statistic,
                        "conserved": conserved,
                        "checkpoint": checkpoint,
                        "checkpoint_order": checkpoint_order,
                        "cumulative_tokens": tokens,
                        **_summary(
                            matrix[checkpoint_order, mask],
                            data.block[mask],
                            n_block_levels=data.n_blocks,
                            bootstrap_replicates=bootstrap_replicates,
                            rng=rng,
                        ),
                    }
                )
    return rows


def _controlled_rows(
    data: RegionData,
    *,
    checkpoints: list[str],
    cumulative_tokens: list[int],
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    valid = np.isfinite(data.window_gc) & np.isfinite(data.kmer7_nll)
    assert valid.any()
    conserved = data.conserved[valid].astype(np.float64)
    gc = data.window_gc[valid].astype(np.float64) - 0.5
    kmer = data.kmer7_nll[valid].astype(np.float64) - np.log(4)
    position = (data.target_pos[valid].astype(np.float64) - 127.0) / 127.0
    design = np.column_stack(
        [
            np.ones(valid.sum()),
            conserved,
            gc,
            gc**2,
            kmer,
            kmer**2,
            position,
            position**2,
            position**3,
        ]
    )
    outcomes = np.column_stack(
        [
            values[valid]
            for checkpoint_order in range(len(checkpoints))
            for values in (
                -data.nll[checkpoint_order],
                -data.entropy[checkpoint_order],
            )
        ]
    ).astype(np.float64)
    outcome_keys = [
        (checkpoint_order, statistic)
        for checkpoint_order in range(len(checkpoints))
        for statistic in ("loss", "entropy")
    ]
    block = data.block[valid]
    order = np.argsort(block, kind="stable")
    block_sorted = block[order]
    design = design[order]
    outcomes = outcomes[order]
    boundaries = np.flatnonzero(
        np.r_[True, block_sorted[1:] != block_sorted[:-1], True]
    )
    n_blocks = len(boundaries) - 1
    xtx_blocks = np.empty(
        (n_blocks, len(CONTROL_TERMS), len(CONTROL_TERMS)),
        dtype=np.float64,
    )
    xty_blocks = np.empty(
        (n_blocks, len(CONTROL_TERMS), len(outcome_keys)),
        dtype=np.float64,
    )
    for block_index, (start, end) in enumerate(pairwise(boundaries)):
        x_block = design[start:end]
        xtx_blocks[block_index] = x_block.T @ x_block
        xty_blocks[block_index] = x_block.T @ outcomes[start:end]
    beta = np.linalg.lstsq(
        xtx_blocks.sum(axis=0),
        xty_blocks.sum(axis=0),
        rcond=None,
    )[0]
    draws = rng.integers(
        0,
        n_blocks,
        size=(bootstrap_replicates, n_blocks),
    )
    bootstrap = np.empty(
        (bootstrap_replicates, len(CONTROL_TERMS), len(outcome_keys)),
        dtype=np.float64,
    )
    for draw_index, draw in enumerate(draws):
        bootstrap[draw_index] = np.linalg.lstsq(
            xtx_blocks[draw].sum(axis=0),
            xty_blocks[draw].sum(axis=0),
            rcond=None,
        )[0]

    rows: list[dict[str, Any]] = []
    for outcome_index, (checkpoint_order, statistic) in enumerate(outcome_keys):
        for term_index, term in enumerate(CONTROL_TERMS):
            ci_low, ci_high = np.quantile(
                bootstrap[:, term_index, outcome_index],
                [0.025, 0.975],
            )
            rows.append(
                {
                    "scope": data.region,
                    "statistic": statistic,
                    "score_direction": f"negative_{statistic}",
                    "checkpoint": checkpoints[checkpoint_order],
                    "checkpoint_order": checkpoint_order,
                    "cumulative_tokens": cumulative_tokens[checkpoint_order],
                    "term": term,
                    "estimate": float(beta[term_index, outcome_index]),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "n_positions": int(valid.sum()),
                    "n_blocks": n_blocks,
                }
            )
    return rows


def analyze_likelihood_dynamics_489(
    atom_paths: dict[tuple[str, str], str | Path],
    manifest_paths: dict[tuple[str, str], str | Path],
    *,
    checkpoints: list[str],
    cumulative_tokens: list[int],
    regions: list[str],
    primary_start: int,
    primary_end_exclusive: int,
    block_bp: int,
    bootstrap_replicates: int,
    top_fraction: float,
    n_bins: int,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Run the prespecified #489 analyses on the immutable full atom cache."""
    assert len(checkpoints) == len(cumulative_tokens) == 5
    assert len(regions) == 5
    assert all(later > earlier for earlier, later in pairwise(cumulative_tokens))
    assert 0 <= primary_start < primary_end_exclusive
    assert block_bp > 0 and bootstrap_replicates > 0
    rng = np.random.default_rng(seed)
    region_data: list[RegionData] = []
    population_rows: list[dict[str, Any]] = []
    input_cells: list[dict[str, Any]] = []
    for region in regions:
        data, population, cells = _load_region(
            atom_paths,
            manifest_paths,
            region=region,
            checkpoints=checkpoints,
            cumulative_tokens=cumulative_tokens,
            primary_start=primary_start,
            primary_end_exclusive=primary_end_exclusive,
            block_bp=block_bp,
        )
        region_data.append(data)
        population_rows.append(population)
        input_cells.extend(cells)
    global_data = _combine_regions(region_data)
    all_scopes = [*region_data, global_data]

    classification_rows = [
        row
        for data in all_scopes
        for row in _classification_rows(
            data,
            checkpoints=checkpoints,
            cumulative_tokens=cumulative_tokens,
        )
    ]
    region_groups: dict[str, np.ndarray] = {}
    trajectory_rows: list[dict[str, Any]] = []
    for data in region_data:
        groups, early_mean, terminal_mean = _trajectory_groups(data)
        region_groups[data.region] = groups
        trajectory_rows.extend(
            _trajectory_rows(
                data,
                groups,
                checkpoints=checkpoints,
                cumulative_tokens=cumulative_tokens,
                early_mean=early_mean,
                terminal_mean=terminal_mean,
                bootstrap_replicates=bootstrap_replicates,
                rng=rng,
            )
        )
    global_groups = np.concatenate([region_groups[data.region] for data in region_data])
    trajectory_rows.extend(
        _trajectory_rows(
            global_data,
            global_groups,
            checkpoints=checkpoints,
            cumulative_tokens=cumulative_tokens,
            early_mean=None,
            terminal_mean=None,
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
        )
    )

    selection_rows = _selection_rows(
        region_data,
        checkpoints=checkpoints,
        cumulative_tokens=cumulative_tokens,
        fraction=top_fraction,
    )
    future_rows = _future_loss_rows(
        region_data,
        checkpoints=checkpoints,
        cumulative_tokens=cumulative_tokens,
        n_bins=n_bins,
        bootstrap_replicates=bootstrap_replicates,
        rng=rng,
    )
    distribution_rows = [
        row
        for data in all_scopes
        for row in _distribution_rows(
            data,
            checkpoints=checkpoints,
            cumulative_tokens=cumulative_tokens,
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
        )
    ]
    controlled_rows = [
        row
        for data in region_data
        for row in _controlled_rows(
            data,
            checkpoints=checkpoints,
            cumulative_tokens=cumulative_tokens,
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
        )
    ]

    frames = {
        "population": pd.DataFrame(population_rows),
        "conservation_auprc": pd.DataFrame(classification_rows),
        "trajectory_groups": pd.DataFrame(trajectory_rows),
        "selection_jaccard": pd.DataFrame(selection_rows),
        "future_loss_deciles": pd.DataFrame(future_rows),
        "distributions": pd.DataFrame(distribution_rows),
        "controlled_contrasts": pd.DataFrame(controlled_rows),
    }
    expected_scopes = len(regions) + 1
    assert len(frames["conservation_auprc"]) == expected_scopes * 2 * len(checkpoints)
    assert frames["conservation_auprc"]["auprc"].between(0, 1).all()
    assert frames["selection_jaccard"]["jaccard"].between(0, 1).all()
    assert np.isfinite(frames["future_loss_deciles"]["mean"]).all()
    assert np.isfinite(frames["controlled_contrasts"]["estimate"]).all()
    manifest = {
        "artifact_schema_version": "v1",
        "issue": 489,
        "primary_population": (
            f"target_pos in [{primary_start}, {primary_end_exclusive}), "
            "scorable, nonambiguous, nonrepeat"
        ),
        "checkpoints": [
            {
                "name": checkpoint,
                "order": order,
                "cumulative_tokens": tokens,
            }
            for order, (checkpoint, tokens) in enumerate(
                zip(checkpoints, cumulative_tokens, strict=True)
            )
        ],
        "regions": regions,
        "top_fraction": top_fraction,
        "selection_tie_break": "ascending token_index",
        "loss_bin_definition": (
            f"{n_bins} equal-count rank bins within each region; "
            "1 is lowest current loss"
        ),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
        "bootstrap_unit": f"region-specific genomic {block_bp}-bp block",
        "auprc": "exact average precision from negative loss or negative entropy",
        "trajectory_threshold": (
            "region mean NLL at the earliest and terminal checkpoints"
        ),
        "control_formula": (
            "negative score ~ conserved + window_gc + window_gc^2 + "
            "kmer7_nll + kmer7_nll^2 + position + position^2 + position^3"
        ),
        "input_cells": input_cells,
        "outputs": {
            name: {
                "n_rows": len(frame),
                "columns": list(frame.columns),
            }
            for name, frame in frames.items()
        },
        "validation": {
            "passed": True,
            "n_input_cells": len(input_cells),
            "n_primary_positions": global_data.n_positions,
            "n_scopes": expected_scopes,
        },
    }
    assert manifest["validation"]["n_input_cells"] == len(checkpoints) * len(regions)
    return frames, manifest
