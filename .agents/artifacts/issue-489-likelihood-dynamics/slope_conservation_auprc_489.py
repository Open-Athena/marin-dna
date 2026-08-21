"""Measure conservation AUPRC from per-token loss slopes for issue #489."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score


ARTIFACT_ROOT = Path(__file__).resolve().parent
SCRATCH_ROOT = Path("/tmp/marin-dna-ld489-slope")
S3_ROOT = (
    "oa-bolinas/snakemake/analysis/evals_v2/results/"
    "m13_likelihood_dynamics_489/v1/scoring/full/atoms"
)
REGION_ORDER = ("cds", "upstream", "downstream", "ncrna", "enhancer")
PRIMARY_START = 32
PRIMARY_END = 223
AP_CHUNK_SIZE = 250_000


def as_numpy(
    table: pa.Table,
    column: str,
    dtype: np.dtype[object] | type[object],
) -> np.ndarray:
    """Return one combined Arrow column as a NumPy array."""
    array = table[column].combine_chunks()
    return np.asarray(array.to_numpy(zero_copy_only=False), dtype=dtype)


def primary_mask(table: pa.Table) -> np.ndarray:
    """Select the issue-489 nonrepeat central-position population."""
    target_pos = as_numpy(table, "target_pos", np.int16)
    return (
        (target_pos >= PRIMARY_START)
        & (target_pos < PRIMARY_END)
        & as_numpy(table, "is_scorable", bool)
        & ~as_numpy(table, "is_ambiguous", bool)
        & ~as_numpy(table, "is_repeat", bool)
    )


def exact_average_precision(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    chunk_size: int = AP_CHUNK_SIZE,
) -> float:
    """Compute sklearn-compatible AP with one global sort and chunked reductions."""
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    n_positive = int(labels.sum(dtype=np.int64))
    if not 0 < n_positive < len(labels):
        raise ValueError("labels must contain both classes")

    order = np.argsort(scores, kind="quicksort")[::-1]
    seen = 0
    true_positive = 0
    weighted_precision = 0.0
    start = 0
    while start < len(order):
        end = min(start + chunk_size, len(order))
        while (
            end < len(order)
            and scores[order[end - 1]] == scores[order[end]]
        ):
            end += 1

        indices = order[start:end]
        chunk_scores = scores[indices]
        chunk_labels = labels[indices].astype(np.int64, copy=False)
        boundaries_mask = np.empty(len(chunk_scores) + 1, dtype=bool)
        boundaries_mask[0] = True
        boundaries_mask[-1] = True
        boundaries_mask[1:-1] = chunk_scores[1:] != chunk_scores[:-1]
        boundaries = np.flatnonzero(boundaries_mask)
        group_counts = np.diff(boundaries)
        group_positives = np.add.reduceat(chunk_labels, boundaries[:-1])
        cumulative_counts = np.cumsum(group_counts, dtype=np.int64)
        cumulative_positives = np.cumsum(group_positives, dtype=np.int64)
        precision = (true_positive + cumulative_positives) / (
            seen + cumulative_counts
        )
        weighted_precision += float(np.dot(precision, group_positives))
        seen += len(indices)
        true_positive += int(group_positives.sum())
        start = end

    assert seen == len(labels)
    assert true_positive == n_positive
    return weighted_precision / n_positive


def validate_average_precision() -> None:
    """Check the bounded implementation against sklearn, including tied scores."""
    labels = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    scores = np.array([0.1, 0.8, 0.8, 0.3, 0.7, 0.2, 0.1, 0.9])
    expected = float(average_precision_score(labels, scores))
    observed = exact_average_precision(labels, scores, chunk_size=3)
    assert np.isclose(observed, expected, rtol=0, atol=1e-15)


def checkpoint_metadata() -> tuple[list[str], np.ndarray]:
    """Read the pinned checkpoint names and cumulative-token coordinates."""
    manifest = json.loads((ARTIFACT_ROOT / "manifest.json").read_text())
    checkpoints = sorted(manifest["checkpoints"], key=lambda row: row["order"])
    names = [str(row["name"]) for row in checkpoints]
    tokens_billions = np.array(
        [int(row["cumulative_tokens"]) for row in checkpoints],
        dtype=np.float64,
    ) / 1e9
    assert len(names) == 5
    assert np.all(np.diff(tokens_billions) > 0)
    return names, tokens_billions


def stream_loss_slopes(
    s3: pafs.S3FileSystem,
    *,
    checkpoints: list[str],
    tokens_billions: np.ndarray,
    expected_by_region: dict[str, int],
) -> tuple[np.memmap, np.memmap, dict[str, slice]]:
    """Write aligned labels and negative OLS slopes into bounded memmaps."""
    total_positions = sum(expected_by_region.values())
    score_path = SCRATCH_ROOT / "loss-slope-scores.npy"
    label_path = SCRATCH_ROOT / "loss-slope-labels.npy"
    scores = np.lib.format.open_memmap(
        score_path,
        mode="w+",
        dtype=np.float64,
        shape=(total_positions,),
    )
    labels = np.lib.format.open_memmap(
        label_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_positions,),
    )
    centered_tokens = tokens_billions - tokens_billions.mean()
    slope_weights = centered_tokens / np.square(centered_tokens).sum()

    region_slices: dict[str, slice] = {}
    global_cursor = 0
    for region in REGION_ORDER:
        paths = [f"{S3_ROOT}/{checkpoint}/{region}.parquet" for checkpoint in checkpoints]
        files = [pq.ParquetFile(s3.open_input_file(path)) for path in paths]
        assert len({file.num_row_groups for file in files}) == 1
        region_start = global_cursor
        for row_group in range(files[0].num_row_groups):
            first = files[0].read_row_group(
                row_group,
                columns=(
                    "token_index",
                    "target_pos",
                    "is_scorable",
                    "is_ambiguous",
                    "is_repeat",
                    "is_conserved",
                    "nll",
                ),
            )
            mask = primary_mask(first)
            token_index = as_numpy(first, "token_index", np.int64)
            slope = slope_weights[0] * as_numpy(first, "nll", np.float32)[mask]
            for checkpoint_index, file in enumerate(files[1:], start=1):
                table = file.read_row_group(
                    row_group,
                    columns=("token_index", "nll"),
                )
                assert np.array_equal(
                    token_index,
                    as_numpy(table, "token_index", np.int64),
                )
                slope += (
                    slope_weights[checkpoint_index]
                    * as_numpy(table, "nll", np.float32)[mask]
                )
            n_selected = int(mask.sum())
            destination = slice(global_cursor, global_cursor + n_selected)
            scores[destination] = -slope
            labels[destination] = as_numpy(first, "is_conserved", bool)[mask]
            global_cursor += n_selected
            del first, table, mask, token_index, slope
            gc.collect()
            pa.default_memory_pool().release_unused()

        region_slices[region] = slice(region_start, global_cursor)
        observed = global_cursor - region_start
        assert observed == expected_by_region[region]
        print(f"streamed {region}: {observed:,} positions", flush=True)

    assert global_cursor == total_positions
    scores.flush()
    labels.flush()
    return scores, labels, region_slices


def summarize(
    scores: np.ndarray,
    labels: np.ndarray,
    region_slices: dict[str, slice],
) -> pd.DataFrame:
    """Compute exact pooled slope AUPRC globally and by validation region."""
    rows: list[dict[str, object]] = []
    for scope in (*REGION_ORDER, "global"):
        selected = (
            slice(0, len(labels))
            if scope == "global"
            else region_slices[scope]
        )
        scope_labels = labels[selected]
        scope_scores = scores[selected]
        auprc = exact_average_precision(scope_labels, scope_scores)
        prevalence = float(scope_labels.mean(dtype=np.float64))
        rows.append(
            {
                "scope": scope,
                "statistic": "loss_slope",
                "score_direction": "negative_ols_loss_slope",
                "auprc": auprc,
                "prevalence": prevalence,
                "auprc_minus_prevalence": auprc - prevalence,
                "n_positions": len(scope_labels),
                "n_conserved": int(scope_labels.sum(dtype=np.int64)),
            }
        )
        print(
            f"{scope}: slope AUPRC={auprc:.6f}, prevalence={prevalence:.6f}",
            flush=True,
        )
        gc.collect()
    return pd.DataFrame(rows)


def main() -> None:
    """Stream the cache and write exact slope-based conservation AUPRC."""
    validate_average_precision()
    population = pd.read_parquet(ARTIFACT_ROOT / "population.parquet")
    expected_by_region = dict(
        zip(
            population["region"].astype(str),
            population["n_primary_nonrepeat_scorable"].astype(int),
            strict=True,
        )
    )
    assert set(expected_by_region) == set(REGION_ORDER)
    checkpoints, tokens_billions = checkpoint_metadata()
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    s3 = pafs.S3FileSystem(region="us-east-2")
    scores, labels, region_slices = stream_loss_slopes(
        s3,
        checkpoints=checkpoints,
        tokens_billions=tokens_billions,
        expected_by_region=expected_by_region,
    )
    summary = summarize(scores, labels, region_slices)
    summary.to_csv(
        ARTIFACT_ROOT / "slope_conservation_auprc.csv",
        index=False,
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
