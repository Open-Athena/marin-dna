"""Per-base predictability atoms for conservation × repeat analysis (#478)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna_evals.model.runner import run_per_base_stats_clm
from marin_dna_evals.transforms import _get_special_token_counts

Orientation = Literal["fwd", "rc"]


def _validate_sequences(sequences: pd.DataFrame, window_size: int) -> None:
    for column in ("id", "seq"):
        assert column in sequences.columns, (
            f"sequences missing {column!r}; got {list(sequences.columns)}"
        )
    assert len(sequences) > 0, "empty sequences frame"
    lengths = sequences["seq"].str.len()
    assert (lengths == window_size).all(), (
        f"every seq must be window_size={window_size}; "
        f"got {sorted(lengths.unique())[:5]}"
    )
    assert sequences["id"].is_unique, "window ids must be unique within a region"


def _prediction_array(
    prediction: Any,
    *,
    n_windows: int,
    window_size: int,
    orientation: Orientation,
) -> np.ndarray:
    """Normalize Trainer output to forward-coordinate ``[N, W, 2]``."""
    values = np.asarray(prediction, dtype=np.float32)
    expected = (n_windows, window_size, 2)
    if values.size == int(np.prod(expected)) and values.shape != expected:
        print(
            f"[per_base] WARNING: inference returned {values.shape}; "
            f"reshaping row-major to {expected}"
        )
        values = values.reshape(expected)
    assert values.shape == expected, (
        f"per-base shape {values.shape}, expected {expected}"
    )
    assert np.isfinite(values).all(), "non-finite per-base prediction"
    assert (values[:, :, 0] > 0).all(), "NLL must be positive"
    assert (values[:, :, 1] >= 0).all(), "entropy must be non-negative"
    if orientation == "rc":
        values = values[:, ::-1, :]
    return values


def compute_hf_per_base_stats(
    checkpoint_path: str | Path,
    sequences: pd.DataFrame,
    window_size: int,
    *,
    orientations: tuple[Orientation, ...] = ("fwd", "rc"),
    batch_size: int = 128,
    num_workers: int = 4,
    torch_compile: bool = False,
) -> dict[Orientation, pd.DataFrame]:
    """Score every source base while retaining FWD and RC atoms separately.

    Output is wide: one row per window with fixed-length list columns.
    ``nll[j]`` and ``entropy_4nuc[j]`` always refer to forward genomic
    coordinate ``start + j``. RC predictions are reversed after inference.
    """
    _validate_sequences(sequences, window_size)
    assert orientations and set(orientations) <= {"fwd", "rc"}, orientations
    checkpoint_path = Path(checkpoint_path)
    tokenizer: Any = AutoTokenizer.from_pretrained(checkpoint_path)
    model: Any = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, trust_remote_code=True
    )
    n_prefix, n_suffix = _get_special_token_counts(tokenizer)
    assert (n_prefix, n_suffix) == (1, 0), (
        "per-base genomic alignment requires one prepended BOS and no suffix; "
        f"got (n_prefix, n_suffix)=({n_prefix}, {n_suffix})"
    )

    dataset = Dataset.from_pandas(sequences[["seq"]], preserve_index=False)
    inference_kwargs = {
        "per_device_eval_batch_size": batch_size,
        "torch_compile": torch_compile,
        "bf16_full_eval": True,
        "dataloader_num_workers": num_workers,
        "remove_unused_columns": False,
    }
    out: dict[Orientation, pd.DataFrame] = {}
    for orientation in orientations:
        strand: Literal["+", "-"] = "+" if orientation == "fwd" else "-"
        prediction = run_per_base_stats_clm(
            model,
            tokenizer,
            dataset,
            strand=strand,
            data_transform_on_the_fly=True,
            inference_kwargs=inference_kwargs,
        )
        values = _prediction_array(
            prediction,
            n_windows=len(sequences),
            window_size=window_size,
            orientation=orientation,
        )
        out[orientation] = pd.DataFrame(
            {
                "window_id": sequences["id"].to_numpy(),
                "nll": list(values[:, :, 0]),
                "entropy_4nuc": list(values[:, :, 1]),
            }
        )
    return out


def aggregate_by_case(
    atoms: pd.DataFrame,
    sequences: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct issue-274 FWD per-window sums/counts from per-base NLL."""
    assert list(atoms["window_id"]) == list(sequences["id"]), (
        "atom/source row order or ids differ"
    )
    nll = np.stack(atoms["nll"].to_numpy()).astype(np.float64)
    chars = np.asarray([list(seq) for seq in sequences["seq"]])
    assert chars.shape == nll.shape
    upper = np.char.isupper(chars)
    logp = -nll
    return pd.DataFrame(
        {
            "id": sequences["id"].to_numpy(),
            "ll_sum_upper": np.where(upper, logp, 0.0).sum(axis=1),
            "ll_sum_lower": np.where(~upper, logp, 0.0).sum(axis=1),
            "n_upper": upper.sum(axis=1).astype(np.int64),
            "n_lower": (~upper).sum(axis=1).astype(np.int64),
        }
    )


def compare_ll_gap_cache(
    reconstructed: pd.DataFrame,
    cached: pd.DataFrame,
    *,
    per_window_mean_atol: float = 0.25,
    per_window_q99_atol: float = 0.55,
    min_correlation: float = 0.99999,
    aggregate_mean_atol: float = 2e-5,
) -> dict[str, float | int | bool]:
    """Regression gate against issue-274 cached FWD sums and counts.

    The cache predates the current PyTorch/CUDA runtime, so score tolerances
    admit bounded cross-runtime drift. Exact IDs/counts plus near-unit window
    correlation retain sensitivity to ordering or alignment regressions.
    """
    columns = ["id", "ll_sum_upper", "ll_sum_lower", "n_upper", "n_lower"]
    for name, frame in (("reconstructed", reconstructed), ("cached", cached)):
        missing = set(columns) - set(frame.columns)
        assert not missing, f"{name} missing columns {sorted(missing)}"
        assert frame["id"].is_unique, f"{name} ids are not unique"
    joined = reconstructed.merge(
        cached[columns], on="id", suffixes=("_new", "_cached"), validate="one_to_one"
    )
    assert len(joined) == len(reconstructed) == len(cached), "cache id set differs"
    for count in ("n_upper", "n_lower"):
        assert np.array_equal(joined[f"{count}_new"], joined[f"{count}_cached"]), (
            f"{count} mismatch"
        )

    max_abs = 0.0
    metrics: dict[str, float] = {}
    for value in ("ll_sum_upper", "ll_sum_lower"):
        new_values = joined[f"{value}_new"].to_numpy()
        cached_values = joined[f"{value}_cached"].to_numpy()
        delta = new_values - cached_values
        abs_delta = np.abs(delta)
        value_max = float(abs_delta.max())
        value_q99 = float(np.quantile(abs_delta, 0.99))
        if np.array_equal(new_values, cached_values):
            correlation = 1.0
        else:
            correlation = float(np.corrcoef(new_values, cached_values)[0, 1])
        max_abs = max(max_abs, value_max)
        count = "n_upper" if value.endswith("upper") else "n_lower"
        count_values = joined[f"{count}_new"].to_numpy()
        value_max_per_base = float((abs_delta / np.maximum(count_values, 1)).max())
        mean_diff = float(delta.sum() / count_values.sum())
        label = "ll_upper" if value.endswith("upper") else "ll_lower"
        metrics[f"{label}_max_abs"] = value_max
        metrics[f"{label}_max_abs_per_base"] = value_max_per_base
        metrics[f"{label}_q99_abs"] = value_q99
        metrics[f"{label}_correlation"] = correlation
        metrics[f"{label}_mean_diff"] = mean_diff
        assert value_max_per_base <= per_window_mean_atol, (
            f"{value} per-window max abs diff/base {value_max_per_base:.3g} "
            f"> {per_window_mean_atol}"
        )
        assert value_q99 <= per_window_q99_atol, (
            f"{value} per-window q99 abs diff {value_q99:.3g} > {per_window_q99_atol}"
        )
        assert correlation >= min_correlation, (
            f"{value} window correlation {correlation:.8f} < {min_correlation}"
        )
        assert abs(mean_diff) <= aggregate_mean_atol, (
            f"{value} aggregate mean diff {mean_diff:.3g} > {aggregate_mean_atol}"
        )
    return {
        "gate_schema_version": 2,
        "per_window_mean_atol": per_window_mean_atol,
        "per_window_q99_atol": per_window_q99_atol,
        "min_correlation": min_correlation,
        "aggregate_mean_atol": aggregate_mean_atol,
        "passed": True,
        "n_windows": len(joined),
        "max_abs_per_window_sum_diff": max_abs,
        **metrics,
        "n_upper": int(joined["n_upper_new"].sum()),
        "n_lower": int(joined["n_lower_new"].sum()),
    }
