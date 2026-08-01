"""Reusable LLR-only inference benchmark primitives for issue #430."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from marin_dna.data.dna import NUCLEOTIDES
from marin_dna.data.transforms import (
    _get_nucleotide_token_ids,
    _get_special_token_counts,
)
from marin_dna.model.scoring import (
    compute_variant_llr,
    compute_variant_llr_branch_packed,
    compute_variant_llr_full_pair,
    make_variant_branch_packed_layout,
)


VARIANT_KEY_COLUMNS = ["chrom", "pos", "ref", "alt"]
ExecutionLayout = Literal["prefix-cache", "branch-packed", "full-pair"]
HARNESS_REQUIRED_COLUMNS = [
    *VARIANT_KEY_COLUMNS,
    "target",
    "subset",
    "match_group",
    "context",
    "ref_completion",
    "alt_completion",
    "strand",
]


@dataclass(frozen=True)
class PreparedHarnessLlr:
    """Validated, tokenized rows from the 255-bp Mendelian harness dataset."""

    metadata: pd.DataFrame
    input_ids: Tensor
    alt_token_id: Tensor
    var_pos: int
    nuc_token_ids: Tensor


@dataclass(frozen=True)
class LlrBenchmarkResult:
    """Scores and timing/memory measurements from repeated LLR-only passes."""

    row_indices: np.ndarray
    llr: np.ndarray
    warmup_seconds: float
    repeat_seconds: tuple[float, ...]
    peak_vram_allocated_bytes: int
    peak_vram_reserved_bytes: int

    @property
    def median_seconds(self) -> float:
        return float(np.median(self.repeat_seconds))


def prepare_harness_llr(
    dataset: pd.DataFrame,
    tokenizer: Any,
    *,
    subset: str | None = None,
) -> PreparedHarnessLlr:
    """Validate and tokenize pre-materialized FWD+RC VEP harness rows once.

    The public ``marin-dna/evals_mendelian_traits_harness_255`` dataset stores
    ``context`` (127 bp) and REF/ALT completions (128 bp) for each strand. The
    reference model input is ``context + ref_completion``; ALT differs only at
    completion position zero. This function proves that contract before building
    the fixed tensors used by the timed runner. Tokenization is deliberately
    outside the timed path.
    """
    missing = [col for col in HARNESS_REQUIRED_COLUMNS if col not in dataset.columns]
    assert not missing, f"harness dataset missing required columns: {missing}"
    frame = dataset.loc[:, HARNESS_REQUIRED_COLUMNS].reset_index(drop=True).copy()
    assert len(frame) > 0, "harness dataset is empty"
    assert not frame[HARNESS_REQUIRED_COLUMNS].isna().any().any(), (
        "harness contract columns contain missing values"
    )

    subset_per_group = frame.groupby("match_group")["subset"].nunique()
    bad_groups = subset_per_group[subset_per_group != 1]
    assert bad_groups.empty, (
        f"{len(bad_groups)} match_group(s) span subsets; first: "
        f"{bad_groups.head().to_dict()}"
    )
    if subset is not None:
        frame = frame[frame["subset"] == subset].reset_index(drop=True)
        assert len(frame) > 0, f"no harness rows for subset {subset!r}"

    grouped = frame.groupby(VARIANT_KEY_COLUMNS, sort=False, dropna=False)
    pair_sizes = grouped.size()
    bad_pair_sizes = pair_sizes[pair_sizes != 2]
    assert bad_pair_sizes.empty, (
        f"expected exactly two strand rows per variant, got {len(bad_pair_sizes)} "
        f"bad variants; first: {bad_pair_sizes.head().to_dict()}"
    )
    strand_sets = grouped["strand"].agg(lambda values: frozenset(values))
    bad_strands = strand_sets[strand_sets != frozenset({"+", "-"})]
    assert bad_strands.empty, (
        f"expected one '+' and one '-' row per variant, got {len(bad_strands)} "
        f"bad variants; first: {bad_strands.head().to_dict()}"
    )
    for col in ("target", "subset", "match_group"):
        n_unique = grouped[col].nunique()
        bad = n_unique[n_unique != 1]
        assert bad.empty, (
            f"variant strand pairs disagree on {col!r}; first: {bad.head().to_dict()}"
        )

    context_len = frame["context"].str.len()
    ref_len = frame["ref_completion"].str.len()
    alt_len = frame["alt_completion"].str.len()
    assert (context_len == 127).all(), (
        f"context length must be 127, got {sorted(context_len.unique().tolist())}"
    )
    assert (ref_len == 128).all() and (alt_len == 128).all(), (
        "REF/ALT completion lengths must both be 128"
    )
    assert (
        frame["ref_completion"].str.slice(1) == frame["alt_completion"].str.slice(1)
    ).all(), "REF/ALT completion tails differ beyond the SNV"
    assert (
        frame["ref_completion"].str.slice(0, 1)
        != frame["alt_completion"].str.slice(0, 1)
    ).all(), "REF and ALT alleles must differ"

    sequences = (frame["context"] + frame["ref_completion"]).tolist()
    encoded = [tokenizer.encode(sequence) for sequence in sequences]
    token_lengths = {len(ids) for ids in encoded}
    assert token_lengths == {256}, (
        f"255-bp harness rows must tokenize to 256 tokens including BOS, got "
        f"{sorted(token_lengths)}"
    )
    n_prefix, n_suffix = _get_special_token_counts(tokenizer)
    assert (n_prefix, n_suffix) == (1, 0), (
        f"issue #430 checkpoint requires one BOS and no EOS, got "
        f"n_prefix={n_prefix}, n_suffix={n_suffix}"
    )
    var_pos = n_prefix + 127
    nuc_ids_dict = _get_nucleotide_token_ids(tokenizer)
    nuc_token_ids = torch.tensor(
        [nuc_ids_dict[nuc] for nuc in NUCLEOTIDES], dtype=torch.long
    )
    ref_token_ids = torch.tensor(
        [nuc_ids_dict[base] for base in frame["ref_completion"].str[0]],
        dtype=torch.long,
    )
    alt_token_id = torch.tensor(
        [nuc_ids_dict[base] for base in frame["alt_completion"].str[0]],
        dtype=torch.long,
    )
    input_ids = torch.tensor(encoded, dtype=torch.long)
    assert torch.equal(input_ids[:, var_pos], ref_token_ids), (
        "tokenized reference allele does not occupy the expected variant position"
    )
    assert torch.all(ref_token_ids != alt_token_id), (
        "at least one REF and ALT allele token is unexpectedly identical"
    )
    assert input_ids.shape == (len(frame), 256)
    assert alt_token_id.shape == (len(frame),)

    return PreparedHarnessLlr(
        metadata=frame,
        input_ids=input_ids,
        alt_token_id=alt_token_id,
        var_pos=var_pos,
        nuc_token_ids=nuc_token_ids,
    )


def aggregate_harness_llr(
    prepared: PreparedHarnessLlr,
    row_indices: np.ndarray,
    llr: np.ndarray,
) -> pd.DataFrame:
    """Reduce paired strand rows to one variant with ``minus_llr_avg``."""
    indices = np.asarray(row_indices, dtype=np.int64)
    scores = np.asarray(llr, dtype=float)
    assert indices.ndim == 1 and scores.ndim == 1
    assert len(indices) == len(scores), (
        f"row_indices/LLR length mismatch: {len(indices)} vs {len(scores)}"
    )
    assert len(np.unique(indices)) == len(indices), "row_indices contain duplicates"
    assert np.isfinite(scores).all(), "LLR contains non-finite values"

    frame = prepared.metadata.iloc[indices].reset_index(drop=True).copy()
    frame["llr"] = scores
    grouped = frame.groupby(VARIANT_KEY_COLUMNS, sort=False, dropna=False)
    sizes = grouped.size()
    assert (sizes == 2).all(), "aggregation requires complete two-strand variants"
    for col in ("target", "subset", "match_group"):
        assert (grouped[col].nunique() == 1).all(), (
            f"strand rows disagree on {col!r} during aggregation"
        )

    metadata = grouped[["target", "subset", "match_group"]].first()
    strand_scores = frame.pivot(
        index=VARIANT_KEY_COLUMNS, columns="strand", values="llr"
    )
    assert set(strand_scores.columns) == {"+", "-"}, (
        f"expected '+'/'-' score columns, got {strand_scores.columns.tolist()}"
    )
    output = metadata.join(strand_scores).reset_index()
    output = output.rename(columns={"+": "llr_fwd", "-": "llr_rc"})
    output["minus_llr_avg"] = -0.5 * (output["llr_fwd"] + output["llr_rc"])
    assert len(output) * 2 == len(frame)
    assert np.isfinite(output[["llr_fwd", "llr_rc", "minus_llr_avg"]]).all().all()
    return output


class _LlrOnlyModule(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        sequence_length: int,
        var_pos: int,
        nuc_token_ids: Tensor,
        execution_layout: ExecutionLayout,
    ) -> None:
        super().__init__()
        self.model = model
        self.sequence_length = sequence_length
        self.var_pos = var_pos
        self.execution_layout = execution_layout
        self.register_buffer("nuc_token_ids", nuc_token_ids)
        if execution_layout == "branch-packed":
            position_ids, attention_mask = make_variant_branch_packed_layout(
                sequence_length=sequence_length,
                var_pos=var_pos,
            )
        else:
            position_ids, attention_mask = None, None
        self.register_buffer("branch_position_ids", position_ids)
        self.register_buffer("branch_attention_mask", attention_mask)

    def forward(self, input_ids: Tensor, alt_token_id: Tensor) -> Tensor:
        if self.execution_layout == "prefix-cache":
            return compute_variant_llr(
                self.model,
                input_ids,
                alt_token_id,
                var_pos=self.var_pos,
                nuc_token_ids=self.nuc_token_ids,
            )
        if self.execution_layout == "branch-packed":
            assert self.branch_position_ids is not None
            assert self.branch_attention_mask is not None
            return compute_variant_llr_branch_packed(
                self.model,
                input_ids,
                alt_token_id,
                var_pos=self.var_pos,
                nuc_token_ids=self.nuc_token_ids,
                position_ids=self.branch_position_ids,
                attention_mask=self.branch_attention_mask,
            )
        assert self.execution_layout == "full-pair"
        return compute_variant_llr_full_pair(
            self.model,
            input_ids,
            alt_token_id,
            var_pos=self.var_pos,
            nuc_token_ids=self.nuc_token_ids,
        )


def benchmark_prepared_llr(
    model: nn.Module,
    prepared: PreparedHarnessLlr,
    *,
    batch_size: int,
    device: torch.device | str,
    row_indices: np.ndarray | None = None,
    repetitions: int = 3,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    torch_compile: bool = False,
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] | None = None,
    fullgraph: bool = False,
    use_bf16_autocast: bool = True,
    execution_layout: ExecutionLayout = "prefix-cache",
) -> LlrBenchmarkResult:
    """Run repeated steady-state passes over fixed token tensors.

    One untimed-output warmup batch is synchronized and measured separately; it
    absorbs lazy CUDA initialization and compilation. Every reported repetition
    then covers all selected rows in deterministic order, including DataLoader and
    host-to-device overhead. ``repeat_seconds`` therefore supports the required
    median over at least three full passes while ``warmup_seconds`` reports the
    one-time cost separately.
    """
    assert batch_size > 0, f"batch_size must be positive, got {batch_size}"
    assert repetitions >= 1, f"repetitions must be >=1, got {repetitions}"
    assert num_workers >= 0, f"num_workers must be non-negative, got {num_workers}"
    assert prefetch_factor >= 1, f"prefetch_factor must be >=1, got {prefetch_factor}"
    assert torch_compile or compile_mode is None, (
        "compile_mode requires torch_compile=True"
    )
    assert execution_layout in ("prefix-cache", "branch-packed", "full-pair")

    n_rows = len(prepared.metadata)
    if row_indices is None:
        indices = np.arange(n_rows, dtype=np.int64)
    else:
        indices = np.asarray(row_indices, dtype=np.int64)
    assert indices.ndim == 1 and indices.size > 0, "row_indices must be non-empty 1D"
    assert len(np.unique(indices)) == len(indices), "row_indices contain duplicates"
    assert int(indices.min()) >= 0 and int(indices.max()) < n_rows, (
        f"row_indices out of bounds for {n_rows} rows"
    )

    pad_n = (batch_size - len(indices) % batch_size) % batch_size
    padded_indices = (
        np.concatenate([indices, np.repeat(indices[-1], pad_n)])
        if pad_n > 0
        else indices
    )
    index_tensor = torch.from_numpy(padded_indices)
    tensor_dataset = TensorDataset(
        prepared.input_ids[index_tensor], prepared.alt_token_id[index_tensor]
    )
    resolved_device = torch.device(device)
    loader = DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    assert len(loader) > 0, "benchmark DataLoader is empty"

    wrapped: nn.Module = _LlrOnlyModule(
        model,
        sequence_length=prepared.input_ids.shape[1],
        var_pos=prepared.var_pos,
        nuc_token_ids=prepared.nuc_token_ids,
        execution_layout=execution_layout,
    ).to(resolved_device)
    wrapped.eval()
    if torch_compile:
        compile_kwargs: dict[str, object] = {"fullgraph": fullgraph}
        if compile_mode is not None:
            compile_kwargs["mode"] = compile_mode
        wrapped = torch.compile(wrapped, **compile_kwargs)

    def synchronize() -> None:
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)

    def autocast_context() -> Any:
        return (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if resolved_device.type == "cuda" and use_bf16_autocast
            else nullcontext()
        )

    warm_input_ids, warm_alt = next(iter(loader))
    warm_input_ids = warm_input_ids.to(resolved_device, non_blocking=True)
    warm_alt = warm_alt.to(resolved_device, non_blocking=True)
    synchronize()
    warm_start = perf_counter()
    with torch.inference_mode(), autocast_context():
        warm_output = wrapped(warm_input_ids, warm_alt)
    synchronize()
    warmup_seconds = perf_counter() - warm_start
    assert warm_output.shape == (len(warm_input_ids),)
    assert torch.isfinite(warm_output).all(), "non-finite LLR in warmup batch"

    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)

    repeat_seconds: list[float] = []
    scores = np.empty(len(indices), dtype=np.float32)
    for _ in range(repetitions):
        offset = 0
        synchronize()
        start = perf_counter()
        with torch.inference_mode(), autocast_context():
            for batch_input_ids, batch_alt in loader:
                batch_input_ids = batch_input_ids.to(resolved_device, non_blocking=True)
                batch_alt = batch_alt.to(resolved_device, non_blocking=True)
                output = wrapped(batch_input_ids, batch_alt)
                output_cpu = output.float().cpu().numpy()
                n_keep = min(len(output_cpu), len(indices) - offset)
                if n_keep > 0:
                    scores[offset : offset + n_keep] = output_cpu[:n_keep]
                    offset += n_keep
        synchronize()
        repeat_seconds.append(perf_counter() - start)
        assert offset == len(indices), f"scored {offset} rows, expected {len(indices)}"
        assert np.isfinite(scores).all(), "non-finite LLR in timed pass"

    peak_allocated = (
        int(torch.cuda.max_memory_allocated(resolved_device))
        if resolved_device.type == "cuda"
        else 0
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(resolved_device))
        if resolved_device.type == "cuda"
        else 0
    )
    return LlrBenchmarkResult(
        row_indices=indices,
        llr=scores.copy(),
        warmup_seconds=float(warmup_seconds),
        repeat_seconds=tuple(float(value) for value in repeat_seconds),
        peak_vram_allocated_bytes=peak_allocated,
        peak_vram_reserved_bytes=peak_reserved,
    )
