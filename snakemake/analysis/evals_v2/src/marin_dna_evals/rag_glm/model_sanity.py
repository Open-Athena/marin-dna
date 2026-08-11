"""Model-behavior sanity checks for issue #402's fixed-layout RAG GLMs.

The functions here deliberately encode the experiment's exact 2,048-token
contract.  They are diagnostics for frozen checkpoints, not a generalized
variable-layout RAG API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import polars as pl
import torch
import torch.nn.functional as F
from Bio.Align import PairwiseAligner
from marin_dna_rag_glm.dataset import (
    BASES_PER_SLOT,
    DOCUMENT_TOKENS,
    HUMAN_SEGMENT_START,
    MISSING_SEQUENCE,
    N_NON_HUMAN_SLOTS,
    PROVISIONAL_SPECIES_ORDER,
    SEQUENCE_BOUNDARY,
)
from torch import Tensor

RAG_BOUNDARY_POSITIONS: tuple[int, ...] = tuple(
    (slot + 1) * (BASES_PER_SLOT + 1) for slot in range(N_NON_HUMAN_SLOTS)
)
RAG_VEP_PREFIX_TOKENS = 1_920
RAG_VEP_COMPLETION_TOKENS = 128
RAG_HUMAN_ONLY_PREFIX_TOKENS = 128

ContextAblation = Literal["full", "all_n", "human_only"]
TokenAblation = Literal[
    "full",
    "all_n",
    "roll",
    "unrelated",
    "bos_to_pad",
    "seq_to_unk",
]


def assert_rag_token_geometry(
    input_ids: Tensor,
    *,
    bos_token_id: int,
    boundary_token_id: int,
    pad_token_id: int,
    unk_token_id: int,
    nucleotide_token_ids: Sequence[int],
) -> None:
    """Assert every special/base position in a full or VEP-prefix document.

    This specifically guards the historical failure mode where a CLS/BOS token
    was silently omitted during variant scoring.  Human bases must be A/C/G/T;
    ambiguity/``N`` is allowed only in the seven projected ortholog slots.
    """
    assert input_ids.ndim == 2
    n_rows, n_tokens = input_ids.shape
    assert n_rows > 0
    assert n_tokens in {RAG_VEP_PREFIX_TOKENS, DOCUMENT_TOKENS}
    nucleotide_ids = tuple(int(token_id) for token_id in nucleotide_token_ids)
    assert len(nucleotide_ids) == 4
    assert len(set(nucleotide_ids)) == 4
    special_ids = {bos_token_id, boundary_token_id, pad_token_id, unk_token_id}
    assert not special_ids & set(nucleotide_ids)

    assert bool((input_ids[:, 0] == bos_token_id).all()), (
        "[BOS]/CLS must occupy absolute token 0"
    )
    assert bool((input_ids == bos_token_id).sum(dim=1).eq(1).all()), (
        "each document must contain exactly one [BOS]/CLS"
    )
    assert not bool((input_ids == pad_token_id).any()), (
        "fixed-layout documents must not contain padding"
    )

    for row in input_ids:
        observed_boundaries = (
            torch.nonzero(row == boundary_token_id, as_tuple=False).flatten().tolist()
        )
        assert observed_boundaries == list(RAG_BOUNDARY_POSITIONS), (
            "[SEQ] tokens were deleted, duplicated, or shifted: "
            f"expected {RAG_BOUNDARY_POSITIONS}, got {observed_boundaries}"
        )

    allowed = torch.tensor(
        [bos_token_id, boundary_token_id, unk_token_id, *nucleotide_ids],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    assert bool(torch.isin(input_ids, allowed).all()), "unexpected token in RAG input"
    human = input_ids[:, HUMAN_SEGMENT_START:n_tokens]
    nucleotides = torch.tensor(
        nucleotide_ids, dtype=input_ids.dtype, device=input_ids.device
    )
    assert human.numel() > 0
    assert bool(torch.isin(human, nucleotides).all()), (
        "human prefix/completion context must contain only A/C/G/T tokens"
    )


def split_rag_context(context: str) -> tuple[tuple[str, ...], str]:
    """Split one materialized RAG context into seven orthologs and human text."""
    parts = context.split(SEQUENCE_BOUNDARY)
    assert len(parts) == N_NON_HUMAN_SLOTS + 1
    orthologs = tuple(parts[:N_NON_HUMAN_SLOTS])
    human = parts[-1]
    assert [len(sequence) for sequence in orthologs] == [BASES_PER_SLOT] * 7
    assert len(human) in {127, BASES_PER_SLOT}
    return orthologs, human


def ablate_rag_context(context: str, mode: ContextAblation) -> str:
    """Apply a fixed-geometry all-N or literal human-only context ablation."""
    orthologs, human = split_rag_context(context)
    if mode == "full":
        result = context
    elif mode == "all_n":
        result = SEQUENCE_BOUNDARY.join((*([MISSING_SEQUENCE] * 7), human))
    elif mode == "human_only":
        result = human
    else:  # pragma: no cover - Literal plus loud runtime guard
        raise AssertionError(f"unknown context ablation {mode!r}")
    if mode != "human_only":
        result_orthologs, result_human = split_rag_context(result)
        assert result_human == human
        assert len(result_orthologs) == len(orthologs)
    return result


def ablate_rag_rows(rows: pl.DataFrame, mode: ContextAblation) -> pl.DataFrame:
    """Replace only the context column while preserving row order and metadata."""
    assert "context" in rows.columns
    assert rows.height > 0
    contexts = [ablate_rag_context(context, mode) for context in rows["context"]]
    result = rows.with_columns(pl.Series("context", contexts))
    assert result.height == rows.height
    assert result.drop("context").equals(rows.drop("context"))
    return result


def ablate_rag_token_ids(
    input_ids: Tensor,
    mode: TokenAblation,
    *,
    unk_token_id: int,
    pad_token_id: int,
    boundary_token_id: int,
    donor_input_ids: Tensor | None = None,
    roll_bases: int = 31,
) -> Tensor:
    """Apply controlled ablations to full documents or fixed VEP prefixes."""
    assert input_ids.ndim == 2
    assert input_ids.shape[1] in {RAG_VEP_PREFIX_TOKENS, DOCUMENT_TOKENS}
    result = input_ids.clone()
    if mode == "full":
        pass
    elif mode == "all_n":
        for slot in range(N_NON_HUMAN_SLOTS):
            start = 1 + slot * (BASES_PER_SLOT + 1)
            result[:, start : start + BASES_PER_SLOT] = unk_token_id
    elif mode == "roll":
        assert roll_bases % BASES_PER_SLOT != 0
        for slot in range(N_NON_HUMAN_SLOTS):
            start = 1 + slot * (BASES_PER_SLOT + 1)
            result[:, start : start + BASES_PER_SLOT] = torch.roll(
                result[:, start : start + BASES_PER_SLOT],
                shifts=roll_bases,
                dims=1,
            )
    elif mode == "unrelated":
        assert donor_input_ids is not None
        assert donor_input_ids.shape == input_ids.shape
        for slot in range(N_NON_HUMAN_SLOTS):
            start = 1 + slot * (BASES_PER_SLOT + 1)
            result[:, start : start + BASES_PER_SLOT] = donor_input_ids[
                :, start : start + BASES_PER_SLOT
            ]
    elif mode == "bos_to_pad":
        result[:, 0] = pad_token_id
    elif mode == "seq_to_unk":
        result[:, list(RAG_BOUNDARY_POSITIONS)] = unk_token_id
    else:  # pragma: no cover - Literal plus loud runtime guard
        raise AssertionError(f"unknown token ablation {mode!r}")

    if mode not in {"bos_to_pad", "seq_to_unk"}:
        assert torch.equal(result[:, 0], input_ids[:, 0])
        assert torch.equal(
            result[:, list(RAG_BOUNDARY_POSITIONS)],
            input_ids[:, list(RAG_BOUNDARY_POSITIONS)],
        )
    elif mode == "bos_to_pad":
        assert bool((result[:, 0] == pad_token_id).all())
        assert torch.equal(
            result[:, list(RAG_BOUNDARY_POSITIONS)],
            input_ids[:, list(RAG_BOUNDARY_POSITIONS)],
        )
    else:
        assert torch.equal(result[:, 0], input_ids[:, 0])
        assert bool((result[:, list(RAG_BOUNDARY_POSITIONS)] == unk_token_id).all())
    assert torch.equal(
        result[:, HUMAN_SEGMENT_START:], input_ids[:, HUMAN_SEGMENT_START:]
    )
    return result


def paired_special_token_llr_diagnostics(
    full: pl.DataFrame,
    ablated: pl.DataFrame,
    *,
    benchmark: str,
    ablation: str,
    change_threshold: float = 1.0e-6,
) -> pl.DataFrame:
    """Summarize paired document LLR changes under one token intervention."""
    assert benchmark
    assert ablation in {"bos_to_pad", "seq_to_unk"}
    assert change_threshold > 0
    required = {"document_id", "llr"}
    assert required <= set(full.columns)
    assert required <= set(ablated.columns)
    assert full.height == ablated.height > 1
    assert full["document_id"].n_unique() == full.height
    assert ablated["document_id"].n_unique() == ablated.height
    assert full["document_id"].to_list() == ablated["document_id"].to_list()

    comparison = full.select(
        "document_id", pl.col("llr").alias("full_llr")
    ).with_columns(pl.Series("ablated_llr", ablated["llr"].to_list(), dtype=pl.Float64))
    comparison = comparison.with_columns(
        (pl.col("ablated_llr") - pl.col("full_llr")).alias("llr_delta")
    )
    summary = comparison.select(
        pl.len().alias("n_documents"),
        pl.col("llr_delta").abs().mean().alias("mean_abs_llr_delta"),
        pl.col("llr_delta").abs().max().alias("max_abs_llr_delta"),
        (pl.col("llr_delta").abs() > change_threshold)
        .mean()
        .alias("fraction_llr_changed_gt_1e_6"),
        pl.corr("full_llr", "ablated_llr").alias("llr_pearson"),
    ).with_columns(
        pl.lit(benchmark).alias("benchmark"),
        pl.lit(ablation).alias("ablation"),
    )
    assert summary["n_documents"].item() == full.height
    assert summary.filter(
        ~pl.col("mean_abs_llr_delta").is_finite()
        | ~pl.col("max_abs_llr_delta").is_finite()
        | ~pl.col("llr_pearson").is_finite()
    ).is_empty()
    assert summary["max_abs_llr_delta"].item() > change_threshold, summary
    assert summary["fraction_llr_changed_gt_1e_6"].item() > 0, summary
    return summary.select(
        "benchmark",
        "ablation",
        "n_documents",
        "mean_abs_llr_delta",
        "max_abs_llr_delta",
        "fraction_llr_changed_gt_1e_6",
        "llr_pearson",
    )


def causal_token_losses(logits: Tensor, input_ids: Tensor) -> Tensor:
    """Return full-vocabulary next-token CE at target positions 1..L-1."""
    assert logits.ndim == 3
    assert input_ids.ndim == 2
    assert logits.shape[:2] == input_ids.shape
    assert logits.shape[2] > int(input_ids.max())
    losses = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2),
        input_ids[:, 1:],
        reduction="none",
    )
    assert losses.shape == (input_ids.shape[0], input_ids.shape[1] - 1)
    assert bool(torch.isfinite(losses).all())
    return losses


def rag_target_position_metadata() -> pl.DataFrame:
    """Describe every causal target position in the fixed 2,048-token layout."""
    rows: list[dict[str, object]] = []
    species = PROVISIONAL_SPECIES_ORDER
    for position in range(1, DOCUMENT_TOKENS):
        if position in RAG_BOUNDARY_POSITIONS:
            segment_index = RAG_BOUNDARY_POSITIONS.index(position)
            rows.append(
                {
                    "position": position,
                    "segment_index": segment_index,
                    "segment": species[segment_index],
                    "within_segment_offset": BASES_PER_SLOT,
                    "layout_token_type": "boundary",
                }
            )
        elif position < HUMAN_SEGMENT_START:
            segment_index = (position - 1) // (BASES_PER_SLOT + 1)
            rows.append(
                {
                    "position": position,
                    "segment_index": segment_index,
                    "segment": species[segment_index],
                    "within_segment_offset": (position - 1) % (BASES_PER_SLOT + 1),
                    "layout_token_type": "ortholog_base",
                }
            )
        else:
            rows.append(
                {
                    "position": position,
                    "segment_index": N_NON_HUMAN_SLOTS,
                    "segment": species[-1],
                    "within_segment_offset": position - HUMAN_SEGMENT_START,
                    "layout_token_type": "human_base",
                }
            )
    result = pl.DataFrame(rows)
    assert result.height == DOCUMENT_TOKENS - 1
    assert result["position"].to_list() == list(range(1, DOCUMENT_TOKENS))
    return result


def attention_mask_diagnostics(attention: Tensor) -> dict[str, float]:
    """Measure normalization and exact causal masking for one attention layer."""
    assert attention.ndim == 4
    assert attention.shape[-2:] == (DOCUMENT_TOKENS, DOCUMENT_TOKENS)
    row_error = (attention.float().sum(dim=-1) - 1.0).abs().max()
    future_max = torch.triu(attention, diagonal=1).float().abs().max()
    return {
        "row_sum_max_abs_error": float(row_error.cpu()),
        "future_attention_max_abs": float(future_max.cpu()),
    }


def alignment_attention_rows(
    attention: Tensor,
    availability: Tensor,
    *,
    layer: int,
    radius: int = 32,
    query_stride: int = 4,
) -> pl.DataFrame:
    """Summarize human-query attention near equal-offset ortholog keys.

    Attention is averaged over heads, sampled human offsets, and documents,
    separately for present and missing projected slots.
    """
    assert attention.ndim == 4
    batch, _, queries, keys = attention.shape
    assert (queries, keys) == (DOCUMENT_TOKENS, DOCUMENT_TOKENS)
    assert availability.shape == (batch, N_NON_HUMAN_SLOTS)
    assert radius >= 0
    assert query_stride > 0
    rows: list[dict[str, object]] = []
    for slot in range(N_NON_HUMAN_SLOTS):
        slot_start = 1 + slot * (BASES_PER_SLOT + 1)
        for availability_name, available in (
            ("available", availability[:, slot].bool()),
            ("missing", ~availability[:, slot].bool()),
        ):
            n_documents = int(available.sum())
            if n_documents == 0:
                continue
            selected = attention[available].float()
            for offset in range(-radius, radius + 1):
                base_start = max(0, -offset)
                base_stop = min(BASES_PER_SLOT, BASES_PER_SLOT - offset)
                base_offsets = torch.arange(
                    base_start,
                    base_stop,
                    query_stride,
                    device=attention.device,
                )
                query_positions = HUMAN_SEGMENT_START + base_offsets
                key_positions = slot_start + base_offsets + offset
                values = selected[:, :, query_positions, key_positions]
                rows.append(
                    {
                        "layer": layer,
                        "slot": slot,
                        "species": PROVISIONAL_SPECIES_ORDER[slot],
                        "availability": availability_name,
                        "offset": offset,
                        "mean_attention": float(values.mean().cpu()),
                        "n_documents": n_documents,
                        "n_query_offsets": int(base_offsets.numel()),
                    }
                )
    result = pl.DataFrame(rows)
    assert result.height > 0
    assert result.filter(~pl.col("mean_attention").is_finite()).is_empty()
    return result


def pairwise_alignment_rows(
    ortholog_sequence: str,
    human_sequence: str,
) -> pl.DataFrame:
    """Infer a basewise ortholog↔human map for one fixed projected window.

    This is a manual diagnostic rather than a reconstruction of the HAL path:
    the first optimal global pairwise alignment is used with a fixed scoring
    scheme. Retaining one row per alignment column makes gaps in either
    sequence explicit and lets downstream attention checks map a human target
    base to a possibly shifted ortholog key.
    """
    for name, sequence in (
        ("ortholog_sequence", ortholog_sequence),
        ("human_sequence", human_sequence),
    ):
        assert len(sequence) == BASES_PER_SLOT, (
            f"{name} must have {BASES_PER_SLOT} bases, got {len(sequence)}"
        )
        assert set(sequence.upper()) <= set("ACGTN"), (
            f"{name} contains an unexpected base"
        )

    # Case encodes per-base conservation in the frozen training corpus; it is
    # not a nucleotide difference and must not affect the inferred alignment.
    ortholog_sequence = ortholog_sequence.upper()
    human_sequence = human_sequence.upper()

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -3.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(ortholog_sequence, human_sequence)[0]
    ortholog_indices, human_indices = alignment.indices
    assert len(ortholog_indices) == len(human_indices)

    rows: list[dict[str, object]] = []
    for column, (ortholog_index_raw, human_index_raw) in enumerate(
        zip(ortholog_indices, human_indices, strict=True)
    ):
        ortholog_index = int(ortholog_index_raw)
        human_index = int(human_index_raw)
        ortholog_offset = None if ortholog_index < 0 else ortholog_index
        human_offset = None if human_index < 0 else human_index
        ortholog_base = (
            "-" if ortholog_offset is None else ortholog_sequence[ortholog_offset]
        )
        human_base = "-" if human_offset is None else human_sequence[human_offset]
        if ortholog_offset is None:
            relationship = "ortholog_gap"
        elif human_offset is None:
            relationship = "human_gap"
        elif ortholog_base == human_base:
            relationship = "match"
        else:
            relationship = "mismatch"
        rows.append(
            {
                "alignment_column": column,
                "ortholog_offset": ortholog_offset,
                "human_offset": human_offset,
                "ortholog_base": ortholog_base,
                "human_base": human_base,
                "relationship": relationship,
                "shift": (
                    None
                    if ortholog_offset is None or human_offset is None
                    else ortholog_offset - human_offset
                ),
                "alignment_score": float(alignment.score),
            }
        )
    result = pl.DataFrame(rows)
    assert result["alignment_column"].to_list() == list(range(result.height))
    assert result.filter(pl.col("ortholog_offset").is_not_null())[
        "ortholog_offset"
    ].cast(pl.Int64).to_list() == list(range(BASES_PER_SLOT))
    assert result.filter(pl.col("human_offset").is_not_null())["human_offset"].cast(
        pl.Int64
    ).to_list() == list(range(BASES_PER_SLOT))
    assert result.filter(
        pl.col("relationship").is_in(["ortholog_gap", "human_gap"])
    ).height == 2 * (result.height - BASES_PER_SLOT)
    return result


def indel_mapped_attention_rows(
    attention: Tensor,
    alignment: pl.DataFrame,
    *,
    slot: int,
    layer: int,
    query_stride: int = 4,
) -> pl.DataFrame:
    """Compare attention at pairwise-mapped and naive equal-offset keys.

    Human target offset ``t`` is predicted by the query at ``t - 1``. The
    naive no-indel key is therefore ortholog offset ``t`` (the ``+1`` peak in
    the raw-offset diagnostic); the mapped key comes from
    :func:`pairwise_alignment_rows` and may differ after an indel.
    """
    assert attention.ndim == 4
    assert attention.shape[0] == 1
    assert attention.shape[-2:] == (DOCUMENT_TOKENS, DOCUMENT_TOKENS)
    assert 0 <= slot < N_NON_HUMAN_SLOTS
    assert layer >= 0
    assert query_stride > 0
    required = {
        "ortholog_offset",
        "human_offset",
        "ortholog_base",
        "human_base",
        "relationship",
        "shift",
    }
    assert required <= set(alignment.columns)
    mapped = alignment.filter(
        pl.col("ortholog_offset").is_not_null()
        & pl.col("human_offset").is_between(1, BASES_PER_SLOT - 1)
        & (((pl.col("human_offset") - 1) % query_stride) == 0)
    ).sort("human_offset")
    assert mapped.height > 0
    assert mapped.filter(
        ~pl.col("relationship").is_in(["match", "mismatch"])
    ).is_empty()

    human_targets = torch.tensor(
        mapped["human_offset"].cast(pl.Int64).to_list(),
        dtype=torch.long,
        device=attention.device,
    )
    ortholog_offsets = torch.tensor(
        mapped["ortholog_offset"].cast(pl.Int64).to_list(),
        dtype=torch.long,
        device=attention.device,
    )
    query_positions = HUMAN_SEGMENT_START + human_targets - 1
    slot_start = 1 + slot * (BASES_PER_SLOT + 1)
    mapped_keys = slot_start + ortholog_offsets
    naive_keys = slot_start + human_targets
    assert bool((mapped_keys >= slot_start).all())
    assert bool((mapped_keys < slot_start + BASES_PER_SLOT).all())
    assert bool((naive_keys < slot_start + BASES_PER_SLOT).all())

    mapped_attention = attention[0, :, query_positions, mapped_keys].float().mean(dim=0)
    naive_attention = attention[0, :, query_positions, naive_keys].float().mean(dim=0)
    result = (
        mapped.select(
            pl.col("human_offset").cast(pl.Int64).alias("human_target_offset"),
            pl.col("ortholog_offset").cast(pl.Int64),
            pl.col("shift").cast(pl.Int64),
            "human_base",
            "ortholog_base",
            "relationship",
        )
        .with_columns(
            pl.lit(layer).alias("layer"),
            pl.lit(slot).alias("slot"),
            pl.lit(PROVISIONAL_SPECIES_ORDER[slot]).alias("species"),
            pl.Series("mapped_attention", mapped_attention.cpu().tolist()),
            pl.Series("naive_attention", naive_attention.cpu().tolist()),
        )
        .with_columns(
            (pl.col("mapped_attention") - pl.col("naive_attention")).alias(
                "mapped_minus_naive"
            )
        )
    )
    assert result.filter(
        ~pl.col("mapped_attention").is_finite() | ~pl.col("naive_attention").is_finite()
    ).is_empty()
    return result


def attention_region_rows(
    attention: Tensor,
    availability: Tensor,
    *,
    layer: int,
    query_stride: int = 4,
) -> pl.DataFrame:
    """Summarize human-query mass to BOS, boundaries, and each ortholog slot."""
    assert attention.ndim == 4
    batch, _, queries, keys = attention.shape
    assert (queries, keys) == (DOCUMENT_TOKENS, DOCUMENT_TOKENS)
    assert availability.shape == (batch, N_NON_HUMAN_SLOTS)
    assert query_stride > 0
    query_positions = torch.arange(
        HUMAN_SEGMENT_START,
        DOCUMENT_TOKENS,
        query_stride,
        device=attention.device,
    )
    selected_queries = attention[:, :, query_positions].float()
    rows: list[dict[str, object]] = []
    for region, key_positions in (
        ("bos", [0]),
        ("all_boundaries", list(RAG_BOUNDARY_POSITIONS)),
    ):
        mass = selected_queries[..., key_positions].sum(dim=-1)
        rows.append(
            {
                "layer": layer,
                "region": region,
                "slot": None,
                "species": None,
                "availability": "all",
                "mean_attention_mass": float(mass.mean().cpu()),
                "n_documents": batch,
                "n_query_offsets": int(query_positions.numel()),
            }
        )

    for slot in range(N_NON_HUMAN_SLOTS):
        slot_start = 1 + slot * (BASES_PER_SLOT + 1)
        key_positions = slice(slot_start, slot_start + BASES_PER_SLOT)
        for availability_name, available in (
            ("available", availability[:, slot].bool()),
            ("missing", ~availability[:, slot].bool()),
        ):
            n_documents = int(available.sum())
            if n_documents == 0:
                continue
            mass = selected_queries[available, ..., key_positions].sum(dim=-1)
            rows.append(
                {
                    "layer": layer,
                    "region": "ortholog_slot",
                    "slot": slot,
                    "species": PROVISIONAL_SPECIES_ORDER[slot],
                    "availability": availability_name,
                    "mean_attention_mass": float(mass.mean().cpu()),
                    "n_documents": n_documents,
                    "n_query_offsets": int(query_positions.numel()),
                }
            )
    result = pl.DataFrame(rows)
    assert result.height >= 2
    assert result.filter(~pl.col("mean_attention_mass").is_finite()).is_empty()
    return result
