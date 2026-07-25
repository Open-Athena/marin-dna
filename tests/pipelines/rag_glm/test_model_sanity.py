"""Focused tests for issue #402 model-behavior sanity utilities."""

from __future__ import annotations

import polars as pl
import pytest
import torch

from marin_dna.pipelines.rag_glm.dataset import (
    DOCUMENT_TOKENS,
    HUMAN_SEGMENT_START,
    MISSING_SEQUENCE,
    assemble_document,
)
from marin_dna.pipelines.rag_glm.model_sanity import (
    RAG_BOUNDARY_POSITIONS,
    ablate_rag_context,
    ablate_rag_rows,
    ablate_rag_token_ids,
    alignment_attention_rows,
    assert_rag_token_geometry,
    attention_mask_diagnostics,
    attention_region_rows,
    causal_token_losses,
    indel_mapped_attention_rows,
    pairwise_alignment_rows,
    rag_target_position_metadata,
)
from marin_dna.pipelines.rag_glm.tokenizer import create_rag_char_tokenizer


def _full_ids() -> tuple[object, torch.Tensor]:
    tokenizer = create_rag_char_tokenizer()
    document = assemble_document(
        [
            "ACG" * 85,
            "C" * 255,
            "G" * 255,
            "T" * 255,
            MISSING_SEQUENCE,
            "A" * 255,
            "C" * 255,
            "G" * 255,
        ]
    )
    ids = torch.tensor([tokenizer.encode(document)])
    return tokenizer, ids


def test_exact_geometry_detects_bos_boundary_and_human_corruption() -> None:
    tokenizer, ids = _full_ids()
    kwargs = {
        "bos_token_id": tokenizer.bos_token_id,
        "boundary_token_id": tokenizer.convert_tokens_to_ids("[SEQ]"),
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "nucleotide_token_ids": [4, 5, 6, 7],
    }
    assert_rag_token_geometry(ids, **kwargs)
    for position, replacement, message in (
        (0, 4, "BOS"),
        (RAG_BOUNDARY_POSITIONS[2], 4, "SEQ"),
        (HUMAN_SEGMENT_START, tokenizer.unk_token_id, "human"),
    ):
        corrupted = ids.clone()
        corrupted[:, position] = replacement
        with pytest.raises(AssertionError, match=message):
            assert_rag_token_geometry(corrupted, **kwargs)


def test_context_ablation_preserves_geometry_and_metadata() -> None:
    context = ("A" * 255 + "[SEQ]") * 7 + "C" * 127
    all_n = ablate_rag_context(context, "all_n")
    assert all_n == (MISSING_SEQUENCE + "[SEQ]") * 7 + "C" * 127
    assert ablate_rag_context(context, "human_only") == "C" * 127
    rows = pl.DataFrame({"context": [context], "id": [402]})
    result = ablate_rag_rows(rows, "all_n")
    assert result["id"].to_list() == [402]
    assert result["context"].item() == all_n


def test_token_ablation_changes_only_intended_regions() -> None:
    tokenizer, ids = _full_ids()
    all_n = ablate_rag_token_ids(
        ids,
        "all_n",
        unk_token_id=tokenizer.unk_token_id,
        pad_token_id=tokenizer.pad_token_id,
        boundary_token_id=tokenizer.convert_tokens_to_ids("[SEQ]"),
    )
    assert torch.equal(all_n[:, HUMAN_SEGMENT_START:], ids[:, HUMAN_SEGMENT_START:])
    assert bool((all_n[:, list(RAG_BOUNDARY_POSITIONS)] == 3).all())
    for slot in range(7):
        start = 1 + slot * 256
        assert bool((all_n[:, start : start + 255] == 1).all())

    rolled = ablate_rag_token_ids(
        ids,
        "roll",
        unk_token_id=1,
        pad_token_id=0,
        boundary_token_id=3,
        roll_bases=31,
    )
    assert torch.equal(rolled[:, HUMAN_SEGMENT_START:], ids[:, HUMAN_SEGMENT_START:])
    assert torch.equal(rolled[:, 1:256], torch.roll(ids[:, 1:256], 31, 1))


def test_causal_losses_and_position_metadata() -> None:
    input_ids = torch.tensor([[2, 4, 5]])
    logits = torch.zeros((1, 3, 8))
    logits[0, 0, 4] = 8.0
    logits[0, 1, 5] = 8.0
    losses = causal_token_losses(logits, input_ids)
    assert losses.shape == (1, 2)
    assert float(losses.max()) < 0.01
    metadata = rag_target_position_metadata()
    assert metadata.height == DOCUMENT_TOKENS - 1
    assert metadata.filter(pl.col("layout_token_type") == "boundary")[
        "position"
    ].to_list() == list(RAG_BOUNDARY_POSITIONS)
    assert metadata.filter(pl.col("layout_token_type") == "human_base").height == 255


def test_attention_alignment_and_causal_diagnostics() -> None:
    attention = torch.zeros((1, 1, DOCUMENT_TOKENS, DOCUMENT_TOKENS))
    for query in range(DOCUMENT_TOKENS):
        attention[0, 0, query, query] = 1.0
    diagnostics = attention_mask_diagnostics(attention)
    assert diagnostics == {
        "row_sum_max_abs_error": 0.0,
        "future_attention_max_abs": 0.0,
    }

    # Put equal-offset cross-segment attention into a normalized synthetic map.
    query_offsets = torch.arange(0, 255, 32)
    queries = HUMAN_SEGMENT_START + query_offsets
    keys = 1 + query_offsets
    attention[0, 0, queries, queries] = 0.5
    attention[0, 0, queries, keys] = 0.5
    rows = alignment_attention_rows(
        attention,
        torch.ones((1, 7), dtype=torch.bool),
        layer=0,
        radius=1,
        query_stride=32,
    )
    slot0 = rows.filter((pl.col("slot") == 0) & (pl.col("availability") == "available"))
    assert slot0.sort("mean_attention", descending=True)["offset"].item(0) == 0
    regions = attention_region_rows(
        attention,
        torch.ones((1, 7), dtype=torch.bool),
        layer=0,
        query_stride=32,
    )
    assert set(regions["region"]) == {"bos", "all_boundaries", "ortholog_slot"}
    assert regions.filter(pl.col("region") == "ortholog_slot").height == 7

    # Equal-length windows can still have a leading insertion and trailing
    # deletion. The inferred map shifts the aligned core by +1, so mapped
    # attention must read a different key from the naive +1 geometry.
    human = "ACGT" * 63 + "ACG"
    ortholog = "T" + human[:-1]
    alignment = pairwise_alignment_rows(ortholog, human)
    assert pairwise_alignment_rows(ortholog.lower(), human.lower()).equals(alignment)
    core = alignment.filter(
        pl.col("ortholog_offset").is_not_null() & pl.col("human_offset").is_not_null()
    )
    assert core.height == 254
    assert core["shift"].unique().to_list() == [1]
    for target in range(1, 255, 32):
        query = HUMAN_SEGMENT_START + target - 1
        attention[0, 0, query, 1 + target] = 0.25
        attention[0, 0, query, 1 + target + 1] = 0.75
    mapped = indel_mapped_attention_rows(
        attention,
        alignment,
        slot=0,
        layer=0,
        query_stride=32,
    )
    assert mapped["shift"].unique().to_list() == [1]
    assert mapped["mapped_attention"].to_list() == [0.75] * mapped.height
    assert mapped["naive_attention"].to_list() == [0.25] * mapped.height
