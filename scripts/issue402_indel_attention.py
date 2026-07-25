#!/usr/bin/env python3
"""Inspect issue #402 attention at inferred indel-shifted ortholog positions."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import polars as pl
import torch
from transformers import AutoModelForCausalLM

from marin_dna.pipelines.rag_glm.dataset import (
    BASES_PER_SLOT,
    DOCUMENT_TOKENS,
    PROVISIONAL_SPECIES_ORDER,
)
from marin_dna.pipelines.rag_glm.model_sanity import (
    assert_rag_token_geometry,
    indel_mapped_attention_rows,
    pairwise_alignment_rows,
)
from marin_dna.pipelines.rag_glm.offline_eval import (
    load_rag_model_config_hf,
    load_rag_tokenizer_hf,
    nucleotide_token_ids,
)

TRAIN_DATASET_REPO = "bolinas-dna/zoonomia-rag-v1-v1"
TRAIN_DATASET_REVISION = "5e6b30cf878b61c99e6432ad8ab7865b18cbe0e7"
DEFAULT_ANCHORS = ("win_18_000432221", "win_18_000064770")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True, choices=("46M", "104M"))
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-id", action="append")
    parser.add_argument("--query-stride", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _validation_url() -> str:
    return (
        f"https://huggingface.co/datasets/{TRAIN_DATASET_REPO}/resolve/"
        f"{TRAIN_DATASET_REVISION}/data/validation/part-00000-of-00001.parquet"
    )


def _load_selected_rows(anchor_ids: list[str]) -> list[dict[str, Any]]:
    validation = pl.read_parquet(_validation_url())
    assert validation["anchor_id"].n_unique() == validation.height
    rows: list[dict[str, Any]] = []
    for anchor_id in anchor_ids:
        selected = validation.filter(pl.col("anchor_id") == anchor_id)
        assert selected.height == 1, f"anchor not found exactly once: {anchor_id}"
        row = selected.row(0, named=True)
        assert len(row["seq"]) == 2_075
        assert row["seq"].count("[SEQ]") == 7
        assert len(row["sequence_7"]) == BASES_PER_SLOT
        rows.append(row)
    return rows


def _tokenize_document(tokenizer: Any, sequence: str) -> torch.Tensor:
    encoded = tokenizer(
        [sequence],
        add_special_tokens=True,
        padding=False,
        return_tensors="pt",
    )["input_ids"]
    assert encoded.shape == (1, DOCUMENT_TOKENS)
    assert_rag_token_geometry(
        encoded,
        bos_token_id=tokenizer.bos_token_id,
        boundary_token_id=tokenizer.convert_tokens_to_ids("[SEQ]"),
        pad_token_id=tokenizer.pad_token_id,
        unk_token_id=tokenizer.unk_token_id,
        nucleotide_token_ids=nucleotide_token_ids(tokenizer).tolist(),
    )
    return encoded


def _pairwise_alignments(
    rows: list[dict[str, Any]],
) -> tuple[dict[tuple[str, int], pl.DataFrame], pl.DataFrame, pl.DataFrame]:
    alignments: dict[tuple[str, int], pl.DataFrame] = {}
    alignment_frames: list[pl.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for row in rows:
        anchor_id = str(row["anchor_id"])
        human = str(row["sequence_7"])
        for slot in range(7):
            if not bool(row[f"available_{slot}"]):
                continue
            alignment = pairwise_alignment_rows(str(row[f"sequence_{slot}"]), human)
            alignments[(anchor_id, slot)] = alignment
            alignment_frames.append(
                alignment.with_columns(
                    pl.lit(anchor_id).alias("anchor_id"),
                    pl.lit(slot).alias("slot"),
                    pl.lit(PROVISIONAL_SPECIES_ORDER[slot]).alias("species"),
                )
            )
            aligned = alignment.filter(
                pl.col("ortholog_offset").is_not_null()
                & pl.col("human_offset").is_not_null()
            )
            shift_counts = (
                aligned.group_by("shift")
                .len()
                .sort(["len", "shift"], descending=[True, False])
            )
            summary_rows.append(
                {
                    "anchor_id": anchor_id,
                    "slot": slot,
                    "species": PROVISIONAL_SPECIES_ORDER[slot],
                    "n_aligned_bases": aligned.height,
                    "n_matches": aligned.filter(
                        pl.col("relationship") == "match"
                    ).height,
                    "aligned_identity": aligned.filter(
                        pl.col("relationship") == "match"
                    ).height
                    / aligned.height,
                    "shifted_fraction": aligned.filter(pl.col("shift") != 0).height
                    / aligned.height,
                    "dominant_shift": int(shift_counts["shift"].item(0)),
                    "shift_min": int(aligned["shift"].min()),
                    "shift_max": int(aligned["shift"].max()),
                    "ortholog_gap_columns": alignment.filter(
                        pl.col("relationship") == "ortholog_gap"
                    ).height,
                    "human_gap_columns": alignment.filter(
                        pl.col("relationship") == "human_gap"
                    ).height,
                    "alignment_score": float(alignment["alignment_score"].item(0)),
                }
            )
    assert alignments
    return alignments, pl.concat(alignment_frames), pl.DataFrame(summary_rows)


def _load_attention_model(model_path: str, device: torch.device) -> Any:
    config = load_rag_model_config_hf(model_path)
    kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    if device.type == "cuda":
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.to(device)
    model.eval()
    return model


def _run_attention(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    alignments: dict[tuple[str, int], pl.DataFrame],
    *,
    query_stride: int,
    device: torch.device,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    with torch.inference_mode():
        for document_index, row in enumerate(rows):
            anchor_id = str(row["anchor_id"])
            input_ids = _tokenize_document(tokenizer, str(row["seq"])).to(device)
            output = model(
                input_ids,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
            assert output.attentions is not None
            assert len(output.attentions) == int(model.config.num_hidden_layers)
            for layer, attention in enumerate(output.attentions):
                assert attention is not None
                attention_cpu = attention.float().cpu()
                for slot in range(7):
                    key = (anchor_id, slot)
                    if key not in alignments:
                        continue
                    frames.append(
                        indel_mapped_attention_rows(
                            attention_cpu,
                            alignments[key],
                            slot=slot,
                            layer=layer,
                            query_stride=query_stride,
                        ).with_columns(
                            pl.lit(document_index).alias("document_index"),
                            pl.lit(anchor_id).alias("anchor_id"),
                        )
                    )
            del output, input_ids
            gc.collect()
            torch.cuda.empty_cache()
    result = pl.concat(frames)
    assert result.height > 0
    return result


def main() -> None:
    args = parse_args()
    assert len(args.code_revision) == 40
    assert args.query_stride > 0
    anchor_ids = args.anchor_id or list(DEFAULT_ANCHORS)
    assert len(anchor_ids) >= 2
    assert len(set(anchor_ids)) == len(anchor_ids)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_selected_rows(anchor_ids)
    alignments, alignment_rows, alignment_summary = _pairwise_alignments(rows)
    device = torch.device(args.device)
    tokenizer = load_rag_tokenizer_hf(args.model)
    model = _load_attention_model(args.model, device)
    attention = _run_attention(
        model,
        tokenizer,
        rows,
        alignments,
        query_stride=args.query_stride,
        device=device,
    ).with_columns(pl.lit(args.model_label).alias("model"))
    attention_summary = (
        attention.group_by("model", "anchor_id", "slot", "species", "layer")
        .agg(
            pl.len().alias("n_mapped_targets"),
            pl.col("mapped_attention").mean(),
            pl.col("naive_attention").mean(),
            pl.col("mapped_minus_naive").mean(),
            (pl.col("shift") != 0).mean().alias("shifted_fraction_sampled"),
        )
        .sort("anchor_id", "slot", "layer")
    )
    for frame, name in (
        (alignment_rows, "pairwise_alignment.parquet"),
        (alignment_summary, "alignment_summary.parquet"),
        (attention, "mapped_attention.parquet"),
        (attention_summary, "mapped_attention_summary.parquet"),
    ):
        frame.write_parquet(output_dir / name, compression="zstd")

    manifest = {
        "analysis": "issue402 indel-aware manual attention check",
        "model": args.model,
        "model_label": args.model_label,
        "code_revision": args.code_revision,
        "training_dataset": TRAIN_DATASET_REPO,
        "training_dataset_revision": TRAIN_DATASET_REVISION,
        "anchor_ids": anchor_ids,
        "query_stride": args.query_stride,
        "alignment": {
            "kind": "first optimal global pairwise alignment; diagnostic, not HAL path",
            "match": 2.0,
            "mismatch": -1.0,
            "gap_open": -3.0,
            "gap_extend": -0.5,
        },
        "attention_comparison": (
            "for human target t predicted from query t-1, compare inferred mapped "
            "ortholog key to naive equal-coordinate ortholog key t"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
