#!/usr/bin/env python3
"""Extract per-position and per-document segment losses for issue #402."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl
import torch
from marin_dna_evals.rag_glm.model_sanity import (
    assert_rag_token_geometry,
    causal_token_losses,
    rag_target_position_metadata,
)
from marin_dna_evals.rag_glm.offline_eval import (
    load_rag_model_config_hf,
    load_rag_tokenizer_hf,
    nucleotide_token_ids,
)
from marin_dna_rag_glm.dataset import (
    BASES_PER_SLOT,
    DOCUMENT_TOKENS,
    PROVISIONAL_SPECIES_ORDER,
)
from transformers import AutoModelForCausalLM

TRAIN_DATASET_REPO = "bolinas-dna/zoonomia-rag-v1-v1"
TRAIN_DATASET_REVISION = "5e6b30cf878b61c99e6432ad8ab7865b18cbe0e7"
VALIDATION_URL = (
    f"https://huggingface.co/datasets/{TRAIN_DATASET_REPO}/resolve/"
    f"{TRAIN_DATASET_REVISION}/data/validation/part-00000-of-00001.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local HF checkpoint directory")
    parser.add_argument(
        "--model-source", required=True, help="Immutable checkpoint URI"
    )
    parser.add_argument("--model-label", required=True, choices=("46M", "104M"))
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-rows", type=int, default=2_048)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _model_kwargs(model: str, device: torch.device) -> dict[str, Any]:
    config = load_rag_model_config_hf(model)
    kwargs: dict[str, Any] = {"config": config, "trust_remote_code": True}
    if device.type == "cuda":
        kwargs["torch_dtype"] = torch.bfloat16
    return kwargs


def _tokenize_documents(tokenizer: Any, sequences: list[str]) -> torch.Tensor:
    encoded = tokenizer(
        sequences,
        add_special_tokens=True,
        padding=False,
        return_tensors="pt",
    )["input_ids"]
    assert encoded.shape == (len(sequences), DOCUMENT_TOKENS)
    assert_rag_token_geometry(
        encoded,
        bos_token_id=tokenizer.bos_token_id,
        boundary_token_id=tokenizer.convert_tokens_to_ids("[SEQ]"),
        pad_token_id=tokenizer.pad_token_id,
        unk_token_id=tokenizer.unk_token_id,
        nucleotide_token_ids=nucleotide_token_ids(tokenizer).tolist(),
    )
    return encoded


def _load_validation(max_rows: int) -> pl.DataFrame:
    assert max_rows > 1
    validation = pl.read_parquet(VALIDATION_URL)
    assert validation.height >= max_rows
    validation = validation.head(max_rows)
    assert validation["anchor_id"].n_unique() == max_rows
    assert validation["augmentation"].unique().to_list() == ["+"]
    assert validation.filter(pl.col("chrom") != "18").is_empty()
    assert validation.select(
        pl.col("seq").str.count_matches(r"\[SEQ\]").eq(7).all()
    ).item()
    return validation


def _base_loss_indices() -> dict[int, list[int]]:
    metadata = rag_target_position_metadata().with_row_index("loss_index")
    base_rows = metadata.filter(
        pl.col("layout_token_type").is_in(["ortholog_base", "human_base"])
    )
    assert base_rows.height == len(PROVISIONAL_SPECIES_ORDER) * BASES_PER_SLOT
    result: dict[int, list[int]] = {}
    for segment_index in range(len(PROVISIONAL_SPECIES_ORDER)):
        segment = base_rows.filter(pl.col("segment_index") == segment_index).sort(
            "within_segment_offset"
        )
        assert segment.height == BASES_PER_SLOT
        assert segment["within_segment_offset"].to_list() == list(range(BASES_PER_SLOT))
        result[segment_index] = segment["loss_index"].to_list()
    return result


def extract_losses(
    model: Any,
    tokenizer: Any,
    validation: pl.DataFrame,
    *,
    model_label: str,
    batch_size: int,
    device: torch.device,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute exact causal losses without retaining full per-token logits."""
    assert batch_size > 0
    indices = _base_loss_indices()
    position_sum = torch.zeros(DOCUMENT_TOKENS - 1, dtype=torch.float64)
    position_square_sum = torch.zeros_like(position_sum)
    position_count = 0
    document_frames: list[pl.DataFrame] = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, validation.height, batch_size):
            batch = validation.slice(start, batch_size)
            input_ids = _tokenize_documents(tokenizer, batch["seq"].to_list()).to(
                device
            )
            logits = model(input_ids, use_cache=False).logits
            losses = causal_token_losses(logits, input_ids)
            assert losses.shape == (batch.height, DOCUMENT_TOKENS - 1)
            assert bool(torch.isfinite(losses).all())

            losses_cpu = losses.double().cpu()
            position_sum += losses_cpu.sum(dim=0)
            position_square_sum += losses_cpu.square().sum(dim=0)
            position_count += batch.height

            for segment_index, loss_indices in indices.items():
                segment_loss = losses_cpu[:, loss_indices].mean(dim=1).numpy()
                document_frames.append(
                    batch.select(
                        "anchor_id",
                        "chrom",
                        "start",
                        "end",
                        pl.lit(model_label).alias("model"),
                        pl.lit(segment_index).alias("segment_index"),
                        pl.lit(PROVISIONAL_SPECIES_ORDER[segment_index]).alias(
                            "segment"
                        ),
                        pl.col(f"available_{segment_index}").alias("available"),
                        pl.col(f"quality_pass_{segment_index}").alias("quality_pass"),
                    ).with_columns(pl.Series("mean_loss", segment_loss))
                )

            del logits, losses, losses_cpu, input_ids
            print(f"[{model_label}] scored {position_count}/{validation.height}")

    assert position_count == validation.height
    means = position_sum / position_count
    variance = torch.clamp(
        (position_square_sum - position_sum.square() / position_count)
        / (position_count - 1),
        min=0.0,
    )
    position = rag_target_position_metadata().with_columns(
        pl.Series("mean_loss", means.numpy()),
        pl.Series("se_loss", (variance / position_count).sqrt().numpy()),
        pl.lit(position_count).alias("n_documents"),
        pl.lit(model_label).alias("model"),
    )
    documents = pl.concat(document_frames).sort("model", "anchor_id", "segment_index")
    assert documents.height == validation.height * len(PROVISIONAL_SPECIES_ORDER)
    assert documents.group_by("anchor_id").len()["len"].unique().to_list() == [8]
    assert documents.filter(~pl.col("mean_loss").is_finite()).is_empty()
    assert documents.filter(pl.col("available") != pl.col("quality_pass")).is_empty()
    return position, documents


def main() -> None:
    args = parse_args()
    assert len(args.code_revision) == 40
    assert args.model_source
    assert args.batch_size > 0
    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but unavailable"

    validation = _load_validation(args.validation_rows)
    tokenizer = load_rag_tokenizer_hf(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, **_model_kwargs(args.model, device)
    ).to(device)
    position, documents = extract_losses(
        model,
        tokenizer,
        validation,
        model_label=args.model_label,
        batch_size=args.batch_size,
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    position.write_parquet(
        args.output_dir / "validation_position_loss.parquet", compression="zstd"
    )
    documents.write_parquet(
        args.output_dir / "validation_document_segment_loss.parquet",
        compression="zstd",
    )
    manifest = {
        "code_revision": args.code_revision,
        "model_label": args.model_label,
        "model_source": args.model_source,
        "validation_dataset_repo": TRAIN_DATASET_REPO,
        "validation_dataset_revision": TRAIN_DATASET_REVISION,
        "validation_rows": validation.height,
        "batch_size": args.batch_size,
        "position_rows": position.height,
        "document_segment_rows": documents.height,
        "max_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
