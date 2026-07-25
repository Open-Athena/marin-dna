#!/usr/bin/env python3
"""Run frozen-checkpoint behavioral sanity checks for issue #402.

This intentionally combines the expensive, checkpoint-bound diagnostics in one
SkyPilot job: validation loss profiles/context ablations, attention alignment,
and ortholog-free VEP on all three frozen benchmark harnesses.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForCausalLM

from marin_dna.pipelines.rag_glm.dataset import (
    DOCUMENT_TOKENS,
    HUMAN_SEGMENT_START,
)
from marin_dna.pipelines.rag_glm.hf_scoring import (
    RAG_COMPLETION_TOKENS,
    score_rag_completions_hf,
)
from marin_dna.pipelines.rag_glm.model_sanity import (
    RAG_HUMAN_ONLY_PREFIX_TOKENS,
    RAG_VEP_PREFIX_TOKENS,
    ablate_rag_rows,
    ablate_rag_token_ids,
    alignment_attention_rows,
    assert_rag_token_geometry,
    attention_mask_diagnostics,
    attention_region_rows,
    causal_token_losses,
    rag_target_position_metadata,
)
from marin_dna.pipelines.rag_glm.offline_eval import (
    DOCUMENT_SCORE_COLUMNS,
    RAG_BENCHMARK_DATASETS,
    RAGBenchmark,
    aggregate_rag_variant_scores,
    compute_rag_benchmark_metrics,
    load_rag_eval_split,
    load_rag_model_config_hf,
    load_rag_tokenizer_hf,
    nucleotide_token_ids,
)

TRAIN_DATASET_REPO = "bolinas-dna/zoonomia-rag-v1-v1"
TRAIN_DATASET_REVISION = "5e6b30cf878b61c99e6432ad8ab7865b18cbe0e7"
BENCHMARKS: tuple[RAGBenchmark, ...] = (
    "mendelian_traits",
    "complex_traits",
    "sge",
)
VEP_MODES = ("all_n", "human_only")
LM_ABLATIONS = ("full", "all_n", "roll", "unrelated", "bos_to_pad", "seq_to_unk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local HF checkpoint directory")
    parser.add_argument("--model-label", required=True, choices=("46M", "104M"))
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vep-batch-size", type=int, default=16)
    parser.add_argument("--validation-rows", type=int, default=2_048)
    parser.add_argument("--ablation-rows", type=int, default=512)
    parser.add_argument("--attention-rows", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=1_000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _model_kwargs(args: argparse.Namespace, *, eager: bool = False) -> dict[str, Any]:
    config = load_rag_model_config_hf(args.model)
    kwargs: dict[str, Any] = {"config": config, "trust_remote_code": True}
    if torch.device(args.device).type == "cuda":
        kwargs["torch_dtype"] = torch.bfloat16
    if eager:
        kwargs["attn_implementation"] = "eager"
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


def _load_validation_rows(max_rows: int) -> list[dict[str, Any]]:
    assert max_rows > 0
    url = (
        f"https://huggingface.co/datasets/{TRAIN_DATASET_REPO}/resolve/"
        f"{TRAIN_DATASET_REVISION}/data/validation/part-00000-of-00001.parquet"
    )
    validation = pl.read_parquet(url)
    assert validation.height >= max_rows
    rows = validation.head(max_rows).to_dicts()
    assert all(len(row["seq"]) == 2_075 for row in rows)
    assert all(row["seq"].count("[SEQ]") == 7 for row in rows)
    return rows


def _loss_summaries(
    model: Any,
    tokenizer: Any,
    validation_rows: list[dict[str, Any]],
    *,
    batch_size: int,
    ablation_rows: int,
    device: torch.device,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    assert 0 < ablation_rows <= len(validation_rows)
    position_sum = torch.zeros(DOCUMENT_TOKENS - 1, dtype=torch.float64)
    position_square_sum = torch.zeros_like(position_sum)
    position_count = 0
    mode_sums = {mode: 0.0 for mode in LM_ABLATIONS}
    mode_square_sums = {mode: 0.0 for mode in LM_ABLATIONS}
    mode_counts = {mode: 0 for mode in LM_ABLATIONS}
    human_loss_start = HUMAN_SEGMENT_START - 1

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(validation_rows), batch_size):
            batch_rows = validation_rows[start : start + batch_size]
            intact_cpu = _tokenize_documents(
                tokenizer, [str(row["seq"]) for row in batch_rows]
            )
            intact = intact_cpu.to(device)
            full_logits = model(intact, use_cache=False).logits
            full_losses = causal_token_losses(full_logits, intact)
            position_sum += full_losses.double().sum(dim=0).cpu()
            position_square_sum += full_losses.double().square().sum(dim=0).cpu()
            position_count += intact.shape[0]

            remaining = max(0, ablation_rows - start)
            n_ablation = min(intact.shape[0], remaining)
            if n_ablation == 0:
                continue
            ablation_intact = intact[:n_ablation]
            targets = ablation_intact[:, 1:]
            donor = torch.roll(ablation_intact, shifts=1, dims=0)
            if n_ablation == 1:
                # Pair the singleton with a deterministic document from this batch.
                assert intact.shape[0] > 1
                donor = intact[1:2]
            for mode in LM_ABLATIONS:
                if mode == "full":
                    mode_losses = full_losses[:n_ablation]
                else:
                    altered = ablate_rag_token_ids(
                        ablation_intact,
                        mode,  # type: ignore[arg-type]
                        unk_token_id=tokenizer.unk_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        boundary_token_id=tokenizer.convert_tokens_to_ids("[SEQ]"),
                        donor_input_ids=donor if mode == "unrelated" else None,
                    )
                    logits = model(altered, use_cache=False).logits
                    # Score the frozen intact human targets under the altered context.
                    losses = torch.nn.functional.cross_entropy(
                        logits[:, :-1].float().transpose(1, 2),
                        targets,
                        reduction="none",
                    )
                    assert losses.shape == targets.shape
                    mode_losses = losses
                per_document = mode_losses[:, human_loss_start:].mean(dim=1).double()
                assert bool(torch.isfinite(per_document).all())
                mode_sums[mode] += float(per_document.sum().cpu())
                mode_square_sums[mode] += float(per_document.square().sum().cpu())
                mode_counts[mode] += int(per_document.numel())
            del full_logits, full_losses

    assert position_count == len(validation_rows)
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
    )
    mode_rows: list[dict[str, object]] = []
    for mode in LM_ABLATIONS:
        count = mode_counts[mode]
        assert count == ablation_rows
        mean = mode_sums[mode] / count
        variance = max(
            0.0,
            (mode_square_sums[mode] - mode_sums[mode] ** 2 / count) / (count - 1),
        )
        mode_rows.append(
            {
                "mode": mode,
                "mean_human_loss": mean,
                "se_human_loss": (variance / count) ** 0.5,
                "n_documents": count,
            }
        )
    return position, pl.DataFrame(mode_rows)


def _encode_vep_batch(
    tokenizer: Any, rows: pl.DataFrame, mode: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    assert mode in VEP_MODES
    prefix = tokenizer(
        rows["context"].to_list(),
        add_special_tokens=True,
        padding=False,
        return_tensors="pt",
    )["input_ids"]
    ref = tokenizer(
        rows["ref_completion"].to_list(),
        add_special_tokens=False,
        padding=False,
        return_tensors="pt",
    )["input_ids"]
    alt = tokenizer(
        rows["alt_completion"].to_list(),
        add_special_tokens=False,
        padding=False,
        return_tensors="pt",
    )["input_ids"]
    expected_prefix = (
        RAG_VEP_PREFIX_TOKENS if mode == "all_n" else RAG_HUMAN_ONLY_PREFIX_TOKENS
    )
    assert prefix.shape == (rows.height, expected_prefix)
    assert ref.shape == alt.shape == (rows.height, RAG_COMPLETION_TOKENS)
    nucleotide_ids = nucleotide_token_ids(tokenizer)
    assert bool(torch.isin(ref, nucleotide_ids).all())
    assert bool(torch.isin(alt, nucleotide_ids).all())
    assert bool((ref[:, 1:] == alt[:, 1:]).all())
    assert bool((ref[:, 0] != alt[:, 0]).all())
    if mode == "all_n":
        assert_rag_token_geometry(
            prefix,
            bos_token_id=tokenizer.bos_token_id,
            boundary_token_id=tokenizer.convert_tokens_to_ids("[SEQ]"),
            pad_token_id=tokenizer.pad_token_id,
            unk_token_id=tokenizer.unk_token_id,
            nucleotide_token_ids=nucleotide_ids.tolist(),
        )
        ortholog_prefix = prefix[:, 1 : HUMAN_SEGMENT_START - 1]
        assert bool(
            (
                (ortholog_prefix == tokenizer.unk_token_id)
                | (ortholog_prefix == tokenizer.convert_tokens_to_ids("[SEQ]"))
            ).all()
        )
    else:
        assert bool((prefix[:, 0] == tokenizer.bos_token_id).all())
        assert bool((prefix == tokenizer.bos_token_id).sum(dim=1).eq(1).all())
        assert bool(torch.isin(prefix[:, 1:], nucleotide_ids).all())
    return prefix, ref, alt, expected_prefix


def _score_vep_mode(
    model: Any,
    tokenizer: Any,
    rows: pl.DataFrame,
    *,
    mode: str,
    batch_size: int,
    device: torch.device,
) -> pl.DataFrame:
    transformed = ablate_rag_rows(rows, mode)  # type: ignore[arg-type]
    score_chunks: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, transformed.height, batch_size):
            batch = transformed.slice(start, batch_size)
            prefix, ref, alt, expected_prefix = _encode_vep_batch(
                tokenizer, batch, mode
            )
            score_chunks.append(
                score_rag_completions_hf(
                    model,
                    prefix.to(device),
                    ref.to(device),
                    alt.to(device),
                    nucleotide_token_ids=nucleotide_token_ids(tokenizer).to(device),
                    expected_prefix_tokens=expected_prefix,
                )
                .float()
                .cpu()
            )
    scores = torch.cat(score_chunks).numpy()
    assert scores.shape == (rows.height, 3)
    assert np.isfinite(scores).all()
    return transformed.with_columns(
        *[
            pl.Series(column, scores[:, index], dtype=pl.Float32)
            for index, column in enumerate(DOCUMENT_SCORE_COLUMNS)
        ]
    )


def _run_vep_ablations(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    *,
    batch_size: int,
    n_bootstrap: int,
    device: torch.device,
) -> None:
    for benchmark in BENCHMARKS:
        rows = load_rag_eval_split(benchmark, "test")
        for mode in VEP_MODES:
            mode_dir = output_dir / "vep" / benchmark / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            documents = _score_vep_mode(
                model,
                tokenizer,
                rows,
                mode=mode,
                batch_size=batch_size,
                device=device,
            )
            variants = aggregate_rag_variant_scores(documents, benchmark)
            metrics = compute_rag_benchmark_metrics(
                variants, benchmark, n_bootstrap=n_bootstrap
            ).with_columns(pl.lit(mode).alias("context_mode"))
            documents.write_parquet(mode_dir / "documents.parquet", compression="zstd")
            variants.write_parquet(mode_dir / "variants.parquet", compression="zstd")
            metrics.write_parquet(mode_dir / "metrics.parquet", compression="zstd")
            (mode_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "benchmark": benchmark,
                        "benchmark_repo": RAG_BENCHMARK_DATASETS[benchmark][0],
                        "benchmark_revision": RAG_BENCHMARK_DATASETS[benchmark][1],
                        "context_mode": mode,
                        "n_document_rows": documents.height,
                        "n_variants": variants.height,
                        "geometry": (
                            "fixed_2048_all_ortholog_slots_N"
                            if mode == "all_n"
                            else "literal_256_BOS_plus_human_out_of_distribution"
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )


def _select_attention_rows(
    validation_rows: list[dict[str, Any]], n_rows: int
) -> list[dict[str, Any]]:
    assert n_rows > 0
    selected: list[dict[str, Any]] = []
    signatures: set[tuple[bool, ...]] = set()
    for row in validation_rows:
        signature = tuple(bool(row[f"available_{slot}"]) for slot in range(7))
        if signature not in signatures:
            selected.append(row)
            signatures.add(signature)
        if len(selected) == n_rows:
            break
    if len(selected) < n_rows:
        for row in validation_rows:
            if row not in selected:
                selected.append(row)
            if len(selected) == n_rows:
                break
    assert len(selected) == n_rows
    return selected


def _run_attention(
    model: Any,
    tokenizer: Any,
    validation_rows: list[dict[str, Any]],
    *,
    n_rows: int,
    device: torch.device,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    selected = _select_attention_rows(validation_rows, n_rows)
    alignment_frames: list[pl.DataFrame] = []
    region_frames: list[pl.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []
    model.eval()
    with torch.inference_mode():
        for document_index, row in enumerate(selected):
            input_ids = _tokenize_documents(tokenizer, [str(row["seq"])]).to(device)
            availability = torch.tensor(
                [[bool(row[f"available_{slot}"]) for slot in range(7)]],
                dtype=torch.bool,
                device=device,
            )
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
                # One transfer per layer avoids thousands of scalar GPU syncs in
                # the offset summaries, while the expensive forward stays on GPU.
                attention_cpu = attention.float().cpu()
                availability_cpu = availability.cpu()
                diagnostics = attention_mask_diagnostics(attention_cpu)
                diagnostic_rows.append(
                    {
                        "document_index": document_index,
                        "anchor_id": str(row["anchor_id"]),
                        "layer": layer,
                        **diagnostics,
                    }
                )
                alignment_frames.append(
                    alignment_attention_rows(
                        attention_cpu,
                        availability_cpu,
                        layer=layer,
                        radius=32,
                        query_stride=4,
                    ).with_columns(
                        pl.lit(document_index).alias("document_index"),
                        pl.lit(str(row["anchor_id"])).alias("anchor_id"),
                    )
                )
                region_frames.append(
                    attention_region_rows(
                        attention_cpu,
                        availability_cpu,
                        layer=layer,
                        query_stride=4,
                    ).with_columns(
                        pl.lit(document_index).alias("document_index"),
                        pl.lit(str(row["anchor_id"])).alias("anchor_id"),
                    )
                )
            del output, input_ids
            torch.cuda.empty_cache()
    return (
        pl.concat(alignment_frames),
        pl.concat(region_frames),
        pl.DataFrame(diagnostic_rows),
    )


def main() -> None:
    args = parse_args()
    assert len(args.code_revision) == 40
    assert args.batch_size > 1
    assert args.vep_batch_size > 0
    assert args.validation_rows > 1
    assert 1 < args.ablation_rows <= args.validation_rows
    assert 0 < args.attention_rows <= args.validation_rows
    assert args.n_bootstrap >= 0
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = load_rag_tokenizer_hf(args.model)
    validation_rows = _load_validation_rows(args.validation_rows)

    model = AutoModelForCausalLM.from_pretrained(args.model, **_model_kwargs(args))
    model.to(device)
    position, ablations = _loss_summaries(
        model,
        tokenizer,
        validation_rows,
        batch_size=args.batch_size,
        ablation_rows=args.ablation_rows,
        device=device,
    )
    position.with_columns(pl.lit(args.model_label).alias("model")).write_parquet(
        output_dir / "validation_position_loss.parquet", compression="zstd"
    )
    ablations.with_columns(pl.lit(args.model_label).alias("model")).write_parquet(
        output_dir / "validation_context_ablation.parquet", compression="zstd"
    )
    _run_vep_ablations(
        model,
        tokenizer,
        output_dir,
        batch_size=args.vep_batch_size,
        n_bootstrap=args.n_bootstrap,
        device=device,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    attention_model = AutoModelForCausalLM.from_pretrained(
        args.model, **_model_kwargs(args, eager=True)
    )
    attention_model.to(device)
    alignment, regions, diagnostics = _run_attention(
        attention_model,
        tokenizer,
        validation_rows,
        n_rows=args.attention_rows,
        device=device,
    )
    for frame, filename in (
        (alignment, "attention_alignment.parquet"),
        (regions, "attention_regions.parquet"),
        (diagnostics, "attention_diagnostics.parquet"),
    ):
        frame.with_columns(pl.lit(args.model_label).alias("model")).write_parquet(
            output_dir / filename, compression="zstd"
        )

    manifest = {
        "model": args.model,
        "model_label": args.model_label,
        "code_revision": args.code_revision,
        "training_dataset": TRAIN_DATASET_REPO,
        "training_dataset_revision": TRAIN_DATASET_REVISION,
        "validation_rows": args.validation_rows,
        "ablation_rows": args.ablation_rows,
        "attention_rows": args.attention_rows,
        "lm_ablation_modes": list(LM_ABLATIONS),
        "vep_context_modes": list(VEP_MODES),
        "n_bootstrap": args.n_bootstrap,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
