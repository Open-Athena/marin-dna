"""Pinned Mendelian TSS-proximal CLM evaluation for issue #515."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from pyfaidx import Fasta
from sklearn.metrics import average_precision_score
from transformers import PreTrainedTokenizerBase

from glm_experiments.exp515.config import (
    EVAL_DATASET,
    EVAL_REVISION,
    NUCLEOTIDE_LENGTH,
    REFERENCE_DATASET,
    REFERENCE_FASTA,
    REFERENCE_REVISION,
    SEQUENCE_LENGTH,
)
from glm_experiments.models.components.lm import HFCLM

REFERENCE_ASSETS = (
    REFERENCE_FASTA,
    f"{REFERENCE_FASTA}.fai",
    f"{REFERENCE_FASTA}.gzi",
    "SHA256SUMS",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_reference(destination: Path) -> Path:
    """Download and verify the pinned Ensembl-115 GRCh38 BGZF bundle."""

    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        filename: Path(
            hf_hub_download(
                repo_id=REFERENCE_DATASET,
                filename=filename,
                repo_type="dataset",
                revision=REFERENCE_REVISION,
                local_dir=destination,
            )
        )
        for filename in REFERENCE_ASSETS
    }
    expected = {}
    for line in paths["SHA256SUMS"].read_text(encoding="utf-8").splitlines():
        if line.strip():
            checksum, filename = line.split(maxsplit=1)
            expected[filename.lstrip("* ")] = checksum
    for filename in REFERENCE_ASSETS[:-1]:
        if expected.get(filename) != _sha256(paths[filename]):
            raise ValueError(f"reference checksum mismatch for {filename}")
    return paths[REFERENCE_FASTA]


def load_promoter_frame(cache_dir: Path) -> pd.DataFrame:
    """Load the newer pinned Mendelian endpoint and attach 255-bp windows."""

    cached = cache_dir / "mendelian_tss_proximal_255.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    dataset_path = hf_hub_download(
        repo_id=EVAL_DATASET,
        filename="train.parquet",
        repo_type="dataset",
        revision=EVAL_REVISION,
    )
    frame = pd.read_parquet(dataset_path)
    required = {"chrom", "pos", "ref", "alt", "label", "subset", "match_group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Mendelian endpoint lacks columns {sorted(missing)}")
    mature_groups = set(
        frame.loc[frame["subset"] == "mature_miRNA_variant", "match_group"]
    )
    if mature_groups:
        frame = frame.loc[~frame["match_group"].isin(mature_groups)]
    frame = frame.loc[frame["subset"] == "tss_proximal"].copy()
    frame["chrom"] = frame["chrom"].astype(str)
    frame["ref"] = frame["ref"].str.upper()
    frame["alt"] = frame["alt"].str.upper()
    if frame.empty or frame["label"].nunique() != 2:
        raise ValueError("Mendelian TSS-proximal endpoint is empty or one-class")
    if (
        not frame["ref"].isin(list("ACGT")).all()
        or not frame["alt"].isin(list("ACGT")).all()
    ):
        raise ValueError("Mendelian TSS-proximal endpoint contains non-SNV alleles")

    fasta_path = download_reference(cache_dir / "reference")
    center = NUCLEOTIDE_LENGTH // 2
    sequences: list[str] = []
    with Fasta(fasta_path, as_raw=True, rebuild=False) as genome:
        names = set(genome.keys())
        for row in frame.itertuples(index=False):
            center_zero_based = int(row.pos) - 1
            start = center_zero_based - center
            end = start + NUCLEOTIDE_LENGTH
            if row.chrom not in names or start < 0 or end > len(genome[row.chrom]):
                raise ValueError(f"variant window is outside {row.chrom}:{start}-{end}")
            sequence = str(genome[row.chrom][start:end]).upper()
            if len(sequence) != NUCLEOTIDE_LENGTH:
                raise ValueError(f"short reference window at {row.chrom}:{start}-{end}")
            if sequence[center] != row.ref:
                raise ValueError(
                    f"GRCh38 mismatch at {row.chrom}:{row.pos}: "
                    f"dataset={row.ref}, fasta={sequence[center]}"
                )
            sequences.append(sequence)
    frame["sequence"] = sequences
    cached.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cached, index=False)
    return frame


def _sequence_log_probability(
    model: HFCLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    logits = model.get_logits(input_ids, attention_mask=attention_mask)
    log_probabilities = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    token_log_probabilities = log_probabilities.gather(
        2,
        input_ids[:, 1:].unsqueeze(-1),
    ).squeeze(-1)
    return (token_log_probabilities * attention_mask[:, 1:]).sum(dim=-1)


def _prepare_model_for_evaluation(model: HFCLM) -> torch.device:
    """Place a Lightning-detached model on the best available eval device."""

    current_device = next(model.parameters()).device
    device = torch.device("cuda") if torch.cuda.is_available() else current_device
    model.to(device)
    model.eval()
    return device


@torch.inference_mode()
def evaluate_promoter_auprc(
    model: HFCLM,
    tokenizer: PreTrainedTokenizerBase,
    *,
    frame: pd.DataFrame,
    output_path: Path,
    batch_size: int,
    checkpoint_name: str,
) -> dict[str, Any]:
    """Score the exact ref/alt endpoint and write per-variant CSV evidence."""

    if batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    device = _prepare_model_for_evaluation(model)
    center = NUCLEOTIDE_LENGTH // 2
    scores = np.empty(len(frame), dtype=np.float32)
    started = time.time()
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        references = frame["sequence"].iloc[start:stop].tolist()
        alternates = [
            sequence[:center] + alt + sequence[center + 1 :]
            for sequence, alt in zip(
                references,
                frame["alt"].iloc[start:stop],
                strict=True,
            )
        ]
        encoded = tokenizer(
            [*references, *alternates],
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if input_ids.shape[1] != SEQUENCE_LENGTH:
            raise ValueError(f"evaluation produced {input_ids.shape[1]} tokens")
        with (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        ):
            sequence_scores = _sequence_log_probability(
                model,
                input_ids,
                attention_mask,
            )
        reference_scores, alternate_scores = sequence_scores.chunk(2)
        scores[start:stop] = (alternate_scores - reference_scores).cpu().numpy()
    if not np.isfinite(scores).all():
        raise RuntimeError("Mendelian promoter evaluation produced non-finite scores")
    transformed = -scores
    auprc = float(average_precision_score(frame["label"].astype(int), transformed))
    result_frame = frame[["chrom", "pos", "ref", "alt", "label", "match_group"]].copy()
    result_frame["raw_alt_minus_ref_llr"] = scores
    result_frame["minus_llr_score"] = transformed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output_path, index=False)
    summary = {
        "checkpoint": checkpoint_name,
        "dataset": EVAL_DATASET,
        "dataset_revision": EVAL_REVISION,
        "rows": len(frame),
        "positives": int(frame["label"].sum()),
        "auprc": auprc,
        "elapsed_seconds": time.time() - started,
        "scores_csv": str(output_path),
        "scores_sha256": _sha256(output_path),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
