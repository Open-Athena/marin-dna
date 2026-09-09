"""Frozen geometry and explicit allocated-token AdamH transfer.

The upstream reference is 64 documents of 4096 tokens over 2.5B tokens:
https://github.com/marin-community/marin/blob/efe79892065589b154d969effd49eee3bd286284/experiments/references/completed_adamh.py

Normalizing batch size to reference-length documents extends that fixed-context
heuristic across context lengths. This is a documented experimental choice,
not a claim that Complete(d) proves context-length transfer. Allocated positions
include padding; masked padding does not contribute gradient information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MODEL_TOKENS = 10_240
WINDOW_BP = 255
BATCH_DOCUMENTS = 200
TRAIN_UPDATES = 100_000
BATCH_TOKENS = BATCH_DOCUMENTS * MODEL_TOKENS
TOTAL_TOKENS = TRAIN_UPDATES * BATCH_TOKENS
CHECKPOINT_EVERY = 10_000
TRAIN_SEED = 0
DATA_SEED = 42
REGIONS = ("cds", "tss_utr5", "utr3", "ncrna", "enhancer")
MODEL_DIMENSIONS = {
    "hidden_dim": 640,
    "intermediate_dim": 2560,
    "num_layers": 7,
    "num_heads": 5,
    "num_kv_heads": 5,
    "head_dim": 128,
}
PAD_ID = 0
N_ID = 1
BOS_ID = 2
SEQ_ID = 3
BASE_IDS = {"A": 4, "C": 5, "G": 6, "T": 7, "N": N_ID}


def encode_document(sequence: str) -> dict[str, list[int] | list[float]]:
    """Emit one complete fixed-shape example, with next-token-aligned weights."""
    segments = sequence.split("[SEQ]")
    if not 1 <= len(segments) <= 40:
        raise ValueError("RAG documents contain one through forty available species")
    ids = [BOS_ID]
    for index, segment in enumerate(segments):
        if len(segment) != WINDOW_BP or set(segment.upper()) - set(BASE_IDS):
            raise ValueError("each RAG segment must contain 255 A/C/G/T/N bases")
        if index:
            ids.append(SEQ_ID)
        ids.extend(BASE_IDS[base] for base in segment.upper())
    length = len(ids)
    if length != 256 * len(segments):
        raise ValueError("unexpected RAG token geometry")
    # The final real token predicts padding (or has no next token at full length).
    loss_weights = [1.0] * (length - 1) + [0.0] * (MODEL_TOKENS - length + 1)
    ids.extend([PAD_ID] * (MODEL_TOKENS - length))
    return {"input_ids": ids, "loss_weight": loss_weights}


@dataclass(frozen=True)
class OptimizerRecipe:
    learning_rate: float
    adam_lr: float
    epsilon: float
    beta2: float
    beta1: float = 0.9
    max_grad_norm: float = 0.1
    min_lr_ratio: float = 0.0
    warmup: float = 0.1
    decay: float = 0.2
    lr_schedule: str = "linear"


def resolve_optimizer(
    *, batch_tokens: int = BATCH_TOKENS, total_tokens: int = TOTAL_TOKENS
) -> OptimizerRecipe:
    """Transfer the adapted Complete(d) heuristic with explicit token units."""
    if batch_tokens <= 0 or total_tokens <= 0:
        raise ValueError("optimizer token budgets must be positive")
    batch_ratio = batch_tokens / (64 * 4096)
    duration_ratio = 2.5e9 / total_tokens
    ratio = batch_ratio * duration_ratio
    return OptimizerRecipe(
        learning_rate=min(0.01, 0.00630 * math.sqrt(batch_ratio) * duration_ratio**0.3),
        adam_lr=min(0.01, 0.000656 * math.sqrt(ratio)),
        epsilon=1.85e-8 / math.sqrt(ratio),
        beta2=max(0.9, min(0.9999, 0.9999**batch_ratio)),
    )
