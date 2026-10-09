"""Explicit exp586 choices for the d768 Hero-ladder model."""

import math
from typing import Literal

from experiments.grug.moe_hero_ep.model import GrugModelConfig, QbEstimator

SEQ_LEN = 8192
GLOBAL_BATCH_SIZE = 64
TOKENS_PER_UPDATE = GLOBAL_BATCH_SIZE * SEQ_LEN
EPOCH_TOKENS = 21_615_869_952
TRAINING_EPOCHS = 10
TRAINING_TOKENS = EPOCH_TOKENS * TRAINING_EPOCHS
TOTAL_UPDATES = TRAINING_TOKENS // TOKENS_PER_UPDATE
# Hero warms up for floor(1% of the run), then immediately decays linearly.
# The first decay-phase update uses the exact peak LR, so save after it.
WARMUP_UPDATES = int(0.01 * TOTAL_UPDATES)
PEAK_CHECKPOINT_UPDATE = WARMUP_UPDATES + 1

SOURCE_CHECKPOINT = (
    "s3://marin-us-east-02a/marin/grug/rav-ladder-d768-v2/"
    "2026.08.18/checkpoints/step-11420"
)
SOURCE_METADATA_DIGEST = (
    "01cbe71d30793c7a66dd7fe1f54a50e1198490a58cc364a28b5cf27a26e76cf4"
)
SOURCE_METADATA_RAW_SHA256 = (
    "c8187f5170a01737744b5887ced2ec6007528814a5a6edf8c65056902b7ac1ba"
)


def d768_config(condition: Literal["pretrained", "scratch"]) -> GrugModelConfig:
    """Use the d768 ladder architecture with an 8K DNA context."""
    if condition not in ("pretrained", "scratch"):
        raise ValueError(f"Unknown condition: {condition}")
    hidden_dim = 768
    return GrugModelConfig(
        vocab_size=128_256 if condition == "pretrained" else 8,
        hidden_dim=hidden_dim,
        intermediate_dim=384,
        shared_expert_intermediate_dim=384,
        num_shared_experts=2,
        num_experts=384,
        num_experts_per_token=8,
        latent_dim=384,
        num_layers=8,
        num_heads=6,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=128,
        max_seq_len=SEQ_LEN,
        sliding_window=2048,
        global_every=4,
        capacity_factor=1.15,
        initializer_std=0.5 / math.sqrt(hidden_dim),
        qk_mult=1.3,
        sconv=True,
        attention_implementation="gpu_fa4_cute",
        moe_implementation="fixed_pooled_wave_all_to_all",
        pooled_transport_capacity_factor=1.15,
        num_expert_waves=3,
        qb_estimator=QbEstimator.HIST,
        qb_hist_bins=10_000,
        report_capacity_overflow=True,
        rope_fused=True,
    )
