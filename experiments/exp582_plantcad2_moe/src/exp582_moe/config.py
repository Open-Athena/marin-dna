"""Explicit exp582 choices on the existing Hero model implementation."""

import math
from typing import Literal

from experiments.grug.moe_hero_ep.model import GrugModelConfig, QbEstimator

STAGE_TOKENS = 216_158_699_520
HORIZON_TOKENS = 562_022_055_936
CONTINUATION_TOKENS = 194_542_829_568
# Preserve the original horizon as the optimizer heuristic's reference budget.
TOKENS_PER_UPDATE = 64 * 8192
WARMUP_UPDATES = (HORIZON_TOKENS + 100 * TOKENS_PER_UPDATE - 1) // (
    100 * TOKENS_PER_UPDATE
)
SHORT_STAGE_TOKENS = 2 * WARMUP_UPDATES * TOKENS_PER_UPDATE
EPOCH_TOKENS = STAGE_TOKENS // 10
SOURCE_CHECKPOINT = "s3://marin-us-east-02a/marin/grug/rav-ladder-d1536/2026.08.18/checkpoints/step-15128"
SOURCE_METADATA_DIGEST = (
    "7198833f20fd9d1f442e6d0d9ef9738e634d674ff225ed276e7b56d71cf60227"
)
SOURCE_METADATA_RAW_SHA256 = (
    "128b0e2779b943fbef139132d4add4961e9e12cda3358b65951169940aa556fe"
)


def d1536_config(condition: Literal["pretrained", "random"]) -> GrugModelConfig:
    """Preserve the ladder architecture and select its existing H100 kernels."""
    if condition not in ("pretrained", "random"):
        raise ValueError(f"Unknown condition: {condition}")
    return GrugModelConfig(
        vocab_size=128_256 if condition == "pretrained" else 8,
        hidden_dim=1536,
        intermediate_dim=768,
        shared_expert_intermediate_dim=768,
        num_shared_experts=2,
        num_experts=384,
        num_experts_per_token=8,
        latent_dim=768,
        num_layers=16,
        num_heads=12,
        num_kv_heads=3,
        local_kv_heads=3,
        global_kv_heads=1,
        head_dim=128,
        max_seq_len=8192,
        sliding_window=2048,
        global_every=4,
        capacity_factor=1.15,
        initializer_std=0.5 / math.sqrt(1536),
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
