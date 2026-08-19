"""Pinned experiment constants and DNA-calibrated optimizer scaling."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

MODEL_ID = "marin-dna/marin-dna-exp135-m5.1"
MODEL_REVISION = "a73a5dcfb3d64b8941e7e7596c6e88ef77db3e7a"
MASK_TOKEN = "[MASK]"

SEQUENCE_LENGTH = 256
NUCLEOTIDE_LENGTH = 255
TRAIN_STEPS = 1_000
WARMUP_STEPS = 100
COOLDOWN_START_STEP = 800
CHECKPOINT_INTERVAL = 100

WANDB_PROJECT = "marin"
WANDB_GROUP = "dna-exp479"
EXPERIMENT_TAGS = ("MNTP-479", "issue-479", "bidirectional", "m5.1")

LAMBDA_GH200_PRICE_PER_HOUR_USD = 2.29
BUDGET_USD = 50.0


@dataclass(frozen=True)
class DataComponent:
    """One pinned m5.1 training component and its matched validation probe."""

    name: str
    train_repo: str
    train_revision: str
    train_text_key: str
    validation_repo: str
    validation_revision: str
    validation_text_key: str


DATA_COMPONENTS = (
    DataComponent(
        name="cds",
        train_repo="marin-dna/genomes-v5-genome_set-animals-intervals-v5_255_128",
        train_revision="ffe3e78c99868077c65ad6568e1445d80e480794",
        train_text_key="seq",
        validation_repo="marin-dna/genomes-v5-validation-intervals-v5_255_255",
        validation_revision="daff592f213aaa1cab1711d477a79ff6b1bc4ef4",
        validation_text_key="seq",
    ),
    DataComponent(
        name="upstream",
        train_repo="marin-dna/genomes-v5-genome_set-animals-intervals-v1_255_128",
        train_revision="d93209847b02a0c9be5c03591a0a5e56ee09c35d",
        train_text_key="seq",
        validation_repo="marin-dna/genomes-v5-validation-intervals-v1_255_255",
        validation_revision="a761bc0b663a9827303f3112e4667d53d5326fac",
        validation_text_key="seq",
    ),
    DataComponent(
        name="downstream",
        train_repo="marin-dna/genomes-v5-genome_set-animals-intervals-v15_255_128",
        train_revision="b009afaab756937d75b8da3b1271ad8f0cec0b4d",
        train_text_key="seq",
        validation_repo="marin-dna/genomes-v5-validation-intervals-v15_255_255",
        validation_revision="d7b27eecd68453934ebb3e7e6e78d5401789faa5",
        validation_text_key="seq",
    ),
    DataComponent(
        name="enhancer",
        train_repo="marin-dna/zoonomia-v1-v3_ccre_non_promoter",
        train_revision="862485aa18eed53a53e693ba4c2eb45e0afc5087",
        train_text_key="sequence",
        validation_repo="marin-dna/zoonomia-v1-val_enhancer",
        validation_revision="d40d1e067b2a56ac812af122de029eb79cab1106",
        validation_text_key="sequence",
    ),
    DataComponent(
        name="ncrna",
        train_repo="marin-dna/zoonomia-v1-v3_ncrna_exon",
        train_revision="3e48d9ae7c604b99ccfc8bd07e391b960c1ea21a",
        train_text_key="sequence",
        validation_repo="marin-dna/zoonomia-v1-val_ncrna",
        validation_revision="76a18c1bbf07ac9bd064722431bbdab894b9e6c6",
        validation_text_key="sequence",
    ),
)


@dataclass(frozen=True)
class OptimizerHyperparameters:
    """AdamH/Adam values transferred from the pinned m5.1 scaling heuristic."""

    adamh_learning_rate: float
    adam_learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    max_grad_norm: float
    initializer_range: float
    model_tokens: int
    nucleotide_bases: int

    def to_dict(self) -> dict[str, float | int]:
        """Return JSON-serializable values."""

        return asdict(self)


def optimizer_hyperparameters(
    batch_size: int, train_steps: int = TRAIN_STEPS
) -> OptimizerHyperparameters:
    """Apply the DNA-calibrated CompletedAdamH heuristic to this pilot exposure."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if train_steps <= 0:
        raise ValueError(f"train_steps must be positive, got {train_steps}")

    reference_batch_size = 16_384
    reference_tokens = 2_500_000_000
    lr_base = 0.015566099981405093
    adam_lr_base = 0.02989514059663958
    epsilon_base = 1e-15
    beta1 = 0.6675603345321236
    beta2_base = 0.9067269880630742
    model_tokens = train_steps * batch_size * SEQUENCE_LENGTH
    nucleotide_bases = train_steps * batch_size * NUCLEOTIDE_LENGTH

    batch_ratio = batch_size / reference_batch_size
    token_ratio = reference_tokens / model_tokens
    scaling_ratio = (batch_size * reference_tokens) / (reference_batch_size * model_tokens)

    adamh_lr = min(0.03, lr_base * math.sqrt(batch_ratio) * token_ratio**0.3)
    adam_lr = min(0.03, adam_lr_base * math.sqrt(scaling_ratio))
    epsilon = epsilon_base * math.sqrt(1.0 / scaling_ratio)
    beta2 = max(0.5, min(0.9999, beta2_base**batch_ratio))

    return OptimizerHyperparameters(
        adamh_learning_rate=adamh_lr,
        adam_learning_rate=adam_lr,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        max_grad_norm=0.9951880136348765,
        initializer_range=0.02,
        model_tokens=model_tokens,
        nucleotide_bases=nucleotide_bases,
    )


def wsd_multiplier(
    step: int,
    *,
    warmup_steps: int = WARMUP_STEPS,
    cooldown_start_step: int = COOLDOWN_START_STEP,
    total_steps: int = TRAIN_STEPS,
) -> float:
    """Return the registered linear-warmup/stable/linear-cooldown multiplier."""

    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if not 0 < warmup_steps <= cooldown_start_step < total_steps:
        raise ValueError(
            "expected 0 < warmup_steps <= cooldown_start_step < total_steps, got "
            f"{warmup_steps}, {cooldown_start_step}, {total_steps}"
        )
    if step <= warmup_steps:
        return step / warmup_steps
    if step <= cooldown_start_step:
        return 1.0
    if step >= total_steps:
        return 0.0
    return (total_steps - step) / (total_steps - cooldown_start_step)
