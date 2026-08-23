"""Pinned constants and registered arms for issue #515."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from glm_experiments.models.components.selection import SelectorMode

SOURCE_CHECKPOINT = (
    "gs://marin-us-east5/checkpoints/"
    "dna-exp232-zoonomia-v1-0p25b-v4_tss_region_and_utr5-v0.1-rerun-d08452/"
    "hf/step-2000"
)
TRAIN_DATASET = "marin-dna/zoonomia-v1-v4_tss_region_and_utr5"
TRAIN_REVISION = "80b44bf6129d6ec7988f8cf1b706e4b1464ec9dc"
TRAIN_TEXT_KEY = "sequence"
TRAIN_SPECIES_KEY = "species"
EVAL_DATASET = "marin-dna/evals_mendelian_traits"
EVAL_REVISION = "4aed58e50c5dea0b878a665007af2ef9e5108e9f"
REFERENCE_DATASET = "marin-dna/human-genome"
REFERENCE_REVISION = "11b9433582981bb929af333bc6422f10a8fd71b4"
REFERENCE_FASTA = "Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
GLM_EXPERIMENTS_REVISION = "b46cf87c2926201473797f9b00c13e1781c16403"

NUCLEOTIDE_LENGTH = 255
SEQUENCE_LENGTH = 256
EFFECTIVE_BATCH_SIZE = 2048
BRIDGE_STEPS = 100
CANARY_STEPS = 20
MAX_CONTINUATION_STEPS = 1000
PEAK_LEARNING_RATE = 1e-3
END_LEARNING_RATE = 1e-4
BETAS = (0.9, 0.98)
EPSILON = 1e-9
WEIGHT_DECAY = 0.1
GRADIENT_CLIP_VALUE = 1.0
SEED = 515
SHUFFLE_BUFFER_SIZE = 10_000
GPU_COMPUTE_CAP_USD = 18.0
ALL_IN_CAP_USD = 20.0
GH200_PRICE_PER_HOUR_USD = 2.29
RUNTIME_MARGIN = 1.25
ISSUE_S3_PREFIX = "s3://oa-bolinas/issues/515/online-loss-selection/v1"

Stage = Literal["bridge", "continuation"]


@dataclass(frozen=True)
class Arm:
    """One registered continuation arm."""

    name: str
    selector_mode: SelectorMode
    selector_ratio: float


ARMS = (
    Arm("uniform-100", "uniform", 1.0),
    Arm("random-50", "random", 0.5),
    Arm("student-low-50", "student_low", 0.5),
    Arm("student-middle-50", "student_middle", 0.5),
    Arm("student-high-50", "student_high", 0.5),
)


def continuation_midpoint(steps: int) -> int:
    """Return the registered continuation midpoint in optimizer steps."""

    if steps <= 0:
        raise ValueError("continuation steps must be positive")
    return BRIDGE_STEPS + steps // 2


def continuation_endpoint(steps: int) -> int:
    """Return the global optimizer step at a continuation endpoint."""

    if steps <= 0:
        raise ValueError("continuation steps must be positive")
    return BRIDGE_STEPS + steps
