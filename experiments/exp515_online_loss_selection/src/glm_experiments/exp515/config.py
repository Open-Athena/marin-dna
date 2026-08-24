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
REFSEQ_TRAIN_DATASET = "marin-dna/genomes-v5-genome_set-animals-intervals-v1_255_128"
REFSEQ_TRAIN_REVISION = "d93209847b02a0c9be5c03591a0a5e56ee09c35d"
REFSEQ_TRAIN_TEXT_KEY = "seq"
EXP58_STUDENT_CHECKPOINT = (
    "gs://marin-dna-us-central1/checkpoints/exp58-animals-r01-1e3682/hf/step-1000"
)
EXP58_TEACHER_CHECKPOINT = (
    "gs://marin-dna-us-central1/checkpoints/exp58-animals-r01-1e3682/hf/step-16999"
)
EXP58_TRAIN_DATASET = "marin-dna/genomes-v4-genome_set-animals-intervals-v5_256_128"
EXP58_TRAIN_REVISION = "04d374450a0f78f0ab5e17a8bc7b7c4baeb8295c"
EXP58_TRAIN_TEXT_KEY = "seq"
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
REFSEQ_INITIAL_CONTINUATION_STEPS = 500
REFSEQ_MIDPOINT_STEPS = 250
CDS_NUCLEOTIDE_LENGTH = 256
CDS_SEQUENCE_LENGTH = 256
CDS_GATE_CONTINUATION_STEPS = 100
PEAK_LEARNING_RATE = 1e-3
END_LEARNING_RATE = 1e-4
BETAS = (0.9, 0.98)
EPSILON = 1e-9
WEIGHT_DECAY = 0.1
GRADIENT_CLIP_VALUE = 1.0
SEED = 515
SHUFFLE_BUFFER_SIZE = 10_000
ACCELERATOR = "A100:1"
GPU_COMPUTE_CAP_USD = 48.0
ALL_IN_CAP_USD = 50.0
GPU_PRICE_PER_HOUR_USD = 1.99
RUNTIME_MARGIN = 1.25
ISSUE_S3_PREFIX = "s3://oa-bolinas/issues/515/online-loss-selection/v1"

Stage = Literal["bridge", "continuation"]
ObjectiveKind = Literal["hard_ce", "teacher_kl", "teacher_low"]


@dataclass(frozen=True)
class Arm:
    """One registered continuation arm."""

    name: str
    selector_mode: SelectorMode
    selector_ratio: float
    objective_kind: ObjectiveKind = "hard_ce"


ARMS = (
    Arm("uniform-100", "uniform", 1.0),
    Arm("random-50", "random", 0.5),
    Arm("student-low-50", "student_low", 0.5),
    Arm("student-middle-50", "student_middle", 0.5),
    Arm("student-high-50", "student_high", 0.5),
)

CDS_ARMS = (
    *ARMS,
    Arm("teacher-kl-full", "uniform", 1.0, "teacher_kl"),
    Arm("teacher-low-50", "uniform", 0.5, "teacher_low"),
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
