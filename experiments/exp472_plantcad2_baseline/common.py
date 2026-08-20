"""Shared one-billion-parameter Qwen3 PlantCAD2 sweep recipe."""

import os
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from datetime import timedelta
from typing import Self

import jax
import numpy as np
from fray.types import ResourceConfig
from haliax import Axis
from jaxtyping import PRNGKeyArray
from levanter.callbacks.watch import WatchConfig
from levanter.data.dataset import AsyncDataset
from levanter.data.text.datasets import BlockShuffleConfig, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.layers.attention import AttentionBackend
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.lm_model import LmExample
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from levanter.schedule import BatchSchedule
from marin.execution.lazy import ArtifactStep
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint

TOKENIZER = "kuleshov-group/PlantCAD2-Small-l24-d0768"
VOCAB_SIZE = 7
SEQ_LEN = 8_192
DEFAULT_GLOBAL_BATCH_SIZE = 128
DEFAULT_TRAIN_STEPS = 206_145  # 10 epochs: 2,638,656 examples * 10 / 128
TEMPORARY_CHECKPOINT_INTERVAL = timedelta(minutes=15)
PERMANENT_CHECKPOINT_COUNT = 10
EVALUATION_COUNT = 2 * PERMANENT_CHECKPOINT_COUNT
DATASET_REVISION = "4a444fff5520b992aa978d92a5af509a81977098"
CACHE_VERSION = "2026.08.19"
TOKENIZED_CACHE_RELATIVE = "MarinDNA/tokenized/plantcad/Angiosperm_65_genomes_8192bp"
EXPERIMENT_RELATIVE = "MarinDNA/exp472_plantcad2_baseline"
TRAIN_CACHE_NAME = "inputs/plantcad-angiosperm-train-path-only"
VALIDATION_CACHE_NAME = "inputs/plantcad-angiosperm-validation-path-only"
REVERSE_COMPLEMENT_PROBABILITY = 0.5
REVERSE_COMPLEMENT_SEED = 472
# [PAD], [MASK], and [UNK] are self-complementary; a<->t and c<->g.
REVERSE_COMPLEMENT_TOKEN_IDS = (0, 1, 2, 6, 5, 4, 3)

LEARNING_RATES = (1e-4, 2e-4, 5e-4, 1e-3)
WEIGHT_DECAYS = (0.1, 0.2, 0.8, 1.6)
SKIPPED_SWEEP_POINTS = frozenset({(1e-4, 0.1), (1e-3, 1.6)})

MODEL_CONFIG = Qwen3Config(
    max_seq_len=SEQ_LEN,
    hidden_dim=2_048,
    intermediate_dim=8_192,
    num_layers=16,
    num_heads=16,
    num_kv_heads=4,
    head_dim=128,
    rope=Llama3RotaryEmbeddingsConfig(),
    use_qk_norm=True,
)
MODEL_PARAMS = (
    int(MODEL_CONFIG.total_trainable_params(VOCAB_SIZE))
    + 2 * MODEL_CONFIG.num_layers * MODEL_CONFIG.actual_head_size
)

SHUFFLE = BlockShuffleConfig(
    io_block_size=256,
    window_blocks=512,
    perm_type="feistel",
)
DISABLED_WANDB_WATCH = WatchConfig(watch_targets=[], interval=0)


def reverse_complement_token_ids(token_ids: np.ndarray) -> np.ndarray:
    """Reverse-complement one PlantCAD2 token sequence."""
    if token_ids.ndim != 1:
        raise ValueError(f"expected one token sequence, got shape {token_ids.shape}")
    if not np.issubdtype(token_ids.dtype, np.integer):
        raise ValueError(f"expected integer token IDs, got dtype {token_ids.dtype}")
    if token_ids.size and (token_ids.min() < 0 or token_ids.max() >= VOCAB_SIZE):
        raise ValueError("token sequence contains an out-of-range PlantCAD2 token ID")

    complement = np.asarray(REVERSE_COMPLEMENT_TOKEN_IDS, dtype=token_ids.dtype)
    return complement[token_ids[::-1]]


def _augmentation_rng(seed: int, index: int) -> np.random.Generator:
    if index < 0:
        raise ValueError(f"dataset index must be nonnegative, got {index}")
    return np.random.default_rng(
        np.random.SeedSequence([seed, index & 0xFFFFFFFF, index >> 32])
    )


def reverse_complement_selected(*, seed: int, index: int, probability: float) -> bool:
    """Make a reproducible Bernoulli choice for one training-stream occurrence."""
    if index < 0:
        raise ValueError(f"dataset index must be nonnegative, got {index}")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"augmentation probability must be in [0, 1], got {probability}"
        )
    return probability >= 1.0 or (
        probability > 0.0 and _augmentation_rng(seed, index).random() < probability
    )


def _reverse_complement_lm_example(
    example: LmExample,
    *,
    seed: int,
    index: int,
    probability: float,
) -> LmExample:
    if not reverse_complement_selected(
        seed=seed,
        index=index,
        probability=probability,
    ):
        return example

    original = np.asarray(jax.device_get(example.tokens.array))
    augmented = reverse_complement_token_ids(original)
    token_array = jax.device_put(augmented, example.tokens.array.sharding)
    return replace(example, tokens=replace(example.tokens, array=token_array))


class ReverseComplementAugmentedDataset(AsyncDataset[LmExample]):
    """Apply reverse complements using the absolute training-stream index."""

    def __init__(
        self,
        dataset: AsyncDataset[LmExample],
        *,
        seed: int,
        probability: float,
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"augmentation probability must be in [0, 1], got {probability}"
            )
        self.dataset = dataset
        self.seed = seed
        self.probability = probability

    async def async_len(self) -> int:
        return await self.dataset.async_len()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def get_batch(self, indices: Sequence[int]) -> Sequence[LmExample]:
        examples = await self.dataset.get_batch(indices)
        return [
            _reverse_complement_lm_example(
                example,
                seed=self.seed,
                index=index,
                probability=self.probability,
            )
            for index, example in zip(indices, examples, strict=True)
        ]


def _validate_plantcad_tokenizer(data: LmDataConfig) -> None:
    tokenizer = data.the_tokenizer
    tokens = ("[PAD]", "[MASK]", "[UNK]", "a", "c", "g", "t")
    observed = tuple(tokenizer.convert_tokens_to_ids(list(tokens)))
    if (
        observed != tuple(range(VOCAB_SIZE))
        or len(tokenizer) != VOCAB_SIZE
        or tokenizer.bos_token_id is not None
        or tokenizer.eos_token_id is not None
    ):
        raise ValueError(
            "PlantCAD2 tokenizer contract changed: "
            f"{observed=}, vocab_size={len(tokenizer)}, "
            f"bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}"
        )


@dataclass(frozen=True)
class ReverseComplementDataConfig(LmDataConfig):
    augmentation_seed: int = REVERSE_COMPLEMENT_SEED
    augmentation_probability: float = REVERSE_COMPLEMENT_PROBABILITY

    def train_set(
        self,
        Pos: Axis,
        batch_schedule: BatchSchedule,
        *,
        key: PRNGKeyArray,
    ) -> AsyncDataset[LmExample]:
        _validate_plantcad_tokenizer(self)
        dataset = super().train_set(Pos, batch_schedule, key=key)
        return ReverseComplementAugmentedDataset(
            dataset,
            seed=self.augmentation_seed,
            probability=self.augmentation_probability,
        )


def augment_reverse_complements(data: LmDataConfig) -> LmDataConfig:
    values = {field.name: getattr(data, field.name) for field in fields(LmDataConfig)}
    return ReverseComplementDataConfig(**values)


@dataclass(frozen=True)
class SweepPoint:
    key: str
    learning_rate: float
    weight_decay: float


def _value_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


SWEEP_POINTS = tuple(
    SweepPoint(
        key=f"lr{_value_slug(learning_rate)}-wd{_value_slug(weight_decay)}",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    for learning_rate in LEARNING_RATES
    for weight_decay in WEIGHT_DECAYS
    if (learning_rate, weight_decay) not in SKIPPED_SWEEP_POINTS
)
assert len(SWEEP_POINTS) == 14
SWEEP_POINTS_BY_KEY = {point.key: point for point in SWEEP_POINTS}


class ExistingPlantCadCache(TokenizedCache):
    """Path-only view of the PlantCAD cache produced outside Marin's step graph."""

    @classmethod
    def raw_load(cls, source: str) -> Self:
        return cls(path=source)

    @property
    def cache_dir(self) -> str:
        return self.path

    @property
    def tokenizer(self) -> str:
        return TOKENIZER

    @property
    def format(self) -> TextLmDatasetFormat:
        return TextLmDatasetFormat(text_key="seq")

    @property
    def tags(self) -> list[str]:
        return ["dna", "plantcad", "angiosperm", "pretokenized"]


def env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def global_batch_size() -> int:
    return env_int("EXP472_GLOBAL_BATCH_SIZE", DEFAULT_GLOBAL_BATCH_SIZE)


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"missing required environment variable {name}")
    return value


def require_marin_prefix(expected: str) -> str:
    configured = required_env("MARIN_PREFIX").rstrip("/")
    if configured != expected.rstrip("/"):
        raise ValueError(
            f"MARIN_PREFIX must be exactly {expected!r}, got {configured!r}"
        )
    return configured


def parse_sweep_point() -> SweepPoint:
    key = required_env("TRIAL").strip().lower()
    try:
        return SWEEP_POINTS_BY_KEY[key]
    except KeyError as exc:
        choices = ", ".join(SWEEP_POINTS_BY_KEY)
        raise SystemExit(f"TRIAL must be one of: {choices}") from exc


def existing_plantcad_cache(
    *,
    name: str,
    version: str,
    source: str,
) -> ArtifactStep[TokenizedCache]:
    return ArtifactStep[TokenizedCache].adopt(
        name,
        version,
        source,
        kind=ExistingPlantCadCache,
        config={
            "tokenizer": TOKENIZER,
            "format": {"text_key": "seq"},
            "tags": ["dna", "plantcad", "angiosperm", "pretokenized"],
        },
    )


def build_sweep_run(
    *,
    point: SweepPoint,
    train_cache: ArtifactStep[TokenizedCache],
    validation_cache: ArtifactStep[TokenizedCache],
    resources: ResourceConfig,
    attention_backend: AttentionBackend | None,
    batch_size: int,
    tensor_parallelism: int,
    per_device_parallelism: int,
    runtime_tags: Sequence[str],
    wandb_run_suffix: str | None,
    expected_output_prefix: str,
) -> ArtifactStep[LevanterCheckpoint]:
    steps = env_int("EXP472_STEPS", DEFAULT_TRAIN_STEPS)
    permanent_checkpoint_every = max(
        1,
        (steps + PERMANENT_CHECKPOINT_COUNT - 1) // PERMANENT_CHECKPOINT_COUNT,
    )
    evaluation_every = max(
        1,
        (steps + EVALUATION_COUNT - 1) // EVALUATION_COUNT,
    )
    suffix = os.environ.get("EXP472_RUN_SUFFIX", "v2").strip()
    checkpoint_id = f"exp472-plantcad2-angiosperm-{point.key}"
    if suffix:
        checkpoint_id = f"{checkpoint_id}-{suffix}"
    run_id = checkpoint_id
    if wandb_run_suffix:
        run_id = f"{run_id}-{wandb_run_suffix}"

    wandb_entity = required_env("WANDB_ENTITY")
    wandb_project = required_env("WANDB_PROJECT")
    # TPU workers inherit these credentials from the Iris task environment.
    # Validate their presence without embedding either value in Marin's
    # fingerprinted artifact config.
    required_env("HF_TOKEN")
    required_env("WANDB_API_KEY")
    env_vars = {
        "MARIN_PREFIX": required_env("MARIN_PREFIX"),
        "WANDB_ENTITY": wandb_entity,
        "WANDB_PROJECT": wandb_project,
    }
    if mode := os.environ.get("WANDB_MODE"):
        env_vars["WANDB_MODE"] = mode
    step = train_lm(
        name=f"checkpoints/{checkpoint_id}",
        run_id=run_id,
        model=replace(MODEL_CONFIG, attn_backend=attention_backend),
        optimizer=AdamConfig(
            learning_rate=point.learning_rate,
            weight_decay=point.weight_decay,
            warmup=0.1,
            decay=0.2,
            min_lr_ratio=0.0,
            lr_schedule="linear",
            skip_bad_steps=True,
        ),
        datasets={train_cache: 1.0},
        validation=[validation_cache],
        init_from=None,
        batch_size=batch_size,
        seq_len=SEQ_LEN,
        num_train_steps=steps,
        z_loss_weight=None,
        evals=None,
        resources=resources,
        tensor_parallel_size=tensor_parallelism,
        steps_per_eval=evaluation_every,
        wandb_project=wandb_project,
        wandb_group="exp472-plantcad2-baseline-sweep",
        tags=[
            "MarinDNA",
            "exp472",
            "plantcad2-baseline",
            "angiosperm-65-genomes",
            "augmentation=reverse-complement-p0.5",
            f"lr={point.learning_rate:g}",
            f"wd={point.weight_decay:g}",
            f"batch={batch_size}",
            f"steps={steps}",
            f"eval_every={evaluation_every}",
            f"eval_target_count={EVALUATION_COUNT}",
            f"params={MODEL_PARAMS}",
        ],
        env_vars=env_vars,
    )
    base_build_config = step.build_config

    def build_config(ctx):
        if not ctx.is_fingerprint and ctx.prefix.rstrip(
            "/"
        ) != expected_output_prefix.rstrip("/"):
            raise ValueError(
                f"execution prefix {ctx.prefix!r} must be exactly "
                f"{expected_output_prefix!r}"
            )
        pod = base_build_config(ctx)
        trainer = replace(
            pod.train_config.trainer,
            max_eval_batches=None,
            tracker=replace(
                pod.train_config.trainer.tracker,
                entity=wandb_entity,
            ),
            checkpointer=replace(
                pod.train_config.trainer.checkpointer,
                save_interval=TEMPORARY_CHECKPOINT_INTERVAL,
                keep=[{"every": permanent_checkpoint_every}],
            ),
            watch=DISABLED_WANDB_WATCH,
        )
        if not ctx.is_fingerprint:
            trainer = replace(
                trainer,
                per_device_parallelism=per_device_parallelism,
                per_device_eval_parallelism=per_device_parallelism,
                tracker=replace(
                    trainer.tracker,
                    tags=[*trainer.tracker.tags, *runtime_tags],
                ),
            )
        data = replace(
            pod.train_config.data,
            auto_build_caches=False,
            shuffle=SHUFFLE,
            components={
                name: replace(component, pack=True)
                for name, component in pod.train_config.data.components.items()
            },
            block_cross_document_attention=True,
        )
        data = augment_reverse_complements(data)
        train_config = replace(
            pod.train_config,
            trainer=trainer,
            data=data,
            data_seed=0,
            hf_save_steps=steps + 1,
        )
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config)
