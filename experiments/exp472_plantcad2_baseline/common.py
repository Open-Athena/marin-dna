"""Shared one-billion-parameter Qwen3 PlantCAD2 sweep recipe."""

import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Self

from fray.types import ResourceConfig
from levanter.callbacks.watch import WatchConfig
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.layers.attention import AttentionBackend
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint

TOKENIZER = "kuleshov-group/PlantCAD2-Small-l24-d0768"
VOCAB_SIZE = 7
SEQ_LEN = 8_192
DEFAULT_GLOBAL_BATCH_SIZE = 128
DATASET_REVISION = "4a444fff5520b992aa978d92a5af509a81977098"
CACHE_VERSION = "2026.08.19"
TOKENIZED_CACHE_RELATIVE = "MarinDNA/tokenized/plantcad/Angiosperm_65_genomes_8192bp"
EXPERIMENT_RELATIVE = "MarinDNA/exp472_plantcad2_baseline"
TRAIN_CACHE_NAME = "inputs/plantcad-angiosperm-train-path-only"
VALIDATION_CACHE_NAME = "inputs/plantcad-angiosperm-validation-path-only"

LEARNING_RATES = (3e-4, 1e-3, 3e-3)
WEIGHT_DECAYS = (0.1, 0.2, 0.8)

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
)
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
    steps = env_int("EXP472_STEPS", 2)
    suffix = os.environ.get("EXP472_RUN_SUFFIX", "v1").strip()
    checkpoint_id = f"exp472-plantcad2-angiosperm-{point.key}"
    if suffix:
        checkpoint_id = f"{checkpoint_id}-{suffix}"
    run_id = checkpoint_id
    if wandb_run_suffix:
        run_id = f"{run_id}-{wandb_run_suffix}"

    wandb_entity = required_env("WANDB_ENTITY")
    wandb_project = required_env("WANDB_PROJECT")
    env_vars = {
        "HF_TOKEN": required_env("HF_TOKEN"),
        "MARIN_PREFIX": required_env("MARIN_PREFIX"),
        "WANDB_API_KEY": required_env("WANDB_API_KEY"),
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
            lr_schedule="linear",
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
        steps_per_eval=steps,
        wandb_project=wandb_project,
        wandb_group="exp472-plantcad2-baseline-sweep",
        tags=[
            "MarinDNA",
            "exp472",
            "plantcad2-baseline",
            "angiosperm-65-genomes",
            f"lr={point.learning_rate:g}",
            f"wd={point.weight_decay:g}",
            f"batch={batch_size}",
            f"steps={steps}",
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
            max_eval_batches=1,
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
        train_config = replace(
            pod.train_config,
            trainer=trainer,
            data=data,
            data_seed=0,
            hf_save_steps=steps + 1,
        )
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config)
