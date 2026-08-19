"""Shared one-billion-parameter Qwen3 AFDB smoke-training recipe."""

import os
from dataclasses import replace

from fray.types import ResourceConfig
from levanter.callbacks.watch import WatchConfig
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.layers.attention import AttentionBackend
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint

TOKENIZER = "timodonnell/contacts-v1-tokenizer@5d68a24a899f"
VOCAB_SIZE = 2_845
SEQ_LEN = 8_192

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


def env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"missing required environment variable {name}")
    return value


def existing_afdb_cache(
    *,
    name: str,
    version: str,
    source: str,
) -> ArtifactStep[TokenizedCache]:
    return ArtifactStep[TokenizedCache].adopt(
        name,
        version,
        source,
        kind=TokenizedCache,
        config={
            "tokenizer": TOKENIZER,
            "format": {"text_key": "document"},
            "tags": ["protein", "contacts-v1", "afdb", "pretokenized"],
        },
    )


def build_smoke_run(
    *,
    platform: str,
    train_cache: ArtifactStep[TokenizedCache],
    validation_cache: ArtifactStep[TokenizedCache],
    resources: ResourceConfig,
    attention_backend: AttentionBackend | None,
) -> ArtifactStep[LevanterCheckpoint]:
    steps = env_int("EXP472_STEPS", 2)
    batch_size = env_int("EXP472_BATCH_SIZE", 8)
    suffix = os.environ.get("EXP472_RUN_SUFFIX", "v1").strip()
    run_id = f"exp472-plantcad2-afdb-{platform}-smoke"
    if suffix:
        run_id = f"{run_id}-{suffix}"

    wandb_entity = required_env("WANDB_ENTITY")
    wandb_project = required_env("WANDB_PROJECT")
    env_vars = {
        "HF_TOKEN": required_env("HF_TOKEN"),
        "MARIN_PREFIX": required_env("MARIN_PREFIX"),
        "WANDB_API_KEY": required_env("WANDB_API_KEY"),
        "WANDB_ENTITY": wandb_entity,
        "WANDB_PROJECT": wandb_project,
    }

    step = train_lm(
        name=f"checkpoints/{run_id}",
        run_id=run_id,
        model=replace(MODEL_CONFIG, attn_backend=attention_backend),
        optimizer=AdamConfig(
            learning_rate=1e-3,
            weight_decay=0.2,
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
        steps_per_eval=steps,
        wandb_project=wandb_project,
        wandb_group="exp472-plantcad2-baseline-smoke",
        tags=[
            "marin-dna",
            "exp472",
            "plantcad2-baseline",
            "afdb",
            "smoke",
            platform,
            f"params={MODEL_PARAMS}",
        ],
        env_vars=env_vars,
    )
    base_build_config = step.build_config

    def build_config(ctx):
        pod = base_build_config(ctx)
        trainer = replace(
            pod.train_config.trainer,
            max_eval_batches=1,
            per_device_parallelism=1,
            per_device_eval_parallelism=1,
            watch=DISABLED_WANDB_WATCH,
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
