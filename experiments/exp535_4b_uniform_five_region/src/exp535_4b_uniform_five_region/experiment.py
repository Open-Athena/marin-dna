"""Launch the issue #535 4B full-state five-region continuation."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal

import click
import jmp
from fray.types import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.adaptor import AdaptorConfig, AdaptorExportConfig, NoAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import HFCheckpointConverter, save_hf_checkpoint_callback
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.llama import LlamaConfig
from levanter.optim.adamh import AdamHConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig, run_levanter_train_lm
from rigging.filesystem.storage_path import prefix_join

from exp535_4b_uniform_five_region.formats import DNALmDatasetFormat

MARIN_COMMIT = "53b5b33041f742c7f4991223b0085e41ece4c458"
ISSUE = 535
TOKENIZER_PATH = "tokenizer"
TOKENIZER_SOURCE = "bolinas-dna/tokenizer-char-bos"
TOKENIZER_SOURCE_REVISION = "a73e9d9ee636f722b4c378703c9e2997857809b2"
TOKENIZER_SHA256 = {
    "special_tokens_map.json": "02b7b977703736f58dd672a20b0e6159fa10bfc41c9ca721eba0402552e35f2d",
    "tokenizer.json": "d066e668d7ba6ed48640b7a0ad45b5ae05d5dbd612b2d7f91fae1e2473fc93e9",
    "tokenizer_config.json": "4e814edcdb1cb8f408a2cdf951e9be4093bbcd3d12e58883f56aa06eb39fc8c7",
}

PARENT_RUN_ID = "dna-bolinas-scaling-v0.5-h2944-p4B-fa02c3"
PARENT_STEP = 215_573
PARENT_CHECKPOINT = (
    "gs://marin-us-east5/checkpoints/"
    f"{PARENT_RUN_ID}/checkpoints/step-{PARENT_STEP}/"
)
SEQUENCE_LENGTH = 256
GLOBAL_BATCH_SIZE = 1_536
PER_DEVICE_PARALLELISM = 192
TRAINER_SEED = 0
DATA_SEED = 535

PRODUCTION_ADDED_STEPS = 160_000
SMOKE_ADDED_STEPS = 20
PRODUCTION_TARGET_STEP = PARENT_STEP + PRODUCTION_ADDED_STEPS
SMOKE_TARGET_STEP = PARENT_STEP + SMOKE_ADDED_STEPS
REWARMUP_STEPS = 16_000
COOLDOWN_STEPS = 32_000
COOLDOWN_START_STEP = PARENT_STEP + PRODUCTION_ADDED_STEPS - COOLDOWN_STEPS
PRODUCTION_HF_EXPORT_STEPS = tuple(
    PARENT_STEP + offset for offset in range(20_000, PRODUCTION_ADDED_STEPS + 1, 20_000)
)
SMOKE_HF_EXPORT_STEPS = (SMOKE_TARGET_STEP,)

LEARNING_RATE = 0.001656070288044504
ADAM_LEARNING_RATE = 0.0015719609666189408
BETA1 = 0.6675603345321236
BETA2 = 0.9908625761010076
EPSILON = 1.901773722851381e-14
MAX_GRAD_NORM = 0.9951880136348764
WEIGHT_DECAY = 0.1
Z_LOSS_WEIGHT = 4.312883184368223e-06

TPU_VARIANT = "v5p-16"
TPU_REGION = "us-east5"
TPU_ZONE = "us-east5-a"
WANDB_PROJECT = "marin"
WANDB_GROUP = "dna-exp535"
VERSION = "2026.08.27-v1"
SMOKE_VERSION = "2026.08.27-smoke-v1"


@dataclass(frozen=True)
class RegionCache:
    source_id: str
    cache_dir: str
    text_key: str
    total_tokens: int


REGION_CACHES = {
    "cds": RegionCache(
        "bolinas-dna/genomes-v5-genome_set-animals-intervals-v5_255_128",
        "gs://marin-us-east5/tokenized/bolinas-v5-cds-char-bos-5149-2db477",
        "seq",
        62_037_687_296,
    ),
    "upstream": RegionCache(
        "bolinas-dna/genomes-v5-genome_set-animals-intervals-v1_255_128",
        "gs://marin-us-east5/tokenized/bolinas-v5-upstream-char-bos-5149-e03807",
        "seq",
        17_481_258_496,
    ),
    "downstream": RegionCache(
        "bolinas-dna/genomes-v5-genome_set-animals-intervals-v15_255_128",
        "gs://marin-us-east5/tokenized/bolinas-v5-downstream-char-bos-5149-d31236",
        "seq",
        5_248_475_136,
    ),
    "ncrna_exon": RegionCache(
        "bolinas-dna/zoonomia-v1-v3_ncrna_exon",
        "gs://marin-us-east5/tokenized/bolinas-v5-ncrna_exon-char-bos-5149-a7dbbf",
        "sequence",
        3_910_928_384,
    ),
    "ccre_non_promoter": RegionCache(
        "bolinas-dna/zoonomia-v1-v3_ccre_non_promoter",
        "gs://marin-us-east5/tokenized/bolinas-v5-ccre_non_promoter-char-bos-5149-5ad1e2",
        "sequence",
        24_739_788_800,
    ),
}

MODEL = LlamaConfig(
    max_seq_len=SEQUENCE_LENGTH,
    hidden_dim=2_944,
    intermediate_dim=11_776,
    num_layers=29,
    num_heads=23,
    num_kv_heads=23,
    rope=Llama3RotaryEmbeddingsConfig(
        theta=500_000,
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8_192,
    ),
    initializer_range=0.02,
    layer_norm_epsilon=1e-5,
    tie_word_embeddings=False,
    gradient_checkpointing=True,
    scan_layers=True,
    use_bias=False,
    reference_checkpoint="NousResearch/Llama-2-7b-hf",
    tokenizer=TOKENIZER_PATH,
)


@AdaptorConfig.register_subclass("exact-hf")
@dataclass(frozen=True)
class ExactHfExportConfig(NoAdaptorConfig):
    """Export HF checkpoints at exact absolute steps after a nonzero-step resume."""

    steps: tuple[int, ...] = ()

    def install_export_hooks(
        self,
        *,
        trainer,
        converter: HFCheckpointConverter | None,
        tokenizer,
        export: AdaptorExportConfig,
    ) -> None:
        del tokenizer
        if export.hf_save_path is None:
            raise ValueError("exact HF exports require hf_save_path")
        if converter is None:
            raise ValueError("exact HF exports require an HF-compatible model")
        base_path = export.hf_save_path
        if trainer.config.checkpointer.append_run_id_to_base_path:
            base_path = prefix_join(base_path, trainer.run_id)
        callback = save_hf_checkpoint_callback(
            base_path,
            converter,
            upload_to_hf=False,
            save_dtype=None,
            generation_config=export.generation_config,
        )
        targets = frozenset(self.steps)
        saved: set[int] = set()

        def save_exact_step(info) -> None:
            step = int(info.step)
            if step in targets and step not in saved:
                callback(info)
                saved.add(step)

        trainer.add_hook(save_exact_step, every=1)


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"missing required environment variable {name}")
    return value


def validate_vendored_tokenizer() -> None:
    for filename, expected in TOKENIZER_SHA256.items():
        path = Path(TOKENIZER_PATH, filename)
        if not path.is_file():
            raise FileNotFoundError(f"missing vendored tokenizer file {path}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"{path} sha256 changed: {observed} != {expected}")


def data_config() -> LmDataConfig:
    components = {
        name: DatasetComponent(
            source=None,
            cache_dir=cache.cache_dir,
            format=DNALmDatasetFormat(
                text_key=cache.text_key,
                uppercase_weight=1.0,
                lowercase_weight=0.01,
            ),
            tags=[f"region={name}", f"source={cache.source_id}"],
            split="train",
        )
        for name, cache in REGION_CACHES.items()
    }
    return LmDataConfig(
        tokenizer=TOKENIZER_PATH,
        components=components,
        train_weights={name: 0.2 for name in components},
        cache_dir=None,
        auto_build_caches=False,
        enforce_eos=True,
        permutation_type="feistel",
        block_cross_document_attention=True,
        mixture_block_size=2_048,
    )


def optimizer_config(added_steps: int) -> AdamHConfig:
    return AdamHConfig(
        learning_rate=LEARNING_RATE,
        adam_lr=ADAM_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        max_grad_norm=MAX_GRAD_NORM,
        warmup=0.1,
        rewarmup=added_steps // 10,
        decay=added_steps // 5,
        cycle_length=[PARENT_STEP, added_steps],
        lr_schedule="linear",
        min_lr_ratio=0.0,
    )


def _resources() -> ResourceConfig:
    return ResourceConfig.with_tpu(
        TPU_VARIANT,
        slice_count=1,
        disk="100g",
        regions=[TPU_REGION],
        zone=TPU_ZONE,
        preemptible=True,
    )


def _mesh() -> MeshConfig:
    token_axes = (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)
    return MeshConfig(
        axes={"replica": 1, "data": -1, "model": 1},
        dcn_axes={"replica_dcn": -1},
        compute_mapping={"token": token_axes, "token_repeat": token_axes},
        param_mapping={"embed": "data"},
    )


def _run_on_tpu(pod_config: TrainLmOnPodConfig) -> None:
    from exp535_4b_uniform_five_region import formats as _formats

    del _formats
    run_levanter_train_lm(pod_config)


def _training_job(pod_config: TrainLmOnPodConfig) -> None:
    remote(_run_on_tpu, resources=pod_config.resources)(pod_config)


def build_training(mode: Literal["smoke", "production"]) -> ArtifactStep[LevanterCheckpoint]:
    validate_vendored_tokenizer()
    required_env("WANDB_API_KEY")
    if required_env("MARIN_PREFIX").rstrip("/") != "gs://marin-us-east5":
        raise ValueError("MARIN_PREFIX must be gs://marin-us-east5")

    if mode == "smoke":
        added_steps = SMOKE_ADDED_STEPS
        target_step = SMOKE_TARGET_STEP
        export_steps = SMOKE_HF_EXPORT_STEPS
        run_id = "dna-exp535-4b-uniform-five-region-smoke"
        name = f"checkpoints/{run_id}"
        version = SMOKE_VERSION
        keep: list[dict] = []
    else:
        added_steps = PRODUCTION_ADDED_STEPS
        target_step = PRODUCTION_TARGET_STEP
        export_steps = PRODUCTION_HF_EXPORT_STEPS
        run_id = "dna-exp535-4b-uniform-five-region"
        name = f"checkpoints/{run_id}"
        version = VERSION
        keep = [{"every": COOLDOWN_START_STEP, "until": COOLDOWN_START_STEP}]

    resources = _resources()
    tags = [
        "dna",
        "marin-dna",
        "exp535",
        "4b",
        "uniform-five-region",
        f"mode={mode}",
        f"parent_step={PARENT_STEP}",
        f"added_steps={added_steps}",
        f"data_seed={DATA_SEED}",
        f"accelerator={TPU_VARIANT}",
        f"marin_commit={MARIN_COMMIT}",
    ]
    forwarded_env = {
        "MARIN_PREFIX": "gs://marin-us-east5",
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "UV_LOCK_TIMEOUT": "7200",
    }

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        train_config = TrainLmConfig(
            data=data_config(),
            trainer=TrainerConfig(
                id=run_id,
                tracker=WandbConfig(
                    project=WANDB_PROJECT,
                    name=run_id,
                    group=WANDB_GROUP,
                    tags=tags,
                    replicate_path=ctx.output_path,
                ),
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                train_batch_size=GLOBAL_BATCH_SIZE,
                per_device_parallelism=PER_DEVICE_PARALLELISM,
                per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
                num_train_steps=target_step,
                steps_per_eval=target_step,
                checkpointer=CheckpointerConfig(
                    save_interval=timedelta(minutes=10),
                    keep=keep,
                    keep_last_temporary_checkpoints=1,
                ),
                mesh=_mesh(),
                seed=TRAINER_SEED,
                initialize_from=PARENT_CHECKPOINT,
                allow_partial_checkpoint=False,
                allow_nondivisible_batch_size=True,
            ),
            model=MODEL,
            optimizer=optimizer_config(added_steps),
            z_loss_weight=Z_LOSS_WEIGHT,
            train_seq_len=SEQUENCE_LENGTH,
            data_seed=DATA_SEED,
            adapter=ExactHfExportConfig(steps=export_steps),
            hf_save_steps=1,
            eval_harness=None,
            labeled_eval=None,
        )
        return TrainLmOnPodConfig(
            train_config=train_config,
            resources=ctx.runtime_arg("train_resources"),
            output_path=ctx.output_path,
            env_vars=forwarded_env,
            auto_build_caches=False,
        )

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=LevanterCheckpoint,
        run=_training_job,
        build_config=build_config,
        runtime_args={"train_resources": resources},
    )


@click.command(help=__doc__)
@click.option("--mode", type=click.Choice(("smoke", "production")), required=True)
@build_options
def main(mode: Literal["smoke", "production"]) -> ArtifactStep[LevanterCheckpoint]:
    return build_training(mode)


if __name__ == "__main__":
    main()
