"""Run one exp472 PlantCAD2 LR/WD sweep trial on whole eight-H100 nodes."""

from dataclasses import dataclass

import click
from common import (
    CACHE_VERSION,
    EXPERIMENT_RELATIVE,
    TOKENIZED_CACHE_RELATIVE,
    TRAIN_CACHE_NAME,
    VALIDATION_CACHE_NAME,
    build_sweep_run,
    env_int,
    existing_plantcad_cache,
    global_batch_size,
    parse_sweep_point,
    require_marin_prefix,
)
from fray.types import ResourceConfig
from levanter.layers.attention import AttentionBackend
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import prefix_join

COREWEAVE_ROOT = "s3://marin-us-east-02a"
TOKENIZED_CACHE = prefix_join(COREWEAVE_ROOT, TOKENIZED_CACHE_RELATIVE)
EXPERIMENT_PREFIX = prefix_join(COREWEAVE_ROOT, EXPERIMENT_RELATIVE)
GPUS_PER_NODE = 8
MAX_SEQUENCES_PER_DEVICE = 8


@dataclass(frozen=True)
class GpuBatchConfig:
    data_parallelism: int
    tensor_parallelism: int
    per_device_parallelism: int
    gradient_accumulation: int


def gpu_batch_fit(nodes: int, batch_size: int) -> GpuBatchConfig:
    devices = GPUS_PER_NODE * nodes
    if batch_size % devices:
        raise ValueError(
            f"global batch {batch_size} is not divisible by {devices} H100s"
        )
    full_per_device_batch = batch_size // devices
    per_device_parallelism = min(full_per_device_batch, MAX_SEQUENCES_PER_DEVICE)
    while full_per_device_batch % per_device_parallelism:
        per_device_parallelism -= 1
    return GpuBatchConfig(
        data_parallelism=devices,
        tensor_parallelism=1,
        per_device_parallelism=per_device_parallelism,
        gradient_accumulation=full_per_device_batch // per_device_parallelism,
    )


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    require_marin_prefix(EXPERIMENT_PREFIX)
    nodes = env_int("EXP472_GPU_NODES", 1)
    batch_size = global_batch_size()
    batch = gpu_batch_fit(nodes, batch_size)
    return build_sweep_run(
        point=parse_sweep_point(),
        train_cache=existing_plantcad_cache(
            name=TRAIN_CACHE_NAME,
            version=CACHE_VERSION,
            source=TOKENIZED_CACHE,
        ),
        validation_cache=existing_plantcad_cache(
            name=VALIDATION_CACHE_NAME,
            version=CACHE_VERSION,
            source=TOKENIZED_CACHE,
        ),
        resources=ResourceConfig.with_gpu(
            "H100",
            count=8,
            replicas=nodes,
            cpu=32,
            ram="256g",
            disk="256g",
        ),
        attention_backend=AttentionBackend.JAX_FLASH,
        batch_size=batch_size,
        tensor_parallelism=batch.tensor_parallelism,
        per_device_parallelism=batch.per_device_parallelism,
        runtime_tags=(
            "accelerator=H100",
            f"nodes={nodes}",
            f"data_parallelism={batch.data_parallelism}",
            f"tensor_parallelism={batch.tensor_parallelism}",
            f"per_device_parallelism={batch.per_device_parallelism}",
            f"gradient_accumulation={batch.gradient_accumulation}",
        ),
        expected_output_prefix=EXPERIMENT_PREFIX,
    )


if __name__ == "__main__":
    main()
