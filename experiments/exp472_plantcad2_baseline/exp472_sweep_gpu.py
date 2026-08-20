"""Run one exp472 PlantCAD2 LR/WD sweep trial on whole eight-H100 nodes.

``TRIAL`` selects the logical configuration. ``CLUSTER`` selects either
``cw-us-east-02a`` or ``cw-rno2a``, and ``NODES`` selects a supported whole-node
gang size. Cluster and node count are placement only: reslicing preserves the
trial's W&B ID and checkpoint path.
"""

import os
from dataclasses import dataclass

import click
from fray.types import ResourceConfig
from levanter.layers.attention import AttentionBackend
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import prefix_join

from experiments.exp472_plantcad2_baseline.common import (
    CACHE_VERSION,
    EXPERIMENT_RELATIVE,
    TOKENIZED_CACHE_RELATIVE,
    TRAIN_CACHE_NAME,
    build_sweep_run,
    existing_plantcad_cache,
    global_batch_size,
    parse_sweep_point,
    require_marin_prefix,
)

COREWEAVE_ROOT = "s3://marin-us-east-02a"
TOKENIZED_CACHE = prefix_join(COREWEAVE_ROOT, TOKENIZED_CACHE_RELATIVE)
EXPERIMENT_PREFIX = prefix_join(COREWEAVE_ROOT, EXPERIMENT_RELATIVE)
MAX_SEQUENCES_PER_DEVICE = 8
ALLOWED_NODES = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class ClusterSpec:
    gpu_variant: str
    gpus_per_node: int
    cpu: int
    ram: str
    disk: str


CLUSTERS = {
    "cw-us-east-02a": ClusterSpec(
        gpu_variant="H100",
        gpus_per_node=8,
        cpu=32,
        ram="256g",
        disk="256g",
    ),
    "cw-rno2a": ClusterSpec(
        gpu_variant="H100",
        gpus_per_node=8,
        cpu=32,
        ram="256g",
        disk="256g",
    ),
}


@dataclass(frozen=True)
class GpuBatchConfig:
    data_parallelism: int
    tensor_parallelism: int
    per_device_parallelism: int
    gradient_accumulation: int


def gpu_batch_fit(spec: ClusterSpec, nodes: int, batch_size: int) -> GpuBatchConfig:
    devices = spec.gpus_per_node * nodes
    if batch_size % devices:
        raise ValueError(
            f"global batch {batch_size} is not divisible by {devices} GPUs"
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


def parse_cluster() -> tuple[str, ClusterSpec]:
    cluster = os.environ.get("CLUSTER", "").strip().lower()
    try:
        return cluster, CLUSTERS[cluster]
    except KeyError as exc:
        raise SystemExit(f"CLUSTER must be one of: {', '.join(CLUSTERS)}") from exc


def parse_nodes() -> int:
    raw = os.environ.get("NODES")
    if raw is None:
        raise SystemExit("missing required environment variable NODES")
    nodes = int(raw)
    if nodes not in ALLOWED_NODES:
        choices = ", ".join(str(value) for value in ALLOWED_NODES)
        raise SystemExit(f"NODES must be one of: {choices}")
    return nodes


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    require_marin_prefix(EXPERIMENT_PREFIX)
    cluster, spec = parse_cluster()
    nodes = parse_nodes()
    batch_size = global_batch_size()
    batch = gpu_batch_fit(spec, nodes, batch_size)
    return build_sweep_run(
        point=parse_sweep_point(),
        train_cache=existing_plantcad_cache(
            name=TRAIN_CACHE_NAME,
            version=CACHE_VERSION,
            source=TOKENIZED_CACHE,
        ),
        resources=ResourceConfig.with_gpu(
            spec.gpu_variant,
            count=spec.gpus_per_node,
            replicas=nodes,
            cpu=spec.cpu,
            ram=spec.ram,
            disk=spec.disk,
        ),
        attention_backend=AttentionBackend.JAX_FLASH,
        batch_size=batch_size,
        tensor_parallelism=batch.tensor_parallelism,
        per_device_parallelism=batch.per_device_parallelism,
        runtime_tags=(
            "platform=coreweave",
            f"cluster={cluster}",
            f"gpu={spec.gpu_variant}",
            f"nodes={nodes}",
            f"data_parallelism={batch.data_parallelism}",
            f"tensor_parallelism={batch.tensor_parallelism}",
            f"per_device_parallelism={batch.per_device_parallelism}",
            f"gradient_accumulation={batch.gradient_accumulation}",
        ),
        wandb_run_suffix=None,
        expected_output_prefix=EXPERIMENT_PREFIX,
    )


if __name__ == "__main__":
    main()
