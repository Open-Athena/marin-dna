"""Run one exp472 PlantCAD2 LR/WD sweep trial on a region-local TPU.

``TRIAL`` selects the logical configuration. ``REGION`` selects the regional
W&B run and regional data/checkpoint root; ``TPU`` is resliceable placement
within that region and does not enter either identity.
"""

import math
import os
from dataclasses import dataclass

import click
from fray.types import (
    ResourceConfig,
    get_tpu_topology,
    tpu_family,
    tpu_hbm_capacity_bytes,
)
from levanter.store.cache import ShardedCacheLayout
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.processing.tokenize.cache_stats import read_tokenized_cache_stats
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.cluster_config import StoreType, data_config
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.exp472_plantcad2_baseline.common import (
    CACHE_VERSION,
    DEFAULT_TRAIN_STEPS,
    EXPERIMENT_RELATIVE,
    MODEL_CONFIG,
    MODEL_PARAMS,
    SEQ_LEN,
    TOKENIZED_CACHE_RELATIVE,
    TRAIN_CACHE_NAME,
    VALIDATION_CACHE_NAME,
    build_sweep_run,
    env_int,
    existing_plantcad_cache,
    global_batch_size,
    parse_sweep_point,
    require_marin_prefix,
    required_env,
)

EXPECTED_SPLIT_ROWS = {"train": 2_638_656, "validation": 329_832}
CORRECTION_FACTORS = {"v5e": 0.5, "v6e": 0.3, "v5p": 0.45}
MIN_TPU_CHIPS = 32
MAX_TPU_CHIPS = 512


@dataclass(frozen=True)
class TpuBatchConfig:
    data_parallelism: int
    tensor_parallelism: int
    per_device_parallelism: int
    gradient_accumulation: int


def _batch_memory_bytes(batch_size: int, correction_factor: float) -> int:
    parameter_bytes = MODEL_PARAMS * 4
    optimizer_bytes = MODEL_PARAMS * 8
    hidden = batch_size * SEQ_LEN * MODEL_CONFIG.hidden_dim * 2
    attention = batch_size * SEQ_LEN * MODEL_CONFIG.hidden_dim * 4 * 2
    mlp = batch_size * SEQ_LEN * MODEL_CONFIG.intermediate_dim * 2
    saved_layers = max(math.floor(MODEL_CONFIG.num_layers * 0.75), 4)
    activation_bytes = (hidden + attention + mlp) * saved_layers
    return math.ceil(
        (parameter_bytes + optimizer_bytes + activation_bytes) * correction_factor
    )


def tpu_batch_fit(
    tpu: str,
    batch_size: int,
    *,
    allow_small_smoke: bool = False,
) -> TpuBatchConfig:
    family = tpu_family(tpu)
    try:
        correction_factor = CORRECTION_FACTORS[family]
    except KeyError as exc:
        raise ValueError(f"unsupported TPU family {family!r}") from exc

    chips = get_tpu_topology(tpu).chip_count
    if chips > MAX_TPU_CHIPS or (chips < MIN_TPU_CHIPS and not allow_small_smoke):
        raise ValueError(
            f"{tpu} has {chips} physical chips; exp472 requires "
            f"{MIN_TPU_CHIPS}–{MAX_TPU_CHIPS} except for explicit smoke tests"
        )
    data_parallelism = math.gcd(batch_size, chips)
    tensor_parallelism = chips // data_parallelism
    batch_bytes = _batch_memory_bytes(batch_size, correction_factor)
    capacity_bytes = tpu_hbm_capacity_bytes(tpu)

    if batch_bytes <= capacity_bytes:
        return TpuBatchConfig(
            data_parallelism=data_parallelism,
            tensor_parallelism=tensor_parallelism,
            per_device_parallelism=batch_size // data_parallelism,
            gradient_accumulation=1,
        )

    full_per_device_batch = batch_size // data_parallelism
    for per_device_parallelism in range(full_per_device_batch, 0, -1):
        if full_per_device_batch % per_device_parallelism:
            continue
        microbatch_size = per_device_parallelism * data_parallelism
        microbatch_bytes = math.ceil(batch_bytes * microbatch_size / batch_size)
        if microbatch_bytes <= capacity_bytes:
            return TpuBatchConfig(
                data_parallelism=data_parallelism,
                tensor_parallelism=tensor_parallelism,
                per_device_parallelism=per_device_parallelism,
                gradient_accumulation=batch_size // microbatch_size,
            )
    raise ValueError(f"global batch {batch_size} does not fit on {tpu}")


def regional_root(region: str) -> str:
    spec = data_config().region_buckets.get(region)
    if spec is None:
        raise ValueError(f"no Marin data bucket is configured for region {region!r}")
    if spec.store is not StoreType.GCS:
        raise ValueError(
            f"TPU region {region!r} resolved to non-GCS store {spec.store!r}"
        )
    return f"gs://{spec.name}"


def allow_small_tpu_smoke() -> bool:
    value = os.environ.get("EXP472_ALLOW_SMALL_TPU_SMOKE", "0")
    if value not in {"0", "1"}:
        raise ValueError("EXP472_ALLOW_SMALL_TPU_SMOKE must be 0 or 1")
    if value == "0":
        return False

    steps = env_int("EXP472_STEPS", DEFAULT_TRAIN_STEPS)
    suffix = os.environ.get("EXP472_RUN_SUFFIX", "").strip()
    if steps >= DEFAULT_TRAIN_STEPS or not suffix.startswith("smoke-"):
        raise ValueError(
            "small TPU placement requires a shortened EXP472_STEPS and an "
            "EXP472_RUN_SUFFIX beginning with 'smoke-'"
        )
    return True


def validate_regional_cache(cache_path: str) -> None:
    for split, expected_rows in EXPECTED_SPLIT_ROWS.items():
        split_path = prefix_join(cache_path, split)
        ledger_path = ShardedCacheLayout.parse(split_path).ledger
        if not StoragePath(ledger_path).exists():
            raise FileNotFoundError(
                f"missing region-local {split} token cache ledger: {ledger_path}; "
                "tokenize this dataset in the selected TPU region before training"
            )
        try:
            stats = read_tokenized_cache_stats(cache_path, split)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"missing region-local {split} token cache stats below {cache_path}; "
                "tokenize this dataset in the selected TPU region before training"
            ) from exc
        expected_tokens = expected_rows * SEQ_LEN
        if (
            stats.total_elements != expected_rows
            or stats.total_tokens != expected_tokens
        ):
            raise ValueError(
                f"invalid region-local {split} cache at {cache_path}: {stats}; "
                f"expected {expected_rows} rows and {expected_tokens} tokens"
            )


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    tpu = required_env("TPU").strip().lower()
    region = required_env("REGION").strip().lower()
    get_tpu_topology(tpu)
    root = regional_root(region)
    cache_path = prefix_join(root, TOKENIZED_CACHE_RELATIVE)
    experiment_prefix = prefix_join(root, EXPERIMENT_RELATIVE)
    require_marin_prefix(experiment_prefix)
    validate_regional_cache(cache_path)
    batch_size = global_batch_size()
    batch = tpu_batch_fit(
        tpu,
        batch_size,
        allow_small_smoke=allow_small_tpu_smoke(),
    )

    return build_sweep_run(
        point=parse_sweep_point(),
        train_cache=existing_plantcad_cache(
            name=TRAIN_CACHE_NAME,
            version=CACHE_VERSION,
            source=cache_path,
        ),
        validation_cache=existing_plantcad_cache(
            name=VALIDATION_CACHE_NAME,
            version=CACHE_VERSION,
            source=cache_path,
        ),
        resources=ResourceConfig.with_tpu(tpu, regions=[region]),
        attention_backend=None,
        batch_size=batch_size,
        tensor_parallelism=batch.tensor_parallelism,
        per_device_parallelism=batch.per_device_parallelism,
        runtime_tags=(
            f"region={region}",
            f"tpu={tpu}",
            f"data_parallelism={batch.data_parallelism}",
            f"tensor_parallelism={batch.tensor_parallelism}",
            f"per_device_parallelism={batch.per_device_parallelism}",
            f"gradient_accumulation={batch.gradient_accumulation}",
        ),
        wandb_run_suffix=region,
        expected_output_prefix=experiment_prefix,
    )


if __name__ == "__main__":
    main()
