"""Run the exp472 PlantCAD2 smoke test against a region-local GCS cache."""

import os

import click
from common import (
    SEQ_LEN,
    build_smoke_run,
    existing_plantcad_cache,
    require_marin_prefix,
    required_env,
)
from fray.types import ResourceConfig
from levanter.store.cache import ShardedCacheLayout
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.processing.tokenize.cache_stats import read_tokenized_cache_stats
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.cluster_config import StoreType, data_config
from rigging.filesystem.storage_path import StoragePath, prefix_join

TPU_TYPE = "v6e-4"
DATASET_REVISION = "4a444fff5520b992aa978d92a5af509a81977098"
CACHE_VERSION = "2026.08.19"
CACHE_RELATIVE = "MarinDNA/tokenized/plantcad/Angiosperm_65_genomes_8192bp"
EXPERIMENT_RELATIVE = "MarinDNA/exp472_plantcad2_baseline/tpu"
EXPECTED_SPLIT_ROWS = {"train": 2_638_656, "validation": 329_832}


def regional_root(region: str) -> str:
    spec = data_config().region_buckets.get(region)
    if spec is None:
        raise ValueError(f"no Marin data bucket is configured for region {region!r}")
    if spec.store is not StoreType.GCS:
        raise ValueError(
            f"TPU region {region!r} resolved to non-GCS store {spec.store!r}"
        )
    return f"gs://{spec.name}"


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
    tpu_type = os.environ.get("TPU", TPU_TYPE).strip().lower()
    region = required_env("REGION").strip().lower()
    if tpu_type != TPU_TYPE:
        raise ValueError(f"this smoke test requires {TPU_TYPE}, got {tpu_type!r}")
    root = regional_root(region)
    cache_path = prefix_join(root, CACHE_RELATIVE)
    experiment_prefix = prefix_join(root, EXPERIMENT_RELATIVE)
    require_marin_prefix(experiment_prefix)
    validate_regional_cache(cache_path)

    return build_smoke_run(
        platform=f"tpu-{region}",
        train_cache=existing_plantcad_cache(
            name=f"inputs/plantcad-angiosperm-train-tpu-{region}-path-only",
            version=CACHE_VERSION,
            source=cache_path,
        ),
        validation_cache=existing_plantcad_cache(
            name=f"inputs/plantcad-angiosperm-validation-tpu-{region}-path-only",
            version=CACHE_VERSION,
            source=cache_path,
        ),
        resources=ResourceConfig.with_tpu(TPU_TYPE, regions=[region]),
        attention_backend=None,
        expected_output_prefix=experiment_prefix,
    )


if __name__ == "__main__":
    main()
