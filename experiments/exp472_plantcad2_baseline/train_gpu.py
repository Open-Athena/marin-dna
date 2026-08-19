"""Run the exp472 PlantCAD2 smoke test on whole eight-H100 nodes."""

import click
from common import (
    build_smoke_run,
    env_int,
    existing_plantcad_cache,
    require_marin_prefix,
)
from fray.types import ResourceConfig
from levanter.layers.attention import AttentionBackend
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.training.training import LevanterCheckpoint

DATASET_REVISION = "4a444fff5520b992aa978d92a5af509a81977098"
CACHE_VERSION = "2026.08.19"
TOKENIZED_CACHE = (
    "s3://marin-us-east-02a/MarinDNA/tokenized/plantcad/Angiosperm_65_genomes_8192bp"
)
EXPERIMENT_PREFIX = "s3://marin-us-east-02a/MarinDNA/exp472_plantcad2_baseline/gpu"


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    require_marin_prefix(EXPERIMENT_PREFIX)
    nodes = env_int("EXP472_GPU_NODES", 1)
    return build_smoke_run(
        platform="gpu",
        train_cache=existing_plantcad_cache(
            name="inputs/plantcad-angiosperm-train-gpu-path-only",
            version=CACHE_VERSION,
            source=TOKENIZED_CACHE,
        ),
        validation_cache=existing_plantcad_cache(
            name="inputs/plantcad-angiosperm-validation-gpu-path-only",
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
        expected_output_prefix=EXPERIMENT_PREFIX,
    )


if __name__ == "__main__":
    main()
