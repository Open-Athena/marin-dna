"""Run the exp472 AFDB smoke test on whole eight-H100 nodes."""

import click
from common import build_smoke_run, env_int, existing_afdb_cache
from fray.types import ResourceConfig
from levanter.layers.attention import AttentionBackend
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.training.training import LevanterCheckpoint

AFDB_CACHE = (
    "s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/"
    "tokenized/contacts-v1/2026.07.25"
)
VALIDATION_CACHE = (
    "s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/"
    "tokenized/contacts-v1-val/2026.07.25"
)


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    nodes = env_int("EXP472_GPU_NODES", 1)
    return build_smoke_run(
        platform="gpu",
        train_cache=existing_afdb_cache(
            name="inputs/contacts-v1-afdb-train-gpu",
            version="2026.07.25",
            source=AFDB_CACHE,
        ),
        validation_cache=existing_afdb_cache(
            name="inputs/contacts-v1-afdb-validation-gpu",
            version="2026.07.25",
            source=VALIDATION_CACHE,
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
    )


if __name__ == "__main__":
    main()
