"""Run the exp472 AFDB smoke test on one us-east5 v6e-4 TPU slice."""

import os

import click
from common import build_smoke_run, existing_afdb_cache
from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.training.training import LevanterCheckpoint

TPU_TYPE = "v6e-4"
REGION = "us-east5"
AFDB_CACHE = "gs://marin-us-east5/tokenized/contacts-v1/2026.07.13.1"
VALIDATION_CACHE = "gs://marin-us-east5/tokenized/contacts-v1-val/2026.07.13.1"


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[LevanterCheckpoint]:
    tpu_type = os.environ.get("EXP472_TPU", TPU_TYPE)
    region = os.environ.get("EXP472_REGION", REGION)
    if tpu_type != TPU_TYPE or region != REGION:
        raise ValueError(
            f"this smoke test is fixed to {TPU_TYPE} in {REGION}; got {tpu_type} in {region}"
        )

    return build_smoke_run(
        platform="tpu",
        train_cache=existing_afdb_cache(
            name="inputs/contacts-v1-afdb-train-tpu-path-only",
            version="2026.07.13.1",
            source=AFDB_CACHE,
        ),
        validation_cache=existing_afdb_cache(
            name="inputs/contacts-v1-afdb-validation-tpu-path-only",
            version="2026.07.13.1",
            source=VALIDATION_CACHE,
        ),
        resources=ResourceConfig.with_tpu(TPU_TYPE, regions=[REGION]),
        attention_backend=None,
    )


if __name__ == "__main__":
    main()
