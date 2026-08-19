"""Run the exp472 PlantCAD2 smoke test on one us-east5 v6e-4 TPU slice."""

import json
import os

import click
from common import build_smoke_run, existing_plantcad_cache, required_env
from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.s3_compat import fsspec_s3_conf

TPU_TYPE = "v6e-4"
REGION = "us-east5"
DATASET_REVISION = "4a444fff5520b992aa978d92a5af509a81977098"
CACHE_VERSION = "2026.08.19"
TOKENIZED_CACHE = (
    "s3://marin-us-east-02a/MarinDNA/tokenized/plantcad/Angiosperm_65_genomes_8192bp"
)
CWS3_ENDPOINT = "https://cwobject.com"


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
        train_cache=existing_plantcad_cache(
            name="inputs/plantcad-angiosperm-train-tpu-path-only",
            version=CACHE_VERSION,
            source=TOKENIZED_CACHE,
        ),
        validation_cache=existing_plantcad_cache(
            name="inputs/plantcad-angiosperm-validation-tpu-path-only",
            version=CACHE_VERSION,
            source=TOKENIZED_CACHE,
        ),
        resources=ResourceConfig.with_tpu(TPU_TYPE, regions=[REGION]),
        attention_backend=None,
        extra_env_vars={
            "AWS_ACCESS_KEY_ID": required_env("CW_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": required_env("CW_KEY_SECRET"),
            "AWS_ENDPOINT_URL": CWS3_ENDPOINT,
            "AWS_ENDPOINT_URL_S3": CWS3_ENDPOINT,
            "AWS_REGION": "auto",
            "AWS_DEFAULT_REGION": "auto",
            "FSSPEC_S3": json.dumps(fsspec_s3_conf(CWS3_ENDPOINT)),
        },
    )


if __name__ == "__main__":
    main()
