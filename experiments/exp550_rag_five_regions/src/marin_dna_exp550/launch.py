"""Marin launch for the fixed five-region RAG experiment and a synthetic TPU pilot."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import click
import jmp
from fray.types import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import (
    DatasetComponent,
    LmDataConfig,
    UrlDatasetSourceConfig,
)
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.adamh import AdamHConfig
from levanter.optim.config import OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.training.training import (
    LevanterCheckpoint,
    TrainLmOnPodConfig,
    resolve_training_env,
    run_levanter_train_lm,
)

from marin_dna_exp550.cadence import CompletedUpdateAdaptor
from marin_dna_exp550.formats import RagFormat
from marin_dna_exp550.recipe import (
    BATCH_DOCUMENTS,
    CHECKPOINT_EVERY,
    DATA_SEED,
    MODEL_DIMENSIONS,
    MODEL_TOKENS,
    REGIONS,
    TRAIN_SEED,
    TRAIN_UPDATES,
    resolve_optimizer,
)


def tokenizer_path() -> str:
    return str(Path(__file__).with_name("tokenizer"))


def model_config() -> Qwen3Config:
    return Qwen3Config(
        max_seq_len=MODEL_TOKENS,
        **MODEL_DIMENSIONS,
        rope=Llama3RotaryEmbeddingsConfig(),
        use_sliding_window=False,
        tie_word_embeddings=False,
        tokenizer=tokenizer_path(),
        reference_checkpoint=tokenizer_path(),
    )


@OptimizerConfig.register_subclass("exp550_fixed_horizon_adamh")
@dataclass(frozen=True)
class FixedHorizonAdamH(AdamHConfig):
    def build(self, num_train_steps: int) -> Any:
        # A short pilot follows the identical first updates of the 100k schedule.
        del num_train_steps
        return super().build(TRAIN_UPDATES)


@dataclass(frozen=True)
class TrainingRequest:
    pod: TrainLmOnPodConfig
    datasets: dict[str, dict[str, Any]]
    pilot: bool


def read_dataset_manifest(
    path: str | None, *, pilot: bool
) -> dict[str, dict[str, Any]]:
    if pilot:
        if path:
            raise ValueError("the synthetic pilot does not accept biological inputs")
        return {}
    if not path:
        raise ValueError("production requires the verified public dataset manifest")
    datasets = json.loads(Path(path).read_text())
    if set(datasets) != set(REGIONS):
        raise ValueError("expected exactly five public region datasets")
    for name, spec in datasets.items():
        if spec[
            "repo_id"
        ] != f"marin-dna/rag-five-regions-v1-{name}" or not re.fullmatch(
            "[0-9a-f]{40}", spec["revision"]
        ):
            raise ValueError(
                "training inputs must use the registered immutable public releases"
            )
        if (
            spec["train_rows"] <= 0
            or not 0 < spec["validation_rows"] <= 400
            or not spec["anonymous_verified"]
        ):
            raise ValueError(
                "dataset release did not pass the required publication checks"
            )
    return datasets


def training_config(
    output_path: str, *, run_id: str, pilot: bool, per_device: int
) -> TrainLmConfig:
    if per_device not in {1, 5}:
        raise ValueError("the eight-chip pilot supports microbatch 1 or 5")
    components = {}
    for region in REGIONS:
        cache = f"{output_path}/tokenized/{region}"
        source = UrlDatasetSourceConfig(
            train_urls=[],
            validation_urls=[],
            cache_dir=cache,
            format=RagFormat(),
            tags=[f"region={region}"],
        )
        components[region] = DatasetComponent(
            source=source,
            cache_dir=cache,
            format=RagFormat(),
            tags=[f"region={region}"],
        )
    data = LmDataConfig(
        components=components,
        train_weights={region: 0.2 for region in REGIONS},
        tokenizer=tokenizer_path(),
        cache_dir=None,
        auto_build_caches=False,
        enforce_eos=False,
        shuffle=True,
        permutation_type="feistel",
    )
    cadence = 5 if pilot else CHECKPOINT_EVERY
    token_axes = (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)
    trainer = TrainerConfig(
        id=run_id,
        seed=TRAIN_SEED,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=BATCH_DOCUMENTS,
        per_device_parallelism=per_device,
        per_device_eval_parallelism=per_device,
        allow_nondivisible_batch_size=False,
        num_train_steps=20 if pilot else TRAIN_UPDATES,
        steps_per_eval=cadence,
        max_eval_batches=None,
        checkpointer=CheckpointerConfig(
            save_interval=timedelta(minutes=10), keep=[{"every": cadence}]
        ),
        mesh=MeshConfig(
            axes={"replica": 1, "data": -1, "model": 1},
            compute_mapping={"token": token_axes, "token_repeat": token_axes},
        ),
        tracker=WandbConfig(
            project="marin",
            name=run_id,
            group="dna-exp550-rag-five-regions",
            tags=["dna-exp550", "rag", "synthetic-pilot" if pilot else "five-regions"],
            replicate_path=output_path,
        ),
        require_accelerator=True,
    )
    return TrainLmConfig(
        data=data,
        model=model_config(),
        optimizer=FixedHorizonAdamH(**asdict(resolve_optimizer())),
        trainer=trainer,
        train_seq_len=MODEL_TOKENS,
        data_seed=DATA_SEED,
        z_loss_weight=1e-7,
        hf_save_steps=cadence,
        adapter=CompletedUpdateAdaptor(),
        eval_harness=None,
    )


def train_worker(request: TrainingRequest) -> None:
    """Prepare caches on the free TPU host, then run the standard Marin trainer."""
    pod = request.pod
    if pod.output_path is None:
        raise ValueError("training requires an immutable output path")
    child_env = os.environ.copy()
    child_env.pop("IRIS_TASK_ID", None)
    child_env["JAX_PLATFORMS"] = "cpu"
    with TemporaryDirectory(prefix="exp550-tokenize-") as temporary:
        source = Path(temporary) / "request.json"
        source.write_text(
            json.dumps(
                {
                    "cache_root": f"{pod.output_path}/tokenized",
                    "datasets": request.datasets,
                    "pilot": request.pilot,
                }
            )
        )
        subprocess.run(
            [sys.executable, "-m", "marin_dna_exp550.tokenize_local", str(source)],
            env=child_env,
            check=True,
        )
    # Resolve package resources on the worker; coordinator absolute paths differ.
    inner = replace(
        pod.train_config,
        data=replace(pod.train_config.data, tokenizer=tokenizer_path()),
        model=replace(
            pod.train_config.model,
            tokenizer=tokenizer_path(),
            reference_checkpoint=tokenizer_path(),
        ),
    )
    run_levanter_train_lm(replace(pod, train_config=inner))


def dispatch(request: TrainingRequest) -> None:
    # Secrets are runtime environment only, never fields in the recorded config.
    runtime_env = {
        name: os.environ[name]
        for name in ("WANDB_API_KEY", "WANDB_ENTITY", "MARIN_PREFIX")
    }
    runtime_env.update(
        {
            "WANDB_PROJECT": "marin",
            "UV_LOCK_TIMEOUT": "7200",
            "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        }
    )
    for name in ("UV_PROJECT", "GIT_COMMIT"):
        if name in os.environ:
            runtime_env[name] = os.environ[name]
    runtime_env = resolve_training_env(runtime_env, request.pod.resources)
    remote(train_worker, resources=request.pod.resources, env_vars=runtime_env)(request)


def build_training(
    datasets: dict[str, dict[str, Any]], *, pilot: bool, per_device: int, region: str
) -> ArtifactStep[LevanterCheckpoint]:
    if region != "us-east1":
        raise ValueError("this launch pins the verified us-east1 free TPU placement")
    expected_prefix = "gs://marin-us-east1/MarinDNA/exp550_rag_five_regions"
    if os.environ.get("MARIN_PREFIX") != expected_prefix:
        raise ValueError(f"MARIN_PREFIX must be {expected_prefix}")
    run_id = "dna-exp550-rag46m-five-regions-v1" + (
        f"-pilot-mb{per_device}" if pilot else ""
    )
    resources = ResourceConfig.with_tpu(
        "v6e-8", regions=[region], cpu=16, ram="128g", disk="200g", preemptible=True
    )

    def build_config(ctx: StepContext) -> TrainingRequest:
        inner = training_config(
            ctx.output_path, run_id=run_id, pilot=pilot, per_device=per_device
        )
        return TrainingRequest(
            TrainLmOnPodConfig(
                train_config=inner,
                resources=ctx.runtime_arg("train_resources"),
                output_path=ctx.output_path,
                env_vars={
                    "WANDB_ENTITY": "gonzalobenegas",
                    "WANDB_PROJECT": "marin",
                    "MARIN_PREFIX": expected_prefix,
                },
            ),
            datasets=datasets,
            pilot=pilot,
        )

    return ArtifactStep(
        name=f"checkpoints/{run_id}",
        version=resolve_version(f"checkpoints/{run_id}", None),
        artifact_type=LevanterCheckpoint,
        run=dispatch,
        build_config=build_config,
        runtime_args={"train_resources": resources},
    )


@click.command(help=__doc__)
@click.option("--dataset-manifest", type=click.Path(exists=True))
@click.option("--pilot", is_flag=True)
@click.option("--per-device", type=int, default=5)
@click.option("--region", default="us-east1")
@build_options
def main(
    dataset_manifest: str | None, pilot: bool, per_device: int, region: str
) -> ArtifactStep[LevanterCheckpoint]:
    return build_training(
        read_dataset_manifest(dataset_manifest, pilot=pilot),
        pilot=pilot,
        per_device=per_device,
        region=region,
    )


if __name__ == "__main__":
    main()
