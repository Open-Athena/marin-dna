"""Admit a replacement GPU gang before releasing an exp593 annealing writer.

The gang waits without opening W&B or checkpoint state. An operator publishes a
one-use gate only after the old writer is physically fenced. Training then uses
the unchanged anneal_train.run implementation and stable trial identity.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from botocore.exceptions import ClientError
from experiments.grug.dispatch import _forwarded_env_vars
from experiments.grug.moe_hero_ep.train import _apply_hero_ep_runtime_defaults
from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.rpc.proto_display import priority_band_value
from marin.training.training import resolve_training_env
from rigging.timing import Duration

from exp593_dna.anneal import OWNER
from exp593_dna.anneal_tasks import panel_binding
from exp593_dna.anneal_tokenizer import verify_source_tokenizer
from exp593_dna.anneal_train import TrainingConfig, run
from exp593_dna.pipeline import load_encoder
from exp593_dna.storage import coreweave_client
from exp593_dna.train import WANDB_BASE_URL


@dataclass(frozen=True)
class Gate:
    uri: str
    nonce: str

    def __post_init__(self) -> None:
        if not self.uri.startswith(f"{OWNER}/annealing/operations/gates/"):
            raise ValueError("Standby gate is outside the annealing operations owner")
        if not self.uri.endswith(".json") or len(self.nonce) < 24:
            raise ValueError("Standby gate path or nonce is invalid")


def wait_for_gate(config: TrainingConfig, gate: Gate) -> dict:
    client = coreweave_client()
    parsed = urlparse(gate.uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    while True:
        try:
            client.head_object(Bucket=bucket, Key=key)
        except ClientError as error:
            if error.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                raise
        else:
            payload = json.loads(
                client.get_object(Bucket=bucket, Key=key)["Body"].read()
            )
            if not (
                payload.get("version") == 1
                and payload.get("run_id") == config.identity
                and payload.get("nonce") == gate.nonce
                and payload.get("cluster") == config.cluster
                and payload.get("nodes") == config.nodes
                and payload.get("released_by") == "eczech-exp593-operator"
            ):
                raise ValueError("Standby release gate does not bind this exact gang")
            if config.validation_updates is None:
                if not (
                    payload.get("old_writer_fenced") is True
                    and payload.get("fence_receipt_sha256")
                    and payload.get("checkpoint")
                    and payload.get("metadata_sha256")
                ):
                    raise ValueError(
                        "Production standby gate lacks a writer fence or checkpoint"
                    )
                pointer_uri = urlparse(config.checkpoint_root + "/latest.json")
                pointer = json.loads(
                    client.get_object(
                        Bucket=pointer_uri.netloc, Key=pointer_uri.path.lstrip("/")
                    )["Body"].read()
                )
                if (
                    pointer["checkpoint"] != payload["checkpoint"]
                    or pointer["metadata_sha256"] != payload["metadata_sha256"]
                ):
                    raise ValueError(
                        "Released checkpoint differs from the recovery pointer"
                    )
            elif payload.get("bounded_canary") is not True:
                raise ValueError("Validation standby gate lacks its canary marker")
            return payload
        time.sleep(10)


def wait_then_run(config: TrainingConfig, gate: Gate) -> None:
    wait_for_gate(config, gate)
    run(config)


def dispatch(config: TrainingConfig, gate: Gate) -> None:
    verify_source_tokenizer(load_encoder())
    if not panel_binding()["complete_readback_verified"]:
        raise ValueError("Task panel publication was not fully verified")
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=False,
        moe_implementation="scatter",
        remat_mode="recompute_all",
        processes_per_task=8,
    )
    resources = ResourceConfig.with_gpu(
        "H100", count=8, replicas=config.nodes, cpu=32, ram="600g", disk="900g"
    )
    env = resolve_training_env(
        {
            **_forwarded_env_vars(),
            "MARIN_PREFIX": "s3://marin-us-east-02a/marin",
            **{
                name: os.environ[name]
                for name in ("CW_KEY_ID", "CW_KEY_SECRET", "WANDB_API_KEY")
            },
            "WANDB_BASE_URL": WANDB_BASE_URL,
            "WANDB_ENTITY": "eric-czech",
            "WANDB_PROJECT": "marin",
        },
        resources,
    )
    env.update(
        WANDB_BASE_URL=WANDB_BASE_URL,
        WANDB_ENTITY="eric-czech",
        WANDB_PROJECT="marin",
    )
    job = current_client().submit(
        JobRequest(
            name=f"train-{config.identity}",
            entrypoint=Entrypoint.from_callable(wait_then_run, args=[config, gate]),
            resources=resources,
            processes_per_task=8,
            environment=create_environment(env_vars=env, extras=["gpu"]),
            priority=priority_band_value("batch"),
            timeout=Duration.from_hours(3 if config.validation_updates else 14 * 24),
            max_retries_failure=0,
            max_task_failures=0,
        )
    )
    job.wait(raise_on_failure=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", choices=("181b", "364b"), required=True)
    parser.add_argument(
        "--cluster", choices=("cw-rno2a", "cw-us-east-02a"), required=True
    )
    parser.add_argument("--nodes", type=int, choices=(1, 4, 8), required=True)
    parser.add_argument("--gate-uri", required=True)
    parser.add_argument("--gate-nonce", required=True)
    parser.add_argument("--validation-updates", type=int)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = TrainingConfig(
        trial=args.trial,
        cluster=args.cluster,
        nodes=args.nodes,
        validation_updates=args.validation_updates,
        run_id=args.run_id,
    )
    if config.validation_updates is None and not (
        (config.nodes == 8 and config.cluster == "cw-rno2a")
        or (
            config.trial == "364b"
            and config.nodes in (1, 4)
            and config.cluster == "cw-us-east-02a"
        )
    ):
        raise ValueError("Production standby requires an approved H100 placement")
    dispatch(config, Gate(args.gate_uri, args.gate_nonce))


if __name__ == "__main__":
    main()
