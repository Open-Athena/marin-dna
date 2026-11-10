"""Restore and inspect every native annealing state array without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from importlib.resources import files
from pathlib import Path

import equinox as eqx
import jax
import jmp
from exp586_moe.config import d768_config
from exp586_moe.data_parallel import local_model, replicated_mesh
from exp586_moe.perf5_optimizer import (
    assert_partitioned_optimizer_state,
    match_gradient_placement,
    model_matches_fp32_master,
    partition_expert_optimizer_state,
    sharded_expert_ns,
)
from exp586_moe.schedule import optimizer_config
from exp586_moe.state import fresh_state, state_all_finite
from experiments.grug.dispatch import _forwarded_env_vars
from experiments.grug.moe_hero_ep.model import Transformer
from experiments.grug.moe_hero_ep.train import _apply_hero_ep_runtime_defaults
from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.rpc.proto_display import priority_band_value
from levanter.callbacks.watch import WatchConfig
from levanter.tracker.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from marin.training.training import resolve_training_env
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from exp593_dna import anneal_state
from exp593_dna.anneal import (
    ORIGINAL_TOKENS,
    ORIGINAL_UPDATES,
    OWNER,
    REFERENCE_TOKENS,
    TRIALS,
)
from exp593_dna.publish_inputs import put_immutable
from exp593_dna.storage import coreweave_client


@dataclass(frozen=True)
class AuditConfig:
    trial: str
    checkpoint: str
    metadata_sha256: str
    cluster: str
    allow_temporary: bool = False

    @property
    def identity(self) -> str:
        step = self.checkpoint.rsplit("-", 1)[-1]
        return f"dna-exp593-anneal-nativeaudit-{self.trial}-step-{step}-v1"

    def __post_init__(self) -> None:
        if self.trial not in TRIALS or self.cluster not in (
            "cw-rno2a",
            "cw-us-east-02a",
        ):
            raise ValueError("Unapproved annealing recipe or cluster")
        prefix = (
            f"{OWNER}/annealing/training/{TRIALS[self.trial].run_id}/checkpoints/step-"
        )
        if not self.checkpoint.startswith(prefix):
            raise ValueError("Native audit checkpoint escaped the production namespace")
        if len(self.metadata_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.metadata_sha256
        ):
            raise ValueError("Expected an exact lowercase metadata SHA-256")


def run(config: AuditConfig) -> None:
    recipe = TRIALS[config.trial]
    mp = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    trainer = TrainerConfig(
        id=config.identity,
        seed=0,
        train_batch_size=64,
        num_train_steps=ORIGINAL_UPDATES,
        require_accelerator=True,
        mp=mp,
        use_explicit_mesh_axes=True,
        watch=WatchConfig(interval=0, watch_targets=[]),
        tracker=NoopConfig(),
    )
    trainer.initialize()
    if jax.device_count() != 8 or not all(
        "H100" in d.device_kind for d in jax.devices()
    ):
        raise ValueError("Native audit requires exactly one eight-H100 node")
    raw = (StoragePath(config.checkpoint) / "metadata.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != config.metadata_sha256:
        raise ValueError("Selected checkpoint metadata changed")
    metadata = json.loads(raw)
    binding = metadata["exp593_bindings"]
    if binding["run_id"] != recipe.run_id or binding["validation_updates"] is not None:
        raise ValueError("Selected checkpoint is not bound to production")
    for name, expected in binding["source_code_sha256"].items():
        if (
            hashlib.sha256(files("exp593_dna").joinpath(name).read_bytes()).hexdigest()
            != expected
        ):
            raise ValueError(f"Scientific source code differs: {name}")
    clock = anneal_state.validate_metadata(metadata, recipe, binding)
    if not config.allow_temporary and (
        metadata["is_temporary"] is not False
        or clock.stage.input_tokens != recipe.horizon
    ):
        raise ValueError(
            "The selected checkpoint is not the permanent exact-horizon final"
        )
    mesh = replicated_mesh()
    model = replace(
        local_model(d768_config("pretrained")),
        compact_input_lookup_ids=(),
        compact_input_lookup_one_hot=False,
    )
    optimizer = match_gradient_placement(
        optimizer_config(
            reference_tokens_per_update=REFERENCE_TOKENS,
            training_tokens=ORIGINAL_TOKENS,
        ).build(ORIGINAL_UPDATES, ns4d_impl=sharded_expert_ns(mesh))
    )
    with jax.set_mesh(mesh):
        template = eqx.filter_jit(
            lambda: fresh_state(
                Transformer.init(model, key=jax.random.PRNGKey(0)),
                optimizer,
                mp,
                offload=False,
                retain_master=True,
            )
        )()
        template, _ = partition_expert_optimizer_state(template, mesh)
        state, restored_clock = anneal_state.restore(
            template, config.checkpoint, recipe, mesh, binding
        )
        jax.block_until_ready(state)
        assert_partitioned_optimizer_state(state, mesh)
        if restored_clock != clock or not bool(state_all_finite(state)):
            raise ValueError(
                "Restored native optimizer/model state is nonfinite or its clock differs"
            )
        if not bool(model_matches_fp32_master(state.params, state.master_params)):
            raise ValueError("Restored compute weights differ from their FP32 masters")
        leaves = jax.tree.leaves(state)
        if len(leaves) != 128 or int(state.step) != metadata["step"]:
            raise ValueError("Restored native array count or global step differs")
    if jax.process_index() == 0:
        report = {
            "trial": config.trial,
            "checkpoint": config.checkpoint,
            "metadata_sha256": digest,
            "native_state_array_leaves": len(leaves),
            "all_arrays_read_and_finite": True,
            "compute_matches_fp32_master_cast": True,
            "optimizer_count_matches_global_step": True,
            "clock": asdict(clock),
            "global_step": metadata["step"],
            "audit_cluster": config.cluster,
            "audit_gpus": 8,
            "final": not config.allow_temporary,
        }
        data = (
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        path = Path("/tmp/exp593-native-audit.json")
        path.write_bytes(data)
        key = (
            f"MarinDNA/exp593_augmented_dna/annealing/native_audits/"
            f"{recipe.run_id}/step-{metadata['step']}-{digest}.json"
        )
        receipt = put_immutable(
            coreweave_client(), path, key, hashlib.sha256(data).hexdigest(), len(data)
        )
        print(
            json.dumps({"native_audit": report, "immutable_receipt": receipt}),
            flush=True,
        )


def dispatch(config: AuditConfig) -> None:
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=False,
        moe_implementation="scatter",
        remat_mode="recompute_all",
        processes_per_task=8,
    )
    resources = ResourceConfig.with_gpu(
        "H100", count=8, replicas=1, cpu=32, ram="600g", disk="900g"
    )
    env = resolve_training_env(
        {
            **_forwarded_env_vars(),
            "MARIN_PREFIX": "s3://marin-us-east-02a/marin",
            **{name: os.environ[name] for name in ("CW_KEY_ID", "CW_KEY_SECRET")},
        },
        resources,
    )
    job = current_client().submit(
        JobRequest(
            name=f"train-{config.identity}",
            entrypoint=Entrypoint.from_callable(run, args=[config]),
            resources=resources,
            processes_per_task=8,
            environment=create_environment(env_vars=env, extras=["gpu"]),
            priority=priority_band_value("batch"),
            timeout=Duration.from_hours(3),
            max_retries_failure=0,
            max_task_failures=0,
        )
    )
    job.wait(raise_on_failure=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", choices=tuple(TRIALS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument(
        "--cluster", choices=("cw-rno2a", "cw-us-east-02a"), required=True
    )
    parser.add_argument("--allow-temporary", action="store_true")
    dispatch(AuditConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
