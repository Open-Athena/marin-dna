"""Bounded, isolated full-replica data-parallel training and EP comparisons."""

import argparse
import dataclasses
import gc
import gzip
import hashlib
import inspect
import logging
import os
import re
import statistics
import string
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import equinox as eqx
import jax
import jmp
import numpy as np
from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.rpc.proto_display import priority_band_value
from jax.experimental import multihost_utils
from levanter import tracker
from levanter.callbacks.watch import WatchConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.training.training import resolve_training_env
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from exp586_moe.config import GLOBAL_BATCH_SIZE, SEQ_LEN, TOKENS_PER_UPDATE
from exp586_moe.corpus import distributed_batch, open_dataset
from exp586_moe.data import pretrained_dna_tokenizer, tokenizer_digest
from exp586_moe.data_parallel import (
    assert_replicated_state,
    local_model,
    replicated_mesh,
)
from exp586_moe.perf2 import DEADLINE, PerfModelConfig, remaining_seconds
from exp586_moe.routing_probe import routing_diagnostics
from exp586_moe.schedule import optimizer_config, set_token_learning_rates
from exp586_moe.scientific_equivalence import (
    SOURCE_CHECKPOINT,
    SOURCE_METADATA_SHA256,
    SOURCE_SCIENTIFIC_CONFIG_SHA256,
)
from exp586_moe.state import (
    fresh_state,
    restore_dna_checkpoint,
    save_dna_checkpoint,
    state_error_by_group,
    validate_restored_training_state,
)
from exp586_moe.train import (
    TrainingConfig,
    ValidationSample,
    config_json,
    evaluate,
    training_mesh,
)
from experiments.grug.dispatch import _forwarded_env_vars
from experiments.grug.moe_hero_ep import grugmuon_hero
from experiments.grug.moe_hero_ep.model import Transformer
from experiments.grug.moe_hero_ep.train import (
    MasterParamMode,
    _apply_hero_ep_runtime_defaults,
    _make_train_step,
    grug_trainer_mesh_config,
)

ARTIFACT_ROOT = (
    "s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp586_plantcad2_moe/dp-20261031"
)
VARIANTS = ("scatter", "sonic")


@dataclasses.dataclass(frozen=True)
class DPConfig:
    variant: str
    cluster: str
    attempt: str = "a1"
    updates: int = 100
    verification: bool = False
    batch_size: int = 8
    nodes: int = 1
    ep_reference: str = "best"
    end_to_end: bool = False
    end_to_end_updates: int = 60
    end_to_end_order: str = "dp-first"

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS or self.cluster not in (
            "cw-rno2a",
            "cw-us-east-02a",
        ):
            raise ValueError("Unsupported DP backend or H100 target")
        if self.nodes not in (1, 8) or self.batch_size not in (8, 64):
            raise ValueError("Use one or eight H100 nodes and batch 8 or 64")
        if self.nodes == 8 and self.batch_size != 64:
            raise ValueError("The 64-GPU benchmark requires the production batch")
        if self.ep_reference not in ("best", "live"):
            raise ValueError("Choose the best tested or live EP reference")
        if self.end_to_end and not (self.verification and self.batch_size == 64):
            raise ValueError("End-to-end comparison requires paired production batches")
        if not 30 <= self.end_to_end_updates <= 500:
            raise ValueError("Use 30 through 500 end-to-end updates per backend")
        if self.end_to_end_order not in ("dp-first", "ep-first", "balanced"):
            raise ValueError("Choose an explicit end-to-end backend order")
        if not 10 <= self.updates <= 200 or not self.attempt.replace("-", "").isalnum():
            raise ValueError("Invalid bounded profile length or identity")

    @property
    def run_id(self) -> str:
        suffix = "-b64" if self.batch_size == 64 else ""
        suffix += "-verify" if self.verification else ""
        suffix += f"-n{self.nodes}-{self.ep_reference}" if self.nodes != 1 else ""
        return f"exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-dp-{self.variant}-{self.attempt}{suffix}"

    @property
    def job_name(self) -> str:
        suffix = "-b64" if self.batch_size == 64 else ""
        suffix += "-verify" if self.verification else ""
        suffix += f"-n{self.nodes}-{self.ep_reference}" if self.nodes != 1 else ""
        return f"exp586-dp-{self.variant}-{self.attempt}{suffix}-20261031"

    @property
    def artifact_root(self) -> str:
        return f"{ARTIFACT_ROOT}/{self.run_id}"


def aligned_state(reference: Any, candidate: Any) -> Any:
    """Align equal-shaped arrays to the reference mesh solely for comparison."""
    arrays = iter(jax.tree.leaves(candidate))

    def put(leaf: jax.Array) -> jax.Array:
        value = next(arrays)
        if value.shape != leaf.shape or value.dtype != leaf.dtype:
            raise ValueError("DP changed a state shape or dtype")
        return jax.device_put(value, leaf.sharding)

    result = jax.tree.map(put, reference)
    if next(arrays, None) is not None:
        raise ValueError("DP changed the state structure")
    return result


def _errors(
    reference: Any, candidate: Any, mesh: jax.sharding.Mesh
) -> dict[str, float]:
    with jax.set_mesh(mesh):
        aligned = aligned_state(reference, candidate)
        errors = {
            k: float(v) for k, v in state_error_by_group(reference, aligned).items()
        }
        del aligned
    return errors


def _run(config: DPConfig) -> None:
    if remaining_seconds() < 120:
        raise RuntimeError("Isolated experiment deadline reached")
    production = TrainingConfig(
        "pretrained", 1.0, config.cluster, config.nodes, stage="t1"
    )
    mp = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    TrainerConfig(
        id=config.run_id,
        seed=0,
        train_batch_size=config.batch_size,
        num_train_steps=config.updates,
        require_accelerator=True,
        mp=mp,
        use_explicit_mesh_axes=True,
        mesh=grug_trainer_mesh_config(1),
        watch=WatchConfig(interval=0, watch_targets=[]),
        tracker=WandbConfig(
            entity="eric-czech",
            project="marin",
            name=config.run_id,
            group="exp586-plantcad2-moe-dp",
            tags=["dna-exp586", "profiling", "data-parallel", "H100", config.cluster],
            save_code=False,
            background=False,
            replicate_path=config.artifact_root,
        ),
    ).initialize()
    if jax.device_count() != 8 * config.nodes or any(
        "H100" not in d.device_kind for d in jax.devices()
    ):
        raise ValueError("Unexpected H100 gang size")
    if jax.process_index() == 0 and tracker.get_tracker("wandb").run.resumed:
        raise ValueError("Profiles require a fresh attempt ID after interruption")
    tracker.log_configuration(config)
    tracker.log_summary(
        {
            "profiling/phase": "restoring",
            "profiling/deadline": DEADLINE.isoformat(),
            "profiling/source_checkpoint": SOURCE_CHECKPOINT,
            "profiling/data_preloaded": True,
            "profiling/batch_size": config.batch_size,
            "profiling/processes": jax.process_count(),
            "profiling/nodes": config.nodes,
            "profiling/gpus": jax.device_count(),
            "profiling/ep_reference": config.ep_reference,
            "profiling/source_sha256": hashlib.sha256(
                StoragePath(__file__).read_bytes()
            ).hexdigest(),
            "profiling/model_sha256": hashlib.sha256(
                StoragePath(inspect.getfile(Transformer)).read_bytes()
            ).hexdigest(),
            "profiling/optimizer_source_sha256": hashlib.sha256(
                StoragePath(inspect.getfile(grugmuon_hero)).read_bytes()
            ).hexdigest(),
        }
    )
    tok = pretrained_dna_tokenizer()
    tok_digest = tokenizer_digest(tok)
    rows = [tok.encode(c).ids for c in string.ascii_lowercase]
    if any(len(ids) != 1 for ids in rows) or len({ids[0] for ids in rows}) != 26:
        raise ValueError("Unexpected character tokenizer")
    lookup_ids = tuple(sorted(ids[0] for ids in rows))
    # asdict recurses into RotaryConfig, so retain the original typed field values.
    values = {
        f.name: getattr(production.model, f.name)
        for f in dataclasses.fields(production.model)
    }
    candidate_model = PerfModelConfig(
        **values,
        compact_input_lookup_ids=lookup_ids,
        compact_input_lookup_one_hot=True,
        expert_compute_fast_rows=2048,
    )
    ep_model = (
        PerfModelConfig(**values) if config.ep_reference == "live" else candidate_model
    )
    model = local_model(candidate_model, implementation=config.variant)
    opt_cfg = optimizer_config(
        reference_tokens_per_update=TOKENS_PER_UPDATE,
        training_tokens=production.optimizer_reference_tokens,
        multiplier=1.0,
    )
    optimizer = opt_cfg.build(production.target_updates)
    mesh = replicated_mesh()
    ep_mesh = training_mesh(production)
    tracker.log_summary(
        {
            "profiling/mesh_shape": dict(mesh.shape),
            "profiling/model_config": dataclasses.asdict(model),
            "profiling/ep_model_config": dataclasses.asdict(ep_model),
        }
    )

    def restore(cfg: Any, active_mesh: jax.sharding.Mesh) -> tuple[Any, Any]:
        with jax.set_mesh(active_mesh):
            weights = eqx.filter_jit(Transformer.init)(cfg, key=jax.random.key(0))
            exemplar = eqx.filter_jit(
                lambda w: fresh_state(
                    w, optimizer, mp, offload=False, retain_master=True
                )
            )(weights)
            state, clock = restore_dna_checkpoint(
                exemplar,
                SOURCE_CHECKPOINT,
                active_mesh,
                data_seed=0,
                tokenizer_sha256=tok_digest,
                scientific_config_sha256=SOURCE_SCIENTIFIC_CONFIG_SHA256,
                metadata_sha256=SOURCE_METADATA_SHA256,
            )
            validate_restored_training_state(state, clock)
            return state, clock

    state, clock = restore(model, mesh)
    tracker.log_summary(
        {"profiling/replicated_state_bytes": assert_replicated_state(state)}
    )
    controls = []
    if config.verification:
        for _ in range(2):
            control, c_clock = restore(ep_model, ep_mesh)
            if c_clock != clock:
                raise ValueError("Reference clocks differ")
            initial = _errors(control, state, ep_mesh)
            if any(x != 0 for x in initial.values()):
                raise ValueError("Initial weights/optimizer/router state differ")
            controls.append(control)
        tracker.log_summary({"correctness/initial_state_max_abs": 0})
    step_fn = _make_train_step(
        optimizer,
        mp,
        z_loss_weight=1e-4,
        ema_beta=None,
        offload_opt_state=False,
        master_param_mode=MasterParamMode.FP32_DEVICE,
        watch_config=None,
    )
    dataset = open_dataset("train", 0)
    tracker.log_summary({"profiling/phase": "preloading"})
    with jax.set_mesh(mesh):
        batches = [
            distributed_batch(
                dataset,
                tok,
                mesh,
                clock.examples + i * GLOBAL_BATCH_SIZE,
                config.batch_size,
                augment=True,
            )
            for i in range(config.updates)
        ]
        jax.block_until_ready(batches)
    ep_batches = []
    if controls:
        for batch in batches:
            ep_batches.append(
                jax.tree.map(
                    lambda a: jax.device_put(
                        a, jax.sharding.NamedSharding(ep_mesh, a.sharding.spec)
                    ),
                    batch,
                )
            )
        jax.block_until_ready(ep_batches)
    multihost_utils.sync_global_devices("dp-preloaded")

    def timed(
        s: Any, batch: Any, active_mesh: jax.sharding.Mesh
    ) -> tuple[Any, dict, float]:
        with jax.set_mesh(active_mesh):
            s = dataclasses.replace(
                s,
                opt_state=set_token_learning_rates(
                    s.opt_state, opt_cfg, production.schedule, clock
                ),
            )
            begin = time.monotonic()
            s, metrics, _ = step_fn(s, batch)
            jax.block_until_ready((s, metrics))
            return s, metrics, time.monotonic() - begin

    tracker.log_summary({"profiling/phase": "compiling"})
    times = []
    ep_times = []
    candidate_diff = []
    control_diff = []
    for index, batch in enumerate(batches, 1):
        if remaining_seconds() < 90:
            raise RuntimeError("Experiment hard deadline reached")
        if controls:
            controls[0], ma, _ = timed(controls[0], ep_batches[index - 1], ep_mesh)
            if index % 2:
                state, metrics, elapsed = timed(state, batch, mesh)
                controls[1], mb, control_elapsed = timed(
                    controls[1], ep_batches[index - 1], ep_mesh
                )
            else:
                controls[1], mb, control_elapsed = timed(
                    controls[1], ep_batches[index - 1], ep_mesh
                )
                state, metrics, elapsed = timed(state, batch, mesh)
        else:
            state, metrics, elapsed = timed(state, batch, mesh)
        if index == 1:
            tracker.log_summary(
                {"profiling/phase": "training", "profiling/first_step_seconds": elapsed}
            )
            with jax.set_mesh(mesh):
                compiled = step_fn.lower(state, batch).compile()
                hlo = compiled.as_text()
            forbidden = sorted(
                set(
                    re.findall(
                        r"\b(all-to-all|all-gather(?:-start|-done)?|reduce-scatter(?:-start|-done)?)\(",
                        hlo,
                    )
                )
            )
            tracker.log_summary(
                {
                    "profiling/forbidden_collectives": forbidden,
                    "profiling/all_reduce_hlo_lines": sum(
                        "all-reduce(" in line for line in hlo.splitlines()
                    ),
                }
            )
            if forbidden:
                raise ValueError(
                    f"DP executable contains model-parallel collectives: {forbidden}"
                )
            if jax.process_index() == 0:
                (
                    StoragePath(config.artifact_root) / "train-step.hlo.txt.gz"
                ).write_bytes(gzip.compress(hlo.encode()))
            del compiled, hlo
        values = routing_diagnostics(
            metrics, batch_size=config.batch_size, model_config=model
        )
        loss = float(metrics["train/loss"])
        if values["moe/drop_fraction"] != 0:
            raise ValueError("Local backend dropped assignments")
        if not np.isfinite(loss):
            raise ValueError("Nonfinite loss")
        if controls:
            ca = float(ma["train/loss"])
            cb = float(mb["train/loss"])
            candidate_diff.append(loss - ca)
            control_diff.append(cb - ca)
            for reference in (ma, mb):
                if (
                    routing_diagnostics(
                        reference, batch_size=config.batch_size, model_config=ep_model
                    )["moe/drop_fraction"]
                    != 0
                ):
                    raise ValueError(
                        "EP reference drops invalidate zero-drop equivalence comparison"
                    )
            values.update(
                {
                    "correctness/control_loss": ca,
                    "correctness/control_b_loss": cb,
                    "correctness/candidate_loss_difference": loss - ca,
                    "correctness/control_loss_difference": cb - ca,
                    "throughput/control_step_seconds": control_elapsed,
                }
            )
            if index in (1, 10, config.updates):
                tracker.log_summary(
                    {
                        f"correctness/at_{index}/candidate": _errors(
                            controls[0], state, ep_mesh
                        ),
                        f"correctness/at_{index}/control": _errors(
                            controls[0], controls[1], ep_mesh
                        ),
                    }
                )
        values.update(
            {
                "train/loss": loss,
                "train/step": index,
                "profiling/source_step": clock.updates,
                "train/input_tokens": index * config.batch_size * SEQ_LEN,
                "throughput/step_seconds": elapsed,
                "throughput/tokens_per_second": config.batch_size * SEQ_LEN / elapsed,
                "run_progress": index / config.updates,
            }
        )
        stats = jax.local_devices()[0].memory_stats() or {}
        values.update(
            {
                f"memory/{k}": v
                for k, v in stats.items()
                if k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
            }
        )
        tracker.log(values, step=index)
        if index >= min(10, config.updates):
            times.append(elapsed)
            if controls:
                ep_times.append(control_elapsed)
        clock = clock.advance(
            input_tokens=TOKENS_PER_UPDATE,
            loss_targets=GLOBAL_BATCH_SIZE * (SEQ_LEN - 1),
            examples=GLOBAL_BATCH_SIZE,
        )
    assert_replicated_state(state)
    summary = {
        "profiling/phase": "finishing",
        "profiling/updates": config.updates,
        "profiling/mean_step_seconds": statistics.fmean(times),
        "profiling/median_step_seconds": statistics.median(times),
        "profiling/actual_tokens": config.updates * config.batch_size * SEQ_LEN,
        "profiling/replicated_state_verified": True,
        "profiling/mesh_shape": dict(mesh.shape),
        "profiling/forbidden_collectives": [],
    }
    if controls:
        summary["profiling/paired_ep_mean_step_seconds"] = statistics.fmean(ep_times)
        summary["profiling/paired_throughput_gain"] = (
            statistics.fmean(ep_times) / statistics.fmean(times) - 1
        )
        for label, diffs in (("candidate", candidate_diff), ("control", control_diff)):
            d = np.asarray(diffs)
            summary.update(
                {
                    f"correctness/{label}_loss_rmse": float(np.sqrt(np.mean(d * d))),
                    f"correctness/{label}_loss_mae": float(np.mean(np.abs(d))),
                    f"correctness/{label}_loss_signed_mean": float(np.mean(d)),
                }
            )
    if config.end_to_end:
        # Use the production input/prefetch/metrics path on both backends. Keep
        # the real batch and LR/data clocks; preloaded proxy timings above are
        # reported separately. Compilation is already warm for both backends.
        reference = controls[0]
        controls.clear()
        gc.collect()
        base_clock = clock
        e2e_updates = config.end_to_end_updates
        validation = ValidationSample()

        def end_to_end(
            current: Any, active_mesh: jax.sharding.Mesh, label: str, offset: int
        ) -> tuple[Any, Any, float]:
            active_clock = base_clock
            elapsed_samples = []

            def inputs() -> Any:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        distributed_batch,
                        dataset,
                        tok,
                        active_mesh,
                        base_clock.examples,
                        GLOBAL_BATCH_SIZE,
                        augment=True,
                    )
                    for i in range(e2e_updates):
                        value = future.result()
                        if i + 1 < e2e_updates:
                            future = pool.submit(
                                distributed_batch,
                                dataset,
                                tok,
                                active_mesh,
                                base_clock.examples + (i + 1) * GLOBAL_BATCH_SIZE,
                                GLOBAL_BATCH_SIZE,
                                augment=True,
                            )
                        yield value

            iterator = iter(inputs())
            measured_start = None
            with jax.set_mesh(active_mesh):
                for i in range(e2e_updates):
                    if remaining_seconds() < 120:
                        raise RuntimeError("Experiment hard deadline reached")
                    if i == 5:
                        multihost_utils.sync_global_devices(f"e2e-{label}-start")
                        measured_start = time.monotonic()
                    begin = time.monotonic()
                    batch = next(iterator)
                    current = dataclasses.replace(
                        current,
                        opt_state=set_token_learning_rates(
                            current.opt_state,
                            opt_cfg,
                            production.schedule,
                            active_clock,
                        ),
                    )
                    current, metrics, _ = step_fn(current, batch)
                    jax.block_until_ready((current, metrics))
                    elapsed = time.monotonic() - begin
                    values = routing_diagnostics(
                        metrics,
                        batch_size=GLOBAL_BATCH_SIZE,
                        model_config=model if label == "dp" else ep_model,
                    )
                    if values["moe/drop_fraction"] != 0 or not np.isfinite(
                        float(metrics["train/loss"])
                    ):
                        raise ValueError("Invalid end-to-end loss or routing")
                    if any(not np.isfinite(float(value)) for value in values.values()):
                        raise ValueError("Nonfinite end-to-end metrics")
                    tracker.log(
                        {f"benchmark/{label}/{k}": v for k, v in values.items()}
                        | {
                            f"benchmark/{label}/step_seconds": elapsed,
                            f"benchmark/{label}/loss": float(metrics["train/loss"]),
                        },
                        step=offset + i + 1,
                    )
                    tracker.log_summary({"profiling/phase": f"end_to_end_{label}"})
                    # The production loop coordinates its checkpoint-due decision
                    # on every update, including when no write is due. Include
                    # that host synchronization rather than timing a lighter loop.
                    due = int(
                        multihost_utils.broadcast_one_to_all(
                            np.asarray(0, dtype=np.int32)
                        )
                    )
                    if due:
                        raise AssertionError(
                            "Bounded loop unexpectedly requested a checkpoint"
                        )
                    if i >= 5:
                        elapsed_samples.append(elapsed)
                    active_clock = active_clock.advance(
                        input_tokens=TOKENS_PER_UPDATE,
                        loss_targets=GLOBAL_BATCH_SIZE * (SEQ_LEN - 1),
                        examples=GLOBAL_BATCH_SIZE,
                    )
                multihost_utils.sync_global_devices(f"e2e-{label}-end")
                duration = time.monotonic() - measured_start
                summary.update(
                    {
                        f"benchmark/{label}/measured_updates": len(elapsed_samples),
                        f"benchmark/{label}/wall_seconds": duration,
                        f"benchmark/{label}/wall_tokens_per_second": len(
                            elapsed_samples
                        )
                        * TOKENS_PER_UPDATE
                        / duration,
                        f"benchmark/{label}/mean_step_seconds": statistics.fmean(
                            elapsed_samples
                        ),
                    }
                )
            return current, active_clock, duration

        if config.end_to_end_order == "dp-first":
            state, clock, dp_wall = end_to_end(state, mesh, "dp", config.updates)
            reference, reference_clock, ep_wall = end_to_end(
                reference, ep_mesh, "ep", config.updates + e2e_updates
            )
        else:
            reference, reference_clock, ep_wall = end_to_end(
                reference, ep_mesh, "ep", config.updates
            )
            state, clock, dp_wall = end_to_end(
                state, mesh, "dp", config.updates + e2e_updates
            )
        if clock != reference_clock:
            raise ValueError("End-to-end backend clocks differ")
        if config.end_to_end_order == "balanced":
            # Use both orders on consecutive equal streams, carrying each
            # backend's own training state forward between the two windows.
            first = {
                k: v
                for k, v in summary.items()
                if k.startswith(("benchmark/dp/", "benchmark/ep/"))
            }
            base_clock = clock
            state, clock, dp_second = end_to_end(
                state, mesh, "dp", config.updates + 2 * e2e_updates
            )
            reference, reference_clock, ep_second = end_to_end(
                reference, ep_mesh, "ep", config.updates + 3 * e2e_updates
            )
            if clock != reference_clock:
                raise ValueError("Counterbalanced backend clocks differ")
            summary.update(
                {
                    k.replace("benchmark/", "benchmark/order1/", 1): v
                    for k, v in summary.copy().items()
                    if k.startswith(("benchmark/dp/", "benchmark/ep/"))
                }
            )
            summary.update(
                {
                    k.replace("benchmark/", "benchmark/order0/", 1): v
                    for k, v in first.items()
                }
            )
            dp_wall += dp_second
            ep_wall += ep_second
            for label, duration in (("dp", dp_wall), ("ep", ep_wall)):
                count = 2 * (e2e_updates - 5)
                mean = (
                    first[f"benchmark/{label}/mean_step_seconds"]
                    + summary[f"benchmark/{label}/mean_step_seconds"]
                ) / 2
                summary.update(
                    {
                        f"benchmark/{label}/wall_seconds": duration,
                        f"benchmark/{label}/measured_updates": count,
                        f"benchmark/{label}/wall_tokens_per_second": count
                        * TOKENS_PER_UPDATE
                        / duration,
                        f"benchmark/{label}/mean_step_seconds": mean,
                    }
                )
        summary["benchmark/wall_throughput_gain"] = ep_wall / dp_wall - 1
        for label, current, active_mesh in (
            ("dp", state, mesh),
            ("ep", reference, ep_mesh),
        ):
            tracker.log_summary({"profiling/phase": f"evaluating_{label}"})
            with jax.set_mesh(active_mesh):
                for pass_name in ("cold", "warm"):
                    begin = time.monotonic()
                    values = evaluate(current, validation, tok, active_mesh)
                    summary[f"benchmark/{label}/eval_{pass_name}_seconds"] = (
                        time.monotonic() - begin
                    )
                    summary[f"benchmark/{label}/eval_{pass_name}_metrics"] = values
            tracker.log_summary(summary)
        tracker.log_summary({"profiling/phase": "checkpointing_ep"})
        ep_path = config.artifact_root + f"/ep-checkpoint/step-{clock.updates}"
        with jax.set_mesh(ep_mesh):
            begin = time.monotonic()
            save_dna_checkpoint(
                reference,
                clock,
                ep_path,
                data_seed=0,
                tokenizer_sha256=tok_digest,
                scientific_config_sha256=SOURCE_SCIENTIFIC_CONFIG_SHA256,
                runtime_moe_implementation=ep_model.moe_implementation,
                runtime_metadata={
                    "experimental_dp_reference": True,
                    "not_for_production": True,
                },
                is_temporary=True,
            )
            summary["benchmark/ep/checkpoint_seconds"] = time.monotonic() - begin
        del reference
        gc.collect()
    if config.verification or config.batch_size == 64:
        # Serialize only into this experiment's temporary namespace, never the live owner.
        controls.clear()
        gc.collect()
        tracker.log_summary({"profiling/phase": "checkpointing"})
        path = config.artifact_root + f"/checkpoint/step-{clock.updates}"
        with jax.set_mesh(mesh):
            save_start = time.monotonic()
            save_dna_checkpoint(
                state,
                clock,
                path,
                data_seed=0,
                tokenizer_sha256=tok_digest,
                scientific_config_sha256=SOURCE_SCIENTIFIC_CONFIG_SHA256,
                runtime_moe_implementation=config.variant,
                runtime_metadata={
                    "experimental_dp": True,
                    "proxy_batch_size": config.batch_size,
                    "reference_batch_size": 64,
                    "not_for_production": True,
                },
                is_temporary=True,
            )
            summary["benchmark/dp/checkpoint_seconds"] = time.monotonic() - save_start
            restored, r_clock = restore_dna_checkpoint(
                state,
                path,
                mesh,
                data_seed=0,
                tokenizer_sha256=tok_digest,
                scientific_config_sha256=SOURCE_SCIENTIFIC_CONFIG_SHA256,
            )
            validate_restored_training_state(restored, r_clock)
            if r_clock != clock:
                raise ValueError("Checkpoint changed token/data clocks")
            replay = _errors(state, restored, mesh)
            if any(v != 0 for v in replay.values()):
                raise ValueError("Checkpoint serialization changed state")
            assert_replicated_state(restored)
            summary.update(
                {
                    "correctness/checkpoint_roundtrip_max_abs": 0,
                    "correctness/checkpoint": path,
                }
            )
            replay_batch = distributed_batch(
                dataset, tok, mesh, clock.examples, config.batch_size, augment=True
            )
            state, direct_metrics, _ = timed(state, replay_batch, mesh)
            restored, replay_metrics, _ = timed(restored, replay_batch, mesh)
            next_clock = clock.advance(
                input_tokens=TOKENS_PER_UPDATE,
                loss_targets=GLOBAL_BATCH_SIZE * (SEQ_LEN - 1),
                examples=GLOBAL_BATCH_SIZE,
            )
            validate_restored_training_state(state, next_clock)
            validate_restored_training_state(restored, next_clock)
            summary["correctness/replay_loss_abs"] = abs(
                float(direct_metrics["train/loss"])
                - float(replay_metrics["train/loss"])
            )
            summary["correctness/replay_state_errors"] = _errors(state, restored, mesh)
            if summary["correctness/replay_loss_abs"] > 1e-3:
                raise ValueError("Checkpoint replay changed forward loss")
            del restored
    gc.collect()
    summary["profiling/phase"] = "complete"
    if jax.process_index() == 0:
        (StoragePath(config.artifact_root) / "result.json").write_text(
            config_json(summary)
        )
    tracker.log_summary(summary)
    tracker.current_tracker().finish()


def run_profile(config: DPConfig) -> None:
    try:
        _run(config)
    except Exception as error:
        if jax.process_index() == 0:
            try:
                tracker.log_summary(
                    {
                        "profiling/phase": "failed",
                        "profiling/error": f"{type(error).__name__}: {error}",
                    }
                )
                tracker.get_tracker("wandb").run.finish(exit_code=1)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Could not report isolated DP failure"
                )
        raise


def dispatch(config: DPConfig) -> None:
    if remaining_seconds() < 300:
        raise RuntimeError("No time remains for another isolated job")
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=False,
        moe_implementation=config.variant,
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
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
            "WANDB_ENTITY": "eric-czech",
            "WANDB_PROJECT": "marin",
        },
        resources,
    )
    job = current_client().submit(
        JobRequest(
            name=f"profile-{config.run_id}",
            entrypoint=Entrypoint.from_callable(run_profile, args=[config]),
            resources=resources,
            processes_per_task=8,
            environment=create_environment(env_vars=env, extras=["gpu"]),
            priority=priority_band_value("batch"),
            timeout=Duration.from_seconds(min(2700, remaining_seconds())),
            max_retries_failure=0,
            max_task_failures=0,
        )
    )
    job.wait(raise_on_failure=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--cluster", choices=["cw-rno2a", "cw-us-east-02a"], required=True
    )
    parser.add_argument("--attempt", default="a1")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--verification", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nodes", type=int, choices=[1, 8], default=1)
    parser.add_argument("--ep-reference", choices=["best", "live"], default="best")
    parser.add_argument("--end-to-end", action="store_true")
    parser.add_argument("--end-to-end-updates", type=int, default=60)
    parser.add_argument(
        "--end-to-end-order",
        choices=["dp-first", "ep-first", "balanced"],
        default="dp-first",
    )
    dispatch(DPConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
