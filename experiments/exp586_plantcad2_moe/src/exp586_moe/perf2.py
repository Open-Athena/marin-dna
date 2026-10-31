"""Isolated, deadline-bounded second-round runtime profiling for exp586."""

import argparse
import dataclasses
import datetime as dt
import gc
import gzip
import hashlib
import inspect
import logging
import os
import statistics
import string
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.rpc.proto_display import priority_band_value
from jax.experimental import multihost_utils
from levanter import tracker
from levanter.callbacks.profiler import (
    DEFAULT_XPROF_SERVICE_URL,
    ProfileOptionsConfig,
    profile,
    xprof_viewer_url,
)
from levanter.callbacks.watch import WatchConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.training.training import resolve_training_env
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from exp586_moe.config import GLOBAL_BATCH_SIZE, SEQ_LEN, TOKENS_PER_UPDATE
from exp586_moe.corpus import distributed_batch, open_dataset
from exp586_moe.data import pretrained_dna_tokenizer, tokenizer_digest
from exp586_moe.routing_probe import routing_diagnostics
from exp586_moe.schedule import optimizer_config, set_token_learning_rates
from exp586_moe.scientific_equivalence import (
    SOURCE_CHECKPOINT,
    SOURCE_METADATA_SHA256,
    SOURCE_SCIENTIFIC_CONFIG_SHA256,
    _assert_same_value_layout,
)
from exp586_moe.state import (
    fresh_state,
    restore_dna_checkpoint,
    state_error_by_group,
    validate_restored_training_state,
)
from exp586_moe.train import TrainingConfig, config_json, training_mesh
from experiments.grug.dispatch import _forwarded_env_vars
from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer
from experiments.grug.moe_hero_ep.train import (
    MasterParamMode,
    _apply_hero_ep_runtime_defaults,
    _make_train_step,
    grug_trainer_mesh_config,
)

DEADLINE = dt.datetime(2026, 10, 31, 18, 7, 14, tzinfo=dt.UTC)
ARTIFACT_ROOT = (
    "s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp586_plantcad2_moe/perf2-20261031"
)
VARIANTS = (
    "combined",
    "save-moe",
    "no-remat",
    "unroll2",
    "unroll4",
    "single-process",
    "overlap1",
    "waves1",
    "compact-lookup",
    "onehot-lookup",
    "metrics-batch",
    "trim-expert-rows",
    "onehot-trim",
    "onehot-trim-tiers",
)


@dataclasses.dataclass(frozen=True)
class PerfModelConfig(GrugModelConfig):
    """Isolated runtime knobs, absent from the current scientific binding."""

    layer_scan_unroll: int = 1
    compact_input_lookup_ids: tuple[int, ...] = ()
    compact_input_lookup_one_hot: bool = False
    expert_compute_fast_rows: int | tuple[int, ...] = 0


@dataclasses.dataclass(frozen=True)
class PerfConfig:
    variant: str
    cluster: str
    attempt: str = "a1"
    updates: int = 100
    verification: bool = False
    benchmark: bool = False

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError("Unknown profiling variant")
        if self.cluster not in ("cw-rno2a", "cw-us-east-02a"):
            raise ValueError("Unapproved H100 cluster")
        if not self.attempt.replace("-", "").isalnum():
            raise ValueError("Attempt requires an alphanumeric profiling identity")
        if not 30 <= self.updates <= 200:
            raise ValueError("Profiling is bounded to 30 through 200 updates")

    @property
    def run_id(self) -> str:
        suffix = "-verify" if self.verification else ""
        suffix += "-bench" if self.benchmark else ""
        return f"exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1-perf2-{self.variant}-{self.attempt}{suffix}"

    @property
    def job_name(self) -> str:
        suffix = "-verify" if self.verification else ""
        suffix += "-bench" if self.benchmark else ""
        return f"exp586-perf2-{self.variant}-{self.attempt}{suffix}-20261031"

    @property
    def processes(self) -> int:
        return 1 if self.variant == "single-process" else 8

    @property
    def artifact_root(self) -> str:
        return f"{ARTIFACT_ROOT}/{self.run_id}"

    @property
    def runtime_changes(self) -> dict[str, Any]:
        return {
            "save-moe": {"remat_mode": "save_moe"},
            "no-remat": {"remat_mode": "none"},
            "unroll2": {"layer_scan_unroll": 2},
            "unroll4": {"layer_scan_unroll": 4},
            "waves1": {"num_expert_waves": 1},
            "trim-expert-rows": {"expert_compute_fast_rows": 2048},
            "onehot-trim": {"expert_compute_fast_rows": 2048},
            "onehot-trim-tiers": {"expert_compute_fast_rows": (2048, 4096, 8192)},
        }.get(self.variant, {})


def remaining_seconds() -> int:
    return max(0, int((DEADLINE - dt.datetime.now(dt.UTC)).total_seconds()))


def _state_errors(left: Any, right: Any) -> dict[str, float]:
    return {
        key: float(value) for key, value in state_error_by_group(left, right).items()
    }


def _routing_metrics(metrics: dict, model: GrugModelConfig, *, batched: bool) -> dict:
    if not batched:
        return routing_diagnostics(metrics, batch_size=8, model_config=model)
    host_metrics = jax.device_get(metrics)
    result = routing_diagnostics(host_metrics, batch_size=8, model_config=model)
    result.update(
        (key, float(value))
        for key, value in host_metrics.items()
        if isinstance(value, np.ndarray) and value.ndim == 0
    )
    return result


@eqx.filter_jit
def _row_errors(left: jax.Array, right: jax.Array, rows: jax.Array) -> dict:
    a, b = left[rows].astype(jnp.float32), right[rows].astype(jnp.float32)
    delta = a - b
    return {
        "rmse": jnp.sqrt(jnp.mean(delta**2)),
        "max_abs": jnp.max(jnp.abs(delta)),
        "relative_l2": jnp.linalg.norm(delta) / jnp.maximum(jnp.linalg.norm(a), 1e-30),
    }


def _input_row_errors(left: Any, right: Any, rows: tuple[int, ...]) -> dict:
    result = {}
    left_leaves = jax.tree.leaves_with_path(left)
    right_leaves = jax.tree.leaves_with_path(right)
    for (path, a), (_, b) in zip(left_leaves, right_leaves, strict=True):
        name = jax.tree_util.keystr(path)
        if "token_embed" in name and eqx.is_array(a):
            result[name.lstrip(".")] = {
                key: float(value)
                for key, value in _row_errors(a, b, jnp.asarray(rows)).items()
            }
    if not result:
        raise ValueError("Verification did not find input-embedding state tensors")
    return result


def _run_profile(config: PerfConfig) -> None:
    if remaining_seconds() < 120:
        raise RuntimeError("Six-hour profiling deadline reached")
    production = TrainingConfig("pretrained", 1.0, config.cluster, 1, stage="t1")
    model_values = {
        field.name: getattr(production.model, field.name)
        for field in dataclasses.fields(production.model)
    }
    model_values.update(config.runtime_changes)
    model = PerfModelConfig(**model_values)
    mp = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    TrainerConfig(
        id=config.run_id,
        seed=0,
        train_batch_size=8,
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
            group="exp586-plantcad2-moe-perf2",
            tags=["dna-exp586", "profiling", "perf2", "H100", config.cluster],
            save_code=False,
            background=False,
            replicate_path=config.artifact_root,
        ),
    ).initialize()
    if jax.process_index() == 0 and tracker.get_tracker("wandb").run.resumed:
        raise ValueError(
            "Bounded profiles require a fresh attempt identity after interruption"
        )
    if jax.device_count() != 8 or any(
        "H100" not in device.device_kind for device in jax.devices()
    ):
        raise ValueError("Profiling requires exactly one eight-H100 node")
    tracker.log_configuration(config)
    tracker.log_summary(
        {
            "profiling/phase": "restoring",
            "profiling/deadline": DEADLINE.isoformat(),
            "profiling/source_checkpoint": SOURCE_CHECKPOINT,
            "profiling/processes": config.processes,
            "profiling/batch_size": 8,
            "profiling/data_preloaded": config.benchmark,
            "profiling/trace_captured": not config.benchmark,
            "profiling/runtime_changes": config.runtime_changes,
            "profiling/initial_model_config": dataclasses.asdict(model),
            "profiling/xla_flags": os.environ.get("XLA_FLAGS", ""),
            "profiling/source_fingerprint": hashlib.sha256(
                StoragePath(__file__).read_bytes()
            ).hexdigest(),
            "profiling/model_source_sha256": hashlib.sha256(
                StoragePath(inspect.getfile(Transformer)).read_bytes()
            ).hexdigest(),
        }
    )
    opt_config = optimizer_config(
        reference_tokens_per_update=TOKENS_PER_UPDATE,
        training_tokens=production.optimizer_reference_tokens,
        multiplier=1.0,
    )
    optimizer = opt_config.build(production.target_updates)
    tokenizer = pretrained_dna_tokenizer()
    tok_digest = tokenizer_digest(tokenizer)
    character_ids = [tokenizer.encode(c).ids for c in string.ascii_lowercase]
    if any(len(ids) != 1 for ids in character_ids):
        raise ValueError("Expected one existing vocabulary ID per lowercase character")
    lookup_ids = tuple(sorted(ids[0] for ids in character_ids))
    if len(set(lookup_ids)) != 26:
        raise ValueError("Expected distinct lowercase character vocabulary IDs")
    if config.variant in {
        "compact-lookup",
        "onehot-lookup",
        "onehot-trim",
        "onehot-trim-tiers",
    }:
        model = dataclasses.replace(
            model,
            compact_input_lookup_ids=lookup_ids,
            compact_input_lookup_one_hot=config.variant
            in {"onehot-lookup", "onehot-trim", "onehot-trim-tiers"},
        )
        tracker.log_summary({"profiling/compact_input_lookup_ids": lookup_ids})
    tracker.log_summary({"profiling/resolved_model_config": dataclasses.asdict(model)})
    if model.expert_compute_fast_rows:
        from exp586_moe.perf2_pooled import trimmed_mlp

        tracker.log_summary(
            {
                "profiling/pooled_source_sha256": hashlib.sha256(
                    StoragePath(inspect.getfile(trimmed_mlp)).read_bytes()
                ).hexdigest(),
            }
        )
    mesh = training_mesh(production)

    with jax.set_mesh(mesh):

        def restore(cfg: Any) -> tuple[Any, Any]:
            weights = eqx.filter_jit(Transformer.init)(cfg, key=jax.random.key(0))
            exemplar = eqx.filter_jit(
                lambda w: fresh_state(
                    w, optimizer, mp, offload=False, retain_master=True
                )
            )(weights)
            state, clock = restore_dna_checkpoint(
                exemplar,
                SOURCE_CHECKPOINT,
                mesh,
                data_seed=0,
                tokenizer_sha256=tok_digest,
                scientific_config_sha256=SOURCE_SCIENTIFIC_CONFIG_SHA256,
                metadata_sha256=SOURCE_METADATA_SHA256,
            )
            validate_restored_training_state(state, clock)
            return state, clock

        state, source_clock = restore(model)
        control_a, control_clock = restore(production.model)
        control_b, _ = restore(production.model)
        if source_clock != control_clock:
            raise ValueError("Source clocks differ")
        _assert_same_value_layout(control_a, state)
        _assert_same_value_layout(control_a, control_b)
        initial_errors = _state_errors(state, control_a)
        if any(v != 0.0 for v in initial_errors.values()):
            raise ValueError("Initial state differs")
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
        cached_batches = None
        if config.benchmark:
            tracker.log_summary({"profiling/phase": "preloading"})
            cached_batches = [
                distributed_batch(
                    dataset,
                    tokenizer,
                    mesh,
                    source_clock.examples + i * GLOBAL_BATCH_SIZE,
                    8,
                    augment=True,
                )
                for i in range(config.updates)
            ]
            jax.block_until_ready(cached_batches)
            multihost_utils.sync_global_devices("perf2-preloaded")
            first_batch = cached_batches[0]
        else:
            first_batch = distributed_batch(
                dataset, tokenizer, mesh, source_clock.examples, 8, augment=True
            )

        def scheduled(s: Any, clock: Any) -> Any:
            return dataclasses.replace(
                s,
                opt_state=set_token_learning_rates(
                    s.opt_state, opt_config, production.schedule, clock
                ),
            )

        def timed_step(s: Any, clock: Any, batch: Any, label: str, index: int):
            begun = time.monotonic()
            with jax.profiler.StepTraceAnnotation(label, step_num=index):
                s, metrics, _ = step_fn(scheduled(s, clock), batch)
                jax.block_until_ready((s, metrics))
            return s, metrics, time.monotonic() - begun

        tracker.log_summary({"profiling/phase": "compiling"})
        control_a, ca_metrics, _ = step_fn(
            scheduled(control_a, source_clock), first_batch
        )
        jax.block_until_ready((control_a, ca_metrics))
        control_b, cb_metrics, _ = step_fn(
            scheduled(control_b, source_clock), first_batch
        )
        jax.block_until_ready((control_b, cb_metrics))
        state, candidate_metrics, _ = step_fn(
            scheduled(state, source_clock), first_batch
        )
        jax.block_until_ready((state, candidate_metrics))
        if config.variant == "metrics-batch":
            original_metrics = _routing_metrics(candidate_metrics, model, batched=False)
            batched_metrics = _routing_metrics(candidate_metrics, model, batched=True)
            if original_metrics != batched_metrics:
                raise ValueError("Batched host metrics differ from ordinary metrics")
            tracker.log_summary({"correctness/host_metrics_exact": True})
        compiled = step_fn.lower(scheduled(state, source_clock), first_batch).compile()
        if jax.process_index() == 0:
            (StoragePath(config.artifact_root) / "train-step.hlo.txt.gz").write_bytes(
                gzip.compress(compiled.as_text().encode())
            )
        del compiled
        control_errors = _state_errors(control_a, control_b)
        candidate_errors = _state_errors(control_a, state)
        if config.verification:
            tracker.log_summary(
                {
                    "correctness/input_rows_at_1/control": _input_row_errors(
                        control_a, control_b, lookup_ids
                    ),
                    "correctness/input_rows_at_1/candidate": _input_row_errors(
                        control_a, state, lookup_ids
                    ),
                }
            )
        tracker.log_summary(
            {
                "correctness/initial_errors": initial_errors,
                "correctness/control_errors_step1": control_errors,
                "correctness/candidate_errors_step1": candidate_errors,
                "correctness/control_loss_abs_step1": abs(
                    float(ca_metrics["train/loss"]) - float(cb_metrics["train/loss"])
                ),
                "correctness/candidate_loss_abs_step1": abs(
                    float(ca_metrics["train/loss"])
                    - float(candidate_metrics["train/loss"])
                ),
            }
        )
        control_loss_differences = [
            float(cb_metrics["train/loss"]) - float(ca_metrics["train/loss"])
        ]
        candidate_loss_differences = [
            float(candidate_metrics["train/loss"]) - float(ca_metrics["train/loss"])
        ]
        if not config.verification:
            del control_a, control_b
        del ca_metrics, cb_metrics
        gc.collect()
        profiler = (
            None
            if config.benchmark
            else profile(
                f"/tmp/{config.run_id}/profiler",
                start_step=20,
                num_steps=5,
                create_perfetto_link=False,
                create_perfetto_trace=False,
                profiler_options=ProfileOptionsConfig(
                    host_tracer_level=2,
                    python_tracer_level=0,
                    enable_hlo_proto=True,
                ).build_jax_profile_options(),
                process_index=0,
                upload_uri=f"s3://marin-us-east-02a/tmp/ttl=7d/xprof/{config.run_id}",
                xprof_service_url=DEFAULT_XPROF_SERVICE_URL,
            )
        )
        trace_url = xprof_viewer_url(
            DEFAULT_XPROF_SERVICE_URL,
            f"s3://marin-us-east-02a/tmp/ttl=7d/xprof/{config.run_id}",
        )
        if not config.benchmark:
            tracker.log_summary({"profiling/xprof_url": trace_url})
        times: list[float] = []
        loads: list[float] = []
        metric_times: list[float] = []
        iteration_times: list[float] = []
        paired_control_times: list[float] = []
        losses: list[float] = []
        clock = source_clock.advance(
            input_tokens=TOKENS_PER_UPDATE,
            loss_targets=GLOBAL_BATCH_SIZE * (SEQ_LEN - 1),
            examples=GLOBAL_BATCH_SIZE,
        )
        # Each profile follows node zero's eight examples from each production
        # batch of 64. Thus the data and LR clocks match across all candidates.
        tracker.log_summary({"profiling/phase": "training"})
        previous_completed = time.monotonic()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = (
                None
                if config.benchmark
                else pool.submit(
                    distributed_batch,
                    dataset,
                    tokenizer,
                    mesh,
                    clock.examples,
                    8,
                    augment=True,
                )
            )
            for index in range(2, config.updates + 1):
                if remaining_seconds() < 30:
                    raise RuntimeError("Six-hour profiling deadline reached")
                started = time.monotonic()
                batch = (
                    cached_batches[index - 1]
                    if cached_batches is not None
                    else future.result()
                )
                loading = time.monotonic() - started
                if index < config.updates and not config.benchmark:
                    future = pool.submit(
                        distributed_batch,
                        dataset,
                        tokenizer,
                        mesh,
                        clock.examples + GLOBAL_BATCH_SIZE,
                        8,
                        augment=True,
                    )
                if config.verification:
                    control_a, ca_metrics, _ = step_fn(
                        scheduled(control_a, clock), batch
                    )
                    jax.block_until_ready((control_a, ca_metrics))
                    # Both follow a data-ready control A. Alternate their order
                    # so the paired comparison shares hardware and cache conditions.
                    order = (
                        ("control", "candidate")
                        if index % 2
                        else ("candidate", "control")
                    )
                    for label in order:
                        if label == "control":
                            control_b, cb_metrics, control_elapsed = timed_step(
                                control_b, clock, batch, "control", index
                            )
                        else:
                            state, metrics, elapsed = timed_step(
                                state, clock, batch, "candidate", index
                            )
                else:
                    state = scheduled(state, clock)
                    with jax.profiler.StepTraceAnnotation("train", step_num=index):
                        state, metrics, _ = step_fn(state, batch)
                        jax.block_until_ready((state, metrics))
                    elapsed = time.monotonic() - started
                completed = time.monotonic()
                iteration_elapsed = completed - previous_completed
                previous_completed = completed
                metric_started = time.monotonic()
                values = _routing_metrics(
                    metrics, model, batched=config.variant == "metrics-batch"
                )
                loss = float(metrics["train/loss"])
                if config.verification:
                    values["throughput/paired_control_step_seconds"] = control_elapsed
                    reference = float(ca_metrics["train/loss"])
                    control_loss_differences.append(
                        float(cb_metrics["train/loss"]) - reference
                    )
                    candidate_loss_differences.append(loss - reference)
                    reference_routing = _routing_metrics(
                        ca_metrics, production.model, batched=False
                    )
                    if (
                        values["moe/drop_fraction"]
                        != reference_routing["moe/drop_fraction"]
                    ):
                        raise ValueError(
                            "Candidate and control have different dropped assignment fractions"
                        )
                    values["correctness/control_loss_difference"] = (
                        control_loss_differences[-1]
                    )
                    values["correctness/candidate_loss_difference"] = (
                        candidate_loss_differences[-1]
                    )
                    if index in {2, 10, 50, config.updates}:
                        tracker.log_summary(
                            {
                                f"correctness/state_at_{index}/control": _state_errors(
                                    control_a, control_b
                                ),
                                f"correctness/state_at_{index}/candidate": _state_errors(
                                    control_a, state
                                ),
                                f"correctness/input_rows_at_{index}/control": _input_row_errors(
                                    control_a, control_b, lookup_ids
                                ),
                                f"correctness/input_rows_at_{index}/candidate": _input_row_errors(
                                    control_a, state, lookup_ids
                                ),
                            }
                        )
                if not np.isfinite(loss) or not all(
                    np.isfinite(float(x)) for x in values.values()
                ):
                    raise ValueError("Nonfinite metrics in isolated profile")
                values.update(
                    {
                        "train/loss": loss,
                        "train/step": index,
                        "profiling/source_step": clock.updates,
                        "throughput/step_seconds": elapsed,
                        "throughput/iteration_seconds": iteration_elapsed,
                        "throughput/loading_seconds": loading,
                        "throughput/tokens_per_second": 8 * SEQ_LEN / elapsed,
                        "run_progress": index / config.updates,
                    }
                )
                stats = jax.local_devices()[0].memory_stats()
                if stats:
                    for name in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit"):
                        if name in stats:
                            values[f"memory/{name}"] = stats[name]
                metric_elapsed = time.monotonic() - metric_started
                values["throughput/metrics_seconds"] = metric_elapsed
                tracker.log(values, step=index)
                if index >= 30:
                    times.append(elapsed)
                    iteration_times.append(iteration_elapsed)
                    loads.append(loading)
                    metric_times.append(metric_elapsed)
                    losses.append(loss)
                    if config.verification:
                        paired_control_times.append(control_elapsed)
                if profiler is not None:
                    profiler(type("ProfileStep", (), {"step": index})())
                clock = clock.advance(
                    input_tokens=TOKENS_PER_UPDATE,
                    loss_targets=GLOBAL_BATCH_SIZE * (SEQ_LEN - 1),
                    examples=GLOBAL_BATCH_SIZE,
                )
        summary = {
            "profiling/phase": "complete",
            "profiling/median_step_seconds": statistics.median(times),
            "profiling/mean_step_seconds": statistics.mean(times),
            "profiling/median_iteration_seconds": statistics.median(iteration_times),
            "profiling/mean_iteration_seconds": statistics.mean(iteration_times),
            "profiling/p90_step_seconds": float(np.quantile(times, 0.9)),
            "profiling/median_loading_seconds": statistics.median(loads),
            "profiling/median_metrics_seconds": statistics.median(metric_times),
            "profiling/mean_loss": statistics.mean(losses),
            "profiling/actual_tokens": config.updates * 8 * SEQ_LEN,
            "profiling/updates": config.updates,
            "profiling/no_checkpoints_saved": True,
        }
        if config.verification:
            summary.update(
                {
                    "profiling/paired_control_mean_step_seconds": statistics.mean(
                        paired_control_times
                    ),
                    "profiling/paired_control_median_step_seconds": statistics.median(
                        paired_control_times
                    ),
                    "profiling/paired_mean_throughput_speedup": statistics.mean(
                        paired_control_times
                    )
                    / statistics.mean(times)
                    - 1,
                    "profiling/paired_order_alternated": True,
                }
            )
            summary["correctness/compared_updates"] = config.updates
            for label, differences in (
                ("control", control_loss_differences),
                ("candidate", candidate_loss_differences),
            ):
                values = np.asarray(differences)
                summary.update(
                    {
                        f"correctness/{label}_loss_rmse": float(
                            np.sqrt(np.mean(values**2))
                        ),
                        f"correctness/{label}_loss_mae": float(np.mean(np.abs(values))),
                        f"correctness/{label}_loss_signed_mean": float(np.mean(values)),
                        f"correctness/{label}_loss_max_abs": float(
                            np.max(np.abs(values))
                        ),
                    }
                )
        if jax.process_index() == 0:
            (StoragePath(config.artifact_root) / "result.json").write_text(
                config_json(summary)
            )
        tracker.log_summary(summary)
    tracker.current_tracker().finish()


def run_profile(config: PerfConfig) -> None:
    try:
        _run_profile(config)
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
                    "Could not report profile failure"
                )
        raise


def dispatch(config: PerfConfig) -> None:
    if remaining_seconds() < 300:
        raise RuntimeError("Too little time remains before the six-hour deadline")
    if config.variant == "overlap1":
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + " --xla_gpu_experimental_parallel_collective_overlap_limit=1"
        )
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=False,
        moe_implementation="fixed_pooled_wave_all_to_all",
        remat_mode="recompute_all",
        processes_per_task=config.processes,
    )
    resources = ResourceConfig.with_gpu(
        "H100", count=8, replicas=1, cpu=32, ram="600g", disk="900g"
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
            processes_per_task=config.processes,
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
    parser.add_argument("--benchmark", action="store_true")
    dispatch(PerfConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
