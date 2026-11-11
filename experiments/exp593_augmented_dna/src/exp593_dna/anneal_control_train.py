"""Full-native-state old-1T-distribution annealing control for exp593."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import Any

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
from exp586_moe.schedule import optimizer_config, set_token_learning_rates
from exp586_moe.state import fresh_state, state_all_finite, state_max_abs_error
from exp586_moe.tracking import wandb_history_offset
from exp586_moe.train import primary_json, theoretical_flops
from experiments.grug.dispatch import _forwarded_env_vars
from experiments.grug.moe_hero_ep.model import Transformer
from experiments.grug.moe_hero_ep.train import (
    _apply_hero_ep_runtime_defaults,
    _compute_flops,
)
from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.rpc.proto_display import priority_band_value
from levanter import tracker
from levanter.callbacks.watch import WatchConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.training.training import resolve_training_env
from rigging.timing import Duration

from exp593_dna import anneal_control_batches, anneal_state
from exp593_dna.anneal import (
    ORIGINAL_TOKENS,
    ORIGINAL_UPDATES,
    OWNER,
    REFERENCE_TOKENS,
    TRIALS,
    AnnealTrial,
)
from exp593_dna.anneal_control_batches import OldDistributionSource
from exp593_dna.anneal_evaluate import CompactEvaluator
from exp593_dna.anneal_state import AnnealClock
from exp593_dna.anneal_tasks import (
    TaskEvaluator,
    compact_task_metrics,
    fetch_panels,
    panel_binding,
)
from exp593_dna.anneal_tokenizer import verify_source_tokenizer
from exp593_dna.corpus import open_corpus
from exp593_dna.diagnostics import NormPolicy
from exp593_dna.evaluate import ValidationPanels
from exp593_dna.inputs import INPUTS_MANIFEST_SHA256, PreparedInputs
from exp593_dna.pipeline import load_encoder
from exp593_dna.recovery import canonical_json
from exp593_dna.train import (
    WANDB_BASE_URL,
    evaluation_progress,
    input_progress,
    metric_values,
)
from exp593_dna.train_step import make_train_step

LOGGER = logging.getLogger(__name__)

CONTROL_ID = "dna-exp593-anneal-545b-originaldata-equalbase-seed0-v1"
CONTROL_HORIZON = 228_881 * REFERENCE_TOKENS


class ControlTrial(AnnealTrial):
    """Keep the 545B LR program and forbid FIM training examples."""

    def fim_probability(self, input_tokens: int) -> float:
        if not 0 <= input_tokens <= self.horizon:
            raise ValueError("Control token clock is outside its fixed horizon")
        return 0.0


CONTROL_TRIAL = ControlTrial(
    "545b",
    TRIALS["545b"].source_update,
    TRIALS["545b"].source_metadata_sha256,
    CONTROL_HORIZON,
)

_MOE_CHARTS = {
    "moe/drop_fraction",
    "moe/sender_drop_fraction",
    "moe/receiver_drop_fraction",
    "moe/skipped_padding_fraction",
    "moe/valid_assignments",
}


def compact_training_metrics(values: dict[str, float | int]) -> dict[str, float | int]:
    """Keep aggregate optimization/MoE charts after all raw invariants are checked."""
    active = [
        v
        for k, v in values.items()
        if k.startswith("train/router/layer_") and k.endswith("/active_experts")
    ]
    imbalance = [
        v
        for k, v in values.items()
        if k.startswith("train/router/layer_") and k.endswith("/max_to_mean_load")
    ]
    if len(active) != 8 or len(imbalance) != 8:
        raise ValueError("Expected routing diagnostics from every MoE layer")
    result = {
        k: v
        for k, v in values.items()
        if not k.startswith("train/router/layer_")
        and (not k.startswith("moe/") or k in _MOE_CHARTS)
    }
    result["train/router/active_experts_min"] = min(active)
    result["train/router/max_to_mean_load_max"] = max(imbalance)
    return result


def species_counts(mixture: dict[str, int]) -> dict[str, int]:
    """One W&B summary map, rather than per-species step-history charts."""
    return {
        key.removeprefix("species/"): int(value)
        for key, value in sorted(mixture.items())
        if key.startswith("species/")
    }


@dataclass(frozen=True)
class TrainingConfig:
    trial: str
    cluster: str
    nodes: int = 8
    validation_updates: int | None = None
    run_id: str | None = None
    checkpoint_interval_seconds: int = 600
    keep_temporary_checkpoints: int = 6

    def __post_init__(self) -> None:
        if (
            self.trial != "545b-control"
            or self.cluster not in ("cw-rno2a", "cw-us-east-02a")
            or self.nodes not in (1, 2, 4, 8)
        ):
            raise ValueError("Unapproved annealing recipe or H100 placement")
        if self.checkpoint_interval_seconds < 60 or self.keep_temporary_checkpoints < 1:
            raise ValueError("Invalid checkpoint policy")
        if self.validation_updates is None:
            if self.run_id is not None and self.run_id != CONTROL_ID:
                raise ValueError(
                    "Production must use its new stable annealing identity"
                )
        elif not (
            self.validation_updates == 64
            and self.run_id
            and self.run_id.startswith("dna-exp593-")
            and "smoke" in self.run_id
        ):
            raise ValueError(
                "Disposable full-loop validation requires its own smoke identity"
            )

    @property
    def recipe(self) -> AnnealTrial:
        return CONTROL_TRIAL

    @property
    def identity(self) -> str:
        return self.run_id or CONTROL_ID

    @property
    def checkpoint_root(self) -> str:
        lane = "training" if self.validation_updates is None else "validation"
        return f"{OWNER}/annealing/{lane}/{self.identity}/checkpoints"


def bindings(
    config: TrainingConfig, inputs: PreparedInputs, encoder_digest: str
) -> dict[str, Any]:
    sources = {
        name: hashlib.sha256(
            files("exp593_dna").joinpath(name).read_bytes()
        ).hexdigest()
        for name in (
            "anneal.py",
            "anneal_data.py",
            "anneal_batches.py",
            "anneal_state.py",
            "anneal_tokenizer.py",
            "anneal_evaluate.py",
            "anneal_tasks.py",
            "anneal_train.py",
            "anneal_control_train.py",
            "anneal_control_batches.py",
            "train_step.py",
            "templates.py",
            "annotations.py",
            "corpus.py",
            "schedule.py",
            "pipeline.py",
            "batches.py",
            "evaluate.py",
            "development_evaluate.py",
            "development_readout.py",
            "fim_validation.py",
            "validation.py",
            "diagnostics.py",
        )
    }
    opt = optimizer_config(
        reference_tokens_per_update=REFERENCE_TOKENS, training_tokens=ORIGINAL_TOKENS
    )
    return json.loads(
        canonical_json(
            {
                "protocol": "exp593-original-1t-distribution-annealing-control-v1",
                "run_id": config.identity,
                "trial": asdict(config.recipe),
                "model": asdict(d768_config("pretrained")),
                "optimizer": asdict(opt),
                "optimizer_heuristic_tokens": ORIGINAL_TOKENS,
                "inputs_manifest": INPUTS_MANIFEST_SHA256,
                "tokenizer": encoder_digest,
                "source_code_sha256": sources,
                "validation_updates": config.validation_updates,
                "validation_fim_probability": 0.0,
                "training_distribution": "exp586-fixed8192-full-window-lowercase-rc472-shuffle0",
                "training_horizon_exact_full_batches": CONTROL_HORIZON,
                "norm_policy": asdict(NormPolicy()),
                "primary_manifest": inputs.manifest["files"][
                    "validation/primary_validation_512.jsonl"
                ]["sha256"],
                "stop_policy": "nearest_below_120b_complete_old_distribution_batches",
                "task_panels": panel_binding(),
            }
        )
    )


def _run(config: TrainingConfig) -> None:
    trial = config.recipe
    os.environ.update(
        WANDB_BASE_URL=WANDB_BASE_URL, WANDB_ENTITY="eric-czech", WANDB_PROJECT="marin"
    )
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
        tracker=WandbConfig(
            entity="eric-czech",
            project="marin",
            name=config.identity,
            id=config.identity,
            group="dna-exp593"
            if config.validation_updates is None
            else "dna-exp593-validation",
            background=False,
            save_code=False,
            tags=[
                "dna-exp593",
                "annealing",
                "original-data-control",
                "H100",
                config.cluster,
            ],
            replicate_path=config.checkpoint_root,
            resume="allow",
        ),
    )
    trainer.initialize()
    if jax.device_count() != config.nodes * 8 or not all(
        "H100" in d.device_kind for d in jax.devices()
    ):
        raise ValueError("Annealing did not receive its requested H100 gang")
    device_kind = jax.devices()[0].device_kind
    peak_per_device, peak_total = theoretical_flops(device_kind, jax.device_count())
    flops_per_example = {
        bucket: _compute_flops(
            model_config=replace(d768_config("pretrained"), max_seq_len=bucket)
        )[0]
        for bucket in (2048, 4096, 6144, 8192)
    }
    if config.validation_updates is not None and jax.process_index() == 0:
        # Bounded placement checks expose Python wait locations when numerical
        # progress stops. Only the global primary dumps; production stays quiet.
        faulthandler.dump_traceback_later(120, repeat=True)
    tracker.log_configuration(config)
    log_step = wandb_history_offset()
    tracker.log_summary(
        {
            "training/phase": "loading_inputs",
            "throughput/device_kind": device_kind,
            "throughput/theoretical_flops_per_device": peak_per_device,
            "throughput/theoretical_flops": peak_total,
            "throughput/flops_per_example_by_bucket": {
                str(bucket): value for bucket, value in flops_per_example.items()
            },
            "throughput/mfu_method": "exp586_analytic_padded_bucket_flops_over_H100_BF16_peak",
        }
    )
    inputs = PreparedInputs.open(Path("/tmp/exp593-inputs"))
    encoder = load_encoder()
    scientific = bindings(config, inputs, encoder.digest)
    checkpoint = primary_json(
        lambda: anneal_state.discover(config.checkpoint_root, trial, scientific)
    )
    tracker.log_hyperparameters({"bindings": scientific})
    shuffled = open_corpus(inputs, "train", 0, preload=True, progress=input_progress)
    source = OldDistributionSource(shuffled)
    panels = ValidationPanels.open(inputs, encoder)
    mesh = replicated_mesh()
    evaluator = CompactEvaluator(
        panels, encoder, mesh, mp, progress=evaluation_progress
    )
    tasks = TaskEvaluator(
        evaluator,
        fetch_panels(Path("/tmp/exp593-anneal-task-panels")),
        mp,
        pilot=config.validation_updates is not None,
    )
    tracker.log_hyperparameters({"task_panel_reference": tasks.reference_statistics()})
    model = replace(
        local_model(d768_config("pretrained")),
        compact_input_lookup_ids=(),
        compact_input_lookup_one_hot=False,
    )
    opt_config = optimizer_config(
        reference_tokens_per_update=REFERENCE_TOKENS, training_tokens=ORIGINAL_TOKENS
    )
    optimizer = match_gradient_placement(
        opt_config.build(ORIGINAL_UPDATES, ns4d_impl=sharded_expert_ns(mesh))
    )
    norm_policy = NormPolicy()
    compiled: set[int] = set()

    with jax.set_mesh(mesh):
        tracker.log_summary({"training/phase": "initializing_restore_template"})
        state = eqx.filter_jit(
            lambda: fresh_state(
                Transformer.init(model, key=jax.random.PRNGKey(0)),
                optimizer,
                mp,
                offload=False,
                retain_master=True,
            )
        )()
        state, partitions = partition_expert_optimizer_state(state, mesh)
        if checkpoint:
            state, clock = anneal_state.restore(
                state, checkpoint, trial, mesh, scientific
            )
        else:
            tracker.log_summary(
                {
                    "training/phase": "restoring_full_source",
                    "training/source_checkpoint": trial.source_checkpoint,
                }
            )
            state, clock, evidence = anneal_state.restore_source(
                state, trial, mesh, encoder
            )
            tracker.log_summary(
                {
                    "training/source_restore": evidence,
                    "training/source_state_import": "full_native_state",
                }
            )
        jax.block_until_ready(state)
        assert_partitioned_optimizer_state(state, mesh)
        tracker.log_summary(
            {
                "training/import_integrity": anneal_state.imported_state_integrity(
                    state, TRIALS["545b"] if checkpoint is None else trial, clock
                )
            }
        )
        tracker.log_summary(
            {
                "training/optimizer_partitions": partitions,
                "training/resume_global_step": int(state.step),
                "training/resume_stage_clock": asdict(clock.stage),
            }
        )
        train_step = make_train_step(optimizer, mp, norm_policy=norm_policy)

        def log(values: dict[str, Any]) -> None:
            nonlocal log_step
            log_step += 1
            tracker.log(
                {
                    "train/input_tokens": clock.stage.input_tokens,
                    "train/step": clock.stage.updates,
                    "train/global_step": trial.source_update + clock.stage.updates,
                    "train/lifetime_input_tokens": trial.source_tokens
                    + clock.stage.input_tokens,
                    **values,
                },
                step=log_step,
            )

        def evaluate(event: str, *, include_tasks: bool = False) -> None:
            tracker.log_summary(
                {"training/phase": "evaluating", "training/eval_event": event}
            )
            values = evaluator.evaluate(state)
            if include_tasks:
                tracker.log_summary({"training/phase": "evaluating_tasks"})
                task_details = tasks.evaluate(state)
                values.update(
                    compact_task_metrics(task_details, tasks.panels, pilot=tasks.pilot)
                )
                tracker.log_summary(
                    {
                        "training/last_task_panel_scores_json": json.dumps(
                            task_details, sort_keys=True
                        ),
                        f"training/task_panel_at_{clock.stage.input_tokens}_json": json.dumps(
                            task_details, sort_keys=True
                        ),
                    }
                )
            log(values)
            tracker.log_summary(
                {"training/last_eval_input_tokens": clock.stage.input_tokens, **values}
            )

        def save(permanent: bool) -> str:
            if not bool(state_all_finite(state)):
                raise ValueError("Refusing to checkpoint a nonfinite training state")
            path = primary_json(
                lambda: anneal_state.destination(config.checkpoint_root, trial, clock)
            )
            tracker.log_summary({"training/phase": "checkpointing"})
            anneal_state.save(
                state, path, trial, clock, scientific, permanent=permanent
            )
            digest = primary_json(
                lambda: anneal_state.commit(
                    config.checkpoint_root,
                    path,
                    trial,
                    clock,
                    scientific,
                    keep=config.keep_temporary_checkpoints,
                )
            )
            values = {
                "training/latest_checkpoint": path,
                "training/latest_checkpoint_metadata_sha256": digest,
                "data_mix/species_examples": species_counts(clock.mixture),
            }
            if permanent:
                values["training/latest_permanent_checkpoint"] = path
            tracker.log_summary(values)
            return path

        evaluate(
            "source" if checkpoint is None else "resume",
            include_tasks=checkpoint is None,
        )
        if checkpoint is None and config.validation_updates is None:
            checkpoint = save(True)
        last_save = time.monotonic()

        def complete() -> bool:
            return (
                clock.stage.updates >= config.validation_updates
                if config.validation_updates is not None
                else clock.stage.input_tokens == trial.horizon
            )

        def prepare(c: AnnealClock) -> Any:
            if config.validation_updates is None:
                return anneal_control_batches.prepare(
                    source,
                    mesh,
                    trial,
                    stage_tokens=c.stage.input_tokens,
                    stage_examples=c.stage.examples,
                )
            # Disposable validation uses the identical original-data path.
            return anneal_control_batches.prepare(
                source,
                mesh,
                trial,
                stage_tokens=c.stage.input_tokens,
                stage_examples=c.stage.examples,
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = None if complete() else pool.submit(prepare, clock)

            def phase(name: str) -> None:
                if (
                    config.validation_updates is not None
                    and clock.stage.updates < 3
                    and jax.process_index() == 0
                ):
                    LOGGER.info(
                        "Validation loop stage=%d phase=%s", clock.stage.updates, name
                    )

            while not complete():
                started = time.monotonic()
                phase("awaiting_prefetch")
                local = future.result()
                phase("assembling_batch")
                batch = anneal_control_batches.assemble(local, mesh)
                loaded = time.monotonic()
                next_clock = clock.advance(batch.counts, batch.mixture)
                done = (
                    next_clock.stage.updates >= config.validation_updates
                    if config.validation_updates is not None
                    else next_clock.stage.input_tokens == trial.horizon
                )
                future = None if done else pool.submit(prepare, next_clock)
                factor = trial.factor(clock.stage.input_tokens)
                state = replace(
                    state,
                    opt_state=set_token_learning_rates(
                        state.opt_state, opt_config, trial, clock.stage
                    ),
                )
                norms_due = config.validation_updates is not None or norm_policy.due(
                    next_clock.stage.updates
                )
                first_compile = batch.bucket not in compiled
                phase("running_update")
                state, metrics, norms = train_step(state, batch.model_batch, norms_due)
                jax.block_until_ready((state, metrics, norms))
                if clock.stage.updates == 0:
                    if not bool(
                        model_matches_fp32_master(state.params, state.master_params)
                    ) or not bool(state_all_finite(state)):
                        raise ValueError(
                            "First normal update failed native model/master reconciliation"
                        )
                    tracker.log_summary(
                        {"training/first_update_master_reconciliation": True}
                    )
                compiled.add(batch.bucket)
                previous = clock
                clock = next_clock
                values = compact_training_metrics(
                    metric_values(metrics, norms, batch, collect_norms=norms_due)
                )
                elapsed = time.monotonic() - started
                examples, fim = clock.stage.examples, clock.stage.fim_examples
                achieved_flops_per_second = (
                    flops_per_example[batch.bucket] * 64 / elapsed
                )
                values.update(
                    {
                        "run_progress": min(
                            clock.stage.input_tokens / trial.horizon, 0.999999
                        ),
                        "train/total_progress": clock.stage.input_tokens
                        / trial.horizon,
                        "train/dna_tokens": clock.stage.dna_tokens,
                        "train/text_tokens": clock.stage.text_tokens,
                        "train/loss_targets": clock.stage.loss_targets,
                        "train/examples": examples,
                        "train/fim_examples": fim,
                        "train/rc_examples": clock.stage.rc_examples,
                        "train/fim_fraction": fim / examples,
                        "train/rc_fraction": clock.stage.rc_examples / examples,
                        "train/dna_base_epochs": clock.stage.dna_tokens
                        / 21_615_869_952,
                        "train/source_row_passes": (trial.source_examples + examples)
                        / 2_638_656,
                        "train/lifetime_dna_base_epochs": (
                            trial.source_tokens + clock.stage.dna_tokens
                        )
                        / 21_615_869_952,
                        "train/muon_lr": opt_config.learning_rate * factor,
                        "train/adam_lr": opt_config.adam_lr * factor,
                        "train/bucket": batch.bucket,
                        "train/padding_fraction": 1
                        - (clock.stage.input_tokens - previous.stage.input_tokens)
                        / (64 * batch.bucket),
                        "curriculum/fim_probability": 0.0,
                        "curriculum/batch_fim_fraction": batch.counts["fim_examples"]
                        / 64,
                        "curriculum/fim_supervised_target_fraction": clock.mixture.get(
                            "fim_targets", 0
                        )
                        / clock.stage.loss_targets,
                        "throughput/step_seconds": elapsed,
                        "throughput/loading_seconds": loaded - started,
                        "throughput/includes_first_compile": int(first_compile),
                        "throughput/tokens_per_second": (
                            clock.stage.input_tokens - previous.stage.input_tokens
                        )
                        / elapsed,
                        "throughput/gflops_per_second": achieved_flops_per_second / 1e9,
                        "throughput/mfu": achieved_flops_per_second / peak_total * 100,
                        "throughput/total_gflops": sum(
                            flops_per_example[b] * clock.mixture.get(f"bucket/{b}", 0)
                            for b in flops_per_example
                        )
                        / 1e9,
                        "data_mix/short_fim_examples": clock.mixture.get(
                            "short_fim", 0
                        ),
                        "data_mix/long_fim_examples": clock.mixture.get("long_fim", 0),
                        "data_mix/nearby_fim_examples": clock.mixture.get("nearby", 0),
                        "data_mix/right_flank_first_examples": clock.mixture.get(
                            "right_flank_first", 0
                        ),
                        "data_mix/fim_scored_targets": clock.mixture.get(
                            "fim_targets", 0
                        ),
                        "data_mix/splice_landmark_examples": sum(
                            clock.mixture.get(f"landmark/{kind}", 0)
                            for kind in ("splice_acceptor", "splice_donor")
                        ),
                        "data_mix/translation_landmark_examples": sum(
                            clock.mixture.get(f"landmark/{kind}", 0)
                            for kind in ("translation_start", "translation_stop")
                        ),
                    }
                )
                if first_compile:
                    values[f"compile/bucket_{batch.bucket}_seconds"] = (
                        time.monotonic() - loaded
                    )
                phase("logging_update")
                log(values)
                tracker.log_summary(
                    {
                        "training/phase": "training",
                        "training/last_input_fingerprint": batch.input_fingerprint,
                    }
                )
                if clock.stage.updates <= 5 or clock.stage.updates % 100 == 0:
                    LOGGER.info(
                        "Stage update=%d global=%d inputs=%d loss=%f",
                        clock.stage.updates,
                        int(state.step),
                        clock.stage.input_tokens,
                        values["train/loss"],
                    )
                events = trial.events(
                    previous.stage.input_tokens, clock.stage.input_tokens
                )
                task_events = trial.events(
                    previous.stage.input_tokens, clock.stage.input_tokens, tasks=True
                )
                permanent = bool(task_events) or done
                if events or task_events or done:
                    if future is not None:
                        future.result()
                    evaluate(
                        ",".join(events) or "validation_end",
                        include_tasks=bool(task_events) or done,
                    )
                phase("checkpoint_timer_broadcast")
                due = primary_json(
                    lambda saved_at=last_save, milestone=permanent: (
                        time.monotonic() - saved_at
                        >= config.checkpoint_interval_seconds
                        or milestone
                    )
                )
                phase("checkpoint_timer_complete")
                if due:
                    checkpoint = save(permanent)
                    last_save = time.monotonic()
        if config.validation_updates is not None:
            qualified = {
                int(k.removeprefix("bucket/"))
                for k, v in clock.mixture.items()
                if k.startswith("bucket/") and v > 0
            }
            if qualified != {8192}:
                raise ValueError(
                    "Disposable control validation did not exercise 8192-base windows"
                )
            restored, restored_clock = anneal_state.restore(
                state, checkpoint, trial, mesh, scientific
            )
            error = float(state_max_abs_error(state, restored))
            if restored_clock != clock or error != 0:
                raise ValueError(
                    "Disposable native checkpoint roundtrip changed optimizer/model state"
                )
            log(
                {
                    "validation/roundtrip_max_abs_error": error,
                    "validation/exercised_buckets": len(qualified),
                }
            )
        elif clock.stage.input_tokens != trial.horizon:
            raise ValueError("The fixed annealing input cap was not reached exactly")
        tracker.log_summary(
            {
                "training/phase": "complete",
                "training/final_checkpoint": checkpoint,
                "training/final_clock": asdict(clock),
                "training/nominal_input_tokens": trial.horizon,
            }
        )
        log({"run_progress": 1.0})
    tracker.current_tracker().finish()
    if config.validation_updates is not None and jax.process_index() == 0:
        faulthandler.cancel_dump_traceback_later()


def run(config: TrainingConfig) -> None:
    try:
        _run(config)
    except Exception as error:
        if jax.process_index() == 0:
            try:
                tracker.log_summary(
                    {
                        "training/phase": "failed",
                        "training/error": f"{type(error).__name__}: {error}",
                    }
                )
                tracker.get_tracker("wandb").run.finish(exit_code=1)
            except Exception:
                LOGGER.exception("Could not publish failure")
        raise


def dispatch(config: TrainingConfig) -> None:
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
                n: os.environ[n]
                for n in ("CW_KEY_ID", "CW_KEY_SECRET", "WANDB_API_KEY")
            },
            "WANDB_BASE_URL": WANDB_BASE_URL,
            "WANDB_ENTITY": "eric-czech",
            "WANDB_PROJECT": "marin",
        },
        resources,
    )
    env.update(
        WANDB_BASE_URL=WANDB_BASE_URL, WANDB_ENTITY="eric-czech", WANDB_PROJECT="marin"
    )
    job = current_client().submit(
        JobRequest(
            name=f"train-{config.identity}",
            entrypoint=Entrypoint.from_callable(run, args=[config]),
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial", choices=("545b-control",), required=True)
    p.add_argument("--cluster", choices=("cw-rno2a", "cw-us-east-02a"), required=True)
    p.add_argument("--nodes", type=int, choices=(1, 2, 4, 8), default=8)
    p.add_argument("--validation-updates", type=int)
    p.add_argument("--run-id")
    p.add_argument("--checkpoint-interval-seconds", type=int, default=600)
    p.add_argument("--keep-temporary-checkpoints", type=int, default=6)
    dispatch(TrainingConfig(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
