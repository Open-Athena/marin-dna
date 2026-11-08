"""Augmented d768 runs with fresh initialization and durable token-clock recovery."""

from __future__ import annotations

import argparse
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
import numpy as np
from exp586_moe.config import (
    CONTINUATION_SOURCE_CHECKPOINT,
    SOURCE_CHECKPOINT,
    SOURCE_METADATA_DIGEST,
    d768_config,
)
from exp586_moe.data_parallel import local_model, replicated_mesh
from exp586_moe.perf5_metrics import copy_metrics
from exp586_moe.perf5_optimizer import (
    assert_partitioned_optimizer_state,
    match_gradient_placement,
    model_matches_fp32_master,
    partition_expert_optimizer_state,
    sharded_expert_ns,
)
from exp586_moe.schedule import optimizer_config, set_token_learning_rates
from exp586_moe.state import fresh_state, state_all_finite
from exp586_moe.tracking import wandb_history_offset
from exp586_moe.train import primary_json
from experiments.grug.dispatch import _forwarded_env_vars
from experiments.grug.moe_hero_ep.train import (
    _apply_hero_ep_runtime_defaults,
    _drop_metrics,
)
from experiments.grug.moe_hero_ep.weights import restore_weights
from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.rpc.proto_display import priority_band_value
from levanter import tracker
from levanter.callbacks.watch import WatchConfig
from levanter.grug.grug_moe import (
    MOE_DROPPED_ASSIGNMENTS_METRIC,
    MOE_RECEIVER_DROPPED_ASSIGNMENTS_METRIC,
    MOE_SENDER_DROPPED_ASSIGNMENTS_METRIC,
    MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC,
    MOE_VALID_ASSIGNMENTS_METRIC,
)
from levanter.tracker.tracker import NoopConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.training.training import resolve_training_env
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from exp593_dna.annotations import load_annotations
from exp593_dna.batches import (
    TrainingBatch,
    assemble_distributed_batch,
    prepare_local_batch,
)
from exp593_dna.corpus import open_corpus
from exp593_dna.development_evaluate import DevelopmentEvaluator
from exp593_dna.development_readout import fetch_development
from exp593_dna.development_results import (
    publish_development_protocol,
    publish_development_result,
)
from exp593_dna.diagnostics import NormPolicy
from exp593_dna.evaluate import Evaluator, ValidationPanels
from exp593_dna.inputs import INPUTS_MANIFEST_SHA256, PreparedInputs
from exp593_dna.pipeline import AugmentationConfig, Augmenter, Encoder, load_encoder
from exp593_dna.recovery import (
    bind_and_discover,
    canonical_json,
    commit_and_prune,
    prepare_checkpoint,
)
from exp593_dna.schedule import (
    REFERENCE_TOKENS_PER_UPDATE,
    TRAINING_TOKENS,
    InputTokenSchedule,
    TokenClock,
    validation_events,
)
from exp593_dna.standby import (
    checked_checkpoint,
    committed_promotion,
    publish_once,
    validate_promotion,
    validate_standby_id,
)
from exp593_dna.state import (
    checkpoint_template,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from exp593_dna.train_step import make_train_step

PRODUCTION_RUN_ID = "dna-exp593-augmented-d768-pretrained-seed0-v1"
DNA216B_RUN_ID = "dna-exp593-augmented-d768-dna216b-lrm0p5-seed0-v1"
# restore_weights checks canonical metadata, rather than the raw-file digest.
DNA216B_METADATA_DIGEST = (
    "275e68e8a521333a4dd5c7bf2aa8f9c79c11517677a8af47ce7abc1aa5004a89"
)
WANDB_BASE_URL = "https://forge.coreweave.com/api/wandb"
OWNER = "s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna"
LOGGER = logging.getLogger(__name__)
MAX_UPDATES = (TRAINING_TOKENS + 64 * 2048 - 1) // (64 * 2048)


def evaluation_progress(stage: str, completed: int, total: int) -> None:
    tracker.log_summary(
        {
            "eval/phase": stage,
            "eval/phase_completed": completed,
            "eval/phase_total": total,
        }
    )
    if jax.process_index() == 0:
        LOGGER.info("Evaluation %s: %d/%d", stage, completed, total)


def input_progress(completed: int, total: int) -> None:
    if jax.process_index() == 0 and (completed % 4 == 0 or completed == total):
        tracker.log_summary(
            {"training/cached_train_shards": completed, "training/train_shards": total}
        )
        LOGGER.info("Verified training shards cached: %d/%d", completed, total)


@dataclass(frozen=True)
class TrainingConfig:
    cluster: str
    nodes: int = 1
    run_id: str = PRODUCTION_RUN_ID
    validation_updates: int | None = None
    standby_id: str | None = None
    standby_checkpoint: str | None = None
    standby_from_job: str | None = None
    initialization: str = "text"

    def __post_init__(self) -> None:
        if self.initialization not in ("text", "dna216b"):
            raise ValueError("Unapproved model initialization")
        if self.cluster not in ("cw-rno2a", "cw-us-east-02a") or self.nodes not in (
            1,
            2,
            4,
            8,
        ):
            raise ValueError("Unapproved H100 placement")
        if self.validation_updates is None:
            expected = (
                PRODUCTION_RUN_ID if self.initialization == "text" else DNA216B_RUN_ID
            )
            if self.run_id != expected:
                raise ValueError("Final model identity must match its initialization")
        elif not (
            1 <= self.validation_updates <= 64
            and self.run_id.startswith("dna-exp593-")
            and "smoke" in self.run_id
        ):
            raise ValueError(
                "Bounded validation requires a separate smoke identity and 1..64 updates"
            )
        if self.standby_id is not None:
            validate_standby_id(self.standby_id)
            if (
                self.validation_updates is not None
                or not self.standby_checkpoint
                or not self.standby_from_job
                or not self.standby_from_job.startswith("/eczech/")
                or self.standby_from_job.count("/") != 2
            ):
                raise ValueError(
                    "Standby requires a production checkpoint and exact old root"
                )
        elif self.standby_checkpoint is not None or self.standby_from_job is not None:
            raise ValueError("Standby source requires a unique standby identity")

    @property
    def source_checkpoint(self) -> str:
        return (
            SOURCE_CHECKPOINT
            if self.initialization == "text"
            else CONTINUATION_SOURCE_CHECKPOINT
        )

    @property
    def source_metadata_digest(self) -> str:
        return (
            SOURCE_METADATA_DIGEST
            if self.initialization == "text"
            else DNA216B_METADATA_DIGEST
        )

    @property
    def lr_multiplier(self) -> float:
        return 1.0 if self.initialization == "text" else 0.5

    @property
    def checkpoint_root(self) -> str:
        lane = "training" if self.validation_updates is None else "validation"
        return f"{OWNER}/{lane}/{self.run_id}/checkpoints"

    @property
    def control_root(self) -> str:
        if self.standby_id is None:
            raise ValueError("This allocation has no standby control namespace")
        return f"{OWNER}/placements/{self.standby_id}"

    def complete(self, clock: TokenClock) -> bool:
        if self.validation_updates is not None:
            return clock.updates >= self.validation_updates
        return clock.input_tokens >= TRAINING_TOKENS


def scientific_bindings(
    config: TrainingConfig, inputs: PreparedInputs, encoder: Encoder
) -> dict[str, Any]:
    """Placement is deliberately absent, allowing checkpoint-preserving reslicing."""
    objective_code = {}
    for module in (
        "pipeline.py",
        "templates.py",
        "annotations.py",
        "train_step.py",
        "schedule.py",
        "fim_validation.py",
        "evaluate.py",
    ):
        objective_code[module] = hashlib.sha256(
            files("exp593_dna").joinpath(module).read_bytes()
        ).hexdigest()
    return json.loads(
        canonical_json(
            {
                "protocol": "exp593-augmented-d768-v1",
                "run_id": config.run_id,
                "validation_updates": config.validation_updates,
                "source_checkpoint": config.source_checkpoint,
                "source_metadata": config.source_metadata_digest,
                "optimized_source_commit": "de5f428fc7b2bac8b6e40697f905c8b99ca3580d",
                "model": asdict(d768_config("pretrained")),
                "inputs_manifest": INPUTS_MANIFEST_SHA256,
                "tokenizer": encoder.digest,
                "augmentation": asdict(AugmentationConfig()),
                "schedule": asdict(InputTokenSchedule()),
                "optimizer": asdict(
                    optimizer_config(
                        reference_tokens_per_update=REFERENCE_TOKENS_PER_UPDATE,
                        multiplier=config.lr_multiplier,
                    )
                ),
                "reference_tokens_per_update": REFERENCE_TOKENS_PER_UPDATE,
                "objective_source_sha256": objective_code,
                "validation_manifests": {
                    n: inputs.manifest["files"][n]["sha256"]
                    for n in (
                        "validation/primary_validation_512.jsonl",
                        "audit/validation_disjoint_512.json",
                        "audit/fim_validation_pairs.jsonl",
                    )
                },
                "norm_policy": asdict(NormPolicy()),
                "stop_policy": "first_complete_global_batch_at_or_above_nominal_input_budget",
            }
        )
    )


def permanent_events(previous: int, current: int) -> tuple[str, ...]:
    events = tuple(
        e for e in validation_events(previous, current) if e in ("warmup_end", "final")
    )
    return events + tuple(
        f"quarter_{i}"
        for i in (1, 2, 3)
        if previous < TRAINING_TOKENS * i // 4 <= current
    )


def metric_values(
    metrics: dict[str, Any],
    norms: dict[str, Any],
    batch: TrainingBatch,
    *,
    collect_norms: bool,
) -> dict[str, float | int]:
    """Pack scalar transfers and retain exact int64 routing totals on the host."""
    required = {
        MOE_DROPPED_ASSIGNMENTS_METRIC,
        MOE_SENDER_DROPPED_ASSIGNMENTS_METRIC,
        MOE_RECEIVER_DROPPED_ASSIGNMENTS_METRIC,
        MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC,
        MOE_VALID_ASSIGNMENTS_METRIC,
        "train/router/routing_counts_per_layer",
    }
    selected = {
        k: v
        for k, v in metrics.items()
        if k in required or (isinstance(v, jax.Array) and v.ndim == 0)
    }
    if collect_norms:
        selected.update(norms)
    host = copy_metrics(selected, pack_scalars=True)
    values = _drop_metrics(
        host[MOE_DROPPED_ASSIGNMENTS_METRIC],
        host[MOE_SENDER_DROPPED_ASSIGNMENTS_METRIC],
        host[MOE_RECEIVER_DROPPED_ASSIGNMENTS_METRIC],
        host[MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC],
        host[MOE_VALID_ASSIGNMENTS_METRIC],
        batch_size=64,
        sequence_length=batch.bucket,
        top_k=8,
        num_layers=8,
    )
    actual = batch.counts["dna_tokens"] + batch.counts["text_tokens"]
    if (
        values[MOE_DROPPED_ASSIGNMENTS_METRIC]
        or values[MOE_VALID_ASSIGNMENTS_METRIC] != actual * 64
    ):
        raise ValueError("Router dropped real inputs or changed valid-token accounting")
    values.update({k: float(v) for k, v in host.items() if np.ndim(v) == 0})
    # Keep exact counts instead of replacing them with floating-point scalars.
    values[MOE_VALID_ASSIGNMENTS_METRIC] = actual * 64
    counts = np.asarray(host["train/router/routing_counts_per_layer"])
    for layer, row in enumerate(counts):
        total = max(float(row.sum()), 1)
        values[f"train/router/layer_{layer}/active_experts"] = int(
            np.count_nonzero(row)
        )
        values[f"train/router/layer_{layer}/max_to_mean_load"] = float(
            row.max() * len(row) / total
        )
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Nonfinite training loss, routing or norm metric")
    return values


def _run_training(config: TrainingConfig) -> None:
    os.environ.update(
        WANDB_BASE_URL=WANDB_BASE_URL, WANDB_ENTITY="eric-czech", WANDB_PROJECT="marin"
    )
    mp = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    wandb_config = WandbConfig(
        entity="eric-czech",
        project="marin",
        name=config.run_id,
        id=config.run_id,
        group="dna-exp593"
        if config.validation_updates is None
        else "dna-exp593-validation",
        background=False,
        save_code=False,
        tags=[
            "dna-exp593",
            "H100",
            config.cluster,
            "production" if config.validation_updates is None else "validation",
        ],
        replicate_path=config.checkpoint_root,
        resume="must" if config.standby_id is not None else "allow",
    )
    trainer = TrainerConfig(
        id=config.standby_id or config.run_id,
        seed=0,
        train_batch_size=64,
        num_train_steps=MAX_UPDATES,
        require_accelerator=True,
        mp=mp,
        use_explicit_mesh_axes=True,
        watch=WatchConfig(interval=0, watch_targets=[]),
        tracker=NoopConfig() if config.standby_id is not None else wandb_config,
    )
    trainer.initialize()
    if jax.device_count() != config.nodes * 8 or not all(
        "H100" in d.device_kind for d in jax.devices()
    ):
        raise ValueError("Training did not receive the requested H100 gang")
    warming = config.standby_id is not None
    if warming and primary_json(
        lambda: committed_promotion(
            config.control_root,
            standby_id=config.standby_id,
            old_root=config.standby_from_job,
            run_id=config.run_id,
        )
    ):
        # Iris can restart the same serialized standby entry point after it has
        # become production. Preserve control ownership but use normal recovery,
        # never the possibly pruned original disposable source or readiness path.
        warming = False
        tracker.set_global_tracker(wandb_config.init(config.run_id))
    tracker.log_configuration(config)
    log_step = 0 if warming else wandb_history_offset()
    tracker.log_summary({"training/phase": "loading_inputs"})
    inputs = PreparedInputs.open(Path("/tmp/exp593-inputs"))
    encoder = load_encoder()
    bindings = scientific_bindings(config, inputs, encoder)
    if warming:
        primary_json(
            lambda: asdict(
                checked_checkpoint(
                    config.checkpoint_root, config.standby_checkpoint, bindings
                )
            )
        )
        checkpoint = config.standby_checkpoint
    else:
        checkpoint = primary_json(
            lambda: bind_and_discover(config.checkpoint_root, bindings)
        )
    tracker.log_hyperparameters({"bindings": bindings})
    # Every worker warms the same verified row groups. Eight workers together
    # require about 173 GB of decoded DNA, within the 600 GB node request.
    dataset = open_corpus(inputs, "train", 0, preload=True, progress=input_progress)
    augmenter = Augmenter(
        encoder,
        AugmentationConfig(),
        annotations=load_annotations(inputs.path("annotations/train.parquet")),
    )
    panels = ValidationPanels.open(inputs, encoder)
    mesh = replicated_mesh()
    evaluator = Evaluator(panels, encoder, mesh, mp, progress=evaluation_progress)
    development = DevelopmentEvaluator(
        fetch_development(Path("/tmp/exp593-development"), encoder),
        encoder,
        mesh,
        mp,
        progress=evaluation_progress,
    )
    development_root = (
        config.checkpoint_root.removesuffix("/checkpoints") + "/development"
    )
    development_protocol = None
    if not warming:
        development_protocol = primary_json(
            lambda: publish_development_protocol(development_root)
        )
        tracker.log_summary({"eval/development_protocol": development_protocol["uri"]})
    model = replace(
        local_model(d768_config("pretrained")),
        compact_input_lookup_ids=(),
        compact_input_lookup_one_hot=False,
    )
    opt_config = optimizer_config(
        reference_tokens_per_update=REFERENCE_TOKENS_PER_UPDATE,
        multiplier=config.lr_multiplier,
    )
    optimizer = match_gradient_placement(
        opt_config.build(MAX_UPDATES, ns4d_impl=sharded_expert_ns(mesh))
    )
    schedule = InputTokenSchedule()
    norm_policy = NormPolicy()

    with jax.set_mesh(mesh):
        tracker.log_summary(
            {
                "training/phase": f"loading_{config.initialization}_checkpoint",
                "training/source_checkpoint": config.source_checkpoint,
                "training/source_metadata_digest": config.source_metadata_digest,
                "training/source_state_import": "weights_only",
                "training/initialization": config.initialization,
                "training/lr_multiplier": config.lr_multiplier,
            }
        )
        weights = restore_weights(
            config.source_checkpoint, config.source_metadata_digest, model, mesh
        )
        state = eqx.filter_jit(
            lambda w: fresh_state(w, optimizer, mp, offload=False, retain_master=True)
        )(weights)
        del weights
        state, partitions = partition_expert_optimizer_state(state, mesh)
        clock = TokenClock()
        if checkpoint is not None:
            tracker.log_summary(
                {
                    "training/phase": "restoring",
                    "training/resume_checkpoint": checkpoint,
                }
            )
            state, clock = restore_training_checkpoint(
                state, checkpoint, mesh, bindings
            )
        jax.block_until_ready(state)
        assert_partitioned_optimizer_state(state, mesh)
        if not bool(model_matches_fp32_master(state.params, state.master_params)):
            raise ValueError("BF16 parameters differ from FP32 masters")
        tracker.log_summary(
            {
                "training/optimizer_partitions": partitions,
                "training/resume_update": clock.updates,
            }
        )
        train_step = make_train_step(optimizer, mp, norm_policy=norm_policy)
        compiled_buckets: set[int] = set()

        if warming:
            source_clock = clock
            samples = []
            tested: set[tuple[int, bool]] = set()
            for probe in range(64):
                local = prepare_local_batch(dataset, augmenter, mesh, clock.examples)
                batch = assemble_distributed_batch(local, mesh)
                collect_norms = (batch.bucket, True) not in tested
                state = replace(
                    state,
                    opt_state=set_token_learning_rates(
                        state.opt_state, opt_config, schedule, clock
                    ),
                )
                started = time.monotonic()
                state, metrics, norms = train_step(
                    state, batch.model_batch, collect_norms
                )
                jax.block_until_ready((state, metrics, norms))
                values = metric_values(
                    metrics, norms, batch, collect_norms=collect_norms
                )
                clock = clock.advance(**batch.counts)
                tested.add((batch.bucket, collect_norms))
                compiled_buckets.add(batch.bucket)
                samples.append(
                    {
                        "probe": probe + 1,
                        "bucket": batch.bucket,
                        "collect_norms": collect_norms,
                        "seconds": time.monotonic() - started,
                        "metrics": values,
                    }
                )
                LOGGER.info(
                    "Standby probe=%d bucket=%d norms=%s loss=%f",
                    probe + 1,
                    batch.bucket,
                    collect_norms,
                    values["train/loss"],
                )
                if probe >= 15 and len(tested) == 8:
                    break
            if len(tested) != 8 or not bool(state_all_finite(state)):
                raise ValueError(
                    "Standby failed to exercise all buckets and norm branches"
                )
            validation = evaluator.evaluate(state)
            functional, _ = development.evaluate_development(state)
            if not all(
                np.isfinite(v) for v in (*validation.values(), *functional.values())
            ):
                raise ValueError("Nonfinite standby validation")
            ready = {
                "protocol": "exp593-standby-ready-v1",
                "standby_id": config.standby_id,
                "cluster": config.cluster,
                "nodes": config.nodes,
                "source_checkpoint": checkpoint,
                "source_clock": asdict(source_clock),
                "bindings_sha256": hashlib.sha256(
                    canonical_json(bindings).encode()
                ).hexdigest(),
                "probes": samples,
                "validation": validation,
                "functional": functional,
                "production_writes": False,
            }
            ready_sha = primary_json(
                lambda: publish_once(config.control_root + "/ready.json", ready)
            )
            LOGGER.info(
                "Standby READY identity=%s receipt=%s", config.standby_id, ready_sha
            )
            # Retain the allocation, decoded corpus and compiled executables.
            # Only an explicit operator command after old-writer termination promotes it.
            deadline = time.monotonic() + 3600
            promotion = None
            while promotion is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Standby was not promoted within one hour of readiness"
                    )
                promotion = primary_json(
                    lambda: (
                        json.loads(
                            (
                                StoragePath(config.control_root) / "promote.json"
                            ).read_text()
                        )
                        if (StoragePath(config.control_root) / "promote.json").exists()
                        else None
                    )
                )
                if promotion is None:
                    time.sleep(10)
            # Broadcast metadata checks before any distributed restore.
            approved = primary_json(
                lambda: validate_promotion(
                    promotion,
                    standby_id=config.standby_id,
                    ready_sha256=ready_sha,
                    old_root=config.standby_from_job,
                    checkpoint_root=config.checkpoint_root,
                    bindings=bindings,
                    source_clock=source_clock,
                )[0]
            )
            # Release disposable canary buffers before restoring the real cursor.
            template = checkpoint_template(state, release=True)
            del state, metrics, norms
            state, clock = restore_training_checkpoint(
                template, approved, mesh, bindings
            )
            checkpoint = approved
            jax.block_until_ready(state)
            assert_partitioned_optimizer_state(state, mesh)
            if not bool(state_all_finite(state)) or not bool(
                model_matches_fp32_master(state.params, state.master_params)
            ):
                raise ValueError("Promoted state failed full-state validation")
            tracker.set_global_tracker(wandb_config.init(config.run_id))
            log_step = wandb_history_offset()
            tracker.log_configuration(config)
            tracker.log_hyperparameters({"bindings": bindings})
            development_protocol = primary_json(
                lambda: publish_development_protocol(development_root)
            )
            tracker.log_summary(
                {
                    "training/resume_checkpoint": checkpoint,
                    "training/resume_update": clock.updates,
                    "training/standby_id": config.standby_id,
                    "training/standby_ready_sha256": ready_sha,
                    "eval/development_protocol": development_protocol["uri"],
                }
            )

        def log(values: dict[str, Any]) -> None:
            nonlocal log_step
            log_step += 1
            tracker.log(
                {
                    "train/input_tokens": clock.input_tokens,
                    "train/step": clock.updates,
                    **values,
                },
                step=log_step,
            )

        def evaluate(event: str, *, functional: bool = False) -> None:
            tracker.log_summary(
                {"training/phase": "evaluating", "training/eval_event": event}
            )
            values = evaluator.evaluate(state)
            log(values)
            tracker.log_summary(
                {"training/last_eval_input_tokens": clock.input_tokens, **values}
            )
            LOGGER.info(
                "Validation complete: event=%s update=%d tokens=%d loss=%f",
                event,
                clock.updates,
                clock.input_tokens,
                values["eval/loss"],
            )
            if functional:
                values, predictions = development.evaluate_development(state)
                receipt = primary_json(
                    lambda: publish_development_result(
                        development_root,
                        clock,
                        event,
                        development_protocol,
                        bindings,
                        values,
                        predictions,
                    )
                )
                log(values)
                tracker.log_summary(
                    {**values, "eval/development_result": receipt["uri"]}
                )
                LOGGER.info("Functional development complete: %s", receipt["uri"])

        def save(permanent: bool) -> str:
            if not bool(state_all_finite(state)):
                raise ValueError("Refusing to save a nonfinite optimizer state")
            destination = primary_json(
                lambda: prepare_checkpoint(config.checkpoint_root, clock)
            )
            tracker.log_summary({"training/phase": "checkpointing"})
            save_training_checkpoint(
                state, clock, destination, bindings, temporary=not permanent
            )
            digest = primary_json(
                lambda: commit_and_prune(
                    config.checkpoint_root, destination, clock, bindings
                )
            )
            tracker.log_summary(
                {
                    "training/latest_checkpoint": destination,
                    "training/latest_checkpoint_metadata_sha256": digest,
                }
            )
            if permanent:
                tracker.log_summary(
                    {"training/latest_permanent_checkpoint": destination}
                )
            return destination

        evaluate("initial" if checkpoint is None else "resume", functional=True)
        last_save = time.monotonic()
        # Prefetch only host rendering. Ownership reductions and all device
        # collectives remain ordered on this thread, including eval/checkpoints.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = (
                None
                if config.complete(clock)
                else pool.submit(
                    prepare_local_batch, dataset, augmenter, mesh, clock.examples
                )
            )
            while not config.complete(clock):
                start = time.monotonic()
                local = future.result()
                batch = assemble_distributed_batch(local, mesh)
                loaded = time.monotonic()
                next_clock = clock.advance(**batch.counts)
                future = (
                    None
                    if config.complete(next_clock)
                    else pool.submit(
                        prepare_local_batch,
                        dataset,
                        augmenter,
                        mesh,
                        next_clock.examples,
                    )
                )
                factor = schedule.factor(clock.input_tokens)
                state = replace(
                    state,
                    opt_state=set_token_learning_rates(
                        state.opt_state, opt_config, schedule, clock
                    ),
                )
                collect_norms = norm_policy.due(next_clock.updates)
                first_compile = batch.bucket not in compiled_buckets
                state, metrics, norms = train_step(
                    state, batch.model_batch, collect_norms
                )
                jax.block_until_ready((state, metrics, norms))
                compiled_buckets.add(batch.bucket)
                elapsed = time.monotonic() - start
                previous = clock
                clock = next_clock
                values = metric_values(
                    metrics, norms, batch, collect_norms=collect_norms
                )
                values.update(
                    {
                        "run_progress": min(
                            clock.input_tokens / TRAINING_TOKENS, 0.999999
                        ),
                        "train/total_progress": clock.input_tokens / TRAINING_TOKENS,
                        "train/dna_tokens": clock.dna_tokens,
                        "train/text_tokens": clock.text_tokens,
                        "train/loss_targets": clock.loss_targets,
                        "train/examples": clock.examples,
                        "train/fim_examples": clock.fim_examples,
                        "train/rc_examples": clock.rc_examples,
                        "train/fim_fraction": clock.fim_examples / clock.examples,
                        "train/rc_fraction": clock.rc_examples / clock.examples,
                        "train/source_row_passes": clock.examples / 2_638_656,
                        "train/dna_base_epochs": clock.dna_tokens / 21_615_869_952,
                        "train/muon_lr": opt_config.learning_rate * factor,
                        "train/adam_lr": opt_config.adam_lr * factor,
                        "train/bucket": batch.bucket,
                        "train/padding_fraction": 1
                        - (clock.input_tokens - previous.input_tokens)
                        / (64 * batch.bucket),
                        "throughput/step_seconds": elapsed,
                        "throughput/includes_first_compile": int(first_compile),
                        "throughput/loading_seconds": loaded - start,
                        "throughput/tokens_per_second": (
                            clock.input_tokens - previous.input_tokens
                        )
                        / elapsed,
                        **{
                            f"augmentation/{k}_examples": v
                            for k, v in batch.mixture.items()
                        },
                    }
                )
                if first_compile:
                    values[f"compile/bucket_{batch.bucket}_seconds"] = (
                        time.monotonic() - loaded
                    )
                log(values)
                tracker.log_summary(
                    {
                        "training/phase": "training",
                        "training/last_input_fingerprint": batch.input_fingerprint,
                    }
                )
                if clock.updates % 100 == 0 or clock.updates < 5:
                    LOGGER.info(
                        "Update=%d tokens=%d loss=%.6f tokens/s=%.0f",
                        clock.updates,
                        clock.input_tokens,
                        values["train/loss"],
                        values["throughput/tokens_per_second"],
                    )
                events = validation_events(previous.input_tokens, clock.input_tokens)
                permanent = permanent_events(previous.input_tokens, clock.input_tokens)
                finished = config.complete(clock)
                # Drain host work before sharing the text encoder with eval.
                if (events or permanent or finished) and future is not None:
                    future.result()
                if events or permanent or finished:
                    evaluate(
                        ",".join(events + permanent) or "bounded_validation_end",
                        functional=finished
                        or any(e.startswith("quarter_") for e in permanent),
                    )
                due = primary_json(
                    lambda saved_at=last_save, milestone=bool(permanent), done=finished: (
                        time.monotonic() - saved_at >= 1800 or milestone or done
                    )
                )
                if due:
                    checkpoint = save(bool(permanent) or finished)
                    last_save = time.monotonic()
                if (
                    config.standby_id is not None
                    and clock.updates % 100 == 0
                    and not finished
                ):
                    stop = primary_json(
                        lambda: (
                            json.loads(
                                (
                                    StoragePath(config.control_root) / "stop.json"
                                ).read_text()
                            )
                            if (StoragePath(config.control_root) / "stop.json").exists()
                            else None
                        )
                    )
                    if stop is not None:
                        if stop != {
                            "protocol": "exp593-checkpoint-and-exit-v1",
                            "standby_id": config.standby_id,
                        }:
                            raise ValueError("Invalid handoff request")
                        if future is not None:
                            future.result()
                        if not due:
                            checkpoint = save(True)
                        ack = {"checkpoint": checkpoint, "clock": asdict(clock)}
                        primary_json(
                            lambda receipt=ack: publish_once(
                                config.control_root + "/stopped.json", receipt
                            )
                        )
                        tracker.log_summary(
                            {
                                "training/phase": "handoff_saved",
                                **{"training/handoff_" + k: v for k, v in ack.items()},
                            }
                        )
                        tracker.current_tracker().finish()
                        return
            # A recovered final checkpoint can already satisfy the horizon.
            if checkpoint is None:
                checkpoint = save(True)
        if (
            config.validation_updates is None
            and not TRAINING_TOKENS
            <= clock.input_tokens
            < TRAINING_TOKENS + REFERENCE_TOKENS_PER_UPDATE
        ):
            raise ValueError("Final complete-batch input budget differs")
        tracker.log_summary(
            {
                "training/phase": "complete",
                "training/final_checkpoint": checkpoint,
                "training/final_clock": asdict(clock),
                "training/nominal_input_tokens": TRAINING_TOKENS,
                "training/input_token_overshoot": max(
                    0, clock.input_tokens - TRAINING_TOKENS
                ),
            }
        )
        log({"run_progress": 1.0})
    tracker.current_tracker().finish()


def run_training(config: TrainingConfig) -> None:
    try:
        _run_training(config)
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
                LOGGER.exception("Could not publish failure to W&B")
        raise


def dispatch(config: TrainingConfig) -> None:
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
            "CW_KEY_ID": os.environ["CW_KEY_ID"],
            "CW_KEY_SECRET": os.environ["CW_KEY_SECRET"],
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
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
            name=f"train-{config.run_id}",
            entrypoint=Entrypoint.from_callable(run_training, args=[config]),
            resources=resources,
            processes_per_task=8,
            environment=create_environment(env_vars=env, extras=["gpu"]),
            priority=priority_band_value("batch"),
            timeout=Duration.from_hours(
                2 if config.validation_updates is not None else 14 * 24
            ),
            max_retries_failure=0,
            max_task_failures=0,
        )
    )
    job.wait(raise_on_failure=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cluster", required=True, choices=["cw-rno2a", "cw-us-east-02a"]
    )
    parser.add_argument("--nodes", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--run-id")
    parser.add_argument("--initialization", choices=["text", "dna216b"], default="text")
    parser.add_argument("--validation-updates", type=int)
    parser.add_argument("--standby-id")
    parser.add_argument("--standby-checkpoint")
    parser.add_argument("--standby-from-job")
    arguments = vars(parser.parse_args())
    if arguments["run_id"] is None:
        arguments["run_id"] = (
            PRODUCTION_RUN_ID
            if arguments["initialization"] == "text"
            else DNA216B_RUN_ID
        )
    dispatch(TrainingConfig(**arguments))


if __name__ == "__main__":
    main()
