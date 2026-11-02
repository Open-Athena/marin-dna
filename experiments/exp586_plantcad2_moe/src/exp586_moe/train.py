"""Six d768 DNA trials using the native Hero EP step and exact token clocks."""

import argparse
import dataclasses
import gc
import hashlib
import json
import logging
import math
import os
import resource
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import fsspec
import jax
import jmp
import numpy as np
from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.device_flops import device_flops_for_jax_device
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
from levanter.checkpoint import discover_checkpoint_candidates
from levanter.grug.grug_moe import (
    MOE_DROPPED_ASSIGNMENTS_METRIC,
    MOE_VALID_ASSIGNMENTS_METRIC,
)
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.training.training import resolve_training_env
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from exp586_moe.config import (
    CONTINUATION_SOURCE_CHECKPOINT,
    CONTINUATION_SOURCE_METADATA_SHA256,
    CONTINUATION_SOURCE_SCIENTIFIC_CONFIG_SHA256,
    CONTINUATION_SOURCE_TOKENS,
    CONTINUATION_SOURCE_UPDATES,
    CONTINUATION_TARGET_TOKENS,
    CONTINUATION_TOTAL_UPDATES,
    EPOCH_TOKENS,
    GLOBAL_BATCH_SIZE,
    ONE_TRILLION_PEAK_CHECKPOINT_UPDATE,
    ONE_TRILLION_TARGET_TOKENS,
    ONE_TRILLION_TOTAL_UPDATES,
    ONE_TRILLION_WARMUP_UPDATES,
    PEAK_CHECKPOINT_UPDATE,
    SOURCE_CHECKPOINT,
    SOURCE_METADATA_DIGEST,
    TOKENS_PER_UPDATE,
    TOTAL_UPDATES,
    TRAINING_TOKENS,
    d768_config,
)
from exp586_moe.corpus import corpus_digest, distributed_batch, open_dataset
from exp586_moe.data import (
    PRETRAINED_DNA_INPUT_TOKEN_IDS,
    pretrained_dna_input_token_ids,
    pretrained_dna_tokenizer,
    scratch_tokenizer,
    tokenizer_digest,
)
from exp586_moe.data_parallel import (
    assert_physical_replicas,
    assert_replicated_state,
    local_model,
    replicated_mesh,
)
from exp586_moe.perf4_muon import node_local_expert_ns
from exp586_moe.perf5_optimizer import (
    assert_partitioned_optimizer_state,
    match_gradient_placement,
    model_matches_fp32_master,
    partition_expert_optimizer_state,
    persistent_state_bytes,
    replicate_optimizer_state_for_validation,
    sharded_expert_ns,
)
from exp586_moe.routing_probe import routing_diagnostics
from exp586_moe.schedule import (
    ContinuationTokenSchedule,
    HeroTokenSchedule,
    TokenClock,
    optimizer_config,
    set_token_learning_rates,
)
from exp586_moe.state import (
    fresh_state,
    restore_dna_checkpoint,
    save_dna_checkpoint,
    state_all_finite,
    validate_restored_training_state,
)
from exp586_moe.tracking import wandb_history_offset
from experiments.grug.dispatch import _forwarded_env_vars
from experiments.grug.moe_hero_ep.model import Transformer, apply_qb_betas
from experiments.grug.moe_hero_ep.train import (
    MasterParamMode,
    _apply_hero_ep_runtime_defaults,
    _compute_flops,
    _make_train_step,
    grug_trainer_mesh_config,
)
from experiments.grug.moe_hero_ep.weights import restore_weights

BATCH_SIZE = GLOBAL_BATCH_SIZE
COOLDOWN_UPDATE = PEAK_CHECKPOINT_UPDATE
VALIDATION_EXAMPLES = 512
ARTIFACT_DATE = "2026.10.08"
CONTINUATION_ARTIFACT_DATE = "2026.10.19"
ONE_TRILLION_ARTIFACT_DATE = "2026.10.20"
PERMANENT_CHECKPOINT_COUNT = 8
ONE_TRILLION_PERMANENT_CHECKPOINT_COUNT = 12
PERF4_RUNTIME_TRANSITION_ID = "full-replica-dp-perf4-v1-20261101"
PERF4_SOURCE_COMMIT = "34aa317617c3a0ede609c76b01e44312dbd864e2"
PERF4_RUNTIME_CODE_SHA256 = (
    "16276343806f2c02ef7eccf72f20e559620a5a6605af6fc37a308d0099bd2d94"
)
PERF5_RUNTIME_TRANSITION_ID = "full-replica-dp-perf5-v1-20261102"
PERF5_SOURCE_COMMIT = "pending-production-integration"
PREVIOUS_DP_RUNTIME_TRANSITION_ID = "full-replica-dp-v1-20261031"
PREVIOUS_DP_RUNTIME_CODE_SHA256 = (
    "8ca27ecf5e219150606cccbb6f2835a06c51971def1cf88c2f23cddfb833bf5b"
)
PERF4_READER_OPTIONS = {
    "cache_files": True,
    "cache_file_workers": 4,
    "preload_all": True,
    "preload_workers": 4,
    "cache_groups": 4096,
}
PERF4_COMPILER_OPTIONS = {
    "xla_gpu_enable_pdl": False,
    "xla_gpu_enable_pdl_launch": False,
}

PERF5_RUNTIME_SOURCE_PATHS = (
    "src/exp586_moe/train.py",
    "src/exp586_moe/perf5_optimizer.py",
    "src/exp586_moe/perf5_metrics.py",
    "src/exp586_moe/routing_probe.py",
    "src/experiments/grug/moe_hero_ep/train.py",
)


def perf5_runtime_source_sha256() -> str:
    """Bind the deployed digest to every source file changed by perf5."""
    project = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in PERF5_RUNTIME_SOURCE_PATHS:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((project / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_permanent_checkpoint_updates(
    *,
    start_update: int,
    total_updates: int,
    peak_update: int,
    count: int = PERMANENT_CHECKPOINT_COUNT,
) -> frozenset[int]:
    """Retain the LR peak, evenly spaced stage milestones, and the end."""
    span = total_updates - start_update
    if count < 3 or span <= 0 or not start_update < peak_update <= total_updates:
        raise ValueError("Invalid permanent-checkpoint stage boundaries")
    milestones = {
        start_update + math.ceil(span * checkpoint_index / (count - 1))
        for checkpoint_index in range(1, count - 1)
    }
    result = frozenset(milestones | {peak_update, total_updates})
    if len(result) != count:
        raise ValueError("Permanent-checkpoint milestones overlap")
    return result


PERMANENT_CHECKPOINT_UPDATES = build_permanent_checkpoint_updates(
    start_update=0,
    total_updates=TOTAL_UPDATES,
    peak_update=PEAK_CHECKPOINT_UPDATE,
)


def theoretical_flops(device_kind: str, device_count: int) -> tuple[float, float]:
    """Return the standard dense-BF16 peak for one device and the whole mesh."""
    if device_count <= 0:
        raise ValueError("Device count must be positive")
    per_device = device_flops_for_jax_device(device_kind)
    if per_device is None:
        raise ValueError(f"Unknown theoretical FLOPs for {device_kind}")
    return float(per_device), float(per_device * device_count)


def throughput_flop_metrics(
    *,
    flops_per_example: float,
    completed_examples: int,
    batch_size: int,
    elapsed_seconds: float,
    theoretical_flops_total: float,
) -> dict[str, float]:
    """Match Levanter's analytic MFU convention using the durable example clock."""
    if (
        flops_per_example <= 0
        or completed_examples < 0
        or batch_size <= 0
        or elapsed_seconds <= 0
        or theoretical_flops_total <= 0
    ):
        raise ValueError("FLOP metrics require positive rates and a nonnegative clock")
    model_flops_per_second = flops_per_example * batch_size / elapsed_seconds
    return {
        "throughput/gflops_per_second": model_flops_per_second / 1e9,
        "throughput/mfu": model_flops_per_second / theoretical_flops_total * 100.0,
        "throughput/total_gflops": flops_per_example * completed_examples / 1e9,
    }


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    condition: Literal["scratch", "pretrained"]
    lr_multiplier: float
    cluster: str
    nodes: int
    stage: Literal["s01", "s02", "t1"] = "s01"
    seed: int = 0
    checkpoint_interval_seconds: int = 900
    recovery_checkpoint_delay_updates: int = 25
    moe_backend: Literal["pooled", "fixed", "ring"] | None = None
    validation_updates: int | None = None
    validation_id: str | None = None
    validation_execution_updates: int | None = None
    validation_source_checkpoint: str | None = None
    validation_source_metadata_sha256: str | None = None
    validation_source_update: int | None = None
    runtime_state_mode: Literal["fp32_pinned_host", "fp32_device"] = "fp32_pinned_host"
    runtime_prefetch: bool = False
    runtime_transition_id: str | None = None
    runtime_code_sha256: str | None = None
    runtime_data_parallel: bool = False
    runtime_perf4: bool = False
    runtime_perf5: bool = False
    profiling_updates: int | None = None
    profiling_id: str | None = None
    profiling_batch_size: int = 8
    profiling_state_mode: Literal["production", "device"] = "production"
    profiling_prefetch: bool = False
    profiler_start_step: int = 20
    profiler_num_steps: int = 5

    def __post_init__(self) -> None:
        multipliers = {
            "pretrained": (0.5, 1.0, 2.0),
            "scratch": (1.0, 3.0, 10.0),
        }
        if (
            self.condition not in multipliers
            or self.lr_multiplier not in multipliers[self.condition]
            or self.seed != 0
        ):
            raise ValueError("Select one of the six authorized trials")
        if self.stage not in ("s01", "s02", "t1"):
            raise ValueError("Unknown training stage")
        if self.stage in ("s02", "t1") and (
            self.condition != "pretrained" or self.lr_multiplier != 1.0
        ):
            raise ValueError("This stage is authorized only for pretrained 1x")
        if self.cluster not in ("cw-rno2a", "cw-us-east-02a") or self.nodes not in (
            1,
            2,
            4,
            8,
        ):
            raise ValueError("Training requires an 8-, 16-, 32- or 64-H100 placement")
        if self.checkpoint_interval_seconds <= 0:
            raise ValueError("Checkpoint interval must be positive")
        if self.recovery_checkpoint_delay_updates <= 0:
            raise ValueError("Recovery checkpoint delay must be positive")
        if self.moe_backend not in (None, "pooled", "fixed", "ring") or (
            self.condition == "pretrained" and self.moe_backend not in (None, "pooled")
        ):
            raise ValueError("Select an approved MoE runtime backend")
        if (self.validation_updates is None) != (self.validation_id is None):
            raise ValueError("Validation updates and ID must be set together")
        validation_source = (
            self.validation_source_checkpoint,
            self.validation_source_metadata_sha256,
            self.validation_source_update,
        )
        if any(value is not None for value in validation_source) != all(
            value is not None for value in validation_source
        ):
            raise ValueError(
                "Validation source checkpoint, metadata digest, and update must be set together"
            )
        if (self.profiling_updates is None) != (self.profiling_id is None):
            raise ValueError("Profiling updates and ID must be set together")
        if self.validation_updates is not None and self.profiling_updates is not None:
            raise ValueError("Validation and profiling modes are mutually exclusive")
        if self.validation_updates is not None:
            if (
                self.stage not in ("s02", "t1")
                or not 1 <= self.validation_updates <= 25
            ):
                raise ValueError("Validation must run 1 through 25 updates")
            if (
                not self.validation_id
                or not self.validation_id.replace("-", "").isalnum()
            ):
                raise ValueError(
                    "Validation ID must contain only letters, digits, and hyphens"
                )
        if self.validation_execution_updates is not None and (
            self.validation_updates is None
            or not 1 <= self.validation_execution_updates <= self.validation_updates
        ):
            raise ValueError(
                "Validation execution updates must be within the validation contract"
            )
        if self.validation_source_checkpoint is not None:
            expected = f"{self.production_checkpoint_root}/step-{self.validation_source_update}"
            if (
                self.stage != "t1"
                or self.validation_updates is None
                or self.nodes != 8
                or (
                    self.validation_source_checkpoint != expected
                    and not self.validation_source_checkpoint.startswith(
                        expected + "-retry-"
                    )
                )
                or not isinstance(self.validation_source_update, int)
                or self.validation_source_update < 1
            ):
                raise ValueError(
                    "A validation source must be an exact 64-H100 main-run checkpoint"
                )
            if len(self.validation_source_metadata_sha256 or "") != 64 or any(
                character not in "0123456789abcdef"
                for character in self.validation_source_metadata_sha256 or ""
            ):
                raise ValueError("Validation source metadata digest must be SHA-256")
        if self.runtime_state_mode not in ("fp32_pinned_host", "fp32_device"):
            raise ValueError("Unknown production runtime state mode")
        if (self.runtime_transition_id is None) != (self.runtime_code_sha256 is None):
            raise ValueError(
                "Runtime transition ID and code digest must be set together"
            )
        if self.runtime_transition_id is not None:
            if not self.runtime_transition_id.replace("-", "").isalnum():
                raise ValueError(
                    "Runtime transition ID must contain only letters, digits, and hyphens"
                )
            if len(self.runtime_code_sha256 or "") != 64 or any(
                c not in "0123456789abcdef" for c in self.runtime_code_sha256 or ""
            ):
                raise ValueError("Runtime code digest must be lowercase SHA-256")
        if self.runtime_data_parallel and (
            self.stage != "t1"
            or self.nodes != 8
            or self.runtime_state_mode != "fp32_device"
            or not self.runtime_prefetch
            or self.runtime_transition_id is None
        ):
            raise ValueError(
                "Full-replica DP requires stage t1 on 64 H100s, FP32-device "
                "state, prefetch, and transition metadata"
            )
        if self.runtime_perf4 and not self.runtime_data_parallel:
            raise ValueError("Perf4-derived production requires full-replica DP")
        if self.runtime_perf5 and (
            not self.runtime_perf4
            or self.runtime_transition_id != PERF5_RUNTIME_TRANSITION_ID
            or self.runtime_code_sha256 != perf5_runtime_source_sha256()
        ):
            raise ValueError(
                "Perf5 production requires the perf4 base, its exact transition ID, "
                "and a digest of every changed runtime source"
            )
        if (
            self.runtime_perf4
            and not self.runtime_perf5
            and (self.runtime_transition_id != PERF4_RUNTIME_TRANSITION_ID)
        ):
            raise ValueError(
                "Perf4 production requires full-replica DP and its exact transition ID"
            )
        if self.profiling_updates is not None:
            if self.stage != "t1" or not 10 <= self.profiling_updates <= 200:
                raise ValueError(
                    "Profiling must run 10 through 200 updates of stage t1"
                )
            if (
                not self.profiling_id
                or not self.profiling_id.replace("-", "").isalnum()
            ):
                raise ValueError(
                    "Profiling ID must contain only letters, digits, and hyphens"
                )
            if (
                self.profiler_start_step < 1
                or self.profiler_num_steps < 1
                or self.profiler_start_step + self.profiler_num_steps
                > self.profiling_updates
            ):
                raise ValueError("Profiler window must fit inside the profiling run")
            if self.profiling_batch_size != 8:
                raise ValueError("One-node profiling uses one example per H100")
            if self.profiling_state_mode not in ("production", "device"):
                raise ValueError("Unknown profiling state mode")
            if (
                self.runtime_state_mode != "fp32_pinned_host"
                or self.runtime_prefetch
                or self.runtime_transition_id is not None
                or self.runtime_data_parallel
            ):
                raise ValueError(
                    "Production runtime choices cannot be combined with profiling mode"
                )
        elif (
            self.profiling_batch_size != 8
            or self.profiling_state_mode != "production"
            or self.profiling_prefetch
        ):
            raise ValueError("Profiling runtime choices require profiling mode")

    @property
    def context_axis_size(self) -> int:
        return 1

    @property
    def runtime_batch_size(self) -> int:
        return (
            self.profiling_batch_size
            if self.profiling_updates is not None
            else BATCH_SIZE
        )

    @property
    def effective_runtime_state_mode(
        self,
    ) -> Literal["fp32_pinned_host", "fp32_device"]:
        if self.profiling_updates is not None:
            return (
                "fp32_device"
                if self.profiling_state_mode == "device"
                else "fp32_pinned_host"
            )
        return self.runtime_state_mode

    @property
    def prefetch_depth(self) -> int:
        enabled = (
            self.profiling_prefetch
            if self.profiling_updates is not None
            else self.runtime_prefetch
        )
        return int(enabled)

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        metadata = {
            "state_mode": self.effective_runtime_state_mode,
            "prefetch_depth": self.prefetch_depth,
            "transition_id": self.runtime_transition_id,
            "code_sha256": self.runtime_code_sha256,
            "cluster": self.cluster,
            "nodes": self.nodes,
        }
        if self.runtime_data_parallel:
            metadata.update(
                {
                    "parallelism": "full-replica-data-parallel",
                    "mesh": {
                        "replica_dcn": 8 * self.nodes,
                        "data": 1,
                        "context": 1,
                        "expert": 1,
                        "model": 1,
                    },
                    "moe_implementation": "scatter",
                    "compact_input_lookup_ids": PRETRAINED_DNA_INPUT_TOKEN_IDS,
                    "compact_input_lookup_one_hot": True,
                    "persistent_state_replication": "asserted",
                    "optimizer_newton_schulz_scope": "fully-replicated-local",
                }
            )
        if self.runtime_perf4:
            metadata.update(
                {
                    "perf4_source_commit": PERF4_SOURCE_COMMIT,
                    "reader_options": PERF4_READER_OPTIONS,
                    "compiler_options": PERF4_COMPILER_OPTIONS,
                    "optimizer_newton_schulz_scope": "node-local-8-gpu",
                    "optimizer_group_size": 8,
                    "optimizer_normalize_before_split": False,
                    "optimizer_slice_before_cast": True,
                    "optimizer_cast_before_gather": False,
                    "physical_replica_checks": "restore-and-checkpoint-boundaries",
                }
            )
        if self.runtime_perf5:
            metadata.update(
                {
                    "perf5_source_commit": PERF5_SOURCE_COMMIT,
                    "optimizer_state_placement": (
                        "expert-fp32-partitioned-on-replica-dcn"
                    ),
                    "partitioned_master_banks": 3,
                    "partitioned_momentum_banks": 3,
                    "compute_model_replication": "complete-bf16-on-every-gpu",
                    "optimizer_hyperball_norm": "global-layer-expert-bank",
                    "checkpoint_optimizer_state": "canonical-fully-replicated",
                    "checkpoint_repartition_after_restore": True,
                    "diagnostic_transfer": (
                        "consumed-fp16-bf16-fp32-scalars-packed-exactly"
                    ),
                    "physical_replica_checks": (
                        "canonical-restore-and-checkpoint-boundaries"
                    ),
                    "runtime_source_paths": PERF5_RUNTIME_SOURCE_PATHS,
                }
            )
        return metadata

    @property
    def base_run_id(self) -> str:
        multiplier = {0.5: "0p5", 1.0: "1", 2.0: "2", 3.0: "3", 10.0: "10"}[
            self.lr_multiplier
        ]
        base = (
            f"exp586-plantcad2-d768-{self.condition}-lrm{multiplier}-seed{self.seed}-v1"
        )
        if self.stage == "s02":
            base = base.removesuffix("-v1") + "-s02-v1"
        elif self.stage == "t1":
            base = base.removesuffix("-v1") + "-1t-v1"
        return base

    @property
    def run_id(self) -> str:
        base = self.base_run_id
        if self.validation_id is not None:
            base += f"-smoke-{self.validation_id}"
        elif self.profiling_id is not None:
            base += f"-profile-{self.profiling_id}"
        return base

    @property
    def production_checkpoint_root(self) -> str:
        if self.stage != "t1":
            raise ValueError("Only stage t1 has a validation source checkpoint")
        return (
            "s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/checkpoints/"
            f"{self.base_run_id}/{ONE_TRILLION_ARTIFACT_DATE}/checkpoints"
        )

    @property
    def checkpoint_root(self) -> str:
        if self.validation_id is not None or self.profiling_id is not None:
            return (
                "s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/"
                f"exp586_plantcad2_moe/{self.run_id}/checkpoints"
            )
        artifact_date = {
            "s01": ARTIFACT_DATE,
            "s02": CONTINUATION_ARTIFACT_DATE,
            "t1": ONE_TRILLION_ARTIFACT_DATE,
        }[self.stage]
        return f"s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/checkpoints/{self.run_id}/{artifact_date}/checkpoints"

    @property
    def schedule(self) -> HeroTokenSchedule | ContinuationTokenSchedule:
        if self.stage == "s02":
            return ContinuationTokenSchedule()
        if self.stage == "t1":
            return HeroTokenSchedule(
                total_updates=ONE_TRILLION_TOTAL_UPDATES,
                warmup_updates=ONE_TRILLION_WARMUP_UPDATES,
                peak_update=ONE_TRILLION_PEAK_CHECKPOINT_UPDATE,
                revision="hero-linear-1t-v1",
            )
        return HeroTokenSchedule()

    @property
    def start_update(self) -> int:
        if self.validation_source_update is not None:
            return self.validation_source_update
        return CONTINUATION_SOURCE_UPDATES if self.stage == "s02" else 0

    @property
    def start_input_tokens(self) -> int:
        if self.validation_source_update is not None:
            return self.validation_source_update * TOKENS_PER_UPDATE
        return CONTINUATION_SOURCE_TOKENS if self.stage == "s02" else 0

    @property
    def target_updates(self) -> int:
        return {
            "s01": TOTAL_UPDATES,
            "s02": CONTINUATION_TOTAL_UPDATES,
            "t1": ONE_TRILLION_TOTAL_UPDATES,
        }[self.stage]

    @property
    def target_input_tokens(self) -> int:
        return {
            "s01": TRAINING_TOKENS,
            "s02": CONTINUATION_TARGET_TOKENS,
            "t1": ONE_TRILLION_TARGET_TOKENS,
        }[self.stage]

    @property
    def optimizer_reference_tokens(self) -> int:
        return ONE_TRILLION_TARGET_TOKENS if self.stage == "t1" else TRAINING_TOKENS

    @property
    def execution_stop_update(self) -> int:
        if self.validation_execution_updates is not None:
            return self.start_update + self.validation_execution_updates
        if self.validation_updates is not None:
            return self.start_update + self.validation_updates
        if self.profiling_updates is not None:
            return self.start_update + self.profiling_updates
        return self.target_updates

    @property
    def continuation_source_checkpoint(self) -> str | None:
        return CONTINUATION_SOURCE_CHECKPOINT if self.stage == "s02" else None

    @property
    def permanent_checkpoint_updates(self) -> frozenset[int]:
        if self.profiling_updates is not None:
            return frozenset()
        if self.validation_updates is not None:
            return frozenset({self.execution_stop_update})
        if self.stage == "s01":
            return PERMANENT_CHECKPOINT_UPDATES
        return build_permanent_checkpoint_updates(
            start_update=self.start_update,
            total_updates=self.target_updates,
            peak_update=self.schedule.peak_update,
            count=(
                ONE_TRILLION_PERMANENT_CHECKPOINT_COUNT
                if self.stage == "t1"
                else PERMANENT_CHECKPOINT_COUNT
            ),
        )

    @property
    def scientific_model(self) -> Any:
        """Stable logical model contract shared by transport-compatible resumes."""
        return dataclasses.replace(
            d768_config(self.condition),
            capacity_factor=32.0,
            pooled_transport_capacity_factor=8.0,
            expert_chunks=1,
        )

    @property
    def resolved_moe_backend(self) -> Literal["pooled", "fixed", "ring"]:
        return self.moe_backend or ("ring" if self.condition == "scratch" else "pooled")

    @property
    def model(self) -> Any:
        if self.runtime_data_parallel:
            return local_model(self.scientific_model)
        implementation = {
            "pooled": "fixed_pooled_wave_all_to_all",
            "fixed": "fixed_all_to_all",
            "ring": "ring",
        }[self.resolved_moe_backend]
        return dataclasses.replace(
            self.scientific_model,
            moe_implementation=implementation,
            pooled_transport_capacity_factor=(
                8.0 if self.resolved_moe_backend == "pooled" else None
            ),
        )


def scientific_config(config: TrainingConfig, tokenizer_sha256: str) -> dict[str, Any]:
    """Immutable science and lineage; placement remains operational."""
    stage_one = {
        "schema": 1,
        "condition": config.condition,
        "seed": config.seed,
        "batch_size": BATCH_SIZE,
        "model": dataclasses.asdict(config.scientific_model),
        "optimizer": dataclasses.asdict(
            optimizer_config(
                reference_tokens_per_update=TOKENS_PER_UPDATE,
                training_tokens=config.optimizer_reference_tokens,
                multiplier=config.lr_multiplier,
            )
        ),
        "schedule": dataclasses.asdict(config.schedule),
        "optimizer_reference_tokens": config.optimizer_reference_tokens,
        "source_checkpoint": SOURCE_CHECKPOINT
        if config.condition == "pretrained"
        else None,
        "source_metadata_sha256": SOURCE_METADATA_DIGEST
        if config.condition == "pretrained"
        else None,
        "tokenizer_sha256": tokenizer_sha256,
        "corpus_sha256": corpus_digest(),
        "data_order": "exp472-block256-window512-feistel-mixture2048-seed0",
        "augmentation": "exp472-occurrence-bernoulli-472",
        "precision": "native-bf16-fp32-pinned-host-master",
        "z_loss_weight": 1e-4,
    }
    if config.stage == "s01":
        return stage_one
    if config.stage == "t1":
        stage_t1 = {
            **stage_one,
            "schema": 3,
            "stage": config.stage,
            "start_input_tokens": config.start_input_tokens,
            "target_input_tokens": config.target_input_tokens,
            "validation_updates": config.validation_updates,
        }
        if config.profiling_updates is not None:
            stage_t1.update(
                {
                    "profiling_updates": config.profiling_updates,
                    "profiling_batch_size": config.runtime_batch_size,
                    "profiling_state_mode": config.profiling_state_mode,
                    "profiling_prefetch": config.profiling_prefetch,
                }
            )
        return stage_t1
    return {
        **stage_one,
        "schema": 2,
        "stage": config.stage,
        "schedule": dataclasses.asdict(config.schedule),
        "source_checkpoint": CONTINUATION_SOURCE_CHECKPOINT,
        "source_metadata_sha256": CONTINUATION_SOURCE_METADATA_SHA256,
        "source_scientific_config_sha256": (
            CONTINUATION_SOURCE_SCIENTIFIC_CONFIG_SHA256
        ),
        "start_input_tokens": config.start_input_tokens,
        "target_input_tokens": config.target_input_tokens,
        "validation_updates": config.validation_updates,
    }


def config_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda x: x.name if isinstance(x, Enum) else str(x),
    )


def checkpoint_due(
    update: int,
    initial_update: int,
    seconds_since_save: float,
    interval_seconds: int,
    peak_update: int = COOLDOWN_UPDATE,
    recovery_delay_updates: int = 25,
    permanent_updates: frozenset[int] = PERMANENT_CHECKPOINT_UPDATES,
) -> bool:
    """Bound replay after every restore as well as during uninterrupted training."""
    return (
        is_permanent_checkpoint(update, peak_update, permanent_updates)
        or update == initial_update + recovery_delay_updates
        or seconds_since_save >= interval_seconds
    )


def is_permanent_checkpoint(
    update: int,
    peak_update: int = COOLDOWN_UPDATE,
    permanent_updates: frozenset[int] = PERMANENT_CHECKPOINT_UPDATES,
) -> bool:
    """Retain the LR peak, six horizon milestones, and the final state."""
    return update == peak_update or update in permanent_updates


def primary_json(action: Callable[[], Any]) -> Any:
    """Coordinate primary-only I/O, including failures, across the GPU gang."""
    blob = b""
    if jax.process_index() == 0:
        try:
            result = {"value": action()}
        except Exception as error:  # noqa: BLE001 -- broadcast failure before raising
            result = {"error": f"{type(error).__name__}: {error}"}
        blob = json.dumps(result).encode()
    length = int(
        multihost_utils.broadcast_one_to_all(np.asarray(len(blob), dtype=np.int32))
    )
    payload = (
        np.frombuffer(blob, dtype=np.uint8).copy()
        if blob
        else np.zeros(length, dtype=np.uint8)
    )
    result = json.loads(bytes(multihost_utils.broadcast_one_to_all(payload)))
    if "error" in result:
        raise RuntimeError(result["error"])
    return result["value"]


def bind_and_discover(
    config: TrainingConfig, binding: dict[str, Any]
) -> tuple[str, str | None]:
    serialized = config_json(binding)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    path = StoragePath(config.checkpoint_root) / "training.json"
    if path.exists():
        if path.read_text() != serialized:
            raise ValueError("Trial scientific configuration changed")
    else:
        path.write_text(serialized)
        if path.read_text() != serialized:
            raise ValueError("Scientific configuration readback failed")
    candidates = discover_checkpoint_candidates(config.checkpoint_root)
    for candidate in candidates:
        if candidate.metadata.get("scientific_config_sha256") != digest:
            raise ValueError("Checkpoint scientific configuration mismatch")
    if (
        config.validation_updates is None
        and candidates
        and candidates[-1].step >= config.schedule.peak_update
    ):
        peaks = [c for c in candidates if c.step == config.schedule.peak_update]
        if len(peaks) != 1 or peaks[0].metadata.get("is_temporary") is not False:
            raise ValueError("Permanent peak checkpoint is missing")
    return digest, candidates[-1].path if candidates else None


def validate_runtime_resume(config: TrainingConfig, checkpoint: str) -> dict[str, Any]:
    """Permit optimized runtimes only from exact compatible production lineages."""
    if not config.runtime_perf4:
        return {}
    runtime_perf5 = getattr(config, "runtime_perf5", False)
    validation_source_checkpoint = getattr(config, "validation_source_checkpoint", None)
    if validation_source_checkpoint is not None:
        valid_path = (
            checkpoint == validation_source_checkpoint
            or checkpoint.startswith(config.checkpoint_root + "/step-")
        )
    else:
        valid_path = checkpoint.startswith(config.checkpoint_root + "/step-")
    if not valid_path:
        detail = (
            "unbound checkpoint source"
            if runtime_perf5
            else "non-production checkpoint"
        )
        raise ValueError(f"Optimized production cannot restore a {detail}")
    metadata = json.loads((StoragePath(checkpoint) / "metadata.json").read_text())
    runtime = metadata.get("exp586_runtime")
    if not isinstance(runtime, dict):
        raise TypeError("Optimized production requires bound DP runtime metadata")
    transition = runtime.get("transition_id")
    code_sha256 = runtime.get("code_sha256")
    expected_code = (
        {
            PERF4_RUNTIME_TRANSITION_ID: PERF4_RUNTIME_CODE_SHA256,
            PERF5_RUNTIME_TRANSITION_ID: config.runtime_code_sha256,
        }.get(transition)
        if runtime_perf5
        else {
            PREVIOUS_DP_RUNTIME_TRANSITION_ID: PREVIOUS_DP_RUNTIME_CODE_SHA256,
            PERF4_RUNTIME_TRANSITION_ID: config.runtime_code_sha256,
        }.get(transition)
    )
    if expected_code is None or code_sha256 != expected_code:
        detail = "not compatible" if runtime_perf5 else "not perf4-compatible"
        raise ValueError(f"Checkpoint runtime transition is {detail}")
    if (
        runtime.get("state_mode") != "fp32_device"
        or runtime.get("parallelism") != "full-replica-data-parallel"
        or runtime.get("mesh")
        != {
            "replica_dcn": 64,
            "data": 1,
            "context": 1,
            "expert": 1,
            "model": 1,
        }
        or runtime.get("moe_implementation") != "scatter"
    ):
        raise ValueError("Checkpoint DP topology is not compatible")
    if (
        transition == PERF5_RUNTIME_TRANSITION_ID
        and runtime.get("checkpoint_optimizer_state") != "canonical-fully-replicated"
    ):
        raise ValueError("Perf5 checkpoint does not declare canonical optimizer state")
    return {
        "source_transition_id": transition,
        "source_code_sha256": code_sha256,
        "target_transition_id": getattr(
            config,
            "runtime_transition_id",
            PERF5_RUNTIME_TRANSITION_ID
            if runtime_perf5
            else PERF4_RUNTIME_TRANSITION_ID,
        ),
        "target_code_sha256": config.runtime_code_sha256,
    }


def checkpoint_metadata_binding(checkpoint: str) -> dict[str, Any]:
    """Read the exact metadata digest and scientific binding for a smoke source."""
    blob = (StoragePath(checkpoint) / "metadata.json").read_bytes()
    metadata = json.loads(blob)
    scientific = metadata.get("scientific_config_sha256")
    clock = metadata.get("exp586_clock")
    if not isinstance(scientific, str) or not isinstance(clock, dict):
        raise TypeError("Checkpoint metadata is missing its scientific binding")
    return {
        "metadata_sha256": hashlib.sha256(blob).hexdigest(),
        "scientific_config_sha256": scientific,
        "clock": clock,
    }


def prepare_checkpoint(config: TrainingConfig, clock: TokenClock) -> str:
    """Choose an append-only destination for this checkpoint state.

    Call through primary_json with exactly one active dispatch for this trial.
    Levanter publishes metadata.json only after every shard commits. Partial
    OCDBT writes are never reopened: a prior native TensorStore commit failure
    can leave a prefix that is unsafe to reuse even after object deletion.
    """
    check_clock(
        clock,
        min_updates=config.start_update,
        max_updates=config.execution_stop_update,
    )
    checkpoint = f"{config.checkpoint_root}/step-{clock.updates}"
    fs, path = fsspec.core.url_to_fs(checkpoint)
    fs.invalidate_cache(path)
    if fs.exists(path + "/metadata.json"):
        raise ValueError("Refusing to overwrite a committed checkpoint")
    if fs.exists(path):
        logging.getLogger(__name__).warning(
            "Leaving incomplete checkpoint untouched and selecting a fresh retry prefix: %s",
            checkpoint,
        )
        checkpoint = f"{checkpoint}-retry-{uuid.uuid4().hex[:12]}"
        fs, path = fsspec.core.url_to_fs(checkpoint)
        fs.invalidate_cache(path)
        if fs.exists(path):
            raise RuntimeError("Fresh checkpoint retry prefix already exists")
    return checkpoint


def commit_and_prune(
    config: TrainingConfig, checkpoint: str, digest: str, clock: TokenClock
) -> None:
    """Publish only a committed checkpoint; retain two temporary recovery saves."""
    metadata = json.loads((StoragePath(checkpoint) / "metadata.json").read_text())
    if (
        metadata.get("exp586_clock") != dataclasses.asdict(clock)
        or metadata.get("scientific_config_sha256") != digest
    ):
        raise ValueError("Checkpoint commit readback mismatch")
    expected = f"{config.checkpoint_root}/step-{clock.updates}"
    if checkpoint != expected and not checkpoint.startswith(expected + "-retry-"):
        raise ValueError("Checkpoint is outside the active trial")
    pointer = StoragePath(config.checkpoint_root) / "latest.json"
    payload = json.dumps(
        {"checkpoint": checkpoint, "metadata": metadata}, sort_keys=True
    )
    pointer.write_text(payload)
    if pointer.read_text() != payload:
        raise ValueError("Latest checkpoint pointer readback failed")
    temporary = [
        c
        for c in discover_checkpoint_candidates(config.checkpoint_root)
        if c.metadata.get("is_temporary")
        and c.metadata.get("scientific_config_sha256") == digest
    ]
    for candidate in temporary[:-2]:
        if (
            not candidate.path.startswith(config.checkpoint_root + "/step-")
            or candidate.step >= clock.updates
        ):
            raise ValueError("Unsafe checkpoint retention candidate")
        fs, path = fsspec.core.url_to_fs(candidate.path)
        fs.rm(path, recursive=True)


def check_clock(
    clock: TokenClock, *, min_updates: int = 0, max_updates: int = TOTAL_UPDATES
) -> None:
    if (
        not 0 <= min_updates <= clock.updates <= max_updates
        or clock.examples != clock.updates * BATCH_SIZE
        or clock.input_tokens != clock.updates * TOKENS_PER_UPDATE
        or clock.loss_targets != clock.examples * 8191
    ):
        raise ValueError("Checkpoint data cursor and token clocks disagree")


class ValidationSample:
    """A fixed, evenly spaced sample across all validation Parquet shards."""

    def __init__(self) -> None:
        self.dataset = open_dataset("validation")

    def get_batch(self, indices: list[int]) -> Any:
        return self.dataset.get_batch(
            [i * len(self.dataset) // VALIDATION_EXAMPLES for i in indices]
        )


@eqx.filter_jit
def validation_step(params: Transformer, pending: jax.Array, batch: Any) -> Any:
    mp = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    return mp.cast_to_compute(apply_qb_betas(params, pending)).next_token_loss(
        batch.tokens,
        batch.loss_weight,
        mask=batch.attn_mask,
        reduction="mean",
        logsumexp_weight=None,
        return_router_metrics=True,
    )


def evaluate(
    state: Any, dataset: ValidationSample, tokenizer: Any, mesh: jax.sharding.Mesh
) -> dict[str, float]:
    losses, dropped, valid = [], 0, 0
    for start in range(0, VALIDATION_EXAMPLES, BATCH_SIZE):
        batch = distributed_batch(
            dataset, tokenizer, mesh, start, BATCH_SIZE, augment=False
        )
        loss, metrics = validation_step(state.params, state.pending_qb_betas, batch)
        losses.append(float(loss))
        dropped += int(
            np.asarray(metrics[MOE_DROPPED_ASSIGNMENTS_METRIC]).astype(np.int64).sum()
        )
        valid += int(
            np.asarray(metrics[MOE_VALID_ASSIGNMENTS_METRIC]).astype(np.int64).sum()
        )
    if not all(math.isfinite(x) for x in losses):
        raise ValueError("Nonfinite held-out loss")
    return {
        "eval/loss": float(np.mean(losses)),
        "eval/drop_fraction": dropped / valid,
        "eval/examples": VALIDATION_EXAMPLES,
    }


def _run_training(config: TrainingConfig) -> None:
    mp = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    runtime_batch_size = config.runtime_batch_size
    runtime_tokens_per_update = runtime_batch_size * 8192
    offload_state = config.effective_runtime_state_mode == "fp32_pinned_host"
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=runtime_batch_size,
        num_train_steps=config.execution_stop_update,
        require_accelerator=True,
        mp=mp,
        use_explicit_mesh_axes=True,
        mesh=grug_trainer_mesh_config(config.context_axis_size),
        watch=WatchConfig(interval=0, watch_targets=[]),
        tracker=WandbConfig(
            entity="eric-czech",
            project="marin",
            name=config.run_id,
            group=(
                "exp586-plantcad2-moe-continuation"
                if config.stage == "s02"
                else (
                    "exp586-plantcad2-moe-1t"
                    if config.stage == "t1"
                    else "exp586-plantcad2-moe-sweep"
                )
            ),
            tags=[
                "dna-exp586",
                "H100",
                config.cluster,
                config.condition,
                f"stage={config.stage}",
                *(["profiling"] if config.profiling_updates is not None else []),
            ],
            save_code=False,
            background=False,
            replicate_path=config.checkpoint_root,
        ),
    )
    trainer.initialize()
    assert jax.device_count() == 8 * config.nodes
    assert all("H100" in d.device_kind for d in jax.devices())
    device_kind = jax.devices()[0].device_kind
    flops_per_example, flops_summary = _compute_flops(model_config=config.model)
    flops_per_device, flops_total = theoretical_flops(device_kind, jax.device_count())
    log_offset = wandb_history_offset()
    tracker.log_configuration(config)
    tracker.log_summary(
        {
            "training/phase": "loading",
            "training/moe_runtime_backend": config.model.moe_implementation,
            "training/runtime_batch_size": runtime_batch_size,
            "training/runtime_tokens_per_update": runtime_tokens_per_update,
            "training/state_mode": config.effective_runtime_state_mode,
            "training/prefetch": bool(config.prefetch_depth),
            "training/runtime_state_mode": config.effective_runtime_state_mode,
            "training/prefetch_depth": config.prefetch_depth,
            "training/runtime_transition_id": config.runtime_transition_id,
            "training/runtime_code_sha256": config.runtime_code_sha256,
            "training/runtime_data_parallel": config.runtime_data_parallel,
            "training/runtime_perf4": config.runtime_perf4,
            "training/runtime_perf5": config.runtime_perf5,
            "gpu_type": "H100",
            "nodes": config.nodes,
            **flops_summary,
            "throughput/flops_per_example": flops_per_example,
            "throughput/device_kind": device_kind,
            "throughput/theoretical_flops_per_device": flops_per_device,
            "throughput/theoretical_flops": flops_total,
        }
    )
    tokenizer = (
        scratch_tokenizer()
        if config.condition == "scratch"
        else pretrained_dna_tokenizer()
    )
    tokenizer_sha256 = tokenizer_digest(tokenizer)
    if config.runtime_data_parallel and pretrained_dna_input_token_ids(
        tokenizer
    ) != tuple(config.model.compact_input_lookup_ids):
        raise ValueError("Runtime input-lookup IDs differ from the tokenizer")
    binding = scientific_config(config, tokenizer_sha256)
    digest, checkpoint = primary_json(lambda: bind_and_discover(config, binding))
    tracker.log_hyperparameters(
        {
            **binding,
            "scientific_config_sha256": digest,
            "reference_real_tokens_per_update": TOKENS_PER_UPDATE,
        }
    )
    reader_started = time.monotonic()
    if config.runtime_perf4:
        tracker.log_summary({"training/phase": "prewarming_inputs"})
    dataset = open_dataset(
        "train",
        config.seed,
        **(PERF4_READER_OPTIONS if config.runtime_perf4 else {}),
        **(
            {
                "preload_progress": lambda completed, total: tracker.log_summary(
                    {
                        "training/preloaded_input_items": completed,
                        "training/preload_total_items": total,
                    }
                )
            }
            if config.runtime_perf4
            else {}
        ),
    )
    reader_resources = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(
                [
                    time.monotonic() - reader_started,
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                ],
                dtype=np.float32,
            ),
            tiled=True,
        )
    ).reshape(-1, 2)
    if config.runtime_perf4 and float(reader_resources[:, 1].max()) > 50:
        raise ValueError("Perf4 eager input cache exceeds its validated RSS bound")
    tracker.log_summary(
        {
            "training/reader_setup_max_seconds": float(reader_resources[:, 0].max()),
            "training/reader_max_host_rss_gib": float(reader_resources[:, 1].max()),
        }
    )
    validation = ValidationSample()
    mesh = training_mesh(config)
    opt_config = optimizer_config(
        reference_tokens_per_update=TOKENS_PER_UPDATE,
        training_tokens=config.optimizer_reference_tokens,
        multiplier=config.lr_multiplier,
    )
    optimizer = opt_config.build(
        config.target_updates,
        ns4d_impl=(
            sharded_expert_ns(mesh)
            if config.runtime_perf5
            else node_local_expert_ns(mesh, slice_before_cast=True)
            if config.runtime_perf4
            else None
        ),
    )
    if config.runtime_perf5:
        optimizer = match_gradient_placement(optimizer)
    schedule = config.schedule
    with jax.set_mesh(mesh):
        is_fresh_pretrained = (
            config.stage in ("s01", "t1")
            and config.condition == "pretrained"
            and checkpoint is None
        )
        weights = (
            restore_weights(
                SOURCE_CHECKPOINT, SOURCE_METADATA_DIGEST, config.model, mesh
            )
            if is_fresh_pretrained
            else eqx.filter_jit(Transformer.init)(
                config.model, key=jax.random.key(config.seed)
            )
        )
        state = eqx.filter_jit(
            lambda w: fresh_state(
                w,
                optimizer,
                mp,
                offload=offload_state,
                retain_master=True,
            )
        )(weights)
        del weights
        clock = TokenClock()
        resume_checkpoint = (
            checkpoint
            or config.validation_source_checkpoint
            or config.continuation_source_checkpoint
        )
        runtime_resume = (
            validate_runtime_resume(config, resume_checkpoint)
            if resume_checkpoint is not None
            else {}
        )
        validation_source_binding = None
        if checkpoint is None and config.validation_source_checkpoint is not None:
            validation_source_binding = primary_json(
                lambda: checkpoint_metadata_binding(config.validation_source_checkpoint)
            )
            if (
                validation_source_binding["metadata_sha256"]
                != config.validation_source_metadata_sha256
                or validation_source_binding["clock"].get("updates")
                != config.validation_source_update
            ):
                raise ValueError("Validation source checkpoint binding changed")
        restore_summary: dict[str, float | int] | None = None
        if resume_checkpoint is not None:
            state, clock = restore_dna_checkpoint(
                state,
                resume_checkpoint,
                mesh,
                data_seed=config.seed,
                tokenizer_sha256=tokenizer_sha256,
                scientific_config_sha256=(
                    digest
                    if checkpoint is not None
                    else validation_source_binding["scientific_config_sha256"]
                    if validation_source_binding is not None
                    else CONTINUATION_SOURCE_SCIENTIFIC_CONFIG_SHA256
                ),
                metadata_sha256=(
                    config.validation_source_metadata_sha256
                    if validation_source_binding is not None
                    else CONTINUATION_SOURCE_METADATA_SHA256
                    if checkpoint is None and config.stage == "s02"
                    else None
                ),
            )
            restore_summary = validate_restored_training_state(state, clock)
            if not bool(state_all_finite(state)):
                raise ValueError("Restored training state is nonfinite")
            gc.collect()
        replicated_state_bytes = (
            assert_replicated_state(state) if config.runtime_data_parallel else None
        )
        physical_restore = (
            assert_physical_replicas(state, mesh) if config.runtime_perf4 else None
        )
        optimizer_partitions = None
        partitioned_state_bytes = None
        if config.runtime_perf5:
            state, optimizer_partitions = partition_expert_optimizer_state(state, mesh)
            jax.block_until_ready(state)
            assert_partitioned_optimizer_state(state, mesh)
            if not bool(model_matches_fp32_master(state.params, state.master_params)):
                raise ValueError("Restored BF16 parameters do not match FP32 masters")
            partitioned_state_bytes = persistent_state_bytes(state)
        check_clock(
            clock,
            min_updates=config.start_update,
            max_updates=config.execution_stop_update,
        )
        initial_update = clock.updates
        next_factor = schedule.factor(clock.input_tokens)
        tracker.log_summary(
            {
                "training/resumed_checkpoint": resume_checkpoint,
                "training/resumed_input_tokens": clock.input_tokens,
                "training/restored_state": restore_summary,
                "training/runtime_resume_compatibility": runtime_resume,
                "training/restored_physical_replicas": physical_restore,
                **(
                    {
                        f"training/restored_{key}": value
                        for key, value in restore_summary.items()
                    }
                    if restore_summary is not None
                    else {}
                ),
                "training/schedule_revision": schedule.revision,
                "training/start_input_tokens": config.start_input_tokens,
                "training/target_input_tokens": config.target_input_tokens,
                "training/execution_stop_update": config.execution_stop_update,
                "training/validation_execution_updates": (
                    config.validation_execution_updates
                ),
                "training/peak_update": schedule.peak_update,
                "training/next_muon_lr": opt_config.learning_rate * next_factor,
                "training/next_adam_lr": opt_config.adam_lr * next_factor,
                "training/phase": "compiling",
                "training/profiling_run": config.profiling_updates is not None,
                "training/replicated_state_bytes": replicated_state_bytes,
                "training/optimizer_partitions": optimizer_partitions,
                "training/partitioned_state_bytes": partitioned_state_bytes,
            }
        )
        if initial_update >= schedule.peak_update:
            tracker.log_summary(
                {
                    "training/peak_checkpoint": (
                        f"{config.checkpoint_root}/step-{schedule.peak_update}"
                    )
                }
            )
        if initial_update == schedule.peak_update:
            primary_json(lambda: commit_and_prune(config, checkpoint, digest, clock))
            tracker.log_summary({"training/peak_checkpoint": checkpoint})
            tracker.log(
                evaluate(state, validation, tokenizer, mesh), step=log_offset + 1
            )
            log_offset += 1
        train_step = _make_train_step(
            optimizer,
            mp,
            z_loss_weight=1e-4,
            ema_beta=None,
            offload_opt_state=offload_state,
            master_param_mode=(
                MasterParamMode.FP32_PINNED_HOST
                if offload_state
                else MasterParamMode.FP32_DEVICE
            ),
            watch_config=None,
            compiler_options=(PERF4_COMPILER_OPTIONS if config.runtime_perf4 else None),
            replicate_compute_params=config.runtime_perf5,
        )
        profiler_callback = None
        profile_uri = None
        if config.profiling_updates is not None:
            profile_uri = f"s3://marin-us-east-02a/tmp/ttl=7d/xprof/{config.run_id}"
            profiler_callback = profile(
                f"/tmp/{config.run_id}/profiler",
                start_step=config.profiler_start_step,
                num_steps=config.profiler_num_steps,
                create_perfetto_link=False,
                create_perfetto_trace=False,
                profiler_options=ProfileOptionsConfig(
                    host_tracer_level=2,
                    python_tracer_level=0,
                    device_tracer_level=1,
                ).build_jax_profile_options(),
                process_index=0,
                upload_uri=profile_uri,
                xprof_service_url=DEFAULT_XPROF_SERVICE_URL,
            )
            tracker.log_summary(
                {
                    "profiling/start_step": config.profiler_start_step,
                    "profiling/num_steps": config.profiler_num_steps,
                    "profiling/profile_uri": profile_uri,
                    "profiling/xprof_url": xprof_viewer_url(
                        DEFAULT_XPROF_SERVICE_URL, profile_uri
                    ),
                }
            )
        last_save = time.monotonic()
        batch_iterator = None
        if config.prefetch_depth:

            def prefetch_batches():
                remaining = config.execution_stop_update - clock.updates
                base_examples = clock.examples
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        distributed_batch,
                        dataset,
                        tokenizer,
                        mesh,
                        base_examples,
                        runtime_batch_size,
                        augment=True,
                    )
                    for index in range(remaining):
                        batch = future.result()
                        if index + 1 < remaining:
                            future = pool.submit(
                                distributed_batch,
                                dataset,
                                tokenizer,
                                mesh,
                                base_examples + (index + 1) * runtime_batch_size,
                                runtime_batch_size,
                                augment=True,
                            )
                        yield batch

            batch_iterator = iter(prefetch_batches())
        while clock.updates < config.execution_stop_update:
            start = time.monotonic()
            if batch_iterator is None:
                batch = distributed_batch(
                    dataset,
                    tokenizer,
                    mesh,
                    clock.examples,
                    runtime_batch_size,
                    augment=True,
                )
            else:
                batch = next(batch_iterator)
            loading_time = time.monotonic() - start
            schedule_input_tokens = (
                clock.updates * TOKENS_PER_UPDATE
                if config.profiling_updates is not None
                else clock.input_tokens
            )
            factor = schedule.factor(schedule_input_tokens)
            schedule_clock = (
                dataclasses.replace(clock, input_tokens=schedule_input_tokens)
                if config.profiling_updates is not None
                else clock
            )
            state = dataclasses.replace(
                state,
                opt_state=set_token_learning_rates(
                    state.opt_state, opt_config, schedule, schedule_clock
                ),
            )
            if config.profiling_updates is not None:
                with jax.profiler.StepTraceAnnotation("train", step_num=clock.updates):
                    state, metrics, watch = train_step(state, batch)
                    jax.block_until_ready((state, metrics, watch))
            else:
                state, metrics, watch = train_step(state, batch)
                jax.block_until_ready((state, metrics, watch))
            if config.runtime_data_parallel and clock.updates == initial_update:
                if config.runtime_perf5:
                    assert_partitioned_optimizer_state(state, mesh)
                    if not bool(
                        model_matches_fp32_master(state.params, state.master_params)
                    ):
                        raise ValueError(
                            "Updated BF16 parameters do not match FP32 masters"
                        )
                else:
                    assert_replicated_state(state)
            clock = clock.advance(
                input_tokens=runtime_tokens_per_update,
                loss_targets=runtime_batch_size * 8191,
                examples=runtime_batch_size,
            )
            elapsed = time.monotonic() - start
            values = routing_diagnostics(
                metrics,
                batch_size=runtime_batch_size,
                model_config=config.model,
                prefetch_to_host=config.runtime_perf5,
                pack_floating_scalars=config.runtime_perf5,
            )
            values.update({key: float(value) for key, value in (watch or {}).items()})
            if any(not math.isfinite(float(value)) for value in values.values()):
                tracker.log_summary(
                    {
                        "training/phase": "failed",
                        "training/error": "Nonfinite training metrics",
                    }
                )
                raise ValueError("Nonfinite loss, routing, gradient or update metrics")
            values.update(
                {
                    "run_progress": min(
                        (clock.updates - config.start_update)
                        / (config.execution_stop_update - config.start_update),
                        0.999999,
                    ),
                    "train/total_progress": clock.input_tokens
                    / config.target_input_tokens,
                    "train/step": clock.updates,
                    "train/input_tokens": clock.input_tokens,
                    "train/schedule_input_tokens": schedule_input_tokens,
                    "profiling/actual_input_tokens": (
                        clock.updates * runtime_tokens_per_update
                        if config.profiling_updates is not None
                        else clock.input_tokens
                    ),
                    "train/loss_targets": clock.loss_targets,
                    "train/examples": clock.examples,
                    "train/epochs": clock.input_tokens / EPOCH_TOKENS,
                    "train/schedule_revision": schedule.revision,
                    "train/muon_lr": opt_config.learning_rate * factor,
                    "train/adam_lr": opt_config.adam_lr * factor,
                    "throughput/step_seconds": elapsed,
                    "throughput/loading_seconds": loading_time,
                    "throughput/tokens_per_second": runtime_tokens_per_update / elapsed,
                }
            )
            values.update(
                throughput_flop_metrics(
                    flops_per_example=flops_per_example,
                    completed_examples=clock.examples,
                    batch_size=runtime_batch_size,
                    elapsed_seconds=elapsed,
                    theoretical_flops_total=flops_total,
                )
            )
            for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit"):
                memory = jax.local_devices()[0].memory_stats() or {}
                if key in memory:
                    values[f"memory/{key}"] = memory[key]
            history_step = log_offset + clock.updates - initial_update
            tracker.log(values, step=history_step)
            tracker.log_summary({"training/phase": "training"})
            if profiler_callback is not None:
                profiler_callback(
                    type("ProfileStep", (), {"step": clock.updates - 1})()
                )
            permanent = is_permanent_checkpoint(
                clock.updates,
                peak_update=schedule.peak_update,
                permanent_updates=config.permanent_checkpoint_updates,
            )
            due = 0
            if config.profiling_updates is None:
                due = int(
                    multihost_utils.broadcast_one_to_all(
                        np.asarray(
                            checkpoint_due(
                                clock.updates,
                                initial_update,
                                time.monotonic() - last_save,
                                config.checkpoint_interval_seconds,
                                peak_update=schedule.peak_update,
                                recovery_delay_updates=config.recovery_checkpoint_delay_updates,
                                permanent_updates=config.permanent_checkpoint_updates,
                            ),
                            dtype=np.int32,
                        )
                    )
                )
            if due:
                save_start = time.monotonic()
                if not bool(state_all_finite(state)):
                    raise ValueError("Nonfinite checkpoint state")
                checkpoint_state = state
                if config.runtime_perf5:
                    assert_partitioned_optimizer_state(state, mesh)
                    checkpoint_state = replicate_optimizer_state_for_validation(state)
                    jax.block_until_ready(checkpoint_state)
                    if not bool(
                        model_matches_fp32_master(
                            checkpoint_state.params, checkpoint_state.master_params
                        )
                    ):
                        raise ValueError(
                            "Canonical checkpoint BF16 parameters do not match masters"
                        )
                    assert_replicated_state(checkpoint_state)
                elif config.runtime_data_parallel:
                    assert_replicated_state(state)
                physical_checkpoint = None
                if config.runtime_perf4:
                    physical_started = time.monotonic()
                    physical_checkpoint = assert_physical_replicas(
                        checkpoint_state, mesh
                    )
                    physical_checkpoint["seconds"] = time.monotonic() - physical_started
                    physical_checkpoint["step"] = clock.updates
                    tracker.log_summary(
                        {
                            "training/latest_physical_replica_check": (
                                physical_checkpoint
                            )
                        }
                    )
                tracker.log_summary({"training/phase": "checkpointing"})
                checkpoint = primary_json(
                    lambda clock=clock: prepare_checkpoint(config, clock)
                )
                save_dna_checkpoint(
                    checkpoint_state,
                    clock,
                    checkpoint,
                    data_seed=config.seed,
                    tokenizer_sha256=tokenizer_sha256,
                    scientific_config_sha256=digest,
                    runtime_moe_implementation=config.model.moe_implementation,
                    runtime_metadata=config.runtime_metadata,
                    is_temporary=not permanent,
                )
                if config.runtime_perf5:
                    del checkpoint_state
                    gc.collect()
                primary_json(
                    lambda checkpoint=checkpoint, clock=clock: commit_and_prune(
                        config, checkpoint, digest, clock
                    )
                )
                last_save = time.monotonic()
                tracker.log_summary(
                    {
                        "training/latest_checkpoint": checkpoint,
                        "training/checkpoint_input_tokens": clock.input_tokens,
                        "training/checkpoint_step": clock.updates,
                        "training/checkpoint_seconds": last_save - save_start,
                    }
                )
                if permanent:
                    tracker.log_summary(
                        {"training/latest_permanent_checkpoint": checkpoint}
                    )
                if clock.updates == schedule.peak_update:
                    tracker.log_summary({"training/peak_checkpoint": checkpoint})
            if config.profiling_updates is None and (
                clock.updates == 25 or clock.updates % 2000 == 0 or permanent
            ):
                tracker.log_summary({"training/phase": "evaluating"})
                tracker.log(
                    evaluate(state, validation, tokenizer, mesh), step=history_step
                )
        if (
            config.profiling_updates is None
            and initial_update == config.execution_stop_update
        ):
            primary_json(lambda: commit_and_prune(config, checkpoint, digest, clock))
            tracker.log(
                evaluate(state, validation, tokenizer, mesh), step=log_offset + 1
            )
        tracker.log(
            {"run_progress": 1.0}, step=log_offset + clock.updates - initial_update + 2
        )
        tracker.log_summary(
            {
                "training/phase": "complete",
                "training/final_checkpoint": (
                    checkpoint if config.profiling_updates is None else None
                ),
                "training/validation_run": config.validation_updates is not None,
                "training/profiling_run": config.profiling_updates is not None,
            }
        )
    tracker.current_tracker().finish()


def training_mesh(config: TrainingConfig) -> jax.sharding.Mesh:
    """Select the validated EP8 or fully replicated DP production topology."""
    if config.runtime_data_parallel:
        mesh = replicated_mesh()
        expected = {
            "replica_dcn": 8 * config.nodes,
            "data": 1,
            "context": 1,
            "expert": 1,
            "model": 1,
        }
        if dict(mesh.shape) != expected:
            raise ValueError(f"Unexpected full-replica DP mesh: {dict(mesh.shape)}")
        return mesh
    mesh = compact_grug_mesh(
        expert_axis_size=8,
        replica_axis_size=config.nodes // config.context_axis_size,
        context_axis_size=config.context_axis_size,
    )
    for devices in mesh.devices.reshape(config.nodes, 8):
        if len({d.process_index // 8 for d in devices}) != 1:
            raise ValueError("EP group crosses Iris node boundaries")
    return mesh


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
                logging.getLogger(__name__).exception(
                    "Could not publish the training failure to W&B"
                )
        raise


def dispatch(config: TrainingConfig) -> None:
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=False,
        moe_implementation=config.model.moe_implementation,
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
            "EXP586_CHECKPOINT_DEBUG": os.environ.get("EXP586_CHECKPOINT_DEBUG", "0"),
        },
        resources,
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
                2
                if config.validation_updates is not None
                else (45 if config.stage == "t1" else 14) * 24
            ),
            max_retries_failure=0,
            max_task_failures=0,
        )
    )
    job.wait(raise_on_failure=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["scratch", "pretrained"], required=True)
    parser.add_argument(
        "--lr-multiplier",
        type=float,
        choices=[0.5, 1.0, 2.0, 3.0, 10.0],
        required=True,
    )
    parser.add_argument(
        "--cluster", choices=["cw-rno2a", "cw-us-east-02a"], required=True
    )
    parser.add_argument("--nodes", type=int, choices=[1, 2, 4, 8], required=True)
    parser.add_argument("--stage", choices=["s01", "s02", "t1"], default="s01")
    parser.add_argument("--checkpoint-interval-seconds", type=int, default=900)
    parser.add_argument("--recovery-checkpoint-delay-updates", type=int, default=25)
    parser.add_argument(
        "--moe-backend", choices=["pooled", "fixed", "ring"], default=None
    )
    parser.add_argument("--validation-updates", type=int, default=None)
    parser.add_argument("--validation-id", default=None)
    parser.add_argument("--validation-execution-updates", type=int, default=None)
    parser.add_argument("--validation-source-checkpoint", default=None)
    parser.add_argument("--validation-source-metadata-sha256", default=None)
    parser.add_argument("--validation-source-update", type=int, default=None)
    parser.add_argument(
        "--runtime-state-mode",
        choices=["fp32_pinned_host", "fp32_device"],
        default="fp32_pinned_host",
    )
    parser.add_argument(
        "--runtime-prefetch", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--runtime-transition-id", default=None)
    parser.add_argument("--runtime-code-sha256", default=None)
    parser.add_argument(
        "--runtime-data-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--runtime-perf4",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--runtime-perf5",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--profiling-updates", type=int, default=None)
    parser.add_argument("--profiling-id", default=None)
    parser.add_argument("--profiling-batch-size", type=int, default=8)
    parser.add_argument(
        "--profiling-state-mode", choices=["production", "device"], default="production"
    )
    parser.add_argument(
        "--profiling-prefetch", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--profiler-start-step", type=int, default=20)
    parser.add_argument("--profiler-num-steps", type=int, default=5)
    dispatch(TrainingConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
