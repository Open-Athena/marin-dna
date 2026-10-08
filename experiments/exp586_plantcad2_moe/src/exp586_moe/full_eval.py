"""Sharded final exp586 evaluation on the original full PlantCAD2 splits."""

import argparse
import dataclasses
import hashlib
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jmp
from jax.experimental import multihost_utils
from levanter import tracker
from levanter.callbacks.watch import WatchConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from exp586_moe.config import d768_config
from exp586_moe.data import (
    pretrained_dna_tokenizer,
    scratch_tokenizer,
    tokenizer_digest,
)
from exp586_moe.eval_runtime import PilotScorer
from exp586_moe.evaluation_binding import (
    CheckpointRole,
    validate_checkpoint_binding,
)
from exp586_moe.gpu_smoke import SmokeConfig, dispatch
from exp586_moe.real_eval import (
    commit_chunk,
    fetch_file,
    implementation_digest,
    load_chunk,
    oracle_check,
    read_bytes,
    write_bytes,
)
from exp586_moe.tracking import wandb_history_offset
from experiments.grug.moe_hero_ep.weights import restore_weights

PROFILES = (("context-8192", 8192), ("context-2048", 2048))


@dataclasses.dataclass(frozen=True)
class FullEvalConfig(SmokeConfig):
    checkpoint: str = ""
    checkpoint_metadata_digest: str = ""
    requests_uri: str = ""
    requests_sha256: str = ""
    output_prefix: str = ""
    worker_index: int = -1
    stop_after_chunks: int = 0
    checkpoint_role: CheckpointRole = "dna-trained"

    @property
    def artifact_root(self) -> str:
        return self.output_prefix.rstrip("/")

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.nodes != 1
            or len(self.requests_sha256) != 64
            or len(self.checkpoint_metadata_digest) != 64
            or self.worker_index < 0
            or not self.output_prefix.startswith("s3://")
        ):
            raise ValueError(
                "Full evaluation requires one node and pinned durable inputs"
            )


def centered(sequence: str, length: int) -> str:
    if len(sequence) < length or (len(sequence) - length) % 2:
        raise ValueError("Evaluation sequence cannot be center-cropped")
    start = (len(sequence) - length) // 2
    return sequence[start : start + length]


def task_jobs(manifest: dict, worker: int) -> list[tuple[str, int, str, dict]]:
    grouped: dict[tuple[str, str], list[tuple[str, int, str, dict]]] = {}
    for profile, length in PROFILES:
        grouped[(profile, "tasks")] = []
        grouped[(profile, "af")] = []
        for chunk in manifest["plan"]:
            if chunk["worker"] != worker:
                continue
            # The historical SV score depends on the full 8,192-position
            # indel boundary. Context sensitivity is defined only for the 19
            # centered single-sequence tasks.
            if length < 8192 and chunk["task"] == "sv_impact":
                continue
            grouped[(profile, "tasks")].append((profile, length, "tasks", chunk))
        for chunk in manifest["af_plan"]:
            if chunk["worker"] == worker:
                grouped[(profile, "af")].append((profile, length, "af", chunk))
    ordered = [
        grouped[(profile, kind)] for profile, _ in PROFILES for kind in ("tasks", "af")
    ]
    anchors = [jobs[0] for jobs in ordered if jobs]
    remainder = [job for jobs in ordered for job in jobs[1:]]
    return anchors + remainder


def run_full(config: FullEvalConfig) -> None:
    trainer = TrainerConfig(
        id=config.run_id,
        seed=0,
        train_batch_size=8,
        num_train_steps=1,
        require_accelerator=True,
        use_explicit_mesh_axes=True,
        watch=WatchConfig(interval=0, watch_targets=[]),
        tracker=WandbConfig(
            entity="eric-czech",
            project="marin",
            name=config.run_id,
            group="exp586-plantcad2-moe-final-evaluation",
            tags=[
                "dna-exp586",
                "evaluation",
                "full-splits",
                "H100",
                config.condition,
                config.checkpoint_role,
                config.cluster,
            ],
            background=False,
            save_code=False,
            replicate_path=(
                f"{config.artifact_root}/wandb/worker-{config.worker_index:02d}"
            ),
        ),
    )
    trainer.initialize()
    offset = wandb_history_offset()
    tracker.log_configuration(config)
    tracker.log_summary(
        {"evaluation/phase": "loading", "evaluation/error": None, "training_updates": 0}
    )
    payload = read_bytes(config.requests_uri)
    if hashlib.sha256(payload).hexdigest() != config.requests_sha256:
        raise ValueError("Request digest differs")
    manifest = json.loads(payload)
    if manifest.get("schema") != 5 or manifest.get("real_data_full") is not True:
        raise ValueError("Expected full original PlantCAD2 inputs")
    if not 0 <= config.worker_index < manifest["workers"]:
        raise ValueError("Worker index is outside the immutable execution plan")

    directory = Path(tempfile.mkdtemp(prefix="exp586-full-eval-"))
    wheel = fetch_file(manifest["evaluation_wheel"], directory)
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(directory / "code")
    sys.path.insert(0, str(directory / "code"))
    from exp586_evals.native import NativeAdapter
    from exp586_evals.protocol import original

    protocol = original()
    manifest["samples"] = str(directory)
    tokenizer = (
        scratch_tokenizer()
        if config.condition == "scratch"
        else pretrained_dna_tokenizer()
    )
    token_digest = tokenizer_digest(tokenizer)
    checkpoint_binding = validate_checkpoint_binding(
        checkpoint_role=config.checkpoint_role,
        condition=config.condition,
        checkpoint=config.checkpoint,
        checkpoint_metadata_digest=config.checkpoint_metadata_digest,
        metadata_blob=read_bytes(config.checkpoint + "/metadata.json"),
        tokenizer_sha256=token_digest,
    )

    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    model_config = dataclasses.replace(
        d768_config(config.condition), moe_implementation="sonic", expert_chunks=1
    )
    common_binding = {
        "checkpoint_digest": config.checkpoint_metadata_digest,
        "requests_sha256": config.requests_sha256,
        "tokenizer_sha256": token_digest,
        "schema": 2,
        "implementation_sha256": implementation_digest(),
        "condition": config.condition,
        **checkpoint_binding,
        "model_config": json.loads(json.dumps(dataclasses.asdict(model_config))),
        "precision_policy": "params=bfloat16,compute=bfloat16,output=bfloat16",
    }
    tracker.log_hyperparameters(
        {"binding": common_binding, "worker_index": config.worker_index}
    )
    if jax.device_count() != 8 or any(
        "H100" not in device.device_kind for device in jax.devices()
    ):
        raise ValueError("Final evaluation workers require one eight-H100 node")

    policy = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    jobs = task_jobs(manifest, config.worker_index)
    if not jobs:
        raise ValueError("Worker has no assigned evaluation chunks")
    specs = {spec["key"]: spec for spec in protocol.tasks.LEADERBOARD_TASKS}
    specs["maize-af"] = {"file": "maize-af.parquet"}
    args = argparse.Namespace(
        batch_size=8, sv_batch_size=8, softmax_dtype="fp32", use_cache=False
    )
    completed: list[dict[str, Any]] = []
    reused = 0
    diagnostics: dict[str, Any] = {}
    began_run = time.time()

    with jax.set_mesh(mesh):
        model = eqx.filter_jit(policy.cast_to_param)(
            restore_weights(
                config.checkpoint,
                config.checkpoint_metadata_digest,
                model_config,
                mesh,
            )
        )
        base_ids = [
            tokenizer.encode(base, add_special_tokens=False).ids[0] for base in "ACGT"
        ]
        scorer = PilotScorer(model, mesh, base_ids)
        adapters = {
            profile: NativeAdapter(scorer, tokenizer, sequence_length=length)
            for profile, length in PROFILES
        }
        for index, (profile, length, kind, chunk) in enumerate(jobs):
            spec = specs[chunk["task"]]
            records = manifest["inputs"][spec["file"]]["files"]
            if kind == "tasks":
                selected = [
                    record
                    for record in records
                    if record["path"] == chunk["source_path"]
                ]
                if len(selected) != 1:
                    raise ValueError("Task chunk source file is not unique")
                record = selected[0]
                fetch_file(record, directory)
                local_manifest = {
                    **manifest,
                    "inputs": {
                        spec["file"]: {"rows": record["rows"], "files": [record]}
                    },
                }
                local_start = chunk["start"] - chunk["source_start"]
                frame = protocol.inputs.read_frame(
                    local_manifest,
                    spec,
                    local_start,
                    local_start + chunk["stop"] - chunk["start"],
                )
            else:
                for record in records:
                    fetch_file(record, directory)
                frame = protocol.inputs.read_frame(
                    manifest, spec, chunk["start"], chunk["stop"]
                )
            root = f"{config.artifact_root}/{profile}/{kind}"
            binding = {**common_binding, "profile": profile, "sequence_length": length}
            diagnostic_key = f"{profile}/{kind}"
            if diagnostic_key not in diagnostics:
                sequence_column = "RefSeq" if spec.get("kind") == "sv" else "sequence"
                sequences = [
                    centered(sequence, length)
                    for sequence in frame[sequence_column].iloc[:2]
                ]
                position = length // 2
                items = [
                    {
                        "sequence": sequence,
                        "positions": [position],
                        "feature_position": position,
                    }
                    for sequence in sequences
                ]
                diagnostics[diagnostic_key] = oracle_check(
                    model, mesh, scorer, tokenizer, items, kind == "af"
                )
                tracker.log_summary(
                    {
                        "evaluation/oracles": diagnostics,
                        "evaluation/phase": "inference",
                    }
                )
            receipt = load_chunk(root, chunk, binding)
            embedding_receipt = None
            if kind == "af":
                embedding_receipt = load_chunk(
                    f"{config.artifact_root}/{profile}/embeddings", chunk, binding
                )
            if receipt is not None and (kind != "af" or embedding_receipt is not None):
                reused += 1
            else:
                started = time.time()
                adapter = adapters[profile]
                if kind == "tasks":
                    function = (
                        protocol.tasks._score_sv
                        if spec["kind"] == "sv"
                        else protocol.tasks._score_standard_task
                    )
                    result, arrays = function(
                        spec, frame, adapter, tokenizer, args, compute_metrics=False
                    )
                else:
                    arrays, embeddings = adapter.allele_frequency(
                        frame, include_strand_average=False
                    )
                    result = {}
                    embedding_receipt = commit_chunk(
                        f"{config.artifact_root}/{profile}/embeddings",
                        {
                            **chunk,
                            "binding": binding,
                            "started_epoch": started,
                            "finished_epoch": time.time(),
                        },
                        embeddings,
                    )
                receipt = commit_chunk(
                    root,
                    {
                        **chunk,
                        "binding": binding,
                        "result": result,
                        "started_epoch": started,
                        "finished_epoch": time.time(),
                    },
                    arrays,
                )
            completed.append(
                {
                    "profile": profile,
                    "kind": kind,
                    "receipt": receipt,
                    "embeddings": embedding_receipt,
                }
            )
            tracker.log(
                {
                    "run_progress": 0.99 * (index + 1) / len(jobs),
                    "evaluation/chunks_done": index + 1,
                    "evaluation/chunks_reused": reused,
                    "evaluation/profile": profile,
                    "evaluation/task": chunk["task"],
                },
                step=offset + index + 1,
            )
            if config.stop_after_chunks and len(completed) >= config.stop_after_chunks:
                tracker.log_summary(
                    {
                        "evaluation/phase": "checkpointed_pause",
                        "evaluation/chunks_reused": reused,
                    }
                )
                tracker.current_tracker().finish()
                return

    report = {
        "schema": 1,
        "binding": common_binding,
        "worker": config.worker_index,
        "chunks": completed,
        "oracles": diagnostics,
        "chunks_reused": reused,
        "started_epoch": began_run,
        "finished_epoch": time.time(),
        "run_wall_seconds": time.time() - began_run,
        "gpus": 8,
        "environment": {
            "backend": "native-jax-sonic",
            "batch_size": 8,
            "jax": jax.__version__,
        },
    }
    report_uri = f"{config.artifact_root}/workers/worker-{config.worker_index:02d}.json"
    blob = json.dumps(report, sort_keys=True).encode()
    write_bytes(report_uri, blob)
    multihost_utils.sync_global_devices(f"full-eval-worker-{config.worker_index}")
    if (
        hashlib.sha256(read_bytes(report_uri)).hexdigest()
        != hashlib.sha256(blob).hexdigest()
    ):
        raise ValueError("Worker manifest readback differs")
    tracker.log_summary(
        {
            "evaluation/phase": "complete",
            "evaluation/artifact": config.artifact_root,
            "evaluation/chunks_reused": reused,
            "evaluation/worker_manifest": report_uri,
        }
    )
    tracker.log({"run_progress": 1.0}, step=offset + len(completed) + 1)
    tracker.current_tracker().finish()


def run(config: FullEvalConfig) -> None:
    try:
        run_full(config)
    except Exception as error:
        try:
            wandb_tracker = tracker.get_tracker("wandb")
        except (RuntimeError, KeyError):
            pass
        else:
            wandb_tracker.log_summary(
                {"evaluation/phase": "failed", "evaluation/error": str(error)}
            )
            wandb_tracker.run.finish(exit_code=1)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "condition",
        "run-id",
        "cluster",
        "checkpoint",
        "checkpoint-metadata-digest",
        "requests-uri",
        "requests-sha256",
        "output-prefix",
    ]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--stop-after-chunks", type=int, default=0)
    parser.add_argument(
        "--checkpoint-role",
        choices=("dna-trained", "language-base"),
        default="dna-trained",
    )
    dispatch(
        FullEvalConfig(**vars(parser.parse_args())),
        worker=run,
        moe_implementation="sonic",
        timeout_hours=18,
    )


if __name__ == "__main__":
    main()
