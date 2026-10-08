"""Resumable native inference for the fixed exp582 real-data evaluation pilot."""

import argparse
import dataclasses
import hashlib
import io
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import equinox as eqx
import fsspec
import jax
import jmp
import numpy as np
from jax.experimental import multihost_utils
from levanter import tracker
from levanter.callbacks.watch import WatchConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from scipy.special import log_softmax, softmax
from tokenizers import Tokenizer

from exp582_moe.config import d1536_config
from exp582_moe.data import (
    pretrained_dna_tokenizer,
    scratch_tokenizer,
    tokenizer_digest,
)
from exp582_moe.eval_runtime import NativeScorer, PilotScorer, dna_output_head
from exp582_moe.evaluation_binding import (
    CheckpointRole,
    validate_checkpoint_binding,
)
from exp582_moe.gpu_smoke import SmokeConfig, dispatch
from exp582_moe.tracking import wandb_history_offset
from experiments.grug.moe_hero_ep.model import Transformer
from experiments.grug.moe_hero_ep.weights import restore_weights

LEGACY_SCRATCH_DIGEST = (
    "c003ecade50776f2932184a7cef40ac42ecba3ece1e1967562673949f38a27bd"
)

LEGACY_SCRATCH_CHECKPOINT = (
    "s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp582_plantcad2_moe/"
    "exp582-plantcad2-d1536-random-routing-cf32-long-smoke-v1/checkpoints/step-600"
)


@dataclasses.dataclass(frozen=True)
class RealEvalConfig(SmokeConfig):
    checkpoint: str = ""
    checkpoint_metadata_digest: str = ""
    requests_uri: str = ""
    requests_sha256: str = ""
    allow_legacy_scratch: bool = False
    checkpoint_role: CheckpointRole = "dna-trained"
    stop_after_chunks: int = 0

    @property
    def artifact_root(self) -> str:
        return self.checkpoint_root.removesuffix("/checkpoints") + "/real-evaluation"

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.nodes != 1
            or len(self.requests_sha256) != 64
            or len(self.checkpoint_metadata_digest) != 64
        ):
            raise ValueError(
                "One-node pilot requires pinned request/checkpoint digests"
            )
        if self.allow_legacy_scratch and (
            self.condition != "random"
            or self.checkpoint_metadata_digest != LEGACY_SCRATCH_DIGEST
            or self.checkpoint != LEGACY_SCRATCH_CHECKPOINT
        ):
            raise ValueError(
                "Legacy binding applies only to the verified scratch canary"
            )
        if self.checkpoint_role == "language-base" and self.allow_legacy_scratch:
            raise ValueError("Language-base evaluation cannot use a legacy binding")


def implementation_digest() -> str:
    """Bind all owned model/adapter code and the resolved environment, failing closed."""
    project = Path(__file__).resolve().parents[2]
    files = sorted((project / "src").rglob("*.py")) + [
        project / "pyproject.toml",
        project / "uv.lock",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(project)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_bytes(uri: str) -> bytes:
    with fsspec.open(uri, "rb") as stream:
        return stream.read()


def write_bytes(uri: str, blob: bytes) -> None:
    if jax.process_index() == 0:
        with fsspec.open(uri, "wb") as stream:
            stream.write(blob)


def load_chunk(
    root: str, chunk: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any] | None:
    """Reuse original worker chunk names/metadata, with checkpoint and source binding."""
    uri = f"{root}/chunks/{chunk['key']}"
    fs, path = fsspec.core.url_to_fs(uri + ".json")
    if not fs.exists(path):
        return None
    receipt = json.loads(read_bytes(uri + ".json"))
    if receipt["binding"] != binding or any(receipt[k] != v for k, v in chunk.items()):
        raise ValueError("Existing chunk belongs to another input/checkpoint/source")
    blob = read_bytes(uri + ".npz")
    if hashlib.sha256(blob).hexdigest() != receipt["array_sha256"]:
        raise ValueError("Committed chunk checksum mismatch")
    with np.load(io.BytesIO(blob), allow_pickle=False) as arrays:
        if any(len(arrays[k]) != chunk["stop"] - chunk["start"] for k in arrays.files):
            raise ValueError("Chunk coverage mismatch")
    return receipt


def commit_chunk(
    root: str, receipt: dict[str, Any], arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    chunk = {key: receipt[key] for key in ("key", "task", "start", "stop", "worker")}
    previous = load_chunk(root, chunk, receipt["binding"])
    if previous is not None:
        with np.load(
            io.BytesIO(read_bytes(f"{root}/chunks/{receipt['key']}.npz")),
            allow_pickle=False,
        ) as saved:
            if set(saved.files) != set(arrays):
                raise ValueError("Recomputed chunk fields differ")
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
        return previous
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    blob = stream.getvalue()
    receipt["array_sha256"] = hashlib.sha256(blob).hexdigest()
    uri = f"{root}/chunks/{receipt['key']}"
    write_bytes(uri + ".npz", blob)
    multihost_utils.sync_global_devices("data-" + receipt["key"])
    if hashlib.sha256(read_bytes(uri + ".npz")).hexdigest() != receipt["array_sha256"]:
        raise ValueError("Chunk readback mismatch")
    write_bytes(uri + ".json", json.dumps(receipt, sort_keys=True).encode())
    multihost_utils.sync_global_devices("commit-" + receipt["key"])
    return receipt


def fetch_file(record: dict[str, Any], directory: Path) -> Path:
    """Read the pinned Parquet or evaluation wheel; original loader handles row ranges."""
    target = directory / record["path"]
    if not target.exists():
        blob = read_bytes(record["uri"])
        if hashlib.sha256(blob).hexdigest() != record["sha256"]:
            raise ValueError("Input file checksum differs")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    return target


def oracle_check(
    model: Transformer,
    mesh: jax.sharding.Mesh,
    scorer: PilotScorer,
    tokenizer: Tokenizer,
    items: list[dict],
    allele_frequency: bool,
) -> dict[str, float]:
    ids = [tokenizer.encode(x["sequence"], add_special_tokens=False).ids for x in items]
    arguments = (
        ids,
        [x["positions"] for x in items],
        [x["feature_position"] for x in items],
    )
    first = scorer.evaluate(*arguments, allele_frequency=allele_frequency)
    repeat = scorer.evaluate(*arguments, allele_frequency=allele_frequency)
    for key in first:
        np.testing.assert_array_equal(first[key], repeat[key])
    alone = scorer.evaluate(
        ids[:1], arguments[1][:1], arguments[2][:1], allele_frequency=allele_frequency
    )
    for key in first:
        np.testing.assert_array_equal(first[key][:1], alone[key])
    logs, hidden = NativeScorer(model, mesh).evaluate(ids)
    head = dna_output_head(model, scorer.base_ids)
    head = np.asarray(
        head
        if head.is_fully_addressable
        else multihost_utils.process_allgather(head, tiled=True)
    ).astype(np.float32)
    errors = []
    for i, item in enumerate(items):
        query_logits = (
            hidden[i][np.maximum(np.asarray(item["positions"]) - 1, 0)] @ head
        )
        expected = softmax(query_logits, axis=-1)
        expected[np.asarray(item["positions"]) == 0] = 0
        errors.append(
            float(np.max(np.abs(expected - first["probabilities"][i, : len(expected)])))
        )
        if allele_frequency:
            dna = log_softmax(hidden[i][:-1] @ head, axis=-1)
            targets = np.asarray([list(scorer.base_ids).index(t) for t in ids[i][1:]])
            np.testing.assert_allclose(
                first["acgt_logs"][i],
                dna[np.arange(len(targets)), targets],
                rtol=0,
                atol=2e-4,
            )
            np.testing.assert_allclose(
                first["full_logs"][i], logs[i], rtol=0, atol=2e-4
            )
            np.testing.assert_allclose(
                first["mean"][i],
                hidden[i].mean(axis=0, dtype=np.float32),
                rtol=0,
                atol=2e-4,
            )
            np.testing.assert_array_equal(
                first["variant"][i], hidden[i][item["feature_position"]]
            )
    if max(errors) > 2e-4:
        raise ValueError(
            "Direct DNA projection disagrees with independent CPU projection"
        )
    return {
        "dna_probability_max_abs": max(errors),
        "repeat_max_abs": 0.0,
        "batch_context_max_abs": 0.0,
    }


def run_pilot(config: RealEvalConfig) -> None:
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
            group="exp582-plantcad2-moe-real-eval-validation",
            tags=[
                "dna-exp582",
                "validation",
                "real-data",
                "H100",
                config.condition,
                config.checkpoint_role,
                config.cluster,
            ],
            background=False,
            save_code=False,
            replicate_path=config.artifact_root,
        ),
    )
    trainer.initialize()
    offset = wandb_history_offset()
    tracker.log_configuration(config)
    tracker.log_summary(
        {"validation/phase": "loading", "validation/error": None, "training_updates": 0}
    )
    payload = read_bytes(config.requests_uri)
    if hashlib.sha256(payload).hexdigest() != config.requests_sha256:
        raise ValueError("Request digest differs")
    manifest = json.loads(payload)
    if manifest.get("schema") != 4 or manifest.get("real_data_pilot") is not True:
        raise ValueError("Expected original PlantCAD2 protocol pilot inputs")
    directory = Path(tempfile.mkdtemp(prefix="exp582-eval-"))
    wheel = fetch_file(manifest["evaluation_wheel"], directory)
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(directory / "code")
    sys.path.insert(0, str(directory / "code"))
    from exp582_evals.native import NativeAdapter
    from exp582_evals.protocol import original

    protocol = original()
    manifest["samples"] = str(directory)
    if (
        sum(x["rows"] for k, x in manifest["inputs"].items() if k != "maize-af.parquet")
        != 2560
    ):
        raise ValueError("Pilot inventory differs")
    tokenizer = (
        scratch_tokenizer()
        if config.condition == "random"
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
        allow_legacy_scratch=config.allow_legacy_scratch,
        legacy_scratch_checkpoint=LEGACY_SCRATCH_CHECKPOINT,
        legacy_scratch_digest=LEGACY_SCRATCH_DIGEST,
    )
    binding = {
        "checkpoint_digest": config.checkpoint_metadata_digest,
        "requests_sha256": config.requests_sha256,
        "tokenizer_sha256": token_digest,
        "schema": 1,
        "implementation_sha256": implementation_digest(),
        "condition": config.condition,
        **checkpoint_binding,
    }
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    model_config = dataclasses.replace(
        d1536_config(config.condition), moe_implementation="sonic", expert_chunks=1
    )
    policy = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    binding["model_config"] = json.loads(json.dumps(dataclasses.asdict(model_config)))
    binding["precision_policy"] = "params=bfloat16,compute=bfloat16,output=bfloat16"
    tracker.log_hyperparameters(
        {"binding": binding, "legacy_scratch_binding": config.allow_legacy_scratch}
    )
    assert jax.device_count() == 8 and all(
        "H100" in x.device_kind for x in jax.devices()
    )
    with jax.set_mesh(mesh):
        model = eqx.filter_jit(policy.cast_to_param)(
            restore_weights(
                config.checkpoint, config.checkpoint_metadata_digest, model_config, mesh
            )
        )
        base_ids = [
            tokenizer.encode(base, add_special_tokens=False).ids[0] for base in "ACGT"
        ]
        scorer = PilotScorer(model, mesh, base_ids)
        adapter = NativeAdapter(scorer, tokenizer)
        specs = {spec["key"]: spec for spec in protocol.tasks.LEADERBOARD_TASKS}
        specs["maize-af"] = {"file": "maize-af.parquet"}
        plan = [("tasks", chunk) for chunk in manifest["plan"]] + [
            ("af", chunk) for chunk in manifest["af_plan"]
        ]
        completed, reused, diagnostics = [], 0, {}
        args = argparse.Namespace(
            batch_size=8, sv_batch_size=8, softmax_dtype="fp32", use_cache=False
        )
        began_run = time.time()
        for index, (kind, chunk) in enumerate(plan):
            spec = specs[chunk["task"]]
            for record in manifest["inputs"][spec["file"]]["files"]:
                fetch_file(record, directory)
            frame = protocol.inputs.read_frame(
                manifest, spec, chunk["start"], chunk["stop"]
            )
            root = config.artifact_root + "/" + kind
            if kind not in diagnostics:
                sequence_column = "RefSeq" if spec.get("kind") == "sv" else "sequence"
                items = [
                    {"sequence": x, "positions": [4096], "feature_position": 4096}
                    for x in frame[sequence_column].iloc[:2]
                ]
                diagnostics[kind] = oracle_check(
                    model, mesh, scorer, tokenizer, items, kind == "af"
                )
                tracker.log_summary(
                    {
                        "validation/oracles": diagnostics,
                        "validation/phase": "inference",
                    }
                )
            receipt = load_chunk(root, chunk, binding)
            embedding_receipt = None
            if kind == "af":
                embedding_receipt = load_chunk(
                    config.artifact_root + "/embeddings", chunk, binding
                )
            if receipt is not None and (kind != "af" or embedding_receipt is not None):
                reused += 1
            else:
                started = time.time()
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
                    arrays, embeddings = adapter.allele_frequency(frame)
                    result = {}
                    embedding_receipt = commit_chunk(
                        config.artifact_root + "/embeddings",
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
                {"kind": kind, "receipt": receipt, "embeddings": embedding_receipt}
            )
            tracker.log(
                {
                    "run_progress": 0.99 * (index + 1) / len(plan),
                    "validation/chunks_done": index + 1,
                    "validation/chunks_reused": reused,
                    "validation/task": chunk["task"],
                },
                step=offset + index + 1,
            )
            if config.stop_after_chunks and len(completed) >= config.stop_after_chunks:
                tracker.log_summary(
                    {
                        "validation/phase": "checkpointed_pause",
                        "validation/chunks_reused": reused,
                    }
                )
                tracker.current_tracker().finish()
                return
        report = {
            "binding": binding,
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
            "flash_verification": {
                "backend": "JAX_FLASH",
                "external_pytorch_fa2_verified": False,
            },
        }
        write_bytes(
            config.artifact_root + "/manifest.json",
            json.dumps(report, sort_keys=True).encode(),
        )
        multihost_utils.sync_global_devices("pilot-complete")
        tracker.log_summary(
            {
                "validation/phase": "gpu_complete",
                "validation/artifact": config.artifact_root,
                "validation/chunks_reused": reused,
            }
        )
        tracker.log({"run_progress": 1.0}, step=offset + len(completed) + 1)
    tracker.current_tracker().finish()


def run(config: RealEvalConfig) -> None:
    try:
        run_pilot(config)
    except Exception as error:
        try:
            wandb_tracker = tracker.get_tracker("wandb")
        except (RuntimeError, KeyError):
            # Startup may fail before a tracker exists; preserve that original error.
            pass
        else:
            wandb_tracker.log_summary(
                {"validation/phase": "failed", "validation/error": str(error)}
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
    ]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--allow-legacy-scratch", action="store_true")
    parser.add_argument(
        "--checkpoint-role",
        choices=("dna-trained", "language-base"),
        default="dna-trained",
    )
    parser.add_argument("--stop-after-chunks", type=int, default=0)
    dispatch(
        RealEvalConfig(**vars(parser.parse_args())),
        worker=run,
        moe_implementation="sonic",
    )


if __name__ == "__main__":
    main()
