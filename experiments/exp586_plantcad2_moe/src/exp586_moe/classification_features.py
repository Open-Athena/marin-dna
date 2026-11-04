"""Resumable classification-probe feature extraction on eight-H100 workers."""

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
import numpy as np
from levanter import tracker
from levanter.callbacks.watch import WatchConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from exp586_moe.config import d768_config
from exp586_moe.data import pretrained_dna_tokenizer, tokenizer_digest
from exp586_moe.eval_runtime import NativeScorer
from exp586_moe.evaluation_binding import validate_checkpoint_binding
from exp586_moe.frozen_features import FrozenFeatures
from exp586_moe.gpu_smoke import SmokeConfig, dispatch
from exp586_moe.real_eval import (
    commit_chunk,
    fetch_file,
    implementation_digest,
    load_chunk,
    read_bytes,
    write_bytes,
)
from exp586_moe.stable_scatter_reference import install_stable_scatter_reference
from exp586_moe.tracking import wandb_history_offset
from experiments.grug.moe_hero_ep.weights import restore_weights


@dataclasses.dataclass(frozen=True)
class FeatureConfig(SmokeConfig):
    requests_uri: str = ""
    requests_sha256: str = ""
    model_label: str = ""
    worker_index: int = 0
    output_prefix: str = ""
    gate: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.condition != "pretrained"
            or self.nodes != 1
            or len(self.requests_sha256) != 64
        ):
            raise ValueError("Probe features require one-node final pretrained models")
        if not self.output_prefix.startswith(
            "s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/evaluations/frozen-classification-v1/"
        ):
            raise ValueError("Unexpected probe output owner")


def source_slice(
    manifest: dict[str, Any], spec: dict[str, Any], chunk: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    """Give the established reader a one-file view with local row coordinates."""
    sources = [
        source
        for source in manifest["inputs"][spec["file"]]["files"]
        if source["row_start"] <= chunk["start"] < chunk["stop"] <= source["row_stop"]
    ]
    if len(sources) != 1:
        raise ValueError("A feature chunk must belong to exactly one pinned source")
    source = sources[0]
    view = {
        **manifest,
        "inputs": {spec["file"]: {"rows": source["rows"], "files": [source]}},
    }
    return (
        source,
        view,
        chunk["start"] - source["row_start"],
        chunk["stop"] - source["row_start"],
    )


def extract_rows(
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    scorer: FrozenFeatures,
    tokenizer: Any,
) -> dict[str, np.ndarray]:
    """The visibility builder is the only path from source rows to model inputs."""
    from exp586_evals.probe_protocol import visible_sequences

    prepared = [visible_sequences(spec, row) for row in rows]
    field = "mean" if spec["kind"] == "sv" else "last"
    result = {}
    for name in prepared[0]:
        batches = []
        for offset in range(0, len(rows), scorer.batch_size):
            sequences = [
                row[name] for row in prepared[offset : offset + scorer.batch_size]
            ]
            ids = [
                tokenizer.encode(sequence, add_special_tokens=False).ids
                for sequence in sequences
            ]
            if any(
                len(tokens) != len(sequence)
                for tokens, sequence in zip(ids, sequences, strict=True)
            ):
                raise ValueError("DNA is no longer encoded one nucleotide per token")
            batches.append(scorer.evaluate(ids)[field])
        result[name] = np.concatenate(batches)
    return result


def validate_features(
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    scorer: FrozenFeatures,
    tokenizer: Any,
    mesh: Any,
) -> dict[str, Any]:
    """Require repeated/batch/counterfactual parity and independent full-state reduction."""
    from exp586_evals.probe_protocol import visible_sequences

    rows = rows[:8]
    first = extract_rows(spec, rows, scorer, tokenizer)
    repeat = extract_rows(spec, rows, scorer, tokenizer)
    alone = extract_rows(spec, rows[:1], scorer, tokenizer)
    permuted = extract_rows(spec, rows[::-1], scorer, tokenizer)
    for key in first:
        np.testing.assert_array_equal(first[key], repeat[key])
        np.testing.assert_array_equal(first[key][:1], alone[key])
        np.testing.assert_array_equal(first[key], permuted[key][::-1])
    counterfactual = []
    for row in rows:
        changed = dict(row)
        changed["label"] = 1 - int(row["label"])
        if spec["kind"] != "sv":
            sequence = list(row["sequence"])
            for position in spec["positions"]:
                sequence[position] = "C" if sequence[position].upper() != "C" else "A"
            changed["sequence"] = "".join(sequence)
        counterfactual.append(changed)
    changed_features = extract_rows(spec, counterfactual, scorer, tokenizer)
    for key in first:
        np.testing.assert_array_equal(first[key], changed_features[key])
    visible_change = dict(rows[0])
    field = "RefSeq" if spec["kind"] == "sv" else "sequence"
    sequence = list(visible_change[field])
    position = len(sequence) - 1 if spec["kind"] == "sv" else min(spec["positions"]) - 1
    sequence[position] = "C" if sequence[position].upper() != "C" else "A"
    visible_change[field] = "".join(sequence)
    changed_visible = extract_rows(spec, [visible_change], scorer, tokenizer)
    if all(np.array_equal(first[key][:1], changed_visible[key]) for key in first):
        raise ValueError("Frozen features ignore a visible nucleotide change")
    # A separately implemented adapter materializes full real-token states,
    # then reduces them on CPU. No vocabulary projection is used in production.
    prepared = visible_sequences(spec, rows[0])
    name = next(iter(prepared))
    sequence = prepared[name]
    ids = [tokenizer.encode(sequence, add_special_tokens=False).ids]
    _, hidden = NativeScorer(
        scorer.model, mesh, pad_to=scorer.sequence_length
    ).evaluate(ids)
    reference = (
        hidden[0].mean(axis=0, dtype=np.float32)
        if spec["kind"] == "sv"
        else hidden[0][-1]
    )
    np.testing.assert_allclose(first[name][0], reference, rtol=1e-5, atol=2e-5)
    return {
        "rows": len(rows),
        "repeat_exact": True,
        "batch_exact": True,
        "batch_permutation_exact": True,
        "target_and_label_counterfactual_exact": True,
        "visible_change_affects_features": True,
        "independent_state_reduction_max_abs": float(
            np.max(np.abs(first[name][0] - reference))
        ),
    }


def run(config: FeatureConfig) -> None:
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
            group="exp586-plantcad2-moe-frozen-classification",
            tags=[
                "dna-exp586",
                "evaluation",
                "frozen-probe",
                "H100",
                config.cluster,
                config.model_label,
            ],
            background=False,
            save_code=False,
            replicate_path=f"{config.output_prefix}/wandb/{config.run_id}",
        ),
    )
    trainer.initialize()
    offset = wandb_history_offset()
    tracker.log_configuration(config)
    tracker.log_summary({"evaluation/phase": "loading", "training_updates": 0})
    payload = read_bytes(config.requests_uri)
    if hashlib.sha256(payload).hexdigest() != config.requests_sha256:
        raise ValueError("Probe request manifest identity differs")
    manifest = json.loads(payload)
    if manifest["schema"] != "exp586-classification-features-v1":
        raise ValueError("Unexpected frozen feature protocol")
    directory = Path(tempfile.mkdtemp(prefix="exp586-classification-"))
    wheel = fetch_file(manifest["evaluation_wheel"], directory)
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(directory / "code")
    sys.path.insert(0, str(directory / "code"))
    from exp586_evals.protocol import original

    protocol = original()
    manifest["samples"] = str(directory)
    record = manifest["models"][config.model_label]
    tokenizer = pretrained_dna_tokenizer()
    binding = validate_checkpoint_binding(
        checkpoint_role="dna-trained",
        condition="pretrained",
        checkpoint=record["checkpoint"],
        checkpoint_metadata_digest=record["canonical_metadata_sha256"],
        metadata_blob=read_bytes(record["checkpoint"] + "/metadata.json"),
        tokenizer_sha256=tokenizer_digest(tokenizer),
    )
    binding.update(
        {
            "requests_sha256": config.requests_sha256,
            "implementation_sha256": implementation_digest(),
            "model_label": config.model_label,
        }
    )
    tracker.log_hyperparameters({"binding": binding})
    if jax.device_count() != 8 or any(
        "H100" not in device.device_kind for device in jax.devices()
    ):
        raise ValueError("Expected exactly eight H100 GPUs")
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    model_config = dataclasses.replace(
        d768_config("pretrained"), moe_implementation="sonic", expert_chunks=1
    )
    specs = {spec["key"]: spec for spec in manifest["tasks"]}
    jobs = [
        piece for piece in manifest["plan"] if piece["worker"] == config.worker_index
    ]
    if config.gate:
        jobs = [
            next(piece for piece in manifest["plan"] if piece["task"] == name)
            for name in (
                "conservation_andropogoneae",
                "core_maize_tis",
                "core_tomato_acceptor",
                "sv_impact",
            )
        ]
        # Exercise the final file of a multi-file input, not only row zero.
        jobs.append(
            max(
                (
                    piece
                    for piece in manifest["plan"]
                    if piece["task"] == "conservation_poaceae_non_tis"
                ),
                key=lambda piece: piece["start"],
            )
        )
        if manifest.get("gate_sv_start") is not None:
            start = int(manifest["gate_sv_start"])
            jobs.append(
                {
                    "task": "sv_impact",
                    "start": start,
                    "stop": start + 8,
                    "worker": 0,
                    "key": f"sv_impact--{start:06d}-{start + 8:06d}",
                }
            )
    completed = []
    reference_cases = {}
    started = time.time()
    policy = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    with jax.set_mesh(mesh):
        model = eqx.filter_jit(policy.cast_to_param)(
            restore_weights(
                record["checkpoint"],
                record["canonical_metadata_sha256"],
                model_config,
                mesh,
            )
        )
        scorers = {
            length: FrozenFeatures(model, mesh, sequence_length=length)
            for length in (4096, 8192)
        }
        for index, chunk in enumerate(jobs):
            spec = specs[chunk["task"]]
            source, view, local_start, local_stop = source_slice(manifest, spec, chunk)
            fetch_file(source, directory)
            receipt = (
                load_chunk(config.output_prefix, chunk, binding)
                if not config.gate
                else None
            )
            if receipt is None:
                frame = protocol.inputs.read_frame(
                    view,
                    spec,
                    local_start,
                    min(local_stop, local_start + 8) if config.gate else local_stop,
                )
                rows = frame.to_dict("records")
                scorer = scorers[8192 if spec["kind"] == "sv" else 4096]
                if config.gate:
                    report = validate_features(spec, rows, scorer, tokenizer, mesh)
                    reference_cases[chunk["task"]] = (spec, rows)
                    tracker.log_summary({"validation/" + chunk["task"]: report})
                    if jax.process_index() == 0:
                        write_bytes(
                            config.output_prefix + "/" + chunk["task"] + ".json",
                            json.dumps(report, sort_keys=True).encode(),
                        )
                    receipt = report
                else:
                    arrays = extract_rows(spec, rows, scorer, tokenizer)
                    arrays["row_id"] = np.arange(
                        chunk["start"], chunk["stop"], dtype=np.int64
                    )
                    receipt = commit_chunk(
                        config.output_prefix, {**chunk, "binding": binding}, arrays
                    )
            completed.append(receipt)
            tracker.log_summary(
                {
                    "evaluation/phase": "validating" if config.gate else "extracting",
                    "evaluation/chunks_done": len(completed),
                    "evaluation/chunks_total": len(jobs),
                    "evaluation/task": chunk["task"],
                }
            )
            tracker.log(
                {
                    "run_progress": (index + 1) / len(jobs),
                    "evaluation/chunks_done": index + 1,
                    "evaluation/wall_seconds": time.time() - started,
                },
                step=offset + index + 1,
            )
        reference_report = {}
        if config.gate:
            install_stable_scatter_reference()
            reference_model = eqx.filter_jit(policy.cast_to_param)(
                restore_weights(
                    record["checkpoint"],
                    record["canonical_metadata_sha256"],
                    dataclasses.replace(model_config, moe_implementation="scatter"),
                    mesh,
                )
            )
            references = {
                length: FrozenFeatures(reference_model, mesh, sequence_length=length)
                for length in (4096, 8192)
            }
            tracker.log_summary({"evaluation/phase": "ordered-reference"})
            for name, (spec, rows) in reference_cases.items():
                length = 8192 if spec["kind"] == "sv" else 4096
                candidate = extract_rows(spec, rows, scorers[length], tokenizer)
                reference = extract_rows(spec, rows, references[length], tokenizer)
                for field in candidate:
                    np.testing.assert_array_equal(candidate[field], reference[field])
                reference_report[name] = {"ordered_fp32_reference_exact": True}
            tracker.log_summary({"validation/ordered_reference": reference_report})
        result = {
            "binding": binding,
            "worker": config.worker_index,
            "complete": True,
            "chunks": completed,
            "ordered_reference": reference_report,
        }
        write_bytes(
            config.output_prefix + f"/worker-{config.worker_index:02d}.json",
            json.dumps(result, sort_keys=True).encode(),
        )
    tracker.log_summary(
        {
            "evaluation/phase": "complete",
            "evaluation/result_uri": config.output_prefix
            + f"/worker-{config.worker_index:02d}.json",
        }
    )
    tracker.current_tracker().finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--requests-uri", required=True)
    parser.add_argument("--requests-sha256", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--gate", action="store_true")
    args = vars(parser.parse_args())
    dispatch(
        FeatureConfig(condition="pretrained", **args),
        worker=run,
        moe_implementation="sonic",
        timeout_hours=18,
    )


if __name__ == "__main__":
    main()
