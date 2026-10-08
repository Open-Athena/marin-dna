"""Validate inference backends and task-independent bidirectional DNA scoring."""

import argparse
import dataclasses
import gc
import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any, Literal

import equinox as eqx
import fsspec
import jax
import jmp
import numpy as np
from levanter import tracker
from levanter.callbacks.watch import WatchConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from exp582_moe.config import d1536_config
from exp582_moe.corpus import corpus_digest, open_dataset
from exp582_moe.data import (
    pretrained_dna_tokenizer,
    scratch_tokenizer,
    tokenizer_digest,
)
from exp582_moe.eval_runtime import PilotScorer
from exp582_moe.evaluation_binding import (
    CheckpointRole,
    validate_checkpoint_binding,
)
from exp582_moe.gpu_smoke import SmokeConfig, dispatch
from experiments.grug.moe_hero_ep.weights import restore_weights

DNA = "ACGT"
DNA_WITH_UNKNOWN = "ACGTN"
COMPLEMENT = str.maketrans("ACGT", "TGCA")
CONTEXTS = (128, 512, 2048, 4095)
FORWARD_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
VALIDATION_SOURCES = 256


@dataclasses.dataclass(frozen=True)
class MethodValidationConfig(SmokeConfig):
    checkpoint: str = ""
    checkpoint_metadata_digest: str = ""
    output_prefix: str = ""
    checkpoint_role: CheckpointRole = "dna-trained"

    @property
    def artifact_root(self) -> str:
        return self.output_prefix.rstrip("/") + "/wandb"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.nodes != 1 or len(self.checkpoint_metadata_digest) != 64:
            raise ValueError(
                "Method validation requires one node and a pinned checkpoint"
            )
        expected = (
            "s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/"
            "evaluations/final-v1/validation/"
        )
        if not self.output_prefix.startswith(expected):
            raise ValueError("Method validation output is outside the durable owner")


def reverse_complement(sequence: str) -> str:
    normalized = sequence.upper()
    if set(normalized) - set(DNA_WITH_UNKNOWN):
        raise ValueError("Intrinsic reconstruction requires DNA alphabet characters")
    return normalized.translate(COMPLEMENT)[::-1]


def canonical_window(sequence: str) -> str:
    normalized = sequence.upper()
    if len(normalized) != 8192 or set(normalized) - set(DNA_WITH_UNKNOWN):
        raise ValueError("Intrinsic validation inventory differs")
    return normalized


def validate_target_sequence(sequence: str, target: int) -> tuple[str, int]:
    if len(sequence) > 8192 or target <= 0 or target >= len(sequence):
        raise ValueError("Invalid intrinsic scoring sequence")
    return sequence, target


def validation_sequences() -> tuple[list[str], np.ndarray, str]:
    dataset = open_dataset("validation")
    total = len(dataset)
    # Midpoints between the 512 windows used for training-time held-out loss.
    indices = [
        (2 * i + 1) * total // (2 * VALIDATION_SOURCES)
        for i in range(VALIDATION_SOURCES)
    ]
    windows = [canonical_window(sequence) for sequence in dataset.get_batch(indices)]
    if len(windows) != VALIDATION_SOURCES:
        raise ValueError("Intrinsic validation inventory differs")
    # Unknown bases are valid model inputs, but not four-class reconstruction
    # targets. Replace only an ambiguous-center window with the next unused
    # corpus row whose center is canonical, preserving a deterministic sample.
    used = set(indices)
    for offset, (index, sequence) in enumerate(zip(indices, windows, strict=True)):
        if sequence[4096] in DNA:
            continue
        replacement = (index + 1) % total
        for _ in range(total):
            if replacement not in used:
                candidate = canonical_window(dataset.get_batch([replacement])[0])
                if candidate[4096] in DNA:
                    indices[offset] = replacement
                    windows[offset] = candidate
                    used.add(replacement)
                    break
            replacement = (replacement + 1) % total
        else:
            raise ValueError("Validation corpus has no unambiguous targets")
    oriented, split = [], []
    for index, sequence in enumerate(windows):
        membership = index % 2
        oriented.extend([sequence, reverse_complement(sequence)])
        split.extend([membership, membership])
    identity = hashlib.sha256(
        json.dumps(
            {"indices": indices, "corpus_sha256": corpus_digest()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return oriented, np.asarray(split, dtype=np.int8), identity


def encode(tokenizer: Any, sequences: Sequence[str]) -> list[list[int]]:
    ids = [
        tokenizer.encode(sequence, add_special_tokens=False).ids
        for sequence in sequences
    ]
    if any(
        len(row) != len(sequence) for row, sequence in zip(ids, sequences, strict=True)
    ):
        raise ValueError("Intrinsic examples are not character-tokenized")
    return ids


def score(
    model: Any,
    mesh: jax.sharding.Mesh,
    tokenizer: Any,
    sequences: list[str],
    positions: list[int],
    scorer: PilotScorer,
) -> np.ndarray:
    if len(sequences) != len(positions) or len({len(x) for x in sequences}) != 1:
        raise ValueError("Intrinsic scoring batch shape differs")
    outputs = []
    for start in range(0, len(sequences), scorer.batch_size):
        batch = sequences[start : start + scorer.batch_size]
        points = positions[start : start + scorer.batch_size]
        result = scorer.evaluate(
            encode(tokenizer, batch),
            [[point] for point in points],
            points,
            allele_frequency=False,
        )
        outputs.append(result["probabilities"][:, 0])
    return np.concatenate(outputs)


def align_right(probabilities: np.ndarray) -> np.ndarray:
    return probabilities[:, [3, 2, 1, 0]]


def combine(left: np.ndarray, right: np.ndarray, weight: float) -> np.ndarray:
    tiny = np.finfo(np.float64).tiny
    logits = weight * np.log(np.maximum(left, tiny))
    logits += (1 - weight) * np.log(np.maximum(align_right(right), tiny))
    logits -= logits.max(axis=-1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(axis=-1, keepdims=True)


def metrics(
    probabilities: np.ndarray, targets: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    selected = probabilities[mask]
    truth = targets[mask]
    rows = np.arange(len(selected))
    likelihood = np.maximum(selected[rows, truth], np.finfo(np.float64).tiny)
    return {
        "examples": len(selected),
        "accuracy": float(np.mean(selected.argmax(axis=-1) == truth)),
        "nll": float(-np.mean(np.log(likelihood))),
        "mean_true_probability": float(np.mean(likelihood)),
    }


def intrinsic_report(
    model: Any, mesh: jax.sharding.Mesh, tokenizer: Any
) -> tuple[dict[str, Any], list[str]]:
    windows, split, sample_digest = validation_sequences()
    center = 4096
    targets = np.asarray([DNA.index(sequence[center]) for sequence in windows])
    calibration = split == 0
    test = split == 1
    scorer = PilotScorer(
        model,
        mesh,
        [tokenizer.encode(base, add_special_tokens=False).ids[0] for base in DNA],
    )
    sweeps: dict[str, Any] = {}
    cached: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for context in CONTEXTS:
        left_sequences, right_sequences, positions = [], [], []
        for sequence in windows:
            left, position = validate_target_sequence(
                sequence[center - context : center + 1], context
            )
            right, right_position = validate_target_sequence(
                reverse_complement(sequence[center : center + context + 1]), context
            )
            if position != right_position:
                raise ValueError("Directional target positions differ")
            left_sequences.append(left)
            right_sequences.append(right)
            positions.append(position)
        left = score(model, mesh, tokenizer, left_sequences, positions, scorer)
        right = score(model, mesh, tokenizer, right_sequences, positions, scorer)
        cached[context] = left, right
        candidates = {
            str(weight): metrics(combine(left, right, weight), targets, calibration)
            for weight in FORWARD_WEIGHTS
        }
        selected_weight = min(
            FORWARD_WEIGHTS, key=lambda weight: candidates[str(weight)]["nll"]
        )
        sweeps[str(context)] = {
            "left_test": metrics(left, targets, test),
            "right_test": metrics(align_right(right), targets, test),
            "poe_equal_test": metrics(combine(left, right, 0.5), targets, test),
            "calibration_weights": candidates,
            "selected_forward_weight": selected_weight,
            "poe_selected_test": metrics(
                combine(left, right, selected_weight), targets, test
            ),
            "sequence_length": len(left_sequences[0]),
        }
    context = 2048
    prompts: dict[str, list[str]] = {"rotated": [], "sandwich": []}
    prompt_positions: dict[str, list[int]] = {"rotated": [], "sandwich": []}
    for sequence in windows:
        left = sequence[center - context : center]
        right = sequence[center + 1 : center + context + 1]
        target = sequence[center]
        for name, value in {
            "rotated": right + left + target,
            "sandwich": left + right + left + target,
        }.items():
            prepared, position = validate_target_sequence(value, len(value) - 1)
            prompts[name].append(prepared)
            prompt_positions[name].append(position)
    prompt_metrics = {
        name: metrics(
            score(model, mesh, tokenizer, sequences, prompt_positions[name], scorer),
            targets,
            test,
        )
        for name, sequences in prompts.items()
    }
    left, right = cached[context]
    selected_weight = sweeps[str(context)]["selected_forward_weight"]
    comparison = {
        "left": sweeps[str(context)]["left_test"],
        "right": sweeps[str(context)]["right_test"],
        "poe_equal": sweeps[str(context)]["poe_equal_test"],
        "poe_selected": sweeps[str(context)]["poe_selected_test"],
        **prompt_metrics,
    }
    best = min(comparison, key=lambda name: comparison[name]["nll"])
    return {
        "sample_sha256": sample_digest,
        "source_windows": VALIDATION_SOURCES,
        "oriented_examples": len(windows),
        "split": "source-index parity; both orientations remain in the same membership",
        "context_sweep": sweeps,
        "prompt_methods_2048": prompt_metrics,
        "selected_2048_method": best,
        "selected_2048_metrics": comparison[best],
        "selected_2048_forward_weight": selected_weight,
    }, windows[:64]


def parity_probabilities(
    model: Any,
    mesh: jax.sharding.Mesh,
    tokenizer: Any,
    windows: list[str],
) -> np.ndarray:
    scorer = PilotScorer(
        model,
        mesh,
        [tokenizer.encode(base, add_special_tokens=False).ids[0] for base in DNA],
    )
    positions = [1024, 4096, 7168]
    encoded = encode(tokenizer, windows)
    batches = []
    for start in range(0, len(encoded), scorer.batch_size):
        batch = encoded[start : start + scorer.batch_size]
        result = scorer.evaluate(
            batch,
            [positions] * len(batch),
            [4096] * len(batch),
            allele_frequency=False,
        )
        batches.append(result["probabilities"][:, : len(positions)])
    return np.concatenate(batches)


def pooled_inference_config(
    condition: Literal["pretrained", "random"],
) -> Any:
    """Recreate the exact pooled backend used by the completed exp582 runs."""
    return dataclasses.replace(
        d1536_config(condition),
        capacity_factor=32.0,
        pooled_transport_capacity_factor=8.0,
        expert_chunks=1,
    )


def parity_reference_config(
    condition: Literal["pretrained", "random"], checkpoint_role: CheckpointRole
) -> tuple[Any, str, int]:
    """Select a capacity-safe independent backend for score parity.

    The language-only router has never adapted to DNA and can send most DNA
    assignments to a few experts. A bounded-capacity EP backend would then test
    artificial token dropping rather than model-score parity, so use the exact
    dropless scatter implementation for that negative control.
    """
    if checkpoint_role == "language-base":
        return (
            dataclasses.replace(
                d1536_config(condition),
                moe_implementation="scatter",
                expert_chunks=1,
            ),
            "scatter",
            1,
        )
    return pooled_inference_config(condition), "pooled-ep", 8


def write_report(prefix: str, report: dict[str, Any]) -> tuple[str, str]:
    blob = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    digest = hashlib.sha256(blob).hexdigest()
    uri = prefix.rstrip("/") + "/report-" + digest + ".json"
    fs, path = fsspec.core.url_to_fs(uri)
    if fs.exists(path):
        if fs.cat(path) != blob:
            raise ValueError("Existing method report differs")
    else:
        with fs.open(path, "wb") as stream:
            stream.write(blob)
    if hashlib.sha256(fs.cat(path)).hexdigest() != digest:
        raise ValueError("Method report readback differs")
    return uri, digest


def run_validation(config: MethodValidationConfig) -> None:
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
            group="exp582-plantcad2-moe-final-eval-validation",
            tags=[
                "dna-exp582",
                "validation",
                "inference-methods",
                "H100",
                config.condition,
                config.checkpoint_role,
            ],
            background=False,
            save_code=False,
            replicate_path=config.artifact_root,
        ),
    )
    trainer.initialize()
    tracker.log_configuration(config)
    tracker.log_summary({"validation/phase": "loading_sonic", "validation/error": None})
    if jax.device_count() != 8 or any(
        "H100" not in x.device_kind for x in jax.devices()
    ):
        raise ValueError("Method validation requires one eight-H100 node")
    tokenizer = (
        scratch_tokenizer()
        if config.condition == "random"
        else pretrained_dna_tokenizer()
    )
    with fsspec.open(config.checkpoint + "/metadata.json", "rb") as stream:
        checkpoint_binding = validate_checkpoint_binding(
            checkpoint_role=config.checkpoint_role,
            condition=config.condition,
            checkpoint=config.checkpoint,
            checkpoint_metadata_digest=config.checkpoint_metadata_digest,
            metadata_blob=stream.read(),
            tokenizer_sha256=tokenizer_digest(tokenizer),
        )
    tracker.log_hyperparameters({"checkpoint_binding": checkpoint_binding})
    policy = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    sonic_mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    sonic_config = dataclasses.replace(
        d1536_config(config.condition), moe_implementation="sonic", expert_chunks=1
    )
    with jax.set_mesh(sonic_mesh):
        sonic = eqx.filter_jit(policy.cast_to_param)(
            restore_weights(
                config.checkpoint,
                config.checkpoint_metadata_digest,
                sonic_config,
                sonic_mesh,
            )
        )
        tracker.log_summary({"validation/phase": "intrinsic"})
        intrinsic, parity_windows = intrinsic_report(sonic, sonic_mesh, tokenizer)
        sonic_probabilities = parity_probabilities(
            sonic, sonic_mesh, tokenizer, parity_windows
        )
    del sonic
    gc.collect()
    jax.clear_caches()
    tracker.log_summary(
        {
            "validation/phase": "loading_pooled_ep",
            "validation/intrinsic": intrinsic,
        }
    )
    reference_config, reference_backend, expert_axis_size = parity_reference_config(
        config.condition, config.checkpoint_role
    )
    tracker.log_summary(
        {"validation/phase": f"loading_{reference_backend.replace('-', '_')}"}
    )
    reference_mesh = compact_grug_mesh(
        expert_axis_size=expert_axis_size, replica_axis_size=1
    )
    with jax.set_mesh(reference_mesh):
        reference = eqx.filter_jit(policy.cast_to_param)(
            restore_weights(
                config.checkpoint,
                config.checkpoint_metadata_digest,
                reference_config,
                reference_mesh,
            )
        )
        reference_probabilities = parity_probabilities(
            reference, reference_mesh, tokenizer, parity_windows
        )
    difference = np.abs(sonic_probabilities - reference_probabilities)
    positions = (1024, 4096, 7168)
    target = np.asarray(
        [
            [DNA.find(sequence[position]) for position in positions]
            for sequence in parity_windows
        ]
    )
    valid = target >= 0
    sonic_metrics = metrics(sonic_probabilities, target, valid)
    reference_metrics = metrics(reference_probabilities, target, valid)
    parity = {
        "examples": len(parity_windows),
        "positions_per_example": sonic_probabilities.shape[1],
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "p95_abs": float(np.quantile(difference, 0.95)),
        "p99_abs": float(np.quantile(difference, 0.99)),
        "argmax_agreement": float(
            np.mean(
                sonic_probabilities.argmax(axis=-1)
                == reference_probabilities.argmax(axis=-1)
            )
        ),
        "sonic_metrics": sonic_metrics,
        "reference_backend": reference_backend,
        "reference_metrics": reference_metrics,
        "absolute_nll_delta": abs(sonic_metrics["nll"] - reference_metrics["nll"]),
        "absolute_accuracy_delta": abs(
            sonic_metrics["accuracy"] - reference_metrics["accuracy"]
        ),
        "acceptance": {
            "max_abs_at_most": 0.05,
            "mean_abs_at_most": 0.003,
            "argmax_agreement_at_least": 0.95,
            "absolute_nll_delta_at_most": 0.01,
            "absolute_accuracy_delta_at_most": 0.03,
        },
        "sonic_model": dataclasses.asdict(sonic_config),
        "reference_model": dataclasses.asdict(reference_config),
    }
    accepted = (
        math.isfinite(parity["max_abs"])
        and parity["max_abs"] <= 0.05
        and parity["mean_abs"] <= 0.003
        and parity["argmax_agreement"] >= 0.95
        and parity["absolute_nll_delta"] <= 0.01
        and parity["absolute_accuracy_delta"] <= 0.03
    )
    parity["accepted"] = accepted
    tracker.log_summary({"validation/backend_parity": parity})
    if not accepted:
        raise ValueError(
            "Sonic and pooled-EP inference differ beyond tolerance: "
            f"max_abs={parity['max_abs']:.8f}, "
            f"mean_abs={parity['mean_abs']:.8f}, "
            f"argmax_agreement={parity['argmax_agreement']:.8f}, "
            f"nll_delta={parity['absolute_nll_delta']:.8f}, "
            f"accuracy_delta={parity['absolute_accuracy_delta']:.8f}"
        )
    report = {
        "schema": 1,
        "condition": config.condition,
        "checkpoint_role": config.checkpoint_role,
        "checkpoint_binding": checkpoint_binding,
        "checkpoint": config.checkpoint,
        "checkpoint_metadata_digest": config.checkpoint_metadata_digest,
        "tokenizer_sha256": tokenizer_digest(tokenizer),
        "intrinsic": intrinsic,
        "backend_parity": parity,
    }
    uri, digest = write_report(config.output_prefix, report)
    tracker.log_summary(
        {
            "validation/phase": "complete",
            "validation/report": uri,
            "validation/report_sha256": digest,
            "validation/backend_parity": parity,
        }
    )
    tracker.log({"run_progress": 1.0}, step=1)
    tracker.current_tracker().finish()


def run(config: MethodValidationConfig) -> None:
    try:
        run_validation(config)
    except Exception as error:
        try:
            wandb_tracker = tracker.get_tracker("wandb")
        except (RuntimeError, KeyError):
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
        "output-prefix",
    ]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--checkpoint-role",
        choices=("dna-trained", "language-base"),
        default="dna-trained",
    )
    config = MethodValidationConfig(**vars(parser.parse_args()))
    dispatch(
        config,
        worker=run,
        moe_implementation=d1536_config(config.condition).moe_implementation,
    )


if __name__ == "__main__":
    main()
