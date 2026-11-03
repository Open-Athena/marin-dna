"""Read-only checkpoint integrity, native-loss replay, and full-readout parity."""

import argparse
import dataclasses
import gc
import hashlib
import io
import json
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pyarrow.parquet as pq
from levanter import tracker
from levanter.callbacks.watch import WatchConfig
from levanter.grug.grug_moe import MOE_DROPPED_ASSIGNMENTS_METRIC
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from exp586_moe.config import d768_config
from exp586_moe.data import encode_windows, pretrained_dna_tokenizer, tokenizer_digest
from exp586_moe.data_parallel import local_model, replicated_mesh
from exp586_moe.eval_runtime import PilotScorer
from exp586_moe.evaluation_binding import validate_checkpoint_binding
from exp586_moe.gpu_smoke import SmokeConfig, device_batch, dispatch
from exp586_moe.method_validation import reverse_complement, write_report
from exp586_moe.perf5_optimizer import model_matches_fp32_master
from exp586_moe.real_eval import read_bytes
from exp586_moe.train import PERF4_COMPILER_OPTIONS, validation_step
from experiments.grug import checkpointing
from experiments.grug.moe_hero_ep.model import Transformer, apply_qb_betas
from experiments.grug.moe_hero_ep.weights import restore_weights

# Fixed before looking at the new checkpoint's outputs. Probability limits are
# the previous d768 admission contract; feature/log limits extend that contract
# to the actual frozen-probe and suffix-likelihood inputs.
LIMITS = {
    "probability_mean_abs": 0.003,
    "probability_p99_abs": 0.025,
    "probability_max_abs": 0.075,
    "argmax_agreement_min": 0.95,
    "token_log_mean_abs": 0.01,
    "token_log_p99_abs": 0.075,
    "feature_relative_l2": 0.05,
    "feature_mean_cosine_min": 0.995,
    "native_loss_replay_abs": 0.00005,
    "backend_loss_abs": 0.001,
    "batch_probability_max_abs": 0.0002,
    "batch_feature_max_abs": 0.0002,
}


@dataclasses.dataclass(frozen=True)
class GateConfig(SmokeConfig):
    binding_uri: str = ""
    binding_sha256: str = ""
    output_prefix: str = ""

    @property
    def artifact_root(self) -> str:
        return self.output_prefix + "/wandb"


def compare_readouts(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> dict:
    """Compare distributions, suffix scores, and the features actually fitted."""
    for values in (reference, candidate):
        if any(not np.isfinite(value).all() for value in values.values()):
            raise ValueError("Nonfinite parity output")
    probabilities = np.abs(reference["probabilities"] - candidate["probabilities"])
    result = {
        "probability_mean_abs": float(probabilities.mean()),
        "probability_p99_abs": float(np.quantile(probabilities, 0.99)),
        "probability_max_abs": float(probabilities.max()),
        "argmax_agreement": float(
            np.mean(
                reference["probabilities"].argmax(-1)
                == candidate["probabilities"].argmax(-1)
            )
        ),
    }
    accepted = (
        result["probability_mean_abs"] <= LIMITS["probability_mean_abs"]
        and result["probability_p99_abs"] <= LIMITS["probability_p99_abs"]
        and result["probability_max_abs"] <= LIMITS["probability_max_abs"]
        and result["argmax_agreement"] >= LIMITS["argmax_agreement_min"]
    )
    for name in ("full_logs", "acgt_logs"):
        error = np.abs(reference[name] - candidate[name])
        result[name] = {
            "mean_abs": float(error.mean()),
            "p99_abs": float(np.quantile(error, 0.99)),
            "max_abs": float(error.max()),
        }
        accepted &= (
            result[name]["mean_abs"] <= LIMITS["token_log_mean_abs"]
            and result[name]["p99_abs"] <= LIMITS["token_log_p99_abs"]
        )
    for name in ("mean", "variant"):
        left, right = (
            reference[name].astype(np.float64),
            candidate[name].astype(np.float64),
        )
        relative = np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-30)
        cosine = np.sum(left * right, axis=-1) / np.maximum(
            np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1), 1e-30
        )
        result[name] = {
            "relative_l2": float(relative),
            "mean_cosine": float(cosine.mean()),
            "max_abs": float(np.abs(left - right).max()),
        }
        accepted &= (
            relative <= LIMITS["feature_relative_l2"]
            and cosine.mean() >= LIMITS["feature_mean_cosine_min"]
        )
    result["accepted"] = bool(accepted)
    return result


def windows_for_parity(sequences: list[str]) -> list[str]:
    """Fixed real-genome, reverse-complement, alternate, and low-complexity cases."""
    windows = [sequence.upper() for sequence in sequences[:16]]
    windows += [reverse_complement(sequence) for sequence in windows]
    alternates = []
    for sequence in windows[:16]:
        center = 4096
        base = next(base for base in "ACGT" if base != sequence[center])
        alternates.append(sequence[:center] + base + sequence[center + 1 :])
    windows += alternates
    windows += [
        (motif * 8192)[:8192]
        for motif in ("A", "C", "G", "T", "ACGT", "AT", "ACGTN", "N")
    ]
    return windows


def collect(
    model: Any, mesh: Any, tokenizer: Any, windows: list[str], length: int
) -> dict[str, np.ndarray]:
    scorer = PilotScorer(
        model,
        mesh,
        [tokenizer.token_to_id(base) for base in "acgt"],
        sequence_length=length,
    )
    result: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(windows), scorer.batch_size):
        cropped = [
            sequence[(8192 - length) // 2 : (8192 + length) // 2]
            for sequence in windows[start : start + scorer.batch_size]
        ]
        ids = [
            tokenizer.encode(sequence, add_special_tokens=False).ids
            for sequence in cropped
        ]
        if any(len(row) != length for row in ids):
            raise ValueError("Character-forcing changed")
        values = scorer.evaluate(
            ids,
            [[length // 4, length // 2, 3 * length // 4]] * len(ids),
            [length // 2] * len(ids),
            allele_frequency=True,
        )
        if start == 0:
            repeat = scorer.evaluate(
                ids,
                [[length // 4, length // 2, 3 * length // 4]] * len(ids),
                [length // 2] * len(ids),
                allele_frequency=True,
            )
            single = scorer.evaluate(
                ids[:1],
                [[length // 4, length // 2, 3 * length // 4]],
                [length // 2],
                allele_frequency=True,
            )
            for name in ("probabilities", "mean", "variant"):
                limit = (
                    LIMITS["batch_probability_max_abs"]
                    if name == "probabilities"
                    else LIMITS["batch_feature_max_abs"]
                )
                if (
                    not np.array_equal(values[name], repeat[name])
                    or np.max(np.abs(values[name][:1] - single[name])) > limit
                ):
                    raise ValueError(f"Repeat/batch oracle failed for {name}")
        for name, value in values.items():
            result.setdefault(name, []).append(value)
    return {name: np.concatenate(values) for name, values in result.items()}


def replay_loss(
    params: Any, pending: Any, mesh: Any, tokenizer: Any, windows: list[str]
) -> float:
    values = []
    for start in range(0, len(windows), 64):
        encoded = encode_windows(tokenizer, windows[start : start + 64])
        loss, metrics = validation_step(params, pending, device_batch(encoded, mesh))
        drops = np.asarray(metrics[MOE_DROPPED_ASSIGNMENTS_METRIC])
        if np.any(drops):
            raise ValueError("Native replay dropped expert assignments")
        values.append(float(loss))
    return float(np.mean(values))


def run(config: GateConfig) -> None:
    trainer = TrainerConfig(
        id=config.run_id,
        seed=0,
        train_batch_size=64,
        num_train_steps=1,
        require_accelerator=True,
        use_explicit_mesh_axes=True,
        watch=WatchConfig(interval=0, watch_targets=[]),
        tracker=WandbConfig(
            entity="eric-czech",
            project="marin",
            name=config.run_id,
            group="exp586-1t-final-evaluation",
            tags=["dna-exp586", "validation", "backend-parity", "H100", config.cluster],
            background=False,
            save_code=False,
            replicate_path=config.artifact_root,
        ),
    )
    trainer.initialize()
    tracker.log_configuration(config)
    tracker.log_summary(
        {
            "validation/phase": "binding",
            "training_updates": 0,
            "validation/limits": LIMITS,
        }
    )
    if jax.device_count() != 8 or any(
        "H100" not in device.device_kind for device in jax.devices()
    ):
        raise ValueError("Gate requires eight H100s")
    blob = read_bytes(config.binding_uri)
    if hashlib.sha256(blob).hexdigest() != config.binding_sha256:
        raise ValueError("Gate input binding changed")
    binding = json.loads(blob)
    checkpoint = binding["checkpoint"]
    metadata_blob = read_bytes(checkpoint + "/metadata.json")
    metadata = json.loads(metadata_blob)
    if (
        hashlib.sha256(metadata_blob).hexdigest() != binding["metadata_raw_sha256"]
        or metadata["exp586_clock"]["updates"] != 1907349
    ):
        raise ValueError("Final checkpoint differs")
    tokenizer = pretrained_dna_tokenizer()
    validate_checkpoint_binding(
        checkpoint_role="dna-trained",
        condition="pretrained",
        checkpoint=checkpoint,
        checkpoint_metadata_digest=binding["canonical_metadata_sha256"],
        metadata_blob=metadata_blob,
        tokenizer_sha256=tokenizer_digest(tokenizer),
    )
    sample = binding["validation_sample"]
    sample_blob = read_bytes(sample["uri"])
    if hashlib.sha256(sample_blob).hexdigest() != sample["sha256"]:
        raise ValueError("Loss sample changed")
    sequences = pq.read_table(io.BytesIO(sample_blob), columns=["seq"])[
        "seq"
    ].to_pylist()
    if len(sequences) != 512:
        raise ValueError("Native replay must use all 512 original windows")
    windows = windows_for_parity(sequences)
    report: dict[str, Any] = {
        "schema": 1,
        "checkpoint": checkpoint,
        "checkpoint_digest": binding["canonical_metadata_sha256"],
        "tokenizer_sha256": tokenizer_digest(tokenizer),
        "limits": LIMITS,
        "production_compiler_options": PERF4_COMPILER_OPTIONS,
        "cases": len(windows),
    }
    policy = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    native_mesh = replicated_mesh()
    native_config = local_model(d768_config("pretrained"))
    report["native_config"] = dataclasses.asdict(native_config)
    tracker.log_summary({"validation/phase": "restore-native"})
    with jax.set_mesh(native_mesh):
        template = eqx.filter_eval_shape(
            Transformer.init, native_config, key=jax.random.key(0)
        )
        param_template = eqx.filter_eval_shape(
            lambda: policy.cast_to_param(
                Transformer.init(native_config, key=jax.random.key(0))
            )
        )
        exemplar = {
            "params": param_template,
            "master_params": template,
            "pending_qb_betas": jax.ShapeDtypeStruct((8, 384), jnp.float32),
            "step": jax.ShapeDtypeStruct((), jnp.int32),
        }
        state = checkpointing.load_grug_checkpoint(
            state=exemplar, candidate=checkpoint, mesh=native_mesh, allow_partial=False
        )
        jax.block_until_ready(state)
        if int(state["step"]) != 1907349 or not bool(
            eqx.filter_jit(model_matches_fp32_master)(
                state["params"], state["master_params"]
            )
        ):
            raise ValueError("Stored compute weights do not equal cast FP32 masters")
        report["compute_matches_master_exactly"] = True
        report["parameter_arrays"] = len(jax.tree.leaves(state["params"]))
        if not all(
            bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(state)
        ):
            raise ValueError("Nonfinite stored state")
        tracker.log_summary(
            {
                "validation/phase": "native-loss-replay",
                "validation/compute_matches_master_exactly": True,
            }
        )
        report["native_loss"] = replay_loss(
            state["params"],
            state["pending_qb_betas"],
            native_mesh,
            tokenizer,
            sequences,
        )
        report["training_loss"] = binding["summary"]["eval/loss"]
        if (
            abs(report["native_loss"] - report["training_loss"])
            > LIMITS["native_loss_replay_abs"]
        ):
            raise ValueError(
                f"Native loss replay differs: {report['native_loss']} vs {report['training_loss']}"
            )
        native = eqx.filter_jit(policy.cast_to_compute)(
            apply_qb_betas(state["params"], state["pending_qb_betas"])
        )
        del state
        gc.collect()
        reference = {}
        for length in (2048, 8192):
            tracker.log_summary(
                {
                    "validation/phase": f"native-readouts-{length}",
                    "validation/native_loss": report["native_loss"],
                }
            )
            reference[length] = collect(native, native_mesh, tokenizer, windows, length)
    del native
    gc.collect()
    jax.clear_caches()
    sonic_mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    sonic_config = dataclasses.replace(
        d768_config("pretrained"), moe_implementation="sonic", expert_chunks=1
    )
    report["sonic_config"] = dataclasses.asdict(sonic_config)
    with jax.set_mesh(sonic_mesh):
        tracker.log_summary({"validation/phase": "restore-sonic"})
        sonic = eqx.filter_jit(policy.cast_to_param)(
            restore_weights(
                checkpoint,
                binding["canonical_metadata_sha256"],
                sonic_config,
                sonic_mesh,
            )
        )
        comparisons = {}
        for length in (2048, 8192):
            tracker.log_summary({"validation/phase": f"sonic-readouts-{length}"})
            candidate = collect(sonic, sonic_mesh, tokenizer, windows, length)
            comparisons[str(length)] = compare_readouts(reference[length], candidate)
        report["readout_comparisons"] = comparisons
        # Same full native vocabulary and loss formula, with Sonic computation.
        pending = -sonic.stacked_blocks.stacked.mlp.router_bias.astype(jnp.float32)
        report["sonic_loss"] = replay_loss(
            sonic, pending, sonic_mesh, tokenizer, sequences
        )
        report["backend_loss_abs"] = abs(report["sonic_loss"] - report["native_loss"])
    report["accepted"] = (
        all(record["accepted"] for record in comparisons.values())
        and report["backend_loss_abs"] <= LIMITS["backend_loss_abs"]
    )
    uri, digest = write_report(config.output_prefix, report)
    tracker.log_summary(
        {
            "validation/phase": "complete" if report["accepted"] else "failed",
            "validation/report_uri": uri,
            "validation/report_sha256": digest,
            "validation/accepted": report["accepted"],
            "validation/native_loss": report["native_loss"],
            "validation/sonic_loss": report["sonic_loss"],
            "validation/report": report,
        }
    )
    if not report["accepted"]:
        raise ValueError("Final checkpoint backend gate failed; report was saved")
    tracker.log({"run_progress": 1.0}, step=1)
    tracker.current_tracker().finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--binding-uri", required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--output-prefix", required=True)
    config = GateConfig(condition="pretrained", **vars(parser.parse_args()))
    dispatch(config, worker=run, moe_implementation="scatter", timeout_hours=4)


if __name__ == "__main__":
    main()
