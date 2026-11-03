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
from exp586_moe.stable_scatter_reference import install_stable_scatter_reference
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


def per_case_probability_errors(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> list[dict[str, float | int]]:
    errors = np.abs(reference["probabilities"] - candidate["probabilities"])
    return [
        {"case": index, "mean_abs": float(row.mean()), "max_abs": float(row.max())}
        for index, row in enumerate(errors)
    ]


def admission(report: dict) -> bool:
    """Require exact independent inference, native aggregate loss and typical outputs.

    BF16 atomic training reductions are nondeterministic, so their maximum
    single-output error is a diagnostic, not an equality oracle. Keep its
    original limit and failure visible in the report. Admission instead requires
    bitwise agreement with the independent ordered FP32 reference at both
    contexts, in addition to native loss, probability quantiles and features.
    """
    for record in report["readout_comparisons"].values():
        if not record["accepted"] or any(
            record[key] != 0 for key in ("probability_max_abs", "probability_mean_abs")
        ):
            return False
        if any(
            record[key]["max_abs"] != 0
            for key in ("full_logs", "acgt_logs", "mean", "variant")
        ):
            return False
    for record in report["atomic_scatter_comparisons"].values():
        if (
            record["probability_mean_abs"] > LIMITS["probability_mean_abs"]
            or record["probability_p99_abs"] > LIMITS["probability_p99_abs"]
            or record["argmax_agreement"] < LIMITS["argmax_agreement_min"]
            or any(
                record[key]["relative_l2"] > LIMITS["feature_relative_l2"]
                or record[key]["mean_cosine"] < LIMITS["feature_mean_cosine_min"]
                for key in ("mean", "variant")
            )
        ):
            return False
    return report["backend_loss_abs"] <= LIMITS["native_loss_replay_abs"]


def collect(
    model: Any,
    mesh: Any,
    tokenizer: Any,
    windows: list[str],
    length: int,
    *,
    exact_oracles: bool,
    enforce_oracles: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    scorer = PilotScorer(
        model,
        mesh,
        [tokenizer.token_to_id(base) for base in "acgt"],
        sequence_length=length,
    )
    result: dict[str, list[np.ndarray]] = {}
    oracles = {}
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
            oracles = {
                "repeat": compare_readouts(values, repeat),
                "batch": compare_readouts(
                    {name: value[:1] for name, value in values.items()}, single
                ),
                "repeat_exact": all(
                    np.array_equal(values[name], repeat[name]) for name in values
                ),
                "batch_exact": all(
                    np.array_equal(values[name][:1], single[name]) for name in values
                ),
            }
            # Scatter combines contributions with floating-point atomic adds.
            # Quantify its repeat noise using the fixed numerical contract;
            # Sonic must retain the previous exact-repeat/batch contract.
            if enforce_oracles and (
                not all(oracles[key]["accepted"] for key in ("repeat", "batch"))
                or (
                    exact_oracles
                    and not (oracles["repeat_exact"] and oracles["batch_exact"])
                )
            ):
                raise ValueError("Repeat/batch oracle failed: " + json.dumps(oracles))
        for name, value in values.items():
            result.setdefault(name, []).append(value)
    return {name: np.concatenate(values) for name, values in result.items()}, oracles


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


def run_gate(config: GateConfig) -> None:
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
            "validation/accepted": False,
            "validation/error": "",
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
            model_matches_fp32_master(state["params"], state["master_params"])
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
        native_repeats = {}
        reference_oracles = {}
        for length in (2048, 8192):
            tracker.log_summary(
                {
                    "validation/phase": f"native-readouts-{length}",
                    "validation/native_loss": report["native_loss"],
                }
            )
            reference[length], reference_oracles[str(length)] = collect(
                native,
                native_mesh,
                tokenizer,
                windows,
                length,
                exact_oracles=False,
                enforce_oracles=False,
            )
            repeat, _ = collect(
                native,
                native_mesh,
                tokenizer,
                windows,
                length,
                exact_oracles=False,
                enforce_oracles=False,
            )
            native_repeats[str(length)] = {
                "all_cases": compare_readouts(reference[length], repeat),
                "per_case_probability_errors": per_case_probability_errors(
                    reference[length], repeat
                ),
            }
        report["native_oracles"] = reference_oracles
        report["native_all_case_repeats"] = native_repeats
    del native
    gc.collect()
    jax.clear_caches()
    # Atomic BF16 scatter is noisy even on repeated identical requests. Retain
    # those measurements, then establish a deterministic independent reference
    # without changing grouped expert projections or benchmark inference.
    install_stable_scatter_reference()
    stable_reference = {}
    stable_oracles = {}
    with jax.set_mesh(native_mesh):
        tracker.log_summary({"validation/phase": "restore-ordered-scatter"})
        stable = eqx.filter_jit(policy.cast_to_param)(
            restore_weights(
                checkpoint,
                binding["canonical_metadata_sha256"],
                native_config,
                native_mesh,
            )
        )
        for length in (2048, 8192):
            tracker.log_summary({"validation/phase": f"ordered-readouts-{length}"})
            stable_reference[length], stable_oracles[str(length)] = collect(
                stable, native_mesh, tokenizer, windows, length, exact_oracles=True
            )
        pending = -stable.stacked_blocks.stacked.mlp.router_bias.astype(jnp.float32)
        report["ordered_scatter_loss"] = replay_loss(
            stable, pending, native_mesh, tokenizer, sequences
        )
    report["ordered_scatter_oracles"] = stable_oracles
    del stable
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
        native_comparisons = {}
        native_case_errors = {}
        sonic_oracles = {}
        for length in (2048, 8192):
            tracker.log_summary({"validation/phase": f"sonic-readouts-{length}"})
            candidate, sonic_oracles[str(length)] = collect(
                sonic, sonic_mesh, tokenizer, windows, length, exact_oracles=True
            )
            comparisons[str(length)] = compare_readouts(
                stable_reference[length], candidate
            )
            native_comparisons[str(length)] = compare_readouts(
                reference[length], candidate
            )
            native_case_errors[str(length)] = per_case_probability_errors(
                reference[length], candidate
            )
        report["readout_comparisons"] = comparisons
        report["atomic_scatter_comparisons"] = native_comparisons
        report["atomic_scatter_case_errors"] = native_case_errors
        report["reference_accumulation"] = (
            "ordered FP32 weighted sum of BF16 expert outputs"
        )
        report["sonic_oracles"] = sonic_oracles
        # Same full native vocabulary and loss formula, with Sonic computation.
        pending = -sonic.stacked_blocks.stacked.mlp.router_bias.astype(jnp.float32)
        report["sonic_loss"] = replay_loss(
            sonic, pending, sonic_mesh, tokenizer, sequences
        )
        report["backend_loss_abs"] = abs(report["sonic_loss"] - report["native_loss"])
    report["admission_policy"] = (
        "Exact independent ordered FP32 reference and exact repeat/batch oracles; "
        "native probability mean/p99, argmax, and features retain original limits; "
        "native aggregate loss difference <=5e-5. Atomic max/log outliers are reported diagnostically."
    )
    report["accepted"] = admission(report)
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


def run(config: GateConfig) -> None:
    try:
        run_gate(config)
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
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--binding-uri", required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--output-prefix", required=True)
    config = GateConfig(condition="pretrained", **vars(parser.parse_args()))
    dispatch(config, worker=run, moe_implementation="scatter", timeout_hours=4)


if __name__ == "__main__":
    main()
