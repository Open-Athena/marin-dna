"""Bounded GPU inference checks on synthetic, separate genomic windows."""

import argparse
import dataclasses
import hashlib
import io
import json
from typing import Any, NoReturn

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, reshard
from jax.sharding import PartitionSpec as P
from levanter import tracker
from levanter.callbacks.watch import WatchConfig
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from exp582_moe.config import d1536_config
from exp582_moe.data import encode_windows, pretrained_dna_tokenizer, scratch_tokenizer
from exp582_moe.eval_runtime import NativeScorer
from exp582_moe.gpu_smoke import SmokeConfig, device_batch, dispatch
from exp582_moe.tracking import wandb_history_offset
from experiments.grug.moe_hero_ep.model import _CE_BLOCK_SIZES
from experiments.grug.moe_hero_ep.weights import restore_weights


class _NumericalRejection(ValueError):
    """The shared failure has already been flushed to W&B on every rank."""


@dataclasses.dataclass(frozen=True)
class EvalSmokeConfig(SmokeConfig):
    backend: str = "sonic"
    checkpoint: str = ""
    checkpoint_metadata_digest: str = ""
    requests_uri: str = ""
    requests_sha256: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.nodes != 1:
            raise ValueError("Evaluation smoke currently supports one eight-H100 node")
        if self.backend not in ("scatter", "sonic"):
            raise ValueError(
                "Evaluation requires a supported dropless inference backend"
            )
        prefix = "s3://marin-us-east-02a/tmp/ttl=7d/MarinDNA/exp582_plantcad2_moe/"
        if not self.checkpoint.startswith(prefix) or "smoke" not in self.checkpoint:
            raise ValueError(
                "Evaluation validation requires an exp582 smoke checkpoint"
            )
        if not self.requests_uri.startswith(prefix):
            raise ValueError("Evaluation requests must use the exp582 TTL prefix")
        for digest in (self.requests_sha256, self.checkpoint_metadata_digest):
            if len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest):
                raise ValueError("Expected explicit SHA-256 digests")

    @property
    def artifact_root(self) -> str:
        return self.checkpoint_root.removesuffix("/checkpoints") + "/evaluation"


def pool_features(
    offsets: list[tuple[int, int]], hidden: np.ndarray, bases: int, position: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pool real hidden states using independently checked base-span coverage."""
    spans = np.asarray(offsets)
    if hidden.shape[0] != len(spans) or spans[0, 0] != 0 or spans[-1, 1] != bases:
        raise ValueError("Hidden states and genomic window coverage differ")
    widths = spans[:, 1] - spans[:, 0]
    if np.any(widths <= 0) or np.any(spans[1:, 0] != spans[:-1, 1]):
        raise ValueError("Token offsets overlap or leave gaps")
    match = np.flatnonzero((spans[:, 0] <= position) & (position < spans[:, 1]))
    if len(match) != 1:
        raise ValueError("Variant is not covered by exactly one token")
    mean = np.sum(hidden * widths.astype(np.float32)[:, None], axis=0) / bases
    return mean, hidden[match[0]]


def native_loss_error(native_losses: np.ndarray, token_logs: list[np.ndarray]) -> float:
    """Reject nonfinite oracle values before measuring real-token loss parity."""
    if native_losses.ndim != 2 or len(token_logs) != len(native_losses):
        raise ValueError("Native loss and adapter batch dimensions differ")
    if not np.isfinite(native_losses).all():
        raise ValueError("Nonfinite native per-token loss")
    errors = []
    for losses, logs in zip(native_losses, token_logs, strict=True):
        if not np.isfinite(logs).all() or not 0 < len(logs) <= len(losses):
            raise ValueError("Invalid adapter token log probabilities")
        errors.append(float(np.max(np.abs(losses[: len(logs)] + logs))))
    error = max(errors)
    if not np.isfinite(error):
        raise ValueError("Nonfinite native loss parity error")
    return error


@eqx.filter_jit
def same_hidden_projection(
    hidden: jax.Array, head: jax.Array, labels: jax.Array, weights: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Compare full and blocked vocabulary reductions on identical activations."""
    head = reshard(head, P(None, None))
    logits = jnp.einsum(
        "bsh,hv->bsv",
        hidden,
        head,
        preferred_element_type=jnp.float32,
        out_sharding=P(("replica_dcn", "data", "expert"), None, None),
    )
    logs = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    logs = logs - jax.scipy.special.logsumexp(logits, axis=-1)
    losses = fused_linear_softmax_cross_entropy_loss(
        hidden,
        head,
        labels,
        weight=weights,
        reduction="none",
        dtype=jnp.float32,
        implementation="xla_fast_bwd",
        block_sizes=_CE_BLOCK_SIZES,
    )
    return logs, losses


def _run_evaluation(config: EvalSmokeConfig) -> None:
    mp = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    trainer = TrainerConfig(
        id=config.run_id,
        seed=0,
        train_batch_size=config.nodes * 8,
        num_train_steps=1,
        require_accelerator=True,
        mp=mp,
        use_explicit_mesh_axes=True,
        watch=WatchConfig(interval=0, watch_targets=[]),
        tracker=WandbConfig(
            entity="eric-czech",
            project="marin",
            name=config.run_id,
            group="exp582-plantcad2-moe-eval-validation",
            tags=["dna-exp582", "validation", "synthetic", "H100", config.condition],
            save_code=False,
            background=False,
            replicate_path=config.artifact_root,
        ),
    )
    trainer.initialize()
    log_offset = wandb_history_offset()

    def reject(message: str) -> NoReturn:
        # These checks operate on globally gathered outputs, so every rank agrees.
        tracker.log_summary({"validation/phase": "failed", "validation/error": message})
        tracker.get_tracker("wandb").run.finish(exit_code=1)
        multihost_utils.sync_global_devices("exp582-evaluation-numerical-failure")
        raise _NumericalRejection(message)

    assert jax.device_count() == config.nodes * 8
    assert all("H100" in device.device_kind for device in jax.devices())
    tracker.log_configuration(config)
    tracker.log_summary({"validation/phase": "loading", "validation/error": None})
    with fsspec.open(config.requests_uri, "rb") as source:
        payload = source.read()
    if hashlib.sha256(payload).hexdigest() != config.requests_sha256:
        raise ValueError("Evaluation request artifact changed")
    requests = json.loads(payload)
    if requests.get("schema") != 1 or requests.get("synthetic") is not True:
        raise ValueError("Only explicit synthetic validation requests are allowed")
    if requests.get("bases") != 8192:
        raise ValueError("Full-window validation requires 8192-base source windows")
    items = requests["requests"]
    if not 32 <= len(items) <= 64:
        raise ValueError("Bounded evaluation requires 32..64 requests")
    tokenizer = (
        scratch_tokenizer()
        if config.condition == "random"
        else pretrained_dna_tokenizer()
    )
    if (
        hashlib.sha256(tokenizer.to_str().encode()).hexdigest()
        != requests["tokenizer_sha256"]
    ):
        raise ValueError("Tokenizer differs from the request manifest")
    encodings = []
    for item in items:
        encoding = tokenizer.encode(item["sequence"], add_special_tokens=False)
        if encoding.ids != item["ids"] or not 2 <= len(encoding.ids) <= 8192:
            raise ValueError("Request IDs differ or would need truncation")
        encodings.append(encoding)

    with fsspec.open(config.checkpoint + "/metadata.json") as source:
        checkpoint_metadata = json.load(source)
    if checkpoint_metadata.get("tokenizer_sha256") != requests["tokenizer_sha256"]:
        raise ValueError("Checkpoint and evaluation tokenizer digests differ")

    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=config.nodes)
    model_config = dataclasses.replace(
        d1536_config(config.condition),
        moe_implementation=config.backend,
        expert_chunks=1,
    )
    tracker.log_hyperparameters(
        {"model": dataclasses.asdict(model_config), "training_updates": 0}
    )
    with jax.set_mesh(mesh):
        weights = restore_weights(
            config.checkpoint, config.checkpoint_metadata_digest, model_config, mesh
        )
        model = eqx.filter_jit(mp.cast_to_param)(weights)
        del weights
        adapter = NativeScorer(model, mesh)
        token_logs, mean_features, variant_features = [], [], []
        tracker.log_summary({"validation/phase": "inference"})
        reference_logs = reference_hidden = reference_batch_hidden = None
        for start in range(0, len(items), adapter.batch_size):
            group = items[start : start + adapter.batch_size]
            try:
                logs, hidden = adapter.evaluate([item["ids"] for item in group])
            except ValueError as error:
                reject(str(error))
            for i, (item, values, states) in enumerate(
                zip(group, logs, hidden, strict=True)
            ):
                try:
                    mean, variant = pool_features(
                        encodings[start + i].offsets,
                        states,
                        len(item["sequence"]),
                        item["variant_position"],
                    )
                except ValueError as error:
                    reject(str(error))
                if not all(np.isfinite(x).all() for x in (values, mean, variant)):
                    reject("Nonfinite inference output")
                token_logs.append(values)
                mean_features.append(mean)
                variant_features.append(variant)
            if start == 0:
                reference_logs, reference_hidden = logs[0], hidden[0]
                reference_batch_hidden = hidden
            tracker.log(
                {
                    "run_progress": 0.8
                    * min(len(items), start + len(group))
                    / len(items)
                },
                step=log_offset + start // adapter.batch_size + 1,
            )

        tracker.log_summary({"validation/phase": "native_loss_oracle"})
        first_windows = [item["sequence"] for item in items[: adapter.batch_size]]
        batch = device_batch(encode_windows(tokenizer, first_windows), mesh)
        # Isolate the projection from differences between compiled model forwards.
        hidden_array = np.zeros(
            (adapter.batch_size, 8192, model.config.hidden_dim), dtype=np.float32
        )
        assert reference_batch_hidden is not None
        for row, values in enumerate(reference_batch_hidden):
            hidden_array[row, : len(values)] = values
        if not np.array_equal(
            hidden_array,
            hidden_array.astype(model.output_proj.dtype).astype(np.float32),
        ):
            reject("Saved hidden states cannot be restored to the native dtype exactly")
        hidden_sharding = NamedSharding(
            mesh, P(("replica_dcn", "data", "expert"), None, None)
        )
        saved_hidden = jax.make_array_from_callback(
            hidden_array.shape,
            hidden_sharding,
            lambda index: hidden_array[index].astype(model.output_proj.dtype),
        )
        labels = jnp.pad(batch.tokens[:, 1:], ((0, 0), (0, 1)))
        direct_logs, projected_losses = same_hidden_projection(
            saved_hidden, model.output_proj, labels, batch.loss_weight
        )
        direct_logs, projected_losses = [
            np.asarray(multihost_utils.process_allgather(x, tiled=True))
            for x in (direct_logs, projected_losses)
        ]
        first_logs = token_logs[: len(first_windows)]
        projection_errors = {
            "validation/same_hidden_native_vs_adapter_max_abs": native_loss_error(
                projected_losses, first_logs
            ),
            "validation/same_hidden_dense_vs_adapter_max_abs": native_loss_error(
                -direct_logs, first_logs
            ),
            "validation/same_hidden_native_vs_dense_max_abs": native_loss_error(
                projected_losses,
                [
                    row[: len(values)]
                    for row, values in zip(direct_logs, first_logs, strict=True)
                ],
            ),
        }
        tracker.log_summary(projection_errors)
        if any(value > 2e-4 for value in projection_errors.values()):
            reject(
                "Same-hidden projection comparison exceeds the native-loss tolerance"
            )
        repeated_logs, repeated_hidden = adapter.evaluate(
            [item["ids"] for item in items[: adapter.batch_size]]
        )
        if not all(np.isfinite(x).all() for x in (*repeated_logs, *repeated_hidden)):
            reject("Nonfinite repeated-forward diagnostic output")
        repeat_errors = {
            "validation/repeated_forward_logp_max_abs": max(
                float(np.max(np.abs(a - b)))
                for a, b in zip(repeated_logs, first_logs, strict=True)
            ),
            "validation/repeated_forward_hidden_max_abs": max(
                float(np.max(np.abs(a - b)))
                for a, b in zip(repeated_hidden, reference_batch_hidden, strict=True)
            ),
        }
        tracker.log_summary(repeat_errors)
        if any(value != 0 for value in repeat_errors.values()):
            reject("Identical inference inputs produce different forward outputs")
        native_losses = eqx.filter_jit(
            lambda m, b: m.next_token_loss(
                b.tokens, b.loss_weight, mask=b.attn_mask, reduction="none"
            )
        )(model, batch)
        native_losses = np.asarray(
            multihost_utils.process_allgather(native_losses, tiled=True)
        )
        try:
            loss_error = native_loss_error(
                native_losses, token_logs[: len(first_windows)]
            )
        except ValueError as error:
            reject(str(error))
        # Same model and examples, independently implemented native fused loss.
        # FP32 reductions differ in order; record error before enforcing the bound.
        tracker.log_summary({"validation/native_loss_max_abs_error": loss_error})
        if loss_error > 2e-4:
            reject(f"Evaluation/native per-token loss disagreement: {loss_error}")
        try:
            alone_logs, alone_hidden = adapter.evaluate([items[0]["ids"]])
        except ValueError as error:
            reject(str(error))
        if (
            not np.isfinite(alone_logs[0]).all()
            or not np.isfinite(alone_hidden[0]).all()
        ):
            reject("Nonfinite repeated batch-context outputs")
        context_errors = {
            "validation/batch_context_logp_max_abs": float(
                np.max(np.abs(alone_logs[0] - reference_logs))
            ),
            "validation/batch_context_hidden_max_abs": float(
                np.max(np.abs(alone_hidden[0] - reference_hidden))
            ),
        }
        tracker.log_summary(context_errors)
        # Measure batch-context parity; interpret any nonzero value before approval.
        arrays = {
            "token_logs": np.concatenate(token_logs),
            "lengths": np.asarray([len(x) for x in token_logs]),
            "mean_features": np.asarray(mean_features),
            "variant_features": np.asarray(variant_features),
            "requests_sha256": np.asarray(config.requests_sha256),
        }
        output = io.BytesIO()
        np.savez_compressed(output, **arrays)
        blob = output.getvalue()
        metadata: dict[str, Any] = {
            "schema": 1,
            "config": dataclasses.asdict(config),
            "artifact_sha256": hashlib.sha256(blob).hexdigest(),
            "requests": len(items),
            "native_loss_max_abs_error": loss_error,
            **projection_errors,
            **repeat_errors,
            **context_errors,
        }
        if jax.process_index() == 0:
            with fsspec.open(config.artifact_root + "/outputs.npz", "wb") as target:
                target.write(blob)
            with fsspec.open(config.artifact_root + "/metadata.json", "w") as target:
                json.dump(metadata, target, sort_keys=True)
        multihost_utils.sync_global_devices("exp582-evaluation-artifact-written")
        tracker.log_summary(
            {
                "validation/phase": "gpu_complete",
                "validation/artifact": config.artifact_root,
            }
        )
        tracker.log(
            {"run_progress": 1.0},
            step=log_offset
            + (len(items) + adapter.batch_size - 1) // adapter.batch_size
            + 1,
        )
    tracker.current_tracker().finish()


def run_evaluation(config: EvalSmokeConfig) -> None:
    try:
        _run_evaluation(config)
    except _NumericalRejection:
        raise
    except Exception as error:
        tracker.log_summary(
            {"validation/phase": "failed", "validation/error": str(error)}
        )
        tracker.get_tracker("wandb").run.finish(exit_code=1)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["random", "pretrained"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--cluster", choices=["cw-us-east-02a", "cw-rno2a"], required=True
    )
    parser.add_argument("--nodes", type=int, choices=[1], default=1)
    parser.add_argument("--backend", choices=["scatter", "sonic"], default="sonic")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-metadata-digest", required=True)
    parser.add_argument("--requests-uri", required=True)
    parser.add_argument("--requests-sha256", required=True)
    config = EvalSmokeConfig(**vars(parser.parse_args()))
    dispatch(
        config,
        worker=run_evaluation,
        moe_implementation=config.backend,
    )


if __name__ == "__main__":
    main()
