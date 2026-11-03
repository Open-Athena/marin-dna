"""Read-only native validation-loss execution in the exp586 Marin environment.

The launch snapshot copies this file to ``exp586_moe.validation_loss_worker``;
the owned source stays with the PlantCAD evaluation analysis.
"""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import fsspec
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pyarrow.parquet as pq
from exp586_moe.config import d768_config
from exp586_moe.data import encode_windows, pretrained_dna_tokenizer, scratch_tokenizer
from exp586_moe.gpu_smoke import SmokeConfig, dispatch
from experiments.grug.moe_hero_ep.model import _CE_BLOCK_SIZES
from experiments.grug.moe_hero_ep.weights import restore_weights
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter import tracker
from levanter.callbacks.watch import WatchConfig
from levanter.grug.attention import AttentionMask
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig


@dataclasses.dataclass(frozen=True)
class LossConfig(SmokeConfig):
    request_uri: str = ""
    request_sha256: str = ""
    model_code: str = ""
    output_prefix: str = ""
    mode: str = "full"

    @property
    def artifact_root(self) -> str:
        return self.output_prefix


def read_json(uri: str) -> dict:
    with fsspec.open(uri, "rb") as stream:
        return json.load(stream)


def publish(uri: str, value: dict) -> None:
    if jax.process_index() != 0:
        return
    payload = (json.dumps(value, sort_keys=True) + "\n").encode()
    with fsspec.open(uri, "wb") as stream:
        stream.write(payload)
    with fsspec.open(uri, "rb") as stream:
        if stream.read() != payload:
            raise ValueError("Validation output failed storage readback")


def common_statistics(
    per_position: jax.Array,
    dna_logits: jax.Array,
    labels: jax.Array,
    weights: jax.Array,
    base_ids: jax.Array,
) -> jax.Array:
    matches = labels[..., None] == base_ids
    canonical = jnp.any(matches, axis=-1) & (weights > 0)
    targets = jnp.argmax(matches, axis=-1)
    dna_logs = jax.nn.log_softmax(dna_logits, axis=-1)
    dna_nll = -jnp.take_along_axis(dna_logs, targets[..., None], axis=-1)[..., 0]
    return jnp.stack(
        [
            jnp.sum(per_position),
            jnp.sum(weights),
            jnp.sum(jnp.where(canonical, per_position, 0)),
            jnp.sum(jnp.where(canonical, dna_nll, 0)),
            jnp.sum(canonical.astype(jnp.float32)),
        ]
    )


def restore_dense(record: dict, mesh: Any) -> Any:
    from levanter.checkpoint import load_checkpoint
    from levanter.layers.attention import AttentionBackend
    from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
    from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel

    payload = dict(record["model_config"])
    payload["rope"] = Llama3RotaryEmbeddingsConfig(**payload["rope"])
    payload["attn_backend"] = AttentionBackend.JAX_FLASH
    cfg = Qwen3Config(**payload)
    template = eqx.filter_eval_shape(
        Qwen3LMHeadModel.init, hax.Axis("vocab", 7), cfg, key=jax.random.key(0)
    )
    return load_checkpoint(
        {"model": template}, record["checkpoint"], mesh=mesh, allow_partial=False
    )["model"]


def run(config: LossConfig) -> None:
    with fsspec.open(config.request_uri, "rb") as stream:
        blob = stream.read()
    if hashlib.sha256(blob).hexdigest() != config.request_sha256:
        raise ValueError("Loss request changed")
    request = json.loads(blob)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if source_sha != request["worker_source_sha256"]:
        raise ValueError("Worker source differs from pinned request")
    record = next(x for x in request["models"] if x["code"] == config.model_code)
    checkpoint_metadata = read_json(record["checkpoint"] + "/metadata.json")
    observed_metadata_digest = hashlib.sha256(
        json.dumps(checkpoint_metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if observed_metadata_digest != record["canonical_metadata_sha256"]:
        raise ValueError("Checkpoint metadata differs")
    dense = record["vocab_size"] == 7
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
            group="exp586-validation-loss-correlation",
            tags=[
                "dna-exp586",
                "evaluation",
                "validation-loss",
                "H100",
                config.model_code,
            ],
            background=False,
            save_code=False,
            replicate_path=config.output_prefix + "/wandb",
        ),
    )
    trainer.initialize()
    if jax.device_count() != 8 or any(
        "H100" not in d.device_kind for d in jax.devices()
    ):
        raise ValueError("Validation loss requires one whole eight-H100 node")
    tracker.log_configuration(config)
    tracker.log_hyperparameters(
        {"model_binding": record, "request_sha256": config.request_sha256}
    )
    tracker.log_summary({"evaluation/phase": "restore", "training_updates": 0})
    mesh = compact_grug_mesh(expert_axis_size=1 if dense else 8, replica_axis_size=1)
    policy = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
    tokenizer = (
        pretrained_dna_tokenizer()
        if config.condition == "pretrained"
        else scratch_tokenizer()
    )
    base_ids_host = np.asarray([tokenizer.token_to_id(b) for b in "acgt"], np.int32)
    backend = record.get("source_metadata", {}).get(
        "runtime_moe_implementation", "fixed_pooled_wave_all_to_all"
    )
    if dense:
        base_ids_host = np.asarray([3, 4, 5, 6], np.int32)
    with jax.set_mesh(mesh):
        if dense:
            model = restore_dense(record, mesh)
        else:
            model_cfg = d768_config(config.condition)
            if record["model_config"]["hidden_dim"] == 1536:
                model_cfg = dataclasses.replace(
                    model_cfg,
                    hidden_dim=1536,
                    intermediate_dim=768,
                    shared_expert_intermediate_dim=768,
                    latent_dim=768,
                    num_layers=16,
                    num_heads=12,
                    num_kv_heads=3,
                    local_kv_heads=3,
                    initializer_std=record["model_config"]["initializer_std"],
                )
            model_cfg = dataclasses.replace(
                model_cfg,
                capacity_factor=32,
                pooled_transport_capacity_factor=8,
                moe_implementation=backend,
                remat_mode="none",
            )
            model = restore_weights(
                record["checkpoint"],
                record["canonical_metadata_sha256"],
                model_cfg,
                mesh,
            )
        model = eqx.filter_jit(policy.cast_to_compute)(model)
        base_ids = jnp.asarray(base_ids_host)

        @eqx.filter_jit
        def forward(model: Any, ids: jax.Array) -> jax.Array:
            labels = jnp.pad(ids[:, 1:], ((0, 0), (0, 1)))
            weights = jnp.ones(ids.shape, jnp.float32).at[:, -1].set(0)
            if dense:
                from levanter.layers.attention import AttentionMask as NamedMask
                from levanter.models.lm_model import split_activations
                from levanter.models.loss import maybe_fused_next_token_loss

                Batch, Pos = hax.Axis("batch", ids.shape[0]), model.Pos
                named = hax.named(ids, (Batch, Pos))
                with hax.axis_mapping({"batch": ("replica_dcn", "data", "expert")}):
                    hidden, _ = split_activations(
                        model.activations(named, NamedMask.causal())
                    )
                    head = model.get_lm_head()
                    per_position = (
                        maybe_fused_next_token_loss(
                            Pos,
                            model.Embed,
                            model.Vocab,
                            hidden,
                            head,
                            named,
                            loss_weight=hax.named(weights, (Batch, Pos)),
                            reduction=None,
                            reduction_axis=(),
                            logsumexp_weight=None,
                            dtype=jnp.float32,
                        )
                        .rearrange((Batch, Pos))
                        .array
                    )
                    hidden_array = hidden.rearrange((Batch, Pos, model.Embed)).array
                    head_array = head.rearrange((model.Embed, model.Vocab)).array
                drops = jnp.asarray(0, jnp.float32)
            else:
                hidden_array, router = model(ids, AttentionMask.causal())
                head_array = model.output_proj
                per_position = fused_linear_softmax_cross_entropy_loss(
                    hidden_array,
                    head_array,
                    labels,
                    weight=weights,
                    reduction="none",
                    logsumexp_weight=None,
                    dtype=jnp.float32,
                    implementation="xla_fast_bwd",
                    block_sizes=_CE_BLOCK_SIZES,
                )
                drops = jnp.sum(
                    router["capacity_overflow_per_layer"].astype(jnp.float32)
                )
            dna_logits = jnp.einsum(
                "bsh,hv->bsv",
                hidden_array,
                head_array.at[:, base_ids].get(out_sharding=P(None, None)),
                preferred_element_type=jnp.float32,
                out_sharding=P(("replica_dcn", "data", "expert"), None, None),
            )
            values = common_statistics(
                per_position, dna_logits, labels, weights, base_ids
            )
            return jnp.concatenate([values, drops[None]])

        sharding = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
        result = {
            "schema": 1,
            "model_code": config.model_code,
            "request_sha256": config.request_sha256,
            "checkpoint": record["checkpoint"],
            "checkpoint_digest": record["canonical_metadata_sha256"],
            "training_wandb": record["wandb_url"],
            "validation_wandb": f"https://wandb.ai/eric-czech/marin/runs/{config.run_id}",
            "native_vocabulary": record["vocab_size"],
            "worker_source_sha256": source_sha,
            "runtime_packages": {
                name: importlib.metadata.version(name)
                for name in ("marin-iris", "marin-levanter", "jax", "marin-haliax")
            },
            "backend": "JAX_FLASH" if dense else backend,
            "samples": {},
        }
        phases = ["common-65536"] if dense else ["original-moe-512", "common-65536"]
        began = time.monotonic()
        history_step = 0
        microbatch_size = 128 if dense else 16
        result["forward_microbatch_size"] = microbatch_size
        for phase in phases:
            info = request["inputs"]["samples"][phase]
            tracker.log_summary({"evaluation/phase": phase + "/load"})
            with fsspec.open(info["uri"], "rb") as stream:
                payload = stream.read()
            if hashlib.sha256(payload).hexdigest() != info["sha256"]:
                raise ValueError("Validation windows changed")
            import io

            sequences = pq.read_table(io.BytesIO(payload), columns=["seq"])[
                "seq"
            ].to_pylist()
            if config.mode == "canary":
                sequences = sequences[:64]
            totals = np.zeros(6, np.float64)
            batches = []
            tracker.log_summary({"evaluation/phase": phase + "/score"})
            for start in range(0, len(sequences), microbatch_size):
                encoded = encode_windows(
                    tokenizer, sequences[start : start + microbatch_size]
                )
                host_ids = encoded.token_ids
                ids = jax.make_array_from_callback(
                    host_ids.shape, sharding, lambda index, batch=host_ids: batch[index]
                )
                values = np.asarray(forward(model, ids), dtype=np.float64)
                if not np.isfinite(values).all() or values[-1] != 0:
                    raise ValueError(f"Invalid inference statistics: {values}")
                totals += values
                batches.append(values.tolist())
                history_step += 1
                if (
                    start == 0
                    or (start + microbatch_size) % 1024 == 0
                    or start + microbatch_size >= len(sequences)
                ):
                    tracker.log(
                        {
                            "run_progress": (
                                0.0
                                if phase == "original-moe-512"
                                else min(start + microbatch_size, len(sequences))
                            )
                            / 65536,
                            "evaluation/examples": min(
                                start + microbatch_size, len(sequences)
                            ),
                            "evaluation/native_loss": totals[0] / totals[1],
                            "evaluation/acgt_loss": totals[3] / totals[4],
                            "evaluation/dropped_assignments": totals[5],
                            "evaluation/elapsed_seconds": time.monotonic() - began,
                        },
                        step=history_step,
                    )
            sample = {
                "examples": len(sequences),
                "targets": int(totals[1]),
                "native_loss": totals[0] / totals[1],
                "canonical_native_loss": totals[2] / totals[4],
                "acgt_conditional_loss": totals[3] / totals[4],
                "canonical_targets": int(totals[4]),
                "dropped_assignments": int(totals[5]),
                "input_sha256": info["sha256"],
                "batch_statistics": batches,
            }
            if phase == "original-moe-512" and config.mode != "canary":
                sample["reported_loss"] = record["original_loss"]
                sample["reproduction_delta"] = (
                    sample["native_loss"] - record["original_loss"]
                )
                if abs(sample["reproduction_delta"]) > 2e-4:
                    result["samples"][phase] = sample
                    publish(config.output_prefix + "/reproduction-failed.json", result)
                    raise ValueError(
                        f"Original validation loss not reproduced: {sample['reproduction_delta']}"
                    )
            result["samples"][phase] = sample
            publish(config.output_prefix + "/result.json", result)
            tracker.log_summary({f"validation/{phase}/loss": sample["native_loss"]})
            multihost_utils.sync_global_devices(phase + "-complete")
        tracker.log_summary(
            {
                "evaluation/phase": "complete",
                "evaluation/seconds": time.monotonic() - began,
            }
        )
        tracker.log({"run_progress": 1.0}, step=history_step + 1)
    tracker.current_tracker().finish()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=["scratch", "pretrained"], required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--cluster", required=True)
    p.add_argument("--request-uri", required=True)
    p.add_argument("--request-sha256", required=True)
    p.add_argument("--model-code", required=True)
    p.add_argument("--output-prefix", required=True)
    p.add_argument("--mode", choices=["canary", "full"], default="full")
    args = vars(p.parse_args())
    dispatch(LossConfig(**args), worker=run, timeout_hours=12)


if __name__ == "__main__":
    main()
