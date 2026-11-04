"""Read-only numerical qualification of a checkpoint in the 1T trajectory."""

import argparse
import dataclasses
import gc
import hashlib
import io
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pyarrow.parquet as pq
from levanter import tracker
from levanter.callbacks.watch import WatchConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from exp586_moe.config import d768_config
from exp586_moe.data import pretrained_dna_tokenizer, tokenizer_digest
from exp586_moe.evaluation_binding import validate_checkpoint_binding
from exp586_moe.final_checkpoint_gate import (
    GateConfig,
    collect,
    compare_readouts,
    replay_loss,
    windows_for_parity,
)
from exp586_moe.gpu_smoke import dispatch
from exp586_moe.method_validation import write_report
from exp586_moe.real_eval import read_bytes
from exp586_moe.stable_scatter_reference import install_stable_scatter_reference
from experiments.grug.moe_hero_ep.weights import restore_weights


def run(config: GateConfig) -> None:
    """Check full readouts and original loss without updating any model weights."""
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
            group="exp586-checkpoint-trajectory",
            tags=["dna-exp586", "validation", "backend-parity", "H100", config.cluster],
            background=False,
            save_code=False,
            replicate_path=config.artifact_root,
        ),
    )
    trainer.initialize()
    tracker.log_configuration(config)
    tracker.log_summary({"validation/phase": "binding", "training_updates": 0})
    try:
        blob = read_bytes(config.binding_uri)
        if hashlib.sha256(blob).hexdigest() != config.binding_sha256:
            raise ValueError("Checkpoint gate binding changed")
        binding = json.loads(blob)
        metadata = read_bytes(binding["checkpoint"] + "/metadata.json")
        if hashlib.sha256(metadata).hexdigest() != binding["raw_metadata_sha256"]:
            raise ValueError("Checkpoint metadata changed")
        tokenizer = pretrained_dna_tokenizer()
        validate_checkpoint_binding(
            checkpoint_role=binding["checkpoint_role"],
            condition="pretrained",
            checkpoint=binding["checkpoint"],
            checkpoint_metadata_digest=binding["canonical_metadata_sha256"],
            metadata_blob=metadata,
            tokenizer_sha256=tokenizer_digest(tokenizer),
        )
        sample = binding["validation_sample"]
        payload = read_bytes(sample["uri"])
        if hashlib.sha256(payload).hexdigest() != sample["sha256"]:
            raise ValueError("Original loss sample changed")
        sequences = pq.read_table(io.BytesIO(payload), columns=["seq"])[
            "seq"
        ].to_pylist()
        if len(sequences) != 512:
            raise ValueError("Expected original 512-window validation set")
        if jax.device_count() != 8 or any(
            "H100" not in d.device_kind for d in jax.devices()
        ):
            raise ValueError("Expected eight H100s")
        windows = windows_for_parity(sequences)
        mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
        policy = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16")
        outputs = {}
        report = {"binding": binding, "cases": len(windows), "oracles": {}, "loss": {}}
        for name in ("atomic_scatter", "ordered_scatter", "sonic"):
            if name == "ordered_scatter":
                install_stable_scatter_reference()
            implementation = "sonic" if name == "sonic" else "scatter"
            model_config = dataclasses.replace(
                d768_config("pretrained"),
                moe_implementation=implementation,
                expert_chunks=1,
            )
            tracker.log_summary({"validation/phase": "restoring-" + name})
            with jax.set_mesh(mesh):
                model = eqx.filter_jit(policy.cast_to_param)(
                    restore_weights(
                        binding["checkpoint"],
                        binding["canonical_metadata_sha256"],
                        model_config,
                        mesh,
                    )
                )
                if not all(
                    bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(model)
                ):
                    raise ValueError("Nonfinite checkpoint weights")
                outputs[name] = {}
                report["oracles"][name] = {}
                for length in (2048, 8192):
                    tracker.log_summary({"validation/phase": f"{name}-{length}"})
                    outputs[name][length], report["oracles"][name][str(length)] = (
                        collect(
                            model,
                            mesh,
                            tokenizer,
                            windows,
                            length,
                            exact_oracles=name != "atomic_scatter",
                            enforce_oracles=name != "atomic_scatter",
                        )
                    )
                tracker.log_summary({"validation/phase": name + "-loss"})
                pending = -model.stacked_blocks.stacked.mlp.router_bias.astype(
                    jnp.float32
                )
                report["loss"][name] = replay_loss(
                    model, pending, mesh, tokenizer, sequences
                )
            del model
            gc.collect()
            jax.clear_caches()
        comparisons = {}
        for length in (2048, 8192):
            reference, candidate = (
                outputs["ordered_scatter"][length],
                outputs["sonic"][length],
            )
            comparisons[str(length)] = compare_readouts(reference, candidate)
            if not all(np.array_equal(reference[k], candidate[k]) for k in reference):
                raise ValueError(
                    "Independent ordered-scatter readouts differ from Sonic"
                )
        report["independent_reference"] = comparisons
        report["atomic_scatter_diagnostics"] = {
            str(length): compare_readouts(
                outputs["atomic_scatter"][length], outputs["sonic"][length]
            )
            for length in (2048, 8192)
        }
        if abs(report["loss"]["ordered_scatter"] - report["loss"]["sonic"]) > 1e-6:
            raise ValueError("Independent loss implementations differ")
        if binding.get("training_eval_loss") is not None:
            report["training_loss_abs"] = abs(
                report["loss"]["atomic_scatter"] - binding["training_eval_loss"]
            )
            if report["training_loss_abs"] > 5e-5:
                raise ValueError(
                    "Checkpoint native loss differs from recorded training loss"
                )
        report["accepted"] = True
        uri, sha = write_report(config.output_prefix, report)
        tracker.log_summary(
            {
                "validation/phase": "complete",
                "validation/accepted": True,
                "validation/report_uri": uri,
                "validation/report_sha256": sha,
                "validation/loss": report["loss"],
            }
        )
        tracker.log({"run_progress": 1.0}, step=1)
        tracker.current_tracker().finish()
    except Exception as error:
        tracker.log_summary(
            {"validation/phase": "failed", "validation/error": str(error)}
        )
        tracker.get_tracker("wandb").run.finish(exit_code=1)
        raise


def main() -> None:
    """Dispatch one pinned checkpoint qualification at batch priority."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-id", "cluster", "binding-uri", "binding-sha256", "output-prefix"):
        parser.add_argument("--" + name, required=True)
    dispatch(
        GateConfig(condition="pretrained", **vars(parser.parse_args())),
        worker=run,
        moe_implementation="sonic",
        timeout_hours=3,
    )


if __name__ == "__main__":
    main()
