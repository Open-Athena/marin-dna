"""Reload the retained final LoRA adapter and audit paired-score parity."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import wandb
from peft import PeftModel
from torch.nn.attention import SDPBackend, sdpa_kernel

from exp479_mntp.config import EXPERIMENT_TAGS, MODEL_ID, MODEL_REVISION, WANDB_PROJECT
from exp479_mntp.data import plan_sha256
from exp479_mntp.lora_mntp import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_EVALUATION_ARTIFACT,
    LORA_MODEL_ARTIFACT_PREFIX,
    LORA_RANK,
    LORA_TARGET_MODULES,
    annealed_attention_mask,
)
from exp479_mntp.modeling import ModelBundle, load_model_bundle
from exp479_mntp.paired_nucleotide_gate import (
    EXPECTED_VALIDATION_PLAN_SHA256,
    evaluate_readout,
)
from exp479_mntp.publishing import assert_budget_reserve, write_cost_estimate

WANDB_ENTITY = "gonzalobenegas"
FINAL_ADAPTER_ARTIFACT = (
    f"{WANDB_ENTITY}/{WANDB_PROJECT}/{LORA_MODEL_ARTIFACT_PREFIX}-step-1000:step-1000"
)
FINAL_EVALUATION_ARTIFACT = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{LORA_EVALUATION_ARTIFACT}:step-1000"
RELOAD_AUDIT_RUN_NAME = "dna-exp479-lora-final-reload-parity"
RELOAD_AUDIT_ARTIFACT = RELOAD_AUDIT_RUN_NAME
SERIALIZATION_CE_TOLERANCE = 1e-6
MATH_ATTENTION_CE_TOLERANCE = 2e-3


def assert_source_tokenizer_contract(tokenizer: Any) -> dict[str, Any]:
    """Require the unchanged seven-token source vocabulary and special IDs."""

    contract = {
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "all_special_ids": sorted(int(value) for value in tokenizer.all_special_ids),
    }
    expected = {
        "vocab_size": 7,
        "pad_token_id": 0,
        "unk_token_id": 1,
        "bos_token_id": 2,
        "eos_token_id": None,
        "all_special_ids": [0, 1, 2],
    }
    if contract != expected:
        raise RuntimeError(f"source tokenizer contract changed: {contract}")
    return contract


def assert_reloaded_adapter_contract(model: Any) -> dict[str, Any]:
    """Require the registered frozen rank-16 adapter and no trainable parameters."""

    configurations = getattr(model, "peft_config", {})
    if set(configurations) != {"default"}:
        raise RuntimeError(f"unexpected PEFT adapter names: {sorted(configurations)}")
    config = configurations["default"]
    target_modules = sorted(str(value) for value in config.target_modules)
    contract = {
        "rank": int(config.r),
        "alpha": int(config.lora_alpha),
        "dropout": float(config.lora_dropout),
        "bias": str(config.bias),
        "target_modules": target_modules,
        "modules_to_save": config.modules_to_save,
        "trainable_parameter_names": [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ],
    }
    expected_targets = sorted(LORA_TARGET_MODULES)
    if (
        contract["rank"] != LORA_RANK
        or contract["alpha"] != LORA_ALPHA
        or contract["dropout"] != LORA_DROPOUT
        or contract["bias"] != "none"
        or contract["target_modules"] != expected_targets
        or contract["modules_to_save"] not in (None, [])
        or contract["trainable_parameter_names"]
    ):
        raise RuntimeError(f"reloaded adapter contract changed: {contract}")
    return contract


def paired_score_parity(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    ce_tolerance: float,
) -> dict[str, float | int | bool]:
    """Compare two readouts on identical sample and target identities."""

    if ce_tolerance < 0:
        raise ValueError("CE tolerance must be non-negative")
    identity = ["sample_id", "target_nucleotide_index", "target_base"]
    columns = identity + [
        "nucleotide_ce",
        "nucleotide_correct",
        "full_vocab_ce",
        "full_vocab_correct",
    ]
    paired = reference[columns].merge(
        candidate[columns],
        on=identity,
        how="inner",
        validate="one_to_one",
        suffixes=("_reference", "_candidate"),
    )
    if len(paired) != len(reference) or len(paired) != len(candidate):
        raise RuntimeError("reload audit readouts do not contain identical paired targets")
    nucleotide_delta = (paired["nucleotide_ce_candidate"] - paired["nucleotide_ce_reference"]).abs()
    full_delta = (paired["full_vocab_ce_candidate"] - paired["full_vocab_ce_reference"]).abs()
    nucleotide_mismatches = int(
        (paired["nucleotide_correct_candidate"] != paired["nucleotide_correct_reference"]).sum()
    )
    full_mismatches = int(
        (paired["full_vocab_correct_candidate"] != paired["full_vocab_correct_reference"]).sum()
    )
    nucleotide_maximum = float(nucleotide_delta.max())
    full_maximum = float(full_delta.max())
    passed = (
        nucleotide_maximum <= ce_tolerance
        and full_maximum <= ce_tolerance
        and nucleotide_mismatches == 0
        and full_mismatches == 0
    )
    return {
        "passed": passed,
        "n_targets": len(paired),
        "ce_tolerance": ce_tolerance,
        "nucleotide_ce_maximum_absolute_delta": nucleotide_maximum,
        "nucleotide_ce_mean_absolute_delta": float(nucleotide_delta.mean()),
        "nucleotide_correctness_mismatches": nucleotide_mismatches,
        "full_vocab_ce_maximum_absolute_delta": full_maximum,
        "full_vocab_ce_mean_absolute_delta": float(full_delta.mean()),
        "full_vocab_correctness_mismatches": full_mismatches,
    }


def _training_format_full_mask(
    attention_mask: torch.Tensor,
    sample_ids: torch.Tensor,
) -> torch.Tensor:
    """Recreate the all-open additive mask used after attention annealing."""

    del sample_ids
    return annealed_attention_mask(
        attention_mask,
        future_edge_probability=1.0,
        seed=0,
        dtype=torch.float32,
    )


def _reloaded_bundle(adapter_dir: Path) -> ModelBundle:
    """Load a fresh source model and attach the retained inference adapter."""

    source = load_model_bundle(
        initialization="transferred",
        add_mask=False,
        attention_implementation="sdpa",
    )
    assert_source_tokenizer_contract(source.tokenizer)
    unk_token_id = source.tokenizer.unk_token_id
    if unk_token_id is None or int(unk_token_id) < 0:
        raise RuntimeError("source tokenizer lacks the selected existing UNK mask token")
    adapter = PeftModel.from_pretrained(source.model, adapter_dir, is_trainable=False)
    return ModelBundle(
        model=adapter,
        tokenizer=source.tokenizer,
        canonical_token_ids=source.canonical_token_ids,
        mask_token_id=int(unk_token_id),
        input_output_tied=source.input_output_tied,
    )


def run_lora_reload_audit(
    *,
    artifact_dir: Path,
    output_dir: Path,
    validation_plan: Path,
    batch_size: int,
) -> None:
    """Audit serialization and attention-path parity for the final adapter."""

    if not torch.cuda.is_available():
        raise RuntimeError("LoRA reload audit requires one CUDA GPU")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    validation_hash = plan_sha256(validation_plan)
    if validation_hash != EXPECTED_VALIDATION_PLAN_SHA256:
        raise RuntimeError(
            f"validation plan hash {validation_hash} differs from registered "
            f"{EXPECTED_VALIDATION_PLAN_SHA256}"
        )
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=WANDB_PROJECT,
        group="dna-exp479-lora-mntp-information-gate",
        name=RELOAD_AUDIT_RUN_NAME,
        tags=[*EXPERIMENT_TAGS, "lora", "reload-parity", "paired-information-gate"],
        config={
            "base_model": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "final_adapter_artifact": FINAL_ADAPTER_ARTIFACT,
            "final_evaluation_artifact": FINAL_EVALUATION_ARTIFACT,
            "validation_plan_sha256": validation_hash,
            "batch_size": batch_size,
            "serialization_ce_tolerance": SERIALIZATION_CE_TOLERANCE,
            "math_attention_ce_tolerance": MATH_ATTENTION_CE_TOLERANCE,
        },
    )
    if run is None:
        raise RuntimeError("W&B did not create the LoRA reload-audit run")

    try:
        assert_budget_reserve()
        evaluation_artifact = run.use_artifact(FINAL_EVALUATION_ARTIFACT, type="evaluation")
        evaluation_root = Path(evaluation_artifact.download(root=artifact_dir / "final-evaluation"))
        evaluation_manifest = json.loads(
            (evaluation_root / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            evaluation_manifest.get("status") != "completed"
            or evaluation_manifest.get("base_model") != MODEL_ID
            or evaluation_manifest.get("base_revision") != MODEL_REVISION
            or evaluation_manifest.get("base_frozen") is not True
            or evaluation_manifest.get("mask_token") != "[UNK]"
            or evaluation_manifest.get("validation_plan_sha256") != validation_hash
        ):
            raise RuntimeError("final evaluation manifest differs from the registered LoRA run")
        stored = pd.read_csv(evaluation_root / "paired-nucleotide-scores.csv")
        stored_final = stored[stored["readout"] == "lora_full_step1000"]
        stored_source = stored[stored["readout"] == "source_causal_adapter_disabled_step0"]
        if len(stored_final) != 640 or len(stored_source) != 640:
            raise RuntimeError(
                "retained evaluation artifact lacks the expected 640-target readouts"
            )

        adapter_artifact = run.use_artifact(FINAL_ADAPTER_ARTIFACT, type="model")
        adapter_metadata = dict(adapter_artifact.metadata or {})
        expected_adapter_metadata = {
            "optimizer_step": 1_000,
            "format": "peft_adapter",
            "base_model": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "mask_token": "[UNK]",
        }
        if any(
            adapter_metadata.get(key) != value for key, value in expected_adapter_metadata.items()
        ):
            raise RuntimeError(f"final adapter artifact metadata changed: {adapter_metadata}")
        adapter_root = Path(adapter_artifact.download(root=artifact_dir / "final-adapter"))
        bundle = _reloaded_bundle(adapter_root / "adapter")
        if not isinstance(bundle.model, PeftModel):
            raise TypeError("reloaded adapter did not produce a PEFT model")
        bundle.model.to(device="cuda").eval()

        tokenizer_contract = assert_source_tokenizer_contract(bundle.tokenizer)
        adapter_contract = assert_reloaded_adapter_contract(bundle.model)

        retention = json.loads(
            (evaluation_root / "retention-manifest.json").read_text(encoding="utf-8")
        )
        retained_final = [
            record
            for record in retention.get("artifacts", [])
            if record.get("kind") == "peft_adapter" and record.get("step") == 1_000
        ]
        if (
            retention.get("deletion_performed") is not False
            or len(retained_final) != 1
            or not adapter_artifact.id
            or retained_final[0].get("artifact_id") != adapter_artifact.id
        ):
            raise RuntimeError("final adapter identity differs from the retention manifest")

        reloaded_full = evaluate_readout(
            bundle,
            validation_plan=validation_plan,
            batch_size=batch_size,
            readout="reloaded_standard_full",
            attention_mode="full",
        )
        with bundle.model.disable_adapter():
            reloaded_source = evaluate_readout(
                bundle,
                validation_plan=validation_plan,
                batch_size=batch_size,
                readout="reloaded_source_causal",
                attention_mode="causal",
            )
        with sdpa_kernel([SDPBackend.MATH]):
            math_standard = evaluate_readout(
                bundle,
                validation_plan=validation_plan,
                batch_size=batch_size,
                readout="reloaded_math_standard_full",
                attention_mode="full",
            )
            math_training_format = evaluate_readout(
                bundle,
                validation_plan=validation_plan,
                batch_size=batch_size,
                readout="reloaded_math_training_format_full",
                attention_mode="full",
                attention_mask_transform=_training_format_full_mask,
            )

        checks: dict[str, dict[str, float | int | bool]] = {
            "adapter_serialization": paired_score_parity(
                stored_final,
                reloaded_full,
                ce_tolerance=SERIALIZATION_CE_TOLERANCE,
            ),
            "frozen_source_serialization": paired_score_parity(
                stored_source,
                reloaded_source,
                ce_tolerance=SERIALIZATION_CE_TOLERANCE,
            ),
            "math_attention_encoding": paired_score_parity(
                math_standard,
                math_training_format,
                ce_tolerance=MATH_ATTENTION_CE_TOLERANCE,
            ),
        }
        passed = all(bool(item["passed"]) for item in checks.values())
        if not passed:
            raise RuntimeError(f"LoRA reload parity audit failed: {checks}")

        scores = pd.concat(
            [reloaded_full, reloaded_source, math_standard, math_training_format],
            ignore_index=True,
        )
        scores_path = output_dir / "reload-paired-scores.csv"
        checks_path = output_dir / "reload-parity-checks.json"
        manifest_path = output_dir / "manifest.json"
        scores.to_csv(scores_path, index=False)
        checks_path.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
        cost_path = write_cost_estimate(artifact_dir=artifact_dir)
        manifest: dict[str, Any] = {
            "status": "completed",
            "passed": passed,
            "final_adapter_artifact": FINAL_ADAPTER_ARTIFACT,
            "final_adapter_artifact_id": adapter_artifact.id,
            "final_evaluation_artifact": FINAL_EVALUATION_ARTIFACT,
            "final_evaluation_artifact_id": evaluation_artifact.id,
            "checks": checks,
            "source_tokenizer_contract": tokenizer_contract,
            "reloaded_adapter_contract": adapter_contract,
            "adapter_artifact_metadata": adapter_metadata,
            "retained_final_adapter": retained_final[0],
            "elapsed_seconds": time.time() - started,
            "checkpoint_deletion": "not performed",
            "hugging_face_upload": "not performed",
            "vep_evaluation": "not performed",
            "nucleotide_dependency": "not performed",
            "knowledge_base_update": "not performed",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for group, values in checks.items():
            for key, value in values.items():
                run.summary[f"reload_audit/{group}/{key}"] = value
        run.summary["reload_audit/passed"] = passed
        result = wandb.Artifact(RELOAD_AUDIT_ARTIFACT, type="evaluation")
        for path in (scores_path, checks_path, manifest_path, cost_path):
            result.add_file(str(path))
        logged = run.log_artifact(result, aliases=["final-adapter", "step-1000"])
        logged.wait()
        run.finish(exit_code=0)
        del bundle
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException:
        run.finish(exit_code=1)
        raise
