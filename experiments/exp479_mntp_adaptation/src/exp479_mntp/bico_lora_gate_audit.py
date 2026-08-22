"""Correct the retained BICO LoRA gate and audit its final adapter reload."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import wandb
from peft import PeftModel

from exp479_mntp.bico_attention_diagnostic import install_reflected_future_rope
from exp479_mntp.bico_lora_mntp import (
    BICO_LORA_EVALUATION_ARTIFACT,
    BICO_LORA_MODEL_PREFIX,
    evaluate_bico_readout,
)
from exp479_mntp.config import (
    BUDGET_USD,
    EXPERIMENT_TAGS,
    MODEL_ID,
    MODEL_REVISION,
    WANDB_PROJECT,
)
from exp479_mntp.data import plan_sha256
from exp479_mntp.lora_mntp import plot_lora_trajectory
from exp479_mntp.lora_reload_audit import (
    MATH_ATTENTION_CE_TOLERANCE,
    assert_reloaded_adapter_contract,
    assert_source_tokenizer_contract,
    configure_training_evaluation_numerics,
    paired_score_parity,
)
from exp479_mntp.modeling import ModelBundle, load_model_bundle
from exp479_mntp.paired_nucleotide_gate import (
    EXPECTED_VALIDATION_PLAN_SHA256,
    evaluate_readout,
    information_gate,
    paired_comparison,
    summarize_readouts,
)
from exp479_mntp.publishing import assert_budget_reserve, write_cost_estimate

WANDB_ENTITY = "gonzalobenegas"
FINAL_ADAPTER_ARTIFACT = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{BICO_LORA_MODEL_PREFIX}-step-1000:v0"
FINAL_EVALUATION_ARTIFACT = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{BICO_LORA_EVALUATION_ARTIFACT}:v0"
AUDIT_RUN_NAME = "dna-exp479-bico-lora-corrected-information-gate-audit"
AUDIT_ARTIFACT = AUDIT_RUN_NAME
BASELINE_READOUT = "source_causal_adapter_disabled_step0"
FINAL_READOUT = "lora_full_step1000"
MAXIMUM_INSTANCE_HOURS = 1.0


def corrected_trajectory_tables(
    stored_scores: pd.DataFrame,
    corrected_source: pd.DataFrame,
    reloaded_final: pd.DataFrame,
    *,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Replace the misreported source and final rows before recomputing the gate."""

    if n_bootstrap <= 0:
        raise ValueError("bootstrap count must be positive")
    candidates = stored_scores[stored_scores["readout"].str.startswith("lora_full_step")].copy()
    expected = {
        "lora_full_step0000": 0,
        "lora_full_step0025": 25,
        "lora_full_step0050": 50,
        "lora_full_step0100": 100,
        "lora_full_step0200": 200,
        "lora_full_step0300": 300,
        "lora_full_step0400": 400,
        "lora_full_step0500": 500,
        "lora_full_step0600": 600,
        "lora_full_step0700": 700,
        "lora_full_step0800": 800,
        "lora_full_step0900": 900,
        FINAL_READOUT: 1_000,
    }
    counts = candidates.groupby("readout").size().to_dict()
    if set(counts) != set(expected) or any(int(counts[name]) != 640 for name in expected):
        raise RuntimeError(f"retained BICO trajectory is incomplete: {counts}")
    candidates = candidates[candidates["readout"] != FINAL_READOUT]
    final = reloaded_final.copy()
    final["readout"] = FINAL_READOUT
    final["optimizer_step"] = 1_000
    source = corrected_source.copy()
    source["readout"] = BASELINE_READOUT
    source["optimizer_step"] = 0
    candidates = pd.concat([candidates, final], ignore_index=True)
    candidates["optimizer_step"] = candidates["readout"].map(expected)
    scores = pd.concat([source, candidates], ignore_index=True)

    identity = scores.groupby(["sample_id", "target_nucleotide_index", "target_base"]).size()
    if len(identity) != 640 or not (identity == len(expected) + 1).all():
        raise RuntimeError("corrected BICO trajectory does not use identical paired targets")

    summary = summarize_readouts(scores)
    summary["optimizer_step"] = summary["readout"].map({BASELINE_READOUT: 0} | expected)
    comparisons: list[dict[str, object]] = []
    for candidate, step in expected.items():
        comparison = paired_comparison(
            scores,
            candidate=candidate,
            baseline=BASELINE_READOUT,
            n_bootstrap=n_bootstrap,
        )
        comparison["optimizer_step"] = step
        comparisons.append(comparison)
    comparison_frame = pd.DataFrame(comparisons)
    return scores, summary, comparison_frame, information_gate(comparisons[-1])


def _source_bundle() -> ModelBundle:
    """Load the frozen source while selecting its existing PAD as the masked input."""

    source = load_model_bundle(
        initialization="transferred",
        add_mask=False,
        attention_implementation="sdpa",
    )
    assert_source_tokenizer_contract(source.tokenizer)
    pad_token_id = source.tokenizer.pad_token_id
    if pad_token_id is None or int(pad_token_id) < 0:
        raise RuntimeError("source tokenizer lacks the BICO PAD token")
    return ModelBundle(
        model=source.model,
        tokenizer=source.tokenizer,
        canonical_token_ids=source.canonical_token_ids,
        mask_token_id=int(pad_token_id),
        input_output_tied=source.input_output_tied,
    )


def _attach_adapter(source: ModelBundle, adapter_dir: Path) -> ModelBundle:
    """Attach the retained adapter and install the corrected reflected-RoPE hook."""

    adapter = PeftModel.from_pretrained(source.model, adapter_dir, is_trainable=False)
    install_reflected_future_rope(adapter)
    return ModelBundle(
        model=adapter,
        tokenizer=source.tokenizer,
        canonical_token_ids=source.canonical_token_ids,
        mask_token_id=source.mask_token_id,
        input_output_tied=source.input_output_tied,
    )


def run_bico_lora_gate_audit(
    *,
    artifact_dir: Path,
    output_dir: Path,
    validation_plan: Path,
    batch_size: int,
    n_bootstrap: int,
) -> None:
    """Recompute the true causal gate and verify the retained final adapter."""

    numeric_controls = configure_training_evaluation_numerics()
    if not torch.cuda.is_available():
        raise RuntimeError("BICO LoRA gate audit requires one CUDA GPU")
    if batch_size <= 0 or n_bootstrap <= 0:
        raise ValueError("batch size and bootstrap count must be positive")
    validation_hash = plan_sha256(validation_plan)
    if validation_hash != EXPECTED_VALIDATION_PLAN_SHA256:
        raise RuntimeError("BICO LoRA audit validation plan differs from the fixed gate")
    prior_cost = float(os.getenv("EXP479_PRIOR_COST_USD", "0"))
    price = float(os.getenv("EXP479_INSTANCE_PRICE_PER_HOUR_USD", "1.006"))
    if prior_cost + MAXIMUM_INSTANCE_HOURS * price >= BUDGET_USD:
        raise RuntimeError("BICO LoRA audit projection reaches the issue budget cap")

    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=WANDB_PROJECT,
        group="dna-exp479-bico-lora-information-gate",
        name=AUDIT_RUN_NAME,
        tags=[*EXPERIMENT_TAGS, "bico", "lora", "causal-baseline-fix", "reload-parity"],
        config={
            "base_model": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "final_adapter_artifact": FINAL_ADAPTER_ARTIFACT,
            "invalid_evaluation_artifact": FINAL_EVALUATION_ARTIFACT,
            "validation_plan_sha256": validation_hash,
            "batch_size": batch_size,
            "n_bootstrap": n_bootstrap,
            "maximum_instance_hours": MAXIMUM_INSTANCE_HOURS,
            "numeric_controls": numeric_controls,
        },
    )
    if run is None:
        raise RuntimeError("W&B did not create the BICO LoRA gate-audit run")

    try:
        assert_budget_reserve()
        evaluation_artifact = run.use_artifact(FINAL_EVALUATION_ARTIFACT, type="evaluation")
        evaluation_root = Path(
            evaluation_artifact.download(root=artifact_dir / "invalid-final-evaluation")
        )
        stored_manifest = json.loads(
            (evaluation_root / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            stored_manifest.get("status") != "completed"
            or stored_manifest.get("base_model") != MODEL_ID
            or stored_manifest.get("base_revision") != MODEL_REVISION
            or stored_manifest.get("base_frozen") is not True
            or stored_manifest.get("validation_plan_sha256") != validation_hash
            or stored_manifest.get("physical_batch_size") != 94
            or stored_manifest.get("accumulation_steps") != 1
        ):
            raise RuntimeError("retained BICO evaluation manifest changed")
        stored_scores = pd.read_csv(evaluation_root / "paired-nucleotide-scores.csv")
        stored_invalid_source = stored_scores[stored_scores["readout"] == BASELINE_READOUT]
        stored_final = stored_scores[stored_scores["readout"] == FINAL_READOUT]
        if len(stored_invalid_source) != 640 or len(stored_final) != 640:
            raise RuntimeError("retained BICO evaluation lacks the expected final readouts")

        adapter_artifact = run.use_artifact(FINAL_ADAPTER_ARTIFACT, type="model")
        adapter_metadata = dict(adapter_artifact.metadata or {})
        expected_metadata = {
            "optimizer_step": 1_000,
            "format": "peft_adapter",
            "base_model": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "mask_token": "[PAD]",
            "attention": "BICO reflected future RoPE",
        }
        if any(adapter_metadata.get(key) != value for key, value in expected_metadata.items()):
            raise RuntimeError(f"final BICO adapter metadata changed: {adapter_metadata}")
        adapter_root = Path(adapter_artifact.download(root=artifact_dir / "final-adapter"))

        source = _source_bundle()
        source.model.to(device="cuda").eval()
        source_standard = evaluate_readout(
            source,
            validation_plan=validation_plan,
            batch_size=batch_size,
            readout="source_causal_standard_sdpa",
            attention_mode="causal",
        )
        bundle = _attach_adapter(source, adapter_root / "adapter")
        if not isinstance(bundle.model, PeftModel):
            raise TypeError("reloaded BICO adapter did not produce a PEFT model")
        bundle.model.to(device="cuda").eval()
        tokenizer_contract = assert_source_tokenizer_contract(bundle.tokenizer)
        adapter_contract = assert_reloaded_adapter_contract(bundle.model)

        with bundle.model.disable_adapter():
            source_hook_causal = evaluate_readout(
                bundle,
                validation_plan=validation_plan,
                batch_size=batch_size,
                readout="source_causal_corrected_hook",
                attention_mode="causal",
            )
            source_bug_reproduction = evaluate_readout(
                bundle,
                validation_plan=validation_plan,
                batch_size=batch_size,
                readout="source_full_bico_pad_attended_bug_reproduction",
                attention_mode="full",
            )
        reloaded_final = evaluate_bico_readout(
            bundle,
            validation_plan=validation_plan,
            batch_size=batch_size,
            readout=FINAL_READOUT,
        )

        checks = {
            "causal_hook_matches_standard_sdpa": paired_score_parity(
                source_standard,
                source_hook_causal,
                ce_tolerance=MATH_ATTENTION_CE_TOLERANCE,
            ),
            "stored_mislabel_matches_full_attention_reproduction": paired_score_parity(
                stored_invalid_source,
                source_bug_reproduction,
                ce_tolerance=MATH_ATTENTION_CE_TOLERANCE,
            ),
            "final_adapter_reload_matches_training_artifact": paired_score_parity(
                stored_final,
                reloaded_final,
                ce_tolerance=MATH_ATTENTION_CE_TOLERANCE,
            ),
        }
        audit_passed = all(bool(check["passed"]) for check in checks.values())
        if not audit_passed:
            raise RuntimeError(f"BICO causal/reload audit failed: {checks}")

        scores, summary, comparisons, gate = corrected_trajectory_tables(
            stored_scores,
            source_standard,
            reloaded_final,
            n_bootstrap=n_bootstrap,
        )
        scores_path = output_dir / "corrected-paired-nucleotide-scores.csv"
        summary_path = output_dir / "corrected-paired-nucleotide-summary.csv"
        comparisons_path = output_dir / "corrected-paired-nucleotide-comparisons.csv"
        checks_path = output_dir / "audit-checks.json"
        gate_path = output_dir / "corrected-information-gate.json"
        manifest_path = output_dir / "manifest.json"
        figure_path = output_dir / "figures" / "corrected-paired-nucleotide-trajectory"
        scores.to_csv(scores_path, index=False)
        summary.to_csv(summary_path, index=False)
        comparisons.to_csv(comparisons_path, index=False)
        checks_path.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        plot_lora_trajectory(summary, comparisons, figure_path)
        cost_path = write_cost_estimate(artifact_dir=artifact_dir)
        manifest: dict[str, Any] = {
            "status": "completed",
            "audit_passed": audit_passed,
            "corrected_information_gate": gate,
            "reporting_bug": (
                "the custom SDPA attention hook ignored is_causal, so the retained source "
                "row used full attention; training and full-attention trajectory rows were unaffected"
            ),
            "invalid_evaluation_artifact": FINAL_EVALUATION_ARTIFACT,
            "invalid_evaluation_artifact_id": evaluation_artifact.id,
            "final_adapter_artifact": FINAL_ADAPTER_ARTIFACT,
            "final_adapter_artifact_id": adapter_artifact.id,
            "adapter_artifact_metadata": adapter_metadata,
            "checks": checks,
            "source_tokenizer_contract": tokenizer_contract,
            "reloaded_adapter_contract": adapter_contract,
            "physical_training_batch_size": 94,
            "training_accumulation_steps": 1,
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
                run.summary[f"bico_gate_audit/{group}/{key}"] = value
        final_comparison = comparisons[comparisons["candidate"] == FINAL_READOUT].iloc[0]
        for key, value in final_comparison.items():
            run.summary[f"bico_gate_audit/final/{key}"] = value
        run.summary["bico_gate_audit/corrected_gate_passed"] = bool(gate["passed"])
        result = wandb.Artifact(AUDIT_ARTIFACT, type="evaluation")
        for path in (
            scores_path,
            summary_path,
            comparisons_path,
            checks_path,
            gate_path,
            manifest_path,
            figure_path.with_suffix(".svg"),
            figure_path.with_suffix(".png"),
            cost_path,
        ):
            result.add_file(str(path))
        logged = run.log_artifact(result, aliases=["corrected-gate", "step-1000"])
        logged.wait()
        run.finish(exit_code=0)
        del bundle, source
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException:
        run.finish(exit_code=1)
        raise
