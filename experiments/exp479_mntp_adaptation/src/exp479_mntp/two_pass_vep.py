"""Evaluate frozen two-causal-pass variant scores against the source CLM."""

from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from torch.nn.attention import SDPBackend, sdpa_kernel

from exp479_mntp.config import (
    BUDGET_USD,
    EXPERIMENT_TAGS,
    MODEL_ID,
    MODEL_REVISION,
    NUCLEOTIDE_LENGTH,
    WANDB_PROJECT,
)
from exp479_mntp.gated_lora_mntp import configure_gated_numerics
from exp479_mntp.modeling import ModelBundle, load_model_bundle, model_logits
from exp479_mntp.publishing import assert_budget_reserve, write_cost_estimate
from exp479_mntp.two_pass_information_gate import (
    RC_ALLELE_PERMUTATION,
    reverse_complement_token_ids,
)
from exp479_mntp.vep import (
    DATASETS,
    LoadedArm,
    _protocol_scores,
    attach_reference_windows,
    download_reference,
    load_variant_frame,
    score_strand,
)
from exp479_mntp.vep_metrics import (
    GLOBAL,
    MACRO,
    matched_metrics,
    paired_ap_delta,
    sge_metrics,
)

TWO_PASS_VEP_RUN_NAME = "dna-exp479-frozen-two-pass-vep"
TWO_PASS_VEP_ARTIFACT = TWO_PASS_VEP_RUN_NAME
CALIBRATION_ARTIFACT_ID = "QXJ0aWZhY3Q6MzM3NDU5NzY1Mg=="
CALIBRATION_ARTIFACT_DIGEST = "7d0f9f70f22e39e7e7a7c3e7e8454aeb"
CALIBRATION_TARGETS = 640
FORWARD_ALPHA = 0.615
SYMMETRIC_ALPHA = 0.408
PRIOR_ACGT = np.array(
    [
        0.2702492211838006,
        0.21105919003115264,
        0.2266355140186916,
        0.29205607476635514,
    ],
    dtype=np.float64,
)
PRIMARY_ENDPOINTS = {
    "mendelian_traits": "mendelian_consequence_macro",
    "complex_traits": "complex_global",
    "sge": "accession_consequence_macro",
}
TWO_PASS_VEP_MAX_INSTANCE_HOURS = 2.0


def _normalize_log_scores(values: np.ndarray) -> np.ndarray:
    return values - np.logaddexp.reduce(values, axis=-1, keepdims=True)


def two_pass_log_probabilities(
    left_log_probs: np.ndarray,
    right_log_probs: np.ndarray,
    *,
    log_prior: np.ndarray,
    forward_alpha: float = FORWARD_ALPHA,
    symmetric_alpha: float = SYMMETRIC_ALPHA,
) -> dict[str, np.ndarray]:
    """Construct registered directional and strand-symmetric distributions."""

    if left_log_probs.shape != right_log_probs.shape or left_log_probs.shape[-1] != 4:
        raise ValueError("directional log probabilities must share shape [..., 4]")
    if log_prior.shape != (4,):
        raise ValueError("log prior must have shape [4]")
    if not 0 <= forward_alpha <= 1 or not 0 <= symmetric_alpha <= 1:
        raise ValueError("two-pass alpha values must lie in [0, 1]")
    directional_mean = _normalize_log_scores((left_log_probs + right_log_probs) / 2)
    forward = _normalize_log_scores(left_log_probs + forward_alpha * (right_log_probs - log_prior))
    symmetric_forward = _normalize_log_scores(
        left_log_probs + symmetric_alpha * (right_log_probs - log_prior)
    )
    symmetric_reverse = _normalize_log_scores(
        right_log_probs + symmetric_alpha * (left_log_probs - log_prior)
    )
    symmetric = _normalize_log_scores((symmetric_forward + symmetric_reverse) / 2)
    return {
        "source_conditional_left": left_log_probs,
        "source_conditional_right": right_log_probs,
        "source_conditional_avg": directional_mean,
        "source_two_pass_fwd": forward,
        "source_two_pass_symmetric": symmetric,
    }


def _allele_indices(values: list[str]) -> np.ndarray:
    lookup = {base: index for index, base in enumerate("ACGT")}
    try:
        return np.array([lookup[value] for value in values], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"non-canonical allele {error.args[0]}") from error


def log_probability_ratios(
    log_probabilities: np.ndarray,
    *,
    reference_indices: np.ndarray,
    alternate_indices: np.ndarray,
) -> np.ndarray:
    """Return alternate-minus-reference log-probability ratios."""

    if log_probabilities.ndim != 2 or log_probabilities.shape[1] != 4:
        raise ValueError("nucleotide log probabilities must have shape [rows, 4]")
    if reference_indices.shape != (len(log_probabilities),) or alternate_indices.shape != (
        len(log_probabilities),
    ):
        raise ValueError("allele indices must contain one value per score row")
    rows = np.arange(len(log_probabilities))
    return log_probabilities[rows, alternate_indices] - log_probabilities[rows, reference_indices]


@torch.inference_mode()
def score_conditional_two_pass(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    *,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score one masked central nucleotide from both native causal directions."""

    if batch_size <= 0:
        raise ValueError("VEP batch size must be positive")
    if bundle.mask_token_id is None:
        raise RuntimeError("two-pass VEP requires the source UNK token")
    variant_index = NUCLEOTIDE_LENGTH // 2
    target_input_position = 1 + variant_index
    target_output_position = target_input_position - 1
    reverse_output_position = NUCLEOTIDE_LENGTH - 1 - target_output_position
    if target_output_position != variant_index or reverse_output_position != variant_index:
        raise RuntimeError("central two-pass output mapping changed")

    model = bundle.model
    device = next(model.parameters()).device
    canonical = torch.tensor(bundle.canonical_token_ids, device=device, dtype=torch.long)
    rc_permutation = torch.tensor(RC_ALLELE_PERMUTATION, device=device, dtype=torch.long)
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    round_trip_checks = 0
    for start in range(0, len(frame), batch_size):
        assert_budget_reserve()
        stop = min(len(frame), start + batch_size)
        encoded = bundle.tokenizer(
            frame["sequence"].iloc[start:stop].tolist(),
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if input_ids.shape != (stop - start, NUCLEOTIDE_LENGTH + 1):
            raise RuntimeError(f"unexpected two-pass token shape {tuple(input_ids.shape)}")
        if not torch.all(attention_mask == 1):
            raise RuntimeError("two-pass VEP requires unpadded 256-token inputs")
        if not torch.all(input_ids[:, 0] == bundle.tokenizer.bos_token_id):
            raise RuntimeError("two-pass VEP input lacks the single BOS token")
        reference_indices = torch.tensor(
            _allele_indices(frame["ref"].iloc[start:stop].tolist()),
            device=device,
            dtype=torch.long,
        )
        rows = torch.arange(stop - start, device=device)
        reference_ids = canonical.index_select(0, reference_indices)
        if not torch.equal(input_ids[rows, target_input_position], reference_ids):
            raise RuntimeError(
                "tokenized central base differs from the registered reference allele"
            )
        masked = input_ids.clone()
        masked[:, target_input_position] = bundle.mask_token_id
        if not torch.all(masked[:, target_output_position + 1] == bundle.mask_token_id):
            raise RuntimeError("forward VEP mask is not at output position plus one")

        with sdpa_kernel([SDPBackend.MATH]):
            left_logits = model_logits(
                model,
                input_ids=masked,
                attention_mask=attention_mask,
                attention_mode="causal",
            )[:, target_output_position]
            reversed_ids = reverse_complement_token_ids(
                masked,
                canonical_token_ids=bundle.canonical_token_ids,
                bos_token_id=int(bundle.tokenizer.bos_token_id),
            )
            if not torch.all(reversed_ids[:, reverse_output_position + 1] == bundle.mask_token_id):
                raise RuntimeError("reverse VEP mask is not at mapped output position plus one")
            round_trip = reverse_complement_token_ids(
                reversed_ids,
                canonical_token_ids=bundle.canonical_token_ids,
                bos_token_id=int(bundle.tokenizer.bos_token_id),
            )
            if not torch.equal(round_trip, masked):
                raise RuntimeError("VEP reverse-complement token transform is not an involution")
            round_trip_checks += len(masked)
            right_logits = model_logits(
                model,
                input_ids=reversed_ids,
                attention_mask=attention_mask,
                attention_mode="causal",
            )[:, reverse_output_position]
        left = left_logits.index_select(1, canonical).log_softmax(dim=-1)
        right = (
            right_logits.index_select(1, canonical)
            .index_select(1, rc_permutation)
            .log_softmax(dim=-1)
        )
        left_parts.append(left.float().cpu().numpy().astype(np.float64))
        right_parts.append(right.float().cpu().numpy().astype(np.float64))

    left_log_probs = np.concatenate(left_parts)
    right_log_probs = np.concatenate(right_parts)
    distributions = two_pass_log_probabilities(
        left_log_probs,
        right_log_probs,
        log_prior=np.log(PRIOR_ACGT),
    )
    reference_indices = _allele_indices(frame["ref"].tolist())
    alternate_indices = _allele_indices(frame["alt"].tolist())
    result = pd.DataFrame(
        {
            name: log_probability_ratios(
                values,
                reference_indices=reference_indices,
                alternate_indices=alternate_indices,
            ).astype(np.float32)
            for name, values in distributions.items()
        }
    )
    if not np.isfinite(result.to_numpy()).all():
        raise RuntimeError("non-finite frozen two-pass VEP scores")
    controls = {
        "variant_index_zero_based": variant_index,
        "target_input_position": target_input_position,
        "target_output_position": target_output_position,
        "reverse_output_position": reverse_output_position,
        "round_trip_checks": round_trip_checks,
        "reverse_allele_permutation": list(RC_ALLELE_PERMUTATION),
        "bos_token_id": int(bundle.tokenizer.bos_token_id),
        "unk_token_id": int(bundle.mask_token_id),
        "pad_token_id": int(bundle.tokenizer.pad_token_id),
    }
    return result, controls


def _normal_macro(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("paired macro requires at least one qualifying child")
    count = len(rows)
    delta = float(sum(float(row["delta"]) for row in rows) / count)
    se = float(math.sqrt(sum(float(row["se"]) ** 2 for row in rows)) / count)
    return {
        "delta": delta,
        "se": se,
        "ci_low": delta - 1.96 * se,
        "ci_high": delta + 1.96 * se,
        "n_groups": count,
        "n_rows": sum(int(row["n_rows"]) for row in rows),
    }


def primary_paired_comparison(
    dataset_name: str,
    variants: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    n_bootstrap: int,
) -> dict[str, object]:
    """Compare the registered primary AUPRC endpoint on paired rows."""

    if dataset_name == "complex_traits":
        result = paired_ap_delta(
            variants["label"],
            scores[candidate],
            scores[baseline],
            variants["match_group"],
            n_bootstrap=n_bootstrap,
            seed=0,
        )
    elif dataset_name == "mendelian_traits":
        subset_rows: list[dict[str, float | int]] = []
        for _, cell in variants.groupby("subset", sort=False):
            if cell["match_group"].nunique() < 30:
                continue
            subset_rows.append(
                paired_ap_delta(
                    cell["label"],
                    scores.loc[cell.index, candidate],
                    scores.loc[cell.index, baseline],
                    cell["match_group"],
                    n_bootstrap=n_bootstrap,
                    seed=0,
                )
            )
        result = _normal_macro(subset_rows)
    elif dataset_name == "sge":
        accession_rows: list[dict[str, float | int]] = []
        subsets = sorted(str(value) for value in variants["subset"].unique())
        for _, accession in variants.groupby("mavedb_urn", sort=False):
            consequence_rows: list[dict[str, float | int]] = []
            for subset in subsets:
                cell = accession[accession["subset"] == subset]
                n_positive = int(cell["label"].astype(bool).sum())
                if n_positive < 30 or len(cell) - n_positive < 30:
                    continue
                consequence_rows.append(
                    paired_ap_delta(
                        cell["label"],
                        scores.loc[cell.index, candidate],
                        scores.loc[cell.index, baseline],
                        np.arange(len(cell)),
                        n_bootstrap=n_bootstrap,
                        seed=0,
                    )
                )
            if consequence_rows:
                accession_rows.append(_normal_macro(consequence_rows))
        result = _normal_macro(accession_rows)
    else:
        raise ValueError(f"unknown registered VEP dataset {dataset_name}")
    return {
        "dataset": dataset_name,
        "endpoint": PRIMARY_ENDPOINTS[dataset_name],
        "candidate": candidate,
        "baseline": baseline,
        **result,
    }


def _primary_endpoint_table(dataset_name: str, metrics: pd.DataFrame) -> pd.DataFrame:
    if dataset_name == "mendelian_traits":
        selected = metrics[metrics["subset"] == MACRO]
    elif dataset_name == "complex_traits":
        selected = metrics[metrics["subset"] == GLOBAL]
    elif dataset_name == "sge":
        selected = metrics[
            (metrics["subset"] == MACRO)
            & (metrics["accession"] == MACRO)
            & (metrics["gene"] == MACRO)
        ]
    else:
        raise ValueError(f"unknown registered VEP dataset {dataset_name}")
    return selected.assign(dataset=dataset_name, endpoint=PRIMARY_ENDPOINTS[dataset_name])


def _plot_primary(endpoints: pd.DataFrame, comparisons: pd.DataFrame, output_path: Path) -> None:
    score_order = [
        "source_clm_avg",
        "source_conditional_avg",
        "source_two_pass_fwd",
        "source_two_pass_symmetric",
    ]
    labels = ["Source CLM FWD+RC", "Masked L/R mean", "Two-pass fwd", "Two-pass symmetric"]
    datasets = [spec.name for spec in DATASETS]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#72B7B2", "#59A14F"]
    for axis, dataset_name in zip(axes, datasets, strict=True):
        cell = endpoints[endpoints["dataset"] == dataset_name].set_index("score_type")
        values = [float(cell.loc[name, "value"]) for name in score_order]
        errors = [float(cell.loc[name, "se"]) for name in score_order]
        x = np.arange(len(score_order))
        axis.bar(x, values, yerr=errors, capsize=3, color=colors)
        axis.set_xticks(x, labels, rotation=28, ha="right")
        axis.set_ylabel("AUPRC")
        axis.set_title(dataset_name.replace("_", " ").title())
        axis.grid(axis="y", alpha=0.25)
    gate = comparisons[comparisons["candidate"] == "source_two_pass_symmetric"]
    figure.suptitle(
        "Frozen two-pass VEP versus source CLM; paired primary deltas "
        + ", ".join(f"{row.dataset}: {row.delta:+.3f}" for row in gate.itertuples())
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_two_pass_vep(
    *,
    artifact_dir: Path,
    output_dir: Path,
    batch_size: int,
    n_bootstrap: int,
) -> None:
    """Run the frozen, calibrated two-pass VEP comparison on development labels."""

    numeric_controls = configure_gated_numerics()
    if not torch.cuda.is_available():
        raise RuntimeError("two-pass VEP requires one CUDA GPU")
    if batch_size <= 0 or n_bootstrap <= 0:
        raise ValueError("batch size and bootstrap count must be positive")
    prior_cost = float(os.getenv("EXP479_PRIOR_COST_USD", "0"))
    price = float(os.getenv("EXP479_INSTANCE_PRICE_PER_HOUR_USD", "2.29"))
    if prior_cost + TWO_PASS_VEP_MAX_INSTANCE_HOURS * price >= BUDGET_USD:
        raise RuntimeError("two-pass VEP projection reaches the issue budget cap")
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    budget_path = output_dir / "prelaunch-budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "prior_cost_usd": prior_cost,
                "maximum_instance_hours": TWO_PASS_VEP_MAX_INSTANCE_HOURS,
                "price_per_hour_usd": price,
                "projected_total_usd": prior_cost + TWO_PASS_VEP_MAX_INSTANCE_HOURS * price,
                "budget_cap_usd": BUDGET_USD,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    run = wandb.init(
        project=WANDB_PROJECT,
        group="dna-exp479-causal-information-prerequisite",
        name=TWO_PASS_VEP_RUN_NAME,
        tags=[*EXPERIMENT_TAGS, "frozen-source", "two-causal-pass", "vep"],
        config={
            "base_model": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "calibration_artifact_id": CALIBRATION_ARTIFACT_ID,
            "calibration_artifact_digest": CALIBRATION_ARTIFACT_DIGEST,
            "calibration_targets": CALIBRATION_TARGETS,
            "forward_alpha": FORWARD_ALPHA,
            "symmetric_alpha": SYMMETRIC_ALPHA,
            "prior_acgt": PRIOR_ACGT.tolist(),
            "batch_size": batch_size,
            "n_bootstrap": n_bootstrap,
            "numeric_controls": numeric_controls,
        },
    )
    if run is None:
        raise RuntimeError("W&B did not create the two-pass VEP run")

    try:
        reference = download_reference(artifact_dir / "reference")
        frames = {
            spec.name: attach_reference_windows(load_variant_frame(spec), reference)
            for spec in DATASETS
        }
        loaded = load_model_bundle(
            initialization="transferred",
            add_mask=False,
            attention_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        unk_token_id = loaded.tokenizer.unk_token_id
        if unk_token_id is None:
            raise RuntimeError("source tokenizer lacks UNK for two-pass VEP")
        bundle = ModelBundle(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            canonical_token_ids=loaded.canonical_token_ids,
            mask_token_id=int(unk_token_id),
            input_output_tied=loaded.input_output_tied,
        )
        bundle.model.to(device="cuda", dtype=torch.bfloat16).eval()
        arm = LoadedArm(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            canonical_ids=bundle.canonical_token_ids,
            mask_token_id=None,
        )

        endpoint_parts: list[pd.DataFrame] = []
        comparison_rows: list[dict[str, object]] = []
        runtime_rows: list[dict[str, object]] = []
        control_rows: list[dict[str, object]] = []
        output_files: list[Path] = []
        for spec in DATASETS:
            frame = frames[spec.name]
            scores = pd.DataFrame(index=frame.index)
            source_strands: dict[str, np.ndarray] = {}
            for strand in ("fwd", "rc"):
                torch.cuda.reset_peak_memory_stats()
                strand_started = time.perf_counter()
                source_strands[strand] = score_strand(
                    arm,
                    frame,
                    objective="clm",
                    strand=strand,
                    batch_size=batch_size,
                )
                elapsed = time.perf_counter() - strand_started
                runtime_rows.append(
                    {
                        "dataset": spec.name,
                        "readout": f"source_clm_{strand}",
                        "rows": len(frame),
                        "seconds": elapsed,
                        "variants_per_second": len(frame) / elapsed,
                        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    }
                )
                scores[f"source_clm_{strand}"] = _protocol_scores(
                    source_strands[strand], spec.protocol
                )
            scores["source_clm_avg"] = _protocol_scores(
                (source_strands["fwd"] + source_strands["rc"]) / 2,
                spec.protocol,
            )

            torch.cuda.reset_peak_memory_stats()
            conditional_started = time.perf_counter()
            conditional_llr, controls = score_conditional_two_pass(
                bundle,
                frame,
                batch_size=batch_size,
            )
            conditional_elapsed = time.perf_counter() - conditional_started
            runtime_rows.append(
                {
                    "dataset": spec.name,
                    "readout": "two_causal_passes",
                    "rows": len(frame),
                    "seconds": conditional_elapsed,
                    "variants_per_second": len(frame) / conditional_elapsed,
                    "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                }
            )
            control_rows.append({"dataset": spec.name, **controls})
            for name in conditional_llr:
                scores[name] = _protocol_scores(conditional_llr[name].to_numpy(), spec.protocol)
            if not np.isfinite(scores.to_numpy()).all():
                raise RuntimeError(f"non-finite published scores for {spec.name}")

            if spec.evaluation == "matched":
                metrics = matched_metrics(
                    frame,
                    scores,
                    n_bootstrap=n_bootstrap,
                    seed=0,
                )
            else:
                metrics = sge_metrics(
                    frame,
                    scores,
                    n_bootstrap=n_bootstrap,
                    seed=0,
                )
            endpoint_parts.append(_primary_endpoint_table(spec.name, metrics))
            for candidate in (
                "source_conditional_avg",
                "source_two_pass_fwd",
                "source_two_pass_symmetric",
            ):
                comparison_rows.append(
                    primary_paired_comparison(
                        spec.name,
                        frame,
                        scores,
                        candidate=candidate,
                        baseline="source_clm_avg",
                        n_bootstrap=n_bootstrap,
                    )
                )

            public_columns = [column for column in frame.columns if column != "sequence"]
            scores_path = output_dir / f"{spec.name}.scores.parquet"
            metrics_path = output_dir / f"{spec.name}.metrics.parquet"
            pd.concat([frame[public_columns], scores], axis=1).to_parquet(
                scores_path,
                index=False,
            )
            metrics.to_parquet(metrics_path, index=False)
            output_files.extend((scores_path, metrics_path))

        endpoints = pd.concat(endpoint_parts, ignore_index=True)
        comparisons = pd.DataFrame(comparison_rows)
        primary = comparisons[comparisons["candidate"] == "source_two_pass_symmetric"].reset_index(
            drop=True
        )
        point_noninferiority = bool((primary["delta"] >= 0).all())
        confidence_noninferiority = bool((primary["ci_low"] >= 0).all())
        strict_improvement = bool((primary["ci_low"] > 0).any())
        gate = {
            "criterion": (
                "symmetric frozen two-pass AUPRC is confidence-supported non-inferior "
                "to source CLM FWD+RC on all three registered primary endpoints and "
                "strictly improves at least one"
            ),
            "point_noninferiority_all_three": point_noninferiority,
            "confidence_noninferiority_all_three": confidence_noninferiority,
            "confidence_strict_improvement_any": strict_improvement,
            "passed": bool(
                point_noninferiority and confidence_noninferiority and strict_improvement
            ),
        }
        endpoints_path = output_dir / "primary-endpoints.csv"
        comparisons_path = output_dir / "primary-paired-comparisons.csv"
        controls_path = output_dir / "coordinate-token-controls.csv"
        runtime_path = output_dir / "runtime.csv"
        gate_path = output_dir / "vep-gate.json"
        figure_path = output_dir / "figures" / "two-pass-vep-primary"
        endpoints.to_csv(endpoints_path, index=False)
        comparisons.to_csv(comparisons_path, index=False)
        pd.DataFrame(control_rows).to_csv(controls_path, index=False)
        pd.DataFrame(runtime_rows).to_csv(runtime_path, index=False)
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        _plot_primary(endpoints, comparisons, figure_path)
        elapsed_path = output_dir / "elapsed.json"
        elapsed_path.write_text(
            json.dumps({"elapsed_seconds": time.time() - started}, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_path = output_dir / "manifest.json"
        manifest = {
            "status": "completed",
            "source_model": f"{MODEL_ID}@{MODEL_REVISION}",
            "model_updates": 0,
            "calibration_artifact_id": CALIBRATION_ARTIFACT_ID,
            "calibration_artifact_digest": CALIBRATION_ARTIFACT_DIGEST,
            "calibration_targets": CALIBRATION_TARGETS,
            "forward_alpha": FORWARD_ALPHA,
            "symmetric_alpha": SYMMETRIC_ALPHA,
            "prior_acgt": PRIOR_ACGT.tolist(),
            "datasets": {spec.name: f"{spec.repo_id}@{spec.revision}" for spec in DATASETS},
            "split": "train",
            "allowed_chromosomes": sorted(
                {str(value) for frame in frames.values() for value in frame["chrom"].unique()}
            ),
            "primary_endpoints": PRIMARY_ENDPOINTS,
            "numeric_controls": numeric_controls,
            "gate": gate,
            "nucleotide_dependency": "not performed",
            "knowledge_base_update": "not performed",
            "hugging_face_upload": "not performed",
            "checkpoint_deletion": "not performed",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        cost_path = write_cost_estimate(artifact_dir=artifact_dir)
        output_files.extend(
            (
                endpoints_path,
                comparisons_path,
                controls_path,
                runtime_path,
                gate_path,
                elapsed_path,
                manifest_path,
                budget_path,
                cost_path,
                figure_path.with_suffix(".svg"),
            )
        )

        run.log(
            {
                "two_pass_vep/primary_endpoints": wandb.Table(dataframe=endpoints),
                "two_pass_vep/paired_comparisons": wandb.Table(dataframe=comparisons),
                "two_pass_vep/figure": wandb.Image(str(figure_path.with_suffix(".png"))),
            }
        )
        run.summary["two_pass_vep/passed"] = bool(gate["passed"])
        result = wandb.Artifact(TWO_PASS_VEP_ARTIFACT, type="evaluation")
        for path in output_files:
            result.add_file(str(path))
        logged = run.log_artifact(result, aliases=["vep-gate"])
        logged.wait()
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise
    finally:
        if "bundle" in locals():
            bundle.model.to(device="cpu")
        gc.collect()
        torch.cuda.empty_cache()
