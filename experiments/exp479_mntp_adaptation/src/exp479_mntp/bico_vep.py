"""Within-run odd/X VEP trajectory for one-pass BICO LoRA."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from exp479_mntp.bico_attention_diagnostic import (
    excluded_selected_key_mask,
    install_reflected_future_rope,
)
from exp479_mntp.config import NUCLEOTIDE_LENGTH
from exp479_mntp.modeling import ModelBundle, model_logits
from exp479_mntp.publishing import assert_budget_reserve
from exp479_mntp.two_pass_vep import PRIMARY_ENDPOINTS
from exp479_mntp.vep import (
    DATASETS,
    DatasetSpec,
    _protocol_scores,
    attach_reference_windows,
    download_reference,
    load_variant_frame,
)
from exp479_mntp.vep_metrics import GLOBAL, MACRO, matched_metrics, sge_metrics

BICO_VEP_STEPS = tuple(range(0, 1_001, 100))
SOURCE_CLM_ENDPOINTS = {
    "mendelian_traits": 0.395523,
    "complex_traits": 0.134083,
    "sge": 0.357623,
}


def prepare_bico_vep_frames(artifact_dir: Path) -> dict[str, pd.DataFrame]:
    """Load only public odd-autosome/X labels and attach pinned GRCh38 windows."""

    reference = download_reference(artifact_dir / "reference")
    frames = {
        spec.name: attach_reference_windows(load_variant_frame(spec), reference)
        for spec in DATASETS
    }
    for dataset_name, frame in frames.items():
        if (frame["ref"] == frame["alt"]).any():
            raise ValueError(f"{dataset_name} contains an equal reference/alternate allele")
        if len(frame) == 0:
            raise ValueError(f"{dataset_name} contains no development variants")
    return frames


def _autocast(device: torch.device) -> Any:
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


@torch.inference_mode()
def score_bico_vep(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    *,
    batch_size: int,
) -> np.ndarray:
    """Score reference-orientation alt-minus-ref central conditionals with BICO."""

    if batch_size <= 0:
        raise ValueError("BICO VEP batch size must be positive")
    if bundle.mask_token_id is None:
        raise RuntimeError("BICO VEP requires the existing PAD mask token")
    variant_index = NUCLEOTIDE_LENGTH // 2
    target_input_position = 1 + variant_index
    target_output_position = target_input_position - 1
    if target_output_position != variant_index:
        raise RuntimeError("BICO VEP input/output shift changed")
    model = bundle.model
    device = next(model.parameters()).device
    model.to(device=device)
    install_reflected_future_rope(model)
    canonical = list(bundle.canonical_token_ids)
    base_to_index = {base: index for index, base in enumerate("ACGT")}
    base_to_token = dict(zip("ACGT", canonical, strict=True))
    scores = np.empty(len(frame), dtype=np.float32)

    for start in range(0, len(frame), batch_size):
        assert_budget_reserve()
        stop = min(start + batch_size, len(frame))
        cell = frame.iloc[start:stop]
        encoded = bundle.tokenizer(
            cell["sequence"].tolist(),
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        token_mask = encoded["attention_mask"].to(device)
        if input_ids.shape != (stop - start, NUCLEOTIDE_LENGTH + 1):
            raise RuntimeError(f"unexpected BICO VEP token shape {tuple(input_ids.shape)}")
        if not torch.all(token_mask == 1):
            raise RuntimeError("BICO VEP requires unpadded 256-token inputs")
        if not torch.all(input_ids[:, 0] == bundle.tokenizer.bos_token_id):
            raise RuntimeError("BICO VEP input lacks the single BOS token")
        expected_reference_ids = torch.tensor(
            [base_to_token[value] for value in cell["ref"]],
            device=device,
            dtype=torch.long,
        )
        if not torch.equal(input_ids[:, target_input_position], expected_reference_ids):
            raise RuntimeError("BICO VEP central token differs from the reference allele")
        input_ids[:, target_input_position] = bundle.mask_token_id
        output_positions = torch.full(
            (stop - start,),
            target_output_position,
            device=device,
            dtype=torch.long,
        )
        attention_mask = excluded_selected_key_mask(
            token_mask,
            output_positions,
            dtype=torch.bfloat16,
        )
        with _autocast(device):
            logits = model_logits(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                attention_mode="full",
            )[:, target_output_position, canonical]
        log_probabilities = logits.float().log_softmax(dim=-1)
        rows = torch.arange(stop - start, device=device)
        ref_indices = torch.tensor(
            [base_to_index[value] for value in cell["ref"]],
            device=device,
            dtype=torch.long,
        )
        alt_indices = torch.tensor(
            [base_to_index[value] for value in cell["alt"]],
            device=device,
            dtype=torch.long,
        )
        values = log_probabilities[rows, alt_indices] - log_probabilities[rows, ref_indices]
        scores[start:stop] = values.cpu().numpy()

    if not np.isfinite(scores).all():
        raise RuntimeError("BICO VEP produced non-finite scores")
    return scores


def _select_primary_endpoint(
    dataset_spec: DatasetSpec,
    metrics: pd.DataFrame,
) -> pd.Series:
    """Select the pre-registered development aggregation for one VEP dataset."""

    if dataset_spec.evaluation == "matched":
        subset = MACRO if dataset_spec.name == "mendelian_traits" else GLOBAL
        selected = metrics[metrics["subset"] == subset]
    elif dataset_spec.evaluation == "sge":
        selected = metrics[
            (metrics["subset"] == MACRO)
            & (metrics["accession"] == MACRO)
            & (metrics["gene"] == MACRO)
        ]
    else:
        raise ValueError(f"unknown VEP evaluation protocol {dataset_spec.evaluation}")
    if len(selected) != 1:
        raise RuntimeError(f"registered VEP endpoint is ambiguous for {dataset_spec.name}")
    return selected.iloc[0]


def bico_vep_endpoint(
    dataset_spec: DatasetSpec,
    frame: pd.DataFrame,
    llr: np.ndarray,
    *,
    n_bootstrap: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply the registered protocol and return one primary development endpoint."""

    if n_bootstrap <= 1:
        raise ValueError("BICO VEP endpoint needs at least two bootstrap replicates")
    protocol_scores = _protocol_scores(llr, dataset_spec.protocol)
    score_name = "bico_fwd"
    scores = pd.DataFrame({score_name: protocol_scores})
    if dataset_spec.evaluation == "matched":
        metrics = matched_metrics(frame, scores, n_bootstrap=n_bootstrap, seed=0)
    else:
        metrics = sge_metrics(frame, scores, n_bootstrap=n_bootstrap, seed=0)
    row = _select_primary_endpoint(dataset_spec, metrics)
    return protocol_scores, {
        "dataset": dataset_spec.name,
        "endpoint": PRIMARY_ENDPOINTS[dataset_spec.name],
        "auprc": float(row["value"]),
        "bootstrap_se": float(row["se"]),
        "n_rows": len(frame),
    }


def plot_bico_vep_trajectory(endpoints: pd.DataFrame, output_path: Path) -> None:
    """Plot within-run one-pass VEP AUPRC with source CLM reference lines."""

    figure, axis = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    colors = {
        "mendelian_traits": "#4C78A8",
        "complex_traits": "#F58518",
        "sge": "#54A24B",
    }
    labels = {
        "mendelian_traits": "Mendelian macro",
        "complex_traits": "Complex global",
        "sge": "SGE macro",
    }
    for dataset_name, cell in endpoints.groupby("dataset", sort=False):
        ordered = cell.sort_values("optimizer_step")
        color = colors[dataset_name]
        axis.plot(
            ordered["optimizer_step"],
            ordered["auprc"],
            marker="o",
            color=color,
            label=f"{labels[dataset_name]} BICO FWD",
        )
        axis.axhline(
            SOURCE_CLM_ENDPOINTS[dataset_name],
            color=color,
            linestyle="--",
            alpha=0.55,
            label=f"{labels[dataset_name]} source CLM FWD+RC reference",
        )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Development AUPRC")
    axis.set_title("Within-run one-pass BICO LoRA VEP trajectory (odd autosomes/X)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)
