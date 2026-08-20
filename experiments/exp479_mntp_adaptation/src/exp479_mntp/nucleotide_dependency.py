"""Leakage-free central-mask nucleotide-dependency maps for exp479."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from pyfaidx import Fasta

from exp479_mntp.config import NUCLEOTIDE_LENGTH
from exp479_mntp.modeling import model_logits
from exp479_mntp.publishing import assert_budget_reserve
from exp479_mntp.vep import (
    ArmSpec,
    LoadedArm,
    download_reference,
    load_arm,
    reverse_complement,
)


@dataclass(frozen=True)
class Locus:
    name: str
    chrom: str
    start: int
    end: int
    strand: str


LOCI = (
    Locus("LDLR", "19", 11_089_299, 11_089_425, "+"),
    Locus("TH", "11", 2_171_682, 2_171_868, "-"),
    Locus("GRIA4", "11", 105_609_444, 105_609_472, "+"),
    Locus("HBA1", "16", 176_699, 176_954, "+"),
    Locus("tRNA_Arg_TCT", "1", 159_141_610, 159_141_684, "-"),
)


def locus_window(genome: Fasta, locus: Locus) -> tuple[str, int]:
    """Center a 255-bp 0-based half-open context on a preregistered locus."""

    center = (locus.start + locus.end) // 2
    context_start = center - NUCLEOTIDE_LENGTH // 2
    context_end = context_start + NUCLEOTIDE_LENGTH
    if context_start < 0 or context_end > len(genome[locus.chrom]):
        raise ValueError(f"{locus.name} context is outside chromosome {locus.chrom}")
    sequence = str(genome[locus.chrom][context_start:context_end]).upper()
    if len(sequence) != NUCLEOTIDE_LENGTH:
        raise ValueError(f"{locus.name} produced a short reference context")
    return sequence, context_start


@torch.inference_mode()
def orientation_dependency(
    arm: LoadedArm,
    sequence: str,
    *,
    batch_size: int,
    attention_mode: str = "full",
) -> np.ndarray:
    """Compute one orientation's directed L-infinity categorical Jacobian.

    The baseline is the first row of the same model call as its substitutions,
    so BF16 kernel selection cannot masquerade as a nucleotide dependency.
    MNTP arms mask the readout; a causal arm uses its ordinary next-token
    readout, which cannot see the target token or any later nucleotide.
    """

    if batch_size <= 1:
        raise ValueError("dependency-map batch size must leave room for a baseline")
    if arm.mask_token_id is None and attention_mode != "causal":
        raise RuntimeError("full-attention dependency maps require a mask token")
    if len(sequence) != NUCLEOTIDE_LENGTH:
        raise ValueError("dependency-map sequence has the wrong context length")
    device = next(arm.model.parameters()).device
    encoded = arm.tokenizer(sequence, add_special_tokens=True, return_tensors="pt")
    wildtype = encoded["input_ids"][0].to(device)
    prefix = wildtype.shape[0] - NUCLEOTIDE_LENGTH
    if prefix != 1:
        raise ValueError(f"expected one BOS token, observed prefix length {prefix}")
    canonical = torch.tensor(arm.canonical_ids, dtype=torch.long, device=device)
    matrix = np.zeros((NUCLEOTIDE_LENGTH, NUCLEOTIDE_LENGTH), dtype=np.float32)

    for readout in range(NUCLEOTIDE_LENGTH):
        assert_budget_reserve()
        target_token = prefix + readout
        base = wildtype.clone()
        if arm.mask_token_id is not None:
            base[target_token] = arm.mask_token_id

        substitutions = base.view(1, -1).repeat(NUCLEOTIDE_LENGTH * 4, 1)
        for position in range(NUCLEOTIDE_LENGTH):
            if position == readout:
                continue
            rows = slice(position * 4, (position + 1) * 4)
            substitutions[rows, prefix + position] = canonical

        max_change = torch.zeros(NUCLEOTIDE_LENGTH, dtype=torch.float32, device=device)
        candidate_batch_size = batch_size - 1
        for start in range(0, len(substitutions), candidate_batch_size):
            stop = min(len(substitutions), start + candidate_batch_size)
            candidates = substitutions[start:stop]
            paired = torch.cat((base.unsqueeze(0), candidates), dim=0)
            paired_logits = model_logits(
                arm.model,
                input_ids=paired,
                attention_mode=attention_mode,
            )[:, target_token - 1, canonical]
            baseline_logits = paired_logits[0]
            candidate_logits = paired_logits[1:]
            delta = torch.log_softmax(candidate_logits.float(), dim=-1) - torch.log_softmax(
                baseline_logits.float(), dim=-1
            )
            collapsed = delta.abs().amax(dim=-1)
            positions = torch.arange(start, stop, device=device) // 4
            max_change.scatter_reduce_(0, positions, collapsed, reduce="amax", include_self=True)
        max_change[readout] = 0
        matrix[:, readout] = max_change.cpu().numpy()
    if not np.isfinite(matrix).all():
        raise RuntimeError("dependency map contains non-finite values")
    np.fill_diagonal(matrix, 0.0)
    return matrix


def mean_symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Apply the registered within-orientation mean symmetrization."""

    if matrix.shape != (NUCLEOTIDE_LENGTH, NUCLEOTIDE_LENGTH):
        raise ValueError(f"unexpected dependency shape {matrix.shape}")
    result = (matrix + matrix.T) / 2
    np.fill_diagonal(result, 0.0)
    return result


def off_diagonal_spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Spearman correlation over one copy of the off-diagonal position pairs."""

    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("dependency maps must be same-size square matrices")
    indices = np.triu_indices(first.shape[0], k=1)
    first_rank = pd.Series(first[indices]).rank(method="average")
    second_rank = pd.Series(second[indices]).rank(method="average")
    value = first_rank.corr(second_rank, method="pearson")
    if value is None or not np.isfinite(value):
        raise ValueError("off-diagonal Spearman correlation is undefined")
    return float(value)


def plot_comparison(
    single_orientation: np.ndarray,
    fwd_rc: np.ndarray,
    *,
    locus: Locus,
    correlation: float,
    output_path: Path,
) -> None:
    """Render matched-scale side-by-side heatmaps as an SVG research artifact."""

    maximum = float(max(single_orientation.max(), fwd_rc.max()))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    image = None
    for axis, matrix, title in zip(
        axes,
        (single_orientation, fwd_rc),
        ("Reference orientation", "FWD+RC average"),
        strict=True,
    ):
        image = axis.imshow(
            matrix,
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=maximum,
            interpolation="nearest",
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel("Readout position in 255-bp window")
        axis.set_ylabel("Substitution position in 255-bp window")
    assert image is not None
    figure.colorbar(image, ax=axes, label="L∞ change in A/C/G/T log probability")
    figure.suptitle(
        f"{locus.name} ({locus.chrom}:{locus.start}-{locus.end}, {locus.strand}); "
        f"off-diagonal Spearman ρ={correlation:.3f}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="svg")
    plt.close(figure)


def run_dependency_panel(
    *,
    artifact_dir: Path,
    output_dir: Path,
    hf_repo_id: str,
    batch_size: int,
) -> None:
    """Compute, plot, and publish the fixed #237 panel for transferred MNTP."""

    if not torch.cuda.is_available():
        raise RuntimeError("the full dependency panel requires the Lambda GH200")
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = download_reference(artifact_dir / "reference")
    arm = load_arm(ArmSpec("transferred_mntp", "mntp"), artifact_dir, hf_repo_id)
    arm.model.to("cuda").eval()
    summary: list[dict[str, object]] = []
    with Fasta(fasta_path, as_raw=True, rebuild=False) as genome:
        for locus in LOCI:
            sequence, context_start = locus_window(genome, locus)
            forward_directed = orientation_dependency(arm, sequence, batch_size=batch_size)
            reverse_directed = orientation_dependency(
                arm,
                reverse_complement(sequence),
                batch_size=batch_size,
            )[::-1, ::-1]
            single_orientation = mean_symmetrize(forward_directed)
            reverse_orientation = mean_symmetrize(reverse_directed)
            fwd_rc = (single_orientation + reverse_orientation) / 2
            correlation = off_diagonal_spearman(single_orientation, fwd_rc)
            np.savez_compressed(
                output_dir / f"{locus.name}.npz",
                forward_directed=forward_directed,
                reverse_directed_forward_coordinates=reverse_directed,
                single_orientation=single_orientation,
                fwd_rc=fwd_rc,
            )
            plot_comparison(
                single_orientation,
                fwd_rc,
                locus=locus,
                correlation=correlation,
                output_path=output_dir / f"{locus.name}.svg",
            )
            summary.append(
                {
                    "locus": locus.name,
                    "chrom": locus.chrom,
                    "start": locus.start,
                    "end": locus.end,
                    "strand": locus.strand,
                    "context_start": context_start,
                    "context_end": context_start + NUCLEOTIDE_LENGTH,
                    "off_diagonal_spearman": correlation,
                }
            )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    HfApi().upload_folder(
        folder_path=output_dir,
        path_in_repo="evaluation/nucleotide-dependency",
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message="Upload exp479 MNTP nucleotide-dependency panel",
    )
