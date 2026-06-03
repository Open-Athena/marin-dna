"""Nucleotide dependency maps — pipeline glue + plotting (issue #237).

Loads a checkpoint, extracts a ``window_size`` window centred on a locus,
computes the FWD+RC-stitched dependency map via
``marin_dna.model.interpretation.nucleotide_dependency_map``, and renders it as
a heatmap. The categorical-Jacobian math lives in the model module; this is the
thin orchestration layer the Snakemake rule calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.data.genome import Genome
from marin_dna.model.interpretation import nucleotide_dependency_map


def compute_dependency_map(
    checkpoint_path: str | Path,
    genome_path: str | Path,
    chrom: str,
    start: int,
    end: int,
    strand: Literal["+", "-"],
    window_size: int,
    *,
    combine: Literal["mean", "max"] = "mean",
    rc: bool = True,
    norm_ord: float = np.inf,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Compute the nucleotide dependency map for a genomic locus.

    A ``window_size`` window is centred on the locus midpoint and fed to the
    model for context; the returned map is restricted to the locus
    ``[start, end)``. Coordinates are 0-based half-open.

    Args:
        checkpoint_path: HF model directory (``AutoModelForCausalLM``).
        genome_path: Reference FASTA (local path or ``s3://`` URI).
        chrom, start, end, strand: Locus (0-based half-open). ``strand`` only
            sets display orientation — the window is always extracted on ``+``
            and reverse-complemented internally for the RC pass.
        window_size: Model context window (bp); must be ``>= end - start``.
        combine: Symmetrization of the stitched FWD/RC map (``mean`` | ``max``).
        rc: Stitch forward + reverse-complement (the autoregressive fix).
        norm_ord: Vector-norm order collapsing each ``4x4`` block.
        batch_size: Sequences per forward pass.

    Returns:
        Symmetric ``[n, n]`` DataFrame (``n = end - start``) indexed and columned
        by genomic position. For ``strand == "-"`` the axes are reversed so the
        map reads 5'->3' along the locus strand.
    """
    assert end > start, f"locus end {end} must exceed start {start}"
    assert end - start <= window_size, (
        f"locus span {end - start} bp exceeds model window_size {window_size}; "
        f"pick a smaller locus or a longer-context model"
    )

    # Duck-typed model/tokenizer throughout interpretation (see
    # marin_dna.model.interpretation); annotate Any so HF stub overloads on
    # `.to(device)` don't fight mypy.
    tokenizer: Any = AutoTokenizer.from_pretrained(checkpoint_path)
    model: Any = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, trust_remote_code=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    genome = Genome(genome_path)
    center = (start + end) // 2
    context_start = center - window_size // 2
    context_end = context_start + window_size
    seq = genome(chrom, context_start, context_end, strand="+").upper()
    assert len(seq) == window_size, (
        f"extracted {len(seq)} bp, expected window_size={window_size}"
    )

    M = nucleotide_dependency_map(
        model,
        tokenizer,
        seq,
        rc=rc,
        combine=combine,
        norm_ord=norm_ord,
        batch_size=batch_size,
    )

    start_idx = start - context_start
    end_idx = end - context_start
    assert 0 <= start_idx < end_idx <= window_size, (
        f"locus window slice [{start_idx}:{end_idx}] out of bounds for "
        f"window_size={window_size}"
    )
    sub = M[start_idx:end_idx, start_idx:end_idx]
    coords = list(range(start, end))
    df = pd.DataFrame(sub, index=coords, columns=coords)
    if strand == "-":
        df = df.iloc[::-1, ::-1]
    return df


def plot_dependency_map(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    chrom: str | None = None,
    title: str | None = None,
    tick_freq: int | None = None,
    dpi: int = 150,
) -> None:
    """Render a dependency map as a ``coolwarm`` heatmap (SVG, plus a PNG
    sibling for local visual sanity per repo convention).

    Mirrors GPN-Star's nuc-dep plot: square aspect, robust color scaling,
    rasterized cells, sparse genomic-coordinate ticks.

    ``rasterized=True`` is essential: a dependency map is a dense ``L×L``
    QuadMesh (e.g. 255×255 = 65k cells), which as pure SVG vector cells blows
    up to ~12 MB. Rasterizing embeds the cells as a *single* PNG inside the
    SVG (text/axes stay vector), so file size is governed by ``dpi`` — ~210 KB
    at 150, ~130 KB at 72 for a 255 bp map. Lower ``dpi`` for smaller files.
    """
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text, not paths
    import matplotlib.pyplot as plt
    import seaborn as sns

    coords = np.asarray(df.columns, dtype=int)
    span = int(coords.max() - coords.min())
    if tick_freq is None:
        tick_freq = 50 if span > 100 else 10

    plt.figure(figsize=(4, 4))
    g = sns.heatmap(
        df,
        cmap="coolwarm",
        square=True,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        robust=True,
        rasterized=True,
    )
    tick_idx = np.where(coords % tick_freq == 0)[0]
    g.set_xticks(tick_idx)
    g.set_xticklabels(
        [f"{coords[i]:,}" for i in tick_idx], rotation=0, ha="center", fontsize=8
    )
    if chrom is not None:
        g.set_xlabel(f"Genomic position (chr{chrom})")
    if title is not None:
        g.set_title(title, fontsize=9)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=dpi)
    if out.suffix == ".svg":
        plt.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    plt.close()
