"""Embedding UMAP — pipeline glue + plotting (issue #246).

Loads a checkpoint, extracts a center-pooled FWD+RC-averaged embedding for each
labeled 100 bp region window (GPN-Star's ``songlab/gpn-star-umap-regions``),
projects with UMAP, and renders the 2-D scatter colored by functional region
(their Fig 4A) and by conservation (Fig 4B). The embedding kernel lives in
``marin_dna.model.embeddings``; this is the thin orchestration the Snakemake
rules (``rules/embedding_umap.smk``) call.

Coordinates are 0-based half-open everywhere (repo convention) — the dataset's
``start``/``end`` flow straight into ``Genome(chrom, start, end)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.data.genome import Genome
from marin_dna.model.embeddings import window_embeddings

# The dataset's 7 region labels → display names (GPN-Star Fig 4A).
REGION_DISPLAY: dict[str, str] = {
    "CDS": "CDS",
    "five_prime_UTR": "5' UTR",
    "three_prime_UTR": "3' UTR",
    "lnc_RNA": "lncRNA",
    "PLS": "Promoter",
    "dELS": "Enhancer",
    "background": "Background",
}
# Plot order + fixed palette (matplotlib ``Cn`` colors), mirroring the reference.
REGION_ORDER: list[str] = [
    "CDS",
    "5' UTR",
    "3' UTR",
    "Promoter",
    "Enhancer",
    "lncRNA",
    "Background",
]
REGION_PALETTE: dict[str, str] = {
    "CDS": "C2",
    "5' UTR": "C1",
    "3' UTR": "C4",
    "Promoter": "C0",
    "Enhancer": "C3",
    "lncRNA": "C9",
    "Background": "C5",
}

_CARRY_COLUMNS = ["chrom", "start", "end", "label", "cons"]


def load_umap_regions(dataset_id: str, *, split: str = "test") -> pd.DataFrame:
    """Load the labeled region windows (chrom/start/end/label/cons) from HF.

    Returns the windows with ``chrom`` coerced to ``str`` (Ensembl-style, no
    ``chr`` prefix). Asserts the expected columns and that every window is a
    non-empty 0-based half-open interval.
    """
    from datasets import load_dataset

    df = load_dataset(dataset_id, split=split).to_pandas()
    missing = set(_CARRY_COLUMNS) - set(df.columns)
    assert not missing, f"dataset {dataset_id!r} missing columns: {sorted(missing)}"
    df = df[_CARRY_COLUMNS].copy()
    df["chrom"] = df["chrom"].astype(str)
    assert (df["end"] > df["start"]).all(), "every window needs end > start"
    return df


def compute_region_embeddings(
    checkpoint_path: str | Path,
    genome_path: str | Path,
    regions: pd.DataFrame,
    window_size: int,
    *,
    layer_index: int = -1,
    n_center_bp: int = 100,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Embed every region window; return embeddings + carried metadata.

    For each window the model context (``window_size`` bp) is centered on the
    window midpoint; the center ``n_center_bp`` positions are mean-pooled and
    the FWD/RC strands averaged (see ``model.embeddings.window_embeddings``).
    Windows whose expanded context runs off a chromosome end or covers an
    assembly gap (any ``N``) are **dropped** — these are bulk data, not the
    curated handful the nuc_dep pipeline hard-asserts on. The reference FASTA is
    ``dna_sm`` (soft-masked), so repeats are lowercase and survive ``.upper()``;
    only true ``N`` (gaps) are dropped.

    Returns a DataFrame of the carried columns (``chrom/start/end/label/cons``)
    plus ``emb_0…emb_{D-1}``, one row per surviving window.
    """
    assert window_size >= n_center_bp, (
        f"window_size {window_size} must be >= n_center_bp {n_center_bp}"
    )
    tokenizer: Any = AutoTokenizer.from_pretrained(checkpoint_path)
    model: Any = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, trust_remote_code=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    genome = Genome(genome_path)
    # Sort by (chrom, start) so consecutive S3 byte-range reads hit nearby
    # bgzip blocks (fsspec block-cache locality); UMAP is order-independent.
    regions = regions.sort_values(["chrom", "start"]).reset_index(drop=True)

    seqs: list[str] = []
    keep: list[int] = []
    n_dropped = 0
    for row in regions.itertuples(index=True):
        center = (row.start + row.end) // 2
        ctx_start = center - window_size // 2
        seq = genome(str(row.chrom), ctx_start, ctx_start + window_size, "+").upper()
        if len(seq) != window_size or "N" in seq:
            n_dropped += 1
            continue
        seqs.append(seq)
        keep.append(row.Index)

    n_total = len(regions)
    frac = n_dropped / max(n_total, 1)
    print(f"[umap] dropped {n_dropped}/{n_total} ({frac:.3%}) N/out-of-bounds windows")
    assert frac < 0.05, (
        f"dropped {frac:.1%} of windows (>5%) — investigate genome build / coords"
    )
    assert seqs, "no windows survived N/bounds filtering"

    emb = window_embeddings(
        model,
        tokenizer,
        seqs,
        layer_index=layer_index,
        n_center_bp=n_center_bp,
        rc=True,
        batch_size=batch_size,
    )
    kept = regions.loc[keep, _CARRY_COLUMNS].reset_index(drop=True)
    emb_df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    out = pd.concat([kept, emb_df], axis=1)
    assert len(out) == len(seqs), f"row mismatch: {len(out)} vs {len(seqs)}"
    return out


def fit_umap(emb_df: pd.DataFrame, *, random_state: int = 42) -> pd.DataFrame:
    """Standardize the ``emb_*`` columns and project to 2-D with UMAP.

    Uses UMAP library defaults (``n_neighbors=15``, ``min_dist=0.1``,
    ``metric="euclidean"``, ``n_components=2``) with a fixed seed, matching the
    GPN-Star recipe. Lazy-imports ``umap`` (the optional ``umap`` dependency
    group) so the rest of the library imports without the heavy numba/llvmlite
    stack.

    Returns the carried metadata columns plus ``UMAP1`` / ``UMAP2``.
    """
    try:
        import umap
    except ImportError as e:  # pragma: no cover - exercised only without the group
        raise ImportError(
            "fit_umap requires `umap-learn`; install the optional group with "
            "`uv sync --group umap` (kept out of core deps because it pulls a "
            "~56 MB LLVM wheel via numba/llvmlite)."
        ) from e
    from sklearn.preprocessing import StandardScaler

    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    assert emb_cols, "no emb_* columns in embeddings frame"
    x = emb_df[emb_cols].to_numpy(dtype=np.float32)
    assert np.isfinite(x).all(), "embeddings contain non-finite values"
    x = StandardScaler().fit_transform(x)
    proj = umap.UMAP(random_state=random_state).fit_transform(x)
    out = emb_df.drop(columns=emb_cols).reset_index(drop=True)
    out["UMAP1"] = proj[:, 0]
    out["UMAP2"] = proj[:, 1]
    return out


def plot_umap(
    umap_df: pd.DataFrame,
    output_path: str | Path,
    *,
    color_by: Literal["region", "conservation"],
    dpi: int = 200,
) -> None:
    """Scatter the 2-D UMAP, colored by region label or by conservation.

    Small-file recipe (GPN-Star): ``figsize=(3, 3)``, point size
    ``s = 100/sqrt(N)`` (legible at any density), ``alpha=0.5``, points
    ``rasterized=True`` inside an otherwise-vector SVG (axes/text stay vector).
    Emits ``.svg`` (dashboard artifact) plus a ``.png`` sibling (repo
    convention — ``Read``-back sanity checks).
    """
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text, not paths
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(umap_df)
    assert n > 0, "empty UMAP frame"
    assert {"UMAP1", "UMAP2"} <= set(umap_df.columns), "missing UMAP1/UMAP2 columns"
    s = 100.0 / np.sqrt(n)
    fig, ax = plt.subplots(figsize=(3, 3))

    if color_by == "region":
        labels = umap_df["label"].map(REGION_DISPLAY)
        unmapped = sorted(set(umap_df["label"]) - set(REGION_DISPLAY))
        assert not unmapped, f"unmapped region labels: {unmapped}"
        sns.scatterplot(
            x=umap_df["UMAP1"],
            y=umap_df["UMAP2"],
            hue=labels,
            hue_order=REGION_ORDER,
            palette=REGION_PALETTE,
            s=s,
            alpha=0.5,
            linewidth=0,
            rasterized=True,
            ax=ax,
        )
        sns.move_legend(
            ax, "upper left", bbox_to_anchor=(1, 1), frameon=False, title=None
        )
        leg = ax.get_legend()
        if leg is not None:  # bump legend dots to readable size/opacity
            for handle in leg.legend_handles:
                handle.set_markersize(8)  # type: ignore[attr-defined]  # Line2D handle
                handle.set_alpha(1.0)
    elif color_by == "conservation":
        sc = ax.scatter(
            umap_df["UMAP1"],
            umap_df["UMAP2"],
            c=umap_df["cons"],
            cmap="viridis",
            s=s,
            alpha=0.5,
            linewidths=0,
            rasterized=True,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("conservation (phastCons, 75th pct)")
    else:
        raise ValueError(
            f"color_by must be 'region' or 'conservation', got {color_by!r}"
        )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    if out.suffix == ".svg":
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
