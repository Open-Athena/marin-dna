"""Embedding UMAP — pipeline glue + plotting (issue #246).

Loads a checkpoint, extracts a center-pooled FWD+RC-averaged embedding for each
labeled 100 bp region window (GPN-Star's ``songlab/gpn-star-umap-regions``),
projects with UMAP, and renders the 2-D scatter colored by functional region
(their Fig 4A) and by conservation (Fig 4B). The embedding kernel is the shared
HF Trainer harness: ``marin_dna_evals.model.runner.run_window_embeddings`` (FWD+RC,
strand-aware center bounds) over ``model.scoring.compute_window_embedding`` +
``data.transforms.transform_window_embedding``. This is the thin orchestration
the Snakemake rules (``rules/embedding_umap.smk``) call.

Coordinates are 0-based half-open everywhere (repo convention) — the dataset's
``start``/``end`` flow straight into ``Genome(chrom, start, end)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from datasets import Dataset

from marin_dna.data.genome import Genome
from marin_dna_evals.hf_compat import load_hf_base_model_and_tokenizer
from marin_dna_evals.model.runner import run_window_embeddings

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
    batch_size: int = 128,
    num_workers: int = 4,
    torch_compile: bool = False,
    bf16: bool = True,
) -> pd.DataFrame:
    """Embed every region window; return embeddings + carried metadata.

    Runs through the shared HF Trainer harness
    (``marin_dna_evals.model.runner.run_window_embeddings``) — optional bf16 and
    ``torch.compile``, ``num_workers`` dataloader workers — on a base
    ``AutoModel`` (reads ``last_hidden_state``; no LM head). For each window the
    ``window_size`` context is centered on the region midpoint, the center
    ``n_center_bp`` token positions are mean-pooled, and the FWD/RC strands are
    averaged.

    **Every** window is embedded — none are dropped. The labeled 100 bp locus is
    interior real sequence and stays centered in the window; only the expanded
    flanks can run off a chromosome end or reach an assembly gap, where ``Genome``
    pads with ``N`` and the tokenizer maps those to ``[UNK]`` (a token the model
    saw in training). Keeping the full set is deliberate: the windows are then
    identical across models of *different* ``window_size`` (each scores the exact
    same loci), so their UMAPs are point-for-point comparable. The pooled center
    is never padding, so this only ever touches flank context.

    Returns a DataFrame of the carried columns (``chrom/start/end/label/cons``)
    plus ``emb_0…emb_{D-1}``, one row per window (input order, after a
    ``(chrom, start)`` sort).
    """
    assert window_size >= n_center_bp, (
        f"window_size {window_size} must be >= n_center_bp {n_center_bp}"
    )
    # Use the base model because this path reads hidden states and does not need
    # the LM head. The shared loader validates the raw RoPE schema first.
    tokenizer, model = load_hf_base_model_and_tokenizer(checkpoint_path)

    genome = Genome(genome_path)
    # Sort by (chrom, start) so the dataloader transform's S3 byte-range reads
    # hit nearby bgzip blocks; UMAP is order-independent.
    regions = regions.sort_values(["chrom", "start"]).reset_index(drop=True)

    # The pooled center must stay inside the labeled locus, or the embedding
    # would represent flank context the label doesn't describe. Enforce uniform
    # region width and that the center pool fits within it.
    widths = regions["end"] - regions["start"]
    assert (widths == widths.iloc[0]).all(), "region windows must be uniform width"
    assert n_center_bp <= int(widths.iloc[0]), (
        f"n_center_bp {n_center_bp} exceeds the {int(widths.iloc[0])} bp region "
        f"width; the pooled center would extend past the labeled locus into flanks"
    )

    hf_dataset = Dataset.from_pandas(
        regions[["chrom", "start", "end"]], preserve_index=False
    )
    emb = run_window_embeddings(
        model,
        tokenizer,
        hf_dataset,
        genome,
        window_size,
        n_center_bp=n_center_bp,
        layer_index=layer_index,
        rc=True,
        data_transform_on_the_fly=True,
        inference_kwargs={
            "per_device_eval_batch_size": batch_size,
            "torch_compile": torch_compile,
            "bf16_full_eval": bf16,
            "dataloader_num_workers": num_workers,
            "remove_unused_columns": False,
        },
    )  # [N, D]
    assert emb.shape[0] == len(regions), (
        f"row mismatch: {emb.shape[0]} vs {len(regions)}"
    )
    # Fail at the (expensive) compute step rather than letting a NaN/Inf row
    # reach the parquet and surface only in the downstream fit_umap rule.
    assert np.isfinite(emb).all(), "embeddings contain non-finite values"
    emb_df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    # regions is already 0..N-1 indexed (reset above); concat aligns by position.
    return pd.concat([regions[_CARRY_COLUMNS], emb_df], axis=1)


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

    # Both panels go through seaborn scatterplot, mirroring GPN-Star's
    # interpretation_umap_plot exactly (so the colors match the paper):
    #   region       -> hue=label, fixed Cn palette + label order
    #   conservation -> hue=cons (numeric), NO palette/cmap => seaborn's default
    #                   continuous palette (white->dark) + a representative-level
    #                   legend, not a colorbar.
    if color_by == "region":
        hue: Any = umap_df["label"].map(REGION_DISPLAY)
        unmapped = sorted(set(umap_df["label"]) - set(REGION_DISPLAY))
        assert not unmapped, f"unmapped region labels: {unmapped}"
        hue_kwargs: dict[str, Any] = {
            "hue": hue,
            "hue_order": REGION_ORDER,
            "palette": REGION_PALETTE,
        }
        legend_title = "Region"
    elif color_by == "conservation":
        hue_kwargs = {"hue": umap_df["cons"]}
        legend_title = "Conservation"
    else:
        raise ValueError(
            f"color_by must be 'region' or 'conservation', got {color_by!r}"
        )

    sns.scatterplot(
        x=umap_df["UMAP1"],
        y=umap_df["UMAP2"],
        s=s,
        alpha=0.5,
        linewidth=0,
        edgecolor=None,
        rasterized=True,
        ax=ax,
        **hue_kwargs,
    )
    # Guard move_legend on the legend existing: seaborn omits it for a
    # degenerate hue (e.g. a constant/all-NaN cons column), and move_legend
    # raises "Legend data not found" with no legend present.
    if ax.get_legend() is not None:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        leg = ax.get_legend()  # move_legend rebuilds the legend
        leg.set_title(legend_title)  # title + bump dots to readable size/opacity
        for handle in leg.legend_handles:
            handle.set_markersize(8)  # type: ignore[attr-defined]  # Line2D handle
            handle.set_alpha(1.0)

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
