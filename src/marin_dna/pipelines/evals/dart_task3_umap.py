"""DART-Eval Task 3 cell-type UMAP — load + plot (issue #298).

The cell-type-discrimination analogue of the GPN-Star region UMAP (#246 / #248):
embed the ``bolinas-dna/evals_dart_task3`` cell-type peak windows with a gLM and
project to 2-D, colored by the 5-way cell-type ``label``. The heavy lifting is
reused from the GPN-Star harness — the embedding kernel
(``embedding_umap.compute_region_embeddings``) and the UMAP fit
(``embedding_umap.fit_umap``); this module holds only the DART-specific bits:
loading the interval dataset (no conservation column, fixed 500 bp peaks),
guarding the long-context arm against a too-small position budget, and the
cell-type-colored scatter.

The same model is embedded at several context sizes (the ``context_sizes`` knob,
e.g. 255 + 500): each arm sets ``window_size = n_center_bp = ctx`` so the model
sees ``ctx`` bp and every DNA token is mean-pooled (whole-window pooling). The
255 bp arm is exp136's native context; 500 bp is the full peak — ~2x the
training context, i.e. RoPE extrapolation (out-of-distribution).

Coordinates are 0-based half-open everywhere (repo convention).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from marin_dna.pipelines.evals.dart_task3 import CELL_TYPES, PEAK_WIDTH

# Plot order + fixed palette (matplotlib ``Cn`` colors) for the 5 ENCODE cell
# lines, stable across arms/runs so the two context-size UMAPs share colors.
# Labels are already display-ready (``GM12878`` …), so there is no label→display
# remap like GPN-Star's region map — the ``label`` column is the hue directly.
CELLTYPE_ORDER: list[str] = list(CELL_TYPES)
CELLTYPE_PALETTE: dict[str, str] = {
    "GM12878": "C0",
    "H1ESC": "C1",
    "HEPG2": "C2",
    "IMR90": "C3",
    "K562": "C4",
}

# Interval-only schema (no ``cons``); ``label`` is the cell type (one of CELL_TYPES).
_CARRY_COLUMNS = ["chrom", "start", "end", "label"]


def load_dart_regions(dataset_id: str, *, split: str = "validation") -> pd.DataFrame:
    """Load the DART Task 3 cell-type peak windows (chrom/start/end/label) from HF.

    Asserts the interval schema, that every window is a 0-based half-open
    ``PEAK_WIDTH`` (500 bp) interval, and that every label is one of the 5 cell
    types. Coerces ``chrom`` to ``str`` (Ensembl-style, no ``chr`` prefix) so it
    flows straight into ``Genome(chrom, start, end)``. Returns the windows ready
    for ``embedding_umap.compute_region_embeddings``.
    """
    from datasets import load_dataset

    df = load_dataset(dataset_id, split=split).to_pandas()
    missing = set(_CARRY_COLUMNS) - set(df.columns)
    assert not missing, f"dataset {dataset_id!r} missing columns: {sorted(missing)}"
    df = df[_CARRY_COLUMNS].copy()
    df["chrom"] = df["chrom"].astype(str)
    assert (df["end"] > df["start"]).all(), "every window needs end > start"
    widths = df["end"] - df["start"]
    assert (widths == PEAK_WIDTH).all(), (
        f"every window must be {PEAK_WIDTH} bp; got widths "
        f"{sorted(set(widths.astype(int).tolist()))}"
    )
    bad = sorted(set(df["label"]) - set(CELL_TYPES))
    assert not bad, f"unexpected cell-type labels: {bad} (known: {CELL_TYPES})"
    return df


def read_position_limits(checkpoint_path: str | Path) -> tuple[int, bool]:
    """Return ``(max_position_embeddings, uses_rope)`` from a HF checkpoint config.

    ``uses_rope`` is True when the config declares rotary embeddings
    (``rope_theta`` or ``rope_scaling``) — our qwen3/llama gLMs all do. A RoPE
    model computes positions dynamically, so a window longer than
    ``max_position_embeddings`` *runs* (extrapolating, OOD) rather than erroring or
    truncating; a model with learned absolute positions would instead index past
    its table and crash. ``check_window_fits`` uses the flag to decide warn-vs-fail.
    """
    cfg = json.loads((Path(checkpoint_path) / "config.json").read_text())
    mpe = cfg.get("max_position_embeddings")
    assert mpe is not None, (
        f"checkpoint {checkpoint_path}: config.json has no max_position_embeddings"
    )
    uses_rope = cfg.get("rope_theta") is not None or cfg.get("rope_scaling") is not None
    return int(mpe), bool(uses_rope)


def check_window_fits(
    max_position_embeddings: int,
    window_size: int,
    *,
    uses_rope: bool,
    n_special: int = 1,
) -> None:
    """Guard the embedding context against the model's position budget (issue #298).

    ``need = window_size + n_special``. If it fits ``max_position_embeddings``,
    no-op. If it exceeds:

    - **RoPE model** (``uses_rope=True``): positions are computed dynamically, so
      the forward runs — but positions ``max_position_embeddings…need-1`` are
      never-trained (extrapolation / OOD). We **warn and proceed**, which is what
      lets an intentional long-context arm run (e.g. DART 500 bp = 501 tokens on
      exp136, whose ``max_position_embeddings`` is 256 = its 255 bp + BOS training
      context).
    - **non-RoPE model**: the forward would index past the learned absolute-
      position table and crash, so we **hard-fail** before spending GPU time.
    """
    need = window_size + n_special
    if need <= max_position_embeddings:
        return
    detail = (
        f"window {window_size} bp + {n_special} special = {need} tokens > model "
        f"max_position_embeddings {max_position_embeddings}"
    )
    if uses_rope:
        print(
            f"[dart_umap] WARNING: {detail}; RoPE extrapolation — positions "
            f"{max_position_embeddings}..{need - 1} were never seen in training (OOD)"
        )
        return
    raise AssertionError(
        f"{detail}; non-RoPE model would index past its learned position table"
    )


def plot_dart_umap(
    umap_df: pd.DataFrame, output_path: str | Path, *, dpi: int = 200
) -> None:
    """Scatter the 2-D UMAP colored by the 5-way cell-type ``label``.

    Same small-file recipe as the GPN-Star region plot
    (``embedding_umap.plot_umap``): ``figsize=(3, 3)``, point size
    ``s = 100/sqrt(N)``, ``alpha=0.5``, points ``rasterized=True`` inside an
    otherwise-vector SVG (axes/text stay vector). Emits ``.svg`` (the artifact to
    post / upload) plus a ``.png`` sibling (repo convention — ``Read``-back
    sanity checks).
    """
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text, not paths
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(umap_df)
    assert n > 0, "empty UMAP frame"
    assert {"UMAP1", "UMAP2"} <= set(umap_df.columns), "missing UMAP1/UMAP2 columns"
    unmapped = sorted(set(umap_df["label"]) - set(CELLTYPE_PALETTE))
    assert not unmapped, f"unmapped cell-type labels: {unmapped}"

    s = 100.0 / np.sqrt(n)
    fig, ax = plt.subplots(figsize=(3, 3))
    hue_kwargs: dict[str, Any] = dict(
        hue=umap_df["label"], hue_order=CELLTYPE_ORDER, palette=CELLTYPE_PALETTE
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
    # Guard on the legend existing (seaborn omits it for a degenerate hue), then
    # bump the dots to a readable size/opacity — mirrors plot_umap.
    if ax.get_legend() is not None:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        leg = ax.get_legend()  # move_legend rebuilds the legend
        leg.set_title("Cell type")
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
