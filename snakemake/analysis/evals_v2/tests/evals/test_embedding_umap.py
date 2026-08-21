"""Tests for ``marin_dna_evals.embedding_umap`` (#246).

The model-dependent ``compute_region_embeddings`` is covered end-to-end by the
pipeline smoke run; here we unit-test the label/palette consistency, the
plotting recipe (rasterized, small SVG), and the UMAP fit (skipped when the
optional ``umap`` group isn't installed).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from marin_dna_evals import embedding_umap as eu
from marin_dna_evals.embedding_umap import (
    REGION_DISPLAY,
    REGION_ORDER,
    REGION_PALETTE,
    compute_region_embeddings,
    fit_umap,
    plot_umap,
)

# The 7 region labels in the HF dataset songlab/gpn-star-umap-regions.
_DATASET_LABELS = {
    "CDS",
    "five_prime_UTR",
    "three_prime_UTR",
    "lnc_RNA",
    "PLS",
    "dELS",
    "background",
}


def test_region_maps_are_consistent():
    # Every dataset label maps to a display name; order + palette agree.
    assert set(REGION_DISPLAY) == _DATASET_LABELS
    assert set(REGION_DISPLAY.values()) == set(REGION_ORDER)
    assert set(REGION_ORDER) == set(REGION_PALETTE)


def _synthetic(n: int = 60, d: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "chrom": "1",
            "start": np.arange(n) * 100,
            "end": np.arange(n) * 100 + 100,
            "label": rng.choice(sorted(_DATASET_LABELS), size=n),
            "cons": rng.random(n).astype(np.float32),
            **{f"emb_{i}": rng.standard_normal(n).astype(np.float32) for i in range(d)},
        }
    )


def _with_coords(df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.drop(columns=[c for c in df.columns if c.startswith("emb_")]).copy()
    df["UMAP1"] = rng.standard_normal(len(df))
    df["UMAP2"] = rng.standard_normal(len(df))
    return df


@pytest.mark.parametrize("color_by", ["region", "conservation"])
def test_plot_umap_writes_small_rasterized_svg(tmp_path: Path, color_by: str):
    df = _with_coords(_synthetic())
    out = tmp_path / f"{color_by}.svg"
    plot_umap(df, out, color_by=color_by, dpi=150)
    assert out.exists() and out.with_suffix(".png").exists()
    svg = out.read_text()
    assert "<image" in svg  # points are rasterized inside the otherwise-vector SVG
    assert out.stat().st_size < 500_000  # stays small


def test_plot_umap_rejects_unmapped_label(tmp_path: Path):
    df = pd.DataFrame(
        {"label": ["NOTALABEL"], "cons": [0.5], "UMAP1": [0.0], "UMAP2": [0.0]}
    )
    with pytest.raises(AssertionError, match="unmapped region labels"):
        plot_umap(df, tmp_path / "x.svg", color_by="region")


def test_plot_umap_rejects_bad_color_by(tmp_path: Path):
    df = _with_coords(_synthetic(n=5))
    with pytest.raises(ValueError, match="color_by"):
        plot_umap(df, tmp_path / "x.svg", color_by="rainbow")  # type: ignore[arg-type]


def test_fit_umap_shape_columns_and_determinism():
    pytest.importorskip("umap")  # optional `umap` dependency-group
    df = _synthetic(n=50, d=8)
    coords = fit_umap(df, random_state=42)
    assert {"UMAP1", "UMAP2"} <= set(coords.columns)
    assert not any(c.startswith("emb_") for c in coords.columns)
    assert {"chrom", "start", "end", "label", "cons"} <= set(coords.columns)
    assert len(coords) == 50
    coords2 = fit_umap(df, random_state=42)
    np.testing.assert_allclose(coords["UMAP1"], coords2["UMAP1"])


def test_compute_region_embeddings_embeds_all_windows(monkeypatch):
    """No N-dropping: every window is embedded (the tokenizer maps gap/edge bases
    to [UNK]), so the point set is identical across context sizes. Loaders +
    run_window_embeddings mocked — no checkpoint, no GPU (the transform's genome
    fetch happens inside the mocked harness, not here)."""
    n = 8
    starts = [1000 * i for i in range(n)]
    regions = pd.DataFrame(
        {
            "chrom": ["1"] * n,
            "start": starts,
            "end": [s + 100 for s in starts],
            "label": [sorted(_DATASET_LABELS)[i % 7] for i in range(n)],
            "cons": [i / n for i in range(n)],
        }
    )
    emb_block = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
    captured: dict[str, object] = {}

    def fake_embeddings(*args, **kwargs):
        captured.update(kwargs["inference_kwargs"])
        return emb_block

    monkeypatch.setattr(eu.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(eu.AutoModel, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(eu, "Genome", lambda *a, **k: object())
    monkeypatch.setattr(eu, "run_window_embeddings", fake_embeddings)

    out = compute_region_embeddings(
        "/ckpt",
        "/genome.fa",
        regions,
        window_size=20,
        n_center_bp=10,
        torch_compile=True,
        bf16=False,
    )

    assert captured["torch_compile"] is True
    assert captured["bf16_full_eval"] is False
    assert len(out) == n  # nothing dropped — full point set kept
    assert set(out["start"]) == set(starts)
    assert list(out.columns) == [
        "chrom",
        "start",
        "end",
        "label",
        "cons",
        "emb_0",
        "emb_1",
    ]
    np.testing.assert_array_equal(out[["emb_0", "emb_1"]].to_numpy(), emb_block)
