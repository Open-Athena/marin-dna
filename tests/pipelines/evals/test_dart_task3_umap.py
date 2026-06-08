"""Tests for ``marin_dna.pipelines.evals.dart_task3_umap`` (#298).

The model-dependent embedding is reused from ``embedding_umap`` (covered there +
by the pipeline smoke run); here we unit-test the DART-specific bits: the
cell-type palette consistency, the interval-dataset loader's validation, the
long-context fit guard, the cell-type scatter, and that the shared embedding
kernel carries a ``cons``-less (DART) frame.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marin_dna.pipelines.evals.dart_task3 import CELL_TYPES
from marin_dna.pipelines.evals.dart_task3_umap import (
    CELLTYPE_ORDER,
    CELLTYPE_PALETTE,
    check_window_fits,
    load_dart_regions,
    plot_dart_umap,
    read_position_limits,
)


def test_celltype_maps_are_consistent():
    # Order is the canonical CELL_TYPES; the palette covers exactly those labels.
    assert CELLTYPE_ORDER == list(CELL_TYPES)
    assert set(CELLTYPE_PALETTE) == set(CELL_TYPES)
    assert set(CELLTYPE_ORDER) == set(CELLTYPE_PALETTE)


# --- load_dart_regions -------------------------------------------------------


class _FakeHF:
    """Stand-in for a ``datasets.Dataset`` exposing only ``.to_pandas()``."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df


def _patch_load(monkeypatch, df: pd.DataFrame) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _FakeHF(df))


def _valid_source(n: int = 10) -> pd.DataFrame:
    """A raw evals_dart_task3 frame: 500 bp windows, int chrom, an extra column
    the loader should drop."""
    starts = np.arange(n) * 1000
    half = n // 2
    return pd.DataFrame(
        {
            "chrom": [6] * half + [21] * (n - half),  # int → loader coerces to str
            "start": starts,
            "end": starts + 500,
            "label": [CELL_TYPES[i % len(CELL_TYPES)] for i in range(n)],
            "extra": 1,  # not in the carry schema — dropped
        }
    )


def test_load_dart_regions_valid(monkeypatch):
    _patch_load(monkeypatch, _valid_source(10))
    out = load_dart_regions("bolinas-dna/evals_dart_task3", split="validation")
    assert list(out.columns) == ["chrom", "start", "end", "label"]  # extra dropped
    assert out["chrom"].tolist() == ["6"] * 5 + ["21"] * 5  # coerced to str
    assert ((out["end"] - out["start"]) == 500).all()
    assert set(out["label"]) <= set(CELL_TYPES)


def test_load_dart_regions_missing_column_raises(monkeypatch):
    _patch_load(monkeypatch, _valid_source(4).drop(columns=["label"]))
    with pytest.raises(AssertionError, match="missing columns"):
        load_dart_regions("x")


def test_load_dart_regions_bad_width_raises(monkeypatch):
    df = _valid_source(4)
    df.loc[0, "end"] = df.loc[0, "start"] + 400  # 400 bp window
    _patch_load(monkeypatch, df)
    with pytest.raises(AssertionError, match="500 bp"):
        load_dart_regions("x")


def test_load_dart_regions_bad_label_raises(monkeypatch):
    df = _valid_source(4)
    df.loc[0, "label"] = "NOTACELL"
    _patch_load(monkeypatch, df)
    with pytest.raises(AssertionError, match="unexpected cell-type labels"):
        load_dart_regions("x")


# --- assert_window_fits ------------------------------------------------------


def test_read_position_limits(tmp_path: Path):
    import json

    (tmp_path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 256, "rope_theta": 500000})
    )
    assert read_position_limits(tmp_path) == (256, True)  # rope_theta ⇒ uses_rope
    (tmp_path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 512})  # no rope fields
    )
    assert read_position_limits(tmp_path) == (512, False)


def test_check_window_fits_in_budget():
    # Fits the budget → no-op (no print, no raise), regardless of rope.
    check_window_fits(32768, 500, uses_rope=True)
    check_window_fits(256, 255, uses_rope=True)  # exp136 native: 255 bp + BOS == 256
    check_window_fits(500, 500, uses_rope=False, n_special=0)  # 500 <= 500


def test_check_window_fits_rope_extrapolates(capsys):
    # Over budget + RoPE → warn and PROCEED (the DART 500 bp OOD arm on exp136:
    # 501 tokens > max_position_embeddings 256). No raise.
    check_window_fits(256, 500, uses_rope=True)
    out = capsys.readouterr().out
    assert "RoPE extrapolation" in out and "OOD" in out


def test_check_window_fits_non_rope_hard_fails():
    # Over budget + non-RoPE would index past the learned position table → fail.
    with pytest.raises(AssertionError, match="non-RoPE"):
        check_window_fits(256, 500, uses_rope=False)


# --- plot_dart_umap ----------------------------------------------------------


def _coords(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    starts = np.arange(n) * 500
    return pd.DataFrame(
        {
            "chrom": "6",
            "start": starts,
            "end": starts + 500,
            "label": rng.choice(CELL_TYPES, size=n),
            "UMAP1": rng.standard_normal(n),
            "UMAP2": rng.standard_normal(n),
        }
    )


def test_plot_dart_umap_writes_small_rasterized_svg(tmp_path: Path):
    out = tmp_path / "celltype.svg"
    plot_dart_umap(_coords(), out, dpi=150)
    assert out.exists() and out.with_suffix(".png").exists()
    svg = out.read_text()
    assert "<image" in svg  # points rasterized inside the otherwise-vector SVG
    assert out.stat().st_size < 500_000  # stays small


def test_plot_dart_umap_rejects_unmapped_label(tmp_path: Path):
    df = pd.DataFrame({"label": ["NOTACELL"], "UMAP1": [0.0], "UMAP2": [0.0]})
    with pytest.raises(AssertionError, match="unmapped cell-type labels"):
        plot_dart_umap(df, tmp_path / "x.svg")


# --- shared embedding kernel carries a cons-less frame -----------------------


def test_compute_region_embeddings_carries_without_cons(monkeypatch):
    """DART task3 has no ``cons`` column; the shared kernel must carry the 4
    interval columns + emb_*. Loaders + run_window_embeddings mocked (no
    checkpoint, no GPU). Also exercises the long arm (window_size == n_center_bp
    == 500, whole-window pooling)."""
    from marin_dna.pipelines.evals import embedding_umap as eu
    from marin_dna.pipelines.evals.embedding_umap import compute_region_embeddings

    n = 6
    starts = [500 * i for i in range(n)]
    regions = pd.DataFrame(
        {
            "chrom": ["6"] * n,
            "start": starts,
            "end": [s + 500 for s in starts],
            "label": [CELL_TYPES[i % len(CELL_TYPES)] for i in range(n)],
        }
    )
    emb_block = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
    monkeypatch.setattr(eu.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(eu.AutoModel, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(eu, "Genome", lambda *a, **k: object())
    monkeypatch.setattr(eu, "run_window_embeddings", lambda *a, **k: emb_block)

    out = compute_region_embeddings(
        "/ckpt", "/genome.fa", regions, window_size=500, n_center_bp=500
    )

    assert list(out.columns) == ["chrom", "start", "end", "label", "emb_0", "emb_1"]
    assert len(out) == n  # nothing dropped
    np.testing.assert_array_equal(out[["emb_0", "emb_1"]].to_numpy(), emb_block)
