"""Orchestration tests for the variant-embedding cache driver (issue #314).

The model-dependent forward path (``run_variant_embeddings``) is exercised by a
GPU smoke run; here we mock it to test the *driver* logic on CPU — chunking,
the ``{fwd,rc}×{ref,alt}`` stack axis order, keys↔embedding row alignment, and
the shard roundtrip — so a GPU run can't be wasted on an orchestration bug.
"""

import numpy as np
import pandas as pd
import polars as pl

from marin_dna.pipelines.evals import variant_embedding as ve

_L, _D = 4, 3  # tiny per-token shape for the fake


def _fake_run_variant_embeddings(
    model, tokenizer, hf_ds, genome, window_size, *, strand, **kwargs
):
    """Return ``[n, 2(allele), L, D]`` filled with ``strand_code*100 + allele`` so
    the driver's stack/axis handling is verifiable."""
    n = len(hf_ds)
    sc = 1 if strand == "+" else 2
    out = np.zeros((n, 2, _L, _D), dtype=np.float16)
    out[:, 0] = sc * 100 + 0  # ref allele
    out[:, 1] = sc * 100 + 1  # alt allele
    return out


def _variants() -> pd.DataFrame:
    # Unsorted pos so the driver's (chrom,pos) sort is exercised; carries the
    # CV columns the probe needs downstream.
    return pd.DataFrame(
        {
            "chrom": ["3", "1", "1", "3", "1"],
            "pos": [50, 200, 100, 10, 300],
            "ref": list("ACGTA"),
            "alt": list("GTACG"),
            "label": [True, False, True, False, True],
            "subset": ["missense_variant"] * 5,
            "match_group": [2, 1, 1, 2, 1],
        }
    )


def test_driver_chunks_stacks_and_aligns(monkeypatch, tmp_path):
    monkeypatch.setattr(ve, "run_variant_embeddings", _fake_run_variant_embeddings)
    monkeypatch.setattr(ve, "Genome", lambda *a, **k: object())
    import transformers

    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object()
    )

    out_dir = str(tmp_path / "cache")
    n = ve.cache_variant_embeddings(
        "ckpt", _variants(), out_dir, window_size=255, chunk_size=2
    )
    assert n == 5

    # 5 rows / chunk 2 -> shards of 2, 2, 1.
    shards = sorted((tmp_path / "cache").glob("shard_*.npz"))
    assert [s.name for s in shards] == [
        "shard_0000.npz",
        "shard_0001.npz",
        "shard_0002.npz",
    ]

    embs, keys = [], []
    for s in shards:
        embs.append(np.load(s)["emb"])
        keys.append(pl.read_parquet(str(s).replace(".npz", ".keys.parquet")))
    assert [e.shape for e in embs] == [
        (2, 2, 2, _L, _D),
        (2, 2, 2, _L, _D),
        (1, 2, 2, _L, _D),
    ]

    emb = np.concatenate(embs)  # [5, 2(strand), 2(allele), L, D]
    # axis order (strand{fwd,rc}, allele{ref,alt}): fwd=1xx, rc=2xx; ref=x0, alt=x1.
    assert (emb[:, 0, 0] == 100).all() and (emb[:, 0, 1] == 101).all()  # fwd ref/alt
    assert (emb[:, 1, 0] == 200).all() and (emb[:, 1, 1] == 201).all()  # rc  ref/alt

    # Keys are written sorted by (chrom,pos) and carry the CV columns intact.
    keys_df = pl.concat(keys).to_pandas()
    expected = _variants().sort_values(["chrom", "pos"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        keys_df[["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]],
        expected[["chrom", "pos", "ref", "alt", "label", "subset", "match_group"]],
    )


def test_limit_caps_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(ve, "run_variant_embeddings", _fake_run_variant_embeddings)
    monkeypatch.setattr(ve, "Genome", lambda *a, **k: object())
    import transformers

    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object()
    )
    n = ve.cache_variant_embeddings(
        "ckpt", _variants(), str(tmp_path), window_size=255, chunk_size=8, limit=3
    )
    assert n == 3


def test_train_chroms_are_the_odd_autosomes_plus_x():
    assert ve.TRAIN_CHROMS == (
        "1",
        "3",
        "5",
        "7",
        "9",
        "11",
        "13",
        "15",
        "17",
        "19",
        "21",
        "X",
    )
