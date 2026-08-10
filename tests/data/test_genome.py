"""Tests for ``marin_dna.data.genome.Genome``.

Vendored from biofoundation/tests/test_data.py at commit 834dd4c (May 2026):
only the Genome tests are migrated — GenomicSet has its own home at
``marin_dna.data.intervals`` (tested separately in ``tests/test_intervals.py``).
"""

import textwrap

import pytest

from marin_dna.data.genome import Genome


def _write_genome_fasta(tmp_path):
    fasta = textwrap.dedent(">chr1\nACGTACGTAC\n>chr2\nGGGCCCAGTA\n")
    path = tmp_path / "genome.fa"
    path.write_text(fasta)
    return path


def test_genome_returns_subsequence(tmp_path):
    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path)

    seq = genome("chr1", start=2, end=7)

    assert seq == "GTACG"


def test_genome_reverse_complement(tmp_path):
    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path)

    seq = genome("chr2", start=1, end=6, strand="-")

    assert seq == "GGGCC"


def test_genome_left_padding(tmp_path):
    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path)

    seq = genome("chr1", start=-2, end=3)

    assert seq == "NNACG"


def test_genome_right_padding(tmp_path):
    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path)

    seq = genome("chr1", start=7, end=12)

    assert seq == "TACNN"


def test_genome_padding_both_sides(tmp_path):
    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path)

    seq = genome("chr1", start=-3, end=12)

    assert seq == "NNNACGTACGTACNN"


def test_genome_requires_known_chromosome(tmp_path):
    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path)

    with pytest.raises(ValueError, match="chromosome chr3 not found"):
        genome("chr3", start=0, end=1)


def test_genome_validates_span(tmp_path):
    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path)

    assert genome("chr1", start=5, end=5) == ""
    with pytest.raises(ValueError, match="start 6 must be less than or equal to end 4"):
        genome("chr1", start=6, end=4)

    with pytest.raises(ValueError, match="end -1 must be non-negative"):
        genome("chr1", start=-5, end=-1)

    with pytest.raises(ValueError, match="start 11 is out of range"):
        genome("chr1", start=11, end=11)

    with pytest.raises(ValueError, match="start 10 is out of range"):
        genome("chr1", start=10, end=12)


def test_genome_respects_subset(tmp_path):
    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path, subset_chroms={"chr2"})

    assert genome("chr2", start=0, end=4) == "GGGC"

    with pytest.raises(ValueError, match="chromosome chr1 not found"):
        genome("chr1", start=0, end=4)


def test_genome_survives_pickle_roundtrip(tmp_path):
    """A pickled Genome must work after unpickling (spawn-worker scenario)."""
    import pickle

    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path)
    assert genome("chr1", start=2, end=7) == "GTACG"  # populate _fa in parent

    restored = pickle.loads(pickle.dumps(genome))
    assert restored("chr1", start=2, end=7) == "GTACG"
    assert restored("chr2", start=1, end=6, strand="-") == "GGGCC"


def test_genome_ignores_storage_options_for_local_paths(tmp_path):
    fasta_path = _write_genome_fasta(tmp_path)
    genome = Genome(fasta_path, storage_options={"anon": True, "ignored": "kw"})

    assert genome("chr1", start=2, end=7) == "GTACG"


def test_genome_forwards_storage_options_to_fsspec(monkeypatch):
    fsspec = pytest.importorskip("fsspec")

    captured = {}

    class _Sentinel(Exception):
        pass

    def fake_open(path, **kwargs):
        captured.update(path=path, kwargs=kwargs)
        raise _Sentinel

    monkeypatch.setattr(fsspec, "open", fake_open)

    opts = {"anon": True, "endpoint_url": "https://example"}
    # The probe is lazy now (so fsspec init is deferred until first use,
    # avoiding event-loop inheritance breakage in DataLoader fork()'d
    # workers); construction alone shouldn't touch fsspec.
    g = Genome("s3://fake-bucket/x.fa", storage_options=opts)
    assert captured == {}

    with pytest.raises(_Sentinel):
        _ = g.chroms  # triggers _ensure_probed → _open_fasta → fsspec.open

    assert captured["path"] == "s3://fake-bucket/x.fa"
    assert captured["kwargs"] == {"anon": True, "endpoint_url": "https://example"}
