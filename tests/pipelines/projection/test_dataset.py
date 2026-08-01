"""Tests for ``marin_dna.pipelines.projection.dataset``."""

import json
from pathlib import Path

import polars as pl
import pytest

from marin_dna.pipelines.projection.dataset import (
    prepare_shards,
    reverse_complement_col,
)


def _rc_via_expr(seq: str) -> str:
    """Apply reverse_complement_col to a single string and return the result."""
    df = pl.DataFrame({"s": [seq]})
    return df.select(reverse_complement_col(pl.col("s")).alias("rc"))["rc"][0]


def test_rc_basic_acgt() -> None:
    assert _rc_via_expr("ACGT") == "ACGT"  # palindrome
    assert _rc_via_expr("AAAA") == "TTTT"
    assert _rc_via_expr("AAGCT") == "AGCTT"
    assert _rc_via_expr("acgt") == "acgt"  # lowercase preserved
    assert _rc_via_expr("Aa") == "tT"


def test_rc_round_trip_acgt() -> None:
    """rc(rc(seq)) == seq for ACGT strings."""
    for seq in ["ACGT", "GATTACA", "CCCGGG", "ATATAT", "agctAGCT"]:
        assert _rc_via_expr(_rc_via_expr(seq)) == seq


def test_rc_preserves_non_acgt_chars() -> None:
    """N and other IUPAC ambiguity chars are NOT strictly RC'd; they're left as-is.

    This is the documented deviation from canonical RC. Asserting it
    explicitly so the choice is loud.
    """
    # N self-pairs under canonical RC, so identity is "correct" by accident.
    assert _rc_via_expr("ANT") == "ANT"
    # R is NOT canonical-RC'd to Y; it's left as-is.
    assert (
        _rc_via_expr("AR") == "RT"
    )  # complement(A)=T, R unchanged, then reverse → "RT"
    # Multiple ambiguity chars preserved.
    assert _rc_via_expr("ACRYT") == "AYRGT"  # rev: T,Y,R,C,A → comp: A,Y,R,G,T


def _make_source_parquet(tmp_path: Path, n_rows: int = 100) -> Path:
    """Build a tiny source Parquet matching all_species_with_sequence schema."""
    df = pl.DataFrame(
        {
            "query_name": [f"win_1_{i:09d}" for i in range(n_rows)],
            "species": [
                "Mus_musculus" if i % 2 else "Bos_taurus" for i in range(n_rows)
            ],
            "t_chrom": ["chr1"] * n_rows,
            "t_start": [1000 + i for i in range(n_rows)],
            "t_end": [1000 + i + 255 for i in range(n_rows)],
            "t_strand": ["+" if i % 3 else "-" for i in range(n_rows)],
            "t_src_size": [200_000_000] * n_rows,
            "sequence": ["ACGT" * 63 + "ACG" for _ in range(n_rows)],  # 255 bp
        }
    )
    p = tmp_path / "src.parquet"
    df.write_parquet(p)
    return p


def test_prepare_shards_no_rc(tmp_path: Path) -> None:
    """add_rc=False: row count unchanged, no augmentation column."""
    src = _make_source_parquet(tmp_path, n_rows=20)
    shard_paths = [str(tmp_path / f"shard_{i:04d}.jsonl") for i in range(4)]
    prepare_shards(src, shard_paths, add_rc=False, shuffle_seed=42)

    rows: list[dict] = []
    for p in shard_paths:
        for line in Path(p).read_text().splitlines():
            rows.append(json.loads(line))
    assert len(rows) == 20
    assert "augmentation" not in rows[0]
    # All original query_names accounted for.
    assert {r["query_name"] for r in rows} == {f"win_1_{i:09d}" for i in range(20)}


def test_prepare_shards_with_rc_doubles_rows(tmp_path: Path) -> None:
    """add_rc=True: row count doubles; both '+' and '-' augmentation present."""
    src = _make_source_parquet(tmp_path, n_rows=10)
    shard_paths = [str(tmp_path / f"shard_{i:04d}.jsonl") for i in range(4)]
    prepare_shards(src, shard_paths, add_rc=True, shuffle_seed=42)

    rows: list[dict] = []
    for p in shard_paths:
        for line in Path(p).read_text().splitlines():
            rows.append(json.loads(line))
    assert len(rows) == 20  # doubled
    augs = [r["augmentation"] for r in rows]
    assert sum(1 for a in augs if a == "+") == 10
    assert sum(1 for a in augs if a == "-") == 10

    # For each query_name, the '-' row's sequence is RC of the '+' row's.
    by_qn_aug: dict[tuple[str, str], str] = {}
    for r in rows:
        by_qn_aug[(r["query_name"], r["augmentation"])] = r["sequence"]
    qn = next(iter({r["query_name"] for r in rows}))
    pos_seq = by_qn_aug[(qn, "+")]
    neg_seq = by_qn_aug[(qn, "-")]
    assert _rc_via_expr(pos_seq) == neg_seq


def test_prepare_shards_balanced_sizes(tmp_path: Path) -> None:
    """Shard sizes follow np.array_split semantics: |max - min| <= 1."""
    src = _make_source_parquet(tmp_path, n_rows=23)  # 23 / 4 = 5,5,5,5,3 → 6,6,6,5
    shard_paths = [str(tmp_path / f"shard_{i:04d}.jsonl") for i in range(4)]
    prepare_shards(src, shard_paths, add_rc=False, shuffle_seed=42)

    sizes = [sum(1 for _ in Path(p).read_text().splitlines()) for p in shard_paths]
    assert sum(sizes) == 23
    assert max(sizes) - min(sizes) <= 1


def test_prepare_shards_reproducible_with_seed(tmp_path: Path) -> None:
    """Same seed → identical shard contents in identical order."""
    src = _make_source_parquet(tmp_path, n_rows=30)

    out1 = [str(tmp_path / "a" / f"shard_{i:04d}.jsonl") for i in range(3)]
    out2 = [str(tmp_path / "b" / f"shard_{i:04d}.jsonl") for i in range(3)]
    prepare_shards(src, out1, add_rc=False, shuffle_seed=7)
    prepare_shards(src, out2, add_rc=False, shuffle_seed=7)

    for p1, p2 in zip(out1, out2):
        assert Path(p1).read_text() == Path(p2).read_text()


def test_prepare_shards_different_seed_changes_order(tmp_path: Path) -> None:
    """Different seeds produce different orderings."""
    src = _make_source_parquet(tmp_path, n_rows=30)
    out1 = [str(tmp_path / "a" / f"shard_{i:04d}.jsonl") for i in range(3)]
    out2 = [str(tmp_path / "b" / f"shard_{i:04d}.jsonl") for i in range(3)]
    prepare_shards(src, out1, add_rc=False, shuffle_seed=7)
    prepare_shards(src, out2, add_rc=False, shuffle_seed=8)

    contents1 = "".join(Path(p).read_text() for p in out1)
    contents2 = "".join(Path(p).read_text() for p in out2)
    assert contents1 != contents2  # exceedingly unlikely to match with 30 rows


def test_prepare_shards_missing_sequence_column_raises(tmp_path: Path) -> None:
    """Source parquet without ``sequence`` column fires the assertion."""
    df = pl.DataFrame({"query_name": ["x"], "species": ["A"]})
    p = tmp_path / "bad.parquet"
    df.write_parquet(p)
    out = [str(tmp_path / "shard.jsonl")]
    with pytest.raises(AssertionError, match="missing sequence column"):
        prepare_shards(p, out, add_rc=False, shuffle_seed=42)


def test_prepare_shards_preserves_all_columns(tmp_path: Path) -> None:
    """The full all_species_with_sequence schema survives the JSONL round-trip."""
    src = _make_source_parquet(tmp_path, n_rows=5)
    shard_paths = [str(tmp_path / "shard.jsonl")]
    prepare_shards(src, shard_paths, add_rc=False, shuffle_seed=42)

    expected_cols = {
        "query_name",
        "species",
        "t_chrom",
        "t_start",
        "t_end",
        "t_strand",
        "t_src_size",
        "sequence",
    }
    line = Path(shard_paths[0]).read_text().splitlines()[0]
    row = json.loads(line)
    assert expected_cols.issubset(set(row.keys()))


def test_prepare_shards_streaming_is_balanced_and_reproducible(
    tmp_path: Path,
) -> None:
    src = _make_source_parquet(tmp_path, n_rows=23)
    out1 = [str(tmp_path / "a" / f"shard_{i:04d}.jsonl") for i in range(4)]
    out2 = [str(tmp_path / "b" / f"shard_{i:04d}.jsonl") for i in range(4)]

    prepare_shards(
        src,
        out1,
        add_rc=False,
        shuffle_seed=7,
        max_in_memory_rows=1,
    )
    prepare_shards(
        src,
        out2,
        add_rc=False,
        shuffle_seed=7,
        max_in_memory_rows=1,
    )

    sizes = [len(Path(path).read_text().splitlines()) for path in out1]
    assert sizes == [6, 6, 6, 5]
    for path1, path2 in zip(out1, out2):
        assert Path(path1).read_bytes() == Path(path2).read_bytes()

    rows = [
        json.loads(line)
        for path in out1
        for line in Path(path).read_text().splitlines()
    ]
    assert {row["query_name"] for row in rows} == {f"win_1_{i:09d}" for i in range(23)}
    assert all(not key.startswith("__marin_") for row in rows for key in row)


def test_prepare_shards_streaming_seed_changes_order_and_supports_rc(
    tmp_path: Path,
) -> None:
    src = _make_source_parquet(tmp_path, n_rows=10)
    out1 = [str(tmp_path / "a" / f"shard_{i:04d}.jsonl") for i in range(3)]
    out2 = [str(tmp_path / "b" / f"shard_{i:04d}.jsonl") for i in range(3)]

    prepare_shards(
        src,
        out1,
        add_rc=True,
        shuffle_seed=7,
        max_in_memory_rows=1,
    )
    prepare_shards(
        src,
        out2,
        add_rc=True,
        shuffle_seed=8,
        max_in_memory_rows=1,
    )

    rows = [
        json.loads(line)
        for path in out1
        for line in Path(path).read_text().splitlines()
    ]
    assert len(rows) == 20
    assert {row["augmentation"] for row in rows} == {"+", "-"}
    assert "".join(Path(path).read_text() for path in out1) != "".join(
        Path(path).read_text() for path in out2
    )
