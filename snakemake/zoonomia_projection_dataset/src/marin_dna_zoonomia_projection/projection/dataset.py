"""Training-shard prep (RC augmentation, shuffle, shard to JSONL)."""

from pathlib import Path

import polars as pl

from marin_dna.data.utils import get_array_split_pairs

_COMPLEMENT = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "a": "t",
    "t": "a",
    "c": "g",
    "g": "c",
}


def reverse_complement_col(seq: pl.Expr) -> pl.Expr:
    """Vectorised reverse-complement on ACGT only; other chars (N, IUPAC) pass through."""
    return seq.str.replace_many(_COMPLEMENT).str.reverse()


def prepare_shards(
    parquet_path: str | Path,
    shard_paths: list[str],
    add_rc: bool,
    shuffle_seed: int,
) -> None:
    """Read source Parquet → optional RC augment → shuffle → shard to JSONL.

    Args:
        parquet_path: source Parquet. Must include a ``sequence`` column.
        shard_paths: ordered list of N output JSONL paths. Row count is
            split evenly (np.array_split semantics).
        add_rc: if True, append an ``augmentation`` column with values
            ``"+"`` (original) and ``"-"`` (reverse-complemented sequence).
            Total row count doubles.
        shuffle_seed: passed to ``df.sample(fraction=1, shuffle=True, seed=...)``
            for cross-species interleaving.
    """
    assert len(shard_paths) > 0
    df = pl.read_parquet(str(parquet_path))
    assert "sequence" in df.columns, f"missing sequence column: {df.columns}"

    if add_rc:
        df_pos = df.with_columns(pl.lit("+").alias("augmentation"))
        df_neg = df.with_columns(
            reverse_complement_col(pl.col("sequence")).alias("sequence"),
            pl.lit("-").alias("augmentation"),
        )
        df = pl.concat([df_pos, df_neg])

    df = df.sample(fraction=1.0, shuffle=True, seed=shuffle_seed)

    for path, (start, end) in zip(
        shard_paths, get_array_split_pairs(len(df), len(shard_paths))
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.slice(start, end - start).write_ndjson(path)
