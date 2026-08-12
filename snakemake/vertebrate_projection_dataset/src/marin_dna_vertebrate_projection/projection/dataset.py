"""Training-shard prep (RC augmentation, shuffle, shard to JSONL)."""

from pathlib import Path
from tempfile import TemporaryDirectory

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

_SOURCE_ROW_COLUMN = "__marin_source_row"
_SHUFFLE_HASH_COLUMN = "__marin_shuffle_hash"
_SHUFFLE_RANK_COLUMN = "__marin_shuffle_rank"
_SHARD_COLUMN = "__marin_shard"
_DEFAULT_MAX_IN_MEMORY_ROWS = 100_000_000


def reverse_complement_col(seq: pl.Expr) -> pl.Expr:
    """Vectorised reverse-complement on ACGT only; other chars (N, IUPAC) pass through."""
    return seq.str.replace_many(_COMPLEMENT).str.reverse()


def prepare_shards(
    parquet_path: str | Path,
    shard_paths: list[str],
    add_rc: bool,
    shuffle_seed: int,
    max_in_memory_rows: int = _DEFAULT_MAX_IN_MEMORY_ROWS,
) -> None:
    """Read source Parquet → optional RC augment → shuffle → shard to JSONL.

    Inputs up to ``max_in_memory_rows`` use the original eager Polars shuffle.
    Larger inputs use a deterministic hash sort in Polars' streaming engine and
    write balanced round-robin partitions. The latter can spill its global sort
    to disk instead of materializing the complete dataset in RAM.

    Args:
        parquet_path: source Parquet. Must include a ``sequence`` column.
        shard_paths: ordered list of N output JSONL paths. Row count is
            split evenly (np.array_split semantics).
        add_rc: if True, append an ``augmentation`` column with values
            ``"+"`` (original) and ``"-"`` (reverse-complemented sequence).
            Total row count doubles.
        shuffle_seed: deterministic shuffle seed for cross-species interleaving.
        max_in_memory_rows: maximum post-augmentation row count for the eager
            shuffle. Larger datasets use the bounded-memory streaming path.
    """
    assert len(shard_paths) > 0
    assert max_in_memory_rows > 0
    schema = pl.read_parquet_schema(str(parquet_path))
    assert "sequence" in schema, f"missing sequence column: {list(schema)}"
    helper_columns = {
        _SOURCE_ROW_COLUMN,
        _SHUFFLE_HASH_COLUMN,
        _SHUFFLE_RANK_COLUMN,
        _SHARD_COLUMN,
    }
    assert helper_columns.isdisjoint(schema), (
        f"source schema uses reserved sharding columns: {sorted(helper_columns & schema.keys())}"
    )

    source_rows = (
        pl.scan_parquet(str(parquet_path))
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    output_rows = int(source_rows) * (2 if add_rc else 1)
    if output_rows > max_in_memory_rows:
        print(
            "prepare_shards: using bounded-memory streaming shuffle for "
            f"{output_rows:,} rows"
        )
        _prepare_shards_streaming(
            parquet_path=parquet_path,
            shard_paths=shard_paths,
            add_rc=add_rc,
            shuffle_seed=shuffle_seed,
        )
        return

    print(f"prepare_shards: using eager shuffle for {output_rows:,} rows")
    _prepare_shards_eager(
        parquet_path=parquet_path,
        shard_paths=shard_paths,
        add_rc=add_rc,
        shuffle_seed=shuffle_seed,
    )


def _prepare_shards_eager(
    *,
    parquet_path: str | Path,
    shard_paths: list[str],
    add_rc: bool,
    shuffle_seed: int,
) -> None:
    df = pl.read_parquet(str(parquet_path))

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


def _prepare_shards_streaming(
    *,
    parquet_path: str | Path,
    shard_paths: list[str],
    add_rc: bool,
    shuffle_seed: int,
) -> None:
    output_paths = [Path(path) for path in shard_paths]
    output_parents = {path.parent.resolve() for path in output_paths}
    assert len(output_parents) == 1, (
        "bounded-memory shard outputs must share one directory"
    )
    output_parent = output_paths[0].parent
    output_parent.mkdir(parents=True, exist_ok=True)

    lazy = pl.scan_parquet(str(parquet_path))
    if add_rc:
        lazy = pl.concat(
            [
                lazy.with_columns(pl.lit("+").alias("augmentation")),
                lazy.with_columns(
                    reverse_complement_col(pl.col("sequence")).alias("sequence"),
                    pl.lit("-").alias("augmentation"),
                ),
            ],
            how="vertical",
        )

    shuffled = (
        lazy.with_row_index(_SOURCE_ROW_COLUMN)
        .with_columns(
            pl.col(_SOURCE_ROW_COLUMN)
            .hash(seed=shuffle_seed)
            .alias(_SHUFFLE_HASH_COLUMN)
        )
        .sort(_SHUFFLE_HASH_COLUMN, _SOURCE_ROW_COLUMN)
        .with_row_index(_SHUFFLE_RANK_COLUMN)
        .with_columns(
            (pl.col(_SHUFFLE_RANK_COLUMN) % len(output_paths)).alias(_SHARD_COLUMN)
        )
        .drop(_SOURCE_ROW_COLUMN, _SHUFFLE_HASH_COLUMN, _SHUFFLE_RANK_COLUMN)
    )

    with TemporaryDirectory(prefix=".bounded-shuffle-", dir=output_parent) as temp:
        temporary = Path(temp)

        def output_file(args: pl.FileProviderArgs) -> str:
            assert args.index_in_partition == 0
            assert args.partition_keys.shape == (1, 1)
            shard_index = int(args.partition_keys.item())
            assert 0 <= shard_index < len(output_paths)
            return f"shard_{shard_index:04d}.jsonl"

        shuffled.sink_ndjson(
            pl.PartitionBy(
                temporary,
                key=_SHARD_COLUMN,
                include_key=False,
                approximate_bytes_per_file=None,
                file_path_provider=output_file,
            ),
            mkdir=True,
            engine="streaming",
        )

        for shard_index, output_path in enumerate(output_paths):
            temporary_path = temporary / f"shard_{shard_index:04d}.jsonl"
            if not temporary_path.exists():
                temporary_path.touch()
            temporary_path.replace(output_path)
