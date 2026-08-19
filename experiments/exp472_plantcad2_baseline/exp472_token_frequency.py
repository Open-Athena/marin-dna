"""Count every token ID in the completed CoreWeave PlantCAD cache."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import numpy as np
from levanter.store.cache import CacheLedger, ShardedCacheLayout
from levanter.store.tree_store import TreeStore
from rigging.filesystem.s3_compat import configure_coreweave_s3

if __package__:
    from .exp472_tokenize import (
        COREWEAVE_MARINDNA_PREFIX,
        CORPUS_NAME,
        EXPECTED_SPLITS,
        EXPECTED_TOKEN_IDS,
        VOCAB_SIZE,
    )
else:
    from exp472_tokenize import (
        COREWEAVE_MARINDNA_PREFIX,
        CORPUS_NAME,
        EXPECTED_SPLITS,
        EXPECTED_TOKEN_IDS,
        VOCAB_SIZE,
    )

logger = logging.getLogger(__name__)

CACHE_PATH = f"{COREWEAVE_MARINDNA_PREFIX}/tokenized/plantcad/{CORPUS_NAME}"
TOKEN_NAMES = {token_id: token for token, token_id in EXPECTED_TOKEN_IDS.items()}
EXEMPLAR = {"input_ids": np.array([], dtype=np.int64)}


def count_shard(
    split_path: str,
    shard_name: str,
    token_count: int,
    read_chunk_tokens: int,
) -> np.ndarray:
    shard_path = ShardedCacheLayout.parse(split_path).shard(shard_name)
    store = TreeStore.open(EXEMPLAR, shard_path, mode="r", cache_metadata=True)
    data = store.tree["input_ids"].data
    counts = np.zeros(VOCAB_SIZE, dtype=np.int64)

    for start in range(0, token_count, read_chunk_tokens):
        stop = min(start + read_chunk_tokens, token_count)
        values = np.asarray(data[start:stop].read().result())
        if values.size != stop - start:
            raise ValueError(
                f"short read from {split_path}/{shard_name}: "
                f"received {values.size}, expected {stop - start}"
            )
        if values.size and (values.min() < 0 or values.max() >= VOCAB_SIZE):
            raise ValueError(
                f"out-of-vocabulary token in {split_path}/{shard_name}: "
                f"range=[{values.min()}, {values.max()}]"
            )
        counts += np.bincount(values, minlength=VOCAB_SIZE)

    if int(counts.sum()) != token_count:
        raise ValueError(
            f"count mismatch for {split_path}/{shard_name}: "
            f"observed {counts.sum()}, expected {token_count}"
        )
    return counts


def count_split(
    cache_path: str,
    split: str,
    workers: int,
    read_chunk_tokens: int,
) -> np.ndarray:
    split_path = f"{cache_path}/{split}"
    ledger = CacheLedger.load(split_path)
    if not ledger.is_finished:
        raise ValueError(f"cache is incomplete: {split_path}")

    tasks = []
    for shard_name in ledger.finished_shards:
        token_count = ledger.field_counts_by_shard.get(shard_name, {}).get("input_ids")
        if token_count is None:
            raise ValueError(f"missing input_ids count for {split_path}/{shard_name}")
        tasks.append((shard_name, int(token_count)))

    counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                count_shard,
                split_path,
                shard_name,
                token_count,
                read_chunk_tokens,
            ): shard_name
            for shard_name, token_count in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            shard_name = futures[future]
            shard_counts = future.result()
            counts += shard_counts
            logger.info(
                "%s: counted shard %s (%d/%d)",
                split,
                shard_name,
                completed,
                len(futures),
            )

    expected_tokens = int(ledger.field_counts["input_ids"])
    if int(counts.sum()) != expected_tokens:
        raise ValueError(
            f"count mismatch for {split_path}: observed {counts.sum()}, "
            f"expected {expected_tokens}"
        )
    return counts


def frequency_record(counts: np.ndarray) -> dict[str, object]:
    total = int(counts.sum())
    return {
        "total_tokens": total,
        "tokens": {
            str(token_id): {
                "token": TOKEN_NAMES[token_id],
                "count": int(count),
                "fraction": float(count / total),
            }
            for token_id, count in enumerate(counts)
        },
    }


@click.command(help=__doc__)
@click.option("--cache-path", default=CACHE_PATH, show_default=True)
@click.option("--workers", type=click.IntRange(min=1), default=8, show_default=True)
@click.option(
    "--read-chunk-tokens",
    type=click.IntRange(min=1),
    default=16 * 1024 * 1024,
    show_default=True,
)
def main(cache_path: str, workers: int, read_chunk_tokens: int) -> None:
    logging.basicConfig(level=logging.INFO)
    if not cache_path.startswith("s3://"):
        raise ValueError("this audit must read the completed CoreWeave S3 cache")
    configure_coreweave_s3()

    split_counts = {
        split: count_split(cache_path, split, workers, read_chunk_tokens)
        for split in EXPECTED_SPLITS
    }
    total_counts = sum(split_counts.values(), start=np.zeros(VOCAB_SIZE, np.int64))
    result = {
        "cache_path": cache_path,
        "splits": {
            split: frequency_record(counts) for split, counts in split_counts.items()
        },
        "all_splits": frequency_record(total_counts),
    }
    print("TOKEN_FREQUENCY_JSON=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
