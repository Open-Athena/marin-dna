"""Prepare and tokenize PlantCAD's 8,192-base angiosperm corpus on CoreWeave.

The pinned Hugging Face release contains Arrow IPC shards, which Marin cannot
tokenize directly. The prepare phase streams each split to compact,
sequence-only Parquet files on CoreWeave S3. Tokenizer workers then read only
that S3 copy and build a Levanter cache at ``TOKENIZED_PREFIX``.

All three upstream splits remain distinct. The training entrypoints consume
only train and validation; the separately tokenized test split stays held out.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import click
import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from huggingface_hub import HfFileSystem
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.store.cache import CacheLedger, ShardedCacheLayout
from levanter.tokenizers import TokenizerBackend, load_tokenizer
from marin.processing.tokenize._core import (
    drop_sidecars,
    glob_with_sizes,
    parquet_window_hint,
    tokenize_pipeline,
)
from marin.processing.tokenize.cache_stats import read_tokenized_cache_stats
from marin.processing.tokenize.store_builder import (
    build_from_datasets,
    write_stats_json,
)
from pyarrow import ipc
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.dataset import Dataset, FileEntry
from zephyr.execution import ZephyrContext
from zephyr.readers import load_file

logger = logging.getLogger(__name__)

DATASET_REPO = "plantcad/Angiosperm_65_genomes_8192bp"
DATASET_REVISION = "4a444fff5520b992aa978d92a5af509a81977098"
TOKENIZER = "kuleshov-group/PlantCAD2-Small-l24-d0768"
TOKENIZER_REVISION = "f756c255cb76e9f538c3acec04acf4214ed03fb3"
TEXT_KEY = "seq"
SEQ_LEN = 8_192
VOCAB_SIZE = 7

MARINDNA_PREFIX = "s3://marin-us-east-02a/MarinDNA"
PREPARED_PREFIX = f"{MARINDNA_PREFIX}/data/plantcad/{DATASET_REPO.rsplit('/', 1)[1]}"
TOKENIZED_PREFIX = (
    f"{MARINDNA_PREFIX}/tokenized/plantcad/{DATASET_REPO.rsplit('/', 1)[1]}"
)
MANIFEST_PATH = f"{PREPARED_PREFIX}/manifest.json"
SMOKE_PREFIX = f"{MARINDNA_PREFIX}/tmp/exp472/tokenization-smoke"

COPY_RETRIES = 5
EXPECTED_TOKEN_IDS = {
    "[PAD]": 0,
    "[MASK]": 1,
    "[UNK]": 2,
    "a": 3,
    "c": 4,
    "g": 5,
    "t": 6,
}
EXPECTED_SPLITS = {
    "train": {"files": 44, "bytes": 21_750_241_304, "rows": 2_638_656},
    "validation": {"files": 6, "bytes": 2_718_781_128, "rows": 329_832},
    "test": {"files": 6, "bytes": 2_718_782_552, "rows": 329_832},
}
WORKER_RESOURCES = ResourceConfig(cpu=1, ram="8g", disk="8g", preemptible=False)
COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="6g", disk="16g", preemptible=False)


@dataclass(frozen=True)
class SourceShard:
    split: str
    path: str
    size: int
    xet_hash: str | None

    @property
    def filename(self) -> str:
        return self.path.rsplit("/", 1)[1]


@dataclass(frozen=True)
class PreparedShard:
    split: str
    source_path: str
    source_size: int
    destination_path: str
    destination_size: int
    rows: int


def _s3_path(url: str) -> str:
    return url.removeprefix("s3://")


def _source_catalog(hf_fs: HfFileSystem) -> dict[str, list[SourceShard]]:
    catalogs: dict[str, list[SourceShard]] = {}
    repo_root = f"datasets/{DATASET_REPO}@{DATASET_REVISION}"
    for split, expected in EXPECTED_SPLITS.items():
        entries = hf_fs.ls(f"{repo_root}/{split}", detail=True)
        shards = sorted(
            (
                SourceShard(
                    split=split,
                    path=entry["name"],
                    size=int(entry.get("size") or 0),
                    xet_hash=entry.get("xet_hash"),
                )
                for entry in entries
                if entry.get("type") == "file" and entry["name"].endswith(".arrow")
            ),
            key=lambda shard: shard.path,
        )
        observed = {"files": len(shards), "bytes": sum(shard.size for shard in shards)}
        required = {"files": expected["files"], "bytes": expected["bytes"]}
        if observed != required:
            raise ValueError(
                f"{split} source changed at {DATASET_REVISION}: "
                f"observed {observed}, expected {required}"
            )
        catalogs[split] = shards
    return catalogs


def _parquet_metadata(source: SourceShard) -> dict[bytes, bytes]:
    return {
        b"marindna.dataset": DATASET_REPO.encode(),
        b"marindna.dataset_revision": DATASET_REVISION.encode(),
        b"marindna.source_path": source.path.encode(),
        b"marindna.source_size": str(source.size).encode(),
        b"marindna.sequence_length": str(SEQ_LEN).encode(),
    }


def _prepared_shard(
    s3_fs: fsspec.AbstractFileSystem,
    source: SourceShard,
    destination_url: str,
) -> PreparedShard | None:
    destination_path = _s3_path(destination_url)
    if not s3_fs.exists(destination_path):
        return None
    try:
        with s3_fs.open(destination_path, "rb") as stream:
            parquet = pq.ParquetFile(stream)
            schema = parquet.schema_arrow
            if schema.names != [TEXT_KEY] or schema.field(TEXT_KEY).type != pa.string():
                return None
            metadata = schema.metadata or {}
            if any(
                metadata.get(key) != value
                for key, value in _parquet_metadata(source).items()
            ):
                return None
            rows = parquet.metadata.num_rows
        if rows <= 0:
            return None
        return PreparedShard(
            split=source.split,
            source_path=source.path,
            source_size=source.size,
            destination_path=destination_url,
            destination_size=int(s3_fs.info(destination_path).get("size") or 0),
            rows=rows,
        )
    except (OSError, ValueError, pa.ArrowException):
        return None


def _validate_sequences(batch: pa.RecordBatch, source_path: str) -> pa.RecordBatch:
    sequences = batch.column(batch.schema.get_field_index(TEXT_KEY))
    if sequences.null_count:
        raise ValueError(f"{source_path} contains null sequences")
    bounds = pc.min_max(pc.utf8_length(sequences)).as_py()
    if bounds != {"min": SEQ_LEN, "max": SEQ_LEN}:
        raise ValueError(
            f"{source_path} contains non-{SEQ_LEN}-base sequences: {bounds}"
        )
    return pa.record_batch([sequences], names=[TEXT_KEY])


def _convert_one(
    hf_fs: HfFileSystem,
    s3_fs: fsspec.AbstractFileSystem,
    source: SourceShard,
    destination_prefix: str,
    max_rows: int | None,
) -> PreparedShard:
    destination_url = (
        f"{destination_prefix.rstrip('/')}/{source.split}/"
        f"{source.filename.removesuffix('.arrow')}.parquet"
    )
    existing = _prepared_shard(s3_fs, source, destination_url)
    expected_rows = max_rows if max_rows is not None else None
    if existing is not None and (
        expected_rows is None or existing.rows == expected_rows
    ):
        return existing

    for attempt in range(1, COPY_RETRIES + 1):
        try:
            schema = pa.schema(
                [pa.field(TEXT_KEY, pa.string())], metadata=_parquet_metadata(source)
            )
            rows = 0
            with (
                hf_fs.open(source.path, "rb") as source_stream,
                s3_fs.open(_s3_path(destination_url), "wb") as destination_stream,
            ):
                reader = ipc.open_stream(source_stream)
                if TEXT_KEY not in reader.schema.names:
                    raise ValueError(f"{source.path} has no {TEXT_KEY!r} column")
                with pq.ParquetWriter(
                    destination_stream,
                    schema,
                    compression="zstd",
                    use_dictionary=False,
                ) as writer:
                    for batch in reader:
                        if max_rows is not None:
                            remaining = max_rows - rows
                            if remaining <= 0:
                                break
                            batch = batch.slice(0, min(batch.num_rows, remaining))
                        sequence_batch = _validate_sequences(batch, source.path)
                        writer.write_batch(sequence_batch)
                        rows += sequence_batch.num_rows
            if rows <= 0 or (max_rows is not None and rows != max_rows):
                raise ValueError(f"{source.path} produced {rows} rows")
            prepared = _prepared_shard(s3_fs, source, destination_url)
            if prepared is None or prepared.rows != rows:
                raise ValueError(f"failed to validate prepared shard {destination_url}")
            return prepared
        except Exception as exc:
            if attempt == COPY_RETRIES:
                raise RuntimeError(f"failed to prepare {source.path}") from exc
            time.sleep(min(2**attempt, 30))
    raise AssertionError("conversion retry loop did not return or raise")


def _write_manifest(path: str, prepared: dict[str, list[PreparedShard]]) -> None:
    manifest = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "dataset": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "tokenizer": TOKENIZER,
        "tokenizer_revision": TOKENIZER_REVISION,
        "sequence_length": SEQ_LEN,
        "splits": {
            split: {
                "rows": sum(shard.rows for shard in shards),
                "files": [asdict(shard) for shard in shards],
            }
            for split, shards in prepared.items()
        },
    }
    fs, fs_path = fsspec.core.url_to_fs(path)
    with fs.open(fs_path, "wt") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def prepare_sources(
    conversion_workers: int,
    *,
    smoke_test: bool,
    smoke_records: int,
) -> tuple[str, str, dict[str, int]]:
    hf_fs = HfFileSystem(token=os.environ.get("HF_TOKEN") or False)
    s3_fs = fsspec.filesystem("s3")
    catalogs = _source_catalog(hf_fs)
    smoke_run_prefix = f"{SMOKE_PREFIX}/{smoke_records}-rows"
    destination_prefix = f"{smoke_run_prefix}/source" if smoke_test else PREPARED_PREFIX
    manifest_path = f"{smoke_run_prefix}/manifest.json" if smoke_test else MANIFEST_PATH
    selected = {
        split: shards[:1] if smoke_test else shards
        for split, shards in catalogs.items()
    }
    max_rows = smoke_records if smoke_test else None
    prepared: dict[str, list[PreparedShard]] = {split: [] for split in selected}
    futures = {}
    with ThreadPoolExecutor(
        max_workers=conversion_workers, thread_name_prefix="plantcad-prepare"
    ) as pool:
        for split, shards in selected.items():
            for source in shards:
                future = pool.submit(
                    _convert_one,
                    hf_fs,
                    s3_fs,
                    source,
                    destination_prefix,
                    max_rows,
                )
                futures[future] = split
        for completed, future in enumerate(as_completed(futures), start=1):
            prepared[futures[future]].append(future.result())
            logger.info("prepared %d/%d Arrow shards", completed, len(futures))

    expected_rows = {
        split: smoke_records if smoke_test else int(EXPECTED_SPLITS[split]["rows"])
        for split in selected
    }
    for split, shards in prepared.items():
        shards.sort(key=lambda shard: shard.source_path)
        rows = sum(shard.rows for shard in shards)
        if rows != expected_rows[split]:
            raise ValueError(
                f"prepared {split} has {rows} rows; expected {expected_rows[split]}"
            )
    _write_manifest(manifest_path, prepared)
    logger.info("source preparation complete: %s", manifest_path)
    cache_path = f"{smoke_run_prefix}/tokenized" if smoke_test else TOKENIZED_PREFIX
    return destination_prefix, cache_path, expected_rows


def validate_prepared_sources(
    *, smoke_test: bool, smoke_records: int
) -> tuple[str, str, dict[str, int]]:
    hf_fs = HfFileSystem(token=os.environ.get("HF_TOKEN") or False)
    s3_fs = fsspec.filesystem("s3")
    catalogs = _source_catalog(hf_fs)
    smoke_run_prefix = f"{SMOKE_PREFIX}/{smoke_records}-rows"
    destination_prefix = f"{smoke_run_prefix}/source" if smoke_test else PREPARED_PREFIX
    selected = {
        split: shards[:1] if smoke_test else shards
        for split, shards in catalogs.items()
    }
    expected_rows = {
        split: smoke_records if smoke_test else int(EXPECTED_SPLITS[split]["rows"])
        for split in selected
    }
    for split, shards in selected.items():
        rows = 0
        for source in shards:
            destination_url = (
                f"{destination_prefix}/{split}/"
                f"{source.filename.removesuffix('.arrow')}.parquet"
            )
            prepared = _prepared_shard(s3_fs, source, destination_url)
            if prepared is None:
                raise ValueError(f"missing or invalid prepared shard {destination_url}")
            rows += prepared.rows
        if rows != expected_rows[split]:
            raise ValueError(
                f"prepared {split} has {rows} rows; expected {expected_rows[split]}"
            )
    cache_path = f"{smoke_run_prefix}/tokenized" if smoke_test else TOKENIZED_PREFIX
    return destination_prefix, cache_path, expected_rows


def _load_validated_tokenizer():
    tokenizer = load_tokenizer(TOKENIZER, backend="hf")
    observed = {token: tokenizer.get_vocab().get(token) for token in EXPECTED_TOKEN_IDS}
    if (
        len(tokenizer) != VOCAB_SIZE
        or observed != EXPECTED_TOKEN_IDS
        or tokenizer.bos_token_id is not None
        or tokenizer.eos_token_id is not None
        or tokenizer.encode("ACGT", add_special_tokens=True) != [3, 4, 5, 6]
        or len(tokenizer.encode("A" * SEQ_LEN, add_special_tokens=True)) != SEQ_LEN
    ):
        raise ValueError(
            f"PlantCAD2 tokenizer contract changed: vocab_size={len(tokenizer)}, "
            f"token_ids={observed}"
        )
    return tokenizer


def _validate_tokenized_record(record: dict) -> dict:
    input_ids = record.get("input_ids")
    if input_ids is None or len(input_ids) != SEQ_LEN:
        raise ValueError(
            f"tokenized sequence has {0 if input_ids is None else len(input_ids)} tokens"
        )
    if min(input_ids) < 0 or max(input_ids) >= VOCAB_SIZE:
        raise ValueError("tokenized sequence contains an out-of-range token ID")
    if (
        EXPECTED_TOKEN_IDS["[PAD]"] in input_ids
        or EXPECTED_TOKEN_IDS["[MASK]"] in input_ids
    ):
        raise ValueError("tokenized sequence contains padding or mask tokens")
    return record


def _cache_complete(cache_path: str, split: str, expected_rows: int) -> bool:
    split_path = prefix_join(cache_path, split)
    ledger_path = ShardedCacheLayout.parse(split_path).ledger
    if not StoragePath(ledger_path).exists():
        return False
    try:
        stats = read_tokenized_cache_stats(cache_path, split)
    except FileNotFoundError:
        ledger = CacheLedger.load(split_path)
        write_stats_json(split_path, ledger)
        stats = read_tokenized_cache_stats(cache_path, split)
    expected_tokens = expected_rows * SEQ_LEN
    if stats.total_elements != expected_rows or stats.total_tokens != expected_tokens:
        raise ValueError(
            f"invalid completed {split} cache: {stats}; expected "
            f"{expected_rows} rows and {expected_tokens} tokens"
        )
    return True


def _file_groups(pattern: str) -> list[list[str]]:
    files: list[FileEntry] = sorted(
        drop_sidecars(glob_with_sizes([pattern])), key=lambda entry: entry.path
    )
    if not files:
        raise ValueError(f"no prepared Parquet files matched {pattern}")
    return [[file.path] for file in files]


def tokenize_split(
    *,
    split: str,
    prepared_prefix: str,
    cache_path: str,
    expected_rows: int,
    max_workers: int,
) -> None:
    if _cache_complete(cache_path, split, expected_rows):
        logger.info("cache already complete, skipping: %s/%s", cache_path, split)
        return
    groups = _file_groups(f"{prepared_prefix}/{split}/*.parquet")
    dataset = (
        Dataset.from_list(groups).flat_map(lambda paths: paths).flat_map(load_file)
    )
    tokenized_dataset, batch_size = tokenize_pipeline(
        dataset,
        data_format=TextLmDatasetFormat(text_key=TEXT_KEY),
        sample_count=None,
        sample_parquet_path=parquet_window_hint(groups),
        levanter_batch_size=None,
    )
    tokenized_dataset = tokenized_dataset.map(_validate_tokenized_record)
    chunk_scope = cache_path.removeprefix(MARINDNA_PREFIX).strip("/").replace("/", "-")
    context = ZephyrContext(
        resources=WORKER_RESOURCES,
        coordinator_resources=COORDINATOR_RESOURCES,
        max_workers=min(max_workers, len(groups)),
        chunk_storage_prefix=f"{MARINDNA_PREFIX}/tmp/exp472/zephyr/{chunk_scope}/{split}",
        name=f"exp472-plantcad-tokenize-{split}",
    )
    context.put("tokenizer_name", TOKENIZER)
    context.put("tokenizer_backend", TokenizerBackend.HF)
    ledger = build_from_datasets(
        ctx=context,
        dataset=tokenized_dataset,
        output_path=prefix_join(cache_path, split),
        batch_size=batch_size,
    )
    expected_tokens = expected_rows * SEQ_LEN
    if (
        ledger.total_num_rows != expected_rows
        or ledger.field_counts.get("input_ids") != expected_tokens
    ):
        raise ValueError(
            f"invalid {split} cache ledger: {ledger.total_num_rows} rows and "
            f"{ledger.field_counts.get('input_ids')} tokens; expected "
            f"{expected_rows} rows and {expected_tokens} tokens"
        )
    stats_path, _ = write_stats_json(prefix_join(cache_path, split), ledger)
    logger.info("%s tokenization complete: %s", split, stats_path)


@click.command(help=__doc__)
@click.option(
    "--phase",
    type=click.Choice(("all", "prepare", "tokenize"), case_sensitive=False),
    default="all",
    show_default=True,
)
@click.option(
    "--conversion-workers", type=click.IntRange(min=1), default=4, show_default=True
)
@click.option(
    "--tokenize-workers", type=click.IntRange(min=1), default=64, show_default=True
)
@click.option(
    "--smoke-test",
    is_flag=True,
    help="Use 64 rows per split under an isolated temporary prefix.",
)
@click.option(
    "--smoke-records",
    type=click.IntRange(min=1, max=1_000),
    default=64,
    show_default=True,
)
def main(
    phase: str,
    conversion_workers: int,
    tokenize_workers: int,
    smoke_test: bool,
    smoke_records: int,
) -> None:
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    if phase in {"all", "prepare"}:
        prepared_prefix, cache_path, expected_rows = prepare_sources(
            conversion_workers,
            smoke_test=smoke_test,
            smoke_records=smoke_records,
        )
    else:
        prepared_prefix, cache_path, expected_rows = validate_prepared_sources(
            smoke_test=smoke_test,
            smoke_records=smoke_records,
        )
    if phase in {"all", "tokenize"}:
        _load_validated_tokenizer()
        for split in EXPECTED_SPLITS:
            tokenize_split(
                split=split,
                prepared_prefix=prepared_prefix,
                cache_path=cache_path,
                expected_rows=expected_rows[split],
                max_workers=tokenize_workers,
            )


if __name__ == "__main__":
    main()
