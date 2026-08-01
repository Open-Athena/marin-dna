"""Commit-pinned, FWD/RC-balanced five-source stream for experiment 426."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Literal

from datasets import IterableDataset, interleave_datasets, load_dataset
from marin_dna.data.dna import reverse_complement
from transformers import AutoTokenizer, PreTrainedTokenizerBase

WINDOW_BP = 255
CONTEXT_TOKENS = 256
N_SOURCES = 5
ORIENTATIONS = ("forward", "reverse_complement")
N_STREAMS = N_SOURCES * len(ORIENTATIONS)
DNA_IUPAC = frozenset("ACGTNRYSWKMBDHVacgtnryswkmbdhv")
GENOMES_V5_COLUMNS = ("id", "seq")
ZOONOMIA_V1_COLUMNS = (
    "query_name",
    "species",
    "t_chrom",
    "t_start",
    "t_end",
    "t_strand",
    "t_src_size",
    "sequence",
    "augmentation",
)

TOKENIZER_ID = "marin-dna/tokenizer-char-bos"
TOKENIZER_REVISION = "a73e9d9ee636f722b4c378703c9e2997857809b2"


@dataclass(frozen=True)
class SourceSpec:
    """One immutable component of m5.1's equal training mixture."""

    name: str
    dataset_id: str
    revision: str
    text_key: str
    id_key: str


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "cds",
        "marin-dna/genomes-v5-genome_set-animals-intervals-v5_255_128",
        "ffe3e78c99868077c65ad6568e1445d80e480794",
        "seq",
        "id",
    ),
    SourceSpec(
        "upstream",
        "marin-dna/genomes-v5-genome_set-animals-intervals-v1_255_128",
        "d93209847b02a0c9be5c03591a0a5e56ee09c35d",
        "seq",
        "id",
    ),
    SourceSpec(
        "downstream",
        "marin-dna/genomes-v5-genome_set-animals-intervals-v15_255_128",
        "b009afaab756937d75b8da3b1271ad8f0cec0b4d",
        "seq",
        "id",
    ),
    SourceSpec(
        "ncrna_exon",
        "marin-dna/zoonomia-v1-v3_ncrna_exon",
        "3e48d9ae7c604b99ccfc8bd07e391b960c1ea21a",
        "sequence",
        "query_name",
    ),
    SourceSpec(
        "ccre_non_promoter",
        "marin-dna/zoonomia-v1-v3_ccre_non_promoter",
        "862485aa18eed53a53e693ba4c2eb45e0afc5087",
        "sequence",
        "query_name",
    ),
)


def load_pinned_tokenizer() -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        revision=TOKENIZER_REVISION,
    )
    assert tokenizer.bos_token_id == 2
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    encoded = tokenizer(
        "A" * WINDOW_BP,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    assert len(encoded) == CONTEXT_TOKENS and encoded[0] == tokenizer.bos_token_id
    return tokenizer


def normalize_and_tokenize_row(
    row: Mapping[str, Any],
    *,
    source: SourceSpec,
    orientation: Literal["forward", "reverse_complement"],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    sequence = row[source.text_key]
    record_id = row[source.id_key]
    assert isinstance(sequence, str) and isinstance(record_id, str)
    assert len(sequence) == WINDOW_BP, (source.name, len(sequence))
    assert not set(sequence) - DNA_IUPAC, (source.name, sorted(set(sequence)))
    if orientation == "reverse_complement":
        sequence = reverse_complement(sequence)
    else:
        assert orientation == "forward"
    input_ids = tokenizer(
        sequence,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    assert isinstance(input_ids, list) and len(input_ids) == CONTEXT_TOKENS
    assert input_ids[0] == tokenizer.bos_token_id
    assert tokenizer.bos_token_id not in input_ids[1:]
    assert tokenizer.pad_token_id not in input_ids
    return {
        "input_ids": input_ids,
        "source": source.name,
        "orientation": orientation,
        "record_id": record_id,
        "unknown_tokens": sum(token == tokenizer.unk_token_id for token in input_ids),
    }


def tokenize_source(
    dataset: IterableDataset,
    source: SourceSpec,
    orientation: Literal["forward", "reverse_complement"],
    tokenizer: PreTrainedTokenizerBase,
) -> IterableDataset:
    columns = dataset.column_names
    if columns is None:
        columns = (
            GENOMES_V5_COLUMNS if source.text_key == "seq" else ZOONOMIA_V1_COLUMNS
        )
    assert source.text_key in columns and source.id_key in columns
    return dataset.map(
        partial(
            normalize_and_tokenize_row,
            source=source,
            orientation=orientation,
            tokenizer=tokenizer,
        ),
        remove_columns=list(columns),
    )


def interleave_source_orientations(
    datasets: Sequence[IterableDataset],
) -> IterableDataset:
    """Round-robin in source-major, FWD-then-RC order with no mixing."""

    assert len(datasets) == N_STREAMS
    return interleave_datasets(list(datasets), stopping_strategy="first_exhausted")


def build_balanced_dataset(
    tokenizer: PreTrainedTokenizerBase,
    *,
    skip_per_stream: int = 0,
    sources: Sequence[SourceSpec] = SOURCES,
) -> IterableDataset:
    """Load ten pinned streams, skip equally, tokenize, and round-robin them."""

    assert skip_per_stream >= 0
    assert len(sources) == N_SOURCES
    assert len({source.name for source in sources}) == N_SOURCES
    streams: list[IterableDataset] = []
    for source in sources:
        for orientation in ORIENTATIONS:
            raw = load_dataset(
                source.dataset_id,
                split="train",
                streaming=True,
                revision=source.revision,
            )
            assert isinstance(raw, IterableDataset)
            if skip_per_stream:
                raw = raw.skip(skip_per_stream)
            streams.append(tokenize_source(raw, source, orientation, tokenizer))
    return interleave_source_orientations(streams)


def provenance_manifest() -> dict[str, Any]:
    return {
        "tokenizer": {"id": TOKENIZER_ID, "revision": TOKENIZER_REVISION},
        "sources": [asdict(source) for source in SOURCES],
        "orientations": list(ORIENTATIONS),
        "stream_order": [
            {"source": source.name, "orientation": orientation}
            for source in SOURCES
            for orientation in ORIENTATIONS
        ],
    }
