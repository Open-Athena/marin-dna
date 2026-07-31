"""Pinned five-way m5.1 data adapter for the issue #288 SAE experiment.

The m5.1 continuation used five equally weighted sequence sources, but three
store DNA in ``seq`` while two use ``sequence`` and richer projection
metadata.  SAELens expects a pretokenized ``input_ids`` column.  This module
bridges those boundaries without adding SAE dependencies to MarinDNA core.

The source repositories were globally shuffled before sharding, so this
adapter deliberately does not apply a small-buffer streaming shuffle.  It
round-robins the five pinned streams, giving exact source balance for every
complete group of five windows and deterministic replay.

Run a live two-window-per-source audit from the repository root::

    uv run python scripts/issue288/five_way_data.py --windows-per-source 2
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

from datasets import IterableDataset, interleave_datasets, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

WINDOW_BP = 255
CONTEXT_TOKENS = 256
N_SOURCES = 5
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
    """One commit-pinned component of m5.1's equal training mixture."""

    name: str
    dataset_id: str
    revision: str
    text_key: str
    schema: str


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        name="cds",
        dataset_id=("marin-dna/genomes-v5-genome_set-animals-intervals-v5_255_128"),
        revision="ffe3e78c99868077c65ad6568e1445d80e480794",
        text_key="seq",
        schema="genomes_v5",
    ),
    SourceSpec(
        name="upstream",
        dataset_id=("marin-dna/genomes-v5-genome_set-animals-intervals-v1_255_128"),
        revision="d93209847b02a0c9be5c03591a0a5e56ee09c35d",
        text_key="seq",
        schema="genomes_v5",
    ),
    SourceSpec(
        name="downstream",
        dataset_id=("marin-dna/genomes-v5-genome_set-animals-intervals-v15_255_128"),
        revision="b009afaab756937d75b8da3b1271ad8f0cec0b4d",
        text_key="seq",
        schema="genomes_v5",
    ),
    SourceSpec(
        name="ncrna_exon",
        dataset_id="marin-dna/zoonomia-v1-v3_ncrna_exon",
        revision="3e48d9ae7c604b99ccfc8bd07e391b960c1ea21a",
        text_key="sequence",
        schema="zoonomia_v1",
    ),
    SourceSpec(
        name="ccre_non_promoter",
        dataset_id="marin-dna/zoonomia-v1-v3_ccre_non_promoter",
        revision="862485aa18eed53a53e693ba4c2eb45e0afc5087",
        text_key="sequence",
        schema="zoonomia_v1",
    ),
)


@dataclass(frozen=True)
class BalancedBudget:
    """Smallest exactly five-way-balanced window budget above a target."""

    requested_activations: int
    windows_per_source: int
    total_windows: int
    actual_activations: int


def balanced_budget(requested_activations: int) -> BalancedBudget:
    """Round an activation target up to whole windows and equal source counts."""

    assert requested_activations > 0
    windows_per_source = math.ceil(requested_activations / (WINDOW_BP * N_SOURCES))
    total_windows = windows_per_source * N_SOURCES
    actual_activations = total_windows * WINDOW_BP
    assert actual_activations >= requested_activations
    assert total_windows % N_SOURCES == 0
    return BalancedBudget(
        requested_activations=requested_activations,
        windows_per_source=windows_per_source,
        total_windows=total_windows,
        actual_activations=actual_activations,
    )


def _opposite_strand(strand: str) -> str:
    assert strand in {"+", "-"}
    return "-" if strand == "+" else "+"


def _parse_genomes_v5_row(row: Mapping[str, Any]) -> dict[str, Any]:
    record_id = row["id"]
    assert isinstance(record_id, str)
    coordinate_id, augmentation = record_id.rsplit("_", maxsplit=1)
    chrom, interval = coordinate_id.rsplit(":", maxsplit=1)
    start_text, end_text = interval.split("-", maxsplit=1)
    start = int(start_text)
    end = int(end_text)
    assert augmentation in {"+", "-"}
    return {
        "record_id": record_id,
        "species": "",
        "chrom": chrom,
        "start": start,
        "end": end,
        "source_strand": "",
        "augmentation": augmentation,
        "sequence_strand": augmentation,
        "source_size": -1,
    }


def _parse_zoonomia_v1_row(row: Mapping[str, Any]) -> dict[str, Any]:
    source_strand = row["t_strand"]
    augmentation = row["augmentation"]
    assert source_strand in {"+", "-"}
    assert augmentation in {"+", "-"}
    sequence_strand = (
        source_strand if augmentation == "+" else _opposite_strand(source_strand)
    )
    start = int(row["t_start"])
    end = int(row["t_end"])
    source_size = int(row["t_src_size"])
    query_name = row["query_name"]
    species = row["species"]
    chrom = row["t_chrom"]
    assert all(isinstance(value, str) for value in (query_name, species, chrom))
    record_id = (
        f"{query_name}:{species}:{chrom}:{start}-{end}_{source_strand}:{augmentation}"
    )
    return {
        "record_id": record_id,
        "species": species,
        "chrom": chrom,
        "start": start,
        "end": end,
        "source_strand": source_strand,
        "augmentation": augmentation,
        "sequence_strand": sequence_strand,
        "source_size": source_size,
    }


def reference_position(example: Mapping[str, Any], nucleotide_offset: int) -> int:
    """Map a 0-based input nucleotide offset back to a reference coordinate."""

    assert 0 <= nucleotide_offset < WINDOW_BP
    start = int(example["start"])
    end = int(example["end"])
    assert end - start == WINDOW_BP
    sequence_strand = example["sequence_strand"]
    assert sequence_strand in {"+", "-"}
    if sequence_strand == "+":
        position = start + nucleotide_offset
    else:
        position = end - 1 - nucleotide_offset
    assert start <= position < end
    return position


def normalize_and_tokenize_row(
    row: Mapping[str, Any],
    *,
    source: SourceSpec,
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    """Normalize one source row and enforce the m5.1 token/coordinate contract."""

    assert source.text_key in row, (source.name, source.text_key, sorted(row))
    sequence = row[source.text_key]
    assert isinstance(sequence, str)
    assert len(sequence) == WINDOW_BP, (source.name, len(sequence))
    unexpected = set(sequence) - DNA_IUPAC
    assert not unexpected, (source.name, sorted(unexpected))

    if source.schema == "genomes_v5":
        metadata = _parse_genomes_v5_row(row)
    elif source.schema == "zoonomia_v1":
        metadata = _parse_zoonomia_v1_row(row)
    else:
        raise AssertionError(f"unsupported schema {source.schema!r}")

    start = int(metadata["start"])
    end = int(metadata["end"])
    source_size = int(metadata["source_size"])
    assert start >= 0
    assert end - start == WINDOW_BP
    if source_size >= 0:
        assert end <= source_size

    encoded = tokenizer(
        sequence,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = encoded["input_ids"]
    assert isinstance(input_ids, list)
    assert len(input_ids) == CONTEXT_TOKENS, (source.name, len(input_ids))
    bos_token_id = tokenizer.bos_token_id
    pad_token_id = tokenizer.pad_token_id
    assert bos_token_id is not None
    assert input_ids[0] == bos_token_id
    assert bos_token_id not in input_ids[1:]
    if pad_token_id is not None:
        assert pad_token_id not in input_ids

    unk_token_id = tokenizer.unk_token_id
    unknown_tokens = (
        0 if unk_token_id is None else sum(token == unk_token_id for token in input_ids)
    )
    normalized = {
        "input_ids": input_ids,
        "source": source.name,
        "sequence": sequence,
        "unknown_tokens": unknown_tokens,
        "lowercase_bases": sum(base.islower() for base in sequence),
        **metadata,
    }
    assert reference_position(normalized, 0) in {start, end - 1}
    assert reference_position(normalized, WINDOW_BP - 1) in {start, end - 1}
    return normalized


def tokenize_source(
    dataset: IterableDataset,
    source: SourceSpec,
    tokenizer: PreTrainedTokenizerBase,
) -> IterableDataset:
    """Convert one raw stream to the common pretokenized/provenance schema."""

    columns = dataset.column_names
    if columns is None:
        if source.schema == "genomes_v5":
            columns = list(GENOMES_V5_COLUMNS)
        elif source.schema == "zoonomia_v1":
            columns = list(ZOONOMIA_V1_COLUMNS)
        else:
            raise AssertionError(f"unsupported schema {source.schema!r}")
    assert source.text_key in columns, (source.name, source.text_key, columns)
    return dataset.map(
        partial(normalize_and_tokenize_row, source=source, tokenizer=tokenizer),
        remove_columns=list(columns),
    )


def interleave_five_sources(datasets: Sequence[IterableDataset]) -> IterableDataset:
    """Round-robin five normalized streams with exact deterministic balance."""

    assert len(datasets) == N_SOURCES
    return interleave_datasets(list(datasets), stopping_strategy="first_exhausted")


def build_five_way_dataset(
    tokenizer: PreTrainedTokenizerBase,
    sources: Sequence[SourceSpec] = SOURCES,
) -> IterableDataset:
    """Load, pin, normalize, tokenize, and round-robin the m5.1 sources."""

    assert len(sources) == N_SOURCES
    assert len({source.name for source in sources}) == N_SOURCES
    streams = []
    for source in sources:
        raw = load_dataset(
            source.dataset_id,
            split="train",
            streaming=True,
            revision=source.revision,
        )
        assert isinstance(raw, IterableDataset)
        streams.append(tokenize_source(raw, source, tokenizer))
    return interleave_five_sources(streams)


def audit_live_dataset(windows_per_source: int) -> dict[str, Any]:
    """Read a small balanced prefix and summarize its enforced invariants."""

    assert windows_per_source > 0
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        revision=TOKENIZER_REVISION,
    )
    assert tokenizer.bos_token_id == 2
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1

    dataset = build_five_way_dataset(tokenizer)
    expected_order = [source.name for source in SOURCES]
    source_counts: Counter[str] = Counter()
    unknown_tokens: Counter[str] = Counter()
    lowercase_bases: Counter[str] = Counter()
    first_records: dict[str, str] = {}
    total_windows = windows_per_source * N_SOURCES
    for index, row in enumerate(dataset.take(total_windows)):
        source = row["source"]
        assert source == expected_order[index % N_SOURCES]
        assert len(row["input_ids"]) == CONTEXT_TOKENS
        source_counts[source] += 1
        unknown_tokens[source] += int(row["unknown_tokens"])
        lowercase_bases[source] += int(row["lowercase_bases"])
        first_records.setdefault(source, row["record_id"])

    assert source_counts == Counter(
        {source.name: windows_per_source for source in SOURCES}
    )
    budget_1m = balanced_budget(1_000_000)
    budget_5m = balanced_budget(5_000_000)
    return {
        "tokenizer": {
            "dataset_id": TOKENIZER_ID,
            "revision": TOKENIZER_REVISION,
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token_id": tokenizer.unk_token_id,
        },
        "sources": [asdict(source) for source in SOURCES],
        "audit": {
            "source_counts": dict(source_counts),
            "total_windows": total_windows,
            "nucleotide_activations": total_windows * WINDOW_BP,
            "unknown_tokens": dict(unknown_tokens),
            "lowercase_bases": dict(lowercase_bases),
            "first_records": first_records,
        },
        "balanced_budgets": {
            "1m": asdict(budget_1m),
            "5m": asdict(budget_5m),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows-per-source",
        type=int,
        default=2,
        help="number of live rows to audit from each of the five streams",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit_live_dataset(args.windows_per_source), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
