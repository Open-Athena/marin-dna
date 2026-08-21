"""Per-token likelihood dynamics for the m1 to m1.3 lineage (issue #489).

The module is task-specific so the versioned #489 artifacts do not change the
producer of any existing evals_v2 output.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Literal, TextIO
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.data.dna import NUCLEOTIDES
from marin_dna_evals.model.runner import run_inference, run_ll_clm
from marin_dna_evals.model.scoring import _logits_to_logprobs
from marin_dna_evals.transforms import (
    _get_nucleotide_token_ids,
    _get_special_token_counts,
    transform_ll_clm,
)

ReferenceKind = Literal["twobit", "fasta"]

WINDOW_ID = re.compile(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
KMER_ORDER = 6
KMER_PSEUDOCOUNT = 0.5
ARTIFACT_SCHEMA_VERSION = "v1"


@contextmanager
def _open_fasta_file(
    path: str | Path,
    mode: Literal["rb", "rt"],
) -> Iterator[BinaryIO | TextIO]:
    """Open local or remote FASTA assets without scanning the FASTA."""
    path_string = str(path)
    if urlparse(path_string).scheme in ("", "file"):
        with open(path_string, mode) as handle:
            yield handle
        return

    import fsspec

    filesystem, _, paths = fsspec.get_fs_token_paths(path_string)
    assert len(paths) == 1
    if mode == "rt":
        with filesystem.open(paths[0], mode=mode) as handle:
            yield handle
        return
    with filesystem.open(
        paths[0],
        mode=mode,
        block_size=1 << 20,
        cache_type="none",
    ) as handle:
        yield handle


def _read_fai(path: str | Path) -> dict[str, tuple[int, int, int, int]]:
    """Read name -> (length, byte offset, line bases, line width)."""
    index: dict[str, tuple[int, int, int, int]] = {}
    with _open_fasta_file(f"{path}.fai", "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n\r").split("\t")
            assert len(fields) >= 5, f"malformed FASTA index row: {line!r}"
            name = fields[0]
            values = tuple(map(int, fields[1:5]))
            length, offset, line_bases, line_width = values
            assert length >= 0 and offset >= 0
            assert line_bases > 0 and line_width >= line_bases
            assert name not in index, f"duplicate FASTA index sequence {name!r}"
            index[name] = values
    assert index, f"empty FASTA index {path}.fai"
    return index


def _fasta_byte_offset(
    position: int,
    *,
    offset: int,
    line_bases: int,
    line_width: int,
) -> int:
    return offset + (position // line_bases) * line_width + position % line_bases


def _read_indexed_fasta_windows(
    windows: list[tuple[str, int, int]],
    path: str | Path,
) -> list[str]:
    """Query only requested 0-based half-open intervals using FASTA byte ranges."""
    index = _read_fai(path)
    output: list[str] = []
    with _open_fasta_file(path, "rb") as handle:
        for chrom, start, end in windows:
            assert chrom in index, f"{chrom} absent from {path}.fai"
            length, offset, line_bases, line_width = index[chrom]
            assert 0 <= start <= end <= length, (
                f"{chrom}:{start}-{end} outside FASTA length {length}"
            )
            byte_start = _fasta_byte_offset(
                start,
                offset=offset,
                line_bases=line_bases,
                line_width=line_width,
            )
            byte_end = _fasta_byte_offset(
                end,
                offset=offset,
                line_bases=line_bases,
                line_width=line_width,
            )
            handle.seek(byte_start)
            raw = handle.read(byte_end - byte_start)
            sequence = raw.replace(b"\n", b"").replace(b"\r", b"").decode(
                "ascii"
            )
            assert len(sequence) == end - start, (
                f"{chrom}:{start}-{end}: indexed FASTA query returned "
                f"{len(sequence)} bases"
            )
            output.append(sequence)
    return output

def parse_window_id(window_id: str, *, window_size: int) -> tuple[str, int, int]:
    """Parse a 0-based half-open validation-window identifier."""
    match = WINDOW_ID.fullmatch(str(window_id))
    assert match is not None, f"invalid window id {window_id!r}"
    chrom = match.group("chrom")
    start = int(match.group("start"))
    end = int(match.group("end"))
    assert start >= 0 and end - start == window_size, (
        f"{window_id}: expected a {window_size}-bp 0-based half-open interval"
    )
    return chrom, start, end


def validate_sequences(sequences: pd.DataFrame, *, window_size: int) -> None:
    """Validate the common five-probe source contract."""
    missing = {"id", "seq"} - set(sequences.columns)
    assert not missing, f"sequences missing columns {sorted(missing)}"
    assert len(sequences) > 0, "empty sequences frame"
    assert sequences["id"].is_unique, "window ids must be unique within a region"
    lengths = sequences["seq"].astype(str).str.len()
    assert (lengths == window_size).all(), (
        f"every seq must be window_size={window_size}; "
        f"got {sorted(lengths.unique())[:5]}"
    )
    for window_id in sequences["id"]:
        parse_window_id(str(window_id), window_size=window_size)


def _encode_sequence(sequence: str) -> np.ndarray:
    return np.fromiter(
        (BASE_TO_INT.get(base, -1) for base in sequence.upper()),
        dtype=np.int8,
        count=len(sequence),
    )


def _reverse_complement_encoded(encoded: np.ndarray) -> np.ndarray:
    output = encoded[::-1].copy()
    valid = output >= 0
    output[valid] = 3 - output[valid]
    output[~valid] = -1
    return output


def _context_index(context: np.ndarray) -> int:
    index = 0
    for value in context:
        index = index * 4 + int(value)
    return index


def _add_order6_counts(encoded: np.ndarray, counts: np.ndarray) -> None:
    for oriented in (encoded, _reverse_complement_encoded(encoded)):
        for target_pos in range(KMER_ORDER, len(oriented)):
            context = oriented[target_pos - KMER_ORDER : target_pos]
            target = int(oriented[target_pos])
            if target < 0 or (context < 0).any():
                continue
            counts[_context_index(context), target] += 1


def _score_order6(encoded: np.ndarray, counts: np.ndarray) -> np.ndarray:
    output = np.full(len(encoded), np.nan, dtype=np.float32)
    for target_pos in range(KMER_ORDER, len(encoded)):
        context = encoded[target_pos - KMER_ORDER : target_pos]
        target = int(encoded[target_pos])
        if target < 0 or (context < 0).any():
            continue
        row = counts[_context_index(context)]
        probability = (row[target] + KMER_PSEUDOCOUNT) / (
            row.sum() + 4 * KMER_PSEUDOCOUNT
        )
        output[target_pos] = -np.log(probability)
    return output


def leave_one_chrom_7mer_nll(
    sequences: list[str],
    chroms: list[str],
) -> list[np.ndarray]:
    """Return a strand-symmetric order-6 Markov control per target base."""
    assert len(sequences) == len(chroms) and sequences
    shape = (4**KMER_ORDER, 4)
    total = np.zeros(shape, dtype=np.int64)
    by_chrom: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(shape, dtype=np.int64)
    )
    encoded = [_encode_sequence(sequence) for sequence in sequences]
    for array, chrom in zip(encoded, chroms, strict=True):
        _add_order6_counts(array, total)
        _add_order6_counts(array, by_chrom[chrom])

    results: list[np.ndarray] = []
    for array, chrom in zip(encoded, chroms, strict=True):
        held_out = total - by_chrom[chrom]
        forward = _score_order6(array, held_out)
        reverse = _score_order6(_reverse_complement_encoded(array), held_out)[::-1]
        valid_forward = np.isfinite(forward)
        valid_reverse = np.isfinite(reverse)
        combined = np.full(len(array), np.nan, dtype=np.float32)
        both = valid_forward & valid_reverse
        combined[both] = (forward[both] + reverse[both]) / 2
        combined[valid_forward & ~valid_reverse] = forward[
            valid_forward & ~valid_reverse
        ]
        combined[valid_reverse & ~valid_forward] = reverse[
            valid_reverse & ~valid_forward
        ]
        results.append(combined)
    return results


def _read_reference_windows(
    windows: list[tuple[str, int, int]],
    *,
    reference_kind: ReferenceKind,
    reference_path: str | Path,
) -> list[str]:
    if reference_kind == "twobit":
        import py2bit

        two_bit = py2bit.open(str(reference_path), True)
        try:
            known = two_bit.chroms()
            output: list[str] = []
            for chrom, start, end in windows:
                assert chrom in known, f"{chrom} absent from {reference_path}"
                output.append(str(two_bit.sequence(chrom, start, end)))
            return output
        finally:
            two_bit.close()

    assert reference_kind == "fasta", reference_kind
    return _read_indexed_fasta_windows(windows, reference_path)


def build_window_metadata(
    sequences: pd.DataFrame,
    *,
    region: str,
    window_size: int,
    reference_kind: ReferenceKind,
    reference_path: str | Path,
    assembly: str,
    conservation_label_source: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join conservation case to the matching reference's soft-mask case.

    The returned frame stays one row per window with fixed-length list columns.
    The scoring rule expands it only after inference, which keeps the reference
    join reusable across checkpoints.
    """
    validate_sequences(sequences, window_size=window_size)
    windows = [
        parse_window_id(str(window_id), window_size=window_size)
        for window_id in sequences["id"]
    ]
    references = _read_reference_windows(
        windows,
        reference_kind=reference_kind,
        reference_path=reference_path,
    )
    source_sequences = sequences["seq"].astype(str).tolist()

    repeat_masks: list[np.ndarray] = []
    conserved_masks: list[np.ndarray] = []
    case_upper_masks: list[np.ndarray] = []
    ambiguous_masks: list[np.ndarray] = []
    window_gc: list[float] = []
    for (window_id, case_sequence), reference in zip(
        sequences[["id", "seq"]].itertuples(index=False),
        references,
        strict=True,
    ):
        assert len(reference) == window_size, f"{window_id}: reference length mismatch"
        assert reference.upper() == str(case_sequence).upper(), (
            f"assembly/coordinate mismatch at {window_id}"
        )
        reference_chars = np.asarray(list(reference))
        case_chars = np.asarray(list(str(case_sequence)))
        canonical = np.isin(np.char.upper(reference_chars), list("ACGT"))
        case_upper = np.char.isupper(case_chars)
        repeat_masks.append(np.char.islower(reference_chars) & canonical)
        conserved_masks.append(case_upper & canonical)
        case_upper_masks.append(case_upper)
        ambiguous_masks.append(~canonical)
        upper = np.char.upper(reference_chars)
        denominator = int(canonical.sum())
        window_gc.append(
            float(np.isin(upper, list("GC")).sum() / denominator)
            if denominator
            else float("nan")
        )

    chroms = [chrom for chrom, _, _ in windows]
    kmer7_nll = leave_one_chrom_7mer_nll(
        [sequence.upper() for sequence in source_sequences],
        chroms,
    )
    metadata = pd.DataFrame(
        {
            "row_index": np.arange(len(sequences), dtype=np.int32),
            "window_id": sequences["id"].astype(str).to_numpy(),
            "region": region,
            "chrom": chroms,
            "window_start": [start for _, start, _ in windows],
            "window_end": [end for _, _, end in windows],
            "sequence_upper": [sequence.upper() for sequence in source_sequences],
            "case_is_upper": case_upper_masks,
            "is_conserved": conserved_masks,
            "is_repeat": repeat_masks,
            "is_ambiguous": ambiguous_masks,
            "window_gc": np.asarray(window_gc, dtype=np.float32),
            "kmer7_nll": kmer7_nll,
        }
    )
    total_positions = len(metadata) * window_size
    manifest: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "region": region,
        "n_windows": len(metadata),
        "window_size": window_size,
        "n_positions": total_positions,
        "assembly": assembly,
        "coordinate_system": "0-based half-open",
        "reference_kind": reference_kind,
        "reference_path": str(reference_path),
        "repeat_label_source": f"{assembly}_soft_mask",
        "conservation_label_source": conservation_label_source,
        "conservation_missingness": (
            "lowercase_conflates_below_threshold_and_missing_alignment"
        ),
        "n_conserved": int(sum(mask.sum() for mask in conserved_masks)),
        "n_repeat": int(sum(mask.sum() for mask in repeat_masks)),
        "n_ambiguous": int(sum(mask.sum() for mask in ambiguous_masks)),
        "kmer_control": {
            "name": "chromosome-held-out strand-averaged order-6 Markov NLL",
            "pseudocount": KMER_PSEUDOCOUNT,
        },
    }
    return metadata, manifest


def compute_per_token_stats_clm(
    model: Any,
    input_ids: Tensor,
    is_upper: Tensor | None = None,
    *,
    nucleotide_token_ids: Tensor,
) -> Tensor:
    """Return full-vocabulary true-base NLL and A/C/G/T entropy per target."""
    if is_upper is not None:
        assert is_upper.shape == input_ids.shape
    logits = model(input_ids).logits
    logp_true = _logits_to_logprobs(logits, input_ids).float()
    nucleotide_ids = nucleotide_token_ids.to(device=logits.device)
    nucleotide_logits = logits[:, :-1].float().index_select(-1, nucleotide_ids)
    nucleotide_logp = torch.log_softmax(nucleotide_logits, dim=-1)
    entropy = -(nucleotide_logp.exp() * nucleotide_logp).sum(dim=-1)
    assert entropy.shape == logp_true.shape
    return torch.stack([-logp_true, entropy], dim=-1)


def _prediction_array(
    prediction: Any,
    *,
    n_windows: int,
    window_size: int,
) -> np.ndarray:
    values = np.asarray(prediction, dtype=np.float32)
    expected = (n_windows, window_size, 2)
    if values.size == int(np.prod(expected)) and values.shape != expected:
        values = values.reshape(expected)
    assert values.shape == expected, (
        f"per-token shape {values.shape}, expected {expected}"
    )
    assert np.isfinite(values).all(), "non-finite per-token prediction"
    assert (values[:, :, 0] >= 0).all(), "NLL must be non-negative"
    assert (values[:, :, 1] >= 0).all(), "entropy must be non-negative"
    return values


def aggregate_token_stats_by_case(
    stats: pd.DataFrame,
    sequences: pd.DataFrame,
) -> np.ndarray:
    """Reconstruct per-window case sums in the current LL-gap column order."""
    assert list(stats["window_id"]) == list(sequences["id"].astype(str)), (
        "score/source row order or ids differ"
    )
    nll = np.stack(stats["nll"].to_numpy()).astype(np.float64)
    characters = np.asarray([list(str(sequence)) for sequence in sequences["seq"]])
    assert characters.shape == nll.shape
    upper = np.char.isupper(characters)
    logp = -nll
    return np.column_stack(
        [
            np.where(upper, logp, 0.0).sum(axis=1),
            np.where(~upper, logp, 0.0).sum(axis=1),
            upper.sum(axis=1),
            (~upper).sum(axis=1),
        ]
    )


def _aggregate_parity_report(
    stats: pd.DataFrame,
    sequences: pd.DataFrame,
    aggregate_prediction: Any,
    *,
    atol: float = 0.01,
) -> dict[str, Any]:
    reconstructed = aggregate_token_stats_by_case(stats, sequences)
    aggregate = np.asarray(aggregate_prediction, dtype=np.float64)
    if aggregate.size == reconstructed.size and aggregate.shape != reconstructed.shape:
        aggregate = aggregate.reshape(reconstructed.shape)
    assert aggregate.shape == reconstructed.shape
    assert np.array_equal(aggregate[:, 2:].astype(np.int64), reconstructed[:, 2:])
    absolute = np.abs(aggregate[:, :2] - reconstructed[:, :2])
    max_abs = float(absolute.max(initial=0.0))
    assert max_abs <= atol, (
        f"per-token sums disagree with aggregate LL kernel: {max_abs} > {atol}"
    )
    return {
        "passed": True,
        "atol": atol,
        "max_abs_per_window_sum_diff": max_abs,
        "n_windows": len(reconstructed),
    }


def compute_hf_per_token_stats(
    checkpoint_path: str | Path,
    sequences: pd.DataFrame,
    *,
    window_size: int,
    batch_size: int,
    num_workers: int,
    torch_compile: bool,
    validate_aggregate: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score one forward orientation and optionally gate against aggregate LL."""
    validate_sequences(sequences, window_size=window_size)
    checkpoint_path = Path(checkpoint_path)
    tokenizer: Any = AutoTokenizer.from_pretrained(checkpoint_path)
    model: Any = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
    n_prefix, n_suffix = _get_special_token_counts(tokenizer)
    assert (n_prefix, n_suffix) == (1, 0), (
        "per-token genomic alignment requires one prepended BOS and no suffix; "
        f"got ({n_prefix}, {n_suffix})"
    )
    nucleotide_ids = _get_nucleotide_token_ids(tokenizer)
    nucleotide_token_ids = torch.tensor(
        [nucleotide_ids[nucleotide] for nucleotide in NUCLEOTIDES],
        dtype=torch.long,
    )
    dataset = Dataset.from_pandas(sequences[["seq"]], preserve_index=False)
    inference_kwargs = {
        "per_device_eval_batch_size": batch_size,
        "torch_compile": torch_compile,
        "bf16_full_eval": True,
        "dataloader_num_workers": num_workers,
        "remove_unused_columns": False,
    }
    prediction = run_inference(
        model,
        tokenizer,
        dataset,
        compute_fn=partial(
            compute_per_token_stats_clm,
            nucleotide_token_ids=nucleotide_token_ids,
        ),
        data_transform_fn=transform_ll_clm,
        data_transform_on_the_fly=True,
        inference_kwargs=inference_kwargs,
    )
    values = _prediction_array(
        prediction,
        n_windows=len(sequences),
        window_size=window_size,
    )
    stats = pd.DataFrame(
        {
            "window_id": sequences["id"].astype(str).to_numpy(),
            "nll": list(values[:, :, 0]),
            "entropy_4nuc": list(values[:, :, 1]),
        }
    )
    manifest: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "orientation": "fwd",
        "n_windows": len(stats),
        "window_size": window_size,
        "n_positions": len(stats) * window_size,
        "batch_size": batch_size,
        "torch_compile": torch_compile,
        "aggregate_gate": None,
    }
    if validate_aggregate:
        aggregate_prediction = run_ll_clm(
            model,
            tokenizer,
            dataset,
            data_transform_on_the_fly=True,
            inference_kwargs=inference_kwargs,
        )
        manifest["aggregate_gate"] = _aggregate_parity_report(
            stats,
            sequences,
            aggregate_prediction,
        )
    return stats, manifest


def _flatten_list_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    arrays = [np.asarray(value) for value in frame[column]]
    assert arrays, f"{column} is empty"
    width = len(arrays[0])
    assert all(len(array) == width for array in arrays), (
        f"{column} has inconsistent widths"
    )
    return np.concatenate(arrays)


def assemble_token_atoms(
    metadata: pd.DataFrame,
    stats: pd.DataFrame,
    *,
    checkpoint: str,
    checkpoint_order: int,
    stage: str,
    training_step: int,
    cumulative_tokens: int,
    assembly: str,
    conservation_label_source: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expand reusable window metadata and score lists to stable token rows."""
    assert list(metadata["window_id"]) == list(stats["window_id"]), (
        "metadata/score row order or ids differ"
    )
    n_windows = len(metadata)
    assert n_windows > 0
    widths = {len(np.asarray(value)) for value in stats["nll"]}
    assert len(widths) == 1
    window_size = widths.pop()
    n_positions = n_windows * window_size
    row_indices = np.repeat(metadata["row_index"].to_numpy(dtype=np.int64), window_size)
    target_positions = np.tile(np.arange(window_size, dtype=np.int16), n_windows)
    starts = np.repeat(
        metadata["window_start"].to_numpy(dtype=np.int64),
        window_size,
    )
    case_upper = _flatten_list_column(metadata, "case_is_upper").astype(bool)
    conserved = _flatten_list_column(metadata, "is_conserved").astype(bool)
    repeat = _flatten_list_column(metadata, "is_repeat").astype(bool)
    ambiguous = _flatten_list_column(metadata, "is_ambiguous").astype(bool)
    nll = _flatten_list_column(stats, "nll").astype(np.float32)
    entropy = _flatten_list_column(stats, "entropy_4nuc").astype(np.float32)
    bases = np.concatenate(
        [np.asarray(list(sequence)) for sequence in metadata["sequence_upper"]]
    )
    is_scorable = (~ambiguous) & np.isin(bases, list("ACGT"))
    conservation_status = np.full(
        n_positions,
        "below_threshold_or_missing",
        dtype=object,
    )
    conservation_status[conserved] = "conserved"
    conservation_status[ambiguous] = "ambiguous_base"

    atoms = pd.DataFrame(
        {
            "token_index": row_indices * window_size + target_positions,
            "row_index": row_indices.astype(np.int32),
            "window_id": np.repeat(metadata["window_id"].to_numpy(), window_size),
            "region": np.repeat(metadata["region"].to_numpy(), window_size),
            "chrom": np.repeat(metadata["chrom"].to_numpy(), window_size),
            "window_start": starts,
            "window_end": np.repeat(
                metadata["window_end"].to_numpy(dtype=np.int64),
                window_size,
            ),
            "target_pos": target_positions,
            "genomic_pos": starts + target_positions,
            "base": bases,
            "case_is_upper": case_upper,
            "is_conserved": conserved,
            "conservation_label_status": conservation_status,
            "is_repeat": repeat,
            "is_ambiguous": ambiguous,
            "is_scorable": is_scorable,
            "window_gc": np.repeat(
                metadata["window_gc"].to_numpy(dtype=np.float32),
                window_size,
            ),
            "kmer7_nll": _flatten_list_column(metadata, "kmer7_nll").astype(
                np.float32
            ),
            "nll": nll,
            "entropy_4nuc": entropy,
            "checkpoint": checkpoint,
            "checkpoint_order": np.int8(checkpoint_order),
            "stage": stage,
            "training_step": np.int32(training_step),
            "cumulative_tokens": np.int64(cumulative_tokens),
            "assembly": assembly,
            "conservation_label_source": conservation_label_source,
            "repeat_label_source": f"{assembly}_soft_mask",
        }
    )
    for column in (
        "region",
        "base",
        "conservation_label_status",
        "checkpoint",
        "stage",
        "assembly",
        "conservation_label_source",
        "repeat_label_source",
    ):
        atoms[column] = atoms[column].astype("category")
    assert len(atoms) == n_positions
    assert atoms[["token_index", "nll", "entropy_4nuc"]].notna().all().all()
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "checkpoint": checkpoint,
        "checkpoint_order": checkpoint_order,
        "stage": stage,
        "training_step": training_step,
        "cumulative_tokens": cumulative_tokens,
        "n_windows": n_windows,
        "window_size": window_size,
        "n_positions": n_positions,
        "n_scorable": int(is_scorable.sum()),
        "orientation": "fwd",
        "coordinate_system": "0-based half-open",
        "token_identity": ["region", "row_index", "target_pos"],
        "conservation_missingness": (
            "below_threshold_and_missing_alignment_are_not_distinguished"
        ),
    }
    return atoms, manifest


def validate_pilot_artifacts(
    atom_paths: dict[tuple[str, str], str | Path],
    manifest_paths: dict[tuple[str, str], str | Path],
    *,
    checkpoints: list[str],
    regions: list[str],
    expected_windows: int,
    window_size: int,
) -> dict[str, Any]:
    """Validate earliest/terminal pilot identity, finiteness, and aggregate gates."""
    assert len(checkpoints) == 2, checkpoints
    report: dict[str, Any] = {
        "passed": True,
        "checkpoints": checkpoints,
        "regions": regions,
        "expected_windows": expected_windows,
        "window_size": window_size,
        "cells": {},
    }
    identity_columns = [
        "region",
        "row_index",
        "window_id",
        "target_pos",
        "chrom",
        "genomic_pos",
    ]
    for region in regions:
        identities: list[pd.DataFrame] = []
        for checkpoint in checkpoints:
            key = (checkpoint, region)
            atoms = pd.read_parquet(atom_paths[key])
            expected_rows = expected_windows * window_size
            assert len(atoms) == expected_rows, (
                f"{checkpoint}/{region}: {len(atoms)} != {expected_rows}"
            )
            assert not atoms.duplicated(["row_index", "target_pos"]).any()
            assert atoms["target_pos"].min() == 0
            assert atoms["target_pos"].max() == window_size - 1
            scorable = atoms["is_scorable"].to_numpy(dtype=bool)
            assert np.isfinite(atoms.loc[scorable, "nll"]).all()
            assert np.isfinite(atoms.loc[scorable, "entropy_4nuc"]).all()
            manifest = json.loads(Path(manifest_paths[key]).read_text())
            gate = manifest["score_manifest"]["aggregate_gate"]
            assert gate is not None and gate["passed"] is True
            report["cells"][f"{checkpoint}/{region}"] = {
                "n_rows": len(atoms),
                "n_scorable": int(scorable.sum()),
                "aggregate_max_abs": gate["max_abs_per_window_sum_diff"],
            }
            identities.append(atoms[identity_columns].reset_index(drop=True))
        pd.testing.assert_frame_equal(identities[0], identities[1])
    return report


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a deterministic newline-terminated JSON manifest."""
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
