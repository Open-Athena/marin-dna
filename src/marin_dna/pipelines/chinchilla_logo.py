"""Genome-wide two-pass predictive sequence logos for issue #419.

All coordinates in this module are 0-based, half-open.  The implementation is
specific to the chinchilla m5.1 proof of concept: 255-bp canonical-DNA windows,
one forward and one reverse-complement model pass, strand-logit averaging, and
four-channel log-probability/logo BigWigs.

The functions are deliberately kept in the library even though the pipeline is
experimental.  Snakemake only supplies paths and configuration; tiling,
inference, shard encoding, and artifact validation remain directly testable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

import numpy as np
import polars as pl
import pyBigWig
import torch
import torch.nn.functional as F
from datasets import Dataset
from jaxtyping import Float, Int
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin_dna.data.dna import NUCLEOTIDES
from marin_dna.data.genome import Genome
from marin_dna.data.transforms import (
    _get_nucleotide_token_ids,
    _get_special_token_counts,
)
from marin_dna.model.runner import run_inference

ASSEMBLY_ACCESSION = "GCF_000276665.1"
MODEL_REPOSITORY = "marin-dna/marin-dna-exp135-m5.1"
MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
DEFAULT_CONTEXT_SIZE = 255
DEFAULT_STRIDE = 128
DEFAULT_RETAIN_START = 63
DEFAULT_RETAIN_END = 191
RC_CHANNEL_INDICES = np.array([3, 2, 1, 0], dtype=np.int64)
BASE_COLORS: dict[str, str] = {
    "A": "0,128,0",
    "C": "0,0,255",
    "G": "255,166,0",
    "T": "255,0,0",
}

_CANONICAL_RUN = re.compile(r"[ACGTacgt]+")


@dataclass(frozen=True)
class WindowSpec:
    """One canonical context window and the subinterval it publishes."""

    chrom: str
    run_start: int
    run_end: int
    window_start: int
    emit_start: int
    emit_end: int

    def __post_init__(self) -> None:
        assert self.chrom, "chrom must be non-empty"
        assert 0 <= self.run_start < self.run_end
        assert self.run_start <= self.window_start
        assert self.window_start < self.emit_start < self.emit_end
        assert self.emit_end <= self.run_end

    @property
    def n_emitted(self) -> int:
        """Number of one-base scores emitted from this window."""
        return self.emit_end - self.emit_start


@dataclass(frozen=True)
class CoverageStats:
    """Exact reconciliation of one planned scaffold."""

    chrom: str
    chrom_size: int
    canonical_bases: int
    noncanonical_bases: int
    short_run_bases: int
    border_excluded_bases: int
    scored_bases: int
    canonical_run_count: int
    scoreable_run_count: int
    window_count: int

    def __post_init__(self) -> None:
        assert self.chrom_size >= 0
        assert self.canonical_bases + self.noncanonical_bases == self.chrom_size
        assert (
            self.scored_bases
            + self.short_run_bases
            + self.border_excluded_bases
            + self.noncanonical_bases
            == self.chrom_size
        ), "coverage categories do not reconcile to chrom_size"


@dataclass(frozen=True)
class LogoScores:
    """Canonical A/C/G/T score matrices, with channels in A/C/G/T order."""

    log_probabilities: np.ndarray
    probabilities: np.ndarray
    entropy_bits: np.ndarray
    information_bits: np.ndarray
    glyph_heights_bits: np.ndarray


@dataclass(frozen=True)
class ScoreShard:
    """A bounded, resumable score shard loaded from disk."""

    chrom: str
    metadata: dict[str, Any]
    window_start: np.ndarray
    emit_start: np.ndarray
    emit_end: np.ndarray
    score_offsets: np.ndarray
    log_probabilities: np.ndarray


def canonical_runs(sequence: str, *, offset: int = 0) -> list[tuple[int, int]]:
    """Return maximal A/C/G/T intervals in ``sequence``.

    Lowercase A/C/G/T is canonical.  Every other character is a hard boundary.
    Returned coordinates include ``offset`` and remain 0-based, half-open.
    """
    assert offset >= 0, f"offset must be non-negative, got {offset}"
    return [
        (offset + m.start(), offset + m.end())
        for m in _CANONICAL_RUN.finditer(sequence)
    ]


def _validate_tiling_parameters(
    context_size: int,
    stride: int,
    retain_start: int,
    retain_end: int,
    phase: int,
) -> None:
    assert context_size > 0
    assert stride > 0
    assert 0 <= retain_start < retain_end <= context_size
    assert retain_end - retain_start == stride, (
        "exact-once abutting emission requires stride == retained width; got "
        f"stride={stride}, retained=[{retain_start}, {retain_end})"
    )
    assert 0 <= phase < stride, f"phase must be in [0, {stride}), got {phase}"


def tile_canonical_run(
    chrom: str,
    run_start: int,
    run_end: int,
    *,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    stride: int = DEFAULT_STRIDE,
    retain_start: int = DEFAULT_RETAIN_START,
    retain_end: int = DEFAULT_RETAIN_END,
    phase: int = 0,
) -> list[WindowSpec]:
    """Tile one canonical run and emit every intended base exactly once.

    Regular contexts start at ``run_start + phase`` and advance by ``stride``.
    If the last regular context does not reach the run end, a context anchored at
    ``run_end - context_size`` extends coverage to the right boundary.  Its
    overlapping retained prefix is discarded, so no base is emitted twice.

    A non-zero ``phase`` is used only by the seam-sensitivity pilot.  It omits the
    extra left-edge positions and lets the caller compare the shared positions to
    the production phase-zero plan.
    """
    _validate_tiling_parameters(context_size, stride, retain_start, retain_end, phase)
    assert 0 <= run_start < run_end
    first_start = run_start + phase
    last_context_start = run_end - context_size
    if first_start > last_context_start:
        return []

    starts = list(range(first_start, last_context_start + 1, stride))
    assert starts, "a scoreable run must produce at least one regular window"
    if starts[-1] != last_context_start:
        starts.append(last_context_start)

    windows: list[WindowSpec] = []
    last_emit_end: int | None = None
    for window_start in starts:
        raw_emit_start = window_start + retain_start
        raw_emit_end = window_start + retain_end
        emit_start = (
            raw_emit_start
            if last_emit_end is None
            else max(raw_emit_start, last_emit_end)
        )
        if emit_start >= raw_emit_end:
            continue
        spec = WindowSpec(
            chrom=chrom,
            run_start=run_start,
            run_end=run_end,
            window_start=window_start,
            emit_start=emit_start,
            emit_end=raw_emit_end,
        )
        assert spec.window_start + context_size <= run_end
        assert spec.window_start + retain_start <= spec.emit_start
        assert spec.emit_end <= spec.window_start + retain_end
        if windows:
            assert windows[-1].emit_end == spec.emit_start, (
                "retained centers must abut within a canonical run"
            )
        windows.append(spec)
        last_emit_end = spec.emit_end

    assert windows[0].emit_start == first_start + retain_start
    assert windows[-1].emit_end == run_end - (context_size - retain_end)
    return windows


def tile_sequence(
    chrom: str,
    sequence: str,
    *,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    stride: int = DEFAULT_STRIDE,
    retain_start: int = DEFAULT_RETAIN_START,
    retain_end: int = DEFAULT_RETAIN_END,
    phase: int = 0,
) -> tuple[list[WindowSpec], CoverageStats]:
    """Plan all canonical runs on one scaffold and reconcile its coverage."""
    _validate_tiling_parameters(context_size, stride, retain_start, retain_end, phase)
    runs = canonical_runs(sequence)
    canonical_bases = sum(end - start for start, end in runs)
    short_run_bases = 0
    scoreable_run_count = 0
    windows: list[WindowSpec] = []
    for run_start, run_end in runs:
        run_windows = tile_canonical_run(
            chrom,
            run_start,
            run_end,
            context_size=context_size,
            stride=stride,
            retain_start=retain_start,
            retain_end=retain_end,
            phase=phase,
        )
        if run_windows:
            scoreable_run_count += 1
            windows.extend(run_windows)
        else:
            short_run_bases += run_end - run_start

    scored_bases = sum(window.n_emitted for window in windows)
    noncanonical_bases = len(sequence) - canonical_bases
    border_excluded_bases = canonical_bases - short_run_bases - scored_bases
    assert border_excluded_bases >= 0
    for previous, current in zip(windows, windows[1:]):
        if previous.chrom == current.chrom:
            assert previous.emit_end <= current.emit_start

    stats = CoverageStats(
        chrom=chrom,
        chrom_size=len(sequence),
        canonical_bases=canonical_bases,
        noncanonical_bases=noncanonical_bases,
        short_run_bases=short_run_bases,
        border_excluded_bases=border_excluded_bases,
        scored_bases=scored_bases,
        canonical_run_count=len(runs),
        scoreable_run_count=scoreable_run_count,
        window_count=len(windows),
    )
    return windows, stats


def parse_chrom_sizes(path: str | Path) -> list[tuple[str, int]]:
    """Parse a UCSC two-column ``chrom.sizes`` file, preserving its order."""
    entries: list[tuple[str, int]] = []
    seen: set[str] = set()
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split("\t")
            assert len(fields) == 2, (
                f"{path}:{line_number}: expected two tab-separated fields"
            )
            chrom, size_text = fields
            size = int(size_text)
            assert chrom not in seen, f"duplicate chrom {chrom!r} in {path}"
            assert size > 0, f"non-positive size for {chrom!r}: {size}"
            seen.add(chrom)
            entries.append((chrom, size))
    assert entries, f"chrom sizes file is empty: {path}"
    return entries


def _plan_frame(windows: Sequence[WindowSpec]) -> pl.DataFrame:
    schema = {
        "chrom": pl.String,
        "run_start": pl.Int64,
        "run_end": pl.Int64,
        "window_start": pl.Int64,
        "emit_start": pl.Int64,
        "emit_end": pl.Int64,
    }
    if not windows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame([asdict(window) for window in windows], schema=schema)


def write_window_plan(
    genome_path: str | Path,
    chrom_sizes_path: str | Path,
    chrom: str,
    plan_path: str | Path,
    metadata_path: str | Path,
    *,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    stride: int = DEFAULT_STRIDE,
    retain_start: int = DEFAULT_RETAIN_START,
    retain_end: int = DEFAULT_RETAIN_END,
    phase: int = 0,
) -> CoverageStats:
    """Plan one scaffold from FASTA and write Parquet plus coverage metadata."""
    chrom_sizes = dict(parse_chrom_sizes(chrom_sizes_path))
    assert chrom in chrom_sizes, f"{chrom!r} absent from UCSC chrom sizes"
    genome = Genome(genome_path, subset_chroms={chrom})
    assert genome.chroms == {chrom: chrom_sizes[chrom]}, (
        "FASTA and UCSC chrom.sizes disagree for "
        f"{chrom}: {genome.chroms.get(chrom)} vs {chrom_sizes[chrom]}"
    )
    sequence = genome(chrom, 0, chrom_sizes[chrom])
    assert len(sequence) == chrom_sizes[chrom]
    windows, stats = tile_sequence(
        chrom,
        sequence,
        context_size=context_size,
        stride=stride,
        retain_start=retain_start,
        retain_end=retain_end,
        phase=phase,
    )

    plan = Path(plan_path)
    plan.parent.mkdir(parents=True, exist_ok=True)
    _plan_frame(windows).write_parquet(plan)
    metadata = {
        "coordinate_system": "0-based-half-open",
        "parameters": {
            "context_size": context_size,
            "stride": stride,
            "retain_start": retain_start,
            "retain_end": retain_end,
            "phase": phase,
            "canonical_run_policy": "maximal case-insensitive A/C/G/T runs",
        },
        "coverage": asdict(stats),
    }
    _write_json(metadata_path, metadata)
    return stats


def read_window_plan(path: str | Path) -> list[WindowSpec]:
    """Load and defensively validate a window-plan Parquet."""
    frame = pl.read_parquet(path)
    expected = {
        "chrom",
        "run_start",
        "run_end",
        "window_start",
        "emit_start",
        "emit_end",
    }
    assert set(frame.columns) == expected, (
        f"window plan columns {frame.columns} != {sorted(expected)}"
    )
    windows = [WindowSpec(**row) for row in frame.to_dicts()]
    for previous, current in zip(windows, windows[1:]):
        assert previous.chrom <= current.chrom
        if previous.chrom == current.chrom:
            assert previous.window_start <= current.window_start
            assert previous.emit_end <= current.emit_start
    return windows


def transform_genome_logo_window(
    example: dict[str, Any],
    tokenizer: Any,
    genome: Genome,
    context_size: int,
    strand: Literal["+", "-"] = "+",
) -> dict[str, Tensor]:
    """Extract and tokenize one all-canonical genome window."""
    chrom = str(example["chrom"])
    window_start = int(example["window_start"])
    sequence = genome(
        chrom,
        window_start,
        window_start + context_size,
        strand=strand,
    ).upper()
    assert len(sequence) == context_size
    invalid = set(sequence) - set(NUCLEOTIDES)
    assert not invalid, (
        f"planned window {chrom}:{window_start}-{window_start + context_size} "
        f"contains non-canonical bases: {sorted(invalid)}"
    )
    n_prefix, n_suffix = _get_special_token_counts(tokenizer)
    input_ids = tokenizer.encode(sequence)
    expected_length = context_size + n_prefix + n_suffix
    assert len(input_ids) == expected_length, (
        "genome-logo scoring requires one token per nucleotide: "
        f"encoded {context_size} bases as {len(input_ids)} tokens, expected "
        f"{expected_length} including special tokens"
    )
    return {"input_ids": torch.tensor(input_ids, dtype=torch.long)}


def compute_window_nucleotide_logits(
    model: Any,
    input_ids: Int[Tensor, "B L"],
    *,
    nucleotide_token_ids: Int[Tensor, " 4"],
    n_prefix: int,
    context_size: int,
) -> Float[Tensor, "B W 4"]:
    """Read A/C/G/T next-token logits for every DNA position in each window."""
    assert n_prefix >= 1, (
        "genome-logo scoring requires an auto-prepended BOS token so every DNA "
        "position has a next-token prediction"
    )
    logits = model(input_ids).logits
    assert logits.ndim == 3 and logits.shape[:2] == input_ids.shape
    readout_indices = torch.arange(context_size, device=input_ids.device) + n_prefix - 1
    nucleotide_ids = nucleotide_token_ids.to(input_ids.device)
    selected = logits[:, readout_indices][..., nucleotide_ids].float()
    assert selected.shape == (input_ids.shape[0], context_size, len(NUCLEOTIDES))
    assert torch.isfinite(selected).all(), "model produced non-finite logits"
    return selected


def run_window_nucleotide_logits(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: Dataset,
    genome: Genome,
    context_size: int,
    *,
    inference_kwargs: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Run exactly one FWD and one RC ``Trainer.predict`` pass over a chunk."""
    assert len(dataset) > 0, "cannot run inference on an empty window chunk"
    n_prefix, _ = _get_special_token_counts(tokenizer)
    assert n_prefix >= 1, "the pinned m5.1 logo path requires a BOS token"
    token_ids = _get_nucleotide_token_ids(tokenizer)
    nucleotide_token_ids = torch.tensor(
        [token_ids[nucleotide] for nucleotide in NUCLEOTIDES], dtype=torch.long
    )

    def _one(strand: Literal["+", "-"]) -> np.ndarray:
        values = run_inference(
            model,
            tokenizer,
            dataset,
            compute_fn=lambda wrapped_model, input_ids: (
                compute_window_nucleotide_logits(
                    wrapped_model,
                    input_ids,
                    nucleotide_token_ids=nucleotide_token_ids,
                    n_prefix=n_prefix,
                    context_size=context_size,
                )
            ),
            data_transform_fn=lambda example, tokenizer: transform_genome_logo_window(
                example,
                tokenizer,
                genome,
                context_size,
                strand,
            ),
            data_transform_on_the_fly=True,
            inference_kwargs=inference_kwargs,
        )
        out = np.asarray(values, dtype=np.float32)
        assert out.shape == (len(dataset), context_size, len(NUCLEOTIDES))
        assert np.isfinite(out).all()
        return out

    return {"fwd": _one("+"), "rc": _one("-")}


def logo_from_log_probabilities(log_probabilities: np.ndarray) -> LogoScores:
    """Derive probabilities and standard information-content glyph heights."""
    logp = np.asarray(log_probabilities, dtype=np.float32)
    assert logp.ndim >= 2 and logp.shape[-1] == len(NUCLEOTIDES)
    assert np.isfinite(logp).all(), "log-probabilities must be finite"
    assert (logp <= 1e-6).all(), "canonical log-probabilities must be non-positive"
    probabilities = np.exp(logp).astype(np.float32, copy=False)
    assert np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6), (
        "probabilities do not sum to one"
    )

    log2_probabilities = np.zeros_like(probabilities)
    np.log2(
        probabilities,
        out=log2_probabilities,
        where=probabilities > 0,
    )
    entropy_bits = -(probabilities * log2_probabilities).sum(axis=-1)
    entropy_bits = np.clip(entropy_bits, 0.0, 2.0).astype(np.float32, copy=False)
    information_bits = (2.0 - entropy_bits).astype(np.float32, copy=False)
    glyph_heights = (probabilities * information_bits[..., None]).astype(
        np.float32, copy=False
    )

    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert ((0 <= entropy_bits) & (entropy_bits <= 2)).all()
    assert ((0 <= information_bits) & (information_bits <= 2)).all()
    assert np.allclose(glyph_heights.sum(axis=-1), information_bits, atol=1e-6), (
        "glyph heights do not sum to total information"
    )
    return LogoScores(
        log_probabilities=logp,
        probabilities=probabilities,
        entropy_bits=entropy_bits,
        information_bits=information_bits,
        glyph_heights_bits=glyph_heights,
    )


def aggregate_strand_logits(
    forward_logits: np.ndarray,
    reverse_complement_logits: np.ndarray,
) -> LogoScores:
    """Realign RC logits, average logits, and apply canonical log-softmax."""
    forward = np.asarray(forward_logits, dtype=np.float32)
    reverse = np.asarray(reverse_complement_logits, dtype=np.float32)
    assert forward.shape == reverse.shape
    assert forward.ndim >= 2 and forward.shape[-1] == len(NUCLEOTIDES)
    assert np.isfinite(forward).all() and np.isfinite(reverse).all()

    reverse_in_forward_coordinates = reverse[..., ::-1, :][..., RC_CHANNEL_INDICES]
    mean_logits = (forward + reverse_in_forward_coordinates) / np.float32(2.0)
    logp = F.log_softmax(torch.from_numpy(mean_logits), dim=-1).numpy()
    return logo_from_log_probabilities(logp)


def write_score_shard(
    path: str | Path,
    windows: Sequence[WindowSpec],
    window_log_probabilities: np.ndarray,
    *,
    metadata: dict[str, Any],
) -> None:
    """Write one bounded shard containing canonical log-probabilities only.

    Logo heights are deterministic from ``logp`` and are intentionally not
    duplicated in intermediate storage.  ``score_offsets`` maps every window's
    ``[emit_start, emit_end)`` interval to its rows in the flattened score array.
    """
    assert windows, "score shard cannot be empty"
    chroms = {window.chrom for window in windows}
    assert len(chroms) == 1, f"one shard must contain one chrom, got {chroms}"
    logp_windows = np.asarray(window_log_probabilities, dtype=np.float32)
    context_size = int(metadata["context_size"])
    assert logp_windows.shape == (
        len(windows),
        context_size,
        len(NUCLEOTIDES),
    )

    offsets = [0]
    emitted: list[np.ndarray] = []
    for window, values in zip(windows, logp_windows):
        local_start = window.emit_start - window.window_start
        local_end = window.emit_end - window.window_start
        assert 0 <= local_start < local_end <= context_size
        block = values[local_start:local_end]
        assert block.shape == (window.n_emitted, len(NUCLEOTIDES))
        emitted.append(block)
        offsets.append(offsets[-1] + window.n_emitted)
    flat_logp = np.concatenate(emitted, axis=0).astype(np.float32, copy=False)
    logo_from_log_probabilities(flat_logp)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            chrom=np.array(next(iter(chroms))),
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
            window_start=np.array(
                [window.window_start for window in windows], dtype=np.int64
            ),
            emit_start=np.array(
                [window.emit_start for window in windows], dtype=np.int64
            ),
            emit_end=np.array([window.emit_end for window in windows], dtype=np.int64),
            score_offsets=np.array(offsets, dtype=np.int64),
            logp=flat_logp,
        )
    temporary.replace(output)


def load_score_shard(path: str | Path) -> ScoreShard:
    """Load a score shard and validate coordinates, offsets, dtype, and values."""
    with np.load(path, allow_pickle=False) as archive:
        shard = ScoreShard(
            chrom=str(archive["chrom"].item()),
            metadata=json.loads(str(archive["metadata_json"].item())),
            window_start=np.asarray(archive["window_start"], dtype=np.int64),
            emit_start=np.asarray(archive["emit_start"], dtype=np.int64),
            emit_end=np.asarray(archive["emit_end"], dtype=np.int64),
            score_offsets=np.asarray(archive["score_offsets"], dtype=np.int64),
            log_probabilities=np.asarray(archive["logp"]),
        )
    n_windows = len(shard.window_start)
    assert n_windows > 0
    assert shard.emit_start.shape == shard.emit_end.shape == (n_windows,)
    assert shard.score_offsets.shape == (n_windows + 1,)
    assert shard.score_offsets[0] == 0
    assert np.all(np.diff(shard.score_offsets) == shard.emit_end - shard.emit_start)
    assert np.all(shard.window_start < shard.emit_start)
    assert np.all(shard.emit_start < shard.emit_end)
    assert np.all(shard.emit_end[:-1] <= shard.emit_start[1:])
    assert shard.log_probabilities.dtype == np.float32
    assert shard.log_probabilities.shape == (
        int(shard.score_offsets[-1]),
        len(NUCLEOTIDES),
    )
    logo_from_log_probabilities(shard.log_probabilities)
    return shard


def _validate_resumable_shard(
    path: Path,
    windows: Sequence[WindowSpec],
    expected_metadata: dict[str, Any],
) -> float:
    shard = load_score_shard(path)
    assert shard.chrom == windows[0].chrom
    np.testing.assert_array_equal(
        shard.window_start,
        np.array([window.window_start for window in windows], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        shard.emit_start,
        np.array([window.emit_start for window in windows], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        shard.emit_end,
        np.array([window.emit_end for window in windows], dtype=np.int64),
    )
    for key, value in expected_metadata.items():
        assert shard.metadata.get(key) == value, (
            f"resumable shard {path} metadata {key!r} changed: "
            f"{shard.metadata.get(key)!r} != {value!r}"
        )
    return float(shard.metadata["inference_seconds"])


def score_window_plan(
    plan_path: str | Path,
    genome_path: str | Path,
    shard_dir: str | Path,
    runtime_path: str | Path,
    done_path: str | Path,
    *,
    model_repository: str = MODEL_REPOSITORY,
    model_revision: str = MODEL_REVISION,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    windows_per_chunk: int = 4096,
    batch_size: int = 128,
    num_workers: int = 2,
    torch_compile: bool = True,
    bf16_full_eval: bool = True,
    eval_accumulation_steps: int | None = None,
) -> dict[str, Any]:
    """Score a scaffold plan with one resident model and resumable chunks."""
    assert windows_per_chunk > 0 and batch_size > 0 and num_workers >= 0
    windows = read_window_plan(plan_path)
    assert windows, f"window plan has no scoreable windows: {plan_path}"
    chroms = {window.chrom for window in windows}
    assert len(chroms) == 1
    chrom = next(iter(chroms))
    assert all(
        window.window_start + context_size <= window.run_end for window in windows
    )

    output_dir = Path(shard_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_repository,
        revision=model_revision,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_repository,
        revision=model_revision,
        trust_remote_code=True,
    )
    model.eval()
    genome = Genome(genome_path, subset_chroms={chrom})
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    inference_kwargs: dict[str, Any] = {
        "per_device_eval_batch_size": batch_size,
        "dataloader_num_workers": num_workers,
        "torch_compile": torch_compile,
        "bf16_full_eval": bf16_full_eval,
        "remove_unused_columns": False,
    }
    if eval_accumulation_steps is not None:
        inference_kwargs["eval_accumulation_steps"] = eval_accumulation_steps

    wall_start = time.perf_counter()
    total_inference_seconds = 0.0
    resumed_shards = 0
    shard_count = 0
    for chunk_index, start in enumerate(range(0, len(windows), windows_per_chunk)):
        chunk = windows[start : start + windows_per_chunk]
        shard_path = output_dir / f"part-{chunk_index:06d}.npz"
        invariant_metadata = {
            "format_version": 1,
            "coordinate_system": "0-based-half-open",
            "chrom": chrom,
            "context_size": context_size,
            "model_repository": model_repository,
            "model_revision": model_revision,
            "logical_sequences_per_window": 2,
        }
        if shard_path.exists():
            total_inference_seconds += _validate_resumable_shard(
                shard_path, chunk, invariant_metadata
            )
            resumed_shards += 1
            shard_count += 1
            continue

        dataset = Dataset.from_dict(
            {
                "chrom": [window.chrom for window in chunk],
                "window_start": [window.window_start for window in chunk],
            }
        )
        inference_start = time.perf_counter()
        strand_logits = run_window_nucleotide_logits(
            model,
            tokenizer,
            dataset,
            genome,
            context_size,
            inference_kwargs=inference_kwargs,
        )
        elapsed = time.perf_counter() - inference_start
        scores = aggregate_strand_logits(strand_logits["fwd"], strand_logits["rc"])
        shard_metadata = {
            **invariant_metadata,
            "chunk_index": chunk_index,
            "window_count": len(chunk),
            "scored_base_count": sum(window.n_emitted for window in chunk),
            "inference_seconds": elapsed,
            "batch_size": batch_size,
            "torch_compile": torch_compile,
            "bf16_full_eval": bf16_full_eval,
        }
        write_score_shard(
            shard_path,
            chunk,
            scores.log_probabilities,
            metadata=shard_metadata,
        )
        total_inference_seconds += elapsed
        shard_count += 1
        del dataset, strand_logits, scores

    wall_seconds = time.perf_counter() - wall_start
    scored_bases = sum(window.n_emitted for window in windows)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    peak_vram_bytes = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    runtime = {
        "chrom": chrom,
        "window_count": len(windows),
        "logical_scored_sequence_count": 2 * len(windows),
        "scored_base_count": scored_bases,
        "shard_count": shard_count,
        "resumed_shard_count": resumed_shards,
        "model_inference_seconds": total_inference_seconds,
        "wall_seconds_this_invocation": wall_seconds,
        "windows_per_second": (
            len(windows) / total_inference_seconds
            if total_inference_seconds > 0
            else None
        ),
        "bases_per_second": (
            scored_bases / total_inference_seconds
            if total_inference_seconds > 0
            else None
        ),
        "gpu": gpu_name,
        "peak_vram_bytes": peak_vram_bytes,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "torch_compile": torch_compile,
        "bf16_full_eval": bf16_full_eval,
        "eval_accumulation_steps": eval_accumulation_steps,
    }
    _write_json(runtime_path, runtime)
    _write_json(done_path, {"complete": True, **runtime})
    return runtime


def list_score_shards(shard_dirs: Iterable[str | Path]) -> list[Path]:
    """List all score shards, failing if any requested scaffold has none."""
    paths: list[Path] = []
    for shard_dir in shard_dirs:
        directory = Path(shard_dir)
        found = sorted(directory.glob("part-*.npz"))
        assert found, f"no score shards found in {directory}"
        paths.extend(found)
    return paths


def _contiguous_score_blocks(
    shard: ScoreShard,
) -> Iterator[tuple[int, int, int]]:
    """Yield ``(genomic_start, score_offset_start, score_offset_end)`` blocks."""
    block_first = 0
    for index in range(1, len(shard.emit_start) + 1):
        is_boundary = (
            index == len(shard.emit_start)
            or shard.emit_end[index - 1] != shard.emit_start[index]
        )
        if not is_boundary:
            continue
        yield (
            int(shard.emit_start[block_first]),
            int(shard.score_offsets[block_first]),
            int(shard.score_offsets[index]),
        )
        block_first = index


def write_bigwig_sets(
    shard_paths: Sequence[str | Path],
    chrom_sizes_path: str | Path,
    output_root: str | Path,
) -> dict[str, Path]:
    """Write the four log-probability and four logo BigWigs directly."""
    assert shard_paths, "at least one score shard is required"
    chrom_sizes = parse_chrom_sizes(chrom_sizes_path)
    chrom_rank = {chrom: rank for rank, (chrom, _size) in enumerate(chrom_sizes)}

    # Read only the small coordinate arrays to establish deterministic global
    # ordering before opening output BigWigs.
    order: list[tuple[int, int, Path]] = []
    for path_like in shard_paths:
        path = Path(path_like)
        with np.load(path, allow_pickle=False) as archive:
            chrom = str(archive["chrom"].item())
            starts = np.asarray(archive["emit_start"], dtype=np.int64)
        assert chrom in chrom_rank, f"shard chrom {chrom!r} absent from chrom.sizes"
        assert len(starts) > 0
        order.append((chrom_rank[chrom], int(starts[0]), path))
    ordered_paths = [item[2] for item in sorted(order)]

    root = Path(output_root)
    logprob_dir = root / "bigwig" / "logprob"
    logo_dir = root / "bigwig" / "logo"
    logprob_dir.mkdir(parents=True, exist_ok=True)
    logo_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {
        f"logprob/{base}": logprob_dir / f"{base}.bw" for base in NUCLEOTIDES
    }
    final_paths.update(
        {f"logo/{base}": logo_dir / f"{base}.bw" for base in NUCLEOTIDES}
    )
    temporary_paths = {
        key: path.with_name(f".{path.name}.{os.getpid()}.tmp")
        for key, path in final_paths.items()
    }
    writers: dict[str, Any] = {}
    try:
        for key, path in temporary_paths.items():
            writer = pyBigWig.open(str(path), "w")
            writer.addHeader(chrom_sizes)
            writers[key] = writer

        last_end_by_chrom: dict[str, int] = {}
        for path in ordered_paths:
            shard = load_score_shard(path)
            chrom_size = dict(chrom_sizes)[shard.chrom]
            assert int(shard.emit_end[-1]) <= chrom_size
            previous_end = last_end_by_chrom.get(shard.chrom, 0)
            assert previous_end <= int(shard.emit_start[0]), (
                f"overlapping shards on {shard.chrom}: {previous_end} > "
                f"{int(shard.emit_start[0])}"
            )
            logo = logo_from_log_probabilities(shard.log_probabilities)
            for genomic_start, score_start, score_end in _contiguous_score_blocks(
                shard
            ):
                for channel, base in enumerate(NUCLEOTIDES):
                    writers[f"logprob/{base}"].addEntries(
                        shard.chrom,
                        genomic_start,
                        values=shard.log_probabilities[
                            score_start:score_end, channel
                        ].tolist(),
                        span=1,
                        step=1,
                    )
                    writers[f"logo/{base}"].addEntries(
                        shard.chrom,
                        genomic_start,
                        values=logo.glyph_heights_bits[
                            score_start:score_end, channel
                        ].tolist(),
                        span=1,
                        step=1,
                    )
            last_end_by_chrom[shard.chrom] = int(shard.emit_end[-1])
    finally:
        for writer in writers.values():
            writer.close()

    for key, temporary in temporary_paths.items():
        temporary.replace(final_paths[key])
    return final_paths


def write_ucsc_hub(
    output_root: str | Path,
    *,
    assembly_accession: str = ASSEMBLY_ACCESSION,
) -> dict[str, Path]:
    """Write a UCSC hub with the logo visible and log-probabilities hidden."""
    root = Path(output_root)
    ucsc_root = root / "ucsc"
    assembly_root = ucsc_root / assembly_accession
    assembly_root.mkdir(parents=True, exist_ok=True)
    hub_path = ucsc_root / "hub.txt"
    genomes_path = ucsc_root / "genomes.txt"
    track_db_path = assembly_root / "trackDb.txt"

    hub_path.write_text(
        "hub marinDnaChinchillaLogo\n"
        "shortLabel MarinDNA chinchilla\n"
        "longLabel MarinDNA m5.1 two-pass predictive chinchilla sequence logo\n"
        "genomesFile genomes.txt\n"
        "email info@openathena.ai\n"
    )
    genomes_path.write_text(
        f"genome {assembly_accession}\ntrackDb {assembly_accession}/trackDb.txt\n"
    )

    lines = [
        "track marinDnaM51PredictiveLogo",
        "shortLabel MarinDNA m5.1 logo",
        "longLabel MarinDNA m5.1 two-pass predictive sequence logo",
        "container multiWig",
        "aggregate stacked",
        "type bigWig 0 2",
        "autoScale off",
        "viewLimits 0:2",
        "logo on",
        "visibility full",
        "",
    ]
    for priority, base in enumerate(NUCLEOTIDES, start=1):
        lines.extend(
            [
                f"track marinDnaM51Logo{base}",
                "parent marinDnaM51PredictiveLogo",
                f"shortLabel {base}",
                f"longLabel {base} glyph height (bits)",
                f"bigDataUrl ../../bigwig/logo/{base}.bw",
                "type bigWig 0 2",
                f"color {BASE_COLORS[base]}",
                f"priority {priority}",
                "",
            ]
        )
    lines.extend(
        [
            "track marinDnaM51LogProb",
            "shortLabel MarinDNA m5.1 logp",
            "longLabel MarinDNA m5.1 canonical A/C/G/T log-probabilities",
            "container multiWig",
            "aggregate transparentOverlay",
            "autoScale on",
            "visibility hide",
            "",
        ]
    )
    for priority, base in enumerate(NUCLEOTIDES, start=1):
        lines.extend(
            [
                f"track marinDnaM51LogProb{base}",
                "parent marinDnaM51LogProb",
                f"shortLabel log p({base})",
                f"longLabel canonical log p({base}) from strand-averaged logits",
                f"bigDataUrl ../../bigwig/logprob/{base}.bw",
                "type bigWig",
                f"color {BASE_COLORS[base]}",
                f"priority {priority}",
                "",
            ]
        )
    track_db_path.write_text("\n".join(lines))
    return {"hub": hub_path, "genomes": genomes_path, "trackDb": track_db_path}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of ``path`` without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_dataset_readme(
    path: str | Path,
    *,
    application_commit: str,
    repository: str = "Open-Athena/marin-dna",
    model_repository: str = MODEL_REPOSITORY,
    model_revision: str = MODEL_REVISION,
    assembly_accession: str = ASSEMBLY_ACCESSION,
) -> None:
    """Draft the Hugging Face dataset card required before publication."""
    assert len(application_commit) == 40, (
        "dataset README requires a full commit SHA for its immutable pipeline link"
    )
    pipeline_url = (
        f"https://github.com/{repository}/blob/{application_commit}/"
        "snakemake/analysis/chinchilla_logo"
    )
    content = f"""---
tags:
- biology
- genomics
- dna
pretty_name: MarinDNA m5.1 chinchilla predictive sequence logo
---

# MarinDNA m5.1 chinchilla predictive sequence logo

This dataset contains canonical A/C/G/T log-probability and derived glyph-height
BigWigs from the MarinDNA m5.1 two-pass predictive sequence-logo approximation on
the UCSC/NCBI RefSeq chinchilla assembly `{assembly_accession}`. It was produced
with [`{model_repository}`](https://huggingface.co/{model_repository}) at immutable
revision `{model_revision}` by the commit-pinned [chinchilla-logo pipeline]({pipeline_url}).

This is a predictive next-token logo, not an LLR, mutation-effect, or constraint
track. Each scored window uses exactly two logical sequences: the forward
reference and its reverse complement.

## Files

- `bigwig/logprob/{{A,C,G,T}}.bw`: canonical log-probabilities after averaging
  forward and aligned reverse-complement logits.
- `bigwig/logo/{{A,C,G,T}}.bw`: information-content glyph heights in bits.
- `ucsc/hub.txt`: UCSC track-hub entry point.
- `manifest/release.json`: immutable model/code/assembly identities, coverage,
  runtime, and file checksums.
"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)


def write_release_manifest(
    output_root: str | Path,
    chrom_sizes_path: str | Path,
    plan_metadata_paths: Sequence[str | Path],
    runtime_paths: Sequence[str | Path],
    *,
    application_commit: str,
    model_repository: str = MODEL_REPOSITORY,
    model_revision: str = MODEL_REVISION,
    assembly_accession: str = ASSEMBLY_ACCESSION,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    stride: int = DEFAULT_STRIDE,
    retain_start: int = DEFAULT_RETAIN_START,
    retain_end: int = DEFAULT_RETAIN_END,
) -> dict[str, Any]:
    """Write the machine-readable release manifest after artifact validation."""
    assert len(application_commit) == 40
    assert len(plan_metadata_paths) == len(runtime_paths) > 0
    root = Path(output_root)
    chrom_sizes = parse_chrom_sizes(chrom_sizes_path)
    full_assembly_span = sum(size for _chrom, size in chrom_sizes)
    plans = [json.loads(Path(path).read_text()) for path in plan_metadata_paths]
    runtimes = [json.loads(Path(path).read_text()) for path in runtime_paths]
    coverage_rows = [plan["coverage"] for plan in plans]
    scoped_span = sum(int(row["chrom_size"]) for row in coverage_rows)
    coverage = {
        key: sum(int(row[key]) for row in coverage_rows)
        for key in (
            "canonical_bases",
            "noncanonical_bases",
            "short_run_bases",
            "border_excluded_bases",
            "scored_bases",
            "canonical_run_count",
            "scoreable_run_count",
            "window_count",
        )
    }
    assert (
        coverage["scored_bases"]
        + coverage["short_run_bases"]
        + coverage["border_excluded_bases"]
        + coverage["noncanonical_bases"]
        == scoped_span
    )

    artifact_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path != root / "manifest" / "release.json"
    )
    artifacts = {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in artifact_paths
    }
    manifest = {
        "format_version": 1,
        "description": "MarinDNA m5.1 two-pass predictive sequence logo",
        "coordinate_system": "0-based-half-open",
        "model": {
            "repository": model_repository,
            "revision": model_revision,
            "logical_sequences_per_window": 2,
        },
        "application": {
            "repository": "Open-Athena/marin-dna",
            "commit": application_commit,
        },
        "assembly": {
            "accession": assembly_accession,
            "chrom_sizes_sha256": sha256_file(chrom_sizes_path),
            "ucsc_sequence_count": len(chrom_sizes),
            "ucsc_span": full_assembly_span,
            "scoped_span": scoped_span,
            "out_of_scope_bases": full_assembly_span - scoped_span,
        },
        "tiling": {
            "context_size": context_size,
            "stride": stride,
            "retained_interval": [retain_start, retain_end],
            "canonical_run_policy": "maximal case-insensitive A/C/G/T runs",
            "tail_policy": "run-end anchored; emit only previously uncovered bases",
        },
        "coverage": coverage,
        "per_scaffold_coverage": coverage_rows,
        "runtime": runtimes,
        "storage": {
            "dtype": "Float32",
            "decimal_rounded": False,
            "missing_positions": "absent BigWig intervals",
        },
        "validation": {
            "score_shards_validated_before_bigwig_write": True,
            "bigwig_round_trip": "pending external release validation",
            "ucsc_rendering": "pending external release validation",
        },
        "files": artifacts,
    }
    manifest_path = root / "manifest" / "release.json"
    _write_json(manifest_path, manifest)
    return manifest


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
