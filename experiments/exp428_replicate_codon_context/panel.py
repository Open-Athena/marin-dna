"""Build the preregistered phase-and-substitution-matched chr21 panel."""

from __future__ import annotations

import argparse
import bisect
import fcntl
import gzip
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# This experiment shares a small 4-vCPU/15-GiB/no-swap development node with
# other tasks. Apply caps before importing NumPy or Polars, whose thread pools
# otherwise default to every visible CPU.
LOCAL_THREAD_LIMITS = {
    "POLARS_MAX_THREADS": 2,
    "RAYON_NUM_THREADS": 2,
    "OMP_NUM_THREADS": 1,
    "MKL_NUM_THREADS": 1,
    "OPENBLAS_NUM_THREADS": 1,
    "NUMEXPR_NUM_THREADS": 1,
}
for variable, limit in LOCAL_THREAD_LIMITS.items():
    configured = int(os.environ.get(variable, limit))
    assert configured > 0, (variable, configured)
    os.environ[variable] = str(min(configured, limit))

import numpy as np
import polars as pl
from marin_dna.data.genome import Genome

ISSUE = 428
SEED = 428
CHROM = "21"
BLOCK_BP = 1_000_000
DISCOVERY_BLOCKS = 17
VALIDATION_BLOCKS = 6
TEST_BLOCKS = 6
CANDIDATES_PER_LABEL_SUBSTITUTION_BLOCK = 256
MIN_PER_LABEL_STRATUM_SPLIT = 64
CAP_PER_LABEL_STRATUM_SPLIT = {
    "discovery": 256,
    "validation": 128,
    "test": 128,
}
SPLITS = tuple(CAP_PER_LABEL_STRATUM_SPLIT)
POSITIVE_CLASS = "missense_variant"
NEGATIVE_CLASS = "synonymous_variant"
PAIR_CLASSES = (POSITIVE_CLASS, NEGATIVE_CLASS)
SOURCE_DATASET = "songlab/hg38-variant-consequences"
SOURCE_REVISION = "eb3022cc6797b9369cca16af72ff3c4197df343a"
GTF_RELEASE = 109
GTF_BYTES = 54_258_835
GTF_FTP_BSD_CHECKSUM = 26_235
GTF_FTP_BLOCKS = 52_988
CHROM_LENGTH = 46_709_983
NUCLEOTIDES = frozenset("ACGT")
COMPLEMENT = str.maketrans("ACGT", "TGCA")
LOCAL_HEAVY_LOCK = Path("/tmp/marin-dna-local-heavy.lock")
MIN_MEMORY_AVAILABLE_BYTES = 6 * 1024**3
MAX_LOAD_1 = 2.0
GENETIC_CODE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


@dataclass(frozen=True)
class RawCdsSegment:
    start0: int
    end0: int
    phase: int


@dataclass(frozen=True)
class CdsSegment:
    start0: int
    end0: int
    phase: int
    coding_start: int

    @property
    def length(self) -> int:
        return self.end0 - self.start0


@dataclass(frozen=True)
class TranscriptCds:
    transcript_id: str
    gene_id: str
    gene_name: str
    strand: str
    segments: tuple[CdsSegment, ...]
    phase_consistent: bool

    def coding_offset(self, position0: int) -> int | None:
        for segment in self.segments:
            if segment.start0 <= position0 < segment.end0:
                local = (
                    position0 - segment.start0
                    if self.strand == "+"
                    else segment.end0 - 1 - position0
                )
                return segment.coding_start + local
        return None

    def genomic_position0(self, coding_offset: int) -> int | None:
        for segment in self.segments:
            local = coding_offset - segment.coding_start
            if 0 <= local < segment.length:
                return (
                    segment.start0 + local
                    if self.strand == "+"
                    else segment.end0 - 1 - local
                )
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def memory_status_bytes() -> dict[str, int]:
    fields: dict[str, int] = {}
    with Path("/proc/meminfo").open() as handle:
        for line in handle:
            key, value = line.split(":", 1)
            if key in {"MemAvailable", "SwapTotal"}:
                fields[key] = int(value.strip().split()[0]) * 1024
    assert fields.keys() == {"MemAvailable", "SwapTotal"}
    return fields


def lower_process_priority() -> dict[str, Any]:
    current_nice = os.nice(0)
    if current_nice < 10:
        os.nice(10 - current_nice)
    ionice = subprocess.run(
        ["ionice", "-c", "2", "-n", "7", "-p", str(os.getpid())],
        check=False,
        capture_output=True,
        text=True,
    )
    assert ionice.returncode == 0, ionice.stderr
    return {"nice": os.nice(0), "ionice_class": 2, "ionice_priority": 7}


@contextmanager
def local_heavy_guard() -> Iterator[dict[str, Any]]:
    """Serialize local data work and refuse to start under memory pressure."""
    memory = memory_status_bytes()
    available = memory["MemAvailable"]
    load_1 = os.getloadavg()[0]
    assert available >= MIN_MEMORY_AVAILABLE_BYTES, (
        f"only {available / 1024**3:.2f} GiB available; "
        f"require at least {MIN_MEMORY_AVAILABLE_BYTES / 1024**3:.0f} GiB"
    )
    assert load_1 <= MAX_LOAD_1, (
        f"load1 is {load_1:.2f}; require at most {MAX_LOAD_1:.1f}"
    )
    handle = LOCAL_HEAVY_LOCK.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another task holds the shared-node lock {LOCAL_HEAVY_LOCK}"
            ) from error
        policy = {
            "lock_path": str(LOCAL_HEAVY_LOCK),
            "minimum_memory_available_bytes": MIN_MEMORY_AVAILABLE_BYTES,
            "memory_available_at_start_bytes": available,
            "swap_total_bytes": memory["SwapTotal"],
            "maximum_load_1": MAX_LOAD_1,
            "load_1_at_start": load_1,
            "process_priority": lower_process_priority(),
            "thread_limits": {
                variable: int(os.environ[variable]) for variable in LOCAL_THREAD_LIMITS
            },
        }
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps({"issue": ISSUE, "pid": os.getpid(), **policy}, sort_keys=True)
            + "\n"
        )
        handle.flush()
        yield policy
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def assert_commit(value: str) -> None:
    assert len(value) == 40
    assert all(character in "0123456789abcdef" for character in value)


def assert_current_commit(value: str) -> None:
    assert_commit(value)
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert value == current, (value, current)


def bsd_checksum(path: Path) -> tuple[int, int]:
    checksum = 0
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            size += len(chunk)
            for byte in chunk:
                checksum = ((checksum >> 1) + ((checksum & 1) << 15) + byte) & 0xFFFF
    return checksum, (size + 1023) // 1024


def splitmix64(values: np.ndarray) -> np.ndarray:
    """Stable vectorized 64-bit mixing with defined wraparound."""
    assert values.dtype == np.uint64
    with np.errstate(over="ignore"):
        output = values + np.uint64(0x9E3779B97F4A7C15)
        output = (output ^ (output >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        output = (output ^ (output >> 27)) * np.uint64(0x94D049BB133111EB)
    return output ^ (output >> 31)


def variant_hashes(frame: pl.DataFrame, *, seed: int) -> np.ndarray:
    assert {"pos", "ref", "alt"} <= set(frame.columns)
    base_codes = {"A": 0, "C": 1, "G": 2, "T": 3}
    ref_codes = np.asarray(
        [base_codes[value] for value in frame["ref"]], dtype=np.uint64
    )
    alt_codes = np.asarray(
        [base_codes[value] for value in frame["alt"]], dtype=np.uint64
    )
    positions = frame["pos"].to_numpy().astype(np.uint64)
    values = positions ^ (ref_codes << 32) ^ (alt_codes << 34) ^ np.uint64(seed)
    return splitmix64(values)


def assign_blocks(block_ids: Iterable[int]) -> dict[int, str]:
    blocks = np.asarray(sorted(set(block_ids)), dtype=np.uint64)
    assert len(blocks) == DISCOVERY_BLOCKS + VALIDATION_BLOCKS + TEST_BLOCKS
    order = np.lexsort((blocks, splitmix64(blocks ^ np.uint64(SEED))))
    ordered = blocks[order].astype(int).tolist()
    output = {
        block: (
            "discovery"
            if index < DISCOVERY_BLOCKS
            else "validation"
            if index < DISCOVERY_BLOCKS + VALIDATION_BLOCKS
            else "test"
        )
        for index, block in enumerate(ordered)
    }
    assert list(output.values()).count("discovery") == DISCOVERY_BLOCKS
    assert list(output.values()).count("validation") == VALIDATION_BLOCKS
    assert list(output.values()).count("test") == TEST_BLOCKS
    return output


def load_candidate_sample(source_path: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    assert source_path.is_file()
    source = pl.read_parquet(source_path).filter(
        pl.col("consequence_cre").is_in(PAIR_CLASSES)
    )
    assert source.height > 900_000
    assert source["chrom"].unique().to_list() == [CHROM]
    assert set(source["consequence_cre"].unique()) == set(PAIR_CLASSES)
    assert source.null_count().sum_horizontal().sum() == 0
    assert (
        source.select(pl.struct("chrom", "pos", "ref", "alt").n_unique()).item()
        == source.height
    )
    source = source.with_columns(
        ((pl.col("pos") - 1) // BLOCK_BP).cast(pl.UInt32).alias("block_id"),
        (pl.col("ref") + pl.lit(">") + pl.col("alt")).alias("genomic_substitution"),
    )
    block_assignment = assign_blocks(source["block_id"].unique().to_list())
    source = source.with_columns(
        pl.col("block_id").replace_strict(block_assignment).alias("split"),
        pl.Series("candidate_hash", variant_hashes(source, seed=SEED * 100 + 1)),
        pl.Series("sample_hash", variant_hashes(source, seed=SEED * 100 + 2)),
    )
    group_columns = [
        "consequence_cre",
        "block_id",
        "genomic_substitution",
    ]
    candidates = (
        source.sort("candidate_hash")
        .group_by(group_columns, maintain_order=True)
        .head(CANDIDATES_PER_LABEL_SUBSTITUTION_BLOCK)
        .sort(["pos", "ref", "alt"])
    )
    counts = source.group_by("consequence_cre").len().sort("consequence_cre")
    candidate_counts = (
        candidates.group_by("consequence_cre").len().sort("consequence_cre")
    )
    assert candidates.height <= (
        len(block_assignment)
        * len(PAIR_CLASSES)
        * 12
        * CANDIDATES_PER_LABEL_SUBSTITUTION_BLOCK
    )
    metadata = {
        "source_rows": counts.to_dicts(),
        "candidate_rows": candidate_counts.to_dicts(),
        "candidate_total": candidates.height,
        "block_assignment": {
            split: sorted(
                block
                for block, assigned in block_assignment.items()
                if assigned == split
            )
            for split in SPLITS
        },
    }
    return candidates, metadata


def parse_gtf_attributes(value: str) -> dict[str, list[str]]:
    output: dict[str, list[str]] = defaultdict(list)
    for field in value.strip().rstrip(";").split(";"):
        field = field.strip()
        if not field:
            continue
        key, quoted = field.split(" ", 1)
        assert quoted.startswith('"') and quoted.endswith('"'), field
        output[key].append(quoted[1:-1])
    return dict(output)


def build_transcript(
    *,
    transcript_id: str,
    gene_id: str,
    gene_name: str,
    strand: str,
    segments: Iterable[RawCdsSegment],
) -> TranscriptCds:
    assert strand in {"+", "-"}
    ordered = sorted(
        segments,
        key=lambda segment: segment.start0,
        reverse=strand == "-",
    )
    assert ordered and all(segment.start0 < segment.end0 for segment in ordered)
    coding_offset = (3 - ordered[0].phase) % 3
    output: list[CdsSegment] = []
    phase_consistent = True
    for segment in ordered:
        assert segment.phase in {0, 1, 2}
        phase_consistent &= segment.phase == (3 - coding_offset % 3) % 3
        output.append(
            CdsSegment(
                start0=segment.start0,
                end0=segment.end0,
                phase=segment.phase,
                coding_start=coding_offset,
            )
        )
        coding_offset += segment.end0 - segment.start0
    return TranscriptCds(
        transcript_id=transcript_id,
        gene_id=gene_id,
        gene_name=gene_name,
        strand=strand,
        segments=tuple(output),
        phase_consistent=phase_consistent,
    )


def load_chr_cds_transcripts(gtf_path: Path) -> list[TranscriptCds]:
    assert gtf_path.is_file()
    grouped: dict[str, dict[str, Any]] = {}
    with gzip.open(gtf_path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            assert len(fields) == 9
            if fields[0] != CHROM or fields[2] != "CDS":
                continue
            attributes = parse_gtf_attributes(fields[8])
            transcript_id = attributes["transcript_id"][0]
            record = grouped.setdefault(
                transcript_id,
                {
                    "gene_id": attributes["gene_id"][0],
                    "gene_name": attributes.get("gene_name", [""])[0],
                    "strand": fields[6],
                    "segments": [],
                },
            )
            assert record["gene_id"] == attributes["gene_id"][0]
            assert record["strand"] == fields[6]
            start0 = int(fields[3]) - 1
            end0 = int(fields[4])
            assert 0 <= start0 < end0 <= CHROM_LENGTH
            record["segments"].append(
                RawCdsSegment(start0=start0, end0=end0, phase=int(fields[7]))
            )
    transcripts = [
        build_transcript(
            transcript_id=transcript_id,
            gene_id=record["gene_id"],
            gene_name=record["gene_name"],
            strand=record["strand"],
            segments=record["segments"],
        )
        for transcript_id, record in grouped.items()
    ]
    assert transcripts and len({item.transcript_id for item in transcripts}) == len(
        transcripts
    )
    return sorted(transcripts, key=lambda item: item.transcript_id)


def transcripts_by_position(
    positions0: list[int], transcripts: list[TranscriptCds]
) -> dict[int, list[TranscriptCds]]:
    positions = sorted(set(positions0))
    output: dict[int, list[TranscriptCds]] = defaultdict(list)
    for transcript in transcripts:
        for segment in transcript.segments:
            left = bisect.bisect_left(positions, segment.start0)
            right = bisect.bisect_left(positions, segment.end0)
            for position0 in positions[left:right]:
                output[position0].append(transcript)
    return dict(output)


def transcript_base(base: str, strand: str) -> str:
    assert base in NUCLEOTIDES and strand in {"+", "-"}
    return base if strand == "+" else base.translate(COMPLEMENT)


def annotate_transcript_hit(
    transcript: TranscriptCds,
    *,
    position0: int,
    ref: str,
    alt: str,
    sequence: str,
) -> dict[str, Any] | None:
    offset = transcript.coding_offset(position0)
    if offset is None or not transcript.phase_consistent:
        return None
    codon_position0 = offset % 3
    codon_start = offset - codon_position0
    positions = [
        transcript.genomic_position0(codon_start + index) for index in range(3)
    ]
    if any(value is None for value in positions):
        return None
    bases = [sequence[int(value)] for value in positions]
    if transcript.strand == "-":
        bases = [base.translate(COMPLEMENT) for base in bases]
    ref_codon = "".join(bases)
    assert ref_codon in GENETIC_CODE
    assert ref_codon[codon_position0] == transcript_base(ref, transcript.strand)
    alt_codon_bases = list(ref_codon)
    alt_codon_bases[codon_position0] = transcript_base(alt, transcript.strand)
    alt_codon = "".join(alt_codon_bases)
    ref_aa = GENETIC_CODE[ref_codon]
    alt_aa = GENETIC_CODE[alt_codon]
    if ref_aa == alt_aa:
        consequence = NEGATIVE_CLASS
    elif ref_aa != "*" and alt_aa == "*":
        consequence = "stop_gained"
    elif ref_aa == "*" and alt_aa != "*":
        consequence = "stop_lost"
    else:
        consequence = POSITIVE_CLASS
    return {
        "gene_name": transcript.gene_name,
        "strand": transcript.strand,
        "codon_position": codon_position0 + 1,
        "ref_codon": ref_codon,
        "alt_codon": alt_codon,
        "ref_aa": ref_aa,
        "alt_aa": alt_aa,
        "amino_acid_change": f"{ref_aa}>{alt_aa}",
        "predicted_consequence": consequence,
    }


def unique_or_none(values: Iterable[Any]) -> Any | None:
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else None


def load_chrom_sequence(fasta_path: Path) -> str:
    assert fasta_path.is_file()
    assert Path(f"{fasta_path}.fai").is_file() and Path(f"{fasta_path}.gzi").is_file()
    genome = Genome(fasta_path, subset_chroms={CHROM})
    assert set(genome.chroms) == {CHROM}
    sequence = genome(CHROM, 0, CHROM_LENGTH, "+").upper()
    assert len(sequence) == CHROM_LENGTH and set(sequence) <= set("ACGTN")
    return sequence


def annotate_candidates(
    candidates: pl.DataFrame, *, gtf_path: Path, fasta_path: Path
) -> tuple[pl.DataFrame, dict[str, Any]]:
    sequence = load_chrom_sequence(fasta_path)
    transcripts = load_chr_cds_transcripts(gtf_path)
    positions0 = (candidates["pos"] - 1).to_list()
    indexed = transcripts_by_position(positions0, transcripts)
    output: list[dict[str, Any]] = []
    cds_rows = 0
    matching_rows = 0
    for index, row in enumerate(candidates.iter_rows(named=True), start=1):
        position0 = int(row["pos"]) - 1
        assert sequence[position0] == row["ref"]
        hits = [
            hit
            for transcript in indexed.get(position0, [])
            if (
                hit := annotate_transcript_hit(
                    transcript,
                    position0=position0,
                    ref=row["ref"],
                    alt=row["alt"],
                    sequence=sequence,
                )
            )
            is not None
        ]
        cds_rows += bool(hits)
        matching = [
            hit
            for hit in hits
            if hit["predicted_consequence"] == row["consequence_cre"]
        ]
        matching_rows += bool(matching)
        strands = sorted({hit["strand"] for hit in matching})
        codon_positions = sorted({hit["codon_position"] for hit in matching})
        ref_codons = sorted({hit["ref_codon"] for hit in matching})
        alt_codons = sorted({hit["alt_codon"] for hit in matching})
        amino_changes = sorted({hit["amino_acid_change"] for hit in matching})
        gene_names = sorted({hit["gene_name"] for hit in matching if hit["gene_name"]})
        strand = unique_or_none(strands)
        codon_position = unique_or_none(codon_positions)
        output.append(
            {
                **row,
                "cds_hit_count": len(hits),
                "matching_transcript_count": len(matching),
                "consensus_strand": strand,
                "consensus_codon_position": codon_position,
                "consensus_ref_codon": unique_or_none(ref_codons),
                "consensus_alt_codon": unique_or_none(alt_codons),
                "consensus_amino_acid_change": unique_or_none(amino_changes),
                "matching_gene_names": ",".join(gene_names),
                "matching_ref_codons": ",".join(ref_codons),
                "matching_alt_codons": ",".join(alt_codons),
                "matching_amino_acid_changes": ",".join(amino_changes),
                "transcript_substitution": (
                    None
                    if strand is None
                    else f"{transcript_base(row['ref'], strand)}>{transcript_base(row['alt'], strand)}"
                ),
            }
        )
        if index % 20_000 == 0 or index == candidates.height:
            print(
                json.dumps(
                    {
                        "stage": "annotate",
                        "processed": index,
                        "total": candidates.height,
                    }
                ),
                flush=True,
            )
    annotated = pl.DataFrame(output)
    assert annotated.height == candidates.height
    metadata = {
        "candidate_rows": candidates.height,
        "rows_with_cds_hit": cds_rows,
        "rows_with_matching_consequence": matching_rows,
        "rows_with_consensus_strand": annotated.height
        - annotated["consensus_strand"].null_count(),
        "rows_with_consensus_codon_position": annotated.height
        - annotated["consensus_codon_position"].null_count(),
    }
    return annotated, metadata


def balance_panel(annotated: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    eligible = annotated.filter(
        pl.col("consensus_strand").is_not_null()
        & pl.col("consensus_codon_position").is_not_null()
        & pl.col("transcript_substitution").is_not_null()
    ).with_columns(
        (
            pl.col("consensus_codon_position").cast(pl.String)
            + pl.lit("|")
            + pl.col("transcript_substitution")
        ).alias("matching_stratum")
    )
    counts = eligible.group_by(["matching_stratum", "split", "consequence_cre"]).len()
    count_lookup = {
        (row["matching_stratum"], row["split"], row["consequence_cre"]): row["len"]
        for row in counts.iter_rows(named=True)
    }
    common_strata: list[str] = []
    for stratum in sorted(eligible["matching_stratum"].unique()):
        cell_counts = [
            count_lookup.get((stratum, split, label), 0)
            for split in SPLITS
            for label in PAIR_CLASSES
        ]
        if min(cell_counts) >= MIN_PER_LABEL_STRATUM_SPLIT:
            common_strata.append(stratum)
    assert common_strata

    selected_groups: list[pl.DataFrame] = []
    selected_count_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for stratum in common_strata:
            cell_counts = {
                label: count_lookup[(stratum, split, label)] for label in PAIR_CLASSES
            }
            selected_count = min(
                *cell_counts.values(), CAP_PER_LABEL_STRATUM_SPLIT[split]
            )
            assert selected_count >= MIN_PER_LABEL_STRATUM_SPLIT
            for label in PAIR_CLASSES:
                group = (
                    eligible.filter(
                        (pl.col("split") == split)
                        & (pl.col("matching_stratum") == stratum)
                        & (pl.col("consequence_cre") == label)
                    )
                    .sort("sample_hash")
                    .head(selected_count)
                )
                assert group.height == selected_count
                selected_groups.append(group)
                selected_count_rows.append(
                    {
                        "split": split,
                        "matching_stratum": stratum,
                        "consequence_cre": label,
                        "available": cell_counts[label],
                        "selected": selected_count,
                    }
                )
    panel = (
        pl.concat(selected_groups)
        .sort(["pos", "ref", "alt"])
        .with_row_index("panel_row")
    )
    assert (
        panel.select(pl.struct("chrom", "pos", "ref", "alt").n_unique()).item()
        == panel.height
    )
    assert panel["panel_row"].to_list() == list(range(panel.height))
    assert set(panel["matching_stratum"].unique()) == set(common_strata)
    balance = panel.group_by(["split", "matching_stratum", "consequence_cre"]).len()
    for split in SPLITS:
        for stratum in common_strata:
            observed = balance.filter(
                (pl.col("split") == split) & (pl.col("matching_stratum") == stratum)
            )
            assert observed.height == 2
            assert observed["len"].n_unique() == 1
    block_splits = panel.group_by("block_id").agg(pl.col("split").n_unique())
    assert block_splits["split"].max() == 1
    metadata = {
        "eligible_rows": eligible.height,
        "retained_strata": common_strata,
        "retained_strata_count": len(common_strata),
        "selection_counts": selected_count_rows,
        "panel_rows": panel.height,
        "split_rows": panel.group_by("split").len().sort("split").to_dicts(),
        "label_rows": panel.group_by("consequence_cre")
        .len()
        .sort("consequence_cre")
        .to_dicts(),
    }
    return panel, metadata


def build_panel(
    *,
    source_path: Path,
    gtf_path: Path,
    fasta_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assert source_path.is_file() and gtf_path.is_file() and fasta_path.is_file()
    assert not output_dir.exists()
    experiment_commit = os.environ.get("EXPERIMENT_COMMIT", "")
    assert_current_commit(experiment_commit)
    assert gtf_path.stat().st_size == GTF_BYTES
    assert bsd_checksum(gtf_path) == (GTF_FTP_BSD_CHECKSUM, GTF_FTP_BLOCKS)
    assert Path(f"{fasta_path}.fai").is_file() and Path(f"{fasta_path}.gzi").is_file()

    with local_heavy_guard() as resource_policy:
        candidates, candidate_metadata = load_candidate_sample(source_path)
        annotated, annotation_metadata = annotate_candidates(
            candidates, gtf_path=gtf_path, fasta_path=fasta_path
        )
        panel, sampling_metadata = balance_panel(annotated)
        output_dir.mkdir(parents=True)
        panel_path = output_dir / "panel.parquet"
        panel.write_parquet(panel_path, compression="zstd")
        result = {
            "created_at": datetime.now(UTC).isoformat(),
            "issue": ISSUE,
            "experiment_commit": experiment_commit,
            "source": {
                "dataset": SOURCE_DATASET,
                "revision": SOURCE_REVISION,
                "chromosome": CHROM,
                "path_name": source_path.name,
                "bytes": source_path.stat().st_size,
                "sha256": sha256_file(source_path),
            },
            "gtf": {
                "release": GTF_RELEASE,
                "url": "https://ftp.ensembl.org/pub/release-109/gtf/homo_sapiens/Homo_sapiens.GRCh38.109.gtf.gz",
                "path_name": gtf_path.name,
                "bytes": gtf_path.stat().st_size,
                "sha256": sha256_file(gtf_path),
                "ftp_bsd_checksum": GTF_FTP_BSD_CHECKSUM,
                "ftp_blocks": GTF_FTP_BLOCKS,
            },
            "fasta": {
                "path_name": fasta_path.name,
                "chromosome": CHROM,
                "chromosome_length": CHROM_LENGTH,
                "bytes": fasta_path.stat().st_size,
                "sha256": sha256_file(fasta_path),
            },
            "candidate_sampling": {
                "seed": SEED,
                "block_bp": BLOCK_BP,
                "candidate_cap_per_label_genomic_substitution_block": CANDIDATES_PER_LABEL_SUBSTITUTION_BLOCK,
                **candidate_metadata,
            },
            "annotation": annotation_metadata,
            "balanced_sampling": {
                "minimum_per_label_stratum_split": MIN_PER_LABEL_STRATUM_SPLIT,
                "caps_per_label_stratum_split": CAP_PER_LABEL_STRATUM_SPLIT,
                "matching_stratum": "consensus codon position | transcript-oriented ref>alt",
                **sampling_metadata,
            },
            "coordinate_system": {
                "source_pos": "1-based",
                "gtf": "1-based closed",
                "internal": "0-based half-open",
                "conversion": "pos0 = pos1 - 1; GTF start0 = start1 - 1, end0 = end1",
            },
            "local_resource_policy": resource_policy,
            "output": {
                "path_name": panel_path.name,
                "rows": panel.height,
                "bytes": panel_path.stat().st_size,
                "sha256": sha256_file(panel_path),
            },
        }
        write_json(output_dir / "manifest.json", result)
        result["manifest_sha256"] = sha256_file(output_dir / "manifest.json")
        write_json(output_dir / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_panel(
        source_path=args.source,
        gtf_path=args.gtf,
        fasta_path=args.fasta,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
