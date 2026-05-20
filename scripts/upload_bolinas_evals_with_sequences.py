#!/usr/bin/env python
"""Upload sequence-materialized marin_dna eval datasets to HuggingFace.

This script loads existing marin_dna eval datasets, adds ref/alt sequence windows
(centered, max length 4096 by default), and pushes them to a target HF repo.
"""

from __future__ import annotations

import argparse
import logging
import gzip
import shutil
import urllib.request
from pathlib import Path

from datasets import DatasetDict, load_dataset


LOG = logging.getLogger("bolinas_upload")

GENOME_URL = "http://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"

_GENOME_INDEX = None
_GENOME_PATH = None


class FastaRecord:
    def __init__(
        self, name: str, length: int, offset: int, line_blen: int, line_len: int
    ):
        self.name = name
        self.length = length
        self.offset = offset
        self.line_blen = line_blen
        self.line_len = line_len


class FastaIndex:
    def __init__(self, fasta_path: Path, index_path: Path):
        self.fasta_path = fasta_path
        self.index_path = index_path
        self.records = self._load_or_build()

    def _load_or_build(self) -> dict[str, FastaRecord]:
        if self.index_path.exists():
            return self._load_index()
        records = self._build_index()
        self._write_index(records)
        return records

    def _load_index(self) -> dict[str, FastaRecord]:
        records: dict[str, FastaRecord] = {}
        with open(self.index_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                name, length, offset, line_blen, line_len = line.rstrip("\n").split(
                    "\t"
                )
                records[name] = FastaRecord(
                    name=name,
                    length=int(length),
                    offset=int(offset),
                    line_blen=int(line_blen),
                    line_len=int(line_len),
                )
        return records

    def _write_index(self, records: dict[str, FastaRecord]) -> None:
        with open(self.index_path, "w", encoding="utf-8") as handle:
            for name, record in records.items():
                handle.write(
                    f"{name}\t{record.length}\t{record.offset}\t{record.line_blen}\t{record.line_len}\n"
                )

    def _build_index(self) -> dict[str, FastaRecord]:
        LOG.info("Building FASTA index for %s", self.fasta_path)
        records: dict[str, FastaRecord] = {}
        with open(self.fasta_path, "rb") as handle:
            name = None
            seq_len = 0
            line_blen = None
            line_len = None
            seq_offset = None
            while True:
                line = handle.readline()
                if not line:
                    break
                if line.startswith(b">"):
                    if (
                        name is not None
                        and seq_offset is not None
                        and line_blen is not None
                        and line_len is not None
                    ):
                        records[name] = FastaRecord(
                            name=name,
                            length=seq_len,
                            offset=seq_offset,
                            line_blen=line_blen,
                            line_len=line_len,
                        )
                    name = line[1:].split()[0].decode("utf-8")
                    seq_len = 0
                    line_blen = None
                    line_len = None
                    seq_offset = handle.tell()
                    continue
                if name is None:
                    continue
                raw_len = len(line)
                if raw_len == 0:
                    continue
                stripped = line.rstrip(b"\r\n")
                if line_blen is None:
                    line_blen = len(stripped)
                    line_len = raw_len
                seq_len += len(stripped)

            if (
                name is not None
                and seq_offset is not None
                and line_blen is not None
                and line_len is not None
            ):
                records[name] = FastaRecord(
                    name=name,
                    length=seq_len,
                    offset=seq_offset,
                    line_blen=line_blen,
                    line_len=line_len,
                )
        return records

    def fetch(self, name: str, start: int, end: int) -> str:
        record = self.records[name]
        start = max(0, start)
        end = min(end, record.length)
        if end <= start:
            return ""
        remaining = end - start
        pos = start
        chunks: list[bytes] = []
        with open(self.fasta_path, "rb") as handle:
            while remaining > 0:
                line_idx = pos // record.line_blen
                line_pos = pos % record.line_blen
                byte_pos = record.offset + line_idx * record.line_len + line_pos
                handle.seek(byte_pos)
                take = min(remaining, record.line_blen - line_pos)
                chunks.append(handle.read(take))
                pos += take
                remaining -= take
        return b"".join(chunks).decode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload marin_dna eval datasets with precomputed sequences to HF."
    )
    parser.add_argument(
        "--source-prefix",
        default="gonzalobenegas/bolinas_evals",
        help="HF dataset prefix for source datasets (default: gonzalobenegas/bolinas_evals)",
    )
    parser.add_argument(
        "--datasets",
        default="traitgym_mendelian,traitgym_complex,clinvar_missense,gnomad_pls_v1,gnomad_pls_v2,gwas_coding",
        help="Comma-separated dataset names to process.",
    )
    parser.add_argument(
        "--target-repo",
        default="WillHeld/bolinas-evals",
        help="HF repo to upload into (configs will be dataset names).",
    )
    parser.add_argument(
        "--genome",
        default=GENOME_URL,
        help="Path or URL to reference genome FASTA (can be .fa.gz).",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/bolinas-evals",
        help="Cache directory for downloaded genome/index files.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=4096,
        help="Max window length around variant (default: 4096).",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Number of processes for datasets.map (default: 1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process datasets but do not upload to HF.",
    )
    return parser.parse_args()


def ensure_local_genome(genome_path: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if genome_path.startswith("http://") or genome_path.startswith("https://"):
        filename = Path(genome_path).name
        local_path = cache_dir / filename
        if not local_path.exists():
            LOG.info("Downloading genome from %s", genome_path)
            urllib.request.urlretrieve(genome_path, local_path)
        return _ensure_uncompressed(local_path, cache_dir)
    return _ensure_uncompressed(Path(genome_path), cache_dir)


def _ensure_uncompressed(path: Path, cache_dir: Path) -> Path:
    if path.suffix != ".gz":
        return path
    target = cache_dir / path.with_suffix("").name
    if target.exists():
        return target
    LOG.info("Decompressing genome to %s", target)
    with gzip.open(path, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def get_genome_index(genome_path: Path):
    global _GENOME_INDEX, _GENOME_PATH
    if _GENOME_INDEX is None or _GENOME_PATH != genome_path:
        index_path = genome_path.with_suffix(genome_path.suffix + ".fai")
        LOG.info("Indexing genome: %s", genome_path)
        _GENOME_INDEX = FastaIndex(genome_path, index_path)
        _GENOME_PATH = genome_path
    return _GENOME_INDEX


def resolve_chrom(index, chrom: str) -> str:
    if chrom in index.records:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in index.records:
        return chrom[3:]
    prefixed = f"chr{chrom}"
    if prefixed in index.records:
        return prefixed
    raise KeyError(f"Chromosome '{chrom}' not found in genome index.")


def compute_window_bounds(center: int, window: int, chrom_len: int) -> tuple[int, int]:
    half = window // 2
    start = center - half
    end = start + window
    if start < 0:
        start = 0
        end = min(window, chrom_len)
    if end > chrom_len:
        end = chrom_len
        start = max(0, end - window)
    return start, end


def materialize_sequences(batch, genome_path: Path, window: int):
    index = get_genome_index(genome_path)
    ref_seqs = []
    alt_seqs = []
    effects = []
    for chrom, pos, ref, alt, label in zip(
        batch["chrom"],
        batch["pos"],
        batch["ref"],
        batch["alt"],
        batch.get("label", [None] * len(batch["chrom"])),
        strict=True,
    ):
        chrom_key = resolve_chrom(index, str(chrom))
        record = index.records[chrom_key]
        chrom_len = record.length
        ref = str(ref)
        alt = str(alt)

        # pos is 1-based in marin_dna datasets
        variant_start = int(pos) - 1
        center = variant_start + max(len(ref) - 1, 0) // 2
        window_start, window_end = compute_window_bounds(center, window, chrom_len)

        ref_seq = index.fetch(chrom_key, window_start, window_end)
        within_window = variant_start - window_start
        if within_window < 0 or within_window + len(ref) > len(ref_seq):
            raise ValueError(
                f"Variant at {chrom}:{pos} not within window [{window_start}, {window_end})"
            )

        ref_seq_window = ref_seq[within_window : within_window + len(ref)]
        if ref_seq_window.upper() != ref.upper():
            LOG.warning(
                "Reference mismatch at %s:%s (expected %s, got %s)",
                chrom,
                pos,
                ref,
                ref_seq_window,
            )

        alt_seq = ref_seq[:within_window] + alt + ref_seq[within_window + len(ref) :]

        ref_seqs.append(ref_seq)
        alt_seqs.append(alt_seq)
        effects.append(label)

    return {"ref_seq": ref_seqs, "alt_seq": alt_seqs, "effect": effects}


def process_dataset(
    dataset_name: str,
    source_prefix: str,
    genome_path: Path,
    window: int,
    num_proc: int,
):
    source_path = f"{source_prefix}-{dataset_name}"
    LOG.info("Loading %s", source_path)
    dataset = load_dataset(source_path)

    # Build index in main process before spawning workers to avoid race condition
    get_genome_index(genome_path)

    def _map_fn(batch):
        return materialize_sequences(batch, genome_path, window)

    processed = DatasetDict()
    for split, split_ds in dataset.items():
        LOG.info("Processing %s split: %s", dataset_name, split)
        processed_split = split_ds.map(
            _map_fn,
            batched=True,
            num_proc=num_proc,
            desc=f"materialize {dataset_name}:{split}",
        )
        # Remove original label column in favor of effect
        if "label" in processed_split.column_names:
            processed_split = processed_split.remove_columns(["label"])
        processed[split] = processed_split

    return processed


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cache_dir = Path(args.cache_dir)
    genome_path = ensure_local_genome(args.genome, cache_dir)
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]

    for dataset_name in datasets:
        processed = process_dataset(
            dataset_name=dataset_name,
            source_prefix=args.source_prefix,
            genome_path=genome_path,
            window=args.window,
            num_proc=args.num_proc,
        )

        if args.dry_run:
            LOG.info("Dry run: skipping upload for %s", dataset_name)
            continue

        LOG.info(
            "Uploading %s to %s (config: %s)",
            dataset_name,
            args.target_repo,
            dataset_name,
        )
        processed.push_to_hub(args.target_repo, config_name=dataset_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
