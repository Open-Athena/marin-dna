"""Project public gene features onto exact, existing source windows only."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pysam

from exp593_dna.annotations import Landmark
from exp593_dna.pipeline import DATASET_REVISION, reverse_complement


@dataclass(frozen=True)
class GenomicMotif:
    """A biological feature interval before source-row strand/crop transforms."""

    chrom: str
    start: int
    end: int
    kind: str
    source: str
    feature_id: str


def normalized_name(name: str) -> str:
    """Explicit chromosome-only normalization; exact DNA still verifies mapping."""
    value = name.casefold()
    value = value.removeprefix("chr")
    return re.sub(r"^0+(?=[1-9])", "", value)


def contig_aliases(names: tuple[str, ...], report: Path | None) -> dict[str, str]:
    """Resolve aliases only when they identify a single actual reference record."""
    candidates: dict[str, set[str]] = defaultdict(set)
    for name in names:
        candidates[normalized_name(name)].add(name)
    if report:
        for line in report.read_text().splitlines():
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 9:
                continue
            aliases = [fields[i] for i in (0, 4, 6)]
            if len(fields) > 9:
                aliases.append(fields[9])
            if fields[1] == "assembled-molecule":
                aliases.append(fields[2])
            actual = [a for a in aliases if a in names]
            if len(set(actual)) != 1:
                continue
            for alias in aliases:
                if alias.lower() != "na":
                    candidates[normalized_name(alias)].add(actual[0])
    return {
        alias: next(iter(values))
        for alias, values in candidates.items()
        if len(values) == 1
    }


def parse_features(
    gff: Path,
    aliases: dict[str, str],
    source: str,
    statistics: Counter[str] | None = None,
) -> dict[str, list[GenomicMotif]]:
    """Infer codon bounds and introns per transcript, never from merged gene spans."""
    transcripts: dict[str, dict[str, Any]] = {}
    unsupported = set()
    with gzip.open(gff, "rt") if gff.suffix == ".gz" else gff.open() as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) != 9 or fields[2] not in (
                "CDS",
                "exon",
                "start_codon",
                "stop_codon",
            ):
                continue
            chrom = aliases.get(normalized_name(fields[0]))
            if chrom is None or fields[6] not in ("+", "-"):
                continue
            attrs = dict(x.split("=", 1) for x in fields[8].split(";") if "=" in x)
            for parent in attrs.get("Parent", "").split(","):
                if not parent:
                    continue
                if "trans-splicing" in attrs.get("exception", ""):
                    unsupported.add(parent)
                tx = transcripts.setdefault(
                    parent,
                    {
                        "chrom": chrom,
                        "strand": fields[6],
                        "CDS": [],
                        "exon": [],
                        "start_codon": [],
                        "stop_codon": [],
                        "partial": False,
                    },
                )
                if tx["chrom"] != chrom or tx["strand"] != fields[6]:
                    unsupported.add(parent)
                    continue
                tx[fields[2]].append((int(fields[3]) - 1, int(fields[4])))
                tx["partial"] |= attrs.get("partial") == "true"
    result: dict[str, list[GenomicMotif]] = defaultdict(list)
    for parent, tx in transcripts.items():
        if parent in unsupported:
            continue
        cds = sorted(set(tx["CDS"]))
        if not cds:
            continue
        chrom, strand = tx["chrom"], tx["strand"]

        def add(
            start: int,
            end: int,
            kind: str,
            *,
            chrom_name: str = chrom,
            feature_id: str = parent,
        ) -> None:
            if 0 <= start < end:
                result[chrom_name].append(
                    GenomicMotif(chrom_name, start, end, kind, source, feature_id)
                )

        if not tx["partial"]:
            low, high = cds[0][0], cds[-1][1]
            starts = tx["start_codon"] or (
                [(low, low + 3)] if strand == "+" else [(high - 3, high)]
            )
            stops = tx["stop_codon"] or (
                [(high - 3, high)] if strand == "+" else [(low, low + 3)]
            )
            for interval in starts:
                add(*interval, "translation_start")
            for interval in stops:
                add(*interval, "translation_stop")
        exons = sorted(set(tx["exon"] or cds))
        for before, after in pairwise(exons):
            if after[0] - before[1] < 2:
                continue
            left, right = (before[1], before[1] + 2), (after[0] - 2, after[0])
            add(*(left if strand == "+" else right), "splice_donor")
            add(*(right if strand == "+" else left), "splice_acceptor")
    for values in result.values():
        values.sort(key=lambda x: (x.start, x.end, x.kind, x.feature_id))
    if statistics is not None:
        statistics.update(
            {
                "transcripts": len(transcripts),
                "unsupported_trans_splicing_or_conflicting_transcripts": len(
                    unsupported
                ),
            }
        )
    return result


def map_reference(
    config: dict[str, Any],
    partitions: Path,
    output: Path,
    *,
    splits: tuple[str, ...] = ("train", "validation"),
) -> dict[str, Any]:
    """Require one exact DNA hit at a documented tile offset within the parent."""
    code = config["assembly_code"]
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / f"{code}.report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous["reference"] == config:
            return previous
    reference = pysam.FastaFile(config["fasta"])
    assembly_report = config.get("assets", {}).get("assembly_report", {}).get("path")
    aliases = contig_aliases(
        reference.references, Path(assembly_report) if assembly_report else None
    )
    # PlantCAD's peanut source prepended its release identifier to the official
    # chromosome/scaffold names. This is an explicit source alias, not a guessed
    # chromosome-number translation; exact full-window DNA still gates every row.
    for prefix in config.get("source_contig_prefixes", ()):
        aliases.update(
            {normalized_name(prefix + k): v for k, v in list(aliases.items())}
        )
    gff = Path(config["assets"]["gff"]["path"])
    source_id = f"{code}:{config['accession']}"
    feature_statistics: Counter[str] = Counter()
    features = parse_features(gff, aliases, source_id, feature_statistics)
    starts = {chrom: [m.start for m in values] for chrom, values in features.items()}
    counts: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    used_aliases = {}

    @lru_cache(maxsize=256)
    def parent_region(chrom: str, lo: int, hi: int) -> str:
        return reference.fetch(chrom, lo, hi).upper()

    for split in splits:
        source = partitions / split / f"{code}.parquet"
        if not source.exists():
            continue
        final_path = output / f"{code}.{split}.jsonl"
        temporary = final_path.with_suffix(".part")
        with temporary.open("w") as target:
            for batch in pq.ParquetFile(source).iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    counts[f"{split}/rows"] += 1
                    if row["split"] != split or row["assembly"] != code:
                        raise ValueError("Partition row identity changed")
                    chrom = aliases.get(normalized_name(row["chrom"]))
                    if chrom is None:
                        counts[f"{split}/unknown_contig"] += 1
                        continue
                    if row["end"] > reference.get_reference_length(chrom):
                        counts[f"{split}/coordinate_mismatch"] += 1
                        continue
                    used_aliases[row["chrom"]] = chrom
                    dna = row["seq"]
                    query = (
                        dna.upper()
                        if row["strand"] == "+"
                        else reverse_complement(dna).upper()
                    )
                    parent = parent_region(chrom, row["start"], row["end"])
                    hits = []
                    pos = parent.find(query)
                    span = len(parent)
                    tile_offsets = set(range(0, span - 8192 + 1, 4096)) | {span - 8192}
                    while pos >= 0:
                        offset = pos if row["strand"] == "+" else span - pos - 8192
                        if offset in tile_offsets:
                            hits.append(row["start"] + pos)
                        pos = parent.find(query, pos + 1)
                    if len(hits) != 1:
                        counts[f"{split}/nonunique_or_unmatched_DNA"] += 1
                        continue
                    counts[f"{split}/exact_DNA_mapping"] += 1
                    lo, hi = hits[0], hits[0] + 8192
                    chrom_features = features.get(chrom, [])
                    chrom_starts = starts.get(chrom, [])
                    selected = {}
                    for index in range(
                        bisect_left(chrom_starts, lo), bisect_left(chrom_starts, hi)
                    ):
                        motif = chrom_features[index]
                        if motif.end > hi:
                            continue
                        a, b = motif.start - lo, motif.end - lo
                        if row["strand"] == "-":
                            a, b = 8192 - b, 8192 - a
                        key = (motif.kind, a, b)
                        selected[key] = Landmark(
                            a, b, motif.kind, source_id, motif.feature_id
                        )
                    if not selected:
                        counts[f"{split}/no_landmarks"] += 1
                        continue
                    values = [asdict(m) for _, m in sorted(selected.items())]
                    counts[f"{split}/annotated_rows"] += 1
                    classes.update(f"{split}/{m['kind']}" for m in values)
                    target.write(
                        json.dumps(
                            {
                                "dataset_revision": DATASET_REVISION,
                                "split": split,
                                "row": row["row"],
                                "sequence_sha256": hashlib.sha256(
                                    dna.encode()
                                ).hexdigest(),
                                "landmarks": values,
                            }
                        )
                        + "\n"
                    )
        temporary.replace(final_path)
        print(
            json.dumps({"assembly": code, "split": split, "counts": dict(counts)}),
            flush=True,
        )
    reference.close()
    report = {
        "reference": config,
        "counts": dict(counts),
        "landmark_counts": dict(classes),
        "annotation_parsing": dict(feature_statistics),
        "explicit_contig_aliases": used_aliases,
        "mapping": "Unique exact case-insensitive 8192-base hit at an original tiling offset; original case preserved",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--maize-root", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    maize = args.maize_root
    maize_files = (
        ("genome_gzip", "Zm-B73-REFERENCE-NAM-5.0.fa.gz"),
        ("gff", "Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3.gz"),
    )
    maize_assets = {}
    for kind, name in maize_files:
        path = maize / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        maize_assets[kind] = {
            "path": str(path),
            "url": f"https://download.maizegdb.org/Zm-B73-REFERENCE-NAM-5.0/{name}",
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    configs = [
        {
            "assembly_code": "Zmays",
            "accession": "GCA_902167145.1/Zm00001eb.1",
            "fasta": str(maize / "Zm-B73-REFERENCE-NAM-5.0.fa.gz"),
            "assets": maize_assets,
        }
    ]
    configs += [
        json.loads(p.read_text())
        for p in sorted(args.references.glob("*/reference.json"))
    ]
    for config in configs:
        if config["assembly_code"] == "Ahypogaea":
            config["source_contig_prefixes"] = ["arahy.Tifrunner.gnm1."]
    reports = [map_reference(c, args.partitions, args.output) for c in configs]
    for split in ("train", "validation"):
        with (args.output / f"{split}.jsonl").open("w") as target:
            for path in sorted(args.output.glob(f"*.{split}.jsonl")):
                with path.open() as source:
                    for line in source:
                        target.write(line)
    (args.output / "coverage.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
