"""Retrieve public, versioned annotation candidates for exact train-row mapping."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pysam
import requests


def download(url: str, path: Path) -> None:
    """Download atomically; original DNA is retained only as verification context."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=90) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    temporary.replace(path)


def ncbi_directory(accession: str) -> str:
    """Resolve a versioned NCBI FTP directory without guessing assembly names."""
    digits = accession.split("_", 1)[1].split(".")[0]
    base = (
        f"https://ftp.ncbi.nlm.nih.gov/genomes/all/{accession[:3]}/"
        f"{digits[:3]}/{digits[3:6]}/{digits[6:]}/"
    )
    response = requests.get(base, timeout=45)
    response.raise_for_status()
    dirs = [
        n
        for n in re.findall(r'href="([^"?]+)"', response.text)
        if n.startswith(accession + "_") and n.endswith("/")
    ]
    if len(dirs) != 1:
        raise ValueError("Versioned NCBI assembly directory is ambiguous")
    return urljoin(base, dirs[0])


def prepare_reference(candidate: dict[str, Any], output: Path) -> dict[str, Any]:
    """Prefer a submitted reference annotation, with a paired RefSeq fallback."""
    code = candidate["assembly_code"]
    dest = output / code
    dest.mkdir(parents=True, exist_ok=True)
    saved = dest / "reference.json"
    if saved.exists():
        return json.loads(saved.read_text())
    failures = []
    for accession in candidate["accessions"]:
        try:
            directory = ncbi_directory(accession)
            response = requests.get(directory, timeout=45)
            response.raise_for_status()
            names = re.findall(r'href="([^"?]+)"', response.text)
            fasta_names = [
                n
                for n in names
                if n.endswith("_genomic.fna.gz") and "_from_genomic" not in n
            ]
            gff_names = [n for n in names if n.endswith("_genomic.gff.gz")]
            report_names = [n for n in names if n.endswith("_assembly_report.txt")]
            if not (len(fasta_names) == len(gff_names) == len(report_names) == 1):
                failures.append(
                    {"accession": accession, "reason": "missing_reference_assets"}
                )
                continue
            checksum_path = dest / f"{accession}.md5checksums.txt"
            download(urljoin(directory, "md5checksums.txt"), checksum_path)
            sums = {}
            for line in checksum_path.read_text().splitlines():
                digest, name = line.split(maxsplit=1)
                sums[name.removeprefix("./")] = digest
            assets = {}
            for kind, name in [
                ("genome_gzip", fasta_names[0]),
                ("gff", gff_names[0]),
                ("assembly_report", report_names[0]),
            ]:
                path = dest / name
                download(urljoin(directory, name), path)
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "md5").hexdigest()
                if digest != sums[name]:
                    raise ValueError("Official reference checksum differs: " + name)
                with path.open("rb") as stream:
                    sha = hashlib.file_digest(stream, "sha256").hexdigest()
                assets[kind] = {
                    "path": str(path),
                    "url": urljoin(directory, name),
                    "md5": digest,
                    "sha256": sha,
                    "bytes": path.stat().st_size,
                }
            fasta = dest / f"{accession}.genome.fa"
            if not fasta.exists():
                with (
                    gzip.open(assets["genome_gzip"]["path"], "rb") as source,
                    fasta.with_suffix(".part").open("wb") as target,
                ):
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                fasta.with_suffix(".part").replace(fasta)
            if not Path(str(fasta) + ".fai").exists():
                pysam.faidx(str(fasta))
            with pysam.FastaFile(str(fasta)) as reference:
                length = sum(reference.lengths)
            result = {
                **candidate,
                "accession": accession,
                "fasta": str(fasta),
                "assets": assets,
                "reference_length": length,
                "status": "candidate; exact existing-window mapping required",
            }
            saved.write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps(
                    {"assembly": code, "accession": accession, "reference_ready": True}
                ),
                flush=True,
            )
            return result
        except (requests.RequestException, ValueError, OSError) as exc:
            failures.append(
                {"accession": accession, "reason": type(exc).__name__ + ": " + str(exc)}
            )
    result = {"assembly_code": code, "status": "unavailable", "attempts": failures}
    (dest / "unavailable.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    """Resolve annotation candidates; a failed candidate never expands training DNA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    rows = json.loads(args.registry.read_text())
    candidates = []
    for rec in rows:
        if rec["assembly_code"] == "Zmays":
            continue  # The separately verified MaizeGDB reference is already present.
        exact = rec.get("historical_ncbi_exact_size_candidates", [])
        current = rec.get("current_ncbi_reference_candidate")
        accessions = [r["accession"] for r in exact]
        if not accessions and current:
            delta = abs(current["bp"] / rec["paper_genome_size"] - 1)
            if delta <= 0.01 and current["bp"] < 3_000_000_000:
                accessions = [current["accession"]]
        if accessions:
            candidates.append(
                {
                    "assembly_code": rec["assembly_code"],
                    "species": rec["species"],
                    "paper_genome_size": rec["paper_genome_size"],
                    "accessions": accessions,
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            pool.map(lambda c: prepare_reference(c, args.output), candidates)
        )
    (args.output / "references.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
