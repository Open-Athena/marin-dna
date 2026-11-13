import csv
import hashlib
import json
from collections import Counter
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
import pytest

from exp593_dna.development import (
    ConservationSite,
    Intervals,
    canonical_sequence_hash,
    conservation_candidates,
    select_matched_sites,
    training_intervals,
)
from exp593_dna.pipeline import DATASET_REVISION, reverse_complement


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_training_region_exclusion_verifies_every_original_metadata_file(tmp_path):
    path = tmp_path / "original.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "assembly": "Sbicolor",
                    "chrom": "Chr01",
                    "start": 1000,
                    "end": 2000,
                    "split": "train",
                },
                {
                    "assembly": "Zmays",
                    "chrom": "1",
                    "start": 0,
                    "end": 10000,
                    "split": "train",
                },
                {
                    "assembly": "Sbicolor",
                    "chrom": "Chr01",
                    "start": 1500,
                    "end": 3000,
                    "split": "train",
                },
            ]
        ),
        path,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_revision": DATASET_REVISION,
                "files": {
                    "metadata/train/original.parquet": {
                        "bytes": path.stat().st_size,
                        "sha256": sha(path),
                    }
                },
            }
        )
    )
    options = {
        "expected_rows": 3,
        "expected_shards": 1,
        "expected_manifest_sha256": sha(manifest),
    }
    regions, counts = training_intervals(tmp_path, manifest, **options)
    assert counts == {"all_training_rows_checked": 3, "sorghum_training_rows": 2}
    assert regions["1"].values == [(999, 3001)]
    assert regions["1"].touches(3001, 3513)
    assert not regions["1"].touches(3002, 3514)
    with pytest.raises(ValueError, match="coverage changed"):
        training_intervals(tmp_path, manifest, **{**options, "expected_rows": 4})
    path.write_bytes(path.read_bytes() + b"different")
    with pytest.raises(ValueError, match="metadata bytes"):
        training_intervals(tmp_path, manifest, **options)


def test_conservation_filter_binds_coordinates_reference_and_final_test_exclusions(
    tmp_path,
):
    rng = np.random.default_rng(593)
    genomes = {c: "".join(rng.choice(list("ACGT"), 20000)) for c in ("Chr1", "Chr3")}
    fasta = tmp_path / "genome.fa"
    fasta.write_text("".join(f">{c}\n{s}\n" for c, s in genomes.items()))
    pysam.faidx(str(fasta))
    report = tmp_path / "assembly_report.txt"
    report.write_text("")
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "assembly_code": "Sbicolor",
                "accession": "GCF_000003195.3",
                "fasta": str(fasta),
                "assets": {
                    "genome_gzip": {
                        "path": str(fasta),
                        "sha256": sha(fasta),
                        "url": "fixture://reference",
                    },
                    "assembly_report": {"path": str(report)},
                },
            }
        )
    )

    def row(chrom, pos, label):
        return {
            "chrom": chrom,
            "pos": pos,
            "label": label,
            "sequences": genomes[chrom][pos - 256 : pos + 256],
        }

    first = row("Chr1", 4000, 0)
    excluded = row("Chr3", 2000, 0)
    bad = row("Chr1", 12000, 1)
    bad["sequences"] = ("C" if bad["sequences"][0] != "C" else "A") + bad["sequences"][
        1:
    ]
    rows = [
        row("Chr1", 2000, 1),
        first,
        row("Chr1", 8000, 1),
        bad,
        {**first, "chrom": "Chr2"},
        excluded,
        first,
        {**first, "label": 1},
    ]
    source = tmp_path / "source.tsv"
    exclusions = tmp_path / "exclusions.tsv"
    for path, values in (
        (source, rows),
        (
            exclusions,
            [{**excluded, "sequences": reverse_complement(excluded["sequences"])}],
        ),
    ):
        with path.open("w") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["chrom", "pos", "label", "sequences"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(values)
    candidates, counts, provenance = conservation_candidates(
        source,
        exclusions,
        reference,
        {"1": Intervals([(1500, 2500)]), "3": Intervals([])},
        expected_source_sha256=sha(source),
        expected_exclusion_sha256=sha(exclusions),
    )
    assert len(candidates) == 1
    assert candidates[0].position_1based == 8000
    assert candidates[0].sequence[255] == genomes["Chr1"][7999]
    assert candidates[0].use == "evaluation_only"
    assert counts["training_source_region_overlap"] == 1
    assert counts["reference_sequence_mismatch"] == 1
    assert counts["final_benchmark_dna_overlap"] == 1
    assert counts["conflicting_dna_labels"] == 1
    assert provenance["accession"] == "GCF_000003195.3"
    with pytest.raises(ValueError, match="Invalid conservation"):
        replace(candidates[0], use="train")
    with pytest.raises(ValueError, match="Invalid conservation"):
        replace(candidates[0], start=candidates[0].start + 1)
    fasta.write_text(fasta.read_text() + "\n")
    with pytest.raises(ValueError, match="reference bytes"):
        conservation_candidates(
            source,
            exclusions,
            reference,
            {"1": Intervals([])},
            expected_source_sha256=sha(source),
            expected_exclusion_sha256=sha(exclusions),
        )


def test_matched_selection_is_deterministic_and_removes_context_overlap():
    rng = np.random.default_rng(593)
    candidates = []
    for row, (position, label) in enumerate(
        [(1000, 1), (1001, 1), (2000, 0), (5000, 1), (6000, 0)]
    ):
        sequence = list(rng.choice(list("ACGT"), 512))
        sequence[255] = "A"
        sequence = "".join(sequence)
        candidates.append(
            ConservationSite(
                row,
                "1",
                position,
                position - 256,
                position + 256,
                sequence,
                label,
                canonical_sequence_hash(sequence),
                hashlib.sha256(str(row).encode()).hexdigest(),
            )
        )
    selected = select_matched_sites(candidates, per_class=2)
    assert selected == select_matched_sites(list(reversed(candidates)), per_class=2)
    assert Counter(s.label for s in selected) == {0: 2, 1: 2}
    assert all(s.sequence[s.target_index] == "A" for s in selected)
    assert all(left.end < right.start for left, right in pairwise(selected))
    with pytest.raises(ValueError, match="Insufficient"):
        select_matched_sites(candidates, per_class=3)
