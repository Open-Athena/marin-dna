"""Build and evaluate projected-homology fixtures embedded in real genomic tiles."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from marin_dna_linclust_conservation.mmseqs import parse_cluster_assignments
from marin_dna_linclust_conservation.smoke import parse_time_report
from marin_dna_linclust_conservation.staging import parse_s3_uri


@dataclass(frozen=True, slots=True)
class BackgroundFastaSource:
    """One immutable all-tiles FASTA used as scaling background."""

    label: str
    accession: str
    uri: str
    etag: str
    size_bytes: int
    record_count: int

    @classmethod
    def from_dict(cls, row: dict[str, object]) -> BackgroundFastaSource:
        return cls(
            label=str(row["label"]),
            accession=str(row["accession"]),
            uri=str(row["uri"]),
            etag=str(row["etag"]),
            size_bytes=int(row["size_bytes"]),
            record_count=int(row["record_count"]),
        )


def _iter_fasta_records(lines: Iterable[bytes]) -> Iterator[tuple[str, str]]:
    header: str | None = None
    sequence: list[str] = []
    for raw_line in lines:
        line = raw_line.decode("ascii").strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(sequence)
            header = line[1:].split(maxsplit=1)[0]
            assert header
            sequence = []
        else:
            assert header is not None, "FASTA sequence precedes first header"
            sequence.append(line)
    if header is not None:
        yield header, "".join(sequence)


def _write_record(
    handle: BinaryIO,
    digest: Any,
    *,
    identifier: str,
    sequence: str,
) -> int:
    assert identifier and not any(character.isspace() for character in identifier)
    assert len(sequence) == 255
    encoded = f">{identifier}\n{sequence}\n".encode()
    handle.write(encoded)
    digest.update(encoded)
    return len(encoded)


def build_background_fixture(
    *,
    sources: list[BackgroundFastaSource],
    records_per_source: dict[str, int],
    truth_fasta_path: str | Path,
    output_fasta_path: str | Path,
    s3_client: Any,
) -> dict[str, object]:
    """Stream pinned FASTA prefixes and append the bounded truth fixture."""
    assert sources
    assert len({source.label for source in sources}) == len(sources)
    assert set(records_per_source) == {source.label for source in sources}
    requested = {label: int(count) for label, count in records_per_source.items()}
    assert all(count > 0 for count in requested.values())

    output = Path(output_fasta_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    identifiers: set[str] = set()
    source_receipts: list[dict[str, object]] = []
    written_bytes = 0
    with output.open("wb") as handle:
        for source in sources:
            count = requested[source.label]
            assert count <= source.record_count
            bucket, key = parse_s3_uri(source.uri)
            metadata = s3_client.head_object(Bucket=bucket, Key=key)
            observed_etag = str(metadata["ETag"]).strip('"')
            observed_size = int(metadata["ContentLength"])
            assert observed_etag == source.etag
            assert observed_size == source.size_bytes
            response = s3_client.get_object(
                Bucket=bucket,
                Key=key,
                IfMatch=f'"{source.etag}"',
            )
            body = response["Body"]
            retained = 0
            try:
                records = _iter_fasta_records(body.iter_lines(chunk_size=1024 * 1024))
                for identifier, sequence in itertools.islice(records, count):
                    assert identifier.startswith(f"{source.accession}|")
                    assert identifier not in identifiers
                    identifiers.add(identifier)
                    written_bytes += _write_record(
                        handle,
                        digest,
                        identifier=identifier,
                        sequence=sequence,
                    )
                    retained += 1
            finally:
                body.close()
            assert retained == count, (source.label, retained, count)
            source_receipts.append(
                {
                    "accession": source.accession,
                    "etag": source.etag,
                    "label": source.label,
                    "records_selected": retained,
                    "source_record_count": source.record_count,
                    "size_bytes": source.size_bytes,
                    "uri": source.uri,
                }
            )

        truth_records = 0
        truth_lines = Path(truth_fasta_path).read_bytes().splitlines()
        for identifier, sequence in _iter_fasta_records(truth_lines):
            assert identifier not in identifiers
            identifiers.add(identifier)
            written_bytes += _write_record(
                handle,
                digest,
                identifier=identifier,
                sequence=sequence,
            )
            truth_records += 1

    assert truth_records > 0
    assert output.stat().st_size == written_bytes
    background_records = sum(requested.values())
    return {
        "background_record_count": background_records,
        "combined_fasta_bytes": written_bytes,
        "combined_fasta_sha256": digest.hexdigest(),
        "combined_sequence_count": background_records + truth_records,
        "sources": source_receipts,
        "truth_sequence_count": truth_records,
    }


def _read_truth(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows
    assert len({row["record_id"] for row in rows}) == len(rows)
    return rows


def evaluate_background_scaling(
    *,
    truth_path: str | Path,
    assignments_path: str | Path,
    fixture_receipt_path: str | Path,
    resources_path: str | Path,
) -> dict[str, object]:
    """Measure truth recovery and contamination after adding genomic decoys."""
    truth = _read_truth(truth_path)
    fixture = json.loads(Path(fixture_receipt_path).read_text())
    by_record = {row["record_id"]: row for row in truth}
    truth_ids = set(by_record)
    by_anchor: dict[str, set[str]] = defaultdict(set)
    for row in truth:
        by_anchor[row["query_name"]].add(row["record_id"])
    species_counts = {len(records) for records in by_anchor.values()}
    assert len(species_counts) == 1
    species_count = next(iter(species_counts))

    assignments = parse_cluster_assignments(assignments_path)
    assert assignments.height == int(fixture["combined_sequence_count"])
    truth_assignments = assignments.filter(assignments["member"].is_in(truth_ids))
    assert set(truth_assignments["member"].to_list()) == truth_ids
    truth_representatives = set(truth_assignments["representative"].to_list())
    relevant = assignments.filter(
        assignments["representative"].is_in(truth_representatives)
    )
    cluster_sizes = assignments.group_by("representative").len()["len"]

    cluster_members: dict[str, set[str]] = defaultdict(set)
    for representative, member in relevant.iter_rows():
        cluster_members[str(representative)].add(str(member))
    true_pairs = {
        frozenset(pair)
        for members in by_anchor.values()
        for pair in itertools.combinations(sorted(members), 2)
    }
    clustered_truth_pairs: set[frozenset[str]] = set()
    exact_anchor_clusters = 0
    impure_truth_clusters = 0
    contaminated_truth_clusters = 0
    decoy_members_in_truth_clusters = 0
    truth_records_clustered_with_decoys: set[str] = set()
    all_pairs_in_truth_clusters = 0
    for members in cluster_members.values():
        truth_members = members & truth_ids
        if not truth_members:
            continue
        decoy_members = members - truth_ids
        anchors = {by_record[member]["query_name"] for member in truth_members}
        if len(anchors) > 1:
            impure_truth_clusters += 1
        if decoy_members:
            contaminated_truth_clusters += 1
            decoy_members_in_truth_clusters += len(decoy_members)
            truth_records_clustered_with_decoys.update(truth_members)
        if (
            len(truth_members) == species_count
            and len(anchors) == 1
            and not decoy_members
        ):
            exact_anchor_clusters += 1
        for pair in itertools.combinations(sorted(truth_members), 2):
            clustered_truth_pairs.add(frozenset(pair))
        all_pairs_in_truth_clusters += len(members) * (len(members) - 1) // 2

    recovered_true_pairs = true_pairs & clustered_truth_pairs
    false_truth_pairs = clustered_truth_pairs - true_pairs
    time_records = parse_time_report(resources_path)
    return {
        "anchor_count": len(by_anchor),
        "background_record_count": int(fixture["background_record_count"]),
        "cluster_count": assignments["representative"].n_unique(),
        "combined_sequence_count": assignments.height,
        "contaminated_truth_cluster_count": contaminated_truth_clusters,
        "decoy_member_count_in_truth_clusters": decoy_members_in_truth_clusters,
        "exact_anchor_cluster_count": exact_anchor_clusters,
        "exact_anchor_recovery_fraction": exact_anchor_clusters / len(by_anchor),
        "false_truth_pair_count": len(false_truth_pairs),
        "impure_truth_cluster_count": impure_truth_clusters,
        "mmseqs_cpu_seconds": sum(
            float(record["user_seconds"]) + float(record["system_seconds"])
            for record in time_records
        ),
        "mmseqs_peak_rss_kib": max(
            int(record["maximum_rss_kib"]) for record in time_records
        ),
        "mmseqs_stage_resources": time_records,
        "mmseqs_wall_seconds": sum(
            float(record["elapsed_seconds"]) for record in time_records
        ),
        "pair_precision": (
            len(recovered_true_pairs) / len(clustered_truth_pairs)
            if clustered_truth_pairs
            else 1.0
        ),
        "recovered_true_pair_count": len(recovered_true_pairs),
        "singleton_cluster_count": int((cluster_sizes == 1).sum()),
        "singleton_sequence_fraction": int((cluster_sizes == 1).sum())
        / assignments.height,
        "strict_truth_cluster_pair_precision": (
            len(recovered_true_pairs) / all_pairs_in_truth_clusters
            if all_pairs_in_truth_clusters
            else 1.0
        ),
        "true_pair_count": len(true_pairs),
        "true_pair_recall": len(recovered_true_pairs) / len(true_pairs),
        "truth_record_count": len(truth_ids),
        "truth_record_decoy_contamination_fraction": len(
            truth_records_clustered_with_decoys
        )
        / len(truth_ids),
    }
