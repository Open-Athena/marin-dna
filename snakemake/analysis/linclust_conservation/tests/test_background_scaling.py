import json
from pathlib import Path

from marin_dna_linclust_conservation.background_scaling import (
    BackgroundFastaSource,
    build_background_fixture,
    evaluate_background_scaling,
)


class _Body:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.closed = False

    def iter_lines(self, *, chunk_size: int) -> list[bytes]:
        assert chunk_size > 0
        return self.payload.splitlines()

    def close(self) -> None:
        self.closed = True


class _S3:
    def __init__(self, objects: dict[tuple[str, str], bytes]) -> None:
        self.objects = objects

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        payload = self.objects[(Bucket, Key)]
        return {"ETag": '"etag"', "ContentLength": len(payload)}

    def get_object(self, *, Bucket: str, Key: str, IfMatch: str) -> dict[str, object]:
        assert IfMatch == '"etag"'
        return {"Body": _Body(self.objects[(Bucket, Key)])}


def _fasta(identifier: str, base: str) -> bytes:
    return f">{identifier}\n{base * 255}\n".encode()


def test_build_background_fixture_streams_balanced_prefixes(tmp_path: Path) -> None:
    human = _fasta("H|chr1|0|255|+", "A") + _fasta("H|chr1|128|383|+", "C")
    mouse = _fasta("M|chr1|0|255|+", "G")
    s3 = _S3({("bucket", "human.fa"): human, ("bucket", "mouse.fa"): mouse})
    sources = [
        BackgroundFastaSource(
            "human", "H", "s3://bucket/human.fa", "etag", len(human), 2
        ),
        BackgroundFastaSource(
            "mouse", "M", "s3://bucket/mouse.fa", "etag", len(mouse), 1
        ),
    ]
    truth = tmp_path / "truth.fa"
    truth.write_bytes(_fasta("anchor000000__human", "T"))
    output = tmp_path / "combined.fa"

    receipt = build_background_fixture(
        sources=sources,
        records_per_source={"human": 2, "mouse": 1},
        truth_fasta_path=truth,
        output_fasta_path=output,
        s3_client=s3,
    )

    assert receipt["background_record_count"] == 3
    assert receipt["combined_sequence_count"] == 4
    assert receipt["truth_sequence_count"] == 1
    assert output.read_bytes() == human + mouse + truth.read_bytes()


def test_evaluate_background_scaling_penalizes_decoy_contamination(
    tmp_path: Path,
) -> None:
    truth = tmp_path / "truth.tsv"
    truth.write_text(
        "anchor_index\tquery_name\trecord_id\tsource_label\tspecies\tassembly\n"
        "0\tq0\ta0h\th\thuman\thg\n"
        "0\tq0\ta0m\tm\tmouse\tmm\n"
        "1\tq1\ta1h\th\thuman\thg\n"
        "1\tq1\ta1m\tm\tmouse\tmm\n"
    )
    assignments = tmp_path / "clusters.tsv"
    assignments.write_text("a0h\ta0h\na0h\ta0m\na0h\tdecoy\na1h\ta1h\na1m\ta1m\n")
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps({"background_record_count": 1, "combined_sequence_count": 5})
    )
    resources = tmp_path / "resources.txt"
    resources.write_text(
        '\tCommand being timed: "mmseqs linclust"\n'
        "\tUser time (seconds): 1.00\n"
        "\tSystem time (seconds): 0.25\n"
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:02.00\n"
        "\tMaximum resident set size (kbytes): 1024\n"
    )

    result = evaluate_background_scaling(
        truth_path=truth,
        assignments_path=assignments,
        fixture_receipt_path=fixture,
        resources_path=resources,
    )

    assert result["true_pair_recall"] == 0.5
    assert result["pair_precision"] == 1.0
    assert result["exact_anchor_recovery_fraction"] == 0.0
    assert result["contaminated_truth_cluster_count"] == 1
    assert result["truth_record_decoy_contamination_fraction"] == 0.5
    assert result["strict_truth_cluster_pair_precision"] == 1 / 3
