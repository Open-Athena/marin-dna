import json
from pathlib import Path

import pytest
from marin_dna_linclust_conservation.smoke import (
    parse_time_report,
    summarize_smoke,
)

PIPELINE_COMMIT = "a" * 40
CONFIG_SHA256 = "b" * 64
MMSEQS_CONFIGURATION = {"candidate_release": "18.8cc5c"}


def _write_smoke_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = {
        name: tmp_path / filename
        for name, filename in {
            "stats": "stats.json",
            "assignments": "clusters.tsv",
            "alignments": "alignments.tsv",
            "fasta": "windows.fasta",
            "version": "version.txt",
            "resources": "resources.txt",
            "peak": "peak.txt",
            "gate": "gate.json",
        }.items()
    }
    paths["stats"].write_text(
        json.dumps(
            {
                "accession": "GCF_1.1",
                "candidate_windows": 2,
                "retained_bases": 510,
                "retained_windows": 2,
            }
        )
    )
    paths["assignments"].write_text("first\tfirst\nsecond\tsecond\n")
    paths["alignments"].write_text(
        "first\tfirst\t1\t255\t1\t1\t1\t255\t1\t255\t0\t500\n"
        "second\tsecond\t1\t255\t1\t1\t1\t255\t1\t255\t0\t500\n"
    )
    paths["fasta"].write_text(f">first\n{'A' * 255}\n>second\n{'C' * 255}\n")
    paths["version"].write_text("18.8cc5c\n")
    paths["resources"].write_text(
        '\tCommand being timed: "mmseqs linclust"\n'
        "\tUser time (seconds): 1.25\n"
        "\tSystem time (seconds): 0.50\n"
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:02.00\n"
        "\tMaximum resident set size (kbytes): 1024\n"
    )
    paths["peak"].write_text("2048\n")
    paths["gate"].write_text(
        json.dumps(
            {
                "assignment_partition_stable": True,
                "configuration": MMSEQS_CONFIGURATION,
                "mmseqs_version": "18.8cc5c",
                "pipeline_commit": PIPELINE_COMMIT,
                "pipeline_config_sha256": CONFIG_SHA256,
                "release_gate_passed": True,
                "runs": [
                    {
                        "alignment_count": 11,
                        "record_count": 11,
                        "reverse_complement_alignment_verified": True,
                    }
                ],
            }
        )
    )
    return paths


def _summarize(paths: dict[str, Path]) -> dict[str, object]:
    return summarize_smoke(
        stats_paths=[paths["stats"]],
        assignments_path=paths["assignments"],
        alignments_path=paths["alignments"],
        fasta_path=paths["fasta"],
        mmseqs_version_path=paths["version"],
        resources_path=paths["resources"],
        peak_temporary_bytes_path=paths["peak"],
        release_gate_path=paths["gate"],
        expected_mmseqs_version="18.8cc5c",
        expected_configuration=MMSEQS_CONFIGURATION,
        pipeline_commit=PIPELINE_COMMIT,
        pipeline_config_sha256=CONFIG_SHA256,
    )


def test_parse_appended_gnu_time_reports(tmp_path: Path) -> None:
    report = tmp_path / "resources.txt"
    report.write_text(
        '\tCommand being timed: "mmseqs createdb input db"\n'
        "\tUser time (seconds): 1.25\n"
        "\tSystem time (seconds): 0.50\n"
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:02.00\n"
        "\tMaximum resident set size (kbytes): 1024\n"
        '\tCommand being timed: "mmseqs linclust db clusters tmp"\n'
        "\tUser time (seconds): 3.00\n"
        "\tSystem time (seconds): 1.00\n"
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03\n"
        "\tMaximum resident set size (kbytes): 2048\n"
    )
    records = parse_time_report(report)
    assert [record["elapsed_seconds"] for record in records] == [2.0, 3723.0]
    assert [record["maximum_rss_kib"] for record in records] == [1024, 2048]


def test_summarize_smoke_validates_counts_release_and_provenance(
    tmp_path: Path,
) -> None:
    paths = _write_smoke_fixture(tmp_path)
    receipt = _summarize(paths)
    assert receipt["retained_windows"] == 2
    assert receipt["pipeline_commit"] == PIPELINE_COMMIT
    assert receipt["release_gate_passed"] is True


def test_summarize_smoke_rejects_missing_assignment(tmp_path: Path) -> None:
    paths = _write_smoke_fixture(tmp_path)
    paths["assignments"].write_text("first\tfirst\n")
    paths["alignments"].write_text(
        "first\tfirst\t1\t255\t1\t1\t1\t255\t1\t255\t0\t500\n"
    )
    with pytest.raises(AssertionError):
        _summarize(paths)


def test_summarize_smoke_rejects_wrong_release(tmp_path: Path) -> None:
    paths = _write_smoke_fixture(tmp_path)
    paths["version"].write_text("wrong\n")
    with pytest.raises(AssertionError):
        _summarize(paths)
