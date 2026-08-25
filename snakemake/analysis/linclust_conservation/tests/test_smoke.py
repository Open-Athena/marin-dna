from pathlib import Path

from marin_dna_linclust_conservation.smoke import parse_time_report


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
