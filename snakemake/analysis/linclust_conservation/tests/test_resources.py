from pathlib import Path

from marin_dna_linclust_conservation.resources import record_peak_temporary_bytes


def test_record_peak_temporary_bytes_never_decreases(tmp_path: Path) -> None:
    directory = tmp_path / "temporary"
    directory.mkdir()
    output = tmp_path / "peak.txt"
    (directory / "large").write_bytes(b"x" * 10_000)
    first = record_peak_temporary_bytes(directory=directory, output_path=output)
    (directory / "large").write_bytes(b"x")
    second = record_peak_temporary_bytes(directory=directory, output_path=output)
    assert first >= 10_000
    assert second == first
    assert int(output.read_text()) == first
