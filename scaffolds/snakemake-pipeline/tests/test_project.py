from pathlib import Path

from marin_dna_pipeline import write_example


def test_write_example(tmp_path: Path) -> None:
    output = tmp_path / "results" / "example.txt"

    write_example(output)

    assert output.read_text() == "replace-me\n"
