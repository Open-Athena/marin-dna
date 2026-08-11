"""Pipeline-specific logic for a new MarinDNA Snakemake project."""

from pathlib import Path


def write_example(output_path: str | Path) -> None:
    """Write the scaffold's deterministic smoke-test artifact."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("replace-me\n")
