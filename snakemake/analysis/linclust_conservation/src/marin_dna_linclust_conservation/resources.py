"""Resource high-water-mark helpers for bounded workflow receipts."""

from __future__ import annotations

import subprocess
from pathlib import Path


def record_peak_temporary_bytes(
    *,
    directory: str | Path,
    output_path: str | Path,
) -> int:
    """Update a durable peak-byte counter from the current GNU `du` measurement."""
    result = subprocess.run(
        ["du", "-sb", str(directory)],
        check=True,
        capture_output=True,
        text=True,
    )
    current_bytes = int(result.stdout.split("\t", maxsplit=1)[0])
    output = Path(output_path)
    previous_peak = int(output.read_text().strip()) if output.exists() else 0
    peak = max(previous_peak, current_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{peak}\n")
    return peak
