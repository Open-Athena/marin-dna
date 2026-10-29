"""Durably persist compact inference benchmark evidence after every trial."""

from pathlib import Path

import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3


def upload_trial_artifacts(
    output: Path,
    ledger: Path,
    prefix: str,
    *,
    filesystem: fsspec.AbstractFileSystem | None = None,
) -> list[str]:
    """Upload one result, its optional reference, and the current ledger."""
    if not prefix or prefix.endswith("/"):
        raise ValueError("Results prefix must be nonempty and omit a trailing slash")
    if not output.is_file():
        raise RuntimeError(f"Benchmark output does not exist: {output}")
    if filesystem is None:
        configure_coreweave_s3()
        filesystem = fsspec.filesystem("s3")

    candidates = (output, output.with_suffix(".reference.npz"), ledger)
    uploaded: list[str] = []
    for source in candidates:
        if not source.is_file():
            continue
        destination = f"{prefix}/{source.name}"
        filesystem.put_file(str(source), destination)
        if int(filesystem.info(destination)["size"]) != source.stat().st_size:
            raise RuntimeError(f"Incomplete uploaded benchmark artifact: {destination}")
        uploaded.append(destination)
    return uploaded
