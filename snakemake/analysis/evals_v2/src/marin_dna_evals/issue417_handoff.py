"""Fail-closed artifact validation for the issue #417 evaluation handoff."""

from __future__ import annotations

import json

EXPECTED_HF_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
MIN_MODEL_BYTES = 1_000_000_000


def parse_gsutil_listing(listing: str) -> dict[str, int]:
    """Parse gsutil long-listing output into an exact URI-to-size mapping."""
    objects: dict[str, int] = {}
    for raw_line in listing.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("TOTAL:"):
            continue
        fields = line.split(maxsplit=2)
        assert len(fields) == 3, f"unexpected gsutil listing line: {line!r}"
        size_text, _timestamp, uri = fields
        assert size_text.isdecimal(), f"non-numeric object size: {line!r}"
        assert uri.startswith("gs://"), f"unexpected object URI: {uri!r}"
        assert uri not in objects, f"duplicate object in listing: {uri}"
        objects[uri] = int(size_text)
    return objects


def validate_hf_export_listing(prefix: str, listing: str) -> dict[str, int]:
    """Validate one complete four-file Levanter Hugging Face export."""
    root = prefix.rstrip("/")
    expected = {f"{root}/{name}" for name in EXPECTED_HF_FILES}
    objects = parse_gsutil_listing(listing)
    observed = set(objects)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    assert not missing, f"HF export is missing objects: {missing}"
    assert not unexpected, f"HF export has unexpected objects: {unexpected}"

    sizes = {name: objects[f"{root}/{name}"] for name in EXPECTED_HF_FILES}
    assert all(size > 0 for size in sizes.values()), (
        f"HF export has empty files: {sizes}"
    )
    assert sizes["model.safetensors"] >= MIN_MODEL_BYTES, (
        f"HF model export is unexpectedly small: {sizes['model.safetensors']:,} bytes"
    )
    return sizes


def parse_sky_status_json(output: str) -> list[dict[str, object]]:
    """Extract Sky's JSON array after any human-readable warning lines."""
    lines = output.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.startswith("[")),
        None,
    )
    assert start is not None, f"Sky status output has no JSON payload: {output!r}"
    payload = json.loads("\n".join(lines[start:]))
    assert isinstance(payload, list), f"Sky status payload is not a list: {payload!r}"
    assert all(isinstance(row, dict) for row in payload), (
        f"Sky status payload contains non-object rows: {payload!r}"
    )
    return payload
