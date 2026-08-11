"""Shared pytest fixtures for tests."""

from pathlib import Path

import pytest

from marin_dna.data.utils import load_annotation


@pytest.fixture
def chrY_annotation():
    """Load chrY annotation fixture for integration tests.

    Returns a polars DataFrame with all chrY annotations from
    GCF_000001405.40 (human genome). This fixture is useful for
    integration tests that need real annotation structure.
    """
    fixture_path = Path(__file__).parent / "fixtures" / "chrY_annotation.gtf.gz"
    if not fixture_path.exists():
        pytest.skip("chrY annotation fixture not found")
    return load_annotation(str(fixture_path))
