from __future__ import annotations

import pytest

from design_panels import SPLITS, split_context_group


def test_split_context_groups_are_unique_and_explicit() -> None:
    groups = [split_context_group(split) for split in SPLITS]
    assert len(set(groups)) == len(SPLITS)
    assert groups == [
        "response_independent_discovery_hash",
        "response_independent_validation_hash",
        "response_independent_test_hash",
    ]


def test_split_context_group_rejects_unknown_split() -> None:
    with pytest.raises(AssertionError):
        split_context_group("train")
