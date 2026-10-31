"""Guard the authorized production-size comparison and artifact isolation."""

import pytest

from exp586_moe.dp_profile import DPConfig


def test_production_size_comparison_is_separate_from_one_node_identity():
    small = DPConfig("scatter", "cw-rno2a", batch_size=64, verification=True)
    production = DPConfig(
        "scatter",
        "cw-rno2a",
        batch_size=64,
        verification=True,
        nodes=8,
        ep_reference="live",
        end_to_end=True,
    )
    assert small.run_id != production.run_id
    assert small.job_name != production.job_name
    assert production.run_id.endswith("-b64-verify-n8-live")
    assert "/tmp/ttl=7d/" in production.artifact_root
    assert production.run_id in production.artifact_root


@pytest.mark.parametrize(
    "overrides",
    [
        {"nodes": 8, "batch_size": 8},
        {"nodes": 2},
        {"nodes": 8, "batch_size": 64, "end_to_end": True},
        {"ep_reference": "unknown"},
    ],
)
def test_reject_unsupported_comparisons(overrides):
    with pytest.raises(ValueError):
        DPConfig("scatter", "cw-rno2a", **overrides)
