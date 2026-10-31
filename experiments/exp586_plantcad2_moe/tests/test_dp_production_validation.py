import hashlib

import pytest

from exp586_moe.dp_production_validation import (
    RUNTIME_TRANSITION_ID,
    ValidationConfig,
)
from exp586_moe.train import config_json, scientific_config


def test_dp_validation_uses_the_production_scientific_binding() -> None:
    config = ValidationConfig(
        cluster="cw-rno2a",
        validation_id="a1",
        source_checkpoint=("s3://bucket/checkpoints/step-812708"),
        source_metadata_sha256="a" * 64,
        runtime_code_sha256="b" * 64,
    )
    training = config.training
    assert training.runtime_data_parallel
    assert training.runtime_transition_id == RUNTIME_TRANSITION_ID
    assert training.run_id == "exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1"
    assert training.model.moe_implementation == "scatter"
    assert (
        hashlib.sha256(
            config_json(
                scientific_config(
                    training,
                    "d1cf7a63f3e0e0778a7ef137020e6f39d04be1e6da14fe22d6283eb36cddae5e",
                )
            ).encode()
        ).hexdigest()
        == "6a5273b56f94f5da995a882b44208678dbbb8baf8871a4c3ccc37c286f222899"
    )


def test_dp_validation_requires_production_topology_and_checkpoint() -> None:
    kwargs = {
        "cluster": "cw-rno2a",
        "validation_id": "a1",
        "source_checkpoint": "s3://bucket/checkpoints/step-1",
        "source_metadata_sha256": "a" * 64,
        "runtime_code_sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="64-H100"):
        ValidationConfig(**kwargs, nodes=4)
    with pytest.raises(ValueError, match="committed production checkpoint"):
        ValidationConfig(**{**kwargs, "source_checkpoint": "s3://bucket/latest"})
