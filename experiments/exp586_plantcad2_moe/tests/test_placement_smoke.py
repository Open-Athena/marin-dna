import dataclasses
import hashlib
import json

import pytest

from exp586_moe.placement_smoke import PlacementSmokeConfig, verify_source
from exp586_moe.train import TrainingConfig, config_json, scientific_config


@pytest.mark.parametrize(
    "condition,multiplier", [("pretrained", 1.0), ("scratch", 1.0)]
)
def test_placement_smoke_keeps_training_identity_and_source_separate(
    condition, multiplier
):
    config = PlacementSmokeConfig(
        condition,
        f"exp586-{condition}-placement-smoke",
        "cw-rno2a",
        source_metadata_sha256="a" * 64,
    )
    source = config.source_config
    assert source.lr_multiplier == multiplier
    assert source.nodes == 2 and source.context_axis_size == 1
    assert scientific_config(source, "tokenizer") == scientific_config(
        dataclasses.replace(source, nodes=4), "tokenizer"
    )
    assert config.source_checkpoint == source.checkpoint_root + "/step-4123"
    assert "/tmp/ttl=7d/" in config.checkpoint_root
    assert config.checkpoint_root not in config.source_checkpoint
    with pytest.raises(ValueError, match="separate"):
        dataclasses.replace(config, run_id=source.run_id)
    with pytest.raises(ValueError, match="SHA-256"):
        dataclasses.replace(config, source_metadata_sha256="")


def test_placement_source_rejects_temporary_or_changed_checkpoint(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        TrainingConfig, "checkpoint_root", property(lambda self: str(tmp_path))
    )
    config = PlacementSmokeConfig(
        "pretrained",
        "exp586-placement-smoke",
        "cw-rno2a",
        source_metadata_sha256="a" * 64,
    )
    binding = config_json(scientific_config(config.source_config, "tokenizer"))
    digest = hashlib.sha256(binding.encode()).hexdigest()
    (tmp_path / "training.json").write_text(binding)
    root = tmp_path / "step-4123"
    root.mkdir()
    metadata = {
        "scientific_config_sha256": digest,
        "step": 4123,
        "is_temporary": False,
    }
    blob = json.dumps(metadata).encode()
    (root / "metadata.json").write_bytes(blob)
    config = dataclasses.replace(
        config, source_metadata_sha256=hashlib.sha256(blob).hexdigest()
    )
    assert verify_source(config, "tokenizer") == digest
    with pytest.raises(ValueError, match="scientific"):
        verify_source(config, "different-tokenizer")
    metadata["is_temporary"] = True
    blob = json.dumps(metadata).encode()
    (root / "metadata.json").write_bytes(blob)
    with pytest.raises(ValueError, match="digest"):
        verify_source(config, "tokenizer")
    config = dataclasses.replace(
        config, source_metadata_sha256=hashlib.sha256(blob).hexdigest()
    )
    with pytest.raises(ValueError, match="permanence"):
        verify_source(config, "tokenizer")
