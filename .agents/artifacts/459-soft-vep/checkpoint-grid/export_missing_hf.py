"""Export the two native checkpoints missing from issue #459's HF grid."""

import argparse
import json
from dataclasses import dataclass

import fsspec
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.export_lm_to_hf import ConvertLmConfig, main as export_lm_to_hf
from levanter.models.qwen import Qwen3Config
from levanter.trainer import TrainerConfig


@dataclass(frozen=True)
class ExportTarget:
    checkpoint_path: str
    output_dir: str


EXPORT_TARGETS = {
    "exp351-centered-1000": ExportTarget(
        checkpoint_path=(
            "gs://marin-us-east5/checkpoints/"
            "dna-exp351-zoonomia-v1-0p25b-centered-v0.1-8adcec/"
            "checkpoints/step-1000"
        ),
        output_dir=(
            "gs://marin-us-east5/checkpoints/"
            "dna-exp351-zoonomia-v1-0p25b-centered-v0.1-8adcec/hf/step-1000"
        ),
    ),
    "exp232-ncrna-2500": ExportTarget(
        checkpoint_path=(
            "gs://marin-us-east5/checkpoints/"
            "dna-exp232-zoonomia-v1-0p25b-v4_ncrna_exon-v0.1-c3ab58/"
            "checkpoints/step-2500"
        ),
        output_dir=(
            "gs://marin-us-east5/checkpoints/"
            "dna-exp232-zoonomia-v1-0p25b-v4_ncrna_exon-v0.1-c3ab58/"
            "hf/step-2500"
        ),
    ),
}

EXPECTED_ROPE_SCALING = {
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3",
}


def model_config() -> Qwen3Config:
    """Reconstruct the shared exp232/exp351 Qwen3-0.25B geometry."""
    return Qwen3Config(
        hidden_dim=1152,
        intermediate_dim=4608,
        num_layers=12,
        num_heads=9,
        num_kv_heads=9,
        max_seq_len=256,
        rope=Llama3RotaryEmbeddingsConfig(),
        initializer_range=0.02,
    )


def validate_hf_config(output_dir: str) -> dict[str, object]:
    """Fail closed on the Transformers-major RoPE corruption from issue #439."""
    with fsspec.open(f"{output_dir}/config.json", "rt") as config_file:
        config = json.load(config_file)

    assert config["transformers_version"] == "4.57.6"
    assert config["model_type"] == "qwen3"
    assert config["rope_theta"] == 500_000
    assert config["rope_scaling"] == EXPECTED_ROPE_SCALING
    assert config.get("rope_parameters") is None
    return config


def export_target(target: ExportTarget) -> None:
    """Export one target, refusing to overwrite an incomplete remote prefix."""
    fs, output_path = fsspec.core.url_to_fs(target.output_dir)
    existing = fs.find(output_path)
    if existing:
        validate_hf_config(target.output_dir)
        assert any(path.endswith(".safetensors") for path in existing)
        print(f"Validated existing export: {target.output_dir}")
        return

    export_lm_to_hf(
        ConvertLmConfig(
            trainer=TrainerConfig(),
            checkpoint_path=target.checkpoint_path,
            output_dir=target.output_dir,
            model=model_config(),
            tokenizer="bolinas-dna/tokenizer-char-bos",
            use_cpu=True,
        )
    )
    validate_hf_config(target.output_dir)
    exported = fs.find(output_path)
    assert any(path.endswith(".safetensors") for path in exported)
    print(f"Created and validated export: {target.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "targets",
        nargs="*",
        choices=tuple(EXPORT_TARGETS),
        default=list(EXPORT_TARGETS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in args.targets:
        print(f"Exporting {name}")
        export_target(EXPORT_TARGETS[name])


if __name__ == "__main__":
    main()
