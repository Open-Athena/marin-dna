"""Verify the vendored issue #473 tokenizer on a real Iris child worker."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import click
from fray.types import ResourceConfig
from levanter.tokenizers import load_tokenizer
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import build_options

from exp473_center_seeded_projection.experiment import (
    DATA_VERSION,
    TOKENIZER_PATH,
    TOKENIZER_SHA256,
)


@dataclass(frozen=True)
class TokenizerWorkerPreflightConfig:
    """Content-addressed tokenizer contract sent to the child worker."""

    tokenizer_path: str
    sha256: dict[str, str]


def verify_tokenizer_on_worker(config: TokenizerWorkerPreflightConfig) -> None:
    """Hash and exercise the DNA tokenizer in the remote task environment."""
    for filename, expected in config.sha256.items():
        path = Path(config.tokenizer_path, filename)
        assert path.is_file(), f"remote tokenizer file is missing: {path}"
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == expected, f"remote tokenizer drift: {path}"

    tokenizer = load_tokenizer(config.tokenizer_path)
    assert tokenizer.vocab_size == 7
    assert tokenizer.bos_token_id == 2
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.eos_token_id is None
    encoded = tokenizer.as_hf_tokenizer()(
        "ACGTacgt",
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    assert encoded == [2, 3, 4, 5, 6, 3, 4, 5, 6]
    print(
        "issue #473 tokenizer worker preflight passed: "
        "vocab=7 bos=2 pad=0 unk=1 char_ids=3,4,5,6"
    )


def build_preflight() -> ArtifactStep[Artifact]:
    """Build one CPU-only child task with no scientific data dependency."""

    def build_config(_ctx: StepContext) -> TokenizerWorkerPreflightConfig:
        return TokenizerWorkerPreflightConfig(
            tokenizer_path=TOKENIZER_PATH,
            sha256=dict(TOKENIZER_SHA256),
        )

    return ArtifactStep(
        name="preflights/exp473-tokenizer-worker",
        version=DATA_VERSION,
        artifact_type=Artifact,
        run=remote(
            verify_tokenizer_on_worker,
            resources=ResourceConfig.with_cpu(cpu=1, ram="4g", disk="5g"),
        ),
        build_config=build_config,
    )


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep[Artifact]:
    """Return the CPU-only tokenizer worker preflight."""
    return build_preflight()


if __name__ == "__main__":
    main()
