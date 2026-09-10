"""Prepare local harness stubs for an offline CI DAG dry-run only."""

import argparse
from pathlib import Path
from tempfile import mkdtemp

import yaml


def prepare_overlay(config_path: Path, output_directory: Path) -> Path:
    """Preserve registered models while replacing RAG harness URIs with local stubs.

    The stubs contain no data and cannot pass the production checksum validation.
    Only use this overlay with Snakemake's dry-run option.
    """
    config = yaml.safe_load(config_path.read_text())
    output_directory.mkdir(parents=True, exist_ok=True)
    temporary = Path(mkdtemp(prefix="evals-v2-dry-run-", dir=output_directory))
    models = config["models"]
    for index, model in enumerate(models):
        if model.get("inference_backend") != "rag_combined":
            continue
        harness = temporary / f"harness-{index}.parquet"
        harness.touch()
        model["rag_harness"]["uri"] = str(harness.resolve())
    overlay = temporary / "config.yaml"
    overlay.write_text(yaml.safe_dump({"models": models}, sort_keys=False))
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    print(prepare_overlay(args.config, args.output_directory))


if __name__ == "__main__":
    main()
