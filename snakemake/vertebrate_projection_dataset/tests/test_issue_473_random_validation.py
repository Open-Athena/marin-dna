from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import zstandard as zstd
from marin_dna_vertebrate_projection.issue_473.random_validation import (
    validate_publication,
    write_dataset_card,
    write_uniform_random_split,
)
from marin_dna_vertebrate_projection.projection.dataset import prepare_shards


def _source(rows: int = 30) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_name": [f"anchor_{index // 3}" for index in range(rows)],
            "source_chrom": [f"chr{index % 5 + 1}" for index in range(rows)],
            "species": [f"species_{index % 3}" for index in range(rows)],
            "region_label": ["cds"] * rows,
            "sequence": [("ACgt" + "A" * 251) for _ in range(rows)],
        }
    )


def _compress(path: Path) -> None:
    compressed = path.with_suffix(path.suffix + ".zst")
    compressed.write_bytes(zstd.ZstdCompressor().compress(path.read_bytes()))
    path.unlink()


def test_uniform_split_precedes_reverse_complement_augmentation(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _source().write_parquet(source)
    train = tmp_path / "train.parquet"
    validation = tmp_path / "validation.parquet"
    summary_path = tmp_path / "summary.json"

    summary = write_uniform_random_split(
        source,
        train,
        validation,
        summary_path,
        region_label="cds",
        validation_rows=8,
        seed=42,
        target_length=255,
    )
    assert summary["source_rows"] == 30
    assert summary["train_original_rows"] == 22
    assert summary["published_train_rows"] == 44
    assert summary["validation_rows"] == 8
    assert "augmentation" not in pl.read_parquet_schema(train)
    assert set(pl.read_parquet(validation)["augmentation"]) == {"+"}
    assert json.loads(summary_path.read_text()) == summary

    repeated_validation = tmp_path / "validation_repeated.parquet"
    write_uniform_random_split(
        source,
        tmp_path / "train_repeated.parquet",
        repeated_validation,
        tmp_path / "summary_repeated.json",
        region_label="cds",
        validation_rows=8,
        seed=42,
        target_length=255,
    )
    assert pl.read_parquet(validation).equals(pl.read_parquet(repeated_validation))


def test_random_validation_publication_reconciles_exact_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _source(rows=16_390).write_parquet(source)
    train = tmp_path / "train.parquet"
    validation = tmp_path / "validation.parquet"
    summary = tmp_path / "summary.json"
    write_uniform_random_split(
        source,
        train,
        validation,
        summary,
        region_label="cds",
        validation_rows=16_384,
        seed=42,
        target_length=255,
    )

    artifact_root = tmp_path / "hf"
    cohort = "fullwindow_cds_random_validation"
    train_paths = [
        artifact_root / cohort / f"data/train/shard_{index:04d}.jsonl"
        for index in range(2)
    ]
    validation_paths = [artifact_root / cohort / "data/validation/shard_0000.jsonl"]
    prepare_shards(
        train,
        [str(path) for path in train_paths],
        add_rc=True,
        shuffle_seed=42,
    )
    prepare_shards(
        validation,
        [str(path) for path in validation_paths],
        add_rc=False,
        shuffle_seed=42,
    )
    for path in [*train_paths, *validation_paths]:
        _compress(path)

    repo = "marin-dna/vertebrate-v1-issue473-fullwindow-cds-random-val"
    card = artifact_root / cohort / "README.md"
    write_dataset_card(summary, card, hf_repo=repo, pipeline_commit="a" * 40)
    manifest = validate_publication(
        artifact_root,
        train,
        validation,
        summary,
        tmp_path / "manifest.json",
        cohort=cohort,
        hf_repo=repo,
        pipeline_commit="a" * 40,
        config_sha256="b" * 64,
        train_shards=2,
        validation_shards=1,
        target_length=255,
        workers=1,
    )
    assert manifest["cohorts"][cohort]["splits"]["train"]["rows"] == 12
    assert manifest["cohorts"][cohort]["splits"]["validation"]["rows"] == 16_384


def test_random_validation_uses_only_its_standalone_entrypoint() -> None:
    root = Path(__file__).parents[1]
    shared = (root / "workflow/Snakefile").read_text()
    entrypoint = (root / "workflow/Issue473RandomValidation.smk").read_text()
    rules = (root / "workflow/rules/issue_473_random_validation.smk").read_text()
    module = "rules/issue_473_random_validation.smk"
    assert module not in shared
    assert f'include: "{module}"' in entrypoint

    launcher = (root / "sky/issue_473_random_validation.yaml").read_text()
    assert "--snakefile workflow/Issue473RandomValidation.smk" in launcher
    assert "issue_473_random_validation_all_hf_files" in launcher
    assert "issue_473_random_validation_all_hf" in launcher
    assert "ALLOW_HF_UPLOAD" in launcher

    card_rule = rules.split("rule issue_473_random_validation_card:", 1)[1].split(
        "rule issue_473_random_validation_manifest:", 1
    )[0]
    assert "output:\n        local(" in card_rule
