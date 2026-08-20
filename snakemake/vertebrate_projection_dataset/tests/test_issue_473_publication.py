from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import zstandard as zstd
from marin_dna_vertebrate_projection.issue_473 import publication as publication_module
from marin_dna_vertebrate_projection.issue_473.publication import (
    PublicationDataset,
    parse_publication_datasets,
    validate_artifacts,
    write_dataset_card,
)

PRODUCER_COMMIT = "a" * 40
PUBLICATION_COMMIT = "b" * 40
PRODUCER_CONFIG = "c" * 64
PUBLICATION_CONFIG = "d" * 64


def _frame(source_chrom: str, augmentations: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_name": [f"anchor_{index}" for index in range(len(augmentations))],
            "source_chrom": [source_chrom] * len(augmentations),
            "sequence": ["ACgt" + "A" * 251] * len(augmentations),
            "augmentation": augmentations,
        }
    )


def _write_manifest(path: Path) -> None:
    frame = pl.DataFrame(
        {
            "alignment_name": ["Mus_musculus", "galGal4"],
            "scientific_name": ["Mus musculus", "Gallus gallus"],
            "assembly": ["GRCm39", "galGal4"],
            "taxonomy_id": [10090, 9031],
            "family": ["Muridae", "Phasianidae"],
            "clade": ["mammals", "birds"],
            "phylogenetic_rank": [1, 2],
            "backend": ["zoonomia_cactus", "ucsc_multiz100way"],
            "selection_priority": [10, 10],
            "assembly_level": ["Chromosome", "unknown"],
            "contig_n50": [1, 0],
            "selected": [True, True],
            "selection_reason": [
                "selected_best_pinned_assembly_in_family",
                "selected_best_pinned_assembly_in_family",
            ],
        }
    )
    frame.write_csv(path, separator="\t")


def _write_zstd_jsonl(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zstd.ZstdCompressor().compress(frame.write_ndjson().encode()))


def test_parse_publication_datasets_rejects_duplicate_repositories() -> None:
    raw = {
        "first": {
            "hf_repo": "marin-dna/one",
            "projection_policy": "center_1",
            "region_label": "cds",
        },
        "second": {
            "hf_repo": "marin-dna/one",
            "projection_policy": "full_window",
            "region_label": "ccre_enhancer_centered",
        },
    }
    with pytest.raises(AssertionError):
        parse_publication_datasets(raw)


def test_issue_473_card_and_manifest_reconcile_exact_artifacts(tmp_path: Path) -> None:
    dataset = PublicationDataset(
        key="center1_cds",
        hf_repo="marin-dna/vertebrate-v1-issue473-center1-cds",
        projection_policy="center_1",
        region_label="cds",
    )
    source_root = tmp_path / "datasets"
    dataset_source = source_root / "center_1/cds"
    dataset_source.mkdir(parents=True)
    train = _frame("chr1", ["+", "-", "+", "-"])
    validation = _frame("chr18", ["+", "+"])
    train.write_parquet(dataset_source / "train.parquet")
    validation.write_parquet(dataset_source / "validation.parquet")
    summary = {
        "region_label": "cds",
        "species_scope": "all",
        "seed": 42,
        "validation_chrom": "chr18",
        "train_rows": 4,
        "eligible_validation_rows": 2,
        "validation_rows": 2,
        "realized_token_count": 512,
    }
    (dataset_source / "split_summary.json").write_text(json.dumps(summary) + "\n")
    species = tmp_path / "species.tsv"
    _write_manifest(species)

    artifact_root = tmp_path / "hf"
    card = artifact_root / dataset.key / "README.md"
    write_dataset_card(
        dataset_source / "train.parquet",
        dataset_source / "validation.parquet",
        dataset_source / "split_summary.json",
        species,
        card,
        dataset=dataset,
        producer_commit=PRODUCER_COMMIT,
        producer_config_sha256=PRODUCER_CONFIG,
        publication_commit=PUBLICATION_COMMIT,
        validation_chrom="chr18",
        target_length=255,
    )
    assert dataset.hf_repo in card.read_text()
    assert "center_1" in card.read_text()

    _write_zstd_jsonl(
        train.slice(0, 2),
        artifact_root / dataset.key / "data/train/shard_0000.jsonl.zst",
    )
    _write_zstd_jsonl(
        train.slice(2, 2),
        artifact_root / dataset.key / "data/train/shard_0001.jsonl.zst",
    )
    _write_zstd_jsonl(
        validation, artifact_root / dataset.key / "data/validation/shard_0000.jsonl.zst"
    )
    output = tmp_path / "manifest.json"
    manifest = validate_artifacts(
        artifact_root,
        source_root,
        output,
        datasets=[dataset],
        producer_commit=PRODUCER_COMMIT,
        producer_config_sha256=PRODUCER_CONFIG,
        publication_commit=PUBLICATION_COMMIT,
        publication_config_sha256=PUBLICATION_CONFIG,
        train_shards=2,
        validation_shards=1,
        validation_chrom="chr18",
        target_length=255,
        workers=1,
    )
    assert json.loads(output.read_text()) == manifest
    assert manifest["cohorts"][dataset.key]["splits"]["train"]["rows"] == 4

    (artifact_root / dataset.key / "unexpected.txt").write_text("stale\n")
    with pytest.raises(AssertionError, match="unexpected"):
        validate_artifacts(
            artifact_root,
            source_root,
            output,
            datasets=[dataset],
            producer_commit=PRODUCER_COMMIT,
            producer_config_sha256=PRODUCER_CONFIG,
            publication_commit=PUBLICATION_COMMIT,
            publication_config_sha256=PUBLICATION_CONFIG,
            train_shards=2,
            validation_shards=1,
            validation_chrom="chr18",
            target_length=255,
            workers=1,
        )


def test_issue_473_upload_creates_and_rechecks_private_repo(
    monkeypatch, tmp_path: Path
) -> None:
    class PrivateInfo:
        private = True

    class FakeApi:
        def __init__(self) -> None:
            self.created: list[tuple[str, str, bool, bool]] = []
            self.info_calls = 0

        def repo_exists(self, repo_id: str, *, repo_type: str) -> bool:
            assert repo_id == "marin-dna/test-private"
            assert repo_type == "dataset"
            return False

        def create_repo(
            self,
            repo_id: str,
            *,
            repo_type: str,
            private: bool,
            exist_ok: bool,
        ) -> None:
            self.created.append((repo_id, repo_type, private, exist_ok))

        def dataset_info(self, repo_id: str) -> PrivateInfo:
            assert repo_id == "marin-dna/test-private"
            self.info_calls += 1
            return PrivateInfo()

    api = FakeApi()
    uploads: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(publication_module, "HfApi", lambda: api)
    monkeypatch.setattr(
        publication_module,
        "upload_validated_dataset",
        lambda *args, **kwargs: uploads.append((args, kwargs)),
    )

    publication_module.upload_private_validated_dataset(
        tmp_path / "artifacts",
        tmp_path / "manifest.json",
        tmp_path / "upload.done",
        cohort="center1_cds",
        repo_id="marin-dna/test-private",
        workers=4,
    )

    assert api.created == [("marin-dna/test-private", "dataset", True, False)]
    assert api.info_calls == 2
    assert len(uploads) == 1
    assert uploads[0][1] == {
        "cohort": "center1_cds",
        "repo_id": "marin-dna/test-private",
        "workers": 4,
    }


def test_issue_473_upload_rejects_preexisting_public_repo(monkeypatch) -> None:
    class PublicInfo:
        private = False

    class FakeApi:
        def repo_exists(self, repo_id: str, *, repo_type: str) -> bool:
            return True

        def dataset_info(self, repo_id: str) -> PublicInfo:
            return PublicInfo()

    monkeypatch.setattr(publication_module, "HfApi", FakeApi)
    monkeypatch.setattr(
        publication_module,
        "upload_validated_dataset",
        lambda *args, **kwargs: pytest.fail("public repository must not upload"),
    )

    with pytest.raises(AssertionError, match="refusing to upload.*publicly"):
        publication_module.upload_private_validated_dataset(
            "artifacts",
            "manifest.json",
            "upload.done",
            cohort="center1_cds",
            repo_id="marin-dna/public",
            workers=4,
        )
