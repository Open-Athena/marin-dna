from pathlib import Path

from marin_dna_vertebrate_projection.publication import validate_artifacts

from .test_publication import CONFIG_SHA256, PIPELINE_COMMIT, _fixture


def test_publication_validator_accepts_functional_repo_map_and_smoke_cohorts(
    tmp_path: Path,
) -> None:
    artifact_dir, source_dir, config_path = _fixture(tmp_path)
    config_path.write_text(
        config_path.read_text()
        .replace(
            "publication_smoke_train_shards: 1", "publication_smoke_train_shards: 2"
        )
        .replace("smoke_validation_rows: 1", "smoke_validation_rows: 2")
        + "hf_repo_prefix: functional\n"
        + "hf_repo_names:\n  all: functional-short-name\n"
        + "smoke_region_cohorts: [all]\n"
    )
    card_path = artifact_dir / "all/README.md"
    card_path.write_text(
        card_path.read_text().replace(
            "marin-dna/vertebrate-v2-all", "marin-dna/functional-short-name"
        )
    )

    manifest = validate_artifacts(
        artifact_dir,
        source_dir,
        tmp_path / "manifest.json",
        config_path=config_path,
        pipeline_commit=PIPELINE_COMMIT,
        config_sha256=CONFIG_SHA256,
        tier="smoke",
        workers=1,
    )

    assert set(manifest["cohorts"]) == {"all"}
