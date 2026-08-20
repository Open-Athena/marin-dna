from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_v3_stages_each_remote_source_before_reuse() -> None:
    publication = (
        PROJECT_ROOT / "workflow/rules/issue_473_publication_v3.smk"
    ).read_text()

    assert "rule issue_473_v3_stage_dataset_source:" in publication
    assert "rule issue_473_v3_stage_metadata_source:" in publication
    assert "ISSUE_473_V3_SOURCE_DATASETS" in publication
    assert "ISSUE_473_V3_SOURCE_METADATA" in publication
    assert "ISSUE_473_V3_SOURCE_DATASETS," in publication
    assert "producer=local(" in publication


def test_v3_keeps_validator_and_upload_inputs_local() -> None:
    publication = (
        PROJECT_ROOT / "workflow/rules/issue_473_publication_v3.smk"
    ).read_text()

    assert 'local(f"{ISSUE_473_V3_HF_RESULTS}/{{dataset}}/README.md")' in publication
    assert "local(ISSUE_473_V3_HF_MANIFEST)" in publication
    assert "manifest=local(ISSUE_473_V3_HF_MANIFEST)" in publication
    receipt = 'local(f"{ISSUE_473_V3_PUBLICATION_ROOT}/upload.done/{{dataset}}")'
    assert receipt in publication
    assert f"temp({receipt})" not in publication
