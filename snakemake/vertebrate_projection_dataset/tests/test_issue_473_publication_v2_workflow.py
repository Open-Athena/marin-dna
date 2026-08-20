from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_v2_keeps_validator_inputs_and_upload_receipts_local() -> None:
    publication = (
        PROJECT_ROOT / "workflow/rules/issue_473_publication_v2.smk"
    ).read_text()

    assert 'local(f"{ISSUE_473_V2_HF_RESULTS}/{{dataset}}/README.md")' in publication
    assert "local(ISSUE_473_V2_HF_MANIFEST)" in publication
    receipt = 'local(f"{ISSUE_473_V2_PUBLICATION_ROOT}/upload.done/{{dataset}}")'
    assert receipt in publication
    assert f"temp({receipt})" not in publication


def test_v2_archives_manifest_separately_from_local_upload_input() -> None:
    publication = (
        PROJECT_ROOT / "workflow/rules/issue_473_publication_v2.smk"
    ).read_text()

    assert "rule issue_473_v2_archive_hf_artifact_manifest:" in publication
    assert "ISSUE_473_V2_HF_MANIFEST_ARCHIVE" in publication
    assert "manifest=local(ISSUE_473_V2_HF_MANIFEST)" in publication
