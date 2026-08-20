from pathlib import Path


def test_projection_report_v3_syncs_exact_additive_code() -> None:
    launcher = (
        Path(__file__).parents[1] / "sky" / "projection_report_batched_v3.yaml"
    ).read_text()
    assert "workdir: ." in launcher
    assert "fixed_catalog_report" in launcher
    assert "projection_report_v3" in launcher
    assert "PRODUCER_COMMIT=d0e5380a46cd66d4c42d763b3c42da1150c92073" in launcher
    assert 'test "${#FULL_REJECTIONS[@]}" -eq 2424' in launcher
    assert 'test "${#CENTER_REJECTIONS[@]}" -eq 1884' in launcher
