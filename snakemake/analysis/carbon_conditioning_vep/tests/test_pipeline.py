from pathlib import Path
from typing import Any

import pandas as pd
from marin_dna_carbon_conditioning_vep.pipeline import (
    label_blind_smoke_sample,
    load_development_dataset,
    select_analysis_subset,
    stage_analysis_windows,
    stage_analysis_windows_file,
)


def test_development_loader_downloads_only_pinned_train_parquet() -> None:
    calls: list[dict[str, Any]] = []
    expected = pd.DataFrame({"chrom": ["1"]})

    def downloader(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "/cache/train.parquet"

    def parquet_reader(path: str) -> pd.DataFrame:
        assert path == "/cache/train.parquet"
        return expected

    result = load_development_dataset(
        {
            "repo": "marin-dna/evals_mendelian_traits",
            "revision": "frozen-revision",
            "split": "train",
            "filename": "train.parquet",
        },
        downloader=downloader,
        parquet_reader=parquet_reader,
    )

    assert result is expected
    assert calls == [
        {
            "repo_id": "marin-dna/evals_mendelian_traits",
            "filename": "train.parquet",
            "repo_type": "dataset",
            "revision": "frozen-revision",
        }
    ]


def test_development_loader_rejects_non_training_file() -> None:
    for split, filename in [("test", "test.parquet"), ("train", "test.parquet")]:
        try:
            load_development_dataset(
                {
                    "repo": "marin-dna/evals_mendelian_traits",
                    "revision": "frozen-revision",
                    "split": split,
                    "filename": filename,
                },
                downloader=lambda **_: "unused",
            )
        except AssertionError:
            pass
        else:
            raise AssertionError("held-out dataset file was not rejected")


def test_analysis_subset_selects_complete_promoter_groups() -> None:
    frame = pd.DataFrame(
        {
            "subset": ["tss_proximal"] * 20 + ["missense_variant"] * 10,
            "label": [
                True,
                *([False] * 9),
                True,
                *([False] * 9),
                True,
                *([False] * 9),
            ],
            "match_group": [1] * 10 + [2] * 10 + [3] * 10,
        }
    )
    selected = select_analysis_subset(
        frame,
        {
            "subset": "tss_proximal",
            "expected_rows": 20,
            "expected_positives": 2,
            "expected_groups": 2,
            "expected_group_size": 10,
        },
    )
    assert len(selected) == 20
    assert selected["subset"].eq("tss_proximal").all()
    assert selected["match_group"].nunique() == 2


def test_analysis_scope_selects_complete_full_development_set() -> None:
    frame = pd.DataFrame(
        {
            "subset": ["tss_proximal"] * 10 + ["missense_variant"] * 10,
            "label": [True, *([False] * 9), True, *([False] * 9)],
            "match_group": [1] * 10 + [2] * 10,
        }
    )
    selected = select_analysis_subset(
        frame,
        {
            "subset": None,
            "subset_label": "all_development",
            "expected_rows": 20,
            "expected_positives": 2,
            "expected_groups": 2,
            "expected_group_size": 10,
        },
    )
    assert len(selected) == 20
    assert set(selected["subset"]) == {"tss_proximal", "missense_variant"}
    assert selected["match_group"].nunique() == 2


def test_stage_analysis_windows_filters_and_checks_provenance(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "subset": ["tss_proximal"] * 10 + ["missense_variant"] * 10,
            "label": [True, *([False] * 9), True, *([False] * 9)],
            "match_group": [1] * 10 + [2] * 10,
            "dataset_repo": ["dataset"] * 20,
            "dataset_revision": ["dataset-revision"] * 20,
            "dataset_split": ["train"] * 20,
            "reference_path": ["s3://bucket/reference.fa.gz"] * 20,
            "reference_assembly": ["GRCh38"] * 20,
            "reference_ensembl_release": [115] * 20,
            "reference_masking": ["soft-masked"] * 20,
        }
    )
    source_path = tmp_path / "all-windows.parquet"
    source.to_parquet(source_path, index=False)

    selected = stage_analysis_windows(
        source_path,
        dataset_config={
            "repo": "dataset",
            "revision": "dataset-revision",
            "split": "train",
        },
        reference_config={
            "path": "s3://bucket/reference.fa.gz",
            "assembly": "GRCh38",
            "ensembl_release": 115,
            "masking": "soft-masked",
        },
        analysis_config={
            "subset": "tss_proximal",
            "expected_rows": 10,
            "expected_positives": 1,
            "expected_groups": 1,
            "expected_group_size": 10,
        },
    )

    assert len(selected) == 10
    assert selected["analysis_subset"].eq("tss_proximal").all()
    assert selected["dataset_split"].eq("train").all()


def test_stage_analysis_windows_file_streams_full_development_set(
    tmp_path: Path,
) -> None:
    row_count = 20
    source = pd.DataFrame(
        {
            "variant_id": [f"1:{index + 1}:A>C" for index in range(row_count)],
            "subset": ["tss_proximal"] * 10 + ["missense_variant"] * 10,
            "label": [True, *([False] * 9), True, *([False] * 9)],
            "match_group": [1] * 10 + [2] * 10,
            "ref_sequence": ["A" * 12] * row_count,
            "alt_sequence": ["A" * 6 + "C" + "A" * 5] * row_count,
            "analysis_subset": ["legacy_scope"] * row_count,
            "dataset_repo": ["dataset"] * row_count,
            "dataset_revision": ["dataset-revision"] * row_count,
            "dataset_split": ["train"] * row_count,
            "reference_path": ["s3://bucket/reference.fa.gz"] * row_count,
            "reference_assembly": ["GRCh38"] * row_count,
            "reference_ensembl_release": [115] * row_count,
            "reference_masking": ["soft-masked"] * row_count,
        }
    )
    source_path = tmp_path / "all-windows.parquet"
    output_path = tmp_path / "staged-windows.parquet"
    source.to_parquet(source_path, index=False)

    stage_analysis_windows_file(
        source_path,
        output_path,
        dataset_config={
            "repo": "dataset",
            "revision": "dataset-revision",
            "split": "train",
        },
        reference_config={
            "path": "s3://bucket/reference.fa.gz",
            "assembly": "GRCh38",
            "ensembl_release": 115,
            "masking": "soft-masked",
        },
        analysis_config={
            "subset": None,
            "subset_label": "all_development",
            "expected_rows": row_count,
            "expected_positives": 2,
            "expected_groups": 2,
            "expected_group_size": 10,
        },
    )

    staged = pd.read_parquet(output_path)
    assert len(staged) == row_count
    assert staged["variant_id"].is_unique
    assert staged["analysis_subset"].eq("all_development").all()
    assert set(staged["subset"]) == {"tss_proximal", "missense_variant"}


def test_smoke_sample_removes_labels_and_consequence_metadata() -> None:
    frame = pd.DataFrame(
        {
            "variant_id": [f"1:{index}:A>C" for index in range(10)],
            "label": [index == 0 for index in range(10)],
            "subset": ["missense_variant"] * 10,
            "match_group": [1] * 10,
            "consequence": ["missense_variant"] * 10,
            "ref_sequence": ["A" * 12] * 10,
            "alt_sequence": ["A" * 6 + "C" + "A" * 5] * 10,
        }
    )
    sample = label_blind_smoke_sample(frame, n_rows=3, seed=486)
    assert len(sample) == 3
    assert {"label", "subset", "match_group", "consequence"}.isdisjoint(sample.columns)
    assert sample["label_blind_smoke"].all()
