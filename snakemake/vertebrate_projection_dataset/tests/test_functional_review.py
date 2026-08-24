from pathlib import Path

import polars as pl
from marin_dna_vertebrate_projection.functional_anchors import FUNCTIONAL_ARMS
from marin_dna_vertebrate_projection.functional_projection_review import (
    write_functional_projection_review,
)
from marin_dna_vertebrate_projection.functional_review import (
    write_preprojection_review,
)


def test_preprojection_review_covers_all_arms_and_names_ensembl(
    tmp_path: Path,
) -> None:
    training = pl.DataFrame(
        {
            "query_name": [f"training-{arm}" for arm in FUNCTIONAL_ARMS],
            "source_arm": list(FUNCTIONAL_ARMS),
        }
    )
    deferred = pl.DataFrame(
        {
            "query_name": [f"deferred-{arm}" for arm in FUNCTIONAL_ARMS],
            "source_arm": list(FUNCTIONAL_ARMS),
        }
    )
    projection = pl.concat([training, deferred], how="vertical")
    projection_path = tmp_path / "projection.parquet"
    training_path = tmp_path / "training.parquet"
    deferred_path = tmp_path / "deferred.parquet"
    sample_path = tmp_path / "sample.tsv"
    report_path = tmp_path / "report.md"
    projection.write_parquet(projection_path)
    training.write_parquet(training_path)
    deferred.write_parquet(deferred_path)

    write_preprojection_review(
        projection_path,
        training_path,
        deferred_path,
        sample_path,
        report_path,
        rows_per_arm_and_band=1,
    )

    sample = pl.read_csv(sample_path, separator="\t")
    assert sample.height == 2 * len(FUNCTIONAL_ARMS)
    assert set(sample["source_arm"]) == set(FUNCTIONAL_ARMS)
    report = report_path.read_text()
    assert "Ensembl GRCh38 release 115" in report
    assert "RefSeq" in report
    assert "pending human review" in report


def test_projection_review_samples_each_functional_arm(tmp_path: Path) -> None:
    sequences_path = tmp_path / "sequences.parquet"
    pl.DataFrame(
        {
            "query_name": [f"anchor-{arm}" for arm in FUNCTIONAL_ARMS],
            "species": ["Homo sapiens"] * len(FUNCTIONAL_ARMS),
            "alignment_source": ["human_reference"] * len(FUNCTIONAL_ARMS),
            "t_chrom": ["chr1"] * len(FUNCTIONAL_ARMS),
            "t_start": list(range(len(FUNCTIONAL_ARMS))),
            "region_label": list(FUNCTIONAL_ARMS),
            "fragment_count": [1] * len(FUNCTIONAL_ARMS),
            "sequence": ["A" * 255] * len(FUNCTIONAL_ARMS),
        }
    ).write_parquet(sequences_path)
    rejected_path = tmp_path / "rejected.parquet"
    pl.DataFrame(
        {
            "query_name": ["rejected-cds"],
            "species": ["Mus musculus"],
            "rejection_reason": ["multi_mapping"],
            "fragment_count": [2],
        }
    ).write_parquet(rejected_path)
    sample_path = tmp_path / "sample.tsv"
    rejected_sample_path = tmp_path / "rejected.tsv"
    report_path = tmp_path / "report.md"

    write_functional_projection_review(
        sequences_path,
        [str(rejected_path)],
        sample_path,
        rejected_sample_path,
        report_path,
        seed=517,
        rows_per_arm=1,
        fragmented_rows=0,
        rejected_rows_per_reason=1,
    )

    assert set(pl.read_csv(sample_path, separator="\t")["region_label"]) == set(
        FUNCTIONAL_ARMS
    )
    assert pl.read_csv(rejected_sample_path, separator="\t").height == 1
    assert "pending human review" in report_path.read_text()
