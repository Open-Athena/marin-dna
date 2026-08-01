from pathlib import Path

import polars as pl
import pytest

from scripts.issue417_validate_outputs import _dataset_stats
from scripts.issue417_validate_zrs_sidecar import (
    EXPECTED_ZRS,
    validate_zrs_sidecar,
)


def _zrs_row(
    query_name: str,
    *,
    alignment_source: str,
    species: str,
    clade: str,
) -> dict[str, object]:
    chrom, start, end = EXPECTED_ZRS[query_name]
    return {
        "query_name": query_name,
        "source_chrom": chrom,
        "source_start": start,
        "source_end": end,
        "alignment_source": alignment_source,
        "species": species,
        "clade": clade,
        "t_start": 0,
        "t_end": 255,
        "t_strand": "+",
        "sequence": "A" * 255,
    }


def test_zrs_sidecar_validator_requires_broad_recovery(tmp_path: Path) -> None:
    results = tmp_path / "smoke"
    (results / "metadata").mkdir(parents=True)
    (results / "sequences").mkdir()
    (results / "qc").mkdir()

    pl.DataFrame(
        {
            "selected": [True] * 7,
            "backend": ["zoonomia_cactus"] * 2 + ["ucsc_multiz100way"] * 5,
        }
    ).write_csv(results / "metadata/species_active.tsv", separator="\t")

    rows = []
    for query_name in EXPECTED_ZRS:
        rows.extend(
            [
                _zrs_row(
                    query_name,
                    alignment_source="human_reference",
                    species="Homo sapiens",
                    clade="mammals",
                ),
                _zrs_row(
                    query_name,
                    alignment_source="zoonomia_cactus",
                    species="Mus musculus",
                    clade="mammals",
                ),
                _zrs_row(
                    query_name,
                    alignment_source="ucsc_multiz100way",
                    species="Gallus gallus",
                    clade="birds",
                ),
                _zrs_row(
                    query_name,
                    alignment_source="ucsc_multiz100way",
                    species="Danio rerio",
                    clade="fish",
                ),
            ]
        )
    pl.DataFrame(rows).write_parquet(results / "sequences/all_sources.parquet")
    pl.DataFrame(
        {
            "query_name": list(EXPECTED_ZRS),
            "accepted_non_mammal_projections": [2, 2],
        }
    ).write_parquet(results / "qc/per_anchor.parquet")
    pl.DataFrame({"query_name": list(EXPECTED_ZRS)}).write_csv(
        results / "qc/manual_inspection_sample.tsv", separator="\t"
    )
    (results / "qc/manual_inspection.md").write_text(
        "pending human review\n\nRequired ZRS anchors: " + ", ".join(EXPECTED_ZRS)
    )

    summary = validate_zrs_sidecar(
        results,
        expected_pipeline_commit="a" * 40,
    )

    assert summary["status"] == "automated ZRS sidecar validation passed"
    assert len(summary["zrs_recovery"]) == 2


def test_full_dataset_validator_rejects_zrs_rows(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    pl.DataFrame(
        {
            "query_name": ["zrs_forbidden"],
            "sequence": ["A" * 255],
            "source_chrom": ["chr7"],
            "augmentation": ["+"],
            "species": ["Homo sapiens"],
        }
    ).write_parquet(path)

    with pytest.raises(AssertionError, match="zrs_rows=1"):
        _dataset_stats(path, validation=False)
