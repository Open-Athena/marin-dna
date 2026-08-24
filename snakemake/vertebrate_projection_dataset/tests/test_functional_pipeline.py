from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from marin_dna_vertebrate_projection.functional_anchors import FUNCTIONAL_ARMS
from marin_dna_vertebrate_projection.functional_pipeline import (
    read_ccre_v4,
    write_conservation_catalogs,
    write_training_sequences,
)


def test_read_ccre_v4_preserves_registry_id_and_class(tmp_path: Path) -> None:
    path = tmp_path / "ccre.bed"
    path.write_text("chr1\t10\t30\tEH38E1\t0\tdELS\nchr2\t40\t60\tEH38E2\t0\tpELS\n")

    assert read_ccre_v4(path).to_dicts() == [
        {
            "chrom": "chr1",
            "start": 10,
            "end": 30,
            "ccre_id": "EH38E1",
            "cre_class": "dELS",
        },
        {
            "chrom": "chr2",
            "start": 40,
            "end": 60,
            "ccre_id": "EH38E2",
            "cre_class": "pELS",
        },
    ]


def test_write_conservation_catalogs_keeps_nested_smoke_bands(
    tmp_path: Path,
) -> None:
    rows = []
    for arm_index, arm in enumerate(FUNCTIONAL_ARMS):
        for band_index, score in enumerate([0.25, 0.15, 0.05]):
            start = arm_index * 10_000 + band_index * 300
            rows.append(
                {
                    "query_name": f"{arm}-{band_index}",
                    "source_arm": arm,
                    "chrom": "1",
                    "start": start,
                    "end": start + 255,
                    "proportion_conserved": score,
                }
            )
    scored = tmp_path / "scored.parquet"
    pl.DataFrame(rows).write_parquet(scored)
    projection = tmp_path / "projection.parquet"
    training = tmp_path / "training.parquet"
    deferred = tmp_path / "deferred.parquet"
    summary = tmp_path / "summary.json"

    write_conservation_catalogs(
        scored,
        projection,
        training,
        deferred,
        summary,
        projection_min=0.10,
        training_min=0.20,
        smoke_training_per_arm=1,
        smoke_deferred_per_arm=1,
    )

    projection_frame = pl.read_parquet(projection)
    training_frame = pl.read_parquet(training)
    deferred_frame = pl.read_parquet(deferred)
    assert projection_frame.height == 2 * len(FUNCTIONAL_ARMS)
    assert training_frame.height == len(FUNCTIONAL_ARMS)
    assert deferred_frame.height == len(FUNCTIONAL_ARMS)
    assert set(training_frame["query_name"]) <= set(projection_frame["query_name"])
    assert set(deferred_frame["query_name"]) <= set(projection_frame["query_name"])
    assert set(training_frame["query_name"]).isdisjoint(
        set(deferred_frame["query_name"])
    )
    assert set(projection_frame["source_chrom"]) == {"chr1"}
    assert json.loads(summary.read_text())["counts"]["cds"] == {
        "deferred": 1,
        "projection": 2,
        "training": 1,
    }


def test_write_training_sequences_filters_deferred_anchors(tmp_path: Path) -> None:
    combined = tmp_path / "combined.parquet"
    training = tmp_path / "training.parquet"
    output = tmp_path / "training-sequences.parquet"
    pl.DataFrame(
        {
            "query_name": ["train", "deferred", "train"],
            "species": ["human", "human", "mouse"],
            "sequence": ["A" * 255, "C" * 255, "G" * 255],
        }
    ).write_parquet(combined)
    pl.DataFrame({"query_name": ["train"]}).write_parquet(training)

    write_training_sequences(combined, training, output)

    assert pl.read_parquet(output)["query_name"].to_list() == ["train", "train"]
