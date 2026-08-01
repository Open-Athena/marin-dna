from pathlib import Path

import polars as pl

from scripts import issue417_trace_inspection_samples as trace


def test_trace_samples_reads_the_pipeline_inspection_sample(
    tmp_path: Path, monkeypatch
) -> None:
    results = tmp_path / "results"
    qc = results / "qc"
    metadata = results / "metadata"
    multiz_stage = tmp_path / "multiz"
    qc.mkdir(parents=True)
    metadata.mkdir()
    multiz_stage.mkdir()

    pl.DataFrame(
        [
            {
                "alignment_source": "zoonomia_cactus",
                "fragment_count": 2,
                "query_name": "hal_query",
                "alignment_name": "hal_species",
            },
            {
                "alignment_source": "ucsc_multiz100way",
                "fragment_count": 3,
                "query_name": "multiz_query",
                "alignment_name": "multiz_species",
            },
        ]
    ).write_csv(qc / "manual_inspection_sample.tsv", separator="\t")
    pl.DataFrame({"selected": [True]}).write_csv(
        metadata / "species_active.tsv", separator="\t"
    )

    monkeypatch.setattr(
        trace,
        "_trace_hal",
        lambda _results, row: {"query_name": row["query_name"]},
    )
    monkeypatch.setattr(
        trace,
        "_trace_multiz",
        lambda _results, _stage, _manifest, row: {"query_name": row["query_name"]},
    )

    summary = trace.trace_samples(
        results,
        multiz_stage,
        rows_per_backend=1,
        expected_pipeline_commit="a" * 40,
    )

    assert summary["hal"] == [{"query_name": "hal_query"}]
    assert summary["multiz"] == [{"query_name": "multiz_query"}]
    assert summary["status"] == "backend-native fragment traces passed"
