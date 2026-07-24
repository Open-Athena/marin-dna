"""Tests for the Complex-traits/SGE direct-HAL RAG harness builder."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import polars as pl

from marin_dna.data.dna import reverse_complement
from marin_dna.pipelines.rag_glm.benchmark_harness import (
    build_benchmark_variant_windows,
    materialize_benchmark_split,
)
from marin_dna.pipelines.rag_glm.dataset import (
    MISSING_SEQUENCE,
    PROVISIONAL_SPECIES_ORDER,
)
from marin_dna.pipelines.rag_glm.mendelian_harness import PROJECTION_VERSION


def _source_row(*, pos: int, ref: str, alt: str) -> dict[str, object]:
    return {
        "chrom": "1",
        "pos": pos,
        "ref": ref,
        "alt": alt,
        "label": True,
        "subset": "distal",
        "match_group": pos,
        "extra_source_metadata": f"row-{pos}",
    }


def test_build_windows_converts_one_based_coordinates_and_checks_ref(
    tmp_path, monkeypatch
) -> None:
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    pl.DataFrame([_source_row(pos=128, ref="C", alt="T")]).write_parquet(train_path)
    pl.DataFrame([_source_row(pos=400, ref="G", alt="A")]).write_parquet(test_path)
    sequences = {
        ("chr1", 0, 255): "A" * 127 + "C" + "T" * 127,
        ("chr1", 272, 527): "A" * 127 + "G" + "T" * 127,
    }

    class _FakeGenome:
        def chroms(self):
            return {"chr1": 1_000}

        def sequence(self, chrom, start, end):
            return sequences[(chrom, start, end)]

        def close(self):
            return None

    monkeypatch.setitem(
        sys.modules, "py2bit", SimpleNamespace(open=lambda _path: _FakeGenome())
    )
    variants_path = tmp_path / "variants.parquet"
    bed_path = tmp_path / "variants.bed"
    count = build_benchmark_variant_windows(
        benchmark="complex_traits",
        source_urls={"train": str(train_path), "test": str(test_path)},
        human_twobit_path=tmp_path / "human.2bit",
        variants_path=variants_path,
        bed_path=bed_path,
    )

    assert count == 2
    variants = pl.read_parquet(variants_path).sort("pos")
    assert variants.select(
        "pos", "variant_pos0", "human_start", "human_end"
    ).rows() == [
        (128, 127, 0, 255),
        (400, 399, 272, 527),
    ]
    assert variants["projection_version"].to_list() == [
        PROJECTION_VERSION,
        PROJECTION_VERSION,
    ]
    assert bed_path.read_text().splitlines() == [
        "chr1\t0\t255\tcomplex_traits_00000000\t0\t+",
        "chr1\t272\t527\tcomplex_traits_00000001\t0\t+",
    ]


def test_materialize_split_orients_whole_document_and_retains_source_metadata(
    tmp_path,
) -> None:
    source_path = tmp_path / "source.parquet"
    source = pl.DataFrame([_source_row(pos=128, ref="C", alt="T")])
    source.write_parquet(source_path)
    human = "A" * 127 + "C" + "G" * 127
    variants = pl.DataFrame(
        {
            "chrom": ["1"],
            "pos": [128],
            "ref": ["C"],
            "alt": ["T"],
            "variant_id": ["1:128:C>T"],
            "query_name": ["complex_traits_00000000"],
            "variant_pos0": [127],
            "human_start": [0],
            "human_end": [255],
            "human_reference_sequence": [human],
            "projection_version": [PROJECTION_VERSION],
        }
    )
    species_sequences = {}
    for slot, species in enumerate(PROVISIONAL_SPECIES_ORDER[:-1]):
        if slot == 3:
            species_sequences[species] = pl.DataFrame(
                schema={
                    "query_name": pl.String,
                    "t_chrom": pl.String,
                    "t_start": pl.Int64,
                    "t_end": pl.Int64,
                    "t_strand": pl.String,
                    "sequence": pl.String,
                }
            )
        else:
            sequence = "ACGT" * 63 + "ACG"
            species_sequences[species] = pl.DataFrame(
                {
                    "query_name": ["complex_traits_00000000"],
                    "t_chrom": [f"chr{slot + 1}"],
                    "t_start": [slot * 100],
                    "t_end": [slot * 100 + 255],
                    "t_strand": ["+"],
                    "sequence": [sequence],
                }
            )

    rows = materialize_benchmark_split(
        benchmark="complex_traits",
        source_url=str(source_path),
        variants=variants,
        species_sequences=species_sequences,
        species_order=PROVISIONAL_SPECIES_ORDER,
    )
    assert rows.height == 2
    assert set(rows["strand"]) == {"+", "-"}
    assert rows["document_id"].n_unique() == 2
    assert set(rows["extra_source_metadata"]) == {"row-128"}
    assert rows["target"].to_list() == [True, True]

    forward = rows.filter(pl.col("strand") == "+").row(0, named=True)
    reverse = rows.filter(pl.col("strand") == "-").row(0, named=True)
    assert forward["sequence_7"] == human
    assert reverse["sequence_7"] == reverse_complement(human)
    assert forward["ref_completion"][0] == "C"
    assert forward["alt_completion"][0] == "T"
    assert reverse["ref_completion"][0] == "G"
    assert reverse["alt_completion"][0] == "A"
    assert forward["ref_completion"][1:] == forward["alt_completion"][1:]
    assert reverse["ref_completion"][1:] == reverse["alt_completion"][1:]
    assert forward["sequence_3"] == MISSING_SEQUENCE
    assert reverse["sequence_3"] == MISSING_SEQUENCE
    assert not forward["available_3"]
    assert not reverse["quality_pass_3"]
    assert len(forward["context"].replace("[SEQ]", "")) + 7 == 1_919
    assert len(reverse["context"].replace("[SEQ]", "")) + 7 == 1_919
