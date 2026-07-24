import json
from pathlib import Path

import polars as pl
import pytest

from marin_dna.pipelines.rag_glm.dataset import (
    BASES_PER_SLOT,
    MISSING_SEQUENCE,
    PROVISIONAL_SPECIES_ORDER,
)
from marin_dna.pipelines.rag_glm.mendelian_harness import (
    PROJECTION_VERSION,
    build_mendelian_variant_windows,
    extract_ortholog_sequences_from_twobit,
    materialize_mendelian_rag_harness,
    project_mendelian_variant_windows,
    validate_source_harness,
)


def _source_variant_rows(
    *, chrom: str, pos: int, ref: str, alt: str, match_group: int
) -> list[dict[str, object]]:
    complements = {"A": "T", "C": "G", "G": "C", "T": "A"}
    return [
        {
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "target": True,
            "subset": "missense_variant",
            "match_group": match_group,
            "context": "A" * 127,
            "ref_completion": ref + "C" * 127,
            "alt_completion": alt + "C" * 127,
            "strand": "+",
        },
        {
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "target": True,
            "subset": "missense_variant",
            "match_group": match_group,
            "context": "G" * 127,
            "ref_completion": complements[ref] + "T" * 127,
            "alt_completion": complements[alt] + "T" * 127,
            "strand": "-",
        },
    ]


def test_build_mendelian_variant_windows_uses_exact_centered_coordinates(
    tmp_path: Path,
) -> None:
    train = pl.DataFrame(
        _source_variant_rows(chrom="chr2", pos=256, ref="C", alt="T", match_group=2)
    )
    test = pl.DataFrame(
        _source_variant_rows(chrom="1", pos=128, ref="A", alt="G", match_group=1)
    )
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    variants_path = tmp_path / "variants.parquet"
    bed_path = tmp_path / "variants.bed"
    train.write_parquet(train_path)
    test.write_parquet(test_path)

    assert (
        build_mendelian_variant_windows(
            source_harness_urls=[str(train_path), str(test_path)],
            variants_path=variants_path,
            bed_path=bed_path,
        )
        == 2
    )
    variants = pl.read_parquet(variants_path).sort("chrom")
    assert variants["chrom"].to_list() == ["1", "2"]
    assert variants["variant_pos0"].to_list() == [127, 255]
    assert variants["human_start"].to_list() == [0, 128]
    assert variants["human_end"].to_list() == [255, 383]
    assert variants["projection_version"].unique().to_list() == [PROJECTION_VERSION]
    assert variants["query_name"].n_unique() == 2

    bed = pl.read_csv(
        bed_path,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end", "name", "score", "strand"],
    ).sort("chrom")
    assert bed.select("chrom", "start", "end").rows() == [
        ("chr1", 0, 255),
        ("chr2", 128, 383),
    ]
    assert set(bed["strand"]) == {"+"}


def test_project_mendelian_variant_windows_reuses_canonical_quality_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_bed = tmp_path / "source.bed"
    source_bed.write_text("chr1\t0\t255\tmendelian_00000000\t0\t+\n")
    chrom_sizes = tmp_path / "chrom.sizes"
    chrom_sizes.write_text("chrA\t10000\nchrB\t10000\n")
    output = tmp_path / "projection.parquet"
    raw = tmp_path / "raw.bed"

    def fake_halliftover(
        hal_path: str | Path,
        src_species: str,
        src_bed: str | Path,
        tgt_species: str,
        out_bed: str | Path,
        *,
        no_dupes: bool,
    ) -> float:
        assert str(hal_path).endswith("test.hal")
        assert src_species == "Homo_sapiens"
        assert Path(src_bed) == source_bed
        assert tgt_species == "Mus_musculus"
        assert no_dupes
        Path(out_bed).write_text(
            "chrA\t1000\t1100\tmendelian_00000000\t0\t+\n"
            "chrA\t1100\t1200\tmendelian_00000000\t0\t+\n"
            "chrA\t2000\t2100\tmulti_chrom\t0\t+\n"
            "chrB\t2100\t2200\tmulti_chrom\t0\t+\n"
            "chrA\t3000\t3050\ttoo_short\t0\t+\n"
        )
        return 0.01

    monkeypatch.setattr(
        "marin_dna.pipelines.rag_glm.mendelian_harness.run_halliftover",
        fake_halliftover,
    )
    assert (
        project_mendelian_variant_windows(
            hal_path=tmp_path / "test.hal",
            source_bed=source_bed,
            target_species="Mus_musculus",
            target_chrom_sizes=chrom_sizes,
            output_parquet=output,
            raw_bed_path=raw,
        )
        == 1
    )
    got = pl.read_parquet(output)
    assert got["query_name"].to_list() == ["mendelian_00000000"]
    assert got["species"].to_list() == ["Mus_musculus"]
    assert got["t_start"].to_list() == [973]
    assert got["t_end"].to_list() == [1228]
    assert not raw.exists()


def test_extract_twobit_sequences_is_row_aligned_and_oriented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projections = pl.DataFrame(
        {
            "query_name": ["q0", "q1"],
            "species": ["Mus_musculus"] * 2,
            "t_chrom": ["chrA", "chrB"],
            "t_start": [1_000, 2_000],
            "t_end": [1_255, 2_255],
            "t_strand": ["+", "-"],
            "t_src_size": [10_000, 10_000],
        }
    )
    projection_path = tmp_path / "projection.parquet"
    output_path = tmp_path / "sequences.parquet"
    projections.write_parquet(projection_path)

    class FakeTwoBit:
        def chroms(self) -> dict[str, int]:
            return {"chrA": 10_000, "chrB": 10_000}

        def sequence(self, chrom: str, start: int, end: int) -> str:
            assert (chrom, start, end) in {
                ("chrA", 1_000, 1_255),
                ("chrB", 2_000, 2_255),
            }
            return "A" * (end - start)

        def close(self) -> None:
            return None

    monkeypatch.setattr("py2bit.open", lambda _: FakeTwoBit())
    assert (
        extract_ortholog_sequences_from_twobit(
            projection_path, tmp_path / "species.2bit", output_path
        )
        == 2
    )
    got = pl.read_parquet(output_path).sort("query_name")
    assert got["sequence"].to_list() == ["A" * 255, "T" * 255]


def test_materialize_mendelian_rag_harness_geometry_and_missing_slots(
    tmp_path: Path,
) -> None:
    train_source = pl.DataFrame(
        _source_variant_rows(chrom="1", pos=128, ref="A", alt="G", match_group=1)
    )
    test_source = pl.DataFrame(
        _source_variant_rows(chrom="2", pos=256, ref="C", alt="T", match_group=2)
    )
    train_source_path = tmp_path / "source_train.parquet"
    test_source_path = tmp_path / "source_test.parquet"
    train_source.write_parquet(train_source_path)
    test_source.write_parquet(test_source_path)

    variants_path = tmp_path / "variants.parquet"
    build_mendelian_variant_windows(
        source_harness_urls=[str(train_source_path), str(test_source_path)],
        variants_path=variants_path,
        bed_path=tmp_path / "variants.bed",
    )
    variants = pl.read_parquet(variants_path).sort("chrom")
    query_1 = variants["query_name"][0]

    sequence_paths: dict[str, Path] = {}
    projection_schema = {
        "query_name": pl.String,
        "species": pl.String,
        "t_chrom": pl.String,
        "t_start": pl.Int64,
        "t_end": pl.Int64,
        "t_strand": pl.String,
        "t_src_size": pl.Int64,
        "sequence": pl.String,
    }
    for slot, species in enumerate(PROVISIONAL_SPECIES_ORDER[:-1]):
        path = tmp_path / f"{species}.parquet"
        if slot == 0:
            pl.DataFrame(
                {
                    "query_name": [query_1],
                    "species": [species],
                    "t_chrom": ["chrA"],
                    "t_start": [1_000],
                    "t_end": [1_255],
                    "t_strand": ["+"],
                    "t_src_size": [10_000],
                    "sequence": ["A" * BASES_PER_SLOT],
                }
            ).write_parquet(path)
        else:
            pl.DataFrame(schema=projection_schema).write_parquet(path)
        sequence_paths[species] = path

    output_paths = {
        "train": tmp_path / "train.parquet",
        "test": tmp_path / "test.parquet",
    }
    manifest_path = tmp_path / "manifest.json"
    readme_path = tmp_path / "README.md"
    materialize_mendelian_rag_harness(
        source_harness_urls={
            "train": str(train_source_path),
            "test": str(test_source_path),
        },
        variants_path=variants_path,
        species_sequence_paths=sequence_paths,
        output_split_paths=output_paths,
        manifest_path=manifest_path,
        readme_path=readme_path,
        commit_sha="a" * 40,
        hf_repo="bolinas-dna/test-rag-harness",
    )

    train = pl.read_parquet(output_paths["train"]).sort("strand")
    test = pl.read_parquet(output_paths["test"])
    assert train.height == test.height == 2
    assert train["sequence_0"].to_list() == ["A" * 255, "T" * 255]
    assert set(train["sequence_1"]) == {MISSING_SEQUENCE}
    assert set(test["sequence_0"]) == {MISSING_SEQUENCE}
    assert all(value.count("[SEQ]") == 7 for value in train["context"])
    assert all(
        len(value.replace("[SEQ]", "")) + 7 == 1_919 for value in train["context"]
    )
    assert train["document_id"].n_unique() == 2

    manifest = json.loads(manifest_path.read_text())
    assert manifest["split_rows"] == {"train": 2, "test": 2}
    assert manifest["projection_version"] == PROJECTION_VERSION
    assert manifest["projection"]["hal_liftover_no_dupes"] is True
    assert manifest["centered_variant_token_index"] == 1_920
    assert "biology" in readme_path.read_text()


def test_validate_source_harness_rejects_missing_strand_pair() -> None:
    rows = pl.DataFrame(
        _source_variant_rows(chrom="1", pos=128, ref="A", alt="G", match_group=1)
    ).filter(pl.col("strand") == "+")
    with pytest.raises(AssertionError):
        validate_source_harness(rows)
