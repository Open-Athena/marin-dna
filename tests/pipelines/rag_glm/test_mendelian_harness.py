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
    MAPPING_VERSION,
    attach_extracted_ortholog_sequences,
    derive_projected_variant_intervals,
    materialize_mendelian_rag_harness,
    select_containing_projection_anchors,
    validate_source_harness,
    write_extraction_bed,
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


def test_select_containing_projection_anchors_nearest_center_and_missing() -> None:
    variants = pl.DataFrame(
        {
            "chrom": ["1", "1", "1"],
            "pos": [192, 257, 1_001],
            "ref": ["A", "C", "G"],
            "alt": ["G", "T", "A"],
        }
    )
    human = pl.DataFrame(
        {
            "query_name": ["anchor_0", "anchor_128"],
            "t_chrom": ["chr1", "chr1"],
            "t_start": [0, 128],
            "t_end": [255, 383],
            "t_strand": ["+", "+"],
        }
    )

    got = select_containing_projection_anchors(variants, human).sort("pos")

    # pos=192 means pos0=191: both tiles are 64 bases from the SNV, so the
    # documented stable tie-break chooses the lower source start.
    assert got["source_anchor_id"].to_list() == ["anchor_0", "anchor_128", None]
    assert got["source_anchor_offset"].to_list() == [191, 128, None]
    assert got["mapping_version"].unique().to_list() == [MAPPING_VERSION]


def test_derive_projected_variant_intervals_strand_and_bounds() -> None:
    mapping = pl.DataFrame(
        {
            "chrom": ["1", "1", "1"],
            "pos": [128, 256, 1],
            "ref": ["A", "C", "G"],
            "alt": ["G", "T", "A"],
            "variant_id": ["1:128:A>G", "1:256:C>T", "1:1:G>A"],
            "variant_pos0": [127, 255, 0],
            "source_anchor_id": ["plus", "minus", "edge"],
            "source_anchor_start": [0, 128, 0],
            "source_anchor_end": [255, 383, 255],
            "source_anchor_center_distance": [0, 0, 127],
            "source_anchor_offset": [127, 127, 0],
            "mapping_version": [MAPPING_VERSION] * 3,
        }
    )
    projections = pl.DataFrame(
        {
            "query_name": ["plus", "minus", "edge"],
            "species": ["Mus_musculus"] * 3,
            "t_chrom": ["chrA", "chrB", "chrC"],
            "t_start": [1_000, 2_000, 0],
            "t_end": [1_255, 2_255, 255],
            "t_strand": ["+", "-", "+"],
            "t_src_size": [10_000, 10_000, 10_000],
        }
    )

    got = derive_projected_variant_intervals(
        mapping, projections, species="Mus_musculus"
    ).sort("pos")

    assert got.height == 2  # edge target cannot retain a centered 255-base window
    assert got["projected_variant_pos0"].to_list() == [1_127, 2_127]
    assert got["extraction_start"].to_list() == [1_000, 2_000]
    assert got["extraction_end"].to_list() == [1_255, 2_255]


def test_write_and_attach_extracted_sequences_is_row_aligned_and_oriented(
    tmp_path: Path,
) -> None:
    intervals = pl.DataFrame(
        {
            "chrom": ["1", "1"],
            "pos": [128, 256],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "variant_id": ["1:128:A>G", "1:256:C>T"],
            "projection_chrom": ["chrA", "chrB"],
            "extraction_start": [1_000, 2_000],
            "extraction_end": [1_255, 2_255],
            "projection_strand": ["+", "-"],
        }
    )
    interval_path = tmp_path / "intervals.parquet"
    bed_path = tmp_path / "intervals.bed"
    fasta_path = tmp_path / "sequences.fa"
    output_path = tmp_path / "sequences.parquet"
    intervals.write_parquet(interval_path)

    assert write_extraction_bed(interval_path, bed_path) == 2
    assert bed_path.read_text().splitlines() == [
        "chrA\t1000\t1255\t1:128:A>G",
        "chrB\t2000\t2255\t1:256:C>T",
    ]
    fasta_path.write_text(
        ">chrA:1000-1255\n" + "A" * 255 + "\n>chrB:2000-2255\n" + "A" * 255 + "\n"
    )

    assert (
        attach_extracted_ortholog_sequences(interval_path, fasta_path, output_path) == 2
    )
    got = pl.read_parquet(output_path).sort("pos")
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

    mapping = pl.DataFrame(
        {
            "chrom": ["1", "2"],
            "pos": [128, 256],
            "ref": ["A", "C"],
            "alt": ["G", "T"],
            "variant_id": ["1:128:A>G", "2:256:C>T"],
            "variant_pos0": [127, 255],
            "source_anchor_id": ["anchor_1", None],
            "source_anchor_start": [0, None],
            "source_anchor_end": [255, None],
            "source_anchor_center_distance": [0, None],
            "source_anchor_offset": [127, None],
            "mapping_version": [MAPPING_VERSION, MAPPING_VERSION],
        }
    )
    mapping_path = tmp_path / "mapping.parquet"
    mapping.write_parquet(mapping_path)

    sequence_paths: dict[str, Path] = {}
    interval_schema = {
        "variant_id": pl.String,
        "sequence": pl.String,
        "projection_chrom": pl.String,
        "extraction_start": pl.Int64,
        "extraction_end": pl.Int64,
        "projection_strand": pl.String,
        "projected_variant_pos0": pl.Int64,
    }
    for slot, species in enumerate(PROVISIONAL_SPECIES_ORDER[:-1]):
        path = tmp_path / f"{species}.parquet"
        if slot == 0:
            pl.DataFrame(
                {
                    "variant_id": ["1:128:A>G"],
                    "sequence": ["A" * BASES_PER_SLOT],
                    "projection_chrom": ["chrA"],
                    "extraction_start": [1_000],
                    "extraction_end": [1_255],
                    "projection_strand": ["+"],
                    "projected_variant_pos0": [1_127],
                }
            ).write_parquet(path)
        else:
            pl.DataFrame(schema=interval_schema).write_parquet(path)
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
        mapping_path=mapping_path,
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
    assert manifest["n_variants_with_containing_anchor"] == 1
    assert manifest["centered_variant_token_index"] == 1_920
    assert "biology" in readme_path.read_text()


def test_validate_source_harness_rejects_missing_strand_pair() -> None:
    rows = pl.DataFrame(
        _source_variant_rows(chrom="1", pos=128, ref="A", alt="G", match_group=1)
    ).filter(pl.col("strand") == "+")
    with pytest.raises(AssertionError):
        validate_source_harness(rows)
