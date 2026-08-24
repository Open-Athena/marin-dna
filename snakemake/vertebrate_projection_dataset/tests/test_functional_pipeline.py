from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from marin_dna_vertebrate_projection.functional_anchors import FUNCTIONAL_ARMS
from marin_dna_vertebrate_projection.functional_pipeline import (
    read_ccre_v4,
    write_anchor_distribution_summary,
    write_chromosome_summary,
    write_conservation_catalogs,
    write_development_locus_overlap,
    write_drop_summaries,
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


def test_development_overlap_is_half_open_and_excludes_mature_mirna_groups(
    tmp_path: Path,
) -> None:
    anchors = pl.DataFrame(
        [
            {
                "query_name": f"anchor-{arm}",
                "source_arm": arm,
                "chrom": "1",
                "start": 9,
                "end": 264,
                "source_chrom": "chr1",
            }
            for arm in FUNCTIONAL_ARMS
        ]
    )
    projection = tmp_path / "projection.parquet"
    training = tmp_path / "training.parquet"
    anchors.write_parquet(projection)
    anchors.write_parquet(training)
    subsets = [
        "3_prime_UTR_variant",
        "5_prime_UTR_variant",
        "distal",
        "missense_variant",
        "non_coding_transcript_exon_variant",
        "splicing",
        "synonymous_variant",
        "tss_proximal",
    ]
    variants = pl.DataFrame(
        [
            {
                "chrom": "1",
                "pos": 265 if subset == "distal" else 10,
                "label": True,
                "subset": subset,
                "match_group": index,
            }
            for index, subset in enumerate(subsets)
        ]
        + [
            {
                "chrom": "1",
                "pos": 10,
                "label": label,
                "subset": "mature_miRNA_variant",
                "match_group": 99,
            }
            for label in [True, False]
        ]
    )
    variants_path = tmp_path / "variants.parquet"
    variants.write_parquet(variants_path)
    output = tmp_path / "development_overlap.tsv"

    write_development_locus_overlap(
        projection,
        training,
        variants_path,
        output,
        dataset_repo="marin-dna/evals_mendelian_traits",
        dataset_revision="revision",
        dataset_split="train",
    )

    result = pl.read_csv(output, separator="\t")
    assert result.height == 2 * len(FUNCTIONAL_ARMS) * len(subsets)
    assert set(result["subset"]) == set(subsets)
    assert result["excluded_mature_mirna_match_groups"].unique().to_list() == [1]
    assert (
        result.filter(pl.col("subset") == "distal")["overlapping_variant_count"].sum()
        == 0
    )
    assert (
        result.filter(pl.col("subset") != "distal")["overlapping_variant_count"] == 1
    ).all()
    assert set(result["coordinate_conversion"]) == {
        "1-based VEP pos -> 0-based [pos-1,pos)"
    }


def test_preprojection_summary_tables_cover_both_catalogs_and_all_arms(
    tmp_path: Path,
) -> None:
    audited_rows = []
    for arm_index, arm in enumerate(FUNCTIONAL_ARMS):
        for row_index, conserved in enumerate([0.15, 0.25]):
            start = arm_index * 10_000 + row_index * 300
            audited_rows.append(
                {
                    "query_name": f"{arm}-{row_index}",
                    "source_arm": arm,
                    "chrom": "1",
                    "start": start,
                    "end": start + 255,
                    "source_chrom": "chr1",
                    "source_arm_owned_fraction": 0.8,
                    "union_functional_fraction": 0.9,
                    "exon_fraction": 0.0,
                    "repeat_masked_fraction": 0.1,
                    "gc_fraction": 0.5,
                    "ambiguous_base_fraction": 0.0,
                    "proportion_conserved": conserved,
                    "contributing_feature_count": 1,
                }
            )
    audited = pl.DataFrame(audited_rows)
    audited_path = tmp_path / "audited.parquet"
    audited.write_parquet(audited_path)
    distribution = tmp_path / "distribution.tsv"
    write_anchor_distribution_summary(
        audited_path,
        distribution,
        training_min=0.20,
    )
    distribution_frame = pl.read_csv(distribution, separator="\t")
    assert distribution_frame.height == 2 * len(FUNCTIONAL_ARMS) * 8
    assert set(distribution_frame["catalog"]) == {
        "projection_ge_0.10",
        "training_ge_0.20",
    }

    projection = tmp_path / "projection.parquet"
    training = tmp_path / "training.parquet"
    audited.write_parquet(projection)
    audited.filter(pl.col("proportion_conserved") >= 0.20).write_parquet(training)
    chromosome = tmp_path / "chromosome.tsv"
    write_chromosome_summary(projection, training, chromosome)
    chromosome_frame = pl.read_csv(chromosome, separator="\t")
    assert chromosome_frame.height == 2 * len(FUNCTIONAL_ARMS)
    assert (chromosome_frame["within_arm_fraction"] == 1.0).all()

    ownership = tmp_path / "ownership.parquet"
    construction_drops = tmp_path / "construction_drops.parquet"
    pl.DataFrame(
        [
            {
                "source_arm": arm,
                "passes_ownership_gate": passes,
                "ownership_winner": arm if passes else "cds",
            }
            for arm in FUNCTIONAL_ARMS
            for passes in [True, False]
        ]
    ).write_parquet(ownership)
    pl.DataFrame(
        [
            {
                "source_arm": arm,
                "drop_reason": "undefined_sequence",
            }
            for arm in FUNCTIONAL_ARMS
        ]
    ).write_parquet(construction_drops)
    construction_summary = tmp_path / "construction.tsv"
    ownership_summary = tmp_path / "ownership.tsv"
    write_drop_summaries(
        ownership,
        construction_drops,
        construction_summary,
        ownership_summary,
    )
    assert pl.read_csv(construction_summary, separator="\t").height == (
        2 * len(FUNCTIONAL_ARMS)
    )
    assert pl.read_csv(ownership_summary, separator="\t").height == (
        2 * len(FUNCTIONAL_ARMS)
    )
