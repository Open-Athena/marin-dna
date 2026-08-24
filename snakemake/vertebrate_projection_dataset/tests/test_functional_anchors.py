from pathlib import Path

import polars as pl
import pytest
from marin_dna_vertebrate_projection.functional_anchors import (
    FUNCTIONAL_ARMS,
    CandidateWindows,
    FunctionalFeatureSets,
    annotate_sequence_fractions,
    apply_window_ownership_gate,
    build_candidate_windows,
    extract_functional_features,
    feature_audit_table,
    pairwise_raw_overlap_table,
    resolve_base_priority,
    split_conservation_catalogs,
    tile_intervals,
    to_projection_catalog,
)

from marin_dna.data.intervals import GenomicSet
from marin_dna.data.utils import load_annotation


def _gtf_row(
    chrom: str,
    feature: str,
    start: int,
    end: int,
    strand: str,
    attributes: str,
) -> str:
    return f"{chrom}\ttest\t{feature}\t{start}\t{end}\t.\t{strand}\t.\t{attributes}\n"


def _write_annotation(path: Path) -> pl.DataFrame:
    rows = [
        _gtf_row(
            "1",
            "transcript",
            1001,
            3000,
            "+",
            'transcript_id "T1"; transcript_biotype "protein_coding"; gene_biotype "protein_coding";',
        ),
        _gtf_row(
            "1",
            "exon",
            1001,
            3000,
            "+",
            'transcript_id "T1"; exon_id "E1"; exon_number "1"; transcript_biotype "protein_coding"; gene_biotype "protein_coding";',
        ),
        _gtf_row(
            "1",
            "CDS",
            1501,
            2500,
            "+",
            'transcript_id "T1"; exon_id "E1"; exon_number "1"; transcript_biotype "protein_coding"; gene_biotype "protein_coding";',
        ),
        # A non-canonical alternative transcript remains part of the training set.
        _gtf_row(
            "1",
            "transcript",
            2401,
            3400,
            "+",
            'transcript_id "T2"; transcript_biotype "protein_coding"; gene_biotype "protein_coding";',
        ),
        _gtf_row(
            "1",
            "exon",
            2401,
            3400,
            "+",
            'transcript_id "T2"; exon_id "E2"; exon_number "1"; transcript_biotype "protein_coding"; gene_biotype "protein_coding";',
        ),
        _gtf_row(
            "1",
            "CDS",
            2601,
            2700,
            "+",
            'transcript_id "T2"; exon_id "E2"; exon_number "1"; transcript_biotype "protein_coding"; gene_biotype "protein_coding";',
        ),
        _gtf_row(
            "1",
            "transcript",
            5001,
            7000,
            "-",
            'transcript_id "T3"; transcript_biotype "protein_coding"; gene_biotype "protein_coding";',
        ),
        _gtf_row(
            "1",
            "exon",
            5001,
            7000,
            "-",
            'transcript_id "T3"; exon_id "E3"; exon_number "1"; transcript_biotype "protein_coding"; gene_biotype "protein_coding";',
        ),
        _gtf_row(
            "1",
            "CDS",
            5501,
            6500,
            "-",
            'transcript_id "T3"; exon_id "E3"; exon_number "1"; transcript_biotype "protein_coding"; gene_biotype "protein_coding";',
        ),
        _gtf_row(
            "1",
            "transcript",
            8001,
            8300,
            "+",
            'transcript_id "TN1"; transcript_biotype "lncRNA"; gene_biotype "lncRNA";',
        ),
        _gtf_row(
            "1",
            "exon",
            8001,
            8300,
            "+",
            'transcript_id "TN1"; exon_id "EN1"; transcript_biotype "lncRNA"; gene_biotype "lncRNA";',
        ),
        _gtf_row(
            "1",
            "exon",
            8401,
            8500,
            "+",
            'transcript_id "TP"; exon_id "EP"; transcript_biotype "lncRNA"; gene_biotype "processed_pseudogene"; pseudo "true";',
        ),
    ]
    path.write_text("".join(rows))
    return load_annotation(str(path))


def _ccre() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [9000, 10_000, 11_000],
            "end": [9200, 10_200, 11_200],
            "ccre_id": ["EH38E1", "EH38E2", "EH38E3"],
            "cre_class": ["dELS", "pELS", "PLS"],
        }
    )


def _feature(
    chrom: str,
    start: int,
    end: int,
    source_id: str,
    source_feature: str,
    strand: str = "+",
) -> dict[str, object]:
    return {
        "chrom": chrom,
        "start": start,
        "end": end,
        "strand": strand,
        "source_id": source_id,
        "source_feature": source_feature,
    }


def _features(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "chrom": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
            "strand": pl.String,
            "source_id": pl.String,
            "source_feature": pl.String,
        },
    )


def _empty_set() -> GenomicSet:
    return GenomicSet(
        pl.DataFrame(schema={"chrom": pl.String, "start": pl.Int64, "end": pl.Int64})
    )


def test_extracts_ensembl_features_without_canonical_filter(tmp_path: Path) -> None:
    annotation = _write_annotation(tmp_path / "annotation.gtf")
    extracted = extract_functional_features(
        annotation,
        _ccre(),
        standard_chroms=["chr1"],
    )

    assert set(extracted.features) == set(FUNCTIONAL_ARMS)
    cds = extracted.features["cds"]
    assert cds.select("start", "end").rows() == [
        (1500, 2500),
        (2600, 2700),
        (5500, 6500),
    ]
    assert any("T2" in source_id or "E2" in source_id for source_id in cds["source_id"])

    # T1's 3' UTR [2500, 3000) is split around T2's CDS [2600, 2700).
    utr3 = extracted.features["utr3"]
    assert not (extracted.raw_cores["utr3"] & extracted.raw_cores["cds"]).total_size()
    assert ("1", 2500, 2600) in utr3.select("chrom", "start", "end").rows()
    assert ("1", 2700, 3000) in utr3.select("chrom", "start", "end").rows()

    # The minus-strand transcript contributes 3' UTR on the genomic left.
    assert ("1", 5000, 5500) in utr3.select("chrom", "start", "end").rows()
    assert extracted.features["ncrna"].height == 1
    assert extracted.features["enhancer"]["source_feature"].to_list() == [
        "dELS",
        "pELS",
    ]


def test_priority_resolution_and_audit_counts() -> None:
    raw = {
        "cds": GenomicSet(pl.DataFrame({"chrom": ["1"], "start": [100], "end": [200]})),
        "utr3": GenomicSet(
            pl.DataFrame({"chrom": ["1"], "start": [150], "end": [250]})
        ),
        "tss_region": GenomicSet(
            pl.DataFrame({"chrom": ["1"], "start": [200], "end": [300]})
        ),
        "ncrna": GenomicSet(
            pl.DataFrame({"chrom": ["1"], "start": [250], "end": [350]})
        ),
        "enhancer": GenomicSet(
            pl.DataFrame({"chrom": ["1"], "start": [300], "end": [400]})
        ),
    }
    ownership = resolve_base_priority(raw)
    assert [ownership.owned_cores[arm].total_size() for arm in FUNCTIONAL_ARMS] == [
        100,
        50,
        50,
        50,
        50,
    ]

    feature_sets = FunctionalFeatureSets(
        features={
            arm: _features([_feature("1", 0, 1, arm, arm)]) for arm in FUNCTIONAL_ARMS
        },
        raw_cores=raw,
        all_exons=_empty_set(),
    )
    audit = feature_audit_table(feature_sets, ownership)
    assert audit.filter(pl.col("arm") == "utr3").item(0, "priority_owned_bases") == 50
    overlaps = pairwise_raw_overlap_table(raw)
    assert (
        overlaps.filter((pl.col("arm_a") == "cds") & (pl.col("arm_b") == "utr3")).item(
            0, "overlap_bases"
        )
        == 50
    )
    assert (
        overlaps.filter((pl.col("arm_a") == "utr3") & (pl.col("arm_b") == "cds")).item(
            0, "overlap_bases"
        )
        == 50
    )


def test_tile_intervals_uses_interval_origin_and_drops_terminal() -> None:
    intervals = pl.DataFrame({"chrom": ["1"], "start": [1000], "end": [1511]})
    tiled = tile_intervals(intervals, source_arm="cds", window_size=255, step_size=128)
    assert tiled.select("start", "end").rows() == [
        (1000, 1255),
        (1128, 1383),
        (1256, 1511),
    ]


def test_candidate_builder_expands_short_features_and_tracks_duplicates() -> None:
    cds_features = _features(
        [
            _feature("1", 1000, 1020, "cds:T1", "cds"),
            _feature("1", 1000, 1020, "cds:T2", "cds"),
        ]
    )
    feature_map = {
        "cds": cds_features,
        "utr3": _features([_feature("1", 2000, 2020, "utr3:T1", "utr3")]),
        "tss_region": _features([_feature("1", 3000, 3512, "tss:T1", "tss_band")]),
        "ncrna": _features([_feature("1", 4000, 4020, "nc:T1", "ncrna_exon")]),
        "enhancer": _features([_feature("1", 5000, 5200, "enh:1", "dELS", strand=".")]),
    }
    raw = {
        arm: GenomicSet(frame.select("chrom", "start", "end"))
        for arm, frame in feature_map.items()
    }
    feature_sets = FunctionalFeatureSets(
        features=feature_map,
        raw_cores=raw,
        all_exons=_empty_set(),
    )
    ownership = resolve_base_priority(raw)
    candidates = build_candidate_windows(
        feature_sets,
        ownership,
        chrom_sizes=pl.DataFrame({"chrom": ["1"], "size": [10_000]}),
        defined=GenomicSet(
            pl.DataFrame({"chrom": ["1"], "start": [0], "end": [10_000]})
        ),
    )

    cds = candidates.windows.filter(pl.col("source_arm") == "cds")
    assert cds.height == 1
    assert cds.item(0, "end") - cds.item(0, "start") == 255
    assert cds.item(0, "contributing_feature_count") == 2
    tss = candidates.windows.filter(pl.col("source_arm") == "tss_region")
    assert tss.select("start").to_series().to_list() == [3000, 3128, 3256]
    enhancer = candidates.windows.filter(pl.col("source_arm") == "enhancer")
    assert enhancer.select("start", "end").row(0) == (4973, 5228)


def test_candidate_builder_rejects_enhancer_exon_overlap_and_bounds() -> None:
    feature_map = {
        "cds": _features([_feature("1", 1000, 1300, "cds", "cds")]),
        "utr3": _features([_feature("1", 2000, 2300, "utr3", "utr3")]),
        "tss_region": _features([_feature("1", 3000, 3512, "tss", "tss_band")]),
        "ncrna": _features([_feature("1", 4000, 4300, "nc", "ncrna_exon")]),
        "enhancer": _features(
            [
                _feature("1", 5000, 5200, "enh:exon", "dELS", strand="."),
                _feature("1", 9000, 9200, "enh:clean", "pELS", strand="."),
                _feature("1", 20, 100, "enh:bounds", "dELS", strand="."),
            ]
        ),
    }
    raw = {
        arm: GenomicSet(frame.select("chrom", "start", "end"))
        for arm, frame in feature_map.items()
    }
    feature_sets = FunctionalFeatureSets(
        features=feature_map,
        raw_cores=raw,
        all_exons=GenomicSet(
            pl.DataFrame({"chrom": ["1"], "start": [5100], "end": [5110]})
        ),
    )
    candidates = build_candidate_windows(
        feature_sets,
        resolve_base_priority(raw),
        chrom_sizes=pl.DataFrame({"chrom": ["1"], "size": [10_000]}),
        defined=GenomicSet(
            pl.DataFrame({"chrom": ["1"], "start": [0], "end": [10_000]})
        ),
    )
    enhancer = candidates.windows.filter(pl.col("source_arm") == "enhancer")
    assert enhancer.select("start", "end").rows() == [(8973, 9228)]
    drop_reasons = candidates.construction_drops.filter(
        pl.col("source_arm") == "enhancer"
    )["drop_reason"].to_list()
    assert sorted(drop_reasons) == ["annotated_exon_overlap", "chromosome_bounds"]


def test_ownership_gate_breaks_exact_tie_by_priority() -> None:
    raw = {
        "cds": _empty_set(),
        "utr3": _empty_set(),
        "tss_region": GenomicSet(
            pl.DataFrame({"chrom": ["1"], "start": [0], "end": [50]})
        ),
        "ncrna": GenomicSet(
            pl.DataFrame({"chrom": ["1"], "start": [50], "end": [100]})
        ),
        "enhancer": _empty_set(),
    }
    features = {
        "cds": _features([]),
        "utr3": _features([]),
        "tss_region": _features([_feature("1", 0, 50, "tss", "tss_band")]),
        "ncrna": _features([_feature("1", 50, 100, "nc", "ncrna_exon")]),
        "enhancer": _features([]),
    }
    feature_sets = FunctionalFeatureSets(
        features=features,
        raw_cores=raw,
        all_exons=_empty_set(),
    )
    ownership = resolve_base_priority(raw)
    candidates = pl.DataFrame(
        {
            "source_arm": ["tss_region", "ncrna"],
            "chrom": ["1", "1"],
            "start": [0, 0],
            "end": [255, 255],
            "contributing_feature_count": [1, 1],
            "source_feature_types": ["tss_band", "ncrna_exon"],
        }
    )
    result = apply_window_ownership_gate(candidates, feature_sets, ownership)
    assert result.retained["source_arm"].to_list() == ["tss_region"]
    assert result.dropped.select("source_arm", "ownership_winner").row(0) == (
        "ncrna",
        "tss_region",
    )


def test_catalog_split_is_nested_and_ids_are_stable() -> None:
    raw = {
        "cds": GenomicSet(pl.DataFrame({"chrom": ["1"], "start": [0], "end": [255]})),
        "utr3": _empty_set(),
        "tss_region": _empty_set(),
        "ncrna": _empty_set(),
        "enhancer": _empty_set(),
    }
    feature_sets = FunctionalFeatureSets(
        features={arm: _features([]) for arm in FUNCTIONAL_ARMS},
        raw_cores=raw,
        all_exons=_empty_set(),
    )
    candidates = pl.DataFrame(
        {
            "source_arm": ["cds"],
            "chrom": ["1"],
            "start": [0],
            "end": [255],
            "contributing_feature_count": [1],
            "source_feature_types": ["cds"],
        }
    )
    gate = apply_window_ownership_gate(
        candidates, feature_sets, resolve_base_priority(raw)
    )
    base = gate.retained
    query_name = base.item(0, "query_name")
    assert query_name == "fa1_cds_1_000000000000_000000000255"
    scored = pl.concat(
        [
            base.with_columns(pl.lit(0.10).alias("proportion_conserved")),
            base.with_columns(
                pl.lit("fa1_cds_1_000000000128_000000000383").alias("query_name"),
                pl.lit(128, dtype=pl.Int64).alias("start"),
                pl.lit(383, dtype=pl.Int64).alias("end"),
                pl.lit(0.20).alias("proportion_conserved"),
            ),
            base.with_columns(
                pl.lit("fa1_cds_1_000000000256_000000000511").alias("query_name"),
                pl.lit(256, dtype=pl.Int64).alias("start"),
                pl.lit(511, dtype=pl.Int64).alias("end"),
                pl.lit(0.099).alias("proportion_conserved"),
            ),
        ]
    )
    catalogs = split_conservation_catalogs(scored)
    assert catalogs.projection.height == 2
    assert catalogs.training.height == 1
    assert catalogs.deferred.height == 1
    projection = to_projection_catalog(catalogs.projection)
    assert projection["source_chrom"].to_list() == ["chr1", "chr1"]
    assert projection["region_label"].to_list() == ["cds", "cds"]


def test_sequence_qc_counts_softmask_gc_and_ambiguity() -> None:
    anchors = pl.DataFrame({"query_name": ["a"]})
    sequences = pl.DataFrame({"query_name": ["a"], "sequence": ["ACgtNN"]})
    annotated = annotate_sequence_fractions(anchors, sequences)
    assert annotated.item(0, "gc_fraction") == pytest.approx(2 / 6)
    assert annotated.item(0, "repeat_masked_fraction") == pytest.approx(2 / 6)
    assert annotated.item(0, "ambiguous_base_fraction") == pytest.approx(2 / 6)


def test_candidate_windows_dataclass_contract() -> None:
    empty = CandidateWindows(
        windows=pl.DataFrame(),
        provenance=pl.DataFrame(),
        construction_drops=pl.DataFrame(),
    )
    assert empty.windows.is_empty()
