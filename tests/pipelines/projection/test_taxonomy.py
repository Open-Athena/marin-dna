"""Tests for ``marin_dna.pipelines.projection.taxonomy``."""

from __future__ import annotations

from marin_dna.pipelines.projection.taxonomy import (
    LeafMeta,
    dedup_by_family,
    normalize_zoonomia_leaf,
    parse_newick_leaves,
)


def test_parse_newick_leaves_filters_ancestor_labels() -> None:
    text = (
        "(Solenodon_paradoxus:0.10467,(Erinaceus_europaeus:0.27567,"
        "(Crocidura_indochinensis:0.19436,Sorex_araneus:0.31878)"
        "fullTreeAnc116:0.01)fullTreeAnc117:0.00304)fullTreeAnc121:0.00019;"
    )
    leaves = parse_newick_leaves(text)
    assert leaves == [
        "Solenodon_paradoxus",
        "Erinaceus_europaeus",
        "Crocidura_indochinensis",
        "Sorex_araneus",
    ]


def test_parse_newick_leaves_filters_primates_anc_labels() -> None:
    text = "((Homo_sapiens:0.1,Pan_troglodytes:0.1)PrimatesAnc7:0.05,Mus_musculus:0.5);"
    leaves = parse_newick_leaves(text)
    assert leaves == ["Homo_sapiens", "Pan_troglodytes", "Mus_musculus"]


def test_normalize_zoonomia_leaf_strips_ab_suffix() -> None:
    assert normalize_zoonomia_leaf("Ateles_geoffroyi_a") == "Ateles_geoffroyi"
    assert normalize_zoonomia_leaf("Hylobates_pileatus_b") == "Hylobates_pileatus"
    # Non-suffixed names pass through unchanged.
    assert normalize_zoonomia_leaf("Homo_sapiens") == "Homo_sapiens"
    # Don't be fooled by names that end in _a/_b coincidentally.
    assert normalize_zoonomia_leaf("Capra_hircus") == "Capra_hircus"


def test_parse_newick_leaves_keeps_ab_suffix() -> None:
    """Regression: _a/_b suffixed leaves must survive parse_newick_leaves.

    The HAL stores these leaves under their raw `Ateles_geoffroyi_a`
    name; halStats / halLiftover fail with "Genome <X> not found" if we
    pass them the normalized binomial. parse_newick_leaves returns the
    raw names; normalize_zoonomia_leaf is only used at NCBI / ST2 lookup
    time. Caught when the first full-tier run failed on
    `halStats --chromSizes Ateles_geoffroyi <hal>` after the script
    collapsed the suffix.
    """
    text = (
        "((Ateles_geoffroyi_a:0.1,Ateles_geoffroyi_b:0.1)PrimatesAnc1:0.05,"
        "Hylobates_pileatus_a:0.2,Homo_sapiens:0.1);"
    )
    leaves = parse_newick_leaves(text)
    assert "Ateles_geoffroyi_a" in leaves
    assert "Ateles_geoffroyi_b" in leaves
    assert "Hylobates_pileatus_a" in leaves
    # parse_newick_leaves does not collapse — that's the script's job.
    assert "Ateles_geoffroyi" not in leaves


def _meta(
    leaf: str,
    family: str | None,
    *,
    order: str | None = None,
    level: str | None = "Scaffold",
    n50: int | None = 100000,
    source: str = "zoonomia_supp_st2",
    accession: str | None = None,
) -> LeafMeta:
    return LeafMeta(
        leaf=leaf,
        family=family,
        order=order,
        accession=accession,
        assembly_level=level,
        contig_n50=n50,
        quality_source=source,
    )


def test_dedup_picks_one_per_family() -> None:
    rows = [
        _meta("Bos_taurus", "Bovidae"),
        _meta("Capra_hircus", "Bovidae"),
        _meta("Ovis_aries", "Bovidae"),
        _meta("Mus_musculus", "Muridae"),
    ]
    winners = dedup_by_family(rows, force_include=frozenset())
    families = [w.family for w in winners]
    assert sorted(families) == ["Bovidae", "Muridae"]


def test_dedup_force_include_overrides_ranking() -> None:
    rows = [
        # Better-ranked Capra_hircus should be skipped because Bos_taurus is forced.
        _meta("Capra_hircus", "Bovidae", level="Complete Genome", n50=200_000_000),
        _meta("Bos_taurus", "Bovidae", level="Scaffold", n50=276_285),
    ]
    winners = dedup_by_family(rows, force_include=frozenset({"Bos_taurus"}))
    assert [w.leaf for w in winners] == ["Bos_taurus"]


def test_dedup_quality_source_outranks_better_proxy_n50() -> None:
    """An ST2-true entry beats a higher-N50 proxy in the same family.

    This is the Bovidae regression that motivated the ranking order: the
    proxy's "better" stats apply to a different assembly than what is in
    the HAL, so we'd rather pick the lower-stats ST2-true entry that we
    *know* matches the HAL.
    """
    rows = [
        _meta(
            "Ovis_canadensis",
            "Bovidae",
            level="Chromosome",
            n50=105_224_494,
            source="ncbi_taxon_proxy",
        ),
        _meta(
            "Bos_taurus",
            "Bovidae",
            level="Chromosome",
            n50=276_285,
            source="zoonomia_supp_st2",
        ),
    ]
    winners = dedup_by_family(rows, force_include=frozenset())
    assert [w.leaf for w in winners] == ["Bos_taurus"]


def test_dedup_assembly_level_beats_n50_within_same_source() -> None:
    rows = [
        _meta("A_one", "Family1", level="Scaffold", n50=10_000_000),
        _meta(
            "A_two", "Family1", level="Complete Genome", n50=1_000
        ),  # tiny N50, but Complete > Scaffold
    ]
    winners = dedup_by_family(rows, force_include=frozenset())
    assert [w.leaf for w in winners] == ["A_two"]


def test_dedup_alphabetical_final_tiebreak() -> None:
    rows = [
        _meta("Bos_taurus", "Family1"),
        _meta("Aoudad_lervia", "Family1"),
    ]
    winners = dedup_by_family(rows, force_include=frozenset())
    assert [w.leaf for w in winners] == ["Aoudad_lervia"]


def test_dedup_drops_rows_without_family() -> None:
    rows = [
        _meta("X", None),
        _meta("Bos_taurus", "Bovidae"),
    ]
    winners = dedup_by_family(rows, force_include=frozenset())
    assert [w.leaf for w in winners] == ["Bos_taurus"]


def test_dedup_default_force_include() -> None:
    rows = [
        _meta("Homo_sapiens", "Hominidae"),
        _meta("Mus_musculus", "Muridae"),
        _meta("Bos_taurus", "Bovidae"),
        _meta("Pan_troglodytes", "Hominidae", level="Complete Genome"),
    ]
    winners = dedup_by_family(rows)  # uses default force_include
    leaves = sorted(w.leaf for w in winners)
    assert leaves == ["Bos_taurus", "Homo_sapiens", "Mus_musculus"]
