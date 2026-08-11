"""Tests for the dashboard nucleotide-dependency catalog (issue #240)."""

from __future__ import annotations

import pytest
from marin_dna_evals.interpretation_catalog import (
    display_region,
    load_nuc_dep_block,
    load_umap_block,
    nuc_dep_candidates,
    ucsc_browser_url,
    umap_candidates,
)


def _block() -> dict:
    """A minimal two-combine, one-model, one-locus nuc_dep block (LDLR coords)."""
    return {
        "combines": ["mean", "max"],
        "models": ["modelA"],
        "loci": {
            "LDLR": {"chrom": "19", "start": 11089299, "end": 11089425, "strand": "+"},
        },
    }


# --- coordinate boundary conversion (the correctness-critical bit) ----------


def test_ucsc_browser_url_converts_0based_halfopen_to_1based_inclusive() -> None:
    # 0-based half-open [11089299, 11089425) → 1-based inclusive 11089300-11089425.
    assert (
        ucsc_browser_url("19", 11089299, 11089425)
        == "https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg38&position=chr19:11089300-11089425"
    )


def test_ucsc_browser_url_adds_chr_prefix_once() -> None:
    url = ucsc_browser_url("chr7", 0, 10)
    assert "position=chr7:1-10" in url
    assert "chrchr" not in url


def test_ucsc_browser_url_rejects_empty_or_inverted_interval() -> None:
    with pytest.raises(AssertionError):
        ucsc_browser_url("1", 100, 100)
    with pytest.raises(AssertionError):
        ucsc_browser_url("1", 100, 50)


def test_display_region_is_1based_with_thousands_separators() -> None:
    assert display_region("19", 11089299, 11089425) == "chr19:11,089,300-11,089,425"


# --- candidate manifest ----------------------------------------------------


def test_nuc_dep_candidates_is_cartesian_product() -> None:
    cands = nuc_dep_candidates(_block(), model_displays={"modelA": "Model A"})
    # 2 combines × 1 model × 1 locus.
    assert len(cands) == 2
    assert {c["svg"] for c in cands} == {
        "mean/LDLR/modelA.svg",
        "max/LDLR/modelA.svg",
    }


def test_nuc_dep_candidate_fields() -> None:
    cands = nuc_dep_candidates(_block(), model_displays={"modelA": "Model A"})
    mean = next(c for c in cands if c["combine"] == "mean")
    assert mean["locus"] == "LDLR"
    assert mean["title"] == "LDLR"  # from LOCUS_META
    assert mean["model_display"] == "Model A"
    assert mean["span"] == 126
    assert mean["strand"] == "+"
    assert mean["display_region"] == "chr19:11,089,300-11,089,425"
    assert mean["ucsc_url"].endswith("chr19:11089300-11089425")


def test_nuc_dep_candidates_model_display_falls_back_to_id() -> None:
    cands = nuc_dep_candidates(_block())  # no model_displays
    assert cands[0]["model_display"] == "modelA"


def test_nuc_dep_candidates_paper_image_gated_on_committed_file(tmp_path) -> None:
    # No screenshot committed → paper.image is None (page won't show a broken img).
    cands = nuc_dep_candidates(_block(), refs_dir=tmp_path)
    assert cands[0]["paper"]["image"] is None
    assert cands[0]["paper"]["citation"]  # citation still present for context

    # Commit a screenshot → image becomes the zip-relative key the loader bundles.
    (tmp_path / "LDLR.png").write_bytes(b"\x89PNG\r\n")
    cands = nuc_dep_candidates(_block(), refs_dir=tmp_path)
    assert cands[0]["paper"]["image"] == "refs/LDLR.png"


def test_nuc_dep_candidates_rejects_bad_locus() -> None:
    bad = {
        "combines": ["mean"],
        "models": ["m"],
        "loci": {"X": {"chrom": "1", "start": 10, "end": 10, "strand": "+"}},
    }
    with pytest.raises(AssertionError):
        nuc_dep_candidates(bad)


def test_nuc_dep_candidates_rejects_locus_larger_than_window() -> None:
    too_big = {
        "combines": ["mean"],
        "models": ["m"],
        "window_size": 100,
        "loci": {
            "X": {"chrom": "1", "start": 0, "end": 150, "strand": "+"}
        },  # 150 > 100
    }
    with pytest.raises(AssertionError, match="exceeds nuc_dep window_size"):
        nuc_dep_candidates(too_big)


# --- real config sanity (doubles as a config-validity check) ----------------


def test_load_nuc_dep_block_real_config_is_wellformed() -> None:
    block = load_nuc_dep_block()
    assert block.get("loci"), "expected at least one nuc_dep locus in the config"
    for locus, coords in block["loci"].items():
        assert {"chrom", "start", "end", "strand"} <= set(coords), locus
        assert coords["end"] > coords["start"], locus
        assert coords["strand"] in ("+", "-"), locus
    # Candidates build cleanly from the real config + real model registry.
    cands = nuc_dep_candidates(block)
    assert cands, "expected at least one candidate artifact"


# --- embedding-UMAP catalog (issue #246) -----------------------------------


def test_umap_candidates_is_model_by_colorby_product() -> None:
    cands = umap_candidates(
        {"models": ["modelA", "modelB"]}, model_displays={"modelA": "Model A"}
    )
    # 2 models × {region, conservation}.
    assert len(cands) == 4
    assert {c["svg"] for c in cands} == {
        "modelA/region.svg",
        "modelA/conservation.svg",
        "modelB/region.svg",
        "modelB/conservation.svg",
    }


def test_umap_candidate_fields_and_display_fallback() -> None:
    cands = umap_candidates(
        {"models": ["modelA", "modelB"]}, model_displays={"modelA": "Model A"}
    )
    a_region = next(
        c for c in cands if c["model"] == "modelA" and c["color_by"] == "region"
    )
    assert a_region["model_display"] == "Model A"
    assert a_region["svg"] == "modelA/region.svg"
    # No display mapping for modelB → falls back to the id.
    b = next(c for c in cands if c["model"] == "modelB")
    assert b["model_display"] == "modelB"


def test_umap_candidates_empty_when_no_models() -> None:
    assert umap_candidates({"models": []}) == []
    assert umap_candidates({}) == []


def test_load_umap_block_real_config_is_wellformed() -> None:
    block = load_umap_block()
    assert block.get("models"), "expected at least one umap_embeddings model"
    # Candidates build cleanly from the real config + real model registry.
    cands = umap_candidates(block)
    assert cands, "expected at least one candidate artifact"
    assert all(c["color_by"] in ("region", "conservation") for c in cands)
