"""Species-manifest selection and validation for the vertebrate dataset.

The committed candidate manifest is the source of truth.  Selection is
deterministic and deliberately boring: an explicit pinned priority, assembly
level, contig N50, and finally the alignment name decide the winner within a
taxonomic family.  Runtime name matching is never used to decide taxonomy or
backend ownership.
"""

from __future__ import annotations

import polars as pl


BACKENDS = {"zoonomia_cactus", "ucsc_multiz100way"}
ASSEMBLY_LEVEL_RANK = {
    "Complete Genome": 3,
    "Chromosome": 2,
    "Scaffold": 1,
    "Contig": 0,
    "unknown": -1,
}
REQUIRED_MANIFEST_COLUMNS = {
    "alignment_name",
    "scientific_name",
    "assembly",
    "taxonomy_id",
    "family",
    "clade",
    "phylogenetic_rank",
    "backend",
    "selection_priority",
    "assembly_level",
    "contig_n50",
    "selected",
    "selection_reason",
}


def _as_int(value: object) -> int:
    assert isinstance(value, int | str) and not isinstance(value, bool)
    return int(value)


def _ranking_key(row: dict[str, object]) -> tuple[int, int, int, str]:
    """Return the deterministic lower-is-better family selection key."""
    priority = _as_int(row["selection_priority"])
    assembly_level = str(row["assembly_level"])
    contig_n50 = _as_int(row["contig_n50"])
    return (
        -priority,
        -ASSEMBLY_LEVEL_RANK.get(assembly_level, -1),
        -contig_n50,
        str(row["alignment_name"]),
    )


def select_family_representatives(candidates: pl.DataFrame) -> pl.DataFrame:
    """Select exactly one target per family and record every decision.

    The input must contain all required manifest fields except ``selected``
    and ``selection_reason``.  Those two fields are replaced if present so a
    regenerated manifest cannot accidentally retain stale decisions.
    """
    required = REQUIRED_MANIFEST_COLUMNS - {"selected", "selection_reason"}
    missing = required - set(candidates.columns)
    assert not missing, f"candidate manifest missing columns: {sorted(missing)}"
    assert candidates.height > 0, "candidate manifest must not be empty"

    rows = candidates.drop(
        [c for c in ("selected", "selection_reason") if c in candidates.columns]
    ).to_dicts()
    by_family: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        family = str(row["family"])
        assert family, f"empty family for {row['alignment_name']}"
        by_family.setdefault(family, []).append(row)

    selected_rows: list[dict[str, object]] = []
    for family in sorted(by_family):
        family_rows = sorted(by_family[family], key=_ranking_key)
        winner = str(family_rows[0]["alignment_name"])
        for row in family_rows:
            is_selected = str(row["alignment_name"]) == winner
            selected_rows.append(
                {
                    **row,
                    "selected": is_selected,
                    "selection_reason": (
                        "selected_best_pinned_assembly_in_family"
                        if is_selected
                        else f"excluded_lower_ranked_than:{winner}"
                    ),
                }
            )

    return pl.DataFrame(selected_rows).sort("backend", "family", "alignment_name")


def validate_species_manifest(manifest: pl.DataFrame) -> None:
    """Assert the combined target manifest obeys issue #417's species policy."""
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    assert not missing, f"species manifest missing columns: {sorted(missing)}"
    assert manifest.height > 0, "species manifest must not be empty"
    assert sum(manifest.null_count().row(0)) == 0, (
        "species manifest contains null values"
    )

    assert set(manifest["backend"].to_list()) <= BACKENDS
    assert manifest["alignment_name"].n_unique() == manifest.height
    assert (manifest["scientific_name"].str.len_chars() > 0).all()
    assert (manifest["assembly"].str.len_chars() > 0).all()
    assert (manifest["family"].str.len_chars() > 0).all()
    assert (manifest["selection_reason"].str.len_chars() > 0).all()
    assert (manifest["taxonomy_id"] > 0).all()
    assert (manifest["contig_n50"] >= 0).all()
    assert (manifest["phylogenetic_rank"] >= 0).all()

    scientific = set(manifest["scientific_name"].to_list())
    assert "Homo sapiens" not in scientific, (
        "human is a reference row, never a projection target"
    )

    multiz = manifest.filter(pl.col("backend") == "ucsc_multiz100way")
    assert not multiz.is_empty(), "manifest must include MultiZ targets"
    assert (multiz["clade"] != "mammals").all(), (
        "MultiZ target manifest must exclude every mammal"
    )

    zoonomia = manifest.filter(pl.col("backend") == "zoonomia_cactus")
    assert not zoonomia.is_empty(), "manifest must include Zoonomia targets"
    assert (zoonomia["clade"] == "mammals").all(), (
        "Zoonomia target manifest may contain only mammals"
    )

    regenerated = select_family_representatives(manifest)
    decision_columns = ["alignment_name", "selected", "selection_reason"]
    expected = manifest.select(decision_columns).sort("alignment_name")
    observed = regenerated.select(decision_columns).sort("alignment_name")
    assert expected.equals(observed), (
        "committed selected/reason fields do not match deterministic family policy"
    )

    selected = manifest.filter(pl.col("selected"))
    assert selected.height == selected["family"].n_unique(), (
        "selected targets must contain at most one species per family"
    )
    assert selected.height == selected["taxonomy_id"].n_unique(), (
        "duplicate taxonomy IDs across selected targets"
    )
    assert selected.height == selected["assembly"].n_unique(), (
        "duplicate assemblies across selected targets"
    )
    assert manifest["family"].n_unique() == selected.height, (
        "every candidate family must have exactly one selected target"
    )


def read_species_manifest(path: str) -> pl.DataFrame:
    """Read and validate a committed TSV species manifest."""
    manifest = pl.read_csv(path, separator="\t")
    validate_species_manifest(manifest)
    return manifest
