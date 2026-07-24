"""Projection-coverage and redundancy audit for the issue #402 species panel."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from marin_dna.pipelines.rag_glm.dataset import (
    BASES_PER_SLOT,
    PROVISIONAL_SPECIES_ORDER,
    stable_anchor_rank,
    validate_species_order,
)


def projection_audit_table(
    rows: pl.DataFrame,
    *,
    sample_anchor_ids: list[str],
    species: list[str],
) -> pl.DataFrame:
    """Summarize projection success and sequence ambiguity per species."""
    assert sample_anchor_ids, "sample_anchor_ids is empty"
    assert len(sample_anchor_ids) == len(set(sample_anchor_ids))
    assert {"query_name", "species", "sequence"} <= set(rows.columns)

    sample = rows.filter(
        pl.col("query_name").is_in(sample_anchor_ids) & pl.col("species").is_in(species)
    )
    duplicate_pairs = (
        sample.group_by(["query_name", "species"]).len().filter(pl.col("len") != 1)
    )
    assert duplicate_pairs.is_empty(), duplicate_pairs.head(10).to_dicts()
    assert sample.filter(
        pl.col("sequence").str.len_chars() != BASES_PER_SLOT
    ).is_empty()

    observed = (
        sample.with_columns(
            pl.col("sequence")
            .str.replace_all(r"[ACGTacgt]", "")
            .str.len_chars()
            .alias("_ambiguous_bases")
        )
        .group_by("species")
        .agg(
            pl.col("query_name").n_unique().alias("n_projected"),
            (pl.col("_ambiguous_bases") > 0).sum().alias("n_ambiguous_windows"),
            pl.col("_ambiguous_bases").sum().alias("n_ambiguous_bases"),
        )
    )
    n_sample = len(sample_anchor_ids)
    return (
        pl.DataFrame({"species": species})
        .join(observed, on="species", how="left")
        .with_columns(
            pl.col("n_projected").fill_null(0).cast(pl.Int64),
            pl.col("n_ambiguous_windows").fill_null(0).cast(pl.Int64),
            pl.col("n_ambiguous_bases").fill_null(0).cast(pl.Int64),
            pl.lit(n_sample).alias("n_sample_anchors"),
        )
        .with_columns(
            (pl.col("n_projected") / n_sample).alias("projection_success"),
            (pl.col("n_ambiguous_bases") / (pl.col("n_projected") * BASES_PER_SLOT))
            .fill_nan(0.0)
            .alias("ambiguous_base_fraction"),
        )
        .sort(["projection_success", "species"], descending=[True, False])
    )


def pairwise_identity_table(
    rows: pl.DataFrame,
    *,
    species_order: tuple[str, ...] = PROVISIONAL_SPECIES_ORDER,
) -> pl.DataFrame:
    """Compute a same-position identity proxy for projected ortholog redundancy."""
    order = validate_species_order(species_order)
    non_human = order[:-1]
    assert {"query_name", "species", "sequence"} <= set(rows.columns)

    by_species: dict[str, dict[str, str]] = {species: {} for species in order}
    for query_name, species, sequence in rows.select(
        "query_name", "species", "sequence"
    ).iter_rows():
        if species in by_species:
            assert query_name not in by_species[species], (
                f"duplicate row for {(query_name, species)}"
            )
            by_species[species][query_name] = sequence

    records: list[dict[str, Any]] = []
    valid_codes = np.array([ord(base) for base in "ACGTacgt"], dtype=np.uint8)
    for species_a, species_b in combinations(non_human, 2):
        shared = sorted(set(by_species[species_a]) & set(by_species[species_b]))
        if not shared:
            records.append(
                {
                    "species_a": species_a,
                    "species_b": species_b,
                    "n_shared_anchors": 0,
                    "n_compared_bases": 0,
                    "same_position_identity": None,
                }
            )
            continue
        sequence_a = "".join(by_species[species_a][anchor] for anchor in shared)
        sequence_b = "".join(by_species[species_b][anchor] for anchor in shared)
        array_a = np.frombuffer(sequence_a.upper().encode("ascii"), dtype=np.uint8)
        array_b = np.frombuffer(sequence_b.upper().encode("ascii"), dtype=np.uint8)
        comparable = np.isin(array_a, valid_codes) & np.isin(array_b, valid_codes)
        n_compared = int(comparable.sum())
        identity = (
            float((array_a[comparable] == array_b[comparable]).mean())
            if n_compared
            else None
        )
        records.append(
            {
                "species_a": species_a,
                "species_b": species_b,
                "n_shared_anchors": len(shared),
                "n_compared_bases": n_compared,
                "same_position_identity": identity,
            }
        )
    return pl.DataFrame(records).sort(
        "same_position_identity", descending=True, nulls_last=True
    )


def run_species_audit(
    *,
    source_parquet: str,
    species_tsv: str | Path,
    statistics_path: str | Path,
    pairwise_path: str | Path,
    sample_ids_path: str | Path,
    summary_path: str | Path,
    sample_size: int,
    sample_seed: int,
    panel: tuple[str, ...] = PROVISIONAL_SPECIES_ORDER,
) -> None:
    """Stream the canonical projection and write the Phase-A audit artifacts."""
    validate_species_order(panel)
    assert sample_size > 0

    pool = pl.read_csv(species_tsv, separator="\t")["species"].to_list()
    assert "Homo_sapiens" in pool
    assert set(panel) <= set(pool), (
        f"panel species missing from pool: {set(panel) - set(pool)}"
    )

    source = pl.scan_parquet(source_parquet)
    if "augmentation" in source.collect_schema().names():
        source = source.filter(pl.col("augmentation") == "+")

    human_anchor_ids = (
        source.filter(pl.col("species") == "Homo_sapiens")
        .select("query_name")
        .unique()
        .collect(engine="streaming")["query_name"]
        .to_list()
    )
    assert len(human_anchor_ids) >= sample_size, (
        f"only {len(human_anchor_ids)} human anchors; need {sample_size}"
    )
    sample_anchor_ids = sorted(
        human_anchor_ids,
        key=lambda anchor: (stable_anchor_rank(anchor, sample_seed), anchor),
    )[:sample_size]

    sample_rows = (
        source.filter(
            pl.col("query_name").is_in(sample_anchor_ids)
            & pl.col("species").is_in(pool)
        )
        .select("query_name", "species", "sequence")
        .collect(engine="streaming")
    )
    statistics = projection_audit_table(
        sample_rows, sample_anchor_ids=sample_anchor_ids, species=pool
    )
    pairwise = pairwise_identity_table(sample_rows, species_order=panel)

    for output_path in (statistics_path, pairwise_path, sample_ids_path, summary_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    statistics.write_csv(statistics_path, separator="\t")
    pairwise.write_csv(pairwise_path, separator="\t")
    Path(sample_ids_path).write_text("\n".join(sample_anchor_ids) + "\n")

    panel_statistics = (
        pl.DataFrame({"species": list(panel)})
        .join(statistics, on="species", how="left")
        .to_dicts()
    )
    summary = {
        "source_parquet": source_parquet,
        "sample_seed": sample_seed,
        "sample_rank": "blake2b-64-v1",
        "sample_size": sample_size,
        "n_human_anchors": len(human_anchor_ids),
        "candidate_pool_size": len(pool),
        "provisional_species_order": list(panel),
        "panel_statistics": panel_statistics,
        "pairwise_identity_min": pairwise["same_position_identity"].min(),
        "pairwise_identity_max": pairwise["same_position_identity"].max(),
    }
    Path(summary_path).write_text(json.dumps(summary, indent=2) + "\n")
