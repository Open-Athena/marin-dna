"""Deterministic review samples and a human inspection checklist."""

from __future__ import annotations

import hashlib

import polars as pl


_IUPAC = frozenset("ACGTRYSWKMBDHVNacgtryswkmbdhvn")


def _digest(seed: int, row: dict[str, object]) -> str:
    identity = "\t".join(
        str(row.get(column, ""))
        for column in [
            "query_name",
            "species",
            "alignment_source",
            "t_chrom",
            "t_start",
        ]
    )
    return hashlib.sha256(f"{seed}\t{identity}".encode()).hexdigest()


def build_inspection_sample(
    rows: pl.DataFrame,
    *,
    seed: int = 417,
    rows_per_region: int = 3,
    fragmented_rows: int = 5,
) -> pl.DataFrame:
    """Select a reproducible review set with explicit ZRS and clade coverage.

    Every ZRS anchor contributes human plus the most phylogenetically distant
    recovered row in each backend/clade scope.  The remainder is a stable hash
    sample, so reruns are directly comparable.
    """
    required = {
        "query_name",
        "source_chrom",
        "source_start",
        "source_end",
        "region_label",
        "species",
        "alignment_source",
        "clade",
        "phylogenetic_rank",
        "t_chrom",
        "t_start",
        "t_end",
        "t_strand",
        "t_src_size",
        "fragment_count",
        "aligned_bases",
        "sequence",
    }
    missing = required - set(rows.columns)
    assert not missing, f"inspection rows missing columns: {sorted(missing)}"
    assert rows_per_region > 0
    assert fragmented_rows >= 0
    assert rows.select("query_name", "species").is_unique().all()

    records = rows.to_dicts()
    zrs = [row for row in records if str(row["query_name"]).lower().startswith("zrs_")]
    zrs_names = sorted({str(row["query_name"]) for row in zrs})
    selected_keys: set[tuple[str, str]] = set()
    selection_reasons: dict[tuple[str, str], str] = {}
    for query_name in zrs_names:
        query_rows = [row for row in zrs if row["query_name"] == query_name]
        human_rows = [
            row for row in query_rows if row["alignment_source"] == "human_reference"
        ]
        assert len(human_rows) == 1, f"{query_name} must have one human reference row"
        human_key = (query_name, str(human_rows[0]["species"]))
        selected_keys.add(human_key)
        selection_reasons[human_key] = "required_zrs_human"

        scopes = sorted(
            {
                (str(row["alignment_source"]), str(row["clade"]))
                for row in query_rows
                if row["alignment_source"] != "human_reference"
            }
        )
        for backend, clade in scopes:
            candidates = [
                row
                for row in query_rows
                if row["alignment_source"] == backend and row["clade"] == clade
            ]
            representative = sorted(
                candidates,
                key=lambda row: (-int(row["phylogenetic_rank"]), _digest(seed, row)),
            )[0]
            key = (query_name, str(representative["species"]))
            selected_keys.add(key)
            selection_reasons[key] = "required_zrs_backend_clade"

    for region_label in ["cds", "ccre_non_promoter"]:
        region_rows = [row for row in records if row["region_label"] == region_label]
        assert region_rows, f"inspection requires recovered {region_label} rows"
        for row in sorted(region_rows, key=lambda candidate: _digest(seed, candidate))[
            :rows_per_region
        ]:
            key = (str(row["query_name"]), str(row["species"]))
            selected_keys.add(key)
            selection_reasons.setdefault(key, f"stable_{region_label}_sample")

    fragmented = [row for row in records if int(row["fragment_count"]) > 1]
    for row in sorted(fragmented, key=lambda candidate: _digest(seed, candidate))[
        :fragmented_rows
    ]:
        key = (str(row["query_name"]), str(row["species"]))
        selected_keys.add(key)
        selection_reasons.setdefault(key, "fragmented_mapping_sample")

    selected = [
        row
        for row in records
        if (str(row["query_name"]), str(row["species"])) in selected_keys
    ]
    assert selected, "inspection sample is empty"
    clade_counts: dict[str, str] = {}
    for query_name in sorted({str(row["query_name"]) for row in records}):
        query_rows = [row for row in records if row["query_name"] == query_name]
        counts: dict[str, int] = {}
        for row in query_rows:
            clade = str(row["clade"])
            counts[clade] = counts.get(clade, 0) + 1
        clade_counts[query_name] = "; ".join(
            f"{clade}={counts[clade]}" for clade in sorted(counts)
        )

    output = pl.DataFrame(selected, schema=rows.schema).with_columns(
        pl.Series(
            "selection_reason",
            [
                selection_reasons[(str(row["query_name"]), str(row["species"]))]
                for row in selected
            ],
        ),
        pl.col("sequence")
        .map_elements(
            lambda sequence: set(sequence) <= _IUPAC,
            return_dtype=pl.Boolean,
        )
        .alias("valid_iupac"),
        pl.col("sequence").str.len_bytes().alias("sequence_length"),
        (pl.col("t_end") - pl.col("t_start")).alias("target_span"),
        (pl.col("source_end") - pl.col("source_start")).alias("anchor_length"),
        pl.when(pl.col("t_strand") == "-")
        .then(pl.lit("reverse-complemented_to_human_anchor"))
        .otherwise(pl.lit("source_forward_matches_human_anchor"))
        .alias("extracted_orientation"),
        pl.struct("query_name")
        .map_elements(
            lambda row: clade_counts[str(row["query_name"])],
            return_dtype=pl.String,
        )
        .alias("retained_species_by_clade"),
    )
    output = output.with_columns(
        (pl.col("aligned_bases") / pl.col("anchor_length"))
        .clip(upper_bound=1.0)
        .alias("alignment_coverage_fraction"),
        (pl.col("anchor_length") - pl.col("aligned_bases"))
        .clip(lower_bound=0)
        .alias("uncovered_anchor_bases"),
    )
    assert output["valid_iupac"].all()
    assert (output["sequence_length"] == 255).all()
    assert (output["target_span"] == 255).all()
    assert (output["t_start"] >= 0).all()
    assert (output["t_end"] <= output["t_src_size"]).all()
    assert set(zrs_names) <= set(output["query_name"].to_list())
    return output.sort("query_name", "selection_reason", "alignment_source", "species")


def build_rejection_inspection_sample(
    rejected: pl.DataFrame, *, seed: int = 417, rows_per_reason: int = 2
) -> pl.DataFrame:
    """Select deterministic explicit-rejection examples, favoring fragments."""
    required = {
        "query_name",
        "source_chrom",
        "source_start",
        "source_end",
        "region_label",
        "species",
        "alignment_source",
        "rejection_reason",
        "detail",
        "fragment_count",
    }
    missing = required - set(rejected.columns)
    assert not missing, f"inspection rejections missing columns: {sorted(missing)}"
    assert rows_per_reason > 0
    if rejected.is_empty():
        return rejected
    selected: list[dict[str, object]] = []
    for reason in sorted(rejected["rejection_reason"].unique().to_list()):
        reason_rows = rejected.filter(pl.col("rejection_reason") == reason).to_dicts()
        selected.extend(
            sorted(
                reason_rows,
                key=lambda row: (-int(row["fragment_count"]), _digest(seed, row)),
            )[:rows_per_reason]
        )
    return pl.DataFrame(selected, schema=rejected.schema).sort(
        "rejection_reason", "query_name", "species"
    )


def assert_zrs_broad_recovery(rows: pl.DataFrame, *, minimum_clades: int = 2) -> None:
    """Fail the smoke/full build if either ZRS anchor loses broad non-mammal recovery."""
    assert minimum_clades >= 2
    zrs = rows.filter(pl.col("query_name").str.to_lowercase().str.starts_with("zrs_"))
    zrs_names = zrs["query_name"].unique().to_list()
    assert zrs_names, "no ZRS positive-control anchor found"
    for query_name in zrs_names:
        query_rows = zrs.filter(pl.col("query_name") == query_name)
        assert (
            query_rows.filter(pl.col("alignment_source") == "human_reference").height
            == 1
        )
        non_mammal_clades = query_rows.filter(
            pl.col("alignment_source") == "ucsc_multiz100way"
        )["clade"].n_unique()
        assert non_mammal_clades >= minimum_clades, (
            f"{query_name} recovered {non_mammal_clades} non-mammal clades; "
            f"expected at least {minimum_clades}"
        )


def render_inspection_report(
    sample: pl.DataFrame,
    rejected_sample: pl.DataFrame,
    *,
    seed: int,
    require_zrs: bool = True,
) -> str:
    """Render a reviewable Markdown checklist without claiming human approval."""
    assert not sample.is_empty()
    zrs_names = sorted(
        name
        for name in sample["query_name"].unique().to_list()
        if str(name).lower().startswith("zrs_")
    )
    if require_zrs:
        assert zrs_names, "required ZRS controls are absent from inspection sample"
        zrs_status = f"required ZRS anchors: `{', '.join(zrs_names)}`."
        zrs_checklist = (
            "- [ ] Inspect both ZRS anchors across human, mammals, and recovered "
            "non-mammal clades."
        )
    else:
        assert not zrs_names, "sidecar ZRS controls must not enter the full dataset"
        zrs_status = (
            "ZRS positive control: separate sidecar QC; intentionally absent from "
            "the conservation-filtered grid."
        )
        zrs_checklist = "- [ ] Review the separate ZRS sidecar QC before upload."
    table_lines = [
        "| Human interval | Anchor | Species | Backend | Clade recovery | Target | Strand | Coverage | Gaps | Fragments | Reason |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in sample.to_dicts():
        source = f"{row['source_chrom']}:{row['source_start']}-{row['source_end']}"
        target = f"{row['t_chrom']}:{row['t_start']}-{row['t_end']}"
        table_lines.append(
            "| "
            + " | ".join(
                [
                    source,
                    str(row["query_name"]),
                    str(row["species"]),
                    str(row["alignment_source"]),
                    str(row["retained_species_by_clade"]),
                    target,
                    str(row["t_strand"]),
                    f"{float(row['alignment_coverage_fraction']):.3f}",
                    str(row["uncovered_anchor_bases"]),
                    str(row["fragment_count"]),
                    str(row["selection_reason"]),
                ]
            )
            + " |"
        )
    rejection_lines = [
        "| Human interval | Anchor | Species | Backend | Reason | Detail | Fragments |",
        "|---|---|---|---|---|---|---:|",
    ]
    for row in rejected_sample.to_dicts():
        source = f"{row['source_chrom']}:{row['source_start']}-{row['source_end']}"
        rejection_lines.append(
            "| "
            + " | ".join(
                [
                    source,
                    str(row["query_name"]),
                    str(row["species"]),
                    str(row["alignment_source"]),
                    str(row["rejection_reason"]),
                    str(row["detail"]).replace("|", "\\|"),
                    str(row["fragment_count"]),
                ]
            )
            + " |"
        )
    sequence_lines = []
    for row in sample.to_dicts():
        sequence_lines.extend(
            [
                f">{row['query_name']}|{row['species']}|{row['alignment_source']}",
                str(row["sequence"]),
            ]
        )
    return (
        "# Manual projection inspection\n\n"
        f"Status: **pending human review**. Deterministic sample seed: `{seed}`. "
        f"Rows: `{sample.height}`; {zrs_status}\n\n"
        "Automated prechecks passed for every listed row: 255 bp sequence and target "
        "span, valid IUPAC DNA, non-negative 0-based half-open coordinates, and target "
        "bounds within the source assembly sequence.\n\n"
        "## Human checklist\n\n"
        f"{zrs_checklist}\n"
        "- [ ] Confirm reverse-strand rows are oriented to the human anchor.\n"
        "- [ ] Confirm fragmented mappings are biologically plausible and not duplicated.\n"
        "- [ ] Cross-check selected MultiZ rows against the UCSC browser or staged raw MAF.\n"
        "- [ ] Spot-check HAL source-assembly coordinates against the pinned alignment/genome.\n"
        "- [ ] Record reviewer, date, and any exclusions before upload.\n\n"
        "## Review sample\n\n"
        + "\n".join(table_lines)
        + "\n\n## Rejected projection sample\n\n"
        + "\n".join(rejection_lines)
        + "\n\n<details><summary>Case-preserving sequences</summary>\n\n```fasta\n"
        + "\n".join(sequence_lines)
        + "\n```\n\n</details>\n"
    )
