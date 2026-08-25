"""Uniform-window selection from pinned GPN-Star positional entropy scores.

GPN-Star's canonical Parquet files use bare chromosome names and 1-based
positions.  This module converts positions to the project's internal 0-based,
half-open convention at the read boundary and counts selected bases in the
historical 255 bp / 128 bp-stride window grid without expanding an interval
join.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


GPN_ASSIGNMENT_RECIPE = "exp232-v4-plus-exp326-arm-a-remainder-v1"
GPN_ARMS: tuple[str, ...] = (
    "cds",
    "utr3",
    "tss_region_and_utr5",
    "ncrna_exon",
    "enhancer",
    "background",
)
_DIRECT_V4_ARMS = frozenset(GPN_ARMS[:4])
_NON_CCRE_FUNCTIONAL_FRACS = (
    "cds_frac",
    "utr3_frac",
    "tss_region_and_utr5_frac",
    "ncrna_exon_frac",
)


@dataclass(frozen=True)
class GpnEntropyShard:
    """One checksum-pinned canonical GPN-Star entropy chromosome shard."""

    chrom: str
    path: str
    rows: int
    size_bytes: int
    sha256: str
    min_pos: int
    max_pos: int


def read_gpn_entropy_manifest(path: str | Path) -> dict[str, GpnEntropyShard]:
    """Read and validate the task-owned manifest for GPN-Star-P entropy."""
    frame = pl.read_csv(path, separator="\t")
    expected = {
        "chrom",
        "path",
        "rows",
        "size_bytes",
        "sha256",
        "min_pos",
        "max_pos",
    }
    missing = expected - set(frame.columns)
    assert not missing, f"GPN entropy manifest missing columns: {sorted(missing)}"
    assert frame["chrom"].n_unique() == frame.height
    assert frame["path"].n_unique() == frame.height
    assert (frame["rows"] > 0).all()
    assert (frame["size_bytes"] > 0).all()
    assert (frame["min_pos"] >= 1).all()
    assert (frame["max_pos"] >= frame["min_pos"]).all()
    assert frame["sha256"].str.contains(r"^[0-9a-f]{64}$").all()
    assert (
        frame["path"].str.starts_with(
            "data/gpn-star-hg38-p243-200m/entropy/entropy_chr"
        )
    ).all()
    return {
        str(row["chrom"]): GpnEntropyShard(
            chrom=str(row["chrom"]),
            path=str(row["path"]),
            rows=int(row["rows"]),
            size_bytes=int(row["size_bytes"]),
            sha256=str(row["sha256"]),
            min_pos=int(row["min_pos"]),
            max_pos=int(row["max_pos"]),
        )
        for row in frame.to_dicts()
    }


def sha256_file(path: str | Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gpn_entropy_file(path: str | Path, shard: GpnEntropyShard) -> None:
    """Require the downloaded shard to match its immutable release manifest."""
    source = Path(path)
    assert source.stat().st_size == shard.size_bytes
    assert sha256_file(source) == shard.sha256


def _add_grid_counts(
    counts: np.ndarray,
    positions_0based: np.ndarray,
    *,
    step_size: int,
    window_size: int,
) -> None:
    """Add selected positions to every regular-grid window containing them."""
    grid_indices = positions_0based // step_size
    in_current_grid = grid_indices < counts.size
    if in_current_grid.any():
        counts += np.bincount(
            grid_indices[in_current_grid], minlength=counts.size
        ).astype(counts.dtype, copy=False)

    overlap = window_size - step_size
    in_previous_grid = (
        (grid_indices > 0)
        & ((positions_0based % step_size) < overlap)
        & ((grid_indices - 1) < counts.size)
    )
    if in_previous_grid.any():
        counts += np.bincount(
            grid_indices[in_previous_grid] - 1, minlength=counts.size
        ).astype(counts.dtype, copy=False)


def score_gpn_entropy_windows(
    windows_path: str | Path,
    entropy_path: str | Path,
    scored_path: str | Path,
    stats_path: str | Path,
    *,
    chrom: str,
    entropy_cutoff: float,
    expected_rows: int,
    expected_min_pos: int,
    expected_max_pos: int,
    step_size: int = 128,
    window_size: int = 255,
    batch_size: int = 1_048_576,
) -> dict[str, int | float | str]:
    """Score one chromosome's retained grid windows with bounded peak memory.

    A source position passes only when ``entropy_calibrated < entropy_cutoff``.
    Missing source positions therefore contribute zero selected bases.
    """
    assert chrom.startswith("chr")
    assert 0.0 < entropy_cutoff < 1.0
    assert 0 < step_size <= window_size
    assert batch_size > 0
    assert 1 <= expected_min_pos <= expected_max_pos

    windows = pl.read_csv(
        windows_path,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end", "name"],
        schema_overrides={
            "chrom": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
            "name": pl.String,
        },
    )
    assert windows.height > 0
    assert windows["chrom"].unique().to_list() == [chrom]
    assert windows["name"].n_unique() == windows.height
    assert windows["start"].is_sorted()
    assert (windows["start"] % step_size == 0).all()
    assert (windows["end"] - windows["start"] == window_size).all()

    max_grid_index = int(windows["start"].max()) // step_size
    counts = np.zeros(max_grid_index + 1, dtype=np.int32)
    parquet = pq.ParquetFile(entropy_path)
    required = {"chrom", "pos", "ref", "entropy_calibrated"}
    missing = required - set(parquet.schema_arrow.names)
    assert not missing, f"GPN entropy shard missing columns: {sorted(missing)}"
    assert parquet.metadata.num_rows == expected_rows
    assert pa.types.is_integer(parquet.schema_arrow.field("pos").type)
    assert pa.types.is_floating(
        parquet.schema_arrow.field("entropy_calibrated").type
    )

    observed_rows = 0
    selected_source_positions = 0
    first_pos: int | None = None
    previous_pos = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["pos", "entropy_calibrated"],
    ):
        assert batch.column(0).null_count == 0
        assert batch.column(1).null_count == 0
        positions_1based = batch.column(0).to_numpy(zero_copy_only=False)
        entropy = batch.column(1).to_numpy(zero_copy_only=False)
        assert positions_1based.size == entropy.size
        assert positions_1based.size > 0
        if first_pos is None:
            first_pos = int(positions_1based[0])
        assert positions_1based[0] > previous_pos
        assert (positions_1based[1:] > positions_1based[:-1]).all()
        assert np.isfinite(entropy).all()
        previous_pos = int(positions_1based[-1])
        observed_rows += int(positions_1based.size)

        selected = entropy < entropy_cutoff
        selected_source_positions += int(selected.sum())
        if selected.any():
            positions_0based = positions_1based[selected].astype(np.int64) - 1
            assert (positions_0based >= 0).all()
            _add_grid_counts(
                counts,
                positions_0based,
                step_size=step_size,
                window_size=window_size,
            )

    assert observed_rows == expected_rows
    assert first_pos == expected_min_pos
    assert previous_pos == expected_max_pos
    window_counts = counts[(windows["start"].to_numpy() // step_size).astype(int)]
    assert (window_counts >= 0).all()
    assert (window_counts <= window_size).all()
    scored = windows.with_columns(
        pl.Series("gpn_selected_bases", window_counts, dtype=pl.Int32),
        pl.Series(
            "proportion_gpn_selected",
            window_counts / window_size,
            dtype=pl.Float64,
        ),
        pl.lit(entropy_cutoff).alias("gpn_entropy_cutoff"),
    )
    scored_output = Path(scored_path)
    scored_output.parent.mkdir(parents=True, exist_ok=True)
    scored.write_parquet(scored_output)

    stats: dict[str, int | float | str] = {
        "chrom": chrom,
        "entropy_cutoff_strict_lt": entropy_cutoff,
        "source_rows": observed_rows,
        "minimum_source_position_1based": first_pos,
        "maximum_source_position_1based": previous_pos,
        "selected_source_positions": selected_source_positions,
        "uniform_windows": windows.height,
        "window_selected_base_observations": int(window_counts.sum()),
        "maximum_selected_bases_in_window": int(window_counts.max()),
    }
    stats_output = Path(stats_path)
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return stats


def _write_bed_gzip(frame: pl.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.") as temp:
        temporary = Path(temp)
        plain = temporary / "anchors.bed"
        compressed = temporary / "anchors.bed.gz"
        frame.write_csv(plain, separator="\t", include_header=False)
        with (
            plain.open("rb") as source,
            compressed.open("wb") as raw,
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as target,
        ):
            shutil.copyfileobj(source, target, length=1024 * 1024)
        compressed.replace(output)


def write_gpn_selection_outputs(
    scored_paths: list[str],
    stats_paths: list[str],
    selected_path: str | Path,
    bed_path: str | Path,
    summary_path: str | Path,
    *,
    min_selected_bases: int,
    expected_uniform_windows: int | None = None,
    expected_selected_source_positions: int | None = None,
    expected_windows_ge_10pct: int | None = None,
    expected_windows_ge_20pct: int | None = None,
) -> dict[str, object]:
    """Write the >=20% anchors and the complete 10%/20% count audit."""
    assert scored_paths and len(scored_paths) == len(stats_paths)
    assert min_selected_bases == 51
    lazy = pl.concat([pl.scan_parquet(path) for path in scored_paths], how="vertical")
    counts = (
        lazy.group_by("chrom")
        .agg(
            uniform_windows=pl.len(),
            windows_ge_10pct=(pl.col("gpn_selected_bases") >= 26).sum(),
            windows_ge_20pct=(pl.col("gpn_selected_bases") >= 51).sum(),
            window_selected_base_observations=pl.col("gpn_selected_bases").sum(),
        )
        .sort("chrom")
        .collect(engine="streaming")
    )
    totals = counts.select(pl.exclude("chrom").sum()).row(0, named=True)
    source_stats = [json.loads(Path(path).read_text()) for path in stats_paths]
    selected_source_positions = sum(
        int(item["selected_source_positions"]) for item in source_stats
    )

    if expected_uniform_windows is not None:
        assert totals["uniform_windows"] == expected_uniform_windows
    if expected_selected_source_positions is not None:
        assert selected_source_positions == expected_selected_source_positions
    if expected_windows_ge_10pct is not None:
        assert totals["windows_ge_10pct"] == expected_windows_ge_10pct
    if expected_windows_ge_20pct is not None:
        assert totals["windows_ge_20pct"] == expected_windows_ge_20pct

    selected_output = Path(selected_path)
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    lazy.filter(pl.col("gpn_selected_bases") >= min_selected_bases).sort(
        "chrom", "start", "name"
    ).sink_parquet(selected_output)
    selected = pl.read_parquet(selected_output)
    assert selected.height == totals["windows_ge_20pct"]
    assert selected["name"].n_unique() == selected.height
    _write_bed_gzip(
        selected.select(
            pl.col("chrom").str.strip_prefix("chr"), "start", "end", "name"
        ),
        bed_path,
    )

    summary: dict[str, object] = {
        "coordinate_contract": {
            "gpn_parquet": "1-based position",
            "pipeline": "0-based half-open",
            "conversion": "base=[pos-1,pos)",
        },
        "selection_contract": {
            "base": "entropy_calibrated < gpn_entropy_cutoff",
            "window_ge_10pct": "gpn_selected_bases >= 26",
            "window_ge_20pct": "gpn_selected_bases >= 51",
            "missing_positions": "non-passing",
        },
        "totals": {
            **{key: int(value) for key, value in totals.items()},
            "selected_source_positions": selected_source_positions,
        },
        "by_chrom": counts.to_dicts(),
        "source_shards": source_stats,
    }
    summary_output = Path(summary_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _evenly_spaced_by_group(
    frame: pl.DataFrame, per_group: int, *, group_column: str
) -> pl.DataFrame:
    assert per_group > 0
    sampled: list[pl.DataFrame] = []
    for value in sorted(frame[group_column].unique().to_list()):
        group = frame.filter(pl.col(group_column) == value).sort(
            "chrom", "start", "name"
        )
        if group.height > per_group:
            indices = np.linspace(0, group.height - 1, per_group, dtype=int).tolist()
            group = (
                group.with_row_index("_sample_row")
                .filter(pl.col("_sample_row").is_in(indices))
                .drop("_sample_row")
            )
        sampled.append(group)
    return pl.concat(sampled, how="vertical").sort("chrom", "start", "name")


def assign_gpn_six_arms(labels: pl.DataFrame) -> pl.DataFrame:
    """Apply the exhaustive #232 + #326-Arm-A assignment decision.

    The four established #232 v4 labels are retained directly.  A
    ``ccre_non_promoter`` window enters the enhancer arm only when all four
    non-cCRE functional fractions are exactly zero, matching #326 Arm A.
    Every other GPN-eligible window enters the complement background arm.
    """
    required = {
        "name",
        "chrom",
        "start",
        "end",
        "label",
        *_NON_CCRE_FUNCTIONAL_FRACS,
    }
    missing = required - set(labels.columns)
    assert not missing, f"GPN assignment labels missing columns: {sorted(missing)}"
    assert labels["name"].n_unique() == labels.height

    is_arm_a = pl.col("label") == "ccre_non_promoter"
    for fraction in _NON_CCRE_FUNCTIONAL_FRACS:
        is_arm_a &= pl.col(fraction) == 0.0
    assigned = labels.with_columns(
        pl.when(pl.col("label").is_in(_DIRECT_V4_ARMS))
        .then(pl.col("label"))
        .when(is_arm_a)
        .then(pl.lit("enhancer"))
        .otherwise(pl.lit("background"))
        .alias("arm"),
        pl.when(pl.col("label").is_in(_DIRECT_V4_ARMS))
        .then(pl.lit("exp232_v4_direct"))
        .when(is_arm_a)
        .then(pl.lit("exp326_arm_a_zero_other_functional"))
        .when(pl.col("label") == "background")
        .then(pl.lit("v4_background_to_remainder"))
        .otherwise(pl.lit("ccre_rejected_by_arm_a_to_remainder"))
        .alias("assignment_reason"),
        pl.lit(GPN_ASSIGNMENT_RECIPE).alias("assignment_recipe"),
    )
    assert assigned.height == labels.height
    assert set(assigned["arm"].unique().to_list()) <= set(GPN_ARMS)
    assert assigned.select("name", "arm").is_unique().all()
    assert (
        assigned.filter(pl.col("arm") == "enhancer")["label"]
        == "ccre_non_promoter"
    ).all()
    for fraction in _NON_CCRE_FUNCTIONAL_FRACS:
        assert (
            assigned.filter(pl.col("arm") == "enhancer")[fraction] == 0.0
        ).all()
    return assigned


def write_gpn_anchor_catalog(
    labels_path: str | Path,
    selected_path: str | Path,
    catalog_path: str | Path,
    assignments_path: str | Path,
    summary_path: str | Path,
    *,
    score_set: str,
    dataset_revision: str,
    min_selected_bases: int,
    expected_full_count: int | None = None,
    smoke_anchors_per_region: int | None = None,
    required_arms: list[str] | None = None,
) -> dict[str, object]:
    """Write the GPN projection catalog and versioned exhaustive assignments."""
    labels = pl.read_parquet(labels_path)
    selected = pl.read_parquet(selected_path).with_columns(
        pl.col("chrom").str.strip_prefix("chr")
    )
    score_columns = [
        "name",
        "chrom",
        "start",
        "end",
        "gpn_selected_bases",
        "proportion_gpn_selected",
        "gpn_entropy_cutoff",
    ]
    joined = labels.join(
        selected.select(score_columns),
        on=["name", "chrom", "start", "end"],
        how="inner",
        validate="1:1",
    )
    assert joined.height == labels.height == selected.height
    joined = assign_gpn_six_arms(joined)
    pre_cap_count = joined.height
    if smoke_anchors_per_region is not None:
        joined = _evenly_spaced_by_group(
            joined, smoke_anchors_per_region, group_column="arm"
        )
    elif expected_full_count is not None:
        assert joined.height == expected_full_count

    required = set(required_arms or [])
    assert required <= set(joined["arm"].unique().to_list())
    catalog = (
        joined.rename(
            {
                "name": "query_name",
                "start": "source_start",
                "end": "source_end",
                "label": "v4_region_label",
                "arm": "region_label",
            }
        )
        .with_columns(
            (pl.lit("chr") + pl.col("chrom")).alias("source_chrom"),
            pl.lit(score_set).alias("gpn_score_set"),
            pl.lit(dataset_revision).alias("gpn_dataset_revision"),
            pl.lit(min_selected_bases).alias("gpn_min_selected_bases"),
        )
        .drop("chrom")
        .select(
            "query_name",
            "source_chrom",
            "source_start",
            "source_end",
            "region_label",
            pl.exclude(
                "query_name",
                "source_chrom",
                "source_start",
                "source_end",
                "region_label",
            ),
        )
        .sort("source_chrom", "source_start", "query_name")
    )
    assert catalog["query_name"].n_unique() == catalog.height
    assert (catalog["source_end"] - catalog["source_start"] == 255).all()
    assert (catalog["gpn_selected_bases"] >= min_selected_bases).all()
    entropy_cutoffs = catalog["gpn_entropy_cutoff"].unique().to_list()
    assert len(entropy_cutoffs) == 1
    output = Path(catalog_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_parquet(output)

    assignments = catalog.select(
        "query_name",
        "source_chrom",
        "source_start",
        "source_end",
        pl.col("region_label").alias("arm"),
        "assignment_recipe",
        "assignment_reason",
        "v4_region_label",
        "functional_frac",
        *_NON_CCRE_FUNCTIONAL_FRACS,
        "ccre_non_promoter_frac",
        "gpn_selected_bases",
        "proportion_gpn_selected",
        "gpn_entropy_cutoff",
        "gpn_score_set",
        "gpn_dataset_revision",
        "gpn_min_selected_bases",
    )
    assert assignments.height == catalog.height
    assert assignments.select("assignment_recipe", "query_name").is_unique().all()
    assignments_output = Path(assignments_path)
    assignments_output.parent.mkdir(parents=True, exist_ok=True)
    assignments.write_parquet(assignments_output)

    summary: dict[str, object] = {
        "catalog_rows": catalog.height,
        "pre_smoke_cap_rows": pre_cap_count,
        "score_set": score_set,
        "dataset_revision": dataset_revision,
        "minimum_selected_bases": min_selected_bases,
        "smoke_anchors_per_region": smoke_anchors_per_region,
        "by_chrom": catalog.group_by("source_chrom").len().sort("source_chrom").to_dicts(),
        "assignment_recipe": GPN_ASSIGNMENT_RECIPE,
        "assignment_universe": (
            f"windows with >={min_selected_bases} of 255 positions satisfying "
            f"entropy_calibrated < {entropy_cutoffs[0]}"
        ),
        "assignment_is_exhaustive": True,
        "assignment_arm_count_sum": catalog.height,
        "by_arm": catalog.group_by("region_label").len().sort("region_label").to_dicts(),
        "background_by_assignment_reason": (
            catalog.filter(pl.col("region_label") == "background")
            .group_by("assignment_reason")
            .len()
            .sort("assignment_reason")
            .to_dicts()
        ),
    }
    summary_output = Path(summary_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def manifest_records(path: str | Path) -> list[dict[str, object]]:
    """Return manifest rows for concise audit/debug output."""
    return [asdict(shard) for shard in read_gpn_entropy_manifest(path).values()]
