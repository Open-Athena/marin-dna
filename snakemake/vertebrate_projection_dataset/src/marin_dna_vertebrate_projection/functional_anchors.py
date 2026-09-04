"""Annotation-first human anchor construction for issue #517.

All coordinates accepted and emitted by this module are 0-based and half-open.
The Ensembl GTF boundary conversion belongs in ``load_annotation`` before these
helpers are called.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
import polars as pl
import polars_bio as pb

from marin_dna.data.intervals import GenomicSet

FUNCTIONAL_ARMS = ("cds", "utr3", "tss_region", "ncrna", "enhancer")
DEFAULT_PRIORITY = FUNCTIONAL_ARMS
DEFAULT_NCRNA_BIOTYPES = (
    "lncRNA",
    "miRNA",
    "snoRNA",
    "snRNA",
    "ribozyme",
    "scaRNA",
    "vault_RNA",
)
ENHANCER_CLASSES = ("dELS", "pELS")

_FEATURE_SCHEMA = {
    "chrom": pl.String,
    "start": pl.Int64,
    "end": pl.Int64,
    "strand": pl.String,
    "source_id": pl.String,
    "source_feature": pl.String,
}
_WINDOW_SCHEMA = {
    "source_arm": pl.String,
    "chrom": pl.String,
    "start": pl.Int64,
    "end": pl.Int64,
}
_PROVENANCE_SCHEMA = {
    "source_arm": pl.String,
    "chrom": pl.String,
    "start": pl.Int64,
    "end": pl.Int64,
    "source_id": pl.String,
    "source_feature": pl.String,
}


@dataclass(frozen=True)
class FunctionalFeatureSets:
    """Raw source features, merged raw class cores, and the all-exon mask."""

    features: dict[str, pl.DataFrame]
    raw_cores: dict[str, GenomicSet]
    all_exons: GenomicSet


@dataclass(frozen=True)
class FunctionalOwnership:
    """Raw class unions and their pairwise-disjoint priority-owned cores."""

    raw_cores: dict[str, GenomicSet]
    owned_cores: dict[str, GenomicSet]
    priority: tuple[str, ...]


@dataclass(frozen=True)
class CandidateWindows:
    """Construction-valid candidates, source provenance, and early drops."""

    windows: pl.DataFrame
    provenance: pl.DataFrame
    construction_drops: pl.DataFrame


@dataclass(frozen=True)
class OwnershipGateResult:
    """All ownership annotations plus retained and rejected candidates."""

    audit: pl.DataFrame
    retained: pl.DataFrame
    dropped: pl.DataFrame


@dataclass(frozen=True)
class ConservationCatalogs:
    """Nested projection, training, and deferred human anchor catalogs."""

    projection: pl.DataFrame
    training: pl.DataFrame
    deferred: pl.DataFrame


def _empty_features() -> pl.DataFrame:
    return pl.DataFrame(schema=_FEATURE_SCHEMA)


def _empty_windows() -> pl.DataFrame:
    return pl.DataFrame(schema=_WINDOW_SCHEMA)


def _empty_provenance() -> pl.DataFrame:
    return pl.DataFrame(schema=_PROVENANCE_SCHEMA)


def _normalize_chrom_expr(column: str = "chrom") -> pl.Expr:
    return pl.col(column).cast(pl.String).str.strip_prefix("chr")


def _with_gtf_attributes(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("attribute")
        .str.extract(r'transcript_id "([^"]+)"')
        .alias("transcript_id"),
        pl.col("attribute").str.extract(r'exon_id "([^"]+)"').alias("exon_id"),
        pl.col("attribute").str.extract(r'exon_number "([^"]+)"').alias("exon_number"),
        pl.col("attribute")
        .str.extract(r'transcript_biotype "([^"]+)"')
        .alias("transcript_biotype"),
        pl.col("attribute")
        .str.extract(r'gene_biotype "([^"]+)"')
        .alias("gene_biotype"),
        pl.col("attribute").str.contains(r'pseudo "true"').alias("is_pseudo"),
        pl.col("attribute").str.contains(r'partial "true"').alias("is_partial"),
    )


def _stable_source_id(prefix: str) -> pl.Expr:
    return pl.concat_str(
        [
            pl.lit(prefix),
            pl.coalesce(pl.col("transcript_id"), pl.lit("no_transcript")),
            pl.coalesce(pl.col("exon_id"), pl.lit("no_exon")),
            pl.col("chrom"),
            pl.col("start").cast(pl.String),
            pl.col("end").cast(pl.String),
        ],
        separator=":",
    )


def _select_feature_columns(frame: pl.DataFrame, source_feature: str) -> pl.DataFrame:
    if frame.is_empty():
        return _empty_features()
    return (
        frame.with_columns(
            _stable_source_id(source_feature).alias("source_id"),
            pl.lit(source_feature).alias("source_feature"),
        )
        .select(*_FEATURE_SCHEMA)
        .cast(_FEATURE_SCHEMA)
        .sort("chrom", "start", "end", "source_id")
    )


def _utr_features(annotation: pl.DataFrame, *, side: str) -> pl.DataFrame:
    """Derive transcript-aware protein-coding UTR fragments from exon/CDS bounds."""
    if side not in {"utr5", "utr3"}:
        raise ValueError(f"unknown UTR side: {side!r}")
    exons = annotation.filter(
        (pl.col("feature") == "exon")
        & (pl.col("transcript_biotype") == "protein_coding")
        & pl.col("transcript_id").is_not_null()
    )
    cds = annotation.filter(
        (pl.col("feature") == "CDS")
        & (pl.col("transcript_biotype") == "protein_coding")
        & pl.col("transcript_id").is_not_null()
    )
    if exons.is_empty() or cds.is_empty():
        return _empty_features()
    bounds = cds.group_by("transcript_id", "chrom", "strand").agg(
        pl.col("start").min().alias("cds_start"),
        pl.col("end").max().alias("cds_end"),
    )
    joined = exons.join(
        bounds,
        on=["transcript_id", "chrom", "strand"],
        how="inner",
    )
    if side == "utr5":
        joined = joined.with_columns(
            pl.when(pl.col("strand") == "+")
            .then(pl.col("start"))
            .otherwise(pl.max_horizontal("start", "cds_end"))
            .alias("utr_start"),
            pl.when(pl.col("strand") == "+")
            .then(pl.min_horizontal("end", "cds_start"))
            .otherwise(pl.col("end"))
            .alias("utr_end"),
        )
    else:
        joined = joined.with_columns(
            pl.when(pl.col("strand") == "+")
            .then(pl.max_horizontal("start", "cds_end"))
            .otherwise(pl.col("start"))
            .alias("utr_start"),
            pl.when(pl.col("strand") == "+")
            .then(pl.col("end"))
            .otherwise(pl.min_horizontal("end", "cds_start"))
            .alias("utr_end"),
        )
    feature = joined.filter(pl.col("utr_end") > pl.col("utr_start")).select(
        "chrom",
        pl.col("utr_start").alias("start"),
        pl.col("utr_end").alias("end"),
        "strand",
        "transcript_id",
        "exon_id",
        "exon_number",
    )
    return _select_feature_columns(feature, side)


def _subtract_feature_rows(
    features: pl.DataFrame,
    mask: GenomicSet,
) -> pl.DataFrame:
    """Subtract a merged interval mask while retaining source provenance."""
    if features.is_empty() or mask.n_intervals() == 0:
        return features.clone()
    masks_by_chrom: dict[str, tuple[list[int], list[int]]] = {}
    for chrom, group in mask.to_polars().partition_by("chrom", as_dict=True).items():
        chrom_name = str(chrom[0] if isinstance(chrom, tuple) else chrom)
        ordered = group.sort("start", "end")
        masks_by_chrom[chrom_name] = (
            ordered["start"].to_list(),
            ordered["end"].to_list(),
        )

    output: list[dict[str, object]] = []
    for row in features.iter_rows(named=True):
        starts, ends = masks_by_chrom.get(str(row["chrom"]), ([], []))
        feature_start = int(row["start"])
        feature_end = int(row["end"])
        index = bisect_right(ends, feature_start)
        cursor = feature_start
        while index < len(starts) and starts[index] < feature_end:
            if starts[index] > cursor:
                fragment = dict(row)
                fragment["start"] = cursor
                fragment["end"] = min(starts[index], feature_end)
                if int(fragment["start"]) < int(fragment["end"]):
                    output.append(fragment)
            cursor = max(cursor, ends[index])
            if cursor >= feature_end:
                break
            index += 1
        if cursor < feature_end:
            fragment = dict(row)
            fragment["start"] = cursor
            fragment["end"] = feature_end
            output.append(fragment)
    if not output:
        return _empty_features()
    return pl.DataFrame(output, schema=_FEATURE_SCHEMA).sort(
        "chrom", "start", "end", "source_id"
    )


def extract_functional_features(
    annotation: pl.DataFrame,
    ccre: pl.DataFrame,
    *,
    standard_chroms: list[str] | tuple[str, ...],
    ncrna_biotypes: list[str] | tuple[str, ...] = DEFAULT_NCRNA_BIOTYPES,
    tss_radius: int = 256,
) -> FunctionalFeatureSets:
    """Extract the five #517 raw human feature sets with stable provenance.

    ``annotation`` must already use 0-based, half-open coordinates.
    ``ccre`` must contain ``chrom, start, end, cre_class`` and may contain a
    stable ``ccre_id`` column.
    """
    if tss_radius <= 0:
        raise ValueError(f"tss_radius must be positive, got {tss_radius}")
    if not ncrna_biotypes:
        raise ValueError("ncrna_biotypes must be non-empty")
    required_annotation = {
        "chrom",
        "feature",
        "start",
        "end",
        "strand",
        "attribute",
    }
    missing_annotation = required_annotation - set(annotation.columns)
    if missing_annotation:
        raise ValueError(f"annotation missing columns: {sorted(missing_annotation)}")
    required_ccre = {"chrom", "start", "end", "cre_class"}
    missing_ccre = required_ccre - set(ccre.columns)
    if missing_ccre:
        raise ValueError(f"cCRE table missing columns: {sorted(missing_ccre)}")

    chroms = [chrom.removeprefix("chr") for chrom in standard_chroms]
    ann = (
        annotation.with_columns(_normalize_chrom_expr())
        .filter(pl.col("chrom").is_in(chroms))
        .pipe(_with_gtf_attributes)
    )
    if ann.select((pl.col("start") < 0).any()).item():
        raise ValueError(
            "annotation contains negative coordinates after GTF conversion"
        )

    cds_rows = ann.filter(
        (pl.col("feature") == "CDS")
        & (pl.col("transcript_biotype") == "protein_coding")
    )
    cds = _select_feature_columns(cds_rows, "cds")
    cds_core = GenomicSet(cds.select("chrom", "start", "end"))

    utr3 = _subtract_feature_rows(_utr_features(ann, side="utr3"), cds_core)
    utr5 = _utr_features(ann, side="utr5")

    transcripts = ann.filter(
        (pl.col("feature") == "transcript")
        & (pl.col("transcript_biotype") == "protein_coding")
        & pl.col("transcript_id").is_not_null()
    ).unique(subset=["transcript_id", "chrom", "start", "end", "strand"])
    tss_rows = transcripts.with_columns(
        pl.when(pl.col("strand") == "+")
        .then(pl.col("start"))
        .otherwise(pl.col("end"))
        .alias("tss")
    ).with_columns(
        (pl.col("tss") - tss_radius).alias("start"),
        (pl.col("tss") + tss_radius).alias("end"),
        pl.lit(None, dtype=pl.String).alias("exon_id"),
        pl.lit(None, dtype=pl.String).alias("exon_number"),
    )
    tss_bands = _select_feature_columns(tss_rows, "tss_band")
    tss_region = pl.concat([tss_bands, utr5], how="vertical").sort(
        "chrom", "start", "end", "source_id"
    )

    ncrna_rows = (
        ann.filter(pl.col("feature") == "exon")
        .filter(pl.col("transcript_biotype").is_in(list(ncrna_biotypes)))
        .filter(~pl.col("is_pseudo").fill_null(False))
        .filter(~pl.col("is_partial").fill_null(False))
        .filter(
            ~pl.col("transcript_biotype").fill_null("").str.contains("(?i)pseudogenic")
        )
        .filter(~pl.col("gene_biotype").fill_null("").str.contains("(?i)pseudogene"))
    )
    ncrna = _select_feature_columns(ncrna_rows, "ncrna_exon")

    ccre_frame = ccre.with_columns(_normalize_chrom_expr()).filter(
        pl.col("chrom").is_in(chroms)
        & pl.col("cre_class").is_in(list(ENHANCER_CLASSES))
    )
    if "ccre_id" not in ccre_frame.columns:
        ccre_frame = ccre_frame.with_columns(
            pl.concat_str(
                [
                    pl.lit("ccre"),
                    pl.col("chrom"),
                    pl.col("start").cast(pl.String),
                    pl.col("end").cast(pl.String),
                    pl.col("cre_class"),
                ],
                separator=":",
            ).alias("ccre_id")
        )
    enhancer = (
        ccre_frame.with_columns(
            pl.lit(".").alias("strand"),
            pl.concat_str([pl.lit("enhancer"), pl.col("ccre_id")], separator=":").alias(
                "source_id"
            ),
            pl.col("cre_class").alias("source_feature"),
        )
        .select(*_FEATURE_SCHEMA)
        .cast(_FEATURE_SCHEMA)
        .sort("chrom", "start", "end", "source_id")
    )

    features = {
        "cds": cds,
        "utr3": utr3,
        "tss_region": tss_region,
        "ncrna": ncrna,
        "enhancer": enhancer,
    }
    if any(frame.is_empty() for frame in features.values()):
        empty = [arm for arm, frame in features.items() if frame.is_empty()]
        raise ValueError(f"functional extraction produced empty arms: {empty}")
    raw_cores = {
        arm: GenomicSet(frame.select("chrom", "start", "end"))
        for arm, frame in features.items()
    }
    all_exons = GenomicSet(
        ann.filter(pl.col("feature") == "exon").select("chrom", "start", "end")
    )
    return FunctionalFeatureSets(
        features=features,
        raw_cores=raw_cores,
        all_exons=all_exons,
    )


def resolve_base_priority(
    raw_cores: dict[str, GenomicSet],
    *,
    priority: list[str] | tuple[str, ...] = DEFAULT_PRIORITY,
) -> FunctionalOwnership:
    """Assign each functional base to at most one arm in priority order."""
    priority_tuple = tuple(priority)
    if set(priority_tuple) != set(FUNCTIONAL_ARMS) or len(priority_tuple) != len(
        FUNCTIONAL_ARMS
    ):
        raise ValueError(
            f"priority must be a permutation of {FUNCTIONAL_ARMS}, got {priority_tuple}"
        )
    if set(raw_cores) != set(FUNCTIONAL_ARMS):
        raise ValueError(
            f"raw_cores must contain exactly {FUNCTIONAL_ARMS}, got {tuple(raw_cores)}"
        )
    owned: dict[str, GenomicSet] = {}
    higher: GenomicSet | None = None
    for arm in priority_tuple:
        owned[arm] = raw_cores[arm] if higher is None else raw_cores[arm] - higher
        higher = raw_cores[arm] if higher is None else higher | raw_cores[arm]
    for index, arm in enumerate(priority_tuple):
        for other in priority_tuple[index + 1 :]:
            if (owned[arm] & owned[other]).total_size() != 0:
                raise AssertionError(f"priority-owned cores overlap: {arm}, {other}")
    return FunctionalOwnership(
        raw_cores=dict(raw_cores),
        owned_cores=owned,
        priority=priority_tuple,
    )


def feature_audit_table(
    feature_sets: FunctionalFeatureSets,
    ownership: FunctionalOwnership,
) -> pl.DataFrame:
    """Summarize raw feature, merged interval, and priority-owned base counts."""
    rows = []
    for arm in ownership.priority:
        rows.append(
            {
                "arm": arm,
                "raw_feature_count": feature_sets.features[arm]["source_id"].n_unique(),
                "raw_feature_fragment_count": feature_sets.features[arm].height,
                "raw_merged_interval_count": feature_sets.raw_cores[arm].n_intervals(),
                "raw_union_bases": feature_sets.raw_cores[arm].total_size(),
                "priority_owned_interval_count": ownership.owned_cores[
                    arm
                ].n_intervals(),
                "priority_owned_bases": ownership.owned_cores[arm].total_size(),
            }
        )
    return pl.DataFrame(rows).sort("arm")


def pairwise_raw_overlap_table(raw_cores: dict[str, GenomicSet]) -> pl.DataFrame:
    """Return the complete ordered pairwise raw overlap matrix in long form."""
    rows = []
    for arm_a in FUNCTIONAL_ARMS:
        for arm_b in FUNCTIONAL_ARMS:
            rows.append(
                {
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "overlap_bases": (raw_cores[arm_a] & raw_cores[arm_b]).total_size(),
                }
            )
    return pl.DataFrame(rows).sort("arm_a", "arm_b")


def _resize_intervals(frame: pl.DataFrame, target_size: int) -> pl.DataFrame:
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    difference = pl.lit(target_size) - (pl.col("end") - pl.col("start"))
    left = difference // 2
    right = difference - left
    return frame.with_columns(
        (pl.col("start") - left).alias("start"),
        (pl.col("end") + right).alias("end"),
    )


def _expand_short_intervals(frame: pl.DataFrame, min_size: int) -> pl.DataFrame:
    size = pl.col("end") - pl.col("start")
    difference = (pl.lit(min_size) - size).clip(lower_bound=0)
    left = difference // 2
    right = difference - left
    return frame.with_columns(
        (pl.col("start") - left).alias("start"),
        (pl.col("end") + right).alias("end"),
    )


def tile_intervals(
    intervals: pl.DataFrame,
    *,
    source_arm: str,
    window_size: int = 255,
    step_size: int = 128,
) -> pl.DataFrame:
    """Tile merged intervals from their starts and omit incomplete terminals."""
    if source_arm not in FUNCTIONAL_ARMS:
        raise ValueError(f"unknown source arm: {source_arm!r}")
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive")
    rows: list[dict[str, object]] = []
    for row in intervals.sort("chrom", "start", "end").iter_rows(named=True):
        start = int(row["start"])
        end = int(row["end"])
        for window_start in range(start, end - window_size + 1, step_size):
            rows.append(
                {
                    "source_arm": source_arm,
                    "chrom": str(row["chrom"]),
                    "start": window_start,
                    "end": window_start + window_size,
                }
            )
    if not rows:
        return _empty_windows()
    return (
        pl.DataFrame(rows, schema=_WINDOW_SCHEMA)
        .unique()
        .sort("source_arm", "chrom", "start", "end")
    )


def _coverage_bp(windows: pl.DataFrame, regions: GenomicSet) -> np.ndarray:
    if windows.is_empty() or regions.n_intervals() == 0:
        return np.zeros(windows.height, dtype=np.int64)
    coords = windows.select("chrom", "start", "end")
    coords.config_meta.set(coordinate_system_zero_based=True)
    region_frame = regions.to_polars()
    region_frame.config_meta.set(coordinate_system_zero_based=True)
    covered = pb.coverage(coords, region_frame, output_type="polars.DataFrame")
    if not isinstance(covered, pl.DataFrame):
        raise TypeError(f"polars-bio coverage returned {type(covered).__name__}")
    if not covered.select("chrom", "start", "end").equals(coords):
        raise AssertionError("polars-bio coverage changed window order or coordinates")
    return covered["coverage"].to_numpy().astype(np.int64, copy=False)


def _window_provenance(
    windows: pl.DataFrame,
    features: pl.DataFrame,
) -> pl.DataFrame:
    """Map windows to overlapping same-arm features with a chromosome sweep."""
    if windows.is_empty() or features.is_empty():
        return _empty_provenance()
    features_by_chrom: dict[str, list[dict[str, object]]] = {}
    for row in features.sort("chrom", "start", "end", "source_id").iter_rows(
        named=True
    ):
        features_by_chrom.setdefault(str(row["chrom"]), []).append(row)
    output: list[dict[str, object]] = []
    for chrom, group in windows.partition_by("chrom", as_dict=True).items():
        chrom_name = str(chrom[0] if isinstance(chrom, tuple) else chrom)
        feature_rows = features_by_chrom.get(chrom_name, [])
        next_feature = 0
        active: list[dict[str, object]] = []
        for window in group.sort("start", "end").iter_rows(named=True):
            window_start = int(window["start"])
            window_end = int(window["end"])
            active = [row for row in active if int(row["end"]) > window_start]
            while (
                next_feature < len(feature_rows)
                and int(feature_rows[next_feature]["start"]) < window_end
            ):
                candidate = feature_rows[next_feature]
                if int(candidate["end"]) > window_start:
                    active.append(candidate)
                next_feature += 1
            for feature in active:
                if (
                    int(feature["start"]) < window_end
                    and int(feature["end"]) > window_start
                ):
                    output.append(
                        {
                            "source_arm": str(window["source_arm"]),
                            "chrom": chrom_name,
                            "start": window_start,
                            "end": window_end,
                            "source_id": str(feature["source_id"]),
                            "source_feature": str(feature["source_feature"]),
                        }
                    )
    if not output:
        return _empty_provenance()
    return (
        pl.DataFrame(output, schema=_PROVENANCE_SCHEMA)
        .unique()
        .sort("source_arm", "chrom", "start", "end", "source_id")
    )


def _filter_construction_valid(
    windows: pl.DataFrame,
    *,
    chrom_sizes: pl.DataFrame,
    defined: GenomicSet,
    all_exons: GenomicSet,
    reject_exons: bool,
    window_size: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if windows.is_empty():
        drops_schema = {**_WINDOW_SCHEMA, "drop_reason": pl.String}
        return windows, pl.DataFrame(schema=drops_schema)
    sizes = chrom_sizes.select(
        _normalize_chrom_expr(), pl.col("size").cast(pl.Int64)
    ).unique(subset="chrom")
    annotated = windows.join(sizes, on="chrom", how="left").with_columns(
        pl.Series("defined_bases", _coverage_bp(windows, defined), dtype=pl.Int64),
        pl.Series("exon_bases", _coverage_bp(windows, all_exons), dtype=pl.Int64),
    )
    annotated = annotated.with_columns(
        pl.when(pl.col("size").is_null())
        .then(pl.lit("missing_chrom_size"))
        .when((pl.col("start") < 0) | (pl.col("end") > pl.col("size")))
        .then(pl.lit("chromosome_bounds"))
        .when(pl.col("end") - pl.col("start") != window_size)
        .then(pl.lit("incomplete_window"))
        .when(pl.col("defined_bases") != window_size)
        .then(pl.lit("undefined_sequence"))
        .when(pl.lit(reject_exons) & (pl.col("exon_bases") > 0))
        .then(pl.lit("annotated_exon_overlap"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("drop_reason")
    )
    valid = annotated.filter(pl.col("drop_reason").is_null()).select(*_WINDOW_SCHEMA)
    drops = annotated.filter(pl.col("drop_reason").is_not_null()).select(
        *_WINDOW_SCHEMA, "drop_reason"
    )
    return valid, drops


def build_candidate_windows(
    feature_sets: FunctionalFeatureSets,
    ownership: FunctionalOwnership,
    *,
    chrom_sizes: pl.DataFrame,
    defined: GenomicSet,
    window_size: int = 255,
    step_size: int = 128,
    feature_flank: int = 20,
    min_feature_size: int = 20,
    max_feature_size: int = 10_000,
) -> CandidateWindows:
    """Build annotation-first candidates before the window ownership gate."""
    if feature_flank < 0:
        raise ValueError(f"feature_flank must be non-negative, got {feature_flank}")
    if min_feature_size <= 0 or max_feature_size < min_feature_size:
        raise ValueError("invalid feature size bounds")
    windows: list[pl.DataFrame] = []
    provenance: list[pl.DataFrame] = []
    drops: list[pl.DataFrame] = []
    for arm in ownership.priority:
        if arm == "enhancer":
            centered_features = _resize_intervals(
                feature_sets.features[arm], window_size
            )
            arm_windows = centered_features.select(
                pl.lit(arm).alias("source_arm"), "chrom", "start", "end"
            ).cast(_WINDOW_SCHEMA)
            arm_provenance = centered_features.select(
                pl.lit(arm).alias("source_arm"),
                "chrom",
                "start",
                "end",
                "source_id",
                "source_feature",
            ).cast(_PROVENANCE_SCHEMA)
            arm_windows, arm_drops = _filter_construction_valid(
                arm_windows,
                chrom_sizes=chrom_sizes,
                defined=defined,
                all_exons=feature_sets.all_exons,
                reject_exons=True,
                window_size=window_size,
            )
            arm_provenance = arm_provenance.join(
                arm_windows,
                on=["source_arm", "chrom", "start", "end"],
                how="semi",
            )
        else:
            intervals = ownership.owned_cores[arm].to_polars()
            if arm in {"cds", "utr3", "ncrna"}:
                size = pl.col("end") - pl.col("start")
                intervals = intervals.filter(
                    size.is_between(min_feature_size, max_feature_size, closed="both")
                ).with_columns(
                    (pl.col("start") - feature_flank).alias("start"),
                    (pl.col("end") + feature_flank).alias("end"),
                )
            intervals = _expand_short_intervals(intervals, window_size)
            intervals = GenomicSet(intervals).to_polars()
            arm_windows = tile_intervals(
                intervals,
                source_arm=arm,
                window_size=window_size,
                step_size=step_size,
            )
            arm_windows, arm_drops = _filter_construction_valid(
                arm_windows,
                chrom_sizes=chrom_sizes,
                defined=defined,
                all_exons=feature_sets.all_exons,
                reject_exons=False,
                window_size=window_size,
            )
            arm_provenance = _window_provenance(arm_windows, feature_sets.features[arm])
        windows.append(arm_windows)
        provenance.append(arm_provenance)
        drops.append(arm_drops)

    all_windows = (
        pl.concat(windows, how="vertical")
        .unique()
        .sort("source_arm", "chrom", "start", "end")
    )
    all_provenance = (
        pl.concat(provenance, how="vertical")
        .unique()
        .sort("source_arm", "chrom", "start", "end", "source_id")
    )
    counts = all_provenance.group_by("source_arm", "chrom", "start", "end").agg(
        pl.col("source_id").n_unique().alias("contributing_feature_count"),
        pl.col("source_feature").unique().sort().alias("source_feature_types"),
    )
    all_windows = all_windows.join(
        counts,
        on=["source_arm", "chrom", "start", "end"],
        how="left",
    ).with_columns(
        pl.col("contributing_feature_count").fill_null(0).cast(pl.Int64),
        pl.col("source_feature_types").list.join("|").alias("source_feature_types"),
    )
    if (all_windows["contributing_feature_count"] == 0).any():
        raise AssertionError("candidate window lacks source-feature provenance")
    construction_drops = pl.concat(drops, how="vertical").sort(
        "source_arm", "chrom", "start", "end"
    )
    return CandidateWindows(
        windows=all_windows,
        provenance=all_provenance,
        construction_drops=construction_drops,
    )


def apply_window_ownership_gate(
    candidates: pl.DataFrame,
    feature_sets: FunctionalFeatureSets,
    ownership: FunctionalOwnership,
) -> OwnershipGateResult:
    """Annotate raw/owned coverage and retain only source-arm winners."""
    required = {
        "source_arm",
        "chrom",
        "start",
        "end",
        "contributing_feature_count",
        "source_feature_types",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate windows missing columns: {sorted(missing)}")
    if candidates.is_empty():
        raise ValueError("candidate windows must be non-empty")
    widths = (candidates["end"] - candidates["start"]).to_numpy()
    if not np.all(widths == widths[0]):
        raise AssertionError("candidate windows have inconsistent lengths")

    annotated = candidates.clone()
    raw_bp: dict[str, np.ndarray] = {}
    owned_bp: dict[str, np.ndarray] = {}
    coverage_columns: list[pl.Series] = []
    for arm in ownership.priority:
        raw_bp[arm] = _coverage_bp(candidates, feature_sets.raw_cores[arm])
        owned_bp[arm] = _coverage_bp(candidates, ownership.owned_cores[arm])
        coverage_columns.extend(
            [
                pl.Series(f"raw_{arm}_bases", raw_bp[arm], dtype=pl.Int64),
                pl.Series(
                    f"raw_{arm}_fraction", raw_bp[arm] / widths, dtype=pl.Float64
                ),
                pl.Series(f"owned_{arm}_bases", owned_bp[arm], dtype=pl.Int64),
                pl.Series(
                    f"owned_{arm}_fraction", owned_bp[arm] / widths, dtype=pl.Float64
                ),
            ]
        )
    annotated = annotated.with_columns(coverage_columns)
    matrix = np.column_stack([owned_bp[arm] for arm in ownership.priority])
    winner_index = np.argmax(matrix, axis=1)
    winners = np.asarray(ownership.priority, dtype=object)[winner_index]
    source_index = np.asarray(
        [ownership.priority.index(arm) for arm in candidates["source_arm"]],
        dtype=np.int64,
    )
    source_owned = matrix[np.arange(candidates.height), source_index]
    functional_union = matrix.sum(axis=1)
    exon_bp = _coverage_bp(candidates, feature_sets.all_exons)
    passes = source_index == winner_index
    query_names = [
        f"fa1_{arm}_{chrom}_{int(start):012d}_{int(end):012d}"
        for arm, chrom, start, end in candidates.select(
            "source_arm", "chrom", "start", "end"
        ).iter_rows()
    ]
    annotated = annotated.with_columns(
        pl.Series("query_name", query_names, dtype=pl.String),
        pl.Series("ownership_winner", winners.astype(str), dtype=pl.String),
        pl.Series("source_arm_owned_bases", source_owned, dtype=pl.Int64),
        pl.Series("source_arm_owned_fraction", source_owned / widths, dtype=pl.Float64),
        pl.Series("union_functional_bases", functional_union, dtype=pl.Int64),
        pl.Series(
            "union_functional_fraction", functional_union / widths, dtype=pl.Float64
        ),
        pl.Series("exon_bases", exon_bp, dtype=pl.Int64),
        pl.Series("exon_fraction", exon_bp / widths, dtype=pl.Float64),
        pl.Series("passes_ownership_gate", passes, dtype=pl.Boolean),
    )
    if annotated["query_name"].n_unique() != annotated.height:
        raise AssertionError("functional anchor IDs are not unique")
    retained = annotated.filter(pl.col("passes_ownership_gate")).sort(
        "source_arm", "chrom", "start", "end"
    )
    dropped = annotated.filter(~pl.col("passes_ownership_gate")).sort(
        "source_arm", "chrom", "start", "end"
    )
    if not (retained["source_arm"] == retained["ownership_winner"]).all():
        raise AssertionError("retained source arm did not win ownership")
    return OwnershipGateResult(audit=annotated, retained=retained, dropped=dropped)


def annotate_sequence_fractions(
    anchors: pl.DataFrame,
    sequences: pl.DataFrame,
    *,
    sequence_column: str = "sequence",
) -> pl.DataFrame:
    """Attach GC, repeat-mask, and ambiguous-base fractions to human anchors."""
    required = {"query_name", sequence_column}
    missing = required - set(sequences.columns)
    if missing:
        raise ValueError(f"sequence table missing columns: {sorted(missing)}")
    if sequences["query_name"].n_unique() != sequences.height:
        raise AssertionError("sequence table has duplicate query_name rows")

    def fractions(sequence: str) -> dict[str, float]:
        if not sequence:
            raise ValueError("anchor sequence must be non-empty")
        length = len(sequence)
        uppercase = sequence.upper()
        return {
            "gc_fraction": sum(base in {"G", "C"} for base in uppercase) / length,
            "repeat_masked_fraction": sum(base.islower() for base in sequence) / length,
            "ambiguous_base_fraction": sum(
                base not in {"A", "C", "G", "T"} for base in uppercase
            )
            / length,
        }

    qc_rows = []
    for query_name, sequence in sequences.select(
        "query_name", sequence_column
    ).iter_rows():
        qc_rows.append({"query_name": query_name, **fractions(sequence)})
    qc = pl.DataFrame(
        qc_rows,
        schema={
            "query_name": pl.String,
            "gc_fraction": pl.Float64,
            "repeat_masked_fraction": pl.Float64,
            "ambiguous_base_fraction": pl.Float64,
        },
    )
    result = anchors.join(qc, on="query_name", how="left")
    if result["gc_fraction"].null_count() > 0:
        raise AssertionError("one or more anchors lack sequence QC")
    return result


def split_conservation_catalogs(
    scored_anchors: pl.DataFrame,
    *,
    projection_min: float = 0.10,
    training_min: float = 0.20,
) -> ConservationCatalogs:
    """Split scored human anchors into nested projection/training catalogs."""
    if not 0.0 <= projection_min < training_min <= 1.0:
        raise ValueError(
            f"expected 0 <= projection_min < training_min <= 1, got "
            f"{projection_min}, {training_min}"
        )
    if "proportion_conserved" not in scored_anchors.columns:
        raise ValueError("scored anchors lack proportion_conserved")
    projection = scored_anchors.filter(
        pl.col("proportion_conserved") >= projection_min
    ).sort("source_arm", "chrom", "start", "end")
    training = projection.filter(pl.col("proportion_conserved") >= training_min).sort(
        "source_arm", "chrom", "start", "end"
    )
    deferred = projection.filter(pl.col("proportion_conserved") < training_min).sort(
        "source_arm", "chrom", "start", "end"
    )
    if set(training["query_name"]) - set(projection["query_name"]):
        raise AssertionError("training catalog is not a projection-catalog subset")
    if training.height + deferred.height != projection.height:
        raise AssertionError(
            "training and deferred catalogs do not partition projection"
        )
    return ConservationCatalogs(
        projection=projection,
        training=training,
        deferred=deferred,
    )


def to_projection_catalog(anchors: pl.DataFrame) -> pl.DataFrame:
    """Add the shared projection identity columns without dropping audit metadata."""
    required = {"query_name", "source_arm", "chrom", "start", "end"}
    missing = required - set(anchors.columns)
    if missing:
        raise ValueError(f"functional anchors missing columns: {sorted(missing)}")
    result = anchors.with_columns(
        (pl.lit("chr") + pl.col("chrom")).alias("source_chrom"),
        pl.col("start").alias("source_start"),
        pl.col("end").alias("source_end"),
        pl.col("source_arm").alias("region_label"),
    )
    if result["query_name"].n_unique() != result.height:
        raise AssertionError("projection catalog has duplicate query_name values")
    if not (result["source_end"] - result["source_start"] == 255).all():
        raise AssertionError("projection anchors must be exactly 255 bp")
    return result.sort("source_chrom", "source_start", "query_name")
