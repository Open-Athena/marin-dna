"""Parse, validate, and combine MMseqs2 clustering outputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

import polars as pl

CLUSTER_COLUMNS = ["representative", "member"]
CLUSTER_SCHEMA = {"representative": pl.String, "member": pl.String}

ALIGNMENT_COLUMNS = [
    "query",
    "target",
    "fident",
    "alnlen",
    "qcov",
    "tcov",
    "qstart",
    "qend",
    "tstart",
    "tend",
    "evalue",
    "bits",
]
ALIGNMENT_SCHEMA = {
    "query": pl.String,
    "target": pl.String,
    "fident": pl.Float64,
    "alnlen": pl.Int64,
    "qcov": pl.Float64,
    "tcov": pl.Float64,
    "qstart": pl.Int64,
    "qend": pl.Int64,
    "tstart": pl.Int64,
    "tend": pl.Int64,
    "evalue": pl.Float64,
    "bits": pl.Float64,
}

INFERENCE_FEATURES = frozenset(
    {
        "cluster_member_count",
        "distinct_genome_count",
        "max_members_per_genome",
        "dominant_genome_fraction",
        "member_identity_to_representative",
        "member_query_coverage",
        "member_target_coverage",
        "member_bits",
        "member_evalue",
        "identity_mean",
        "identity_median",
        "identity_min",
        "identity_max",
        "identity_q10",
        "coverage_mean",
        "coverage_median",
        "coverage_min",
        "coverage_max",
        "coverage_q10",
        "bits_mean",
        "bits_median",
        "bits_min",
        "bits_max",
        "bits_q10",
        "singleton",
        "assignment_stability",
    }
)
FORBIDDEN_SCORE_INPUTS = frozenset(
    {
        "phylop_fraction",
        "repeat_fraction",
        "gc_content",
        "chromosome",
        "start",
        "end",
        "strand",
        "annotation",
        "cds",
        "ccre",
        "species_tree",
    }
)


def _read_headerless(
    path: str | Path,
    *,
    columns: list[str],
    schema: dict[str, pl.DataTypeClass],
) -> pl.DataFrame:
    path = Path(path)
    if path.stat().st_size == 0:
        return pl.DataFrame(schema=schema)
    return pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=columns,
        schema_overrides=schema,
    )


def parse_cluster_assignments(path: str | Path) -> pl.DataFrame:
    """Read `mmseqs createtsv` output and validate one row per member."""
    assignments = _read_headerless(
        path,
        columns=CLUSTER_COLUMNS,
        schema=CLUSTER_SCHEMA,
    )
    assert assignments.height > 0, "cluster assignment TSV is empty"
    assert assignments["representative"].str.len_chars().min() > 0
    assert assignments["member"].str.len_chars().min() > 0
    duplicated = assignments.group_by("member").len().filter(pl.col("len") != 1)
    assert duplicated.height == 0, "members must occur in exactly one cluster"
    representatives = set(assignments["representative"].to_list())
    members = set(assignments["member"].to_list())
    assert representatives.issubset(members), "every representative must be a member"
    self_representatives = set(
        assignments.filter(pl.col("representative") == pl.col("member"))[
            "representative"
        ].to_list()
    )
    missing_self_rows = representatives - self_representatives
    assert not missing_self_rows, (
        f"clusters lack one self row: {sorted(missing_self_rows)[:10]}"
    )
    return assignments


def _iter_assignment_rows(handle: TextIO) -> Iterable[tuple[str, str]]:
    for line_number, line in enumerate(handle, start=1):
        fields = line.rstrip("\n").split("\t")
        assert len(fields) == 2 and all(fields), (
            f"invalid cluster assignment at line {line_number}"
        )
        yield fields[0], fields[1]


def merge_cluster_assignments(
    *,
    assignment_paths: Iterable[str | Path],
    output_path: str | Path,
) -> dict[str, object]:
    """Union several complete partitions into deterministic components.

    Every representative-member relation is treated as an undirected edge.
    Input files are streamed, while the member universe and union-find state use
    memory linear in the number of sequences.
    """
    paths = [Path(path) for path in assignment_paths]
    assert len(paths) >= 2, "an ensemble requires at least two assignments"

    parent: dict[str, str] = {}
    member_order: list[str] = []

    def find(member: str) -> str:
        root = member
        while parent[root] != root:
            root = parent[root]
        while parent[member] != member:
            next_member = parent[member]
            parent[member] = root
            member = next_member
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        canonical, other = sorted((left_root, right_root))
        parent[other] = canonical

    input_receipts: list[dict[str, int]] = []
    universe: set[str] | None = None
    for index, path in enumerate(paths):
        seen_members: set[str] = set()
        representatives: set[str] = set()
        self_representatives: set[str] = set()
        with path.open() as handle:
            for representative, member in _iter_assignment_rows(handle):
                assert member not in seen_members, (
                    f"member {member!r} occurs more than once in {path}"
                )
                seen_members.add(member)
                representatives.add(representative)
                if representative == member:
                    self_representatives.add(representative)
                if index == 0:
                    parent[member] = member
                    member_order.append(member)
                else:
                    assert universe is not None
                    assert member in universe, f"unexpected member {member!r} in {path}"

        if index == 0:
            assert seen_members, "cluster assignment TSV is empty"
            universe = seen_members
        else:
            assert seen_members == universe, (
                f"assignment member universe differs in {path}"
            )
        assert representatives.issubset(seen_members), (
            f"every representative in {path} must be a member"
        )
        assert representatives == self_representatives, (
            f"every cluster in {path} must contain one representative self row"
        )

        with path.open() as handle:
            for representative, member in _iter_assignment_rows(handle):
                union(representative, member)
        input_receipts.append(
            {
                "cluster_count": len(representatives),
                "edge_count": len(seen_members) - len(representatives),
                "sequence_count": len(seen_members),
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    roots: set[str] = set()
    with output.open("w") as handle:
        for member in member_order:
            root = find(member)
            roots.add(root)
            handle.write(f"{root}\t{member}\n")

    return {
        "algorithm": "union_find_connected_components",
        "input_assignment_count": len(paths),
        "input_assignments": input_receipts,
        "output_cluster_count": len(roots),
        "sequence_count": len(member_order),
    }


def parse_alignments(path: str | Path) -> pl.DataFrame:
    """Read `convertalis` output and convert MMseqs2 coordinates at the boundary."""
    alignments = _read_headerless(
        path,
        columns=ALIGNMENT_COLUMNS,
        schema=ALIGNMENT_SCHEMA,
    )
    if alignments.height == 0:
        return alignments.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("query_start"),
            pl.lit(None, dtype=pl.Int64).alias("query_end"),
            pl.lit(None, dtype=pl.Int64).alias("target_start"),
            pl.lit(None, dtype=pl.Int64).alias("target_end"),
            pl.lit(None, dtype=pl.Boolean).alias("reverse_strand"),
        )
    assert alignments["alnlen"].min() > 0
    assert alignments["fident"].is_between(0.0, 1.0, closed="both").all()
    assert alignments["qcov"].is_between(0.0, 1.0, closed="both").all()
    assert alignments["tcov"].is_between(0.0, 1.0, closed="both").all()
    assert (alignments["evalue"] >= 0).all()
    assert (alignments["bits"] >= 0).all()
    assert (
        alignments.select("qstart", "qend", "tstart", "tend").min().min_horizontal()
        >= 1
    ).item()
    return alignments.with_columns(
        (pl.min_horizontal("qstart", "qend") - 1).alias("query_start"),
        pl.max_horizontal("qstart", "qend").alias("query_end"),
        (pl.min_horizontal("tstart", "tend") - 1).alias("target_start"),
        pl.max_horizontal("tstart", "tend").alias("target_end"),
        (
            (pl.col("qend") < pl.col("qstart")) | (pl.col("tend") < pl.col("tstart"))
        ).alias("reverse_strand"),
    )


def validate_alignment_coverage(
    assignments: pl.DataFrame,
    alignments: pl.DataFrame,
) -> None:
    """Require exactly one representative-member alignment for every assignment."""
    assert set(CLUSTER_COLUMNS).issubset(assignments.columns)
    assert {"query", "target"}.issubset(alignments.columns)
    alignment_pairs = alignments.select(
        pl.col("query").alias("representative"),
        pl.col("target").alias("member"),
    )
    duplicated = (
        alignment_pairs.group_by(CLUSTER_COLUMNS).len().filter(pl.col("len") != 1)
    )
    assert duplicated.height == 0, "alignment pairs must occur exactly once"
    missing = assignments.select(CLUSTER_COLUMNS).join(
        alignment_pairs,
        on=CLUSTER_COLUMNS,
        how="anti",
    )
    assert missing.height == 0, (
        "cluster assignments lack strand-aware alignments: "
        f"{missing.head(10).to_dicts()}"
    )
    unexpected = alignment_pairs.join(
        assignments.select(CLUSTER_COLUMNS),
        on=CLUSTER_COLUMNS,
        how="anti",
    )
    assert unexpected.height == 0, (
        "alignments contain pairs absent from cluster assignments: "
        f"{unexpected.head(10).to_dicts()}"
    )
    assert alignment_pairs.height == assignments.height


def filter_cluster_alignments(
    *,
    assignments_path: str | Path,
    alignments_paths: Iterable[str | Path],
    output_path: str | Path,
) -> pl.DataFrame:
    """Keep strand-aware search alignments belonging to Linclust cluster edges."""
    assignments = parse_cluster_assignments(assignments_path)
    alignment_frames = [parse_alignments(path) for path in alignments_paths]
    assert alignment_frames, "at least one alignment table is required"
    alignments = pl.concat(alignment_frames)
    alignments = alignments.sort(
        [
            "query",
            "target",
            "bits",
            "evalue",
            "fident",
            "alnlen",
            "qcov",
            "tcov",
            "reverse_strand",
        ],
        descending=[False, False, True, False, True, True, True, True, False],
    ).unique(subset=["query", "target"], keep="first", maintain_order=True)
    expected = assignments.select(CLUSTER_COLUMNS)
    filtered = alignments.join(
        expected,
        left_on=["query", "target"],
        right_on=CLUSTER_COLUMNS,
        how="semi",
    )
    validate_alignment_coverage(assignments, filtered)
    assert filtered.height == assignments.height
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    filtered.select(ALIGNMENT_COLUMNS).write_csv(
        output,
        separator="\t",
        include_header=False,
    )
    return filtered


def validate_score_features(feature_names: list[str] | tuple[str, ...]) -> None:
    """Require deployed model inputs to be declared Linclust-only features."""
    normalized = [name.lower() for name in feature_names]
    assert len(normalized) == len(set(normalized)), "duplicate feature names"
    forbidden = set(normalized) & FORBIDDEN_SCORE_INPUTS
    assert not forbidden, f"forbidden score inputs: {sorted(forbidden)}"
    unknown = set(normalized) - INFERENCE_FEATURES
    assert not unknown, f"undeclared score inputs: {sorted(unknown)}"


def cluster_membership_features(
    assignments: pl.DataFrame,
    *,
    member_to_genome: pl.DataFrame,
) -> pl.DataFrame:
    """Build the required membership baselines for every window."""
    assert set(CLUSTER_COLUMNS).issubset(assignments.columns)
    assert {"member", "source_genome"}.issubset(member_to_genome.columns)
    assert member_to_genome["member"].n_unique() == member_to_genome.height
    joined = assignments.join(member_to_genome, on="member", how="left", validate="m:1")
    assert joined["source_genome"].is_not_null().all()
    per_genome = (
        joined.group_by("representative", "source_genome")
        .len()
        .rename({"len": "members_per_genome"})
    )
    per_cluster = joined.group_by("representative").agg(
        pl.len().alias("cluster_member_count"),
        pl.col("source_genome").n_unique().alias("distinct_genome_count"),
    )
    multiplicity = per_genome.group_by("representative").agg(
        pl.col("members_per_genome").max().alias("max_members_per_genome")
    )
    clusters = per_cluster.join(
        multiplicity, on="representative", validate="1:1"
    ).with_columns(
        (pl.col("max_members_per_genome") / pl.col("cluster_member_count")).alias(
            "dominant_genome_fraction"
        ),
        (pl.col("cluster_member_count") == 1).alias("singleton"),
    )
    return assignments.join(clusters, on="representative", validate="m:1")
