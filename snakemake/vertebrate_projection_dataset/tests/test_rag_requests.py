from dataclasses import replace

import pytest

from marin_dna_vertebrate_projection.rag.documents import Locus
from marin_dna_vertebrate_projection.rag.requests import (
    ProjectionIdentity,
    SourceRecord,
    plan_requests,
)


def identity():
    return ProjectionIdentity("bird", "bird-v1", "a" * 64, "b" * 64, "c" * 64, "d" * 64)


def test_exact_requests_deduplicate_across_catalogs_and_benchmarks():
    shared = Locus("chr1", 0, 255)
    nearby = Locus("chr1", 1, 256)
    rows = [
        SourceRecord("training:cds", "1", shared),
        SourceRecord("mendelian", "1", shared),
        SourceRecord("sge", "1", nearby),
    ]
    plan = plan_requests(rows, {"bird": identity()}, chrom_sizes={"chr1": 300})
    assert len(plan.source_rows) == 3
    assert len({row.row_id for row in rows}) == 3
    assert plan.requests == (shared, nearby)
    assert plan.missing["bird"] == plan.requests


@pytest.mark.parametrize(
    "field,value",
    [
        ("chain_sha256", "e" * 64),
        ("genome_sha256", "f" * 64),
        ("assembly", "bird-v2"),
        ("contract", "single-best-liftover"),
        ("source_sizes_sha256", "e" * 64),
        ("target_sizes_sha256", "f" * 64),
    ],
)
def test_cache_reuse_requires_every_identity_component(field, value):
    locus = Locus("chr1", 0, 255)
    original = identity()
    different = replace(original, **{field: value})
    rows = [SourceRecord("training:cds", "1", locus)]
    exact = plan_requests(
        rows,
        {"bird": original},
        chrom_sizes={"chr1": 300},
        cached={"bird": {locus: original.fingerprint}},
    )
    assert exact.reused["bird"] == (locus,)
    assert exact.missing["bird"] == ()
    changed = plan_requests(
        rows,
        {"bird": different},
        chrom_sizes={"chr1": 300},
        cached={"bird": {locus: original.fingerprint}},
    )
    assert changed.reused["bird"] == ()
    assert changed.missing["bird"] == (locus,)


def test_out_of_bounds_and_duplicate_source_rows_fail():
    row = SourceRecord("sge", "1", Locus("chr1", 0, 255))
    with pytest.raises(ValueError, match="outside pinned genome"):
        plan_requests([row], {"bird": identity()}, chrom_sizes={"chr1": 254})
    with pytest.raises(ValueError, match="duplicate namespaced"):
        plan_requests([row, row], {"bird": identity()}, chrom_sizes={"chr1": 300})
