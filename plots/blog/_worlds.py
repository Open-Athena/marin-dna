"""The four metric-worlds of the blog figure family: **dataset × method**.

Each world bundles a display label, a ``read(model_id) -> tidy DataFrame`` reader
(one GPU forward pass produces the scores + embeddings both worlds of a dataset
read from), and the trajectory traits to draw. SGE assays only coding/splice, so
its trait set is narrower (missense / splicing) than Mendelian's.
"""

from __future__ import annotations

from marin_dna.pipelines.evals.blog_metrics import (
    read_llr_metrics,
    read_probe_metrics,
    read_sge_metrics,
)

# (subset, label, EARTH_QUAL color slot) — colors matched across Figs 5–8.
_MISSENSE = ("missense_variant", "missense", 0)
_PROMOTER = ("tss_proximal", "promoter", 1)
_SPLICING = ("splicing", "splicing", 4)

# Trajectory traits per dataset (Fig 7/8). Mendelian shows Eric's headline three;
# SGE its two assayed consequences.
MENDELIAN_TRAJ_TRAITS = (_MISSENSE, _PROMOTER, _SPLICING)
SGE_TRAJ_TRAITS = (_MISSENSE, _SPLICING)


def _read_mendelian_llr(model_id: str):
    return read_llr_metrics(model_id, "mendelian_traits")


def _read_mendelian_probe(model_id: str):
    return read_probe_metrics(model_id, "mendelian_traits")


def _read_sge_llr(model_id: str):
    return read_sge_metrics(model_id, kind="metrics", score_type="minus_llr_avg")


def _read_sge_probe(model_id: str):
    return read_sge_metrics(model_id, kind="probe_metrics", score_type="probe_score")


class World:
    """One (dataset, method) cell of the parallel figure family."""

    def __init__(self, key, label, read, traits):
        self.key = key
        self.label = label
        self.read = read
        self.traits = traits


WORLDS: dict[str, World] = {
    "mendelian_llr": World(
        "mendelian_llr",
        "Mendelian · zero-shot LLR",
        _read_mendelian_llr,
        MENDELIAN_TRAJ_TRAITS,
    ),
    "mendelian_probe": World(
        "mendelian_probe",
        "Mendelian · frozen-embedding probe",
        _read_mendelian_probe,
        MENDELIAN_TRAJ_TRAITS,
    ),
    "sge_llr": World(
        "sge_llr",
        "SGE · zero-shot LLR",
        _read_sge_llr,
        SGE_TRAJ_TRAITS,
    ),
    "sge_probe": World(
        "sge_probe",
        "SGE · frozen-embedding probe",
        _read_sge_probe,
        SGE_TRAJ_TRAITS,
    ),
}
