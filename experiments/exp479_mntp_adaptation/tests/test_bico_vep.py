from __future__ import annotations

import pandas as pd

from exp479_mntp.bico_vep import BICO_VEP_STEPS, _select_primary_endpoint
from exp479_mntp.vep import DatasetSpec
from exp479_mntp.vep_metrics import GLOBAL, MACRO


def _matched(name: str) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        repo_id="example/dataset",
        revision="a" * 40,
        protocol="minus_llr",
        evaluation="matched",
    )


def test_bico_vep_steps_cover_source_and_every_hundred_updates() -> None:
    assert BICO_VEP_STEPS == tuple(range(0, 1_001, 100))


def test_mendelian_selects_macro_and_complex_selects_global() -> None:
    metrics = pd.DataFrame(
        [
            {"subset": GLOBAL, "value": 0.1, "se": 0.01},
            {"subset": MACRO, "value": 0.2, "se": 0.02},
        ]
    )

    mendelian = _select_primary_endpoint(_matched("mendelian_traits"), metrics)
    complex_traits = _select_primary_endpoint(_matched("complex_traits"), metrics)

    assert mendelian["value"] == 0.2
    assert complex_traits["value"] == 0.1


def test_sge_selects_accession_consequence_macro() -> None:
    spec = DatasetSpec(
        name="sge",
        repo_id="example/dataset",
        revision="a" * 40,
        protocol="minus_llr",
        evaluation="sge",
    )
    metrics = pd.DataFrame(
        [
            {
                "subset": MACRO,
                "accession": "urn:mavedb:1",
                "gene": "GENE",
                "value": 0.1,
                "se": 0.01,
            },
            {
                "subset": "both",
                "accession": MACRO,
                "gene": MACRO,
                "value": 0.2,
                "se": 0.02,
            },
            {
                "subset": MACRO,
                "accession": MACRO,
                "gene": MACRO,
                "value": 0.3,
                "se": 0.03,
            },
        ]
    )

    selected = _select_primary_endpoint(spec, metrics)

    assert selected["value"] == 0.3
    assert selected["se"] == 0.03
