from __future__ import annotations

import numpy as np

from exp479_mntp.config import NUCLEOTIDE_LENGTH
from exp479_mntp.mntp_dependency import (
    LDLR_EVALUATION_ARTIFACT,
    LDLR_LOCUS_NAME,
    SOURCE_MODEL_ARTIFACT,
    dependency_checks,
    selected_locus,
)


def test_selected_dependency_is_browser_default_ldlr() -> None:
    locus = selected_locus()
    assert LDLR_LOCUS_NAME == "LDLR"
    assert (locus.chrom, locus.start, locus.end, locus.strand) == (
        "19",
        11_089_299,
        11_089_425,
        "+",
    )
    assert "step-1000:v0" in SOURCE_MODEL_ARTIFACT
    assert "ldlr-dependency" in LDLR_EVALUATION_ARTIFACT


def test_dependency_checks_require_zero_diagonal_and_both_contexts() -> None:
    matrix = np.zeros((NUCLEOTIDE_LENGTH, NUCLEOTIDE_LENGTH), dtype=np.float32)
    matrix[1, 2] = 0.5
    matrix[2, 1] = 0.25
    passing = dependency_checks(matrix)
    assert passing["passed"]
    assert passing["past_context_maximum"] == 0.5
    assert passing["future_context_maximum"] == 0.25

    matrix[0, 0] = 0.1
    assert not dependency_checks(matrix)["passed"]
