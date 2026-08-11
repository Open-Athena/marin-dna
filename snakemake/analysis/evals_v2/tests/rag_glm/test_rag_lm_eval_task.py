"""Request/result tests for the paired issue #402 lm-eval task."""

from __future__ import annotations

import pytest

pytest.importorskip("lm_eval")

from marin_dna_evals.lm_eval.rag_dna_vep_llr_eval import (
    RagDnaVepLlrEvalTask,
)


def _task() -> RagDnaVepLlrEvalTask:
    task = object.__new__(RagDnaVepLlrEvalTask)
    task._metrics = ["auprc"]
    return task


def _doc() -> dict[str, object]:
    return {
        "chrom": "1",
        "pos": 402,
        "ref": "A",
        "alt": "G",
        "target": 1,
        "subset": "all",
        "match_group": 17,
        "strand": "+",
        "ref_completion": "A" + "C" * 127,
        "alt_completion": "G" + "C" * 127,
    }


def test_task_emits_one_paired_cache_request() -> None:
    doc = _doc()
    request = _task().construct_requests(doc, "materialized-context")
    assert request.request_type == "rag_loglikelihood"
    assert request.args == (
        "materialized-context",
        doc["ref_completion"],
        doc["alt_completion"],
    )


def test_task_uses_raw_llr_returned_by_paired_scorer() -> None:
    result = _task().process_results(_doc(), [(-3.25, -2.0, 1.25)])
    assert result == {"auprc": (1.25, 1, "all", ("1", 402, "A", "G"), 17, "+")}
