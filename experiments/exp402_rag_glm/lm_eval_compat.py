"""Compatibility shims for the issue #402 standalone online evaluation."""

from __future__ import annotations

import sys
import types
from typing import Any

import transformers


def stub_unused_transformers5_multimodal_adapter() -> None:
    """Keep lm-eval's unused multimodal adapter from breaking text-only eval.

    The Marin-pinned lm-eval fork imports every registered model adapter. Its
    multimodal HF adapter still references an API removed in Transformers 5,
    while this experiment only uses Levanter's text-only ``TemplateLM``.
    """
    major_version = int(transformers.__version__.split(".", maxsplit=1)[0])
    if major_version < 5:
        return
    module_name = "lm_eval.models.hf_vlms"
    sys.modules.setdefault(module_name, types.ModuleType(module_name))


def allow_zero_shot_rag_sample_logging() -> None:
    """Provide lm-eval's logging-only target for the custom paired request.

    The pinned fork now calls ``doc_to_target`` only while assembling logged
    samples. This task is fixed at zero-shot and constructs its complete paired
    request itself, so an empty logging target cannot affect model inputs or
    metrics.
    """
    from marin_dna.pipelines.evals.lm_eval.rag_dna_vep_llr_eval import (
        RagDnaVepLlrEvalTask,
    )

    def _logging_target(self: RagDnaVepLlrEvalTask, doc: Any) -> str:
        del self
        assert isinstance(doc, dict)
        return ""

    RagDnaVepLlrEvalTask.doc_to_target = _logging_target
