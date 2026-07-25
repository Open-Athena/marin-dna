"""Compatibility shims for the issue #402 standalone online evaluation."""

from __future__ import annotations

import sys
import types

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
