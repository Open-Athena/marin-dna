import importlib
import sys
from pathlib import Path

import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer


class _DummyModel:
    def to(self, device):
        assert device == "cuda"
        return self

    def eval(self):
        return self


def test_app_builds_without_downloading_model(monkeypatch):
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: _DummyModel(),
    )

    def fake_gpu(*args, **kwargs):
        def decorate(function):
            return function

        return decorate

    monkeypatch.setattr(spaces, "GPU", fake_gpu)
    app_directory = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(app_directory))
    sys.modules.pop("app", None)
    module = importlib.import_module("app")

    assert module.MODEL_REVISION == "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
    assert module.demo is not None
    assert module.PROGRESSIVE_MIN_LENGTH is None
