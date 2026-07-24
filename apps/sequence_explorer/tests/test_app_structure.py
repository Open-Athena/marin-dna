import importlib
import sys
from pathlib import Path

import marimo


def test_marimo_app_builds_without_loading_the_model(monkeypatch):
    app_directory = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(app_directory))
    sys.modules.pop("app", None)
    module = importlib.import_module("app")

    assert isinstance(module.app, marimo.App)
    source = (app_directory / "app.py").read_text()
    assert "c0676b2012b8b9c526deb26ff517f6b92b6d375d" in source
    assert source.count("3fe94227186175fa6f5bf074c27d6cd3c3b0f391") == 1
    assert '"jaxtyping==0.3.9"' in source
    assert '"--no-deps"' in source
    assert "marin-dna @ git+" not in source
    assert "cyvcf2" not in source
    assert "PROGRESSIVE_MIN_LENGTH = (" in source
    assert "**Runtime status.**" in source
    assert "Analysis failed during **{_stage}**." in source
    assert "sequence_form" not in source
    assert "Full-sequence navigator" not in source
    assert "navigator_figure" not in source
    assert "mo.state(DEFAULT_EXAMPLE.sequence)" in source
    assert "mo.state(DEFAULT_EXAMPLE.label)" in source
    assert 'label="Recommended examples"' in source
    assert 'CUSTOM_SEQUENCE_LABEL = "Custom sequence"' not in source
    assert "CUSTOM_SEQUENCE_LABEL" in source
    assert "on_change=_select_sequence_source" in source
    assert "on_change=_set_custom_sequence" in source
    assert "set_sequence_source(CUSTOM_SEQUENCE_LABEL)" in source
    assert "get_example_sequence" not in source
    assert "mo.ui.run_button" in source
    assert "not analyze_button.value" in source
    assert "_raw_sequence = sequence_input.value" in source
    assert 'analysis_result["sequence_sha256"]' in source
    assert "_current_sha256 !=" in source
    assert "sequence_tracks_figure" in source
    assert "_current_sequence," in source
    assert "DNA sequence** track" in source
    assert "(forward/RC averaged)" in source
    assert "mo.callout(\n            mo.md(" in source
    assert "These are **model interpretations**" not in source
    assert "marimo.App" in source
    assert "@spaces.GPU" not in source
    assert "import gradio" not in source
