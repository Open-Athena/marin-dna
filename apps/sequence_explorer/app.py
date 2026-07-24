# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "jaxtyping==0.3.9",
#     "marimo==0.23.15",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "plotly==6.9.0",
#     "torch==2.8.0",
#     "transformers==4.57.6",
# ]
# ///

"""Public molab-hosted MarinDNA sequence explorer (issue #387)."""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full", app_title="MarinDNA sequence explorer")


@app.cell
def _():
    import hashlib
    import importlib
    import importlib.util
    import os
    import shutil
    import subprocess
    import sys
    import time
    from functools import cache

    import marimo as mo
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    SOURCE_REVISION = "1c22b21e28cf5506babf077758a365fdf9d5cf70"
    if importlib.util.find_spec("marin_dna") is None:
        _uv = shutil.which("uv")
        if _uv is None:
            raise RuntimeError(
                "The pinned MarinDNA source requires the uv executable supplied "
                "by molab, but uv was not found on PATH."
            )
        subprocess.run(
            [
                _uv,
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-deps",
                f"git+https://github.com/Open-Athena/marin-dna.git@{SOURCE_REVISION}",
            ],
            check=True,
        )
        importlib.invalidate_caches()
    assert importlib.util.find_spec("marin_dna") is not None, (
        "the pinned MarinDNA source package was not installed"
    )

    from marin_dna.apps.sequence_explorer_examples import (
        CUSTOM_SEQUENCE_LABEL,
        DEFAULT_EXAMPLE,
        EXAMPLES,
    )
    from marin_dna.apps.sequence_explorer_ui import (
        dependency_loading_figure,
        download_links_html,
        logo_figure,
        sequence_tracks_figure,
        span_from_plotly_points,
    )
    from marin_dna.model.interpretation import nucleotide_dependency_map
    from marin_dna.model.sequence_interpretation import (
        normalize_dna_sequence,
        nucleotide_logo,
    )

    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        CUSTOM_SEQUENCE_LABEL,
        DEFAULT_EXAMPLE,
        EXAMPLES,
        SOURCE_REVISION,
        cache,
        dependency_loading_figure,
        download_links_html,
        logo_figure,
        hashlib,
        mo,
        normalize_dna_sequence,
        np,
        nucleotide_dependency_map,
        nucleotide_logo,
        os,
        sequence_tracks_figure,
        span_from_plotly_points,
        sys,
        time,
        torch,
    )


@app.cell
def _(SOURCE_REVISION, os):
    MODEL_ID = "bolinas-dna/marin-dna-exp135-m5.1"
    MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
    APPLICATION_REVISION = os.getenv("SOURCE_REVISION", SOURCE_REVISION)
    APPLICATION_SOURCE_URL = (
        "https://github.com/Open-Athena/marin-dna/tree/"
        f"{APPLICATION_REVISION}/apps/sequence_explorer"
    )
    BATCH_SIZE = int(os.getenv("NUCLEOTIDE_DEPENDENCY_BATCH_SIZE", "32"))
    _progressive_threshold = os.getenv("PROGRESSIVE_MIN_LENGTH")
    PROGRESSIVE_MIN_LENGTH = (
        int(_progressive_threshold) if _progressive_threshold else None
    )
    assert BATCH_SIZE >= 1
    assert PROGRESSIVE_MIN_LENGTH is None or PROGRESSIVE_MIN_LENGTH >= 16
    return (
        APPLICATION_REVISION,
        APPLICATION_SOURCE_URL,
        BATCH_SIZE,
        MODEL_ID,
        MODEL_REVISION,
        PROGRESSIVE_MIN_LENGTH,
    )


@app.cell
def _(
    APPLICATION_SOURCE_URL,
    MODEL_ID,
    MODEL_REVISION,
    mo,
):
    mo.vstack(
        [
            mo.md(
                f"""
                # MarinDNA sequence explorer

                Explore the MarinDNA 1B model's **sequence logo** and
                **nucleotide dependency map** (forward/RC averaged) for a DNA
                sequence.

                [Open Athena / MarinDNA source]({APPLICATION_SOURCE_URL}) ·
                [Pinned model](https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION})
                """
            ),
            mo.Html(
                """
                <style>
                  main { max-width: 1180px !important; }
                  .download-links { display: flex; flex-wrap: wrap; gap: .75rem; }
                  .download-links a {
                    border: 1px solid #cbd5e1; border-radius: .5rem;
                    padding: .55rem .8rem; color: #0f172a;
                    text-decoration: none; background: #f8fafc;
                  }
                  .download-links a:hover { background: #e2e8f0; }
                </style>
                """
            ),
        ]
    )
    return


@app.cell
def _(mo, sys, torch):
    try:
        _cuda_available = torch.cuda.is_available()
        _gpu_name = (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if _cuda_available
            else "none"
        )
        _runtime_error = None
    except Exception as _error:
        _cuda_available = False
        _gpu_name = "unavailable"
        _runtime_error = f"{type(_error).__name__}: {_error}"

    _runtime_summary = (
        f"Python **{sys.version.split()[0]}** · PyTorch **{torch.__version__}** · "
        f"PyTorch CUDA **{torch.version.cuda or 'none'}** · "
        f"CUDA available: **{_cuda_available}** · Device: **{_gpu_name}**"
    )
    if _runtime_error is not None:
        _runtime_summary += f"\n\nCUDA initialization error: `{_runtime_error}`"
    mo.callout(
        mo.md(f"**Runtime status.** {_runtime_summary}"),
        kind="success" if _cuda_available else "danger",
    )
    return


@app.cell
def _(DEFAULT_EXAMPLE, mo):
    get_sequence, set_sequence = mo.state(DEFAULT_EXAMPLE.sequence)
    get_sequence_source, set_sequence_source = mo.state(DEFAULT_EXAMPLE.label)
    return get_sequence, get_sequence_source, set_sequence, set_sequence_source


@app.cell
def _(mo):
    get_track_view, set_track_view = mo.state(None, allow_self_loops=True)
    return get_track_view, set_track_view


@app.cell
def _(
    CUSTOM_SEQUENCE_LABEL,
    EXAMPLES,
    get_sequence_source,
    mo,
    set_sequence,
    set_sequence_source,
):
    _examples_by_label = {example.label: example.sequence for example in EXAMPLES}
    _selection_options = [*list(_examples_by_label), CUSTOM_SEQUENCE_LABEL]

    def _select_sequence_source(label):
        assert label in _selection_options
        set_sequence_source(label)
        if label != CUSTOM_SEQUENCE_LABEL:
            set_sequence(_examples_by_label[label])

    example_selector = mo.ui.dropdown(
        options=_selection_options,
        value=get_sequence_source(),
        label="Recommended examples",
        on_change=_select_sequence_source,
        searchable=True,
        full_width=True,
    )
    example_selector
    return (example_selector,)


@app.cell
def _(
    CUSTOM_SEQUENCE_LABEL,
    get_sequence,
    mo,
    set_sequence,
    set_sequence_source,
):
    def _set_custom_sequence(sequence):
        set_sequence(sequence)
        set_sequence_source(CUSTOM_SEQUENCE_LABEL)

    sequence_input = mo.ui.text_area(
        value=get_sequence(),
        placeholder="Enter 16–255 A/C/G/T bases",
        rows=6,
        label="DNA sequence (16–255 bp; A/C/G/T only)",
        debounce=300,
        full_width=True,
        on_change=_set_custom_sequence,
    )
    analyze_button = mo.ui.run_button(
        label="Analyze sequence",
        tooltip="Run the pinned model on the current sequence",
        kind="success",
    )
    mo.vstack(
        [
            sequence_input,
            analyze_button,
            mo.md(
                "Changing the example repopulates the editable sequence. Any edit "
                "clears the current plots; press **Analyze sequence** to recompute."
            ),
        ]
    )
    return analyze_button, sequence_input


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    MODEL_ID,
    MODEL_REVISION,
    cache,
    nucleotide_logo,
    torch,
):
    @cache
    def load_model():
        if not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA GPU is required. In molab, attach the free GPU from the "
                "notebook specs menu; analysis will not fall back to CPU."
            )
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        model.eval()
        # One deterministic warm-up also checks BOS and one-token-per-base
        # tokenizer assumptions before a user sequence reaches inference.
        nucleotide_logo(model, tokenizer, "ACGTACGTACGTACGT")
        torch.cuda.synchronize()
        return model, tokenizer, gpu_name

    return (load_model,)


@app.cell
def _(
    APPLICATION_REVISION,
    analyze_button,
    BATCH_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    PROGRESSIVE_MIN_LENGTH,
    dependency_loading_figure,
    load_model,
    logo_figure,
    hashlib,
    mo,
    normalize_dna_sequence,
    np,
    nucleotide_dependency_map,
    nucleotide_logo,
    sequence_input,
    time,
    torch,
):
    mo.stop(
        not analyze_button.value,
        mo.callout(
            mo.md(
                "Choose or edit a sequence, then press **Analyze sequence**. "
                "Editing clears existing plots and never triggers inference."
            ),
            kind="info",
        ),
    )
    _raw_sequence = sequence_input.value
    try:
        _normalized = normalize_dna_sequence(_raw_sequence)
    except ValueError as _error:
        mo.stop(True, mo.callout(str(_error), kind="danger"))

    _sequence_sha256 = hashlib.sha256(_normalized.encode("ascii")).hexdigest()
    assert len(_sequence_sha256) == 64

    _length = len(_normalized)
    _preparation_started = time.perf_counter()
    _stage = "model download and warm-up"
    try:
        with mo.status.spinner(
            title="Running MarinDNA",
            subtitle="Loading and warming the pinned model for this session…",
        ) as _spinner:
            _model, _tokenizer, _gpu_name = load_model()
            _model_ready = time.perf_counter()
            _stage = "CUDA memory initialization"
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            _stage = "probability logo"
            _logo_started = time.perf_counter()
            _logo = nucleotide_logo(_model, _tokenizer, _normalized)
            torch.cuda.synchronize()
            _logo_finished = time.perf_counter()

            _preparation_seconds = _model_ready - _preparation_started
            _logo_seconds = _logo_finished - _logo_started
            _progressive = (
                PROGRESSIVE_MIN_LENGTH is not None and _length >= PROGRESSIVE_MIN_LENGTH
            )
            if _progressive:
                mo.output.replace(
                    mo.vstack(
                        [
                            mo.md(
                                f"Length: **{_length} bp** · Time to logo: "
                                f"**{_logo_seconds:.2f} s**"
                            ),
                            logo_figure(_logo, span=(0, _length)),
                            dependency_loading_figure(),
                        ]
                    )
                )
            _stage = "nucleotide-dependency map"
            _spinner.update(subtitle="Building the nucleotide-dependency map…")
            _dependency = nucleotide_dependency_map(
                _model,
                _tokenizer,
                _normalized,
                rc=True,
                combine="mean",
                norm_ord=np.inf,
                batch_size=BATCH_SIZE,
            )
            torch.cuda.synchronize()
            _finished = time.perf_counter()
    except Exception as _error:
        _failure = f"{type(_error).__name__}: {_error}"
        print(f"Analysis failed during {_stage}: {_failure}", flush=True)
        mo.stop(
            True,
            mo.callout(
                mo.md(f"Analysis failed during **{_stage}**.\n\n`{_failure}`"),
                kind="danger",
            ),
        )

    _dependency_seconds = _finished - _logo_finished
    _peak_vram_bytes = int(torch.cuda.max_memory_allocated())
    assert _dependency.shape == (_length, _length)
    assert np.isfinite(_dependency).all()
    assert np.allclose(_dependency, _dependency.T, atol=1e-6)

    _metadata = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "application_revision": APPLICATION_REVISION,
        "sequence_length_bp": str(_length),
        "coordinate_system": "0-based sequence-relative",
        "logo_strand_rule": "mean_forward_and_transformed_rc_logits_then_softmax",
        "dependency_combine": "mean",
        "gpu": _gpu_name,
    }
    analysis_result = {
        "logo": _logo,
        "dependency": _dependency,
        "length": _length,
        "sequence_sha256": _sequence_sha256,
        "metadata": _metadata,
        "summary": f"""
            **Analysis complete.** Length: **{_length} bp** · GPU: **{_gpu_name}**<br>
            Session/model preparation: **{_preparation_seconds:.2f} s** ·
            Time to logo: **{_logo_seconds:.2f} s** · Additional dependency-map
            time: **{_dependency_seconds:.2f} s** · Analysis total:
            **{_logo_seconds + _dependency_seconds:.2f} s** · Peak VRAM:
            **{_peak_vram_bytes / 2**30:.2f} GiB**<br>
            Model `{MODEL_REVISION}` · Application `{APPLICATION_REVISION}`
            """,
    }
    return (analysis_result,)


@app.cell
def _(
    analysis_result,
    download_links_html,
    get_sequence,
    get_track_view,
    hashlib,
    mo,
    normalize_dna_sequence,
    sequence_tracks_figure,
    set_track_view,
    span_from_plotly_points,
):
    try:
        _current_sequence = normalize_dna_sequence(get_sequence())
    except ValueError:
        _current_sequence = None
    _current_sha256 = (
        hashlib.sha256(_current_sequence.encode("ascii")).hexdigest()
        if _current_sequence is not None
        else None
    )
    mo.stop(_current_sha256 != analysis_result["sequence_sha256"], mo.md(""))

    _view_state = get_track_view()
    _span = (
        tuple(_view_state["span"])
        if _view_state is not None
        and _view_state["sequence_sha256"] == analysis_result["sequence_sha256"]
        else None
    )

    def _update_track_view(selection):
        _selected_span = span_from_plotly_points(analysis_result["length"], selection)
        if _selected_span is not None:
            set_track_view(
                {
                    "sequence_sha256": analysis_result["sequence_sha256"],
                    "span": list(_selected_span),
                }
            )

    def _reset_track_view(_value):
        set_track_view(None)

    sequence_tracks = mo.ui.plotly(
        sequence_tracks_figure(
            _current_sequence,
            analysis_result["logo"],
            analysis_result["dependency"],
            span=_span,
        ),
        config={
            "displaylogo": False,
            "scrollZoom": False,
            "modeBarButtonsToRemove": [
                "zoom2d",
                "pan2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "resetScale2d",
                "lasso2d",
            ],
        },
        label="Aligned sequence tracks",
        on_change=_update_track_view,
    )
    _reset_view = mo.ui.button(
        label="Reset view",
        tooltip="Restore the full sequence span",
        on_click=_reset_track_view,
    )
    _visible_span = (
        f"Showing positions **{_span[0]}–{_span[1] - 1}** ({_span[1] - _span[0]} bp)."
        if _span is not None
        else f"Showing the full **0–{analysis_result['length'] - 1}** span."
    )
    mo.vstack(
        [
            mo.callout(mo.md(analysis_result["summary"]), kind="success"),
            mo.md(
                "Drag horizontally over the **DNA sequence** track to select and zoom "
                "all three aligned tracks. The dependency map uses the same span on "
                f"both axes. {_visible_span}"
            ),
            sequence_tracks,
            _reset_view,
            mo.Html(
                download_links_html(
                    analysis_result["logo"],
                    analysis_result["dependency"],
                    metadata=analysis_result["metadata"],
                )
            ),
        ],
        gap=1.0,
    )
    return


if __name__ == "__main__":
    app.run()
