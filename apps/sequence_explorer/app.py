# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "marimo==0.23.15",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "plotly==6.9.0",
#     "torch==2.8.0",
#     "transformers==4.57.6",
#     "marin-dna @ git+https://github.com/Open-Athena/marin-dna.git@2653622de8945e4ba2578458cb30b47803e63d9e",
# ]
# ///

"""Public molab-hosted MarinDNA sequence explorer (issue #387)."""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full", app_title="MarinDNA sequence explorer")


@app.cell
def _():
    import os
    import time
    from functools import cache

    import marimo as mo
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from marin_dna.apps.sequence_explorer_examples import DEFAULT_EXAMPLE, EXAMPLES
    from marin_dna.apps.sequence_explorer_ui import (
        dependency_figure,
        dependency_loading_figure,
        download_links_html,
        logo_figure,
        navigator_figure,
        span_from_plotly_ranges,
    )
    from marin_dna.model.interpretation import nucleotide_dependency_map
    from marin_dna.model.sequence_interpretation import (
        normalize_dna_sequence,
        nucleotide_logo,
    )

    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        DEFAULT_EXAMPLE,
        EXAMPLES,
        cache,
        dependency_figure,
        dependency_loading_figure,
        download_links_html,
        logo_figure,
        mo,
        navigator_figure,
        normalize_dna_sequence,
        np,
        nucleotide_dependency_map,
        nucleotide_logo,
        os,
        span_from_plotly_ranges,
        time,
        torch,
    )


@app.cell
def _(os):
    MODEL_ID = "bolinas-dna/marin-dna-exp135-m5.1"
    MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
    APPLICATION_REVISION = os.getenv("SOURCE_REVISION", "local-development")
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

                Paste a DNA sequence to inspect what the headline MarinDNA 1B
                model has learned. The logo averages forward and transformed
                reverse-complement **logits** before softmax. The dependency map
                reuses the tested forward/RC categorical-Jacobian implementation.

                These are **model interpretations**, not measurements of biological
                function or clinical predictions. Submit only public reference,
                synthetic, or otherwise non-sensitive sequences. **Do not submit
                personal, confidential, identifiable, patient, or protected genetic
                data.** Execution occurs on third-party CoreWeave infrastructure;
                see the [privacy policy](https://marimo.io/pages/legal/privacy) and
                [molab terms](https://molab.marimo.io/pages/legal/terms).

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
def _(DEFAULT_EXAMPLE, EXAMPLES, mo):
    _example_options = {
        "Custom sequence": "",
        **{example.label: example.sequence for example in EXAMPLES},
    }
    example_picker = mo.ui.dropdown(
        options=_example_options,
        value=DEFAULT_EXAMPLE.label,
        label="Recommended example",
        searchable=True,
        full_width=True,
    )
    example_picker
    return (example_picker,)


@app.cell
def _(example_picker, mo):
    sequence_form = mo.ui.text_area(
        value=example_picker.value,
        placeholder="Enter 16–255 A/C/G/T bases",
        rows=6,
        label="DNA sequence (16–255 bp; A/C/G/T only)",
        full_width=True,
    ).form(
        submit_button_label="Analyze sequence",
        submit_button_tooltip="Run the pinned model on the submitted sequence",
        clear_on_submit=False,
    )
    sequence_form
    return (sequence_form,)


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
    BATCH_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    PROGRESSIVE_MIN_LENGTH,
    dependency_loading_figure,
    load_model,
    logo_figure,
    mo,
    normalize_dna_sequence,
    np,
    nucleotide_dependency_map,
    nucleotide_logo,
    sequence_form,
    time,
    torch,
):
    _raw_sequence = sequence_form.value
    mo.stop(
        _raw_sequence is None,
        mo.callout(
            "Select an example or enter a custom sequence, then press "
            "**Analyze sequence**. Editing and linked zoom never trigger inference.",
            kind="info",
        ),
    )
    try:
        _normalized = normalize_dna_sequence(_raw_sequence)
    except ValueError as _error:
        mo.stop(True, mo.callout(str(_error), kind="danger"))

    _length = len(_normalized)
    _preparation_started = time.perf_counter()
    with mo.status.spinner(
        title="Running MarinDNA",
        subtitle="Loading and warming the pinned model for this session…",
    ) as _spinner:
        _model, _tokenizer, _gpu_name = load_model()
        _model_ready = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
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
        "metadata": _metadata,
    }
    mo.callout(
        mo.md(
            f"""
            **Analysis complete.** Length: **{_length} bp** · GPU: **{_gpu_name}**<br>
            Session/model preparation: **{_preparation_seconds:.2f} s** ·
            Time to logo: **{_logo_seconds:.2f} s** · Additional dependency-map
            time: **{_dependency_seconds:.2f} s** · Analysis total:
            **{_logo_seconds + _dependency_seconds:.2f} s** · Peak VRAM:
            **{_peak_vram_bytes / 2**30:.2f} GiB**<br>
            Model `{MODEL_REVISION}` · Application `{APPLICATION_REVISION}`
            """
        ),
        kind="success",
    )
    return (analysis_result,)


@app.cell
def _(mo):
    reset_span = mo.ui.button(
        value=0,
        on_click=lambda count: count + 1,
        label="Reset to full sequence",
        tooltip="Clear the navigator brush without rerunning the model",
    )
    return (reset_span,)


@app.cell
def _(analysis_result, mo, navigator_figure, reset_span):
    # Referencing the reset counter recreates the navigator and clears its
    # selection. The analysis cell does not depend on this button.
    _reset_generation = reset_span.value
    navigator = mo.ui.plotly(
        navigator_figure(analysis_result["logo"]),
        config={"displaylogo": False, "scrollZoom": False},
        label="Full-sequence navigator",
    )
    mo.vstack(
        [
            mo.md(
                "Drag a horizontal span below to link both views. Selections use "
                "0-based, half-open coordinates `[start, end)`. Double-clicking "
                "the navigator also clears its Plotly selection."
            ),
            navigator,
            reset_span,
        ]
    )
    return (navigator,)


@app.cell
def _(
    analysis_result,
    dependency_figure,
    download_links_html,
    logo_figure,
    mo,
    navigator,
    span_from_plotly_ranges,
):
    _span = span_from_plotly_ranges(
        analysis_result["length"],
        navigator.ranges,
    )
    mo.vstack(
        [
            mo.md(f"Visible span: **[{_span[0]}, {_span[1]})**"),
            logo_figure(analysis_result["logo"], span=_span),
            dependency_figure(analysis_result["dependency"], span=_span),
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
