"""Public MarinDNA sequence explorer (issue #387)."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Generator
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples import DEFAULT_EXAMPLE, EXAMPLES, EXAMPLES_BY_LABEL
from marin_dna.model.interpretation import nucleotide_dependency_map
from marin_dna.model.sequence_interpretation import (
    NucleotideLogo,
    normalize_dna_sequence,
    nucleotide_logo,
)
from ui import (
    dependency_figure,
    dependency_loading_figure,
    download_links_html,
    logo_figure,
    navigator_dataframe,
)

MODEL_ID = "bolinas-dna/marin-dna-exp135-m5.1"
MODEL_REVISION = "c0676b2012b8b9c526deb26ff517f6b92b6d375d"
APPLICATION_REVISION = os.getenv(
    "SOURCE_REVISION", os.getenv("SPACE_COMMIT_SHA", "local-development")
)
APPLICATION_SOURCE_URL = (
    "https://github.com/Open-Athena/marin-dna/tree/"
    f"{APPLICATION_REVISION}/apps/sequence_explorer"
)
BATCH_SIZE = int(os.getenv("NUCLEOTIDE_DEPENDENCY_BATCH_SIZE", "32"))
GPU_DURATION_SECONDS = 120

# This remains unset until the required ZeroGPU benchmark establishes the
# shortest sequence length whose post-logo delay is >=3 s. Set it to that
# measured threshold at deployment time; do not guess.
_progressive_threshold = os.getenv("PROGRESSIVE_MIN_LENGTH")
PROGRESSIVE_MIN_LENGTH = int(_progressive_threshold) if _progressive_threshold else None


# ZeroGPU's CUDA emulation is active at module load. Hugging Face explicitly
# recommends placing models on CUDA here rather than moving them inside the
# decorated function, so model downloads and transfers are not repeated per
# submission.
TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
)
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda")
MODEL.eval()


def _estimate_text(raw_sequence: str) -> str:
    compact = "".join(raw_sequence.split()).upper()
    if not compact:
        return "Enter 16–255 A/C/G/T bases. Invalid input never requests a GPU."
    invalid = sorted(set(compact) - set("ACGT"))
    if invalid:
        return "Fix unsupported characters before submission; no GPU will be requested."
    if len(compact) < 16:
        return f"{len(compact)} bp entered; at least 16 bp are required."
    if len(compact) > 255:
        return f"{len(compact)} bp entered; the maximum is 255 bp."
    return (
        f"Valid input: **{len(compact)} bp**. The ZeroGPU allocation is capped at "
        f"**{GPU_DURATION_SECONDS} seconds**; the measured length-specific estimate "
        "will replace this cap after the benchmark gate."
    )


def _validate_submission(raw_sequence: str) -> tuple[str, str]:
    try:
        normalized = normalize_dna_sequence(raw_sequence)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    return normalized, normalized


def _select_example(label: str, current_sequence: str) -> tuple[str, str]:
    if label == "Custom sequence":
        return current_sequence, _estimate_text(current_sequence)
    example = EXAMPLES_BY_LABEL[label]
    return example.sequence, _estimate_text(example.sequence)


def _mark_custom(raw_sequence: str) -> tuple[Any, str]:
    return gr.Dropdown(value="Custom sequence"), _estimate_text(raw_sequence)


def _metadata(length: int) -> dict[str, str]:
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "application_revision": APPLICATION_REVISION,
        "sequence_length_bp": str(length),
        "coordinate_system": "0-based sequence-relative",
        "logo_strand_rule": "mean_forward_and_transformed_rc_logits_then_softmax",
        "dependency_combine": "mean",
    }


def _result_state(
    logo: NucleotideLogo,
    dependency: np.ndarray,
) -> dict[str, Any]:
    assert dependency.shape == (
        logo.probabilities.shape[0],
        logo.probabilities.shape[0],
    )
    return {"logo": logo, "dependency": dependency}


def _status_markdown(
    *,
    length: int,
    time_to_logo: float,
    dependency_seconds: float | None,
    peak_vram_bytes: int | None,
) -> str:
    runtime = (
        "Dependency map is still running."
        if dependency_seconds is None
        else (
            f"Additional dependency-map time: **{dependency_seconds:.2f} s** · "
            f"Total: **{time_to_logo + dependency_seconds:.2f} s**"
        )
    )
    vram = (
        "Peak VRAM: pending"
        if peak_vram_bytes is None
        else f"Peak VRAM: **{peak_vram_bytes / 2**30:.2f} GiB**"
    )
    return (
        f"Length: **{length} bp** · Time to logo: **{time_to_logo:.2f} s** · "
        f"{runtime} · {vram}  \n"
        f"Model `{MODEL_REVISION}` · Application `{APPLICATION_REVISION}`"
    )


@spaces.GPU(duration=GPU_DURATION_SECONDS)
def _analyze_sequence(
    sequence: str,
) -> Generator[tuple[Any, ...], None, None]:
    """One GPU allocation and one Gradio generator for both interpretations."""
    if not torch.cuda.is_available():
        raise gr.Error(
            "A ZeroGPU allocation was not provided. Analysis will not fall back to CPU."
        )
    # Validation ran in the preceding CPU event. Assert again inside the trust
    # boundary without logging or persisting the submitted sequence.
    normalized = normalize_dna_sequence(sequence)
    length = len(normalized)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started_at = time.perf_counter()

    logo = nucleotide_logo(MODEL, TOKENIZER, normalized)
    torch.cuda.synchronize()
    logo_finished_at = time.perf_counter()
    time_to_logo = logo_finished_at - started_at
    nav_data = navigator_dataframe(logo)
    full_span = (0, length)

    progressive = (
        PROGRESSIVE_MIN_LENGTH is not None and length >= PROGRESSIVE_MIN_LENGTH
    )
    if progressive:
        yield (
            logo_figure(logo, span=full_span),
            dependency_loading_figure(),
            nav_data,
            f"Visible span: **[0, {length})**",
            _status_markdown(
                length=length,
                time_to_logo=time_to_logo,
                dependency_seconds=None,
                peak_vram_bytes=None,
            ),
            "<p>Download links appear when the dependency map finishes.</p>",
            None,
        )

    dependency = nucleotide_dependency_map(
        MODEL,
        TOKENIZER,
        normalized,
        rc=True,
        combine="mean",
        norm_ord=np.inf,
        batch_size=BATCH_SIZE,
    )
    torch.cuda.synchronize()
    finished_at = time.perf_counter()
    dependency_seconds = finished_at - logo_finished_at
    peak_vram = int(torch.cuda.max_memory_allocated())
    result = _result_state(logo, dependency)

    yield (
        logo_figure(logo, span=full_span),
        dependency_figure(dependency, span=full_span),
        nav_data,
        f"Visible span: **[0, {length})**",
        _status_markdown(
            length=length,
            time_to_logo=time_to_logo,
            dependency_seconds=dependency_seconds,
            peak_vram_bytes=peak_vram,
        ),
        download_links_html(
            logo,
            dependency,
            metadata=_metadata(length),
        ),
        result,
    )


def _span_from_selection(length: int, selection: gr.SelectData) -> tuple[int, int]:
    assert isinstance(selection.index, (list, tuple)) and len(selection.index) == 2
    raw_start, raw_end = sorted(float(value) for value in selection.index)
    start = max(0, min(length - 1, math.ceil(raw_start)))
    end = max(start + 1, min(length, math.floor(raw_end) + 1))
    assert 0 <= start < end <= length
    return start, end


def _select_span(
    result: dict[str, Any] | None,
    selection: gr.SelectData,
) -> tuple[Any, Any, str]:
    if result is None:
        raise gr.Error("Run an analysis before selecting a visible span.")
    logo = result["logo"]
    dependency = result["dependency"]
    span = _span_from_selection(len(logo.information_bits), selection)
    return (
        logo_figure(logo, span=span),
        dependency_figure(dependency, span=span),
        f"Visible span: **[{span[0]}, {span[1]})**",
    )


def _reset_span(
    result: dict[str, Any] | None,
) -> tuple[Any, Any, str, pd.DataFrame]:
    if result is None:
        raise gr.Error("Run an analysis before resetting the visible span.")
    logo = result["logo"]
    dependency = result["dependency"]
    length = len(logo.information_bits)
    return (
        logo_figure(logo),
        dependency_figure(dependency),
        f"Visible span: **[0, {length})**",
        navigator_dataframe(logo),
    )


INTRODUCTION = f"""
# MarinDNA sequence explorer

Paste a DNA sequence to inspect what the headline MarinDNA 1B model has learned.
The logo averages forward and reverse-complement **logits** in forward-sequence
coordinates, then applies softmax. The dependency map reuses the tested
forward/RC categorical-Jacobian implementation.

These are **model interpretations**, not measurements of biological function or
clinical predictions. Do not submit personal, confidential, or identifiable
genetic data. Requests run on third-party Hugging Face infrastructure; see the
[Hugging Face privacy policy](https://huggingface.co/privacy).

[Open Athena / MarinDNA source]({APPLICATION_SOURCE_URL})
· [Pinned model](https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION})
"""

CSS = """
.gradio-container { max-width: 1180px !important; }
.download-links { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 0.5rem 0; }
.download-links a {
  border: 1px solid #cbd5e1; border-radius: 0.5rem; padding: 0.55rem 0.8rem;
  color: #0f172a; text-decoration: none; background: #f8fafc;
}
.download-links a:hover { background: #e2e8f0; }
"""

empty_navigator = pd.DataFrame(
    {"position": pd.Series(dtype=int), "information_bits": pd.Series(dtype=float)}
)
example_choices = ["Custom sequence", *[example.label for example in EXAMPLES]]

with gr.Blocks(title="MarinDNA sequence explorer", css=CSS) as demo:
    result_state = gr.State(value=None)
    validated_sequence = gr.State(value="")
    gr.Markdown(INTRODUCTION)
    with gr.Row():
        example_dropdown = gr.Dropdown(
            choices=example_choices,
            value=DEFAULT_EXAMPLE.label,
            label="Recommended example",
            scale=2,
        )
        run_button = gr.Button("Analyze sequence", variant="primary", scale=1)
    sequence_input = gr.Textbox(
        value=DEFAULT_EXAMPLE.sequence,
        label="DNA sequence (16–255 bp; A/C/G/T only)",
        lines=5,
        max_lines=8,
        show_copy_button=True,
    )
    estimate = gr.Markdown(_estimate_text(DEFAULT_EXAMPLE.sequence))

    status = gr.Markdown(
        f"Model `{MODEL_REVISION}` · Application `{APPLICATION_REVISION}`"
    )
    logo_plot = gr.Plot(label="Information-content sequence logo")
    dependency_plot = gr.Plot(label="Nucleotide-dependency map")
    gr.Markdown(
        "Drag a horizontal region in the navigator to link both views. "
        "Selections are interpreted as 0-based, half-open intervals."
    )
    navigator = gr.LinePlot(
        value=empty_navigator,
        x="position",
        y="information_bits",
        title="Full-sequence navigator",
        x_title="Sequence position (0-based)",
        y_title="Information (bits)",
        height=180,
        label="Full-sequence navigator",
    )
    with gr.Row():
        visible_span = gr.Markdown("Visible span: run an analysis")
        reset_button = gr.Button("Reset to full sequence")
    downloads = gr.HTML("<p>Download links appear after analysis.</p>")

    example_dropdown.input(
        _select_example,
        inputs=[example_dropdown, sequence_input],
        outputs=[sequence_input, estimate],
        queue=False,
    )
    sequence_input.input(
        _mark_custom,
        inputs=sequence_input,
        outputs=[example_dropdown, estimate],
        queue=False,
    )
    validation_event = run_button.click(
        _validate_submission,
        inputs=sequence_input,
        outputs=[validated_sequence, sequence_input],
        queue=False,
    )
    validation_event.success(
        _analyze_sequence,
        inputs=validated_sequence,
        outputs=[
            logo_plot,
            dependency_plot,
            navigator,
            visible_span,
            status,
            downloads,
            result_state,
        ],
    )
    navigator.select(
        _select_span,
        inputs=result_state,
        outputs=[logo_plot, dependency_plot, visible_span],
        queue=False,
    )
    navigator.double_click(
        _reset_span,
        inputs=result_state,
        outputs=[logo_plot, dependency_plot, visible_span, navigator],
        queue=False,
    )
    reset_button.click(
        _reset_span,
        inputs=result_state,
        outputs=[logo_plot, dependency_plot, visible_span, navigator],
        queue=False,
    )

demo.queue(default_concurrency_limit=1, max_size=8)

if __name__ == "__main__":
    demo.launch()
