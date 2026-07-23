import base64
import re

import numpy as np

from marin_dna.apps.sequence_explorer_ui import (
    dependency_figure,
    download_links_html,
    logo_figure,
    navigator_figure,
    span_from_plotly_ranges,
)
from marin_dna.model.sequence_interpretation import NucleotideLogo


def _logo() -> NucleotideLogo:
    probabilities = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.2, 0.3, 0.4],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=np.float32,
    )
    entropy = -(probabilities * np.log2(probabilities)).sum(axis=1)
    information = 2.0 - entropy
    return NucleotideLogo(
        probabilities=probabilities,
        entropy_bits=entropy,
        information_bits=information,
        glyph_heights_bits=probabilities * information[:, None],
    )


def test_logo_uses_glyph_paths_and_has_required_hover():
    figure = logo_figure(_logo(), span=(1, 3))
    assert figure.layout.xaxis.range == (0.5, 2.5)
    assert figure.layout.yaxis.range == (0, 2)
    assert figure.layout.shapes
    assert {shape.type for shape in figure.layout.shapes} == {"path"}
    assert "Probability" in figure.data[0].hovertemplate
    assert "Glyph height" in figure.data[0].hovertemplate
    assert "Stack information" in figure.data[0].hovertemplate


def test_dependency_figure_has_same_half_open_span_on_both_axes():
    dependency = np.arange(25, dtype=float).reshape(5, 5)
    dependency = (dependency + dependency.T) / 2.0
    figure = dependency_figure(dependency, span=(1, 4))
    assert figure.layout.xaxis.range == (0.5, 3.5)
    assert figure.layout.yaxis.range == (3.5, 0.5)
    assert figure.layout.yaxis.scaleanchor == "x"
    assert "Position i" in figure.data[0].hovertemplate
    assert "Dependency" in figure.data[0].hovertemplate


def test_downloads_are_in_memory_csvs_with_revisions_and_no_sequence():
    logo = _logo()
    dependency = np.eye(3)
    html = download_links_html(
        logo,
        dependency,
        metadata={
            "model_revision": "model-sha",
            "application_revision": "app-sha",
        },
    )
    urls = re.findall(r'href="data:text/csv;base64,([^"]+)"', html)
    assert len(urls) == 2
    decoded = [base64.b64decode(url).decode("utf-8") for url in urls]
    assert "# model_revision=model-sha" in decoded[0]
    assert "# application_revision=app-sha" in decoded[1]
    assert "A_probability" in decoded[0]
    assert decoded[1].splitlines()[-1].startswith("2,")
    assert "ACGT" not in decoded[0]
    assert "ACGT" not in decoded[1]


def test_navigator_supports_horizontal_brush_and_full_range():
    figure = navigator_figure(_logo())
    assert figure.layout.dragmode == "select"
    assert figure.layout.selectdirection == "h"
    assert figure.layout.xaxis.range == (-0.5, 2.5)
    assert figure.layout.yaxis.range == (0, 2)


def test_plotly_ranges_convert_to_zero_based_half_open_span():
    assert span_from_plotly_ranges(10, None) == (0, 10)
    assert span_from_plotly_ranges(10, {}) == (0, 10)
    assert span_from_plotly_ranges(10, {"x": [1.2, 4.7]}) == (2, 5)
    assert span_from_plotly_ranges(10, {"x": [9.8, -2.0]}) == (0, 10)
    assert span_from_plotly_ranges(10, {"xaxis": [6.1, 6.2]}) == (7, 8)
