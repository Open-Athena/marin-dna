import base64
import re

import numpy as np
import pytest

from marin_dna.apps.sequence_explorer_ui import (
    dependency_figure,
    download_links_html,
    logo_figure,
    sequence_tracks_figures,
    span_from_plotly_points,
    span_from_plotly_selection,
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


def test_plotly_points_convert_to_half_open_span():
    assert span_from_plotly_points(10, None) is None
    assert span_from_plotly_points(10, []) is None
    assert span_from_plotly_points(10, [{"x": 4}, {"x": 2}, {"x": 3}]) == (
        2,
        5,
    )
    assert span_from_plotly_points(
        10,
        [
            {"x2": 0},
            {"Sequence position (0-based)": 9},
            {"x": float("nan")},
            {"x": 2.5},
            {"x": 12},
        ],
    ) == (0, 10)
    assert span_from_plotly_points(10, [{"not_x": 4}]) is None


def test_plotly_selection_converts_to_half_open_span():
    assert span_from_plotly_selection(10, None) is None
    assert span_from_plotly_selection(10, {}) is None
    assert span_from_plotly_selection(10, {"range": {"x": [1.2, 4.7]}}) == (2, 5)
    assert span_from_plotly_selection(10, {"range": {"x3": [9.8, -2.0]}}) == (
        0,
        10,
    )
    assert span_from_plotly_selection(10, {"range": {"x2": [6.1, 6.2]}}) == (
        7,
        8,
    )


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


def test_sequence_tracks_are_aligned_and_dependency_is_not_selectable():
    dependency = np.arange(9, dtype=float).reshape(3, 3)
    dependency = (dependency + dependency.T) / 2.0
    sequence_figure, dependency_figure = sequence_tracks_figures(
        "ACG",
        _logo(),
        dependency,
        span=(1, 3),
    )

    assert sequence_figure.layout.dragmode == "select"
    assert sequence_figure.layout.selectdirection == "h"
    assert dependency_figure.layout.dragmode is False
    assert sequence_figure.layout.xaxis.range == (0.5, 2.5)
    assert sequence_figure.layout.xaxis2.range == (0.5, 2.5)
    assert dependency_figure.layout.xaxis.range == (0.5, 2.5)
    assert (
        sequence_figure.layout.xaxis.domain
        == sequence_figure.layout.xaxis2.domain
        == dependency_figure.layout.xaxis.domain
        == (0.0, 1.0)
    )
    assert sequence_figure.layout.xaxis.matches == "x2"
    assert sequence_figure.layout.yaxis.fixedrange
    assert sequence_figure.layout.yaxis2.fixedrange
    assert dependency_figure.layout.xaxis.fixedrange
    assert dependency_figure.layout.yaxis.fixedrange
    assert dependency_figure.layout.yaxis.scaleanchor == "x"
    assert dependency_figure.layout.yaxis.scaleratio == 1
    assert dependency_figure.layout.yaxis.range == (2.5, 0.5)

    raw_positions = {
        int(position) for trace in sequence_figure.data[:4] for position in trace.x
    }
    assert raw_positions == {0, 1, 2}
    assert sequence_figure.data[0].xaxis == "x"
    assert sequence_figure.data[4].xaxis == "x2"
    assert dependency_figure.data[0].xaxis is None
    assert dependency_figure.data[0].yaxis is None
    assert all(
        shape.xref == "x2" and shape.yref == "y2"
        for shape in sequence_figure.layout.shapes
    )

    assert sequence_figure.layout.width == dependency_figure.layout.width
    assert sequence_figure.layout.margin.l == dependency_figure.layout.margin.l
    assert sequence_figure.layout.margin.r == dependency_figure.layout.margin.r
    plot_width = (
        dependency_figure.layout.width
        - dependency_figure.layout.margin.l
        - dependency_figure.layout.margin.r
    )
    dependency_height = (
        dependency_figure.layout.height
        - dependency_figure.layout.margin.t
        - dependency_figure.layout.margin.b
    )
    assert dependency_height == pytest.approx(plot_width)
    assert {annotation.text for annotation in sequence_figure.layout.annotations} == {
        "DNA sequence · drag horizontally to zoom",
        "Sequence logo",
    }
    [dependency_title] = dependency_figure.layout.annotations
    assert dependency_title.text == "Nucleotide dependency"
    assert dependency_title.y > 1
    assert dependency_title.yanchor == "bottom"
