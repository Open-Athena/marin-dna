import base64
import re

import numpy as np
import pytest

from marin_dna.apps.sequence_explorer_ui import (
    dependency_figure,
    download_links_html,
    logo_figure,
    sequence_tracks_figure,
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


def test_sequence_tracks_are_aligned_and_raw_dna_drives_square_zoom():
    dependency = np.arange(9, dtype=float).reshape(3, 3)
    dependency = (dependency + dependency.T) / 2.0
    figure = sequence_tracks_figure("ACG", _logo(), dependency)

    assert figure.layout.dragmode == "zoom"
    assert figure.layout.selectdirection == "h"
    assert figure.layout.xaxis.range == (-0.5, 2.5)
    assert figure.layout.xaxis2.range == (-0.5, 2.5)
    assert figure.layout.xaxis3.range == (-0.5, 2.5)
    assert (
        figure.layout.xaxis.domain
        == figure.layout.xaxis2.domain
        == figure.layout.xaxis3.domain
        == (0.0, 1.0)
    )
    assert figure.layout.xaxis.matches == "x3"
    assert figure.layout.xaxis2.matches == "x3"
    assert figure.layout.yaxis3.matches == "x3"
    assert figure.layout.yaxis3.range == (2.5, -0.5)
    assert figure.layout.yaxis.fixedrange
    assert figure.layout.yaxis2.fixedrange

    raw_positions = {int(position) for trace in figure.data[:4] for position in trace.x}
    assert raw_positions == {0, 1, 2}
    assert figure.data[0].xaxis == "x"
    assert figure.data[4].xaxis == "x2"
    assert figure.data[5].xaxis == "x3"
    assert figure.data[5].yaxis == "y3"
    assert all(
        shape.xref == "x2" and shape.yref == "y2" for shape in figure.layout.shapes
    )

    assert figure.layout.margin.autoexpand is False
    plot_width = figure.layout.width - figure.layout.margin.l - figure.layout.margin.r
    plot_height = figure.layout.height - figure.layout.margin.t - figure.layout.margin.b
    dependency_domain = figure.layout.yaxis3.domain
    dependency_height = plot_height * (dependency_domain[1] - dependency_domain[0])
    assert dependency_height == pytest.approx(plot_width)
    assert {annotation.text for annotation in figure.layout.annotations} == {
        "DNA sequence · drag horizontally to zoom",
        "Sequence logo",
        "Nucleotide dependency",
    }
