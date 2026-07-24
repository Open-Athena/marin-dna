"""Pure plotting and download helpers for the hosted sequence explorer."""

from __future__ import annotations

import base64
import csv
import html
import io
from functools import cache
from typing import Any

import numpy as np
import plotly.graph_objects as go
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath
from plotly.subplots import make_subplots

from marin_dna.data.dna import NUCLEOTIDES
from marin_dna.model.sequence_interpretation import NucleotideLogo

NUCLEOTIDE_COLORS = {
    "A": "#2E8B57",
    "C": "#2563EB",
    "G": "#F59E0B",
    "T": "#DC2626",
}


def _format_number(value: float) -> str:
    return f"{value:.6g}"


@cache
def _normalized_glyph_commands(base: str) -> tuple[tuple[Any, ...], ...]:
    """Convert one bold nucleotide glyph to normalized Plotly path commands."""
    assert base in NUCLEOTIDES
    text_path = TextPath(
        (0, 0),
        base,
        size=1,
        prop=FontProperties(family="DejaVu Sans", weight="bold"),
    )
    bbox = text_path.get_extents()
    assert bbox.width > 0 and bbox.height > 0
    vertices = text_path.vertices.copy()
    vertices[:, 0] = (vertices[:, 0] - bbox.xmin) / bbox.width
    vertices[:, 1] = (vertices[:, 1] - bbox.ymin) / bbox.height
    codes = text_path.codes
    assert codes is not None

    commands: list[tuple[Any, ...]] = []
    i = 0
    while i < len(codes):
        code = codes[i]
        x, y = vertices[i]
        if code == Path.MOVETO:
            commands.append(("M", x, y))
            i += 1
        elif code == Path.LINETO:
            commands.append(("L", x, y))
            i += 1
        elif code == Path.CURVE3:
            assert i + 1 < len(codes) and codes[i + 1] == Path.CURVE3
            end_x, end_y = vertices[i + 1]
            commands.append(("Q", x, y, end_x, end_y))
            i += 2
        elif code == Path.CURVE4:
            assert i + 2 < len(codes)
            assert codes[i + 1] == Path.CURVE4
            assert codes[i + 2] == Path.CURVE4
            control_2_x, control_2_y = vertices[i + 1]
            end_x, end_y = vertices[i + 2]
            commands.append(("C", x, y, control_2_x, control_2_y, end_x, end_y))
            i += 3
        elif code == Path.CLOSEPOLY:
            commands.append(("Z",))
            i += 1
        else:
            raise AssertionError(f"unsupported matplotlib path code: {code}")
    return tuple(commands)


def _glyph_path(
    base: str,
    *,
    center_x: float,
    bottom_y: float,
    width: float,
    height: float,
) -> str:
    pieces: list[str] = []
    left_x = center_x - width / 2.0
    for command in _normalized_glyph_commands(base):
        op = command[0]
        values = command[1:]
        if op == "Z":
            pieces.append("Z")
            continue
        transformed: list[float] = []
        for index in range(0, len(values), 2):
            transformed.extend(
                [
                    left_x + float(values[index]) * width,
                    bottom_y + float(values[index + 1]) * height,
                ]
            )
        pieces.append(f"{op} " + " ".join(_format_number(v) for v in transformed))
    return " ".join(pieces)


def _normalized_span(length: int, span: tuple[int, int] | None) -> tuple[int, int]:
    if span is None:
        return 0, length
    start, end = span
    assert 0 <= start < end <= length, (
        f"invalid 0-based half-open span [{start}, {end}) for length {length}"
    )
    return start, end


def logo_figure(
    logo: NucleotideLogo,
    *,
    span: tuple[int, int] | None = None,
) -> go.Figure:
    """Interactive information-content logo rendered with true glyph paths."""
    length = int(logo.probabilities.shape[0])
    assert logo.probabilities.shape == (length, 4)
    assert logo.glyph_heights_bits.shape == (length, 4)
    start, end = _normalized_span(length, span)

    shapes: list[dict[str, Any]] = []
    hover_x: list[int] = []
    hover_y: list[float] = []
    hover_data: list[list[Any]] = []
    for position in range(length):
        heights = logo.glyph_heights_bits[position]
        bottom = 0.0
        for nucleotide_index in np.argsort(heights):
            height = float(heights[nucleotide_index])
            if height <= 1e-9:
                continue
            nucleotide = NUCLEOTIDES[int(nucleotide_index)]
            shapes.append(
                {
                    "type": "path",
                    "path": _glyph_path(
                        nucleotide,
                        center_x=float(position),
                        bottom_y=bottom,
                        width=0.88,
                        height=height,
                    ),
                    "fillcolor": NUCLEOTIDE_COLORS[nucleotide],
                    "line": {"width": 0},
                    "layer": "above",
                }
            )
            hover_x.append(position)
            hover_y.append(bottom + height / 2.0)
            hover_data.append(
                [
                    nucleotide,
                    float(logo.probabilities[position, nucleotide_index]),
                    height,
                    float(logo.information_bits[position]),
                ]
            )
            bottom += height

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker={"size": 14, "color": "rgba(0,0,0,0.001)"},
            customdata=hover_data,
            hovertemplate=(
                "Position %{x}<br>"
                "Nucleotide %{customdata[0]}<br>"
                "Probability %{customdata[1]:.4f}<br>"
                "Glyph height %{customdata[2]:.4f} bits<br>"
                "Stack information %{customdata[3]:.4f} bits"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.update_layout(
        shapes=shapes,
        height=430,
        margin={"l": 64, "r": 24, "t": 54, "b": 52},
        plot_bgcolor="white",
        paper_bgcolor="white",
        title={"text": "Model-predicted information-content logo", "x": 0.02},
        dragmode="pan",
        hovermode="closest",
        uirevision=f"sequence-logo-{start}-{end}",
    )
    figure.update_xaxes(
        title="Sequence position (0-based)",
        range=[start - 0.5, end - 0.5],
        showgrid=False,
        zeroline=False,
    )
    figure.update_yaxes(
        title="Information (bits)",
        range=[0, 2],
        tick0=0,
        dtick=0.5,
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        fixedrange=False,
    )
    return figure


def dependency_figure(
    dependency: np.ndarray,
    *,
    span: tuple[int, int] | None = None,
) -> go.Figure:
    """Interactive symmetric dependency heatmap with a square linked span."""
    length = int(dependency.shape[0])
    assert dependency.shape == (length, length)
    start, end = _normalized_span(length, span)
    positions = np.arange(length)
    finite = dependency[np.isfinite(dependency)]
    assert finite.size == dependency.size
    color_max = float(np.quantile(finite, 0.99))
    if color_max <= 0:
        color_max = 1.0

    figure = go.Figure(
        go.Heatmap(
            z=dependency,
            x=positions,
            y=positions,
            colorscale="RdBu_r",
            zmin=0,
            zmax=color_max,
            colorbar={"title": "Dependency"},
            hovertemplate=(
                "Position i %{x}<br>Position j %{y}<br>"
                "Dependency %{z:.5f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        height=650,
        margin={"l": 72, "r": 52, "t": 54, "b": 60},
        plot_bgcolor="white",
        paper_bgcolor="white",
        title={"text": "Forward/reverse-complement dependency map", "x": 0.02},
        dragmode="pan",
        uirevision=f"dependency-map-{start}-{end}",
    )
    figure.update_xaxes(
        title="Sequence position (0-based)",
        range=[start - 0.5, end - 0.5],
        constrain="domain",
    )
    figure.update_yaxes(
        title="Sequence position (0-based)",
        range=[end - 0.5, start - 0.5],
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
    )
    return figure


def sequence_tracks_figure(
    logo: NucleotideLogo,
    dependency: np.ndarray,
) -> go.Figure:
    """Aligned logo and dependency tracks with one shared nucleotide x-axis."""
    length = int(logo.probabilities.shape[0])
    assert dependency.shape == (length, length)
    logo_track = logo_figure(logo)
    dependency_track = dependency_figure(dependency)

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.34, 0.66],
    )
    for trace in logo_track.data:
        figure.add_trace(trace, row=1, col=1)
    for trace in dependency_track.data:
        trace.colorbar.update(
            x=1.02,
            xanchor="left",
            y=0.31,
            len=0.6,
            thickness=16,
        )
        figure.add_trace(trace, row=2, col=1)

    shapes: list[dict[str, Any]] = []
    for shape in logo_track.layout.shapes:
        shape_json = shape.to_plotly_json()
        shape_json.update(xref="x", yref="y")
        shapes.append(shape_json)

    shared_range = [-0.5, length - 0.5]
    figure.update_layout(
        shapes=shapes,
        height=960,
        margin={"l": 72, "r": 112, "t": 76, "b": 60},
        plot_bgcolor="white",
        paper_bgcolor="white",
        dragmode="zoom",
        hovermode="closest",
        selectdirection="h",
        newselection={"line": {"color": "#4F46E5", "width": 2}},
        uirevision=f"sequence-tracks-{length}",
    )
    figure.update_xaxes(
        range=shared_range,
        domain=[0.0, 1.0],
        showgrid=False,
        zeroline=False,
        row=1,
        col=1,
    )
    figure.update_xaxes(
        title="Sequence position (0-based)",
        range=shared_range,
        domain=[0.0, 1.0],
        row=2,
        col=1,
    )
    figure.update_yaxes(
        title="Information (bits)",
        range=[0, 2],
        tick0=0,
        dtick=0.5,
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title="Sequence position (0-based)",
        range=[length - 0.5, -0.5],
        row=2,
        col=1,
    )
    figure.add_annotation(
        text="Sequence logo",
        x=0,
        y=1.04,
        xref="paper",
        yref="paper",
        xanchor="left",
        showarrow=False,
        font={"size": 18},
    )
    figure.add_annotation(
        text="Nucleotide dependency",
        x=0,
        y=0.64,
        xref="paper",
        yref="paper",
        xanchor="left",
        showarrow=False,
        font={"size": 18},
    )
    return figure


def dependency_loading_figure() -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text="Building dependency map…",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 20, "color": "#475569"},
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(height=650, margin={"l": 20, "r": 20, "t": 20, "b": 20})
    return figure


def _csv_data_url(text: str) -> str:
    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"data:text/csv;base64,{payload}"


def _metadata_rows(metadata: dict[str, str]) -> str:
    return "".join(f"# {key}={value}\n" for key, value in metadata.items())


def download_links_html(
    logo: NucleotideLogo,
    dependency: np.ndarray,
    *,
    metadata: dict[str, str],
) -> str:
    """In-memory CSV download links; no submitted sequence or file persistence."""
    probability_buffer = io.StringIO()
    probability_buffer.write(_metadata_rows(metadata))
    probability_writer = csv.writer(probability_buffer, lineterminator="\n")
    probability_writer.writerow(
        [
            "position",
            "A_probability",
            "C_probability",
            "G_probability",
            "T_probability",
            "entropy_bits",
            "information_bits",
            "A_height_bits",
            "C_height_bits",
            "G_height_bits",
            "T_height_bits",
        ]
    )
    for position in range(len(logo.entropy_bits)):
        probability_writer.writerow(
            [
                position,
                *logo.probabilities[position].tolist(),
                float(logo.entropy_bits[position]),
                float(logo.information_bits[position]),
                *logo.glyph_heights_bits[position].tolist(),
            ]
        )

    dependency_buffer = io.StringIO()
    dependency_buffer.write(_metadata_rows(metadata))
    dependency_writer = csv.writer(dependency_buffer, lineterminator="\n")
    dependency_writer.writerow(["position", *range(dependency.shape[1])])
    for position, row in enumerate(dependency):
        dependency_writer.writerow([position, *row.tolist()])

    probability_url = html.escape(
        _csv_data_url(probability_buffer.getvalue()), quote=True
    )
    dependency_url = html.escape(
        _csv_data_url(dependency_buffer.getvalue()), quote=True
    )
    return (
        '<div class="download-links">'
        f'<a download="marindna_probabilities.csv" href="{probability_url}">'
        "Download probability/logo matrix (CSV)</a>"
        f'<a download="marindna_dependency.csv" href="{dependency_url}">'
        "Download dependency matrix (CSV)</a>"
        "</div>"
    )
