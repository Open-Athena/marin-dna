"""Pure plotting and download helpers for the hosted sequence explorer."""

from __future__ import annotations

import base64
import csv
import html
import io
import math
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
        height=280,
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


def span_from_plotly_points(
    length: int,
    points: list[dict[str, Any]] | None,
) -> tuple[int, int] | None:
    """Convert selected Plotly nucleotide points to a 0-based half-open span."""
    assert length >= 1
    if not points:
        return None
    positions: list[int] = []
    for point in points:
        raw_position = next(
            (
                point[key]
                for key in ("x", "x2", "x3", "Sequence position (0-based)")
                if key in point
            ),
            None,
        )
        try:
            value = float(raw_position)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        position = round(value)
        if not math.isclose(value, position, abs_tol=1e-6):
            continue
        if 0 <= position < length:
            positions.append(position)
    if not positions:
        return None
    start = min(positions)
    end = max(positions) + 1
    assert 0 <= start < end <= length
    return start, end


def span_from_plotly_selection(
    length: int,
    selection: dict[str, Any] | None,
) -> tuple[int, int] | None:
    """Convert a horizontal Plotly selection to a 0-based half-open span."""
    assert length >= 1
    if not selection:
        return None
    ranges = selection.get("range")
    if not isinstance(ranges, dict):
        return None
    raw_range = next(
        (ranges[key] for key in ("x", "x2", "x3") if key in ranges),
        None,
    )
    if not isinstance(raw_range, list) or len(raw_range) != 2:
        return None
    raw_start, raw_end = sorted(float(value) for value in raw_range)
    start = max(0, min(length - 1, math.ceil(raw_start)))
    end = max(start + 1, min(length, math.floor(raw_end) + 1))
    assert 0 <= start < end <= length
    return start, end


def sequence_tracks_figures(
    sequence: str,
    logo: NucleotideLogo,
    dependency: np.ndarray,
    *,
    span: tuple[int, int] | None = None,
) -> tuple[go.Figure, go.Figure]:
    """Aligned selectable sequence/logo and static square dependency figures."""
    length = len(sequence)
    assert length >= 1
    assert set(sequence) <= set(NUCLEOTIDES)
    assert logo.probabilities.shape == (length, 4)
    assert dependency.shape == (length, length)
    start, end = _normalized_span(length, span)
    logo_track = logo_figure(logo)
    dependency_track = dependency_figure(dependency)

    figure_width = 1050
    left_margin = 72
    right_margin = 112
    plot_width = figure_width - left_margin - right_margin
    top_margin = 70
    selection_bottom_margin = 30
    dependency_bottom_margin = 60
    raw_sequence_height = 56
    logo_height = 150
    gap = 32
    selection_plot_height = raw_sequence_height + logo_height + gap
    selection_figure_height = (
        top_margin + selection_plot_height + selection_bottom_margin
    )
    dependency_figure_height = top_margin + plot_width + dependency_bottom_margin

    logo_domain = [0.0, logo_height / selection_plot_height]
    raw_sequence_domain = [
        (logo_height + gap) / selection_plot_height,
        1.0,
    ]
    assert plot_width > 0
    assert 0 < logo_domain[1] < raw_sequence_domain[0] < 1

    sequence_figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0,
    )
    positions = np.arange(length)
    sequence_array = np.asarray(list(sequence))
    for nucleotide in NUCLEOTIDES:
        nucleotide_positions = positions[sequence_array == nucleotide]
        sequence_figure.add_trace(
            go.Scatter(
                x=nucleotide_positions,
                y=np.full(len(nucleotide_positions), 0.5),
                text=[nucleotide] * len(nucleotide_positions),
                mode="text",
                textfont={
                    "color": NUCLEOTIDE_COLORS[nucleotide],
                    "family": "DejaVu Sans Mono, monospace",
                    "size": 14,
                },
                hovertemplate=(
                    f"Position %{{x}}<br>Nucleotide {nucleotide}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    for trace in logo_track.data:
        sequence_figure.add_trace(trace, row=2, col=1)

    shapes: list[dict[str, Any]] = []
    for shape in logo_track.layout.shapes:
        shape_json = shape.to_plotly_json()
        shape_json.update(xref="x2", yref="y2")
        shapes.append(shape_json)

    shared_range = [start - 0.5, end - 0.5]
    sequence_figure.update_layout(
        shapes=shapes,
        width=figure_width,
        height=selection_figure_height,
        autosize=False,
        margin={
            "l": left_margin,
            "r": right_margin,
            "t": top_margin,
            "b": selection_bottom_margin,
            "autoexpand": False,
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        dragmode="select",
        hovermode="closest",
        selectdirection="h",
        newselection={"line": {"color": "#4F46E5", "width": 2}},
        uirevision=f"sequence-tracks-{sequence}-{start}-{end}",
    )
    sequence_figure.update_xaxes(
        range=shared_range,
        domain=[0.0, 1.0],
        showgrid=False,
        zeroline=False,
        row=1,
        col=1,
    )
    sequence_figure.update_xaxes(
        range=shared_range,
        domain=[0.0, 1.0],
        showgrid=False,
        zeroline=False,
        row=2,
        col=1,
    )
    sequence_figure.update_yaxes(
        range=[0, 1],
        domain=raw_sequence_domain,
        visible=False,
        fixedrange=True,
        row=1,
        col=1,
    )
    sequence_figure.update_yaxes(
        title="Information (bits)",
        range=[0, 2],
        domain=logo_domain,
        tick0=0,
        dtick=0.5,
        showgrid=True,
        gridcolor="#E5E7EB",
        zeroline=False,
        fixedrange=True,
        row=2,
        col=1,
    )
    sequence_figure.add_annotation(
        text="DNA sequence · drag horizontally to zoom",
        x=0,
        y=1.02,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        font={"size": 18},
    )
    sequence_figure.add_annotation(
        text="Sequence logo",
        x=0,
        y=logo_domain[1] + 0.025,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        font={"size": 18},
    )

    dependency_figure_panel = go.Figure()
    for trace in dependency_track.data:
        trace.colorbar.update(
            x=1.02,
            xanchor="left",
            y=0.5,
            len=1.0,
            thickness=16,
        )
        dependency_figure_panel.add_trace(trace)
    dependency_figure_panel.update_layout(
        width=figure_width,
        height=dependency_figure_height,
        autosize=False,
        margin={
            "l": left_margin,
            "r": right_margin,
            "t": top_margin,
            "b": dependency_bottom_margin,
            "autoexpand": False,
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        dragmode=False,
        hovermode="closest",
        uirevision=f"dependency-track-{start}-{end}",
    )
    dependency_figure_panel.update_xaxes(
        title="Sequence position (0-based)",
        range=shared_range,
        domain=[0.0, 1.0],
        constrain="domain",
        fixedrange=True,
    )
    dependency_figure_panel.update_yaxes(
        title="Sequence position (0-based)",
        range=[shared_range[1], shared_range[0]],
        domain=[0.0, 1.0],
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
        fixedrange=True,
    )
    dependency_figure_panel.add_annotation(
        text="Nucleotide dependency",
        x=0,
        y=1.02,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        font={"size": 18},
    )
    return sequence_figure, dependency_figure_panel


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
