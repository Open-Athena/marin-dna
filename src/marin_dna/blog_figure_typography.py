"""Normalize and validate typography in SVGs used by the MarinDNA blog post."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from statistics import median
from xml.etree import ElementTree


FIGURE_TYPOGRAPHY_VERSION = "lato-v1"
FIGURE_FONT_FAMILY = "Lato, sans-serif"
# The article's SVG frame uses 1.25rem (20px) of padding on each side.
FIGURE_FRAME_HORIZONTAL_PADDING_PX = 40.0
FIGURE_RENDER_WIDTH_PX = 700.0

# Matplotlib authors plots at its native defaults. The browser then renders the
# complete SVG 1.2x larger, preserving the intended proportions among text,
# lines, markers, and whitespace instead of resizing fonts independently.
MATPLOTLIB_BASE_SIZE = 10.0
FIGURE_GLOBAL_RENDER_SCALE = 1.2
MATPLOTLIB_TITLE_SIZE_RATIO = 1.2
MATPLOTLIB_SMALL_SIZE_RATIO = 5.0 / 6.0

MATPLOTLIB_NOTE_SIZE = MATPLOTLIB_BASE_SIZE * MATPLOTLIB_SMALL_SIZE_RATIO
FIGURE_BASE_SIZE_PX = MATPLOTLIB_BASE_SIZE * FIGURE_GLOBAL_RENDER_SCALE
FIGURE_BODY_SIZE_PX = FIGURE_BASE_SIZE_PX
FIGURE_AXIS_LABEL_SIZE_PX = FIGURE_BASE_SIZE_PX
FIGURE_TICK_SIZE_PX = FIGURE_BASE_SIZE_PX
FIGURE_LEGEND_SIZE_PX = FIGURE_BASE_SIZE_PX
FIGURE_TITLE_SIZE_PX = FIGURE_BASE_SIZE_PX * MATPLOTLIB_TITLE_SIZE_RATIO
FIGURE_PANEL_TITLE_SIZE_PX = FIGURE_BASE_SIZE_PX * MATPLOTLIB_TITLE_SIZE_RATIO
FIGURE_NOTE_SIZE_PX = FIGURE_BASE_SIZE_PX * MATPLOTLIB_SMALL_SIZE_RATIO

# The normalizer also processes hand-authored explanatory diagrams. Keep their
# existing 16px top-level headings valid; plot semantics use the ratios above.
FIGURE_SVG_MAX_SIZE_PX = 16.0
FIGURE_MATH_MIN_SIZE_PX = 7.0

_NUMBER = r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)"
_SVG_ROOT_RE = re.compile(r"<svg\b[^>]*>", re.IGNORECASE | re.DOTALL)
_SVG_TEXT_TAG_RE = re.compile(
    r"<(?P<name>text|tspan|g)\b[^>]*>", re.IGNORECASE | re.DOTALL
)
_CSS_FONT_SIZE_RE = re.compile(
    rf"(?P<prefix>\bfont-size\s*:\s*)(?P<value>{_NUMBER})(?P<unit>px|pt)?",
    re.IGNORECASE,
)
_ATTRIBUTE_FONT_SIZE_RE = re.compile(
    rf"(?P<prefix>\bfont-size\s*=\s*[\"'])(?P<value>{_NUMBER})"
    rf"(?P<unit>px|pt)?(?P<suffix>[\"'])",
    re.IGNORECASE,
)
_CSS_FONT_FAMILY_RE = re.compile(
    r"(?P<prefix>\bfont-family\s*:\s*)(?P<value>[^;\"]+)",
    re.IGNORECASE,
)
_ATTRIBUTE_FONT_FAMILY_RE = re.compile(
    r"(?P<prefix>\bfont-family\s*=\s*)(?P<quote>[\"'])"
    r"(?P<value>.*?)(?P=quote)",
    re.IGNORECASE,
)
_TYPOGRAPHY_MARKER_RE = re.compile(
    r"\bdata-figure-typography\s*=\s*[\"'][^\"']*[\"']", re.IGNORECASE
)
_RENDER_WIDTH_MARKER_RE = re.compile(
    r"\bdata-figure-render-width\s*=\s*[\"'][^\"']*[\"']", re.IGNORECASE
)
_RENDER_SCALE_MARKER_RE = re.compile(
    r"\bdata-figure-render-scale\s*=\s*[\"'][^\"']*[\"']", re.IGNORECASE
)
_PRESERVE_TYPOGRAPHY_RE = re.compile(
    r"\bdata-figure-preserve-typography\s*=\s*[\"']true[\"']", re.IGNORECASE
)


def matplotlib_typography_rcparams() -> dict[str, float]:
    """Return Matplotlib's fixed default semantic font hierarchy."""
    title_size = MATPLOTLIB_BASE_SIZE * MATPLOTLIB_TITLE_SIZE_RATIO
    return {
        "font.size": MATPLOTLIB_BASE_SIZE,
        "axes.titlesize": title_size,
        "figure.titlesize": title_size,
        "axes.labelsize": MATPLOTLIB_BASE_SIZE,
        "xtick.labelsize": MATPLOTLIB_BASE_SIZE,
        "ytick.labelsize": MATPLOTLIB_BASE_SIZE,
        "legend.fontsize": MATPLOTLIB_BASE_SIZE,
        "legend.title_fontsize": MATPLOTLIB_BASE_SIZE,
    }


def _view_box_width(svg: str, path: Path | None = None) -> float:
    root = ElementTree.fromstring(svg)
    assert root.tag.rsplit("}", maxsplit=1)[-1] == "svg", (
        f"not an SVG root: {path or '<string>'}"
    )
    view_box = root.get("viewBox")
    assert view_box is not None, f"referenced SVG lacks viewBox: {path or '<string>'}"
    values = view_box.replace(",", " ").split()
    assert len(values) == 4, f"invalid SVG viewBox={view_box!r}: {path or '<string>'}"
    width = float(values[2])
    assert math.isfinite(width) and width > 0, (
        f"non-positive SVG viewBox={view_box!r}: {path or '<string>'}"
    )
    return width


def _to_user_units(value: float, unit: str | None) -> float:
    if unit is None or unit.lower() == "px":
        return value
    assert unit.lower() == "pt", f"unsupported SVG font-size unit: {unit}"
    return value * 96.0 / 72.0


def _from_user_units(value: float, unit: str | None) -> float:
    if unit is None or unit.lower() == "px":
        return value
    assert unit.lower() == "pt", f"unsupported SVG font-size unit: {unit}"
    return value * 72.0 / 96.0


def _format_number(value: float) -> str:
    rendered = f"{value:.3f}".rstrip("0").rstrip(".")
    return rendered or "0"


def _font_size_declarations(svg: str) -> list[tuple[str, float]]:
    declarations: list[tuple[str, float]] = []
    for tag_match in _SVG_TEXT_TAG_RE.finditer(svg):
        tag = tag_match.group(0)
        if _PRESERVE_TYPOGRAPHY_RE.search(tag):
            continue
        name = tag_match.group("name").lower()
        for pattern in (_CSS_FONT_SIZE_RE, _ATTRIBUTE_FONT_SIZE_RE):
            for match in pattern.finditer(tag):
                declarations.append(
                    (
                        name,
                        _to_user_units(
                            float(match.group("value")), match.group("unit")
                        ),
                    )
                )
    return declarations


def _modal_body_size(declarations: list[tuple[str, float]]) -> float:
    body_sizes = [round(value, 6) for tag, value in declarations if tag != "tspan"]
    assert body_sizes, "SVG has no text font-size declarations"
    counts = Counter(body_sizes)
    highest_count = max(counts.values())
    candidates = [value for value, count in counts.items() if count == highest_count]
    midpoint = median(body_sizes)
    return min(candidates, key=lambda value: (abs(value - midpoint), value))


def _normalize_size(
    value: float,
    unit: str | None,
    tag_name: str,
    factor: float,
    view_box_width: float,
    render_width_px: float,
) -> str:
    scaled = _to_user_units(value, unit) * factor
    effective = scaled * render_width_px / view_box_width
    minimum = FIGURE_MATH_MIN_SIZE_PX if tag_name == "tspan" else FIGURE_NOTE_SIZE_PX
    effective = min(max(effective, minimum), FIGURE_SVG_MAX_SIZE_PX)
    normalized = effective * view_box_width / render_width_px
    return _format_number(_from_user_units(normalized, unit))


def _normalize_sizes(svg: str, view_box_width: float, render_width_px: float) -> str:
    declarations = _font_size_declarations(svg)
    modal_size = _modal_body_size(declarations)
    target_size = FIGURE_BODY_SIZE_PX * view_box_width / render_width_px
    factor = target_size / modal_size

    def normalize_tag(tag_match: re.Match[str]) -> str:
        tag_name = tag_match.group("name").lower()
        tag = tag_match.group(0)
        if _PRESERVE_TYPOGRAPHY_RE.search(tag):
            return tag

        def replace_css(match: re.Match[str]) -> str:
            value = _normalize_size(
                float(match.group("value")),
                match.group("unit"),
                tag_name,
                factor,
                view_box_width,
                render_width_px,
            )
            return f"{match.group('prefix')}{value}{match.group('unit') or ''}"

        def replace_attribute(match: re.Match[str]) -> str:
            value = _normalize_size(
                float(match.group("value")),
                match.group("unit"),
                tag_name,
                factor,
                view_box_width,
                render_width_px,
            )
            return (
                f"{match.group('prefix')}{value}{match.group('unit') or ''}"
                f"{match.group('suffix')}"
            )

        tag = _CSS_FONT_SIZE_RE.sub(replace_css, tag)
        return _ATTRIBUTE_FONT_SIZE_RE.sub(replace_attribute, tag)

    return _SVG_TEXT_TAG_RE.sub(normalize_tag, svg)


def _normalize_font_families(svg: str) -> str:
    def replace_css(match: re.Match[str]) -> str:
        family = match.group("value").lower()
        if "monospace" in family or "emoji" in family:
            return match.group(0)
        return f"{match.group('prefix')}'Lato', sans-serif"

    def replace_attribute(match: re.Match[str]) -> str:
        family = match.group("value").lower()
        if "monospace" in family or "emoji" in family:
            return match.group(0)
        quote = match.group("quote")
        return f"{match.group('prefix')}{quote}{FIGURE_FONT_FAMILY}{quote}"

    svg = _CSS_FONT_FAMILY_RE.sub(replace_css, svg)
    return _ATTRIBUTE_FONT_FAMILY_RE.sub(replace_attribute, svg)


def _mark_root(
    svg: str, render_width_px: float, render_scale: float | None = None
) -> str:
    match = _SVG_ROOT_RE.search(svg)
    assert match is not None, "SVG has no root element"
    root = match.group(0)
    if _TYPOGRAPHY_MARKER_RE.search(root):
        root = _TYPOGRAPHY_MARKER_RE.sub(
            f'data-figure-typography="{FIGURE_TYPOGRAPHY_VERSION}"', root
        )
    else:
        root = root[:-1] + (f' data-figure-typography="{FIGURE_TYPOGRAPHY_VERSION}">')
    render_width_marker = (
        f'data-figure-render-width="{_format_number(render_width_px)}"'
    )
    if _RENDER_WIDTH_MARKER_RE.search(root):
        root = _RENDER_WIDTH_MARKER_RE.sub(render_width_marker, root)
    else:
        root = root[:-1] + f" {render_width_marker}>"
    if render_scale is not None:
        render_scale_marker = (
            f'data-figure-render-scale="{_format_number(render_scale)}"'
        )
        if _RENDER_SCALE_MARKER_RE.search(root):
            root = _RENDER_SCALE_MARKER_RE.sub(render_scale_marker, root)
        else:
            root = root[:-1] + f" {render_scale_marker}>"
    if not _ATTRIBUTE_FONT_FAMILY_RE.search(root):
        root = root[:-1] + f' font-family="{FIGURE_FONT_FAMILY}">'
    return svg[: match.start()] + root + svg[match.end() :]


def _root_render_scale(svg: str) -> float | None:
    root = ElementTree.fromstring(svg)
    raw_scale = root.get("data-figure-render-scale")
    if raw_scale is None:
        return None
    scale = float(raw_scale)
    assert math.isfinite(scale) and scale > 0, raw_scale
    return scale


def normalize_svg_typography(
    svg: str, render_width_px: float = FIGURE_RENDER_WIDTH_PX
) -> str:
    """Return an SVG using Lato and a normalized final-width size hierarchy."""
    assert math.isfinite(render_width_px) and render_width_px > 0, render_width_px
    view_box_width = _view_box_width(svg)
    render_scale = _root_render_scale(svg)
    if render_scale is not None:
        assert math.isclose(render_scale, FIGURE_GLOBAL_RENDER_SCALE), (
            "Matplotlib plots must use the one shared figure render scale: "
            f"{FIGURE_GLOBAL_RENDER_SCALE:g}, not {render_scale:g}"
        )
        expected_width = view_box_width * render_scale
        assert math.isclose(render_width_px, expected_width, abs_tol=0.01), (
            render_width_px,
            expected_width,
        )
        normalized = _normalize_font_families(svg)
        return _mark_root(normalized, expected_width, render_scale)
    normalized = _normalize_sizes(svg, view_box_width, render_width_px)
    normalized = _normalize_font_families(normalized)
    return _mark_root(normalized, render_width_px)


def normalize_matplotlib_svg_typography(svg: str) -> str:
    """Apply the one shared whole-figure scale to a Matplotlib SVG."""
    render_width_px = _view_box_width(svg) * FIGURE_GLOBAL_RENDER_SCALE
    normalized = _normalize_font_families(svg)
    return _mark_root(normalized, render_width_px, FIGURE_GLOBAL_RENDER_SCALE)


def normalize_matplotlib_svg_typography_file(path: Path) -> bool:
    """Normalize one Matplotlib SVG in place and report whether it changed."""
    original = path.read_text()
    normalized = normalize_matplotlib_svg_typography(original)
    if normalized == original:
        return False
    path.write_text(normalized)
    return True


def sync_article_figure_width(
    article_path: Path, figure_id: str, svg_path: Path
) -> float:
    """Sync one plot frame to the whole-SVG render width declared by its SVG."""
    svg = svg_path.read_text(encoding="utf-8")
    root = ElementTree.fromstring(svg)
    raw_render_width = root.get("data-figure-render-width")
    raw_render_scale = root.get("data-figure-render-scale")
    assert raw_render_width is not None, svg_path
    assert raw_render_scale is not None, svg_path
    render_width = float(raw_render_width)
    render_scale = float(raw_render_scale)
    assert math.isclose(render_scale, FIGURE_GLOBAL_RENDER_SCALE), (
        svg_path,
        render_scale,
    )
    assert math.isclose(
        render_width,
        _view_box_width(svg, svg_path) * render_scale,
        abs_tol=0.01,
    )
    frame_width = render_width + FIGURE_FRAME_HORIZONTAL_PADDING_PX
    article = article_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf'(<figure id="{re.escape(figure_id)}" data-figure-width=")'
        r"[0-9]+(?:\.[0-9]+)?(\">)"
    )
    replacement = _format_number(frame_width)
    updated, count = pattern.subn(rf"\g<1>{replacement}\g<2>", article)
    assert count == 1, f"expected one {figure_id} declaration, found {count}"
    if updated != article:
        article_path.write_text(updated, encoding="utf-8")
    return frame_width


def normalize_svg_typography_file(
    path: Path, render_width_px: float = FIGURE_RENDER_WIDTH_PX
) -> bool:
    """Normalize one SVG in place and report whether its bytes changed."""
    original = path.read_text()
    normalized = normalize_svg_typography(original, render_width_px)
    if normalized == original:
        return False
    path.write_text(normalized)
    return True


def validate_svg_typography(
    path: Path, expected_render_width_px: float | None = None
) -> None:
    """Reject SVG typography that diverges from the blog's Lato hierarchy."""
    svg = path.read_text()
    root = ElementTree.fromstring(svg)
    assert root.get("data-figure-typography") == FIGURE_TYPOGRAPHY_VERSION, (
        f"referenced SVG lacks normalized Lato typography marker: {path}"
    )
    raw_render_width = root.get("data-figure-render-width")
    assert raw_render_width is not None, (
        f"referenced SVG lacks normalized render-width marker: {path}"
    )
    render_width_px = float(raw_render_width)
    assert math.isfinite(render_width_px) and render_width_px > 0, (
        f"referenced SVG has invalid render width {raw_render_width!r}: {path}"
    )
    if expected_render_width_px is not None:
        assert math.isclose(render_width_px, expected_render_width_px), (
            f"referenced SVG was normalized for {render_width_px:g}px but the article "
            f"renders it at {expected_render_width_px:g}px: {path}"
        )
    raw_render_scale = root.get("data-figure-render-scale")
    if raw_render_scale is not None:
        render_scale = float(raw_render_scale)
        assert math.isclose(render_scale, FIGURE_GLOBAL_RENDER_SCALE), (
            f"referenced SVG uses {render_scale:g}x rather than the shared "
            f"{FIGURE_GLOBAL_RENDER_SCALE:g}x plot scale: {path}"
        )
        assert math.isclose(
            render_width_px, _view_box_width(svg, path) * render_scale, abs_tol=0.01
        ), f"referenced SVG whole-figure scale is inconsistent: {path}"

    families = [
        match.group("value").strip()
        for pattern in (_CSS_FONT_FAMILY_RE, _ATTRIBUTE_FONT_FAMILY_RE)
        for match in pattern.finditer(svg)
    ]
    assert families, f"referenced SVG has no font-family declaration: {path}"
    unexpected = [
        family
        for family in families
        if "lato" not in family.lower()
        and "monospace" not in family.lower()
        and "emoji" not in family.lower()
    ]
    assert not unexpected, (
        f"referenced SVG uses unexpected font families {sorted(set(unexpected))}: "
        f"{path}"
    )

    view_box_width = _view_box_width(svg, path)
    declarations = _font_size_declarations(svg)
    assert declarations, f"referenced SVG has no text font-size declarations: {path}"
    for tag_name, size in declarations:
        effective = size * render_width_px / view_box_width
        minimum = (
            FIGURE_MATH_MIN_SIZE_PX if tag_name == "tspan" else FIGURE_NOTE_SIZE_PX
        )
        assert minimum - 0.01 <= effective <= FIGURE_SVG_MAX_SIZE_PX + 0.01, (
            f"referenced SVG {tag_name} font renders at {effective:.2f}px, outside "
            f"the {minimum:g}–{FIGURE_SVG_MAX_SIZE_PX:g}px hierarchy: {path}"
        )
