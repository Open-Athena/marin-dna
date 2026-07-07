"""Save a figure as PNG + PDF + SVG with transparent backgrounds.

The SVG is the web-facing format: a transparent background lets it sit
seamlessly on an HTML page (the page background shows through instead of an
opaque white rectangle). Text is kept as real ``<text>`` elements (not vector
outlines) so that, once the SVG is inlined into the blog page, labels render in
the page's webfont and follow the page theme — see ``utils.figure_theme`` and
the inline/scrub step in ``site/build.py``. PNG/PDF are kept for previews and
the paper, and are made transparent too so all three formats stay visually
consistent.
"""

from __future__ import annotations

from pathlib import Path

# Imported for its side effect: applies the web-native rcParams (svg.fonttype,
# despine, page-ink colors) to every figure, since all figures import this
# module to save. (Only this import is adapted from the vendored original —
# package-relative here instead of `from utils import figure_theme`.)
from . import figure_theme  # noqa: F401


def save_figure(fig, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, extra in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        path = directory / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", transparent=True, **extra)
        paths.append(path)
    print("Wrote " + ", ".join(str(p) for p in paths))
