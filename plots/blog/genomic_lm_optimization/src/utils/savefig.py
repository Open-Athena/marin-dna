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
import sys
import matplotlib as mpl

# Plot recipes run in their small, isolated uv project. Add this monorepo's
# tracked library source so the shared SVG normalizer remains the single source
# of truth without pulling the full training dependency graph into that lock.
REPO_SRC = Path(__file__).resolve().parents[5] / "src"
assert (REPO_SRC / "marin_dna" / "blog_figure_typography.py").is_file(), REPO_SRC
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from marin_dna.blog_figure_typography import (  # noqa: E402
    normalize_svg_typography_file,
)

# Imported for its side effect: applies the web-native rcParams (svg.fonttype,
# despine, page-ink colors) to every figure, since all figures import this
# module to save.
from utils import figure_theme  # noqa: E402, F401


def _normalize_svg(path: Path) -> None:
    """Remove generator timestamps and trailing whitespace for stable diffs."""
    lines = path.read_text().splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n")


def save_figure(fig, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with mpl.rc_context({"svg.hashsalt": name}):
        for ext, extra in (
            ("png", {"dpi": 300}),
            ("pdf", {"metadata": {"CreationDate": None, "ModDate": None}}),
            ("svg", {"metadata": {"Date": None}}),
        ):
            path = directory / f"{name}.{ext}"
            fig.savefig(path, bbox_inches="tight", transparent=True, **extra)
            if ext == "svg":
                _normalize_svg(path)
                normalize_svg_typography_file(path)
            paths.append(path)
    print("Wrote " + ", ".join(str(p) for p in paths))
