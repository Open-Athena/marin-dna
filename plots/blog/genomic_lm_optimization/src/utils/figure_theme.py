"""Web-native rcParams for the figure set.

Importing this module applies a small set of matplotlib rcParams (a side
effect, on purpose) that make the saved SVGs look like they were drawn
directly on the blog page rather than dropped in as screenshots:

  * ``svg.fonttype = 'none'`` keeps text as real ``<text>`` elements instead of
    vector outlines. ``utils.savefig`` then normalizes the web-facing SVG to
    the blog's Lato hierarchy; the article safely inlines those SVGs so they use
    the exact same loaded webfont as the prose.
  * top/right spines off — an open, D3/Observable-Plot-style frame.
  * text, axis, and tick colors set to the page's near-black body ink.

Metrics still come from matplotlib's bundled DejaVu Sans (always present at
build time), so layout is deterministic regardless of which fonts the machine
has; only the *rendered* font is swapped to the page stack in the browser.

Imported for its side effect by ``utils.savefig`` (which every figure imports),
so the theme applies to all figures without each script opting in.
"""

from __future__ import annotations

import matplotlib as mpl

from marin_dna.blog_figure_typography import (
    FIGURE_AXIS_LABEL_SIZE_PX,
    FIGURE_BODY_SIZE_PX,
    FIGURE_PANEL_TITLE_SIZE_PX,
    FIGURE_TITLE_SIZE_PX,
)

# The page's body ink (--text in the Open Athena stylesheet).
INK = "#1f1e1b"

# (On-page sizing lives in figure_style.SCALE / figure_style.figsize(): figures
# are authored at their final size rather than resized here.)

mpl.rcParams.update(
    {
        # Match the canonical MarinDNA blog figures explicitly instead of
        # relying on whichever sans-serif happens to be first on the host.
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        # Keep text as <text> (font-independent layout via DejaVu metrics, but
        # restyleable/selectable once inlined).
        "svg.fonttype": "none",
        # Semantic defaults. Individual dense panels may opt down within the
        # validated 11–16 px final-width range.
        "font.size": FIGURE_BODY_SIZE_PX,
        "axes.titlesize": FIGURE_PANEL_TITLE_SIZE_PX,
        "axes.labelsize": FIGURE_AXIS_LABEL_SIZE_PX,
        "xtick.labelsize": FIGURE_BODY_SIZE_PX,
        "ytick.labelsize": FIGURE_BODY_SIZE_PX,
        "legend.fontsize": FIGURE_BODY_SIZE_PX,
        "legend.title_fontsize": FIGURE_BODY_SIZE_PX,
        "figure.titlesize": FIGURE_TITLE_SIZE_PX,
        # Open frame: drop the top/right spines like a native web chart.
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Page ink for every line of text and the axis furniture.
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.titlecolor": INK,
    }
)
