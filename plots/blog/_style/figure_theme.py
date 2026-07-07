"""Web-native rcParams for the figure set.

Importing this module applies a small set of matplotlib rcParams (a side
effect, on purpose) that make the saved SVGs look like they were drawn
directly on the blog page rather than dropped in as screenshots:

  * ``svg.fonttype = 'none'`` keeps text as real ``<text>`` elements instead of
    vector outlines, so once the SVG is inlined into the page the labels render
    in the page's webfont and stay selectable. (The site build rewrites the
    emitted ``font-family`` to the page stack and maps black to ``currentColor``
    so axes/text follow the page theme — see ``site/build.py``.)
  * top/right spines off — an open, D3/Observable-Plot-style frame.
  * text, axis, and tick colors set to the page's near-black ink so the mapped
    ``currentColor`` matches the body text exactly.

Metrics still come from matplotlib's bundled DejaVu Sans (always present at
build time), so layout is deterministic regardless of which fonts the machine
has; only the *rendered* font is swapped to the page stack in the browser.

Imported for its side effect by ``utils.savefig`` (which every figure imports),
so the theme applies to all figures without each script opting in.
"""

from __future__ import annotations

import matplotlib as mpl

# The page's body ink (--text in the Open Athena stylesheet). Emitted as
# #1f1e1b, which the site build maps to currentColor for theme-reactivity.
INK = "#1f1e1b"

# (On-page sizing lives in figure_style.SCALE / figure_style.figsize(): figures
# are authored at their final size rather than resized here.)

mpl.rcParams.update(
    {
        # Keep text as <text> (font-independent layout via DejaVu metrics, but
        # restyleable/selectable once inlined).
        "svg.fonttype": "none",
        # Open frame: drop the top/right spines like a native web chart.
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Page ink for every line of text and the axis furniture, so the
        # build's black -> currentColor mapping lands on the page text color.
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.titlecolor": INK,
    }
)
