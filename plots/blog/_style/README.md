# Blog figure style — vendored from Eric's post repo

`figure_style.py`, `figure_theme.py`, and `savefig.py` are ported from
**[`eric-czech/marin-dna-post-202606`](https://github.com/eric-czech/marin-dna-post-202606)**
@ `2abef91b37a16fde9c9cdf1cfa0046942442b97f` (`src/utils/`). They are vendored so
the redone blog figures (epic #361) match the **untouched** Figures 1–4 exactly —
same earthy palette, `svg.fonttype=none` (real `<text>`), and ink `#1f1e1b`, which
the site build maps to the page webfont + `currentColor`.

| File | Provenance |
|------|-----------|
| `figure_style.py` | **verbatim** — palettes (`PARAM_CMAP`/`HEATMAP_CMAP`/`EARTH_QUAL`), `figsize`/`SCALE`, legend + tick helpers, formatters |
| `figure_theme.py` | **verbatim** — web-native rcParams, applied on import |
| `savefig.py` | verbatim **except** the `figure_theme` import (now package-relative: `from . import figure_theme`); emits transparent PNG+PDF+SVG |

**Updating:** if Eric's style changes, re-fetch from the source repo and re-pin the
SHA above:

```bash
ERIC=eric-czech/marin-dna-post-202606
for f in figure_style figure_theme savefig; do
  gh api repos/$ERIC/contents/src/utils/$f.py --jq .content | base64 -d > plots/blog/_style/$f.py
done
# then re-apply the one savefig import edit
```
